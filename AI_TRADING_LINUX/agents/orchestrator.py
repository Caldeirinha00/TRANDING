import asyncio
import logging
import time
from core.mt5_connection import mt5
from typing import Dict, Any, Optional
from datetime import datetime
from core.mt5_executor import mt5_call, mt5_exec

from agents.llm.mistral_client import mistral_client
from agents.market_analyzer import market_analyzer, _compute_atr
from agents.news_analyzer import news_analyzer
from agents.calendar_analyzer import calendar_analyzer
from agents.trade_executor import trade_executor
from core.mcp_client import mcp_client
from core.config import CONFIG
from risk.rules import RiskManager
from core.redis_broadcaster import broadcaster

logger = logging.getLogger(__name__)

# News cache: stores {symbol: {"result": ..., "timestamp": ...}}
_news_cache: Dict[str, Dict[str, Any]] = {}
_NEWS_CACHE_TTL = 180  # 3 minutes


class Orchestrator:
    """CIO Agent — coordinates the AI committee and makes final trading decisions."""
    
    def __init__(self):
        self.risk_manager = RiskManager()
        self.trades_today: list[Dict[str, Any]] = []
        self.trade_day: str = ""
        self._global_exec_lock = asyncio.Lock()
        
    def _reset_daily_counter(self):
        """Reset trade counter at the start of each new day."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self.trade_day != today:
            self.trades_today = []
            self.trade_day = today
    
    def _calculate_kelly_risk_pct(self, setup_type: str, setup_stats: dict, min_trades_full_kelly: int = 20) -> Optional[float]:
        """Calculates Risk % using Half-Kelly with Bayesian Shrinkage."""
        stats = setup_stats.get(setup_type, {})
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        total_profit = stats.get("total_profit", 0.0)
        total_loss = stats.get("total_loss", 0.0)
        n = wins + losses

        if n < 3:
            return None  # Insufficient data, fallback to legacy

        # Shrinkage prior = global win rate, equivalent to 10 trades
        global_wr = setup_stats.get("_global", {}).get("win_rate", 0.5)
        prior_strength = 10
        W = (wins + global_wr * prior_strength) / (n + prior_strength)

        avg_win = total_profit / wins if wins > 0 else 0.0
        avg_loss = total_loss / losses if losses > 0 else 0.0
        if avg_loss <= 0:
            return None

        R = avg_win / avg_loss
        if R <= 0:
            return 0.0
            
        kelly_pct = W - ((1 - W) / R)
        half_kelly = kelly_pct / 2

        # Scale down while sample size is small
        if n < min_trades_full_kelly:
            half_kelly *= (n / min_trades_full_kelly)

        risk_fraction = max(0.0, half_kelly)
        
        # Convert fraction to percentage (e.g. 0.01 -> 1.0%)
        risk_pct_scaled = risk_fraction * 100.0
        return min(risk_pct_scaled, float(CONFIG.MAX_SINGLE_TRADE_LOSS_PCT))

    def _calculate_lot_size(self, risk_pct: float, symbol_info, account_balance: float, sl_distance_monetary: float) -> float:
        """Dynamic lot sizing based on Account Balance risk % and SL distance."""
        if risk_pct <= 0:
            return 0.0
            
        risk_usd = account_balance * (risk_pct / 100.0)
        
        # 2. Calculate the lot size based on exactly how much 1 lot loses
        if sl_distance_monetary <= 0:
            logger.warning(f"Invalid monetary SL distance {sl_distance_monetary}. Aborting trade.")
            return 0.0
            
        # Lot_Size = Risk_USD / Loss_per_1_lot_USD
        calculated_lot = risk_usd / sl_distance_monetary
        
        # 3. Respect symbol minimum, maximum, and step, plus absolute Sanity Cap
        min_lot = symbol_info.volume_min if symbol_info else 0.01
        max_lot = symbol_info.volume_max if symbol_info else 100.0
        lot_step = symbol_info.volume_step if symbol_info else 0.01
        
        # Sanity cap (glass ceiling)
        safe_lot = min(calculated_lot, float(CONFIG.MAX_LOT_SIZE))
        
        if min_lot > safe_lot:
            logger.warning(f"Calculated safe lot {safe_lot} is smaller than broker minimum {min_lot}. Aborting to respect risk.")
            return 0.0
            
        lot = max(safe_lot, min_lot)
        lot = min(lot, max_lot)
        
        # Round to nearest lot_step
        if lot_step > 0:
            lot = round(lot / lot_step) * lot_step
            
        return round(lot, 2)

    async def _get_news_cached(self, symbol: str) -> Dict[str, Any]:
        """Get news analysis with 3-minute cache to save API tokens."""
        now = time.time()
        cached = _news_cache.get(symbol)
        
        if cached and (now - cached["timestamp"]) < _NEWS_CACHE_TTL:
            logger.info(f"Using cached news for {symbol} (age: {int(now - cached['timestamp'])}s)")
            return cached["result"]
        
        # Fetch fresh news
        result = await news_analyzer.analyze_sentiment(symbol)
        _news_cache[symbol] = {"result": result, "timestamp": now}
        return result

    async def manage_open_positions(self):
        """Active Position Management: Protects open trades using News and Technicals."""
        positions = await mt5_call(mt5.positions_get)
        if not positions:
            return
            
        for pos in positions:
            symbol = pos.symbol
            ticket = pos.ticket
            direction = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
            logger.info(f"Active Management: Reviewing {direction} position on {symbol} (Ticket: {ticket})")
            
            # 1. News Protection (Every cycle via Hash, 0 cost if no change)
            news_result = await self._get_news_cached(symbol)
            if news_result and news_result.get("sentiment") != "NEUTRAL":
                sentiment = news_result.get("sentiment")
                conf = news_result.get("confidence", 0)
                
                # Check for strong reversal (>=80%)
                is_reversal = (direction == "BUY" and sentiment == "BEARISH" and conf >= 80) or \
                              (direction == "SELL" and sentiment == "BULLISH" and conf >= 80)
                
                if is_reversal:
                    current_profit = pos.profit
                    if current_profit <= 0:
                        # Position is losing or breakeven: close immediately
                        logger.critical(f"NEWS REVERSAL on losing {direction} {symbol}. Profit: {current_profit:.2f}. Closing. Reason: {news_result.get('reasoning_summary')}")
                        await trade_executor.close_position(ticket)
                        continue
                    else:
                        # Position is profitable: tighten SL to lock profit instead of closing
                        logger.warning(f"NEWS REVERSAL on profitable {direction} {symbol}. Profit: {current_profit:.2f}. Tightening SL instead of closing.")
                        # Move SL to breakeven + small buffer
                        symbol_info = await mt5_call(mt5.symbol_info, symbol)
                        if symbol_info:
                            stops_level = getattr(symbol_info, 'trade_stops_level', 0)
                            buffer = symbol_info.point * max(10, stops_level + 2)
                            if direction == "BUY":
                                new_sl = pos.price_open + buffer
                                if new_sl > pos.sl:  # Only improve SL
                                    request = {"action": mt5.TRADE_ACTION_SLTP, "position": ticket, "symbol": symbol, "sl": new_sl, "tp": pos.tp}
                                    await mt5_call(mt5.order_send, request)
                                    logger.info(f"SL tightened to breakeven+ {new_sl} for {symbol} (Ticket: {ticket})")
                            else:
                                new_sl = pos.price_open - buffer
                                if pos.sl == 0 or new_sl < pos.sl:  # Only improve SL
                                    request = {"action": mt5.TRADE_ACTION_SLTP, "position": ticket, "symbol": symbol, "sl": new_sl, "tp": pos.tp}
                                    await mt5_call(mt5.order_send, request)
                                    logger.info(f"SL tightened to breakeven+ {new_sl} for {symbol} (Ticket: {ticket})")
                        continue
                    
            # 2. Proteção de Lucro em 2 Fases (Break-Even e Trailing Stop Dinâmico)
            current_price = pos.price_current
            symbol_info = await mt5_call(mt5.symbol_info, symbol)
            digits = symbol_info.digits if symbol_info else 5
            
            # Verifica direção e distância atual de lucro
            is_in_profit = (direction == "BUY" and current_price > pos.price_open) or \
                           (direction == "SELL" and current_price < pos.price_open)
            dist_current = (current_price - pos.price_open) if direction == "BUY" else (pos.price_open - current_price)
            
            if not is_in_profit or dist_current <= 0:
                continue

            # Parâmetros de Volatilidade baseados em ATR (M15)
            atr_value = None
            try:
                candles = await mt5_call(mt5.copy_rates_from_pos, symbol, mt5.TIMEFRAME_M15, 0, 20)
                if candles is not None and len(candles) > 0:
                    candles_list = [{"close": c[4], "high": c[2], "low": c[3]} for c in candles]
                    atr_value = _compute_atr(candles_list, 14)
                else:
                    logger.warning(f"Trailing stop bypassed for {symbol}: Insufficient M15 candles to calculate ATR")
            except Exception as e:
                logger.warning(f"Trailing stop bypassed for {symbol}: Error fetching candles for ATR - {e}")
            
            if atr_value is None or atr_value <= 0:
                continue
            
            # --- Configuração dos Degraus de Proteção ---
            # Degrau 1: Break-Even (Risco Zero) logo aos 1.0x ATR
            breakeven_threshold = atr_value * 1.0
            # Buffer de Break-Even: cobre spread e comissões (mínimo de 15 pontos ou 0.1x ATR)
            point = symbol_info.point if symbol_info else 0.0001
            be_buffer = max(point * 15, atr_value * 0.1)
            
            # Degrau 2: Trailing Stop Ativo aos 1.2x ATR
            activation_threshold = atr_value * 1.2
            trail_distance = atr_value * 1.0        # Persegue a 1.0x ATR atrás do preço atual
            step_min_move = atr_value * 0.15        # Mínimo de 0.15x ATR para registrar ordem no MT5
            
            target_sl = None
            stage_name = None
            
            # Avaliação de Fase (prioridade à mais avançada)
            if dist_current >= activation_threshold:
                # FASE 2: Trailing Stop Ativo
                # O SL persegue o preço, mas NUNCA pode ser inferior ao nível de Break-Even + buffer
                if direction == "BUY":
                    calc_trailing = current_price - trail_distance
                    target_sl = max(calc_trailing, pos.price_open + be_buffer)
                else:  # SELL
                    calc_trailing = current_price + trail_distance
                    target_sl = min(calc_trailing, pos.price_open - be_buffer)
                stage_name = "Trailing Stop"
                
            elif dist_current >= breakeven_threshold:
                # FASE 1: Break-Even
                if direction == "BUY":
                    target_sl = pos.price_open + be_buffer
                else:  # SELL
                    target_sl = pos.price_open - be_buffer
                stage_name = "Break-Even"
                
            if target_sl is not None:
                new_sl = 0.0
                # REGRA DE MONOTONICIDADE: O SL só pode mover-se a favor do LUCRO, NUNCA recuar
                if direction == "BUY":
                    if target_sl > (pos.sl + step_min_move):
                        new_sl = target_sl
                else:  # SELL
                    if pos.sl == 0.0 or target_sl < (pos.sl - step_min_move):
                        new_sl = target_sl
                        
                if new_sl > 0.0:
                    new_sl = round(new_sl, digits)
                    logger.info(f"{stage_name} Triggered for {symbol} (Ticket: {ticket}). Profit Dist: {dist_current:.5f} (ATR: {atr_value:.5f}). Moving SL from {pos.sl} to {new_sl}")
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": ticket,
                        "symbol": symbol,
                        "sl": float(new_sl),
                        "tp": float(pos.tp),
                    }
                    res = await mt5_call(mt5.order_send, request)
                    if res and res.retcode != mt5.TRADE_RETCODE_DONE:
                        logger.error(f"Failed to move {stage_name} SL for {symbol}: {res.comment}")

    async def evaluate_and_trade(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Coordinates the AI committee to analyze market and place a trade.
        Returns trade details dict on success, None otherwise.
        """
        t0 = time.perf_counter()
        result = await self._evaluate_and_trade_internal(symbol)
        t1 = time.perf_counter()
        latency = t1 - t0
        if latency > 30.0:
            logger.warning(f"Ciclo lento para {symbol}: {latency:.2f} segundos")
        else:
            logger.debug(f"Ciclo concluído para {symbol} em {latency:.2f}s")
        return result

    async def _evaluate_and_trade_internal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Internal logic for evaluate_and_trade."""
        self._reset_daily_counter()
        logger.info(f"=== Orchestrator starting evaluation for {symbol} ===")
        
        # Pre-flight: Check daily trade limit
        if len(self.trades_today) >= CONFIG.MAX_TRADES_PER_DAY:
            logger.warning(f"Daily trade limit reached ({CONFIG.MAX_TRADES_PER_DAY}). Skipping {symbol}.")
            return None
            
        # Pre-flight: Check if market is closed (Weekend)
        from datetime import timezone
        now_utc = datetime.now(timezone.utc)
        is_weekend = False
        # Forex and commodities close Friday 22:00 UTC (00:00/01:00 Greek time) and open Sunday 21:00/22:00 UTC
        if now_utc.weekday() == 4 and now_utc.hour >= 22: # Friday night
            is_weekend = True
        elif now_utc.weekday() == 5: # Saturday
            is_weekend = True
        elif now_utc.weekday() == 6 and now_utc.hour < 21: # Sunday before 21:00 UTC
            is_weekend = True
            
        # Identify cryptos which trade 24/7
        is_crypto = any(crypto in symbol.upper() for crypto in ["BTC", "ETH", "SOL", "XRP", "LTC", "ADA", "DOGE"])
        
        if is_weekend and not is_crypto:
            logger.info(f"Market for {symbol} is CLOSED (Weekend). Skipping evaluation.")
            return None
        
        # Pre-flight: Check account risk
        account = await mt5_call(mt5.account_info)
        if account:
            account_dict = account._asdict()
            if not await self.risk_manager.evaluate_account_risk(account_dict):
                logger.critical("Account risk check FAILED. Aborting trade.")
                return None
            
            # Check position limits
            positions = await mt5_call(mt5.positions_get)
            if positions:
                pos_list = [p._asdict() for p in positions]
                
                # Check if we already have an open position for THIS symbol (case-insensitive)
                symbol_positions = [p for p in pos_list if p.get("symbol", "").upper() == symbol.upper()]
                if symbol_positions:
                    logger.info(f"Already have an open position for {symbol}. Skipping.")
                    return None
                    
                if not self.risk_manager.evaluate_positions_risk(pos_list):
                    logger.critical("Position risk check FAILED. Aborting trade.")
                    return None
        
        # 1. Gather raw candle data
        candles = await mcp_client.get_candles_latest(symbol, count=100)
        if not candles:
            logger.error(f"Failed to get candles for {symbol}. Aborting.")
            return None
            
        # 2. Economic Calendar Veto
        calendar = await calendar_analyzer.get_upcoming_events(symbol)
        if calendar.get("veto"):
            logger.warning(f"Economic Calendar VETO for {symbol}. High impact events pending. Skipping trade.")
            return None
            
        # 3. Run Technical analysis + News (cached) concurrently
        market_result, news_result = await asyncio.gather(
            market_analyzer.analyze_candles(symbol, candles),
            self._get_news_cached(symbol),
            return_exceptions=True
        )
        
        # Handle exceptions from gather
        if isinstance(market_result, Exception):
            logger.error(f"Market analysis failed: {market_result}")
            market_result = {"trend": "NEUTRAL", "confidence": 0}
        else:
            asyncio.create_task(broadcaster.send_market_update(symbol, market_result.get("trend", "NEUTRAL"), int(market_result.get("confidence", 0))))
            
        if isinstance(news_result, Exception):
            logger.error(f"News analysis failed: {news_result}")
            news_result = {"sentiment": "NEUTRAL", "confidence": 0}
        else:
            asyncio.create_task(broadcaster.send_news_alert(symbol, news_result.get("sentiment", "NEUTRAL"), int(news_result.get("confidence", 0)), news_result.get("reasoning_summary", "")))
        
        # 3. Market Regime Flag (soft — let CIO decide, don't hard-veto)
        regime_warning = ""
        if market_result.get("regime") == "CHOPPY":
            logger.info(f"Market Regime is CHOPPY for {symbol}. Flagging for CIO (no longer a hard veto).")
            regime_warning = "WARNING: The market regime is currently CHOPPY. Be extra cautious but still trade if there is a clear directional signal with strong conviction."

        # 4. RAG: Consult past trades
        context_query = f"Symbol: {symbol}, Technical: {market_result.get('trend')}, News: {news_result.get('sentiment')}"
        try:
            from memory.trade_journal import trade_journal
            past_trades = await trade_journal.find_similar_context(context_query, symbol=symbol, limit=15)
            # Aggregate total stats
            wins, losses, total_profit, total_loss = 0, 0, 0.0, 0.0
            setup_stats = {"_global": {"wins": 0, "losses": 0, "total_profit": 0.0, "total_loss": 0.0}}
            
            for pt in past_trades:
                meta = pt.get("metadata", {})
                try:
                    profit = float(meta.get("profit", 0.0))
                except (ValueError, TypeError):
                    profit = 0.0
                setup_type = meta.get("setup_type", "UNKNOWN")
                
                if setup_type not in setup_stats:
                    setup_stats[setup_type] = {"wins": 0, "losses": 0, "total_profit": 0.0, "total_loss": 0.0}
                    
                if profit >= 0:
                    wins += 1
                    total_profit += profit
                    setup_stats[setup_type]["wins"] += 1
                    setup_stats[setup_type]["total_profit"] += profit
                    setup_stats["_global"]["wins"] += 1
                    setup_stats["_global"]["total_profit"] += profit
                else:
                    losses += 1
                    total_loss += abs(profit)
                    setup_stats[setup_type]["losses"] += 1
                    setup_stats[setup_type]["total_loss"] += abs(profit)
                    setup_stats["_global"]["losses"] += 1
                    setup_stats["_global"]["total_loss"] += abs(profit)
            
            total_global = setup_stats["_global"]["wins"] + setup_stats["_global"]["losses"]
            setup_stats["_global"]["win_rate"] = (setup_stats["_global"]["wins"] / total_global) if total_global > 0 else 0.5
            
            total = wins + losses
            win_rate = (wins / total * 100) if total > 0 else 0
            profit_factor = (total_profit / total_loss) if total_loss > 0 else (999.0 if total_profit > 0 else 0.0)
            
            rag_context = ""
            if total >= 5:
                rag_context += f"Contexto Estatístico Agregado: Win Rate {win_rate:.1f}%, Profit Factor {profit_factor:.2f} (N={total} amostras totais).\n"
                
                # Setup breakdown
                breakdown_lines = []
                for stype, stats in setup_stats.items():
                    if stype == "_global": continue
                    st_total = stats["wins"] + stats["losses"]
                    if st_total >= 5:
                        st_wr = (stats["wins"] / st_total * 100) if st_total > 0 else 0
                        st_pf = (stats["total_profit"] / stats["total_loss"]) if stats["total_loss"] > 0 else (999.0 if stats["total_profit"] > 0 else 0.0)
                        breakdown_lines.append(f"   - {stype}: Win Rate {st_wr:.1f}%, Profit Factor {st_pf:.2f} (N={st_total})")
                
                if breakdown_lines:
                    rag_context += "   Breakdown por Setup (apenas categorias com 5+ amostras):\n" + "\n".join(breakdown_lines)
            else:
                rag_context = f"Nenhum histórico semelhante estatisticamente significante encontrado (apenas {total} amostras). Avalia puramente com base nos dados atuais."
        except Exception as e:
            logger.error(f"Failed to query Qdrant for RAG: {e}")
            rag_context = "No historical context available."

        # 4. CIO Final Decision
        logger.info("Consulting CIO for final decision...")
        
        # Sanitize dicts to prevent prompt injection from raw news headlines
        safe_market = {k: v for k, v in market_result.items() if k in ["trend", "regime", "atr_14", "momentum", "volatility", "support_resistance_bounce_detected", "volume_breakout_detected", "confidence"]}
        safe_news = {k: v for k, v in news_result.items() if k in ["sentiment", "confidence", "summary"]}
        
        prompt = f"""You are the Chief Investment Officer (CIO) of an automated trading firm.
You must decide whether to place a trade for {symbol} right now.

Committee Reports:
1. Technical Analysis: {safe_market}
2. Fundamental/News Analysis: {safe_news}
3. Historical Statistics (RAG): {rag_context}
{regime_warning}

Use historical stats as probabilistic context, not as deterministic rules.

CRITICAL — STOP LOSS AND TAKE PROFIT RULES:
- The Technical Analysis includes 'atr_14' — this is the Average True Range on M15 candles, measured in raw price units.
- You MUST base your sl_distance and tp_distance on the ATR, NOT on fixed numbers.
- sl_distance = 1.5x to 2.0x the ATR (this keeps the stop loss OUTSIDE normal market noise).
- tp_distance = 2.0x to 3.0x the ATR (this gives a Risk:Reward ratio of at least 1:1.5).
- Example: If atr_14 = 45.0 for US30, then sl_distance should be 67.5 to 90.0, and tp_distance should be 90.0 to 135.0.
- Example: If atr_14 = 0.0020 for EURUSD, then sl_distance should be 0.0030 to 0.0040.
- NEVER set sl_distance smaller than 1.0x ATR — that guarantees being stopped out by noise.

TRADE DECISION RULES:
- You SHOULD trade when there is a reasonable directional bias. You do NOT need perfect conditions.
- BUY if the technical trend is BULLISH with confidence >= 55, even if news is NEUTRAL.
- SELL if the technical trend is BEARISH with confidence >= 55, even if news is NEUTRAL.
- Only HOLD if technicals are genuinely NEUTRAL with no clear direction, or if news DIRECTLY CONTRADICTS the technical signal (e.g., tech is BULLISH but news is strongly BEARISH).
- NEUTRAL news should NOT prevent trading. Most of the time news is neutral — that is normal.
- RESPECT DETERMINISTIC FLAGS:
  - If 'volume_breakout_detected': true, you MUST trade in the breakout direction (setup: VOLUME_BREAKOUT).
  - If 'sr_bounce_detected': true, prioritize 'SR_BOUNCE'.
- If the regime is 'CHOPPY', you may still trade if there is a clear signal (e.g., RSI divergence, SR bounce, or volume breakout).

Return ONLY a JSON object:
{{
    "action": "BUY" | "SELL" | "HOLD",
    "setup_type": "TREND_PULLBACK" | "TREND_BREAKOUT" | "MEAN_REVERSION" | "NEWS_MOMENTUM" | "SR_BOUNCE" | "VOLUME_BREAKOUT" | "OTHER",
    "reasoning": "Brief 1-2 sentence explanation including ATR value (MUST be a plain string, never a nested object)",
    "confidence": integer 0-100,
    "sl_distance": float,
    "tp_distance": float
}}
"""
        
        decision = await mistral_client.get_json_analysis(
            prompt=prompt,
            system_prompt="You are a pragmatic CIO. You balance risk management with opportunity capture. Missing good trades is as costly as taking bad ones. Output ONLY valid JSON."
        )
        
        if not decision:
            logger.error("CIO failed to return a decision.")
            return None
            
        action = decision.get("action", "HOLD")
        
        try:
            confidence = int(decision.get("confidence", 0))
        except (ValueError, TypeError):
            confidence = 0
            
        setup_type = decision.get("setup_type", "UNKNOWN")
        
        raw_reasoning = decision.get("reasoning", "")
        if isinstance(raw_reasoning, dict):
            reasoning_str = raw_reasoning.get("summary") or json.dumps(raw_reasoning)
        else:
            reasoning_str = str(raw_reasoning)
        decision["reasoning"] = reasoning_str

        logger.info(f"CIO Decision: {action} (Conf: {confidence}%, Setup: {setup_type}). Reasoning: {reasoning_str}")
        
        try:
            sl_broadcast = float(decision.get('sl_distance', 0.0))
            tp_broadcast = float(decision.get('tp_distance', 0.0))
        except (ValueError, TypeError):
            sl_broadcast, tp_broadcast = 0.0, 0.0
            
        asyncio.create_task(broadcaster.send_decision(
            symbol=symbol,
            action=action,
            confidence=confidence,
            reasoning=decision.get('reasoning', ''),
            sl=sl_broadcast,
            tp=tp_broadcast
        ))
        
        # 4. Execute if actionable
        if action in ("BUY", "SELL"):
            # Get symbol info for lot sizing
            symbol_info = await mt5_call(mt5.symbol_info, symbol)
            if not symbol_info:
                logger.error(f"Cannot get symbol info for {symbol}")
                return None
                
            tick = await mt5_call(mt5.symbol_info_tick, symbol)
            if not tick:
                logger.error(f"Cannot get tick data for {symbol}. Market may be closed.")
                return None
            current_price = tick.ask if action == "BUY" else tick.bid
            
            raw_sl_distance = decision.get("sl_distance")
            try:
                sl_distance = float(raw_sl_distance)
            except (ValueError, TypeError):
                sl_distance = 0.0
                
            if sl_distance <= 0:
                logger.error(f"LLM decision missing valid sl_distance for {symbol}. Aborting trade.")
                return None
            
            # --- CALCULATE KELLY POSITION SIZING ---
            risk_pct = self._calculate_kelly_risk_pct(setup_type, setup_stats)
            if risk_pct is None:
                logger.info(f"Not enough data for Kelly sizing on {setup_type}. Fallback to confidence ({confidence}%).")
                max_risk_pct = float(CONFIG.MAX_SINGLE_TRADE_LOSS_PCT)
                if confidence >= 85:
                    risk_pct = max_risk_pct * 1.0
                elif confidence >= 75:
                    risk_pct = max_risk_pct * 0.7
                elif confidence >= 65:
                    risk_pct = max_risk_pct * 0.5
                elif confidence >= 55:
                    risk_pct = max_risk_pct * 0.3
                else:
                    logger.warning(f"Confidence {confidence}% is below trade threshold (55%). Aborting.")
                    return None
            elif risk_pct <= 0:
                logger.warning(f"Kelly criterion risk is {risk_pct}% (zero edge). Aborting trade to protect capital.")
                return None
            else:
                logger.info(f"Kelly Sizing applied for {setup_type}: Risk {risk_pct:.2f}% of account.")
            
            # --- DOUBLE CHECK CONCURRENCY LIMITS BEFORE EXECUTION ---
            async with self._global_exec_lock:
                positions_before_exec = await mt5_call(mt5.positions_get)
                if positions_before_exec:
                    pos_list_exec = [p._asdict() for p in positions_before_exec]
                    # Check symbol again
                    sym_open = [p for p in pos_list_exec if p.get("symbol", "").upper() == symbol.upper()]
                    if sym_open:
                        logger.warning(f"Concurrency prevention: {symbol} was opened while waiting for LLM. Aborting.")
                        return None
                    # Check global limit again
                    if not self.risk_manager.evaluate_positions_risk(pos_list_exec):
                        logger.warning(f"Concurrency prevention: MAX_POSITIONS reached while waiting for LLM. Aborting {symbol}.")
                        return None
                # --------------------------------------------------------
                
                # Calculate actual SL price
                if action == "BUY":
                    sl_price = current_price - sl_distance
                else:
                    sl_price = current_price + sl_distance
                sl_price = round(sl_price, symbol_info.digits)
                
                # Determine exactly how much 1 lot loses if this SL is hit
                order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
                profit_for_1_lot = await mt5_call(mt5.order_calc_profit, order_type, symbol, 1.0, current_price, sl_price)
                
                if profit_for_1_lot is None:
                    logger.error(f"Failed to calculate profit for 1 lot on {symbol} natively. Using formula fallback.")
                    tick_val = getattr(symbol_info, 'trade_tick_value_loss', getattr(symbol_info, 'trade_tick_value', 0.0))
                    tick_size = getattr(symbol_info, 'trade_tick_size', 1.0)
                    if tick_size > 0:
                        sl_distance_monetary = (abs(current_price - sl_price) / tick_size) * tick_val
                    else:
                        sl_distance_monetary = 100.0 # Extreme fallback
                else:
                    sl_distance_monetary = abs(profit_for_1_lot)
                    
                account_info = await mt5_call(mt5.account_info)
                if not account_info:
                    logger.error("Failed to fetch account info for lot sizing. Aborting trade to protect capital.")
                    return None
                account_balance = account_info.balance
                
                lot_size = self._calculate_lot_size(risk_pct, symbol_info, account_balance, sl_distance_monetary)
                
                if lot_size <= 0:
                    logger.warning(f"Lot size calculated as {lot_size} (likely due to low risk_pct). Trade aborted.")
                    return None
                
                # --- PRE-FLIGHT MARGIN PROTECTION ---
                free_margin = getattr(account_info, 'margin_free', 0.0)
                required_margin = await mt5_call(mt5.order_calc_margin, order_type, symbol, lot_size, current_price)
                
                if required_margin is not None and free_margin > 0:
                    # Never commit more than 70% of available free margin to a single order
                    max_allowed_margin = free_margin * 0.70
                    if required_margin > max_allowed_margin:
                        margin_scale = max_allowed_margin / required_margin
                        scaled_lot = lot_size * margin_scale
                        lot_step = symbol_info.volume_step if symbol_info else 0.01
                        min_lot = symbol_info.volume_min if symbol_info else 0.01
                        
                        if lot_step > 0:
                            scaled_lot = round(scaled_lot / lot_step) * lot_step
                        scaled_lot = round(scaled_lot, 2)
                        
                        if scaled_lot < min_lot:
                            logger.warning(
                                f"Margin protection: Required margin ${required_margin:.2f} exceeds 70% free margin "
                                f"(${max_allowed_margin:.2f}). Scaling down results in {scaled_lot} < min_lot ({min_lot}). Aborting trade."
                            )
                            return None
                            
                        logger.warning(
                            f"Margin protection active for {symbol}: Required margin ${required_margin:.2f} > 70% free margin (${max_allowed_margin:.2f}). "
                            f"Scaling down lot from {lot_size} to {scaled_lot}."
                        )
                        lot_size = scaled_lot
                
                signal = {
                    "action": action,
                    "symbol": symbol,
                    "lot_size": lot_size,
                    "sl_distance": sl_distance,
                    "tp_distance": float(decision.get("tp_distance", sl_distance * 1.5))
                }
                
                logger.info(f"Forwarding signal to Trade Executor: {signal}")
                trade_result = await trade_executor.execute_signal(signal)
            
            if trade_result:
                self.trades_today.append(trade_result)
                logger.info(f"Trade #{len(self.trades_today)} today. Ticket: {trade_result.get('ticket')}")
                
                # Cache the context for the Trade Monitor
                context = f"CIO Reasoning: {decision.get('reasoning')} | News: {news_result.get('sentiment')} | Tech: {market_result.get('trend')}"
                asyncio.create_task(broadcaster.cache_trade_context(trade_result.get('ticket'), context, confidence, setup_type))
                
                return trade_result
            else:
                logger.error("Trade Executor failed to execute signal.")
                return None
        
        logger.info(f"No trade for {symbol}. Decision: HOLD.")
        return None

orchestrator = Orchestrator()
