import asyncio
import logging
from logging.handlers import RotatingFileHandler
import os
import time
import MetaTrader5 as mt5
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
from core.ipc_pipe_server import NamedPipeIPCServer
from risk.dynamic_risk import dynamic_risk_manager
import numpy as np

logger = logging.getLogger(__name__)

# Setup persistent logging for AI decisions
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'ai_trading.log')

file_handler = RotatingFileHandler(log_file, maxBytes=10_000_000, backupCount=5)
file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(file_handler)

# News cache: stores {symbol: {"result": ..., "timestamp": ...}}
_news_cache: Dict[str, Dict[str, Any]] = {}
_NEWS_CACHE_TTL = 180  # 3 minutes


class Orchestrator:
    """CIO Agent — coordinates the AI committee and makes final trading decisions."""
    
    def __init__(self):
        self.risk_manager = RiskManager()
        self.dynamic_risk_manager = dynamic_risk_manager
        self.pipe_server = NamedPipeIPCServer(self._handle_pipe_request)
        self.trades_today: list[Dict[str, Any]] = []
        self.trade_day: str = ""
        self._global_exec_lock = asyncio.Lock()
        
    def _reset_daily_counter(self):
        """Reset trade counter at the start of each new day."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self.trade_day != today:
            self.trades_today = []
            self.trade_day = today

    async def _handle_pipe_request(self, payload: dict) -> dict:
        """
        Handles real-time handshake request via Named Pipe from MT5.
        Executes sub-millisecond evaluation (< 5ms) returning:
        {"approved": bool, "risk_pct": float, "sl_mult": float, "tp_mult": float, "reason": str}
        """
        symbol = payload.get("sym") or payload.get("symbol", "EURUSD")
        direction = payload.get("dir") or payload.get("direction", "BUY")
        setup_type = payload.get("setup") or payload.get("setup_type", "H1_BULL_PULLBACK_M5")
        
        # 1. Fundamental LLM / News Veto
        is_vetoed = False
        try:
            is_vetoed = await self._apply_llm_veto(symbol, {"direction": direction})
        except Exception as e:
            logger.error(f"[Pipe IPC] LLM Veto check error: {e}")
            is_vetoed = False

        if is_vetoed:
            logger.warning(f"[Pipe IPC VETO] Order {direction} {symbol} blocked by AI sentiment.")
            return {
                "approved": False,
                "risk_pct": 0.0,
                "sl_mult": float(payload.get("sl_mult", 1.2)),
                "tp_mult": float(payload.get("tp_mult", 1.25)),
                "reason": "VETOED_BY_AI"
            }

        # 2. Bayesian Half-Kelly Risk Sizing
        setup_stats = getattr(self, "setup_stats", {})
        kelly_risk = self._calculate_kelly_risk_pct(setup_type, setup_stats=setup_stats)
        if kelly_risk is None or kelly_risk <= 0.0:
            kelly_risk = float(getattr(CONFIG, "MAX_SINGLE_TRADE_LOSS_PCT", 1.0))

        # 3. Dynamic Portfolio Risk Scaling (EWMA Covariance + Online PCA ENB)
        scaled_risk = kelly_risk
        try:
            import MetaTrader5 as mt5
            positions = await mt5_call(mt5.positions_get)
            if positions:
                symbols_set = list({p.symbol for p in positions} | {symbol})
                returns_list = []
                for sym in symbols_set:
                    rates = await mt5_call(mt5.copy_rates_from_pos, sym, mt5.TIMEFRAME_M5, 0, 51)
                    if rates is not None and len(rates) >= 2:
                        closes = np.array([r[4] for r in rates], dtype=np.float64)
                        rets = np.diff(closes) / closes[:-1]
                        returns_list.append(rets[-50:])
                
                if returns_list and len(returns_list) == len(symbols_set):
                    returns_matrix = np.column_stack(returns_list)
                    weights = []
                    for sym in symbols_set:
                        if sym == symbol:
                            weights.append(kelly_risk if direction == "BUY" else -kelly_risk)
                        else:
                            pos = next((p for p in positions if p.symbol == sym), None)
                            w = kelly_risk if pos and pos.type == mt5.ORDER_TYPE_BUY else -kelly_risk
                            weights.append(w)
                    
                    scale = self.dynamic_risk_manager.evaluate_new_trade_scaling(
                        returns_matrix=returns_matrix,
                        risk_weights=np.array(weights, dtype=np.float64)
                    )
                    scaled_risk = round(kelly_risk * scale, 4)
        except Exception as e:
            logger.debug(f"[Pipe IPC] Dynamic risk scaling skipped: {e}")

        logger.info(f"[Pipe IPC APPROVED] {direction} {symbol} authorized with Risk={scaled_risk:.2f}% (Kelly Base: {kelly_risk:.2f}%)")
        return {
            "approved": True,
            "risk_pct": scaled_risk,
            "sl_mult": float(payload.get("sl_mult", 1.2)),
            "tp_mult": float(payload.get("tp_mult", 1.25)),
            "reason": "APPROVED_BY_AI"
        }
    
    def _calculate_kelly_risk_pct(
        self,
        setup_type: str,
        setup_stats: dict,
        meta_prob: Optional[float] = None,
        meta_threshold: float = 0.55,
        min_trades_full_kelly: int = 20,
        prior_wr: float = 0.50,
        prior_rr: float = 1.25,
        prior_strength: float = 10.0
    ) -> Optional[float]:
        """
        Calculates Risk % using Half-Kelly with Double Bayesian Shrinkage (W and R).
        Prevents division by zero on winning streaks and regularizes outliers.
        """
        if meta_prob is not None and meta_prob < meta_threshold:
            return 0.0  # Vetoed by Meta-Model

        stats = setup_stats.get(setup_type, {})
        wins = int(stats.get("wins", 0))
        losses = int(stats.get("losses", 0))
        total_profit = abs(float(stats.get("total_profit", 0.0)))
        total_loss = abs(float(stats.get("total_loss", 0.0)))
        n = wins + losses

        if n < 3:
            return None  # Insufficient data, fallback to legacy static risk

        # Bayesian Posterior for Win Rate W (Beta-Binomial)
        emp_W = (wins + prior_wr * prior_strength) / (n + prior_strength)
        W = (0.60 * meta_prob + 0.40 * emp_W) if meta_prob is not None else emp_W

        # Bayesian Posterior for Payoff Ratio R (regularized against prior_rr to avoid div-by-zero on 0 losses)
        avg_win = (total_profit / wins) if wins > 0 else prior_rr
        avg_loss = (total_loss / losses) if losses > 0 else 1.0
        raw_R = avg_win / max(avg_loss, 1e-8)
        R = (n * raw_R + prior_strength * prior_rr) / (n + prior_strength)

        if R <= 0.0:
            return 0.0

        kelly_fraction = W - ((1.0 - W) / R)
        if kelly_fraction <= 0.0:
            return 0.0

        half_kelly = 0.5 * kelly_fraction
        sample_scale = min(1.0, max(0.25, n / float(min_trades_full_kelly)))
        risk_pct = half_kelly * sample_scale * 100.0
        return min(risk_pct, float(CONFIG.MAX_SINGLE_TRADE_LOSS_PCT))

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
            actual_risk_usd_if_min_lot = sl_distance_monetary * min_lot
            actual_risk_pct = (actual_risk_usd_if_min_lot / account_balance) * 100
            
            hard_limit = float(CONFIG.MAX_SINGLE_TRADE_LOSS_PCT)
            
            if actual_risk_pct <= hard_limit:
                logger.info(f"Calculated lot ({safe_lot}) < min ({min_lot}). Forcing {min_lot} (Actual risk: {actual_risk_pct:.2f}% <= Hard Limit {hard_limit}%).")
                safe_lot = min_lot
            else:
                logger.warning(f"Broker min lot ({min_lot}) would cause {actual_risk_pct:.2f}% loss. Breaches hard limit ({hard_limit}%). Aborting.")
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
            # If position was opened by on-chart MQL5 Twin (magic == 202609), skip!
            # The EA on the MT5 chart manages its own breakeven and trailing stop on every tick.
            if getattr(pos, 'magic', 0) == 202609:
                continue
                
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
            if not symbol_info:
                logger.error(f"Failed to get symbol_info for {symbol}. Skipping SL trailing.")
                continue
            digits = symbol_info.digits
            
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
            
            # Institutional Breakeven Buffer calculation
            tick = await mt5_call(mt5.symbol_info_tick, symbol)
            if not tick or tick.time == 0:
                continue
                
            current_spread = tick.ask - tick.bid
            
            # Commission calculation
            commission_per_lot = 7.0 # Example: 7 USD per round lot
            volume = pos.volume
            tick_size = symbol_info.trade_tick_size
            tick_value = symbol_info.trade_tick_value
            
            if volume > 0 and tick_value > 0:
                commission_in_account_currency = commission_per_lot * volume
                commission_cost_in_price = (commission_in_account_currency * tick_size) / (volume * tick_value)
            else:
                commission_cost_in_price = 0.0
            
            stops_level = symbol_info.trade_stops_level * symbol_info.point
            be_buffer = max(
                current_spread * 2.0,           # Absorve alargamento atual
                commission_cost_in_price * 2.0, # Cobre comissões e swaps curtos
                atr_value * 0.05,               # Fundo mínimo de volatilidade
                stops_level                     # Respeita limite da corretora
            )
            
            # Para BUY, SL compara com Bid. Precisamos de superar o spread pago na entrada (Ask) + comissão.
            be_price_buy = pos.price_open + commission_cost_in_price + be_buffer
            # Para SELL, SL compara com Ask. Precisamos de cair abaixo do spread + comissão.
            be_price_sell = pos.price_open - commission_cost_in_price - be_buffer
            
            # Degrau 2: Trailing Stop Ativo aos 1.2x ATR
            activation_threshold = atr_value * 1.2
            trail_distance = atr_value * 1.0        # Persegue a 1.0x ATR atrás do preço atual
            step_min_move = atr_value * 0.15        # Mínimo de 0.15x ATR para registrar ordem no MT5
            
            target_sl = None
            stage_name = None
            
            # Avaliação de Fase (prioridade à mais avançada)
            if dist_current >= activation_threshold:
                # FASE 2: Trailing Stop Ativo
                if direction == "BUY":
                    calc_trailing = current_price - trail_distance
                    target_sl = max(calc_trailing, be_price_buy)
                else:  # SELL
                    calc_trailing = current_price + trail_distance
                    target_sl = min(calc_trailing, be_price_sell)
                stage_name = "Trailing Stop"
                
            elif dist_current >= breakeven_threshold:
                # FASE 1: Break-Even
                if direction == "BUY":
                    target_sl = be_price_buy
                else:  # SELL
                    target_sl = be_price_sell
                stage_name = "Break-Even"
                
            if target_sl is not None:
                new_sl = 0.0
                current_sl = pos.sl
                # REGRA DE MONOTONICIDADE (Stricter): O SL só pode mover-se a favor do LUCRO, NUNCA recuar
                if direction == "BUY":
                    # target_sl cannot be lower than current_sl
                    if current_sl > 0:
                        target_sl = max(target_sl, current_sl)
                    if target_sl > (current_sl + step_min_move):
                        new_sl = target_sl
                else:  # SELL
                    # target_sl cannot be higher than current_sl
                    if current_sl > 0:
                        target_sl = min(target_sl, current_sl)
                    if current_sl == 0.0 or target_sl < (current_sl - step_min_move):
                        new_sl = target_sl
                        
                if new_sl > 0.0:
                    new_sl = round(new_sl, digits)
                    logger.info(f"{stage_name} Triggered for {symbol} (Ticket: {ticket}). Profit Dist: {dist_current:.5f} (ATR: {atr_value:.5f}). Moving SL from {current_sl} to {new_sl}")
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": ticket,
                        "symbol": symbol,
                        "sl": float(new_sl),
                        "tp": float(pos.tp),
                    }
                    res = await mt5_call(mt5.order_send, request)
                    if res and res.retcode != mt5.TRADE_RETCODE_DONE:
                        if res.retcode == 10016: # mt5.TRADE_RETCODE_INVALID_STOPS
                            logger.debug(f"SL too close for {symbol}, skipping modification (broker protection)")
                        else:
                            logger.error(f"Failed to move {stage_name} SL for {symbol}: {res.comment}")

    async def evaluate_and_trade(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Coordinates the AI committee to analyze market and place a trade.
        Returns trade details dict on success, None otherwise.
        """
        if getattr(CONFIG, "TWIN_MODE", True) and symbol.upper() in getattr(CONFIG, "twin_symbols_list", ["EURUSD"]):
            logger.debug(f"{symbol} is managed by on-chart MQL5 Twin EA. Python direct execution bypassed.")
            return None

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
            
        # Synchronization: Verify if the latest closed candle is truly closed according to Broker Time
        tick = await mt5_call(mt5.symbol_info_tick, symbol)
        
        latest_closed_time = candles[-1]['time'].timestamp()
        
        if not tick or tick.time == 0:
            broker_current_time = latest_closed_time + 300 # Fallback
        else:
            broker_current_time = tick.time
            
        timeframe_seconds = 300 # M5
        
        seconds_since_close = broker_current_time - (latest_closed_time + timeframe_seconds)
        
        if seconds_since_close < 0:
            return None # The previous candle hasn't officially closed in broker time yet
            
        if seconds_since_close > 10:
            logger.warning(f"Decision lag for {symbol}: {seconds_since_close}s")
            
        # 2. Economic Calendar Veto
        calendar = await calendar_analyzer.get_upcoming_events(symbol)
        if calendar.get("veto"):
            logger.warning(f"Economic Calendar VETO for {symbol} [{calendar.get('events')}]. Skipping trade.")
            return None
            
        # 3. Get H1 Candles for Trend Context
        h1_candles = await mt5_call(mt5.copy_rates_from_pos, symbol, mt5.TIMEFRAME_H1, 1, 100)
        if not h1_candles:
            logger.error(f"Failed to get H1 candles for {symbol}. Aborting.")
            return None
            
        h1_candles = [{"time": c[0], "open": c[1], "high": c[2], "low": c[3], "close": c[4]} for c in h1_candles]
        
        # 4. Evaluate Quantitative Signal
        quant_signal = await self._evaluate_quant_signal(symbol, candles, h1_candles, broker_current_time)
        
        # Logging obligations for WFO/EA MQL5 translation
        import json
        log_entry = {
            "bar_time_m5": quant_signal["bar_time_m5"],
            "h1_ctx_bar_time": quant_signal["h1_ctx_bar_time"],
            "symbol": symbol,
            "direction": quant_signal["direction"],
            "setup_type": quant_signal["setup_type"],
            "h1_adx": quant_signal["h1_adx"],
            "m5_atr": quant_signal["m5_atr"],
            "m5_atr_median": quant_signal["m5_atr_median"],
            "m5_ema20": quant_signal["m5_ema20"],
            "trigger_open": quant_signal["trigger_open"],
            "trigger_high": quant_signal["trigger_high"],
            "trigger_low": quant_signal["trigger_low"],
            "trigger_close": quant_signal["trigger_close"],
            "vetoed_by_llm": False,
            "reason_code": quant_signal["reason_code"]
        }
        
        if not quant_signal["direction"]:
            logger.info(f"Quant Signal REJECTED for {symbol}: {json.dumps(log_entry)}")
            return None
            
        # 5. Apply LLM Veto Overlay
        vetoed = await self._apply_llm_veto(symbol, quant_signal)
        log_entry["vetoed_by_llm"] = vetoed
        
        if vetoed:
            log_entry["reason_code"] = "LLM_VETO"
            logger.warning(f"Trade VETOED by LLM: {json.dumps(log_entry)}")
            return None
            
        logger.info(f"Quant Signal APPROVED: {json.dumps(log_entry)}")
        
        action = quant_signal["direction"]
        confidence = quant_signal["confidence"]
        setup_type = quant_signal["setup_type"]
        
        # Deterministic SL/TP based strictly on Volatility (ATR)
        # Using standard 1.5x ATR for SL and 2.0x ATR for TP
        atr_14 = quant_signal["m5_atr"]
        sl_distance = 1.5 * atr_14
        tp_distance = 2.0 * atr_14
        
        asyncio.create_task(broadcaster.send_decision(
            symbol=symbol,
            action=action,
            confidence=confidence,
            reasoning=f"Quant Signal: {setup_type} (ADX: {quant_signal['h1_adx']:.1f}, ATR: {atr_14:.5f})",
            sl=sl_distance,
            tp=tp_distance
        ))
        
        # 4. Execute if actionable
        if action in ("BUY", "SELL"):
            if atr_14 <= 0:
                logger.error(f"Invalid ATR ({atr_14}) for {symbol}. Cannot calculate SL/TP distances.")
                return None
                
            setup_stats = {"_global": {"win_rate": 0.5}}
                
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
                    "tp_distance": tp_distance
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

    def _calculate_ema(self, series: list, period: int) -> float:
        import numpy as np
        if len(series) < period:
            return series[-1] if series else 0.0
        ema = np.mean(series[:period])
        multiplier = 2 / (period + 1)
        for price in series[period:]:
            ema = (price - ema) * multiplier + ema
        return ema

    def _calculate_atr(self, candles: list, period: int) -> float:
        if len(candles) <= period:
            return 0.0
        tr_list = []
        for i in range(1, len(candles)):
            c = candles[i]
            pc = candles[i-1]
            high_low = c['high'] - c['low']
            high_close = abs(c['high'] - pc['close'])
            low_close = abs(c['low'] - pc['close'])
            tr_list.append(max(high_low, high_close, low_close))
            
        # Wilder's Smoothing (RMA/SMMA)
        atr = sum(tr_list[:period]) / period
        for i in range(period, len(tr_list)):
            atr = (atr * (period - 1) + tr_list[i]) / period
        return float(atr)
        
    def _calculate_atr_series(self, candles: list, period: int, lookback: int = 50) -> list:
        if len(candles) <= period + lookback:
            return [0.0]
        tr_list = []
        for i in range(1, len(candles)):
            c = candles[i]
            pc = candles[i-1]
            high_low = c['high'] - c['low']
            high_close = abs(c['high'] - pc['close'])
            low_close = abs(c['low'] - pc['close'])
            tr_list.append(max(high_low, high_close, low_close))
        
        atr = sum(tr_list[:period]) / period
        atr_full_series = [atr]
        for i in range(period, len(tr_list)):
            atr = (atr * (period - 1) + tr_list[i]) / period
            atr_full_series.append(atr)
            
        # Return only the last 'lookback' values
        return atr_full_series[-lookback:]

    def _calculate_adx(self, candles: list, period: int) -> float:
        import numpy as np
        if len(candles) <= period * 2:
            return 0.0
            
        tr_list = []
        plus_dm_list = []
        minus_dm_list = []
        
        for i in range(1, len(candles)):
            c = candles[i]
            pc = candles[i-1]
            
            high_low = c['high'] - c['low']
            high_close = abs(c['high'] - pc['close'])
            low_close = abs(c['low'] - pc['close'])
            tr_list.append(max(high_low, high_close, low_close))
            
            up_move = c['high'] - pc['high']
            down_move = pc['low'] - c['low']
            
            if up_move > down_move and up_move > 0:
                plus_dm_list.append(up_move)
                minus_dm_list.append(0.0)
            elif down_move > up_move and down_move > 0:
                plus_dm_list.append(0.0)
                minus_dm_list.append(down_move)
            else:
                plus_dm_list.append(0.0)
                minus_dm_list.append(0.0)
                
        def _smoothed(series, length):
            sm = [np.sum(series[:length])]
            for val in series[length:]:
                sm.append(sm[-1] - (sm[-1] / length) + val)
            return sm
            
        tr_smooth = _smoothed(tr_list, period)
        plus_dm_smooth = _smoothed(plus_dm_list, period)
        minus_dm_smooth = _smoothed(minus_dm_list, period)
        
        dx_list = []
        for i in range(len(tr_smooth)):
            tr = tr_smooth[i]
            if tr == 0:
                dx_list.append(0.0)
                continue
            plus_di = 100 * (plus_dm_smooth[i] / tr)
            minus_di = 100 * (minus_dm_smooth[i] / tr)
            if (plus_di + minus_di) == 0:
                dx_list.append(0.0)
            else:
                dx_list.append(100 * abs(plus_di - minus_di) / (plus_di + minus_di))
                
        if len(dx_list) < period:
            return 0.0
            
        adx = np.mean(dx_list[:period])
        for dx in dx_list[period:]:
            adx = (adx * (period - 1) + dx) / period
        return float(adx)

    async def _evaluate_quant_signal(self, symbol: str, closed_candles: list, h1_candles: list, broker_time: int):
        import numpy as np
        from datetime import timezone
        
        last_candle = closed_candles[-1]
        
        # 1. FILTRO DE JANELA DE LIQUIDEZ (Horário Londres + NY: 08:00 - 16:00 UTC)
        candle_dt = datetime.fromtimestamp(last_candle['time'], tz=timezone.utc)
        if candle_dt.hour < 8 or candle_dt.hour > 16:
            return {
                "direction": None, "setup_type": "NONE", "confidence": 0,
                "h1_adx": 0.0, "m5_atr": 0.0, "m5_atr_median": 0.0, "m5_ema20": 0.0,
                "reason_code": "OUTSIDE_LIQUIDITY_WINDOW",
                "h1_ctx_bar_time": int(h1_candles[-1]['time']),
                "bar_time_m5": int(last_candle['time']),
                "trigger_open": float(last_candle['open']),
                "trigger_high": float(last_candle['high']),
                "trigger_low": float(last_candle['low']),
                "trigger_close": float(last_candle['close'])
            }
        
        # 2. FILTRO DE TENDÊNCIA H1 (WFO: EMA 25/70 + ADX 30)
        h1_closes = [c['close'] for c in h1_candles]
        h1_ema_fast = self._calculate_ema(h1_closes, 25)
        h1_ema_slow = self._calculate_ema(h1_closes, 70)
        h1_adx = self._calculate_adx(h1_candles, 14)
        
        m5_closes = [c['close'] for c in closed_candles]
        m5_atr = self._calculate_atr(closed_candles, 14)
        m5_atr_series = self._calculate_atr_series(closed_candles, 14, lookback=50)
        m5_atr_median = np.median(m5_atr_series) if m5_atr_series else m5_atr
        m5_ema20 = self._calculate_ema(m5_closes, 20)
        
        base_signal = {
            "direction": None,
            "setup_type": "NONE",
            "confidence": 0,
            "h1_adx": float(h1_adx),
            "m5_atr": float(m5_atr),
            "m5_atr_median": float(m5_atr_median),
            "m5_ema20": float(m5_ema20),
            "reason_code": "VALID",
            "h1_ctx_bar_time": int(h1_candles[-1]['time']),
            "bar_time_m5": int(last_candle['time']),
            "trigger_open": float(last_candle['open']),
            "trigger_high": float(last_candle['high']),
            "trigger_low": float(last_candle['low']),
            "trigger_close": float(last_candle['close'])
        }
        
        # Só operamos em forte tendência definida (ADX >= 30.0)
        if h1_adx < 30.0:
            base_signal["reason_code"] = "ADX_LOW"
            return base_signal
        
        trend_bullish = h1_ema_fast > h1_ema_slow
        trend_bearish = h1_ema_fast < h1_ema_slow
        
        # Filtro de Volatilidade ATR (1.4x a 2.5x mediana)
        if m5_atr < (m5_atr_median * 1.4):
            base_signal["reason_code"] = "ATR_LOW"
            return base_signal
        if m5_atr > (m5_atr_median * 2.5):
            base_signal["reason_code"] = "ATR_HIGH"
            return base_signal
        
        # 3. GATILHO DE ENTRADA (M5 Pullback + Filtro de Rejeição de Price Action)
        mid_candle = (last_candle['high'] + last_candle['low']) * 0.5
        if trend_bullish:
            rejection = last_candle['close'] > mid_candle
            if last_candle['low'] <= m5_ema20 and last_candle['close'] > m5_ema20 and rejection:
                base_signal["direction"] = "BUY"
                base_signal["setup_type"] = "H1_BULL_PULLBACK_M5"
                base_signal["confidence"] = min(95, int(60 + (h1_adx - 30.0) * 2))
            else:
                base_signal["reason_code"] = "NO_BULL_PULLBACK"
                
        elif trend_bearish:
            rejection = last_candle['close'] < mid_candle
            if last_candle['high'] >= m5_ema20 and last_candle['close'] < m5_ema20 and rejection:
                base_signal["direction"] = "SELL"
                base_signal["setup_type"] = "H1_BEAR_PULLBACK_M5"
                base_signal["confidence"] = min(95, int(60 + (h1_adx - 30.0) * 2))
            else:
                base_signal["reason_code"] = "NO_BEAR_PULLBACK"
        else:
            base_signal["reason_code"] = "NO_TREND"
                
        return base_signal

    async def _apply_llm_veto(self, symbol: str, quant_signal: dict) -> bool:
        if not quant_signal:
            return False
        
        try:
            llm_analysis = await self._get_news_cached(symbol)
        except Exception as e:
            logger.error(f"News analysis failed during LLM veto check: {e}")
            return False
            
        sentiment = llm_analysis.get("sentiment", "NEUTRAL")  # "BULLISH", "BEARISH", "NEUTRAL"
        llm_confidence = int(llm_analysis.get("confidence", 0))
        
        # Veto absoluto apenas em extrema convicção contrária
        if quant_signal["direction"] == "BUY" and sentiment == "BEARISH" and llm_confidence >= 80:
            logger.warning(f"[LLM VETO] {symbol}: BUY bloqueado por sentimento BEARISH extremo ({llm_confidence}%)")
            return True
            
        if quant_signal["direction"] == "SELL" and sentiment == "BULLISH" and llm_confidence >= 80:
            logger.warning(f"[LLM VETO] {symbol}: SELL bloqueado por sentimento BULLISH extremo ({llm_confidence}%)")
            return True
        
        return False

orchestrator = Orchestrator()
