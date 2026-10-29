import logging
from typing import Dict, Any, List
from agents.llm.mistral_client import mistral_client

logger = logging.getLogger(__name__)


def _compute_sma(candles: List[Dict[str, Any]], period: int) -> float:
    """Compute Simple Moving Average over the last `period` candles."""
    if len(candles) < period:
        return 0.0
    closes = [float(c.get('close', 0)) for c in candles[-period:]]
    return sum(closes) / period

def _compute_bbw(candles: List[Dict[str, Any]], period: int = 20, num_std_dev: float = 2.0) -> float:
    """Compute Bollinger Band Width (BBW) over the last `period` candles as % of price."""
    if len(candles) < period:
        return 0.0
    import math
    closes = [float(c.get('close', 0)) for c in candles[-period:]]
    sma = sum(closes) / period
    variance = sum((c - sma) ** 2 for c in closes) / period
    std_dev = math.sqrt(variance)
    upper_band = sma + (num_std_dev * std_dev)
    lower_band = sma - (num_std_dev * std_dev)
    if sma == 0:
        return 0.0
    return (upper_band - lower_band) / sma * 100



def _compute_rsi(candles: List[Dict[str, Any]], period: int = 14) -> float:
    """Compute RSI (Relative Strength Index) over the last `period` candles."""
    if len(candles) < period + 1:
        return 50.0  # neutral default
    closes = [float(c.get('close', 0)) for c in candles[-(period + 1):]]
    gains, losses = [], []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        if delta > 0:
            gains.append(delta)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(delta))
    avg_gain = sum(gains) / period if gains else 0.0
    avg_loss = sum(losses) / period if losses else 0.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def _compute_atr(candles: List[Dict[str, Any]], period: int = 14) -> float:
    """Compute Average True Range (ATR) over the last `period` candles."""
    if len(candles) < period + 1:
        return 0.0
    
    true_ranges = []
    for i in range(1, len(candles)):
        high = float(candles[i].get('high', 0))
        low = float(candles[i].get('low', 0))
        prev_close = float(candles[i-1].get('close', 0))
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_ranges.append(max(tr1, tr2, tr3))
        
    if not true_ranges:
        return 0.0
        
    recent_trs = true_ranges[-period:]
    return sum(recent_trs) / len(recent_trs)

def _detect_sr_bounce(candles: List[Dict[str, Any]], atr: float) -> bool:
    """
    Detects if the current or previous candle touched a confirmed support/resistance
    pivot point and rejected it.
    """
    if len(candles) < 50 or atr == 0:
        return False
        
    tolerance = atr * 0.5
    
    # 1. Find Confirmed Pivot Points (N=3)
    highs = [float(c.get('high', 0)) for c in candles]
    lows = [float(c.get('low', 0)) for c in candles]
    closes = [float(c.get('close', 0)) for c in candles]
    opens = [float(c.get('open', 0)) for c in candles]
    
    resistances = []
    supports = []
    
    N = 3
    for i in range(N, len(candles) - N):
        is_high = True
        for j in range(1, N + 1):
            if highs[i] < highs[i-j] or highs[i] < highs[i+j]:
                is_high = False
                break
        if is_high:
            resistances.append(highs[i])
            
        is_low = True
        for j in range(1, N + 1):
            if lows[i] > lows[i-j] or lows[i] > lows[i+j]:
                is_low = False
                break
        if is_low:
            supports.append(lows[i])
            
    if not resistances and not supports:
        return False
        
    # 2. Check if recent candles touched and rejected these levels
    for idx in [-1, -2]:
        curr_high = highs[idx]
        curr_low = lows[idx]
        curr_close = closes[idx]
        curr_open = opens[idx]
        
        for r in resistances:
            if abs(curr_high - r) <= tolerance:
                if curr_close < curr_open: # Rejected downwards
                    return True
                    
        for s in supports:
            if abs(curr_low - s) <= tolerance:
                if curr_close > curr_open: # Rejected upwards
                    return True
                    
    return False

def _detect_volume_breakout(candles: List[Dict[str, Any]], regime: str, volume_multiplier: float = 1.5) -> bool:
    """
    Detects if the current candle breaks a 20-period structure with high volume.
    Aborts if the regime is CHOPPY.
    """
    if regime == "CHOPPY" or len(candles) < 21:
        return False
        
    volumes = [float(c.get('tick_volume', 0)) for c in candles]
    highs = [float(c.get('high', 0)) for c in candles]
    lows = [float(c.get('low', 0)) for c in candles]
    closes = [float(c.get('close', 0)) for c in candles]
    
    recent_vols = volumes[-21:-1]
    avg_vol = sum(recent_vols) / len(recent_vols) if recent_vols else 0
    
    curr_vol = volumes[-1]
    curr_close = closes[-1]
    
    if curr_vol <= avg_vol * volume_multiplier:
        return False
        
    prev_20_highs = highs[-21:-1]
    prev_20_lows = lows[-21:-1]
    
    max_high = max(prev_20_highs)
    min_low = min(prev_20_lows)
    
    if curr_close > max_high or curr_close < min_low:
        return True
        
    return False


class MarketAnalyzer:
    """Technical analysis agent. Computes local indicators and asks LLM for interpretation."""
    
    async def analyze_candles(self, symbol: str, candles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Performs technical analysis on candlestick data.
        Computes indicators locally (SMA, RSI) then sends to LLM for interpretation.
        """
        if not candles or len(candles) < 2:
            logger.error(f"Insufficient candle data for {symbol}")
            return {"trend": "NEUTRAL", "support_level": 0.0, "resistance_level": 0.0, "confidence": 0}
        
        logger.info(f"MarketAnalyzer analyzing {len(candles)} candles for {symbol}...")
        
        # Compute local indicators
        sma_20 = _compute_sma(candles, 20)
        sma_50 = _compute_sma(candles, 50)
        rsi_14 = _compute_rsi(candles, 14)
        bbw_20 = _compute_bbw(candles, 20)
        atr_14 = _compute_atr(candles, 14)
        
        current_price = float(candles[-1].get('close', 0))
        high_range = max(float(c.get('high', 0)) for c in candles[-50:])
        low_range = min(float(c.get('low', 0)) for c in candles[-50:])
        
        # Send last 50 candles as OHLC summary
        recent_data = []
        for c in candles[-50:]:
            recent_data.append(f"O:{c.get('open')} H:{c.get('high')} L:{c.get('low')} C:{c.get('close')} V:{c.get('tick_volume', 0)}")
        data_str = "\n".join(recent_data)
        
        prompt = f"""
Analyze the following candlestick data for {symbol}:

[OHLCV DATA — last 50 candles]
{data_str}

[PRE-COMPUTED INDICATORS]
- Current Price: {current_price}
- SMA(20): {sma_20:.5f}
- SMA(50): {sma_50:.5f}
- RSI(14): {rsi_14:.1f}
- Bollinger Band Width (20): {bbw_20:.3f}%
- Range High (50 candles): {high_range}
- Range Low (50 candles): {low_range}

Based on this data, provide a technical analysis. Evaluate the BBW to determine if the market is trending, ranging, or too choppy (noise).
Return ONLY a JSON object with the following schema:
{{
    "trend": "BULLISH" | "BEARISH" | "NEUTRAL",
    "regime": "TRENDING" | "RANGING" | "CHOPPY",
    "support_level": float,
    "resistance_level": float,
    "confidence": 0-100
}}
"""
        
        result = await mistral_client.get_json_analysis(
            prompt=prompt,
            system_prompt="You are a highly skilled quantitative technical analyst. Output ONLY valid JSON."
        )
        
        if result:
            regime = result.get('regime', 'CHOPPY')
            result["atr_14"] = round(atr_14, 5)
            result["sr_bounce_detected"] = _detect_sr_bounce(candles, atr_14)
            result["volume_breakout_detected"] = _detect_volume_breakout(candles, regime)
            logger.info(f"Market analysis for {symbol}: {result.get('trend')} | Regime: {regime} (Conf: {result.get('confidence')}%) | ATR: {atr_14:.5f} | SR Bounce: {result['sr_bounce_detected']} | Vol Break: {result['volume_breakout_detected']}")
            return result
        else:
            logger.error(f"Market analysis failed for {symbol}.")
            return {"trend": "NEUTRAL", "regime": "CHOPPY", "support_level": 0.0, "resistance_level": 0.0, "confidence": 0, "atr_14": round(atr_14, 5)}

    async def confirm_reversal(self, symbol: str, candles: List[Dict[str, Any]], current_direction: str) -> bool:
        """
        Escalation path: Asks the AI if the technical trend has reversed strongly against our position.
        current_direction: 'BUY' or 'SELL'.
        Returns True if AI confirms a strong reversal (>60% confidence).
        """
        if not candles or len(candles) < 2:
            return False
            
        logger.warning(f"MarketAnalyzer ESCALATING to AI: Checking for reversal on {symbol} (Position: {current_direction})")
        
        sma_20 = _compute_sma(candles, 20)
        sma_50 = _compute_sma(candles, 50)
        rsi_14 = _compute_rsi(candles, 14)
        current_price = float(candles[-1].get('close', 0))
        
        recent_data = []
        for c in candles[-15:]:  # Just last 15 candles for quick reversal check
            recent_data.append(f"O:{c.get('open')} H:{c.get('high')} L:{c.get('low')} C:{c.get('close')}")
        data_str = "\n".join(recent_data)
        
        prompt = f"""
We currently hold a {current_direction} position on {symbol}.
Local indicators flagged a potential reversal. Please review the latest 15 candles and indicators to confirm.

[OHLC DATA — last 15 candles]
{data_str}

[PRE-COMPUTED INDICATORS]
- Current Price: {current_price}
- SMA(20): {sma_20:.5f}
- SMA(50): {sma_50:.5f}
- RSI(14): {rsi_14:.1f}

Based strictly on this data, has the technical trend COMPLETELY REVERSED against our {current_direction} position?
Return ONLY a JSON object with this schema:
{{
    "reversed": true | false,
    "confidence": 0-100,
    "reasoning": "One short sentence why"
}}
"""
        result = await mistral_client.get_json_analysis(
            prompt=prompt,
            system_prompt="You are a strict risk-manager AI. Output ONLY valid JSON."
        )
        
        if result and result.get('reversed') is True:
            conf = result.get('confidence', 0)
            if conf >= 60:
                logger.critical(f"AI CONFIRMED REVERSAL for {symbol} with {conf}% confidence! Reason: {result.get('reasoning')}")
                return True
            else:
                logger.info(f"AI sees reversal for {symbol} but confidence too low ({conf}%). Ignoring.")
                return False
                
        logger.info(f"AI denied reversal for {symbol}. Trend remains valid.")
        return False

market_analyzer = MarketAnalyzer()
