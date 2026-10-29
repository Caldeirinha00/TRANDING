import numpy as np
import pandas as pd
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

LOOKBACK_WINDOW = 101  # Constante partilhada entre live e treino

@dataclass
class RegimeState:
    regime: str       # TRENDING | RANGING | CHOPPY
    trend_dir: float  # [-1, +1]
    strength: float   # [0, 1]

def compute_sma(candles: List[Dict[str, Any]], period: int) -> float:
    if len(candles) < period: return 0.0
    closes = [float(c.get('close', 0)) for c in candles[-period:]]
    return sum(closes) / period

def compute_ema(candles: List[Dict[str, Any]], period: int) -> float:
    if len(candles) < period: return 0.0
    closes = [float(c.get('close', 0)) for c in candles]
    ema = sum(closes[:period]) / period
    multiplier = 2 / (period + 1)
    for price in closes[period:]:
        ema = (price - ema) * multiplier + ema
    return ema

def ema_slope_bps(candles: List[Dict[str, Any]], period: int) -> float:
    if len(candles) < period + 1: return 0.0
    closes = [float(c.get('close', 0)) for c in candles]
    ema = sum(closes[:period]) / period
    multiplier = 2 / (period + 1)
    prev_ema = ema
    for i in range(period, len(closes) - 1):
        prev_ema = (closes[i] - prev_ema) * multiplier + prev_ema
    curr_ema = (closes[-1] - prev_ema) * multiplier + prev_ema
    if prev_ema == 0: return 0.0
    return (curr_ema / prev_ema - 1.0) * 1e4

def compute_bbw(candles: List[Dict[str, Any]], period: int = 20, num_std_dev: float = 2.0) -> float:
    if len(candles) < period: return 0.0
    closes = [float(c.get('close', 0)) for c in candles[-period:]]
    sma = sum(closes) / period
    variance = sum((c - sma) ** 2 for c in closes) / period
    std_dev = math.sqrt(variance)
    if sma == 0: return 0.0
    upper_band = sma + (num_std_dev * std_dev)
    lower_band = sma - (num_std_dev * std_dev)
    return (upper_band - lower_band) / sma * 100

def bollinger_position(candles: List[Dict[str, Any]], period: int = 20, num_std_dev: float = 2.0) -> float:
    if len(candles) < period: return 0.5
    closes = [float(c.get('close', 0)) for c in candles[-period:]]
    sma = sum(closes) / period
    variance = sum((c - sma) ** 2 for c in closes) / period
    std_dev = math.sqrt(variance)
    upper = sma + (num_std_dev * std_dev)
    lower = sma - (num_std_dev * std_dev)
    curr = closes[-1]
    if upper == lower: return 0.5
    pos = (curr - lower) / (upper - lower)
    return max(0.0, min(1.0, pos))

def compute_rsi(candles: List[Dict[str, Any]], period: int = 14) -> float:
    if len(candles) < period + 1: return 50.0
    closes = [float(c.get('close', 0)) for c in candles]
    
    gains, losses = [], []
    for i in range(1, period + 1):
        delta = closes[i] - closes[i-1]
        gains.append(max(0.0, delta))
        losses.append(max(0.0, -delta))
        
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    
    for i in range(period + 1, len(closes)):
        delta = closes[i] - closes[i-1]
        gain = max(0.0, delta)
        loss = max(0.0, -delta)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_atr(candles: List[Dict[str, Any]], period: int = 14) -> float:
    if len(candles) < period + 1: return 0.0
    true_ranges = []
    for i in range(1, len(candles)):
        high = float(candles[i].get('high', 0))
        low = float(candles[i].get('low', 0))
        prev_close = float(candles[i-1].get('close', 0))
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        true_ranges.append(max(tr1, tr2, tr3))
    recent_trs = true_ranges[-period:]
    return sum(recent_trs) / len(recent_trs)

def compute_momentum(candles: List[Dict[str, Any]], period: int) -> float:
    if len(candles) < period + 1: return 0.0
    curr = float(candles[-1].get('close', 0))
    prev = float(candles[-(period+1)].get('close', 0))
    if prev == 0: return 0.0
    return (curr / prev - 1.0) * 1e4

def compute_macd_normalized(candles: List[Dict[str, Any]], atr: float) -> Tuple[float, float]:
    if len(candles) < 34 or atr == 0: return 0.0, 0.0
    c_list = [float(c.get('close', 0)) for c in candles[-60:]] if len(candles) >= 60 else [float(c.get('close', 0)) for c in candles]
    
    mult12 = 2 / 13
    mult26 = 2 / 27
    mult9 = 2 / 10
    
    if len(c_list) < 26:
        return 0.0, 0.0
        
    ema12_h = [c_list[0]]
    for p in c_list[1:]:
        ema12_h.append((p - ema12_h[-1]) * mult12 + ema12_h[-1])
        
    ema26_h = [c_list[0]]
    for p in c_list[1:]:
        ema26_h.append((p - ema26_h[-1]) * mult26 + ema26_h[-1])
        
    macd_history = [e12 - e26 for e12, e26 in zip(ema12_h[-len(c_list)+25:], ema26_h[-len(c_list)+25:])]
    if len(macd_history) < 9:
        return macd_history[-1] / atr, 0.0
        
    signal_h = macd_history[0]
    for m in macd_history[1:]:
        signal_h = (m - signal_h) * mult9 + signal_h
        
    return macd_history[-1] / atr, signal_h / atr

def volume_ratio(candles: List[Dict[str, Any]], period: int) -> float:
    if len(candles) < period + 1: return 1.0
    curr_vol = float(candles[-1].get('tick_volume', 0))
    vols = [float(c.get('tick_volume', 0)) for c in candles[-(period+1):-1]]
    avg_vol = sum(vols) / len(vols) if vols else 1.0
    if avg_vol == 0: return 1.0
    return curr_vol / avg_vol

def candle_body_ratio(candle: Dict[str, Any]) -> float:
    o = float(candle.get('open', 0))
    h = float(candle.get('high', 0))
    l = float(candle.get('low', 0))
    c = float(candle.get('close', 0))
    rng = h - l
    if rng == 0: return 0.0
    return abs(c - o) / rng

def candle_size_vs_atr(candles: List[Dict[str, Any]], period: int) -> float:
    atr = compute_atr(candles, period)
    if atr == 0: return 0.0
    h = float(candles[-1].get('high', 0))
    l = float(candles[-1].get('low', 0))
    return (h - l) / atr

def compute_stochastic(candles: List[Dict[str, Any]], k_period: int = 14, d_period: int = 3) -> float:
    if len(candles) < k_period: return 50.0
    recent = candles[-k_period:]
    highs = [float(c.get('high', 0)) for c in recent]
    lows = [float(c.get('low', 0)) for c in recent]
    c = float(candles[-1].get('close', 0))
    hh = max(highs)
    ll = min(lows)
    if hh == ll: return 50.0
    return (c - ll) / (hh - ll) * 100.0

def distance_to_support(candles: List[Dict[str, Any]]) -> float:
    if len(candles) < 20: return 0.0
    lows = [float(c.get('low', 0)) for c in candles[-20:]]
    c = float(candles[-1].get('close', 0))
    return c - min(lows)

def distance_to_resistance(candles: List[Dict[str, Any]]) -> float:
    if len(candles) < 20: return 0.0
    highs = [float(c.get('high', 0)) for c in candles[-20:]]
    c = float(candles[-1].get('close', 0))
    return max(highs) - c

def session_position(candles: List[Dict[str, Any]]) -> float:
    if len(candles) == 0: return 0.5
    ts = candles[-1].get('time')
    if isinstance(ts, int) or isinstance(ts, float):
        dt = pd.to_datetime(ts, unit='s')
    else:
        dt = pd.to_datetime(ts)
    return (dt.hour * 60 + dt.minute) / (24 * 60)

def session_range_vs_atr(candles: List[Dict[str, Any]], period: int) -> float:
    if len(candles) < 60: return 0.0
    atr = compute_atr(candles, period)
    if atr == 0: return 0.0
    highs = [float(c.get('high', 0)) for c in candles[-60:]]
    lows = [float(c.get('low', 0)) for c in candles[-60:]]
    return (max(highs) - min(lows)) / atr

def time_sin(candle: Dict[str, Any]) -> float:
    ts = candle.get('time')
    if isinstance(ts, int) or isinstance(ts, float):
        dt = pd.to_datetime(ts, unit='s')
    else:
        dt = pd.to_datetime(ts)
    minutes = dt.hour * 60 + dt.minute
    return math.sin(2 * math.pi * minutes / (24 * 60))

def day_sin(candle: Dict[str, Any]) -> float:
    ts = candle.get('time')
    if isinstance(ts, int) or isinstance(ts, float):
        dt = pd.to_datetime(ts, unit='s')
    else:
        dt = pd.to_datetime(ts)
    return math.sin(2 * math.pi * dt.dayofweek / 7)

def compute_fracdiff(closes: List[float], d: float = 0.40, threshold: float = 1e-4) -> float:
    """Computes fixed-width fractionally differentiated close value to preserve stationary memory."""
    if len(closes) < 10:
        return 0.0
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < threshold or k >= len(closes):
            break
        w.append(w_k)
        k += 1
    w = np.array(w[::-1])
    recent_closes = np.array(closes[-len(w):])
    return float(np.dot(w, recent_closes))

def compute_garman_klass_ratio(candles: List[Dict[str, Any]], atr: float) -> float:
    """Garman-Klass intraday volatility normalized by ATR."""
    if not candles or atr <= 0:
        return 1.0
    c = candles[-1]
    h, l = float(c.get('high', 0)), float(c.get('low', 0))
    open_p, close_p = float(c.get('open', 0)), float(c.get('close', 0))
    if l <= 0 or open_p <= 0:
        return 1.0
    log_hl = math.log(max(h, 1e-8) / max(l, 1e-8))
    log_co = math.log(max(close_p, 1e-8) / max(open_p, 1e-8))
    sigma2 = 0.5 * (log_hl ** 2) - (2.0 * math.log(2.0) - 1.0) * (log_co ** 2)
    gk_vol = math.sqrt(max(0.0, sigma2)) * close_p
    return gk_vol / atr

def compute_bvc_ofi(candles: List[Dict[str, Any]], period: int = 14) -> float:
    """Bulk Volume Classification (BVC) Order Flow Imbalance estimate."""
    if len(candles) < period:
        return 0.0
    returns = []
    for i in range(1, len(candles)):
        c_curr = float(candles[i].get('close', 0))
        c_prev = float(candles[i-1].get('close', 0))
        returns.append(c_curr - c_prev)
    sigma_dp = float(np.std(returns[-period:])) if len(returns) >= period else 1e-5
    if sigma_dp <= 0:
        sigma_dp = 1e-5
    last = candles[-1]
    delta = float(last.get('close', 0)) - float(last.get('open', 0))
    z = delta / sigma_dp
    phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return 2.0 * phi - 1.0

def compute_spread_zscore(spread_history: List[float], current_spread: float) -> float:
    """Computes median-absolute-deviation (MAD) standardized spread expansion z-score."""
    if len(spread_history) < 10:
        return 0.0
    median_s = float(np.median(spread_history))
    mad = float(np.median(np.abs(np.array(spread_history) - median_s)))
    if mad <= 0:
        return 0.0
    return (current_spread - median_s) / (1.4826 * mad + 1e-8)

def compute_adx_full(candles: List[Dict[str, Any]], period: int = 14) -> Tuple[float, float, float]:
    if len(candles) < period * 2: return 0.0, 0.0, 0.0
    
    tr_list, plus_dm_list, minus_dm_list = [], [], []
    for i in range(1, len(candles)):
        h = float(candles[i].get('high', 0))
        l = float(candles[i].get('low', 0))
        ph = float(candles[i-1].get('high', 0))
        pl = float(candles[i-1].get('low', 0))
        pc = float(candles[i-1].get('close', 0))
        
        tr = max(h - l, abs(h - pc), abs(l - pc))
        tr_list.append(tr)
        
        up_move = h - ph
        down_move = pl - l
        
        if up_move > down_move and up_move > 0:
            plus_dm_list.append(up_move)
        else:
            plus_dm_list.append(0.0)
            
        if down_move > up_move and down_move > 0:
            minus_dm_list.append(down_move)
        else:
            minus_dm_list.append(0.0)
            
    atr = sum(tr_list[:period])
    plus_di14 = sum(plus_dm_list[:period])
    minus_di14 = sum(minus_dm_list[:period])
    
    dx_list = []
    
    for i in range(period, len(tr_list)):
        atr = atr - (atr / period) + tr_list[i]
        plus_di14 = plus_di14 - (plus_di14 / period) + plus_dm_list[i]
        minus_di14 = minus_di14 - (minus_di14 / period) + minus_dm_list[i]
        
        if atr == 0:
            dx_list.append(0.0)
            continue
            
        pdi = 100 * (plus_di14 / atr)
        mdi = 100 * (minus_di14 / atr)
        
        if (pdi + mdi) == 0:
            dx_list.append(0.0)
        else:
            dx_list.append(100 * abs(pdi - mdi) / (pdi + mdi))
            
    if len(dx_list) < period:
        return 0.0, 0.0, 0.0
        
    adx = sum(dx_list[:period]) / period
    for i in range(period, len(dx_list)):
        adx = (adx * (period - 1) + dx_list[i]) / period
        
    pdi = 100 * (plus_di14 / atr) if atr > 0 else 0
    mdi = 100 * (minus_di14 / atr) if atr > 0 else 0
    return adx, pdi, mdi

def compute_bbw_history(candles: List[Dict[str, Any]], period: int = 20, lookback: int = 100) -> List[float]:
    if len(candles) < lookback + period: return []
    history = []
    for i in range(len(candles) - lookback, len(candles) + 1):
        window = candles[i - period:i]
        history.append(compute_bbw(window, period))
    return history

def percentile_rank(val: float, history: List[float]) -> float:
    if not history: return 50.0
    count = sum(1 for x in history if x < val)
    return (count / len(history)) * 100.0

def classify_regime(candles: List[Dict], adx_period: int = 14, bbw_lookback: int = 100) -> RegimeState:
    adx, plus_di, minus_di = compute_adx_full(candles, adx_period)
    bbw = compute_bbw(candles, 20)
    
    if len(candles) >= bbw_lookback + 20:
        bbw_history = compute_bbw_history(candles, 20, bbw_lookback)
        bbw_percentile = percentile_rank(bbw, bbw_history)
    else:
        bbw_percentile = 50.0
    
    if bbw_percentile < 20:
        regime = "CHOPPY"
        strength = 1.0 - (bbw_percentile / 20)
    elif adx > 25:
        regime = "TRENDING"
        strength = min(adx / 50.0, 1.0)
    else:
        regime = "RANGING"
        strength = 1.0 - (adx / 25.0)
    
    if plus_di > minus_di:
        trend_dir = strength if regime == "TRENDING" else strength * 0.3
    else:
        trend_dir = -strength if regime == "TRENDING" else -strength * 0.3
    
    return RegimeState(regime=regime, trend_dir=trend_dir, strength=strength)

def compute_scalp_features(candles: List[Dict], spread: float) -> Optional[np.ndarray]:
    """
    30 features normalizados. UMA SÓ IMPLEMENTAÇÃO — live e treino.
    """
    if len(candles) < LOOKBACK_WINDOW:
        return None
    
    candles = candles[-LOOKBACK_WINDOW:]
    
    price = float(candles[-1]["close"])
    atr = compute_atr(candles, 14)
    if atr <= 0:
        return None
    
    regime = classify_regime(candles)
    macd, macd_signal = compute_macd_normalized(candles, atr)
    
    closes = [float(c.get("close", 0)) for c in candles]
    spread_history = [float(c.get("spread_price", spread)) for c in candles[-50:]]
    fracdiff_val = (compute_fracdiff(closes, d=0.40) / price - 1.0) * 1e4 if price > 0 else 0.0
    gk_ratio = compute_garman_klass_ratio(candles, atr)
    bvc_ofi = compute_bvc_ofi(candles, 14)
    spread_z = compute_spread_zscore(spread_history, spread)

    features = np.array([
        compute_rsi(candles, 7) / 100.0,                          # 0
        compute_rsi(candles, 14) / 100.0,                         # 1
        compute_stochastic(candles, 14, 3) / 100.0,               # 2
        (compute_ema(candles, 8) / price - 1.0) * 1e4,            # 3
        (compute_ema(candles, 21) / price - 1.0) * 1e4,           # 4
        (compute_sma(candles, 50) / price - 1.0) * 1e4,           # 5
        ema_slope_bps(candles, 8),                                 # 6
        ema_slope_bps(candles, 21),                                # 7
        compute_bbw(candles, 20),                                  # 8
        bollinger_position(candles, 20),                           # 9
        atr / price * 1e4,                                        # 10
        compute_momentum(candles, 5),                              # 11
        compute_momentum(candles, 10),                             # 12
        macd,                                                      # 13
        macd_signal,                                               # 14
        volume_ratio(candles, 1),                                  # 15
        volume_ratio(candles, 5),                                  # 16
        volume_ratio(candles, 20),                                 # 17
        candle_body_ratio(candles[-1]),                             # 18
        candle_size_vs_atr(candles, 14),                           # 19
        distance_to_support(candles) / atr,                        # 20
        distance_to_resistance(candles) / atr,                     # 21
        spread / atr,                                              # 22
        session_position(candles),                                 # 23
        session_range_vs_atr(candles, 14),                         # 24
        time_sin(candles[-1]),                                     # 25
        day_sin(candles[-1]),                                      # 26
        regime.trend_dir,                                          # 27
        regime.strength,                                           # 28
        {"TRENDING": 1.0, "RANGING": 0.0, "CHOPPY": -1.0}[regime.regime],  # 29
        fracdiff_val,                                              # 30
        gk_ratio,                                                  # 31
        bvc_ofi,                                                   # 32
        spread_z,                                                  # 33
    ], dtype=np.float32)
    
    return features

FEATURE_NAMES = [
    "rsi_7", "rsi_14", "stoch_k",
    "ema8_bps", "ema21_bps", "sma50_bps", "ema8_slope", "ema21_slope",
    "bbw", "bb_pos", "atr_bps",
    "mom_5", "mom_10", "macd", "macd_signal",
    "vol_1", "vol_5", "vol_20",
    "body_ratio", "candle_vs_atr",
    "dist_support", "dist_resist", "spread_atr",
    "session_pos", "session_range", "time_sin", "day_sin",
    "trend_dir", "regime_strength", "regime_type",
    "fracdiff_bps", "gk_ratio", "bvc_ofi", "spread_zscore"
]

def compute_scalp_features_from_df(df: pd.DataFrame, idx: int) -> Optional[np.ndarray]:
    """Adaptador: converte slice do DataFrame para List[Dict] e chama a mesma função."""
    start = max(0, idx - LOOKBACK_WINDOW + 1)
    if idx - start + 1 < LOOKBACK_WINDOW:
        return None
    window = df.iloc[start : idx + 1].to_dict("records")
    spread = df.iloc[idx].get("spread_price", 0.0)
    return compute_scalp_features(window, spread)
