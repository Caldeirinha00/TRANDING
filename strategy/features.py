import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "ker_d1",
    "atr_regime_ratio",
    "cs_momentum_spread",
    "breakout_extension_atr",
    "bar_conviction_ratio",
    "d1_trend_age_norm",
    "net_carry_to_risk_ratio",
    "liquidity_session_sin",
    "liquidity_session_cos",
]


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def compute_kaufman_efficiency_ratio(close: pd.Series, window: int = 10) -> pd.Series:
    """
    1. ker_d1: |P_t - P_{t-10}| / sum_{i=0..9} |P_{t-i} - P_{t-i-1}| in [0, 1]
    """
    direction = (close - close.shift(window)).abs()
    volatility = close.diff(1).abs().rolling(window=window, min_periods=window).sum()
    ker = direction / volatility.replace(0.0, np.nan)
    return ker.clip(0.0, 1.0).fillna(0.0)


def compute_d1_trend_age_norm(close_d1: pd.Series, ema50_d1: pd.Series, cap: int = 30) -> pd.Series:
    """
    6. d1_trend_age_norm: Consecutive D1 bars on the same side of EMA50_D1,
    capped at 30 and normalized into [0, 1].
    """
    side = np.where(close_d1 >= ema50_d1, 1, -1)
    side_series = pd.Series(side, index=close_d1.index)
    regime_group = (side_series != side_series.shift(1)).cumsum()
    streak = side_series.groupby(regime_group).cumcount() + 1
    return (streak.clip(upper=cap) / float(cap)).astype(float)


def align_d1_to_h4_strict_causal(
    h4_open_index: pd.DatetimeIndex,
    d1_df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Align D1 features to H4 bars strictly by bar close time (zero intraday lookahead).
    MT5 timestamps denote bar OPEN times:
      - H4 bar opened at T_h4 closes at T_h4 + 4h.
      - D1 bar opened at T_d1 closes at T_d1 + 24h.
    An H4 bar closing at T_h4 + 4h may only read D1 bars with T_d1 + 24h <= T_h4 + 4h.
    """
    h4_close = pd.DataFrame({"h4_open_time": h4_open_index, "obs_time": h4_open_index + pd.Timedelta(hours=4)})
    d1_obs = d1_df[columns].copy()
    d1_obs["obs_time"] = d1_df.index + pd.Timedelta(hours=24)
    d1_obs = d1_obs.sort_values("obs_time")
    h4_sorted = h4_close.sort_values("obs_time")

    aligned = pd.merge_asof(
        h4_sorted,
        d1_obs,
        on="obs_time",
        direction="backward",
    )
    aligned = aligned.set_index("h4_open_time").reindex(h4_open_index)
    return aligned[columns]


def build_stationary_feature_frame(
    df_h4: pd.DataFrame,
    df_d1: pd.DataFrame,
    cs_momentum_spread_series: pd.Series,
    net_carry_to_risk_series: pd.Series,
    side_series: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Pure single-source-of-truth function for Layer 2 stationary features.
    Strictly aligns closed D1 bars (+24h close time) to closed H4 bars (+4h close time).
    When `side_series` (+1 / -1) is provided, directional features (`cs_momentum_spread`
    and `breakout_extension_atr`) are signed relative to the trade direction so shallow
    trees treat Long and Short candidates symmetrically.
    """
    # --- Strictly causal D1 calculations ---
    d1 = df_d1.copy()
    d1["ema50_d1"] = d1["close"].ewm(span=50, min_periods=50, adjust=False).mean()
    d1["ker_d1"] = compute_kaufman_efficiency_ratio(d1["close"], window=10)
    d1["d1_trend_age_norm"] = compute_d1_trend_age_norm(d1["close"], d1["ema50_d1"], cap=30)

    # --- Strictly causal H4 calculations ---
    h4 = df_h4.copy()
    h4["atr14_h4"] = compute_atr(h4, period=14)
    h4["ema20_h4"] = h4["close"].ewm(span=20, min_periods=20, adjust=False).mean()

    # 2. atr_regime_ratio: ATR14_H4 / Median_120(ATR14_H4)
    atr_median_120 = h4["atr14_h4"].rolling(window=120, min_periods=30).median()
    h4["atr_regime_ratio"] = (h4["atr14_h4"] / atr_median_120.replace(0.0, np.nan)).fillna(1.0)

    # 4. breakout_extension_atr: unsigned |Close - EMA20| / ATR, or signed dip depth -side * (Close - EMA20) / ATR
    raw_ext = (h4["close"] - h4["ema20_h4"]) / h4["atr14_h4"].replace(0.0, np.nan)
    if side_series is not None:
        aligned_side = side_series.reindex(h4.index).fillna(1.0).astype(float)
        h4["breakout_extension_atr"] = (-aligned_side * raw_ext).fillna(0.0)
    else:
        aligned_side = None
        h4["breakout_extension_atr"] = raw_ext.abs().fillna(0.0)

    # 5. bar_conviction_ratio: (|Close - Open| / (High - Low)) * (tick_volume / SMA20(tick_volume))
    bar_range = (h4["high"] - h4["low"]).replace(0.0, np.nan)
    body_ratio = ((h4["close"] - h4["open"]).abs() / bar_range).fillna(0.0)
    vol_col = "tick_volume" if "tick_volume" in h4.columns else "volume"
    vol_mean_20 = h4[vol_col].rolling(window=20, min_periods=5).mean().replace(0.0, np.nan)
    vol_ratio = (h4[vol_col] / vol_mean_20).fillna(1.0).clip(0.0, 5.0)
    h4["bar_conviction_ratio"] = body_ratio * vol_ratio

    # 8. liquidity_session_flag: Cyclic encoding of H4 close hour (bar open + 4h)
    close_hours = ((h4.index.hour + 4) % 24).astype(float)
    h4["liquidity_session_sin"] = np.sin(2.0 * np.pi * close_hours / 24.0)
    h4["liquidity_session_cos"] = np.cos(2.0 * np.pi * close_hours / 24.0)

    # Strict causal D1 -> H4 alignment by close timestamp
    aligned_d1 = align_d1_to_h4_strict_causal(h4.index, d1, ["ker_d1", "d1_trend_age_norm"])
    merged = h4.join(aligned_d1)
    merged["ker_d1"] = merged["ker_d1"].fillna(0.0)
    merged["d1_trend_age_norm"] = merged["d1_trend_age_norm"].fillna(0.0)

    # 3. cs_momentum_spread & 7. net_carry_to_risk_ratio
    cs_aligned = cs_momentum_spread_series.reindex(merged.index).ffill().fillna(0.0)
    if aligned_side is not None:
        merged["cs_momentum_spread"] = aligned_side * cs_aligned
    else:
        merged["cs_momentum_spread"] = cs_aligned
    merged["net_carry_to_risk_ratio"] = net_carry_to_risk_series.reindex(merged.index).ffill().fillna(0.0)

    return merged[FEATURE_COLUMNS]
