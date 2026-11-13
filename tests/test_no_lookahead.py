import numpy as np
import pandas as pd
from strategy.features import build_stationary_feature_frame
from strategy.primary_trend import PrimaryStrategyConfig, PrimaryTrendStrategy


def _make_ohlcv(idx: pd.DatetimeIndex, seed: int = 42, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, 0.0015, size=len(idx))
    close = 1.0800 + np.cumsum(steps)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.0008, size=len(idx)))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.0008, size=len(idx)))
    vol = rng.integers(500, 5000, size=len(idx)).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "tick_volume": vol},
        index=idx,
    )


def test_strict_temporal_causality_no_lookahead():
    """
    Truncating future H4 and D1 bars must never alter past stationary features or past primary signals.
    """
    idx_h4 = pd.date_range("2024-01-01", periods=300, freq="4h", tz="UTC")
    idx_d1 = pd.date_range("2023-10-01", periods=150, freq="1D", tz="UTC")

    df_h4_full = _make_ohlcv(idx_h4, seed=42)
    df_d1_full = _make_ohlcv(idx_d1, seed=43)
    df_h4_2 = _make_ohlcv(idx_h4, seed=99)
    df_d1_2 = _make_ohlcv(idx_d1, seed=100)

    cs_mom = pd.Series(0.35, index=idx_h4)
    carry = pd.Series(-0.02, index=idx_h4)

    features_full = build_stationary_feature_frame(df_h4_full, df_d1_full, cs_mom, carry)

    cutoff_time = idx_h4[240]
    df_h4_trunc = df_h4_full.loc[:cutoff_time]
    df_d1_trunc = df_d1_full.loc[:cutoff_time]

    features_trunc = build_stationary_feature_frame(
        df_h4_trunc, df_d1_trunc, cs_mom.loc[:cutoff_time], carry.loc[:cutoff_time]
    )

    max_diff = (features_full.loc[:cutoff_time] - features_trunc).abs().max().max()
    assert max_diff <= 1e-12, f"Temporal causality violation detected! Max diff: {max_diff}"

    # Verify primary signal causality on a 2-pair basket
    strategy = PrimaryTrendStrategy()
    sigs_full, _ = strategy.generate_signals_for_basket(
        {"EURUSD": df_h4_full, "USDJPY": df_h4_2},
        {"EURUSD": df_d1_full, "USDJPY": df_d1_2},
    )
    sigs_trunc, _ = strategy.generate_signals_for_basket(
        {"EURUSD": df_h4_trunc, "USDJPY": df_h4_2.loc[:cutoff_time]},
        {"EURUSD": df_d1_trunc, "USDJPY": df_d1_2.loc[:cutoff_time]},
    )
    assert (sigs_full["EURUSD"].loc[:cutoff_time, "side"] == sigs_trunc["EURUSD"].loc[:cutoff_time, "side"]).all()


def test_intraday_d1_bar_does_not_leak_before_d1_close():
    """
    Mutating an unclosed D1 bar's close at 00:00 of day D must NOT affect any H4 bar
    closing earlier than (D + 24h) — i.e., H4 bars opened at 00:00..16:00 on day D.
    """
    idx_h4 = pd.date_range("2024-01-01", periods=360, freq="4h", tz="UTC")
    idx_d1 = pd.date_range("2023-10-01", periods=160, freq="1D", tz="UTC")

    df_h4 = _make_ohlcv(idx_h4, seed=7)
    df_d1_a = _make_ohlcv(idx_d1, seed=8)
    df_d1_b = df_d1_a.copy()

    # Mutate today's D1 close on 2024-02-15 (which opens 2024-02-15 00:00 and closes 2024-02-16 00:00)
    target_day = pd.Timestamp("2024-02-15 00:00:00+00:00")
    df_d1_b.loc[target_day, "close"] = df_d1_b.loc[target_day, "close"] + 0.15

    cs_mom = pd.Series(0.1, index=idx_h4)
    carry = pd.Series(0.0, index=idx_h4)
    feat_a = build_stationary_feature_frame(df_h4, df_d1_a, cs_mom, carry)
    feat_b = build_stationary_feature_frame(df_h4, df_d1_b, cs_mom, carry)

    # All H4 bars before 2024-02-15 20:00 close before 2024-02-16 00:00, so must be bit-identical
    pre_close_mask = feat_a.index < pd.Timestamp("2024-02-15 20:00:00+00:00")
    assert (feat_a.loc[pre_close_mask] - feat_b.loc[pre_close_mask]).abs().max().max() <= 1e-12

    # Same intraday causality check across DUAL_REGIME primary signals (D1_TREND_CARRY + BB_TREND_DIP + BB_RANGE_REV)
    strat = PrimaryTrendStrategy(config=PrimaryStrategyConfig(mode="DUAL_REGIME"))
    df_h4_2 = _make_ohlcv(idx_h4, seed=11)
    df_d1_2 = _make_ohlcv(idx_d1, seed=12)
    sigs_a, _ = strat.generate_signals_for_basket({"EURUSD": df_h4, "USDJPY": df_h4_2}, {"EURUSD": df_d1_a, "USDJPY": df_d1_2})
    sigs_b, _ = strat.generate_signals_for_basket({"EURUSD": df_h4, "USDJPY": df_h4_2}, {"EURUSD": df_d1_b, "USDJPY": df_d1_2})
    assert (sigs_a["EURUSD"].loc[pre_close_mask, "side"] == sigs_b["EURUSD"].loc[pre_close_mask, "side"]).all()


def test_signed_directional_features_causality_and_symmetry():
    """
    Passing `side_series` to `build_stationary_feature_frame` must preserve strict temporal
    causality while orienting directional features symmetrically for Long (+1) and Short (-1).
    """
    idx_h4 = pd.date_range("2024-01-01", periods=160, freq="4h", tz="UTC")
    idx_d1 = pd.date_range("2023-10-01", periods=100, freq="1D", tz="UTC")
    df_h4 = _make_ohlcv(idx_h4, seed=21)
    df_d1 = _make_ohlcv(idx_d1, seed=22)

    cs_mom = pd.Series(-0.80, index=idx_h4)
    carry = pd.Series(0.01, index=idx_h4)
    short_sides = pd.Series(-1, index=idx_h4)
    long_sides = pd.Series(1, index=idx_h4)

    feat_short_full = build_stationary_feature_frame(df_h4, df_d1, cs_mom, carry, side_series=short_sides)
    cutoff = idx_h4[120]
    feat_short_trunc = build_stationary_feature_frame(
        df_h4.loc[:cutoff], df_d1.loc[:cutoff], cs_mom.loc[:cutoff], carry.loc[:cutoff], side_series=short_sides.loc[:cutoff]
    )
    assert (feat_short_full.loc[:cutoff] - feat_short_trunc).abs().max().max() <= 1e-12

    feat_long_pos = build_stationary_feature_frame(df_h4, df_d1, -cs_mom, carry, side_series=long_sides)
    assert np.allclose(feat_short_full["cs_momentum_spread"].iloc[-1], 0.80)
    assert np.allclose(feat_long_pos["cs_momentum_spread"].iloc[-1], 0.80)
    assert np.allclose(
        feat_short_full["breakout_extension_atr"].iloc[-1],
        -feat_long_pos["breakout_extension_atr"].iloc[-1],
    )
