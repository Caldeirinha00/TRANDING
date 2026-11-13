import numpy as np
import pandas as pd
from backtest.engine import PortfolioBacktestEngine
from strategy.primary_trend import PrimaryStrategyConfig
from tests.test_esma_risk_math import make_mock_spec


def _make_h4_bars(timestamps: pd.DatetimeIndex, opens, highs, lows, closes) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "tick_volume": [1000] * len(timestamps),
            "spread": [0] * len(timestamps),
            "real_volume": [0] * len(timestamps),
        },
        index=timestamps,
    )


def test_backtest_bid_ask_spread_tp_and_wednesday_swap():
    """
    Verify exact MT5 Bid/Ask execution and Wednesday triple rollover accounting in PortfolioBacktestEngine:
    - Signal at t=0 (Tuesday 2025-03-04 20:00) enters Long at t=1 Open (Wednesday 00:00 Ask = Bid + spread).
    - Holds across Wednesday->Thursday 00:00 (3x rollover) and hits TP on Thursday 04:00 bar.
    """
    idx = pd.date_range("2025-03-04 20:00:00", periods=10, freq="4h", tz="UTC")
    opens = [1.1000] * 10
    highs = [1.1010] * 10
    lows = [1.0990] * 10
    closes = [1.1005] * 10
    # Hit TP (+2.25R = +0.01125 -> 1.11135) on bar idx[8] (Thu 04:00)
    highs[8] = 1.1120
    df_h4 = _make_h4_bars(idx, opens, highs, lows, closes)

    sig_df = pd.DataFrame(
        {
            "side": [1] + [0] * 9,
            "atr14_h4": [0.0025] * 10,
            "sl_distance_price": [0.0050] * 10,
            "tp_distance_price": [0.01125] * 10,
            "cs_spread": [1.0] * 10,
            "ker_d1": [0.50] * 10,
            "trigger_type": ["DONCHIAN_DUAL_MOM"] + ["NONE"] * 9,
        },
        index=idx,
    )

    spec = make_mock_spec(
        "EURUSD",
        pip_value_001_eur=0.10,
        digits=5,
        spread_points=10,
        swap_long=-2.0,  # -0.02 EUR/day -> Wed 3x = -0.06 EUR
        swap_short=0.5,
    )
    engine = PortfolioBacktestEngine(specs={"EURUSD": spec}, initial_capital_eur=1000.0)
    report = engine.run(h4_bars_by_symbol={"EURUSD": df_h4}, signals_by_symbol={"EURUSD": sig_df}, fixed_lots=0.01)

    assert report.total_trades == 1
    trade = report.trades[0]
    assert trade.exit_reason == "TP"
    assert round(trade.entry_price, 5) == 1.10010
    assert round(trade.exit_price, 5) == 1.11135
    assert round(trade.gross_1r_eur, 2) == 5.00
    assert round(trade.swap_eur, 4) == -0.06
    assert round(trade.net_pnl_eur, 2) == 11.19  # +11.25 - 0.06 swap


def test_backtest_conservative_sl_before_tp_and_time_stop():
    """
    If both SL and TP occur inside the same candle, SL wins conservatively;
    otherwise a flat trade closes at vertical barrier bar 30.
    """
    idx = pd.date_range("2025-03-03 00:00:00", periods=36, freq="4h", tz="UTC")
    opens = np.full(36, 1.1000)
    highs = np.full(36, 1.1010)
    lows = np.full(36, 1.0990)
    closes = np.full(36, 1.1002)

    df_h4 = _make_h4_bars(idx, opens, highs, lows, closes)
    sig_df = pd.DataFrame(
        {
            "side": [1] + [0] * 35,
            "atr14_h4": [0.0025] * 36,
            "sl_distance_price": [0.0050] * 36,
            "tp_distance_price": [0.01125] * 36,
            "cs_spread": [1.0] * 36,
            "ker_d1": [0.50] * 36,
            "trigger_type": ["DONCHIAN_DUAL_MOM"] + ["NONE"] * 35,
        },
        index=idx,
    )
    spec = make_mock_spec("EURUSD", pip_value_001_eur=0.10, digits=5, spread_points=10, swap_long=0.0)
    legacy_cfg = PrimaryStrategyConfig(
        mode="DONCHIAN_DUAL_MOM",
        sl_atr_mult=2.0,
        tp_atr_mult=4.5,
        breakeven_trigger_r=1.20,
        breakeven_lock_r=0.10,
        trailing_activation_r=1.80,
        trailing_atr_mult=2.0,
        vertical_barrier_bars_h4=30,
    )
    engine = PortfolioBacktestEngine(specs={"EURUSD": spec}, trend_config=legacy_cfg, initial_capital_eur=1000.0)

    report_ts = engine.run(h4_bars_by_symbol={"EURUSD": df_h4}, signals_by_symbol={"EURUSD": sig_df}, fixed_lots=0.01)
    assert report_ts.total_trades == 1
    assert report_ts.trades[0].exit_reason == "TIME_STOP"
    assert report_ts.trades[0].bars_held == 30

    df_ambig = df_h4.copy()
    df_ambig.iloc[3, df_ambig.columns.get_loc("high")] = 1.1250
    df_ambig.iloc[3, df_ambig.columns.get_loc("low")] = 1.0800
    report_sl = engine.run(h4_bars_by_symbol={"EURUSD": df_ambig}, signals_by_symbol={"EURUSD": sig_df}, fixed_lots=0.01)
    assert report_sl.total_trades == 1
    assert report_sl.trades[0].exit_reason == "SL"


def test_backtest_breakeven_lock_and_chandelier_trailing():
    """
    Verify BE activates at +1.20R locking +0.10R, and Chandelier Trailing activates at +1.80R
    and closes cleanly on pullback without intrabar lookahead.
    """
    idx = pd.date_range("2025-03-03 00:00:00", periods=10, freq="4h", tz="UTC")
    opens = np.full(10, 1.1000)
    highs = np.full(10, 1.1010)
    lows = np.full(10, 1.0990)
    closes = np.full(10, 1.1005)
    # Entry at bar 1 Ask = 1.10010, sl_dist = 0.00500 (ATR = 0.0025), TP = 1.11135
    # Bar 2 spikes to +1.30R (1.10660) -> activates BE lock at 1.10010 + 0.00050 = 1.10060
    highs[2] = 1.1068
    # Bar 3 pulls back to 1.1004 (hits locked BE at 1.10060!) -> exits as BE with positive +0.10R!
    lows[3] = 1.1004

    df_h4 = _make_h4_bars(idx, opens, highs, lows, closes)
    sig_df = pd.DataFrame(
        {
            "side": [1] + [0] * 9,
            "atr14_h4": [0.0025] * 10,
            "sl_distance_price": [0.0050] * 10,
            "tp_distance_price": [0.01125] * 10,
            "cs_spread": [1.0] * 10,
            "ker_d1": [0.50] * 10,
            "trigger_type": ["DONCHIAN_DUAL_MOM"] + ["NONE"] * 9,
        },
        index=idx,
    )
    spec = make_mock_spec("EURUSD", pip_value_001_eur=0.10, digits=5, spread_points=10, swap_long=0.0)
    legacy_cfg = PrimaryStrategyConfig(
        mode="DONCHIAN_DUAL_MOM",
        sl_atr_mult=2.0,
        tp_atr_mult=4.5,
        breakeven_trigger_r=1.20,
        breakeven_lock_r=0.10,
        trailing_activation_r=1.80,
        trailing_atr_mult=2.0,
        vertical_barrier_bars_h4=30,
    )
    engine = PortfolioBacktestEngine(specs={"EURUSD": spec}, trend_config=legacy_cfg, initial_capital_eur=1000.0)

    rep_be = engine.run({"EURUSD": df_h4}, {"EURUSD": sig_df}, fixed_lots=0.01)
    assert rep_be.total_trades == 1
    assert rep_be.trades[0].exit_reason == "BE"
    assert round(rep_be.trades[0].realized_r, 2) == 0.10

    # Now test Chandelier Trail after +1.80R (entry 1.10010 + 0.0090 = 1.10910)
    # Use 0.0020 bar range so eff_atr=0.0025 -> trail_sl = 1.1100 - 2*0.0025 = 1.1050
    df_tr = df_h4.copy()
    df_tr.iloc[2, df_tr.columns.get_loc("high")] = 1.1100  # +1.98R peak (< TP 1.11135)
    df_tr.iloc[3, df_tr.columns.get_loc("low")] = 1.1045   # Hits trail at 1.1050 (BE was 1.10060)
    rep_tr = engine.run({"EURUSD": df_tr}, {"EURUSD": sig_df}, fixed_lots=0.01)
    assert rep_tr.total_trades == 1
    assert rep_tr.trades[0].exit_reason == "TRAIL"
    assert round(rep_tr.trades[0].realized_r, 2) == 0.98


def test_adaptive_swap_veto_4pct_vs_10pct_by_ker():
    """
    Verify adaptive swap filter: rejects >4% 1R under normal regime (ker_d1 < 0.45),
    but allows up to 10% 1R under strong trend regime (ker_d1 >= 0.45).
    """
    idx = pd.date_range("2025-03-03 00:00:00", periods=10, freq="4h", tz="UTC")
    opens = np.full(10, 1.1000)
    highs = np.full(10, 1.1010)
    lows = np.full(10, 1.0990)
    closes = np.full(10, 1.1002)
    df_h4 = _make_h4_bars(idx, opens, highs, lows, closes)

    # Swap long = -4.0 pts/day (-0.04€/day); 30 H4 bars (5d across Wed) = 7x = -0.28€ vs 5.00€ 1R = 5.6% (>4%, <10%)
    spec = make_mock_spec("EURUSD", pip_value_001_eur=0.10, digits=5, spread_points=10, swap_long=-4.0)
    engine = PortfolioBacktestEngine(specs={"EURUSD": spec}, trend_config=PrimaryStrategyConfig(), initial_capital_eur=1000.0)

    def _run_with_ker(ker_val: float):
        sig_df = pd.DataFrame(
            {
                "side": [1] + [0] * 9,
                "atr14_h4": [0.0025] * 10,
                "sl_distance_price": [0.0050] * 10,
                "tp_distance_price": [0.01125] * 10,
                "cs_spread": [1.0] * 10,
                "ker_d1": [ker_val] * 10,
                "trigger_type": ["DONCHIAN_DUAL_MOM"] + ["NONE"] * 9,
            },
            index=idx,
        )
        return engine.run({"EURUSD": df_h4}, {"EURUSD": sig_df}, fixed_lots=0.01)

    rep_low_ker = _run_with_ker(0.30)
    assert rep_low_ker.total_trades == 0
    assert rep_low_ker.rejected_counts.get("VETO_HOSTILE_SWAP", 0) == 1

    rep_high_ker = _run_with_ker(0.55)
    assert rep_high_ker.total_trades == 1
