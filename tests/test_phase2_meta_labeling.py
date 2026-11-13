import numpy as np
import pandas as pd
from backtest.engine import PortfolioBacktestEngine
from backtest.validation import (
    PurgedEmbargoedWalkForwardSplit,
    compute_brier_skill_score,
    compute_deflated_sharpe_ratio,
    compute_sample_uniqueness_weights,
)
from data.cost_model import CostModel
from strategy.features import FEATURE_COLUMNS
from strategy.meta_model import (
    MetaModelHyperParams,
    run_purged_walk_forward_meta_labeling,
)
from strategy.primary_trend import PrimaryStrategyConfig
from strategy.triple_barrier import TripleBarrierConfig, TripleBarrierLabeler
from tests.test_esma_risk_math import make_mock_spec


def test_purged_and_embargoed_walk_forward_zero_leakage():
    """
    Training events whose exit_time overlaps or falls inside the embargo buffer before
    `test_start` must be strictly purged/embargoed.
    """
    sig_times = pd.to_datetime(
        [
            "2024-01-05 00:00:00",
            "2024-02-10 00:00:00",
            "2024-03-28 00:00:00",  # Overlaps April 1 test boundary (purged)
            "2024-03-31 12:00:00",  # Closes exactly in 24h embargo buffer (embargoed)
            "2024-04-02 00:00:00",  # First OOS test sample
            "2024-04-15 00:00:00",
        ],
        utc=True,
    )
    exit_times = pd.to_datetime(
        [
            "2024-01-08 00:00:00",
            "2024-02-14 00:00:00",
            "2024-04-03 00:00:00",
            "2024-03-31 20:00:00",
            "2024-04-05 00:00:00",
            "2024-04-18 00:00:00",
        ],
        utc=True,
    )
    splitter = PurgedEmbargoedWalkForwardSplit(
        initial_train_months=3,
        test_step_months=1,
        expanding=True,
        embargo_hours=24,
        min_train_samples=1,
        anchor_calendar_month_start=True,
    )
    folds = splitter.split(sig_times, exit_times)
    assert len(folds) == 1
    f = folds[0]
    assert list(f.train_indices) == [0, 1]
    assert list(f.test_indices) == [4, 5]
    assert f.purged_count == 1
    assert f.embargoed_count == 1
    assert all(exit_times[i] <= pd.Timestamp("2024-03-31 00:00:00+00:00") for i in f.train_indices)


def test_sample_uniqueness_weights_penalize_concurrency():
    """
    Overlapping simultaneous events receive lower weights than isolated non-overlapping events,
    while normalized weights always average to 1.0.
    """
    starts = pd.to_datetime(
        [
            "2024-01-01 00:00:00",
            "2024-01-01 00:00:00",
            "2024-01-01 00:00:00",
            "2024-01-10 00:00:00",  # Completely isolated
        ],
        utc=True,
    )
    ends = pd.to_datetime(
        [
            "2024-01-03 00:00:00",
            "2024-01-03 00:00:00",
            "2024-01-03 00:00:00",
            "2024-01-12 00:00:00",
        ],
        utc=True,
    )
    w = compute_sample_uniqueness_weights(starts, ends)
    assert len(w) == 4
    assert np.isclose(w.mean(), 1.0, atol=1e-9)
    assert w[3] > w[0]
    assert np.isclose(w[0], w[1]) and np.isclose(w[1], w[2])
    assert np.isclose(w[3] / w[0], 3.0, atol=1e-6)


def test_deflated_sharpe_ratio_and_brier_skill_score():
    """Verify multiple-testing penalty (K=15 < K=1) and Brier Skill Score properties."""
    rng = np.random.default_rng(7)
    daily_rets = rng.normal(0.0008, 0.010, size=1000)

    res_k1 = compute_deflated_sharpe_ratio(daily_rets, num_trials_k=1, periods_per_year=252)
    res_k15 = compute_deflated_sharpe_ratio(daily_rets, num_trials_k=15, periods_per_year=252)
    assert res_k1.observed_sr_annualized > 0.0
    assert res_k15.sr_threshold_annualized > 0.0
    assert res_k15.dsr < res_k1.dsr

    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    good_probs = np.array([0.85, 0.15, 0.80, 0.75, 0.20, 0.25, 0.90, 0.10])
    bad_probs = np.array([0.15, 0.85, 0.20, 0.25, 0.80, 0.75, 0.10, 0.90])
    assert compute_brier_skill_score(y_true, good_probs)["brier_skill_score"] > 0.5
    assert compute_brier_skill_score(y_true, bad_probs)["brier_skill_score"] < 0.0


def test_triple_barrier_parity_with_portfolio_backtest_engine():
    """
    TripleBarrierLabeler must produce bit-for-bit identical entry_price, exit_price,
    swap_eur, net_pnl_eur, and realized_r as PortfolioBacktestEngine on Wednesday rollover.
    """
    idx_h4 = pd.date_range("2025-02-20 00:00:00", periods=80, freq="4h", tz="UTC")
    opens = np.full(80, 1.1000)
    highs = np.full(80, 1.1010)
    lows = np.full(80, 1.0990)
    closes = np.full(80, 1.1005)
    # Place signal at Tuesday 2025-03-04 20:00 (idx 74 -> enter Wed 00:00, exit TP Thu 04:00 idx 82? Let's extend)
    idx_h4 = pd.date_range("2025-02-20 00:00:00", periods=90, freq="4h", tz="UTC")
    opens = np.full(90, 1.1000)
    highs = np.full(90, 1.1010)
    lows = np.full(90, 1.0990)
    closes = np.full(90, 1.1005)
    sig_ts = pd.Timestamp("2025-03-04 20:00:00+00:00")
    sig_pos = idx_h4.get_loc(sig_ts)
    highs[sig_pos + 8] = 1.1130  # Thu 04:00 crosses TP

    df_h4 = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "tick_volume": np.full(90, 1000.0),
            "spread": np.zeros(90, dtype=int),
            "real_volume": np.zeros(90, dtype=int),
        },
        index=idx_h4,
    )
    idx_d1 = pd.date_range("2024-12-01 00:00:00", periods=100, freq="1D", tz="UTC")
    df_d1 = pd.DataFrame(
        {
            "open": np.full(100, 1.1000),
            "high": np.full(100, 1.1020),
            "low": np.full(100, 1.0980),
            "close": np.full(100, 1.1005),
            "tick_volume": np.full(100, 5000.0),
        },
        index=idx_d1,
    )
    sides = np.zeros(90, dtype=int)
    sides[sig_pos] = 1
    sig_df = pd.DataFrame(
        {
            "side": sides,
            "atr14_h4": np.full(90, 0.0025),
            "sl_distance_price": np.full(90, 0.0050),
            "tp_distance_price": np.full(90, 0.01125),
            "cs_spread": np.where(sides != 0, 0.90, 0.0),
            "cs_long_spread": np.full(90, 0.90),
            "ker_d1": np.full(90, 0.50),
            "trigger_type": np.where(sides != 0, "BB_TREND_DIP", "NONE"),
        },
        index=idx_h4,
    )

    spec = make_mock_spec(
        "EURUSD",
        pip_value_001_eur=0.10,
        digits=5,
        spread_points=10,
        swap_long=-2.0,
        swap_short=0.5,
    )
    cfg = PrimaryStrategyConfig()
    cost_model = CostModel(commission_per_lot_roundtrip_eur=0.0)

    labeler = TripleBarrierLabeler(
        cost_model=cost_model,
        strategy_cfg=cfg,
        barrier_cfg=TripleBarrierConfig(filter_risk_and_swap_ineligible=False, drop_unresolved_end_of_sample=False),
    )
    events = labeler.label_symbol_events("EURUSD", df_h4, df_d1, sig_df, spec)
    engine = PortfolioBacktestEngine(specs={"EURUSD": spec}, cost_model=cost_model, trend_config=cfg)
    report = engine.run({"EURUSD": df_h4}, {"EURUSD": sig_df}, fixed_lots=0.01)

    assert len(events) == 1
    assert report.total_trades == 1
    ev = events.iloc[0]
    tr = report.trades[0]
    assert ev["exit_reason"] == tr.exit_reason == "TP"
    assert round(float(ev["entry_price"]), 5) == round(tr.entry_price, 5)
    assert round(float(ev["exit_price"]), 5) == round(tr.exit_price, 5)
    assert round(float(ev["swap_eur"]), 4) == round(tr.swap_eur, 4) == -0.06
    assert round(float(ev["net_pnl_eur"]), 4) == round(tr.net_pnl_eur, 4)
    assert round(float(ev["realized_r"]), 4) == round(tr.realized_r, 4)


def test_purged_walk_forward_meta_labeling_end_to_end():
    """Smoke test full CalibratedLightGBMMetaModel pipeline across purged walk-forward folds."""
    rng = np.random.default_rng(123)
    n_events = 220
    sig_times = pd.date_range("2023-01-01 00:00:00", periods=n_events, freq="2D", tz="UTC")
    exit_times = sig_times + pd.Timedelta(hours=24)

    x_mat = rng.normal(0.0, 1.0, size=(n_events, len(FEATURE_COLUMNS)))
    # Inject informative structure in cs_momentum_spread (col 2) and ker_d1 (col 0)
    logit = 1.1 * x_mat[:, 2] + 0.7 * x_mat[:, 0]
    probs_true = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n_events) < probs_true).astype(int)
    realized_r = np.where(y == 1, 1.25, -0.98)

    events_df = pd.DataFrame(x_mat, columns=FEATURE_COLUMNS)
    events_df["symbol"] = np.where(np.arange(n_events) % 2 == 0, "EURUSD", "USDJPY")
    events_df["signal_time"] = sig_times
    events_df["entry_time"] = sig_times + pd.Timedelta(hours=4)
    events_df["exit_time"] = exit_times
    events_df["realized_r"] = realized_r
    events_df["meta_label"] = y
    events_df["is_risk_swap_eligible"] = True
    events_df["is_resolved_label"] = True

    splitter = PurgedEmbargoedWalkForwardSplit(
        initial_train_months=8,
        test_step_months=3,
        expanding=True,
        embargo_hours=24,
        min_train_samples=50,
    )
    params = MetaModelHyperParams(n_estimators=40, min_child_samples=10)
    wf_res = run_purged_walk_forward_meta_labeling(events_df, splitter=splitter, params=params)

    assert not wf_res.oos_predictions_df.empty
    assert len(wf_res.fold_summaries) >= 2
    assert set(wf_res.meta_decisions_by_symbol.keys()) == {"EURUSD", "USDJPY"}
    oos_probs = wf_res.oos_predictions_df["meta_prob"].to_numpy()
    assert np.all((oos_probs >= 0.0) & (oos_probs <= 1.0))
    assert sum(wf_res.aggregate_feature_importances.values()) > 0.99
