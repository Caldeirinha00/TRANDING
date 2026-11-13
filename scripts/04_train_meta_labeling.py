from collections import defaultdict
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.engine import BacktestReport, PortfolioBacktestEngine
from backtest.validation import (
    PurgedEmbargoedWalkForwardSplit,
    compute_brier_skill_score,
    compute_deflated_sharpe_ratio,
)
from config.settings import settings
from core.mt5_connector import MT5Connector
from core.storage import StorageClient
from data.cost_model import CostModel
from strategy.meta_model import MetaModelHyperParams, run_purged_walk_forward_meta_labeling
from strategy.primary_trend import PrimaryTrendStrategy
from strategy.triple_barrier import TripleBarrierConfig, TripleBarrierLabeler

PRIOR_PHASE1_TRIALS_K = 14  # 6 trials in 03b ablation + 8 trials in 03c hypothesis tournament
PHASE2_TRIALS_K = 1         # Single pre-specified Calibrated LGBM Meta-Labeling test
TOTAL_TRIALS_K = PRIOR_PHASE1_TRIALS_K + PHASE2_TRIALS_K


def _extract_daily_returns(report: BacktestReport) -> np.ndarray:
    if report.equity_curve.empty:
        return np.array([], dtype=float)
    eq_daily = report.equity_curve["equity_eur"].resample("1D").last().dropna()
    rets = eq_daily.pct_change().dropna().to_numpy(dtype=float)
    return rets[np.isfinite(rets)]


def main() -> None:
    storage = StorageClient()
    connector = MT5Connector()
    if not connector.connect():
        print("[ERRO] Falha ao ligar ao MT5 para obter especificações contratuais.")
        sys.exit(1)

    h4_bars = {}
    d1_bars = {}
    specs = {}
    benchmark_spreads = settings.benchmark_spread_points
    for sym in settings.approved_symbols:
        spec = connector.get_historical_benchmark_spec(
            sym,
            benchmark_spread_points=benchmark_spreads.get(sym),
        )
        if spec is None:
            print(f"[ERRO] Spec indisponível para {sym}")
            sys.exit(1)
        specs[sym] = spec
        h4_bars[sym] = storage.load_ohlcv_dataframe("ohlcv_h4", sym)
        d1_bars[sym] = storage.load_ohlcv_dataframe("ohlcv_d1", sym)
    connector.shutdown()

    strategy = PrimaryTrendStrategy()  # Locked H2_BB_TREND_DIP_8P
    signals_by_symbol, _ = strategy.generate_signals_for_basket(h4_bars, d1_bars)
    cost_model = CostModel()

    # Keep unresolved trailing bars & risk/swap ineligible events labeled for clean scoring,
    # while training strictly on resolved & risk/swap-eligible events.
    labeler = TripleBarrierLabeler(
        cost_model=cost_model,
        strategy_cfg=strategy.cfg,
        barrier_cfg=TripleBarrierConfig(
            reference_capital_eur=settings.initial_capital_eur,
            reference_lots=0.01,
            filter_risk_and_swap_ineligible=False,
            drop_unresolved_end_of_sample=False,
        ),
    )
    events_df = labeler.label_basket(h4_bars, d1_bars, signals_by_symbol, specs)
    if events_df.empty:
        print("[ERRO] Nenhuma ocorrência candidata gerada pela Camada 1.")
        sys.exit(1)

    train_pool = events_df[events_df["is_resolved_label"] & events_df["is_risk_swap_eligible"]]
    print("\n" + "=" * 100)
    print(" FASE 2 — TREINO META-LABELING PURGED & EMBARGOED WALK-FORWARD (H2_BB_TREND_DIP_8P)")
    print("=" * 100)
    print(f"Eventos candidatos Camada 1 (total)    : {len(events_df)}")
    print(
        f"Eventos elegíveis risco/swap resolvidos: {len(train_pool)} "
        f"(Win Rate base = {train_pool['meta_label'].mean() * 100:.2f}%, "
        f"Média R = {train_pool['realized_r'].mean():+.4f}R)"
    )

    splitter = PurgedEmbargoedWalkForwardSplit(
        initial_train_months=48,
        test_step_months=6,
        expanding=True,
        embargo_hours=24,
        min_train_samples=120,
        anchor_calendar_month_start=True,
    )
    params = MetaModelHyperParams()
    wf_result = run_purged_walk_forward_meta_labeling(
        events_df=events_df,
        splitter=splitter,
        params=params,
    )

    min_sig = events_df["signal_time"].min()
    oos_anchor = pd.Timestamp(year=min_sig.year, month=min_sig.month, day=1, tz="UTC") + pd.DateOffset(months=48)
    oos_start_dt = oos_anchor.to_pydatetime()

    oos_all = wf_result.oos_predictions_df
    eval_oos = oos_all[oos_all["is_resolved_label"] & oos_all["is_risk_swap_eligible"]]
    if eval_oos.empty:
        eval_oos = oos_all

    bss_global = compute_brier_skill_score(
        y_true=eval_oos["meta_label"].to_numpy(dtype=int),
        y_prob=eval_oos["meta_prob"].to_numpy(dtype=float),
    )

    probs = eval_oos["meta_prob"].to_numpy(dtype=float)
    exp_rs = eval_oos["expected_net_r"].to_numpy(dtype=float)
    pass_gate = (probs >= settings.meta_min_probability) & (exp_rs >= settings.meta_min_expected_net_r)
    high_conv = pass_gate & (probs >= settings.meta_high_conviction_probability)

    print(
        f"Amostras OOS ({oos_anchor.date()} .. {oos_all['signal_time'].max().date()}) : "
        f"{len(oos_all)} total | {len(eval_oos)} elegíveis | "
        f"Aprovadas Meta (>= {settings.meta_min_probability:.2f} & >= +{settings.meta_min_expected_net_r:.2f}R): "
        f"{int(pass_gate.sum())} ({pass_gate.mean() * 100:.1f}%) | High-Conv 0.02 lots: {int(high_conv.sum())}"
    )
    print(
        f"Calibração OOS (p) : mean={probs.mean():.4f} | std={probs.std():.4f} | "
        f"p25={np.percentile(probs, 25):.4f} | med={np.median(probs):.4f} | "
        f"p75={np.percentile(probs, 75):.4f} | max={probs.max():.4f}"
    )
    print(
        f"Brier Score Model={bss_global['brier_model']:.5f} vs Climatology={bss_global['brier_baseline']:.5f} "
        f"=> BSS = {bss_global['brier_skill_score']:+.5f}"
    )

    print("\n--- RESUMO POR FOLD WALK-FORWARD (PURGED & EMBARGOED) ---")
    print(
        f"{'FOLD':>4} | {'JANELA OOS':<23} | {'N_TRN':>5} | {'PRG':>3} | {'EMB':>3} | "
        f"{'N_TST':>5} | {'TRN_WR':>6} | {'TST_WR':>6} | {'P_MEAN':>6} | {'BSS':>8}"
    )
    print("-" * 92)
    for fs in wf_result.fold_summaries:
        win_str = f"{fs['test_start']}..{fs['test_end']}"
        print(
            f"{fs['fold_id']:>4} | {win_str:<23} | {fs['n_train']:>5} | {fs['purged']:>3} | {fs['embargoed']:>3} | "
            f"{fs['n_test_eval']:>5} | {fs['train_win_rate']*100:>5.1f}% | {fs['test_win_rate']*100:>5.1f}% | "
            f"{fs['mean_oos_prob']:>6.3f} | {fs['brier_skill_score']:>+8.4f}"
        )

    print("\n--- IMPORTÂNCIA MÉDIA DAS FEATURES (LIGHTGBM OOS) ---")
    for feat_name, imp_val in sorted(wf_result.aggregate_feature_importances.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat_name:<26}: {imp_val * 100:5.2f}%")

    # Execute strictly paired Out-of-Sample Portfolio Backtests on identical [oos_start_dt .. end]
    engine_base = PortfolioBacktestEngine(
        specs=specs,
        cost_model=cost_model,
        trend_config=strategy.cfg,
        initial_capital_eur=settings.initial_capital_eur,
    )
    rep_base_oos = engine_base.run(
        h4_bars_by_symbol=h4_bars,
        signals_by_symbol=signals_by_symbol,
        fixed_lots=0.01,
        meta_decisions_by_symbol=None,
        start_time=oos_start_dt,
    )

    engine_meta = PortfolioBacktestEngine(
        specs=specs,
        cost_model=cost_model,
        trend_config=strategy.cfg,
        initial_capital_eur=settings.initial_capital_eur,
    )
    rep_meta_oos = engine_meta.run(
        h4_bars_by_symbol=h4_bars,
        signals_by_symbol=signals_by_symbol,
        fixed_lots=None,
        meta_decisions_by_symbol=wf_result.meta_decisions_by_symbol,
        start_time=oos_start_dt,
    )

    dsr_base = compute_deflated_sharpe_ratio(
        _extract_daily_returns(rep_base_oos),
        num_trials_k=PRIOR_PHASE1_TRIALS_K,
        periods_per_year=252,
    )
    dsr_meta = compute_deflated_sharpe_ratio(
        _extract_daily_returns(rep_meta_oos),
        num_trials_k=TOTAL_TRIALS_K,
        periods_per_year=252,
    )

    trade_r_meta = np.array([t.realized_r for t in rep_meta_oos.trades], dtype=float)
    dsr_meta_trades = compute_deflated_sharpe_ratio(
        trade_r_meta,
        num_trials_k=TOTAL_TRIALS_K,
        periods_per_year=max(1, int(round(len(trade_r_meta) / max(1.0, (oos_all["signal_time"].max() - oos_anchor).days / 365.25)))),
    )

    precision_lift_pp = rep_meta_oos.win_rate_pct - rep_base_oos.win_rate_pct
    lots_dist = defaultdict(int)
    for t in rep_meta_oos.trades:
        lots_dist[round(t.lots, 2)] += 1

    print("\n" + "=" * 100)
    print(f" COMPARAÇÃO OUT-OF-SAMPLE STRICT ({oos_anchor.date()} .. {rep_base_oos.equity_curve.index[-1].date()})")
    print("=" * 100)
    print(f"{'MÉTRICA':<34} | {'FASE 1 OOS (BASE)':>22} | {'FASE 2 OOS (HÍBRIDO)':>22} | {'DELTA / STATUS':>16}")
    print("-" * 100)
    print(f"{'Capital Final (Base 1000 EUR)':<34} | {rep_base_oos.final_equity_eur:>20.2f}€ | {rep_meta_oos.final_equity_eur:>20.2f}€ | {rep_meta_oos.final_equity_eur - rep_base_oos.final_equity_eur:>+14.2f}€")
    print(f"{'Retorno Líquido (%)':<34} | {rep_base_oos.net_return_pct:>+21.2f}% | {rep_meta_oos.net_return_pct:>+21.2f}% | {rep_meta_oos.net_return_pct - rep_base_oos.net_return_pct:>+14.2f}pp")
    print(f"{'Total Trades Executados':<34} | {rep_base_oos.total_trades:>22} | {rep_meta_oos.total_trades:>22} | {rep_meta_oos.total_trades - rep_base_oos.total_trades:>+16}")
    print(f"{'Precision / Win Rate (%)':<34} | {rep_base_oos.win_rate_pct:>21.2f}% | {rep_meta_oos.win_rate_pct:>21.2f}% | {precision_lift_pp:>+14.2f}pp")
    print(f"{'Profit Factor Líquido':<34} | {rep_base_oos.profit_factor:>22.3f} | {rep_meta_oos.profit_factor:>22.3f} | {rep_meta_oos.profit_factor - rep_base_oos.profit_factor:>+16.3f}")
    print(f"{'Expectância Líquida (R)':<34} | {rep_base_oos.expectancy_r:>+21.4f}R | {rep_meta_oos.expectancy_r:>+21.4f}R | {rep_meta_oos.expectancy_r - rep_base_oos.expectancy_r:>+15.4f}R")
    print(f"{'Max Drawdown OOS (%)':<34} | {rep_base_oos.max_drawdown_pct:>21.2f}% | {rep_meta_oos.max_drawdown_pct:>21.2f}% | {rep_meta_oos.max_drawdown_pct - rep_base_oos.max_drawdown_pct:>+14.2f}pp")
    print(f"{'Sharpe Anualizado (Diário)':<34} | {rep_base_oos.sharpe_annualized:>22.3f} | {rep_meta_oos.sharpe_annualized:>22.3f} | {rep_meta_oos.sharpe_annualized - rep_base_oos.sharpe_annualized:>+16.3f}")
    print(f"{'Rácio de Calmar OOS':<34} | {rep_base_oos.calmar_ratio:>22.3f} | {rep_meta_oos.calmar_ratio:>22.3f} | {rep_meta_oos.calmar_ratio - rep_base_oos.calmar_ratio:>+16.3f}")
    print(f"{'Fricção S+C / 1R (%)':<34} | {rep_base_oos.friction_to_gross_risk_pct:>21.2f}% | {rep_meta_oos.friction_to_gross_risk_pct:>21.2f}% | {rep_meta_oos.friction_to_gross_risk_pct - rep_base_oos.friction_to_gross_risk_pct:>+14.2f}pp")
    print(f"{'DSR Diário (K=' + str(PRIOR_PHASE1_TRIALS_K) + ' vs K=' + str(TOTAL_TRIALS_K) + ')':<34} | {dsr_base.dsr:>22.4f} | {dsr_meta.dsr:>22.4f} | {dsr_meta.dsr - dsr_base.dsr:>+16.4f}")
    print(f"{'DSR por Trade (K=' + str(TOTAL_TRIALS_K) + ')':<34} | {'-':>22} | {dsr_meta_trades.dsr:>22.4f} | {'':>16}")
    print(f"Alocação Discreta Híbrido (lotes): {dict(lots_dist)}")
    print(f"Vetos Híbrido OOS                : {rep_meta_oos.rejected_counts}")

    # Evaluate strict Fase 2 Gate
    c1_pf = rep_meta_oos.profit_factor >= 1.20
    c2_exp = rep_meta_oos.expectancy_r >= 0.10
    c3_lift = precision_lift_pp >= 5.0
    c4_dd = rep_meta_oos.max_drawdown_pct < rep_base_oos.max_drawdown_pct
    c5_ratios = (rep_meta_oos.sharpe_annualized > rep_base_oos.sharpe_annualized) and (
        rep_meta_oos.calmar_ratio > rep_base_oos.calmar_ratio
    )
    c6_bss = bss_global["brier_skill_score"] > 0.0
    c7_dsr = dsr_meta.dsr >= 0.95

    checks = [
        ("1. Profit Factor OOS >= 1.20", c1_pf, f"{rep_meta_oos.profit_factor:.3f}"),
        ("2. Expectância OOS >= +0.10R", c2_exp, f"{rep_meta_oos.expectancy_r:+.4f}R"),
        ("3. Lift Precision >= +5.0 pp", c3_lift, f"{precision_lift_pp:+.2f} pp"),
        ("4. MaxDD OOS < Fase 1 OOS", c4_dd, f"{rep_meta_oos.max_drawdown_pct:.2f}% vs {rep_base_oos.max_drawdown_pct:.2f}%"),
        (
            "5. Sharpe & Calmar > Fase 1 OOS",
            c5_ratios,
            f"Sh={rep_meta_oos.sharpe_annualized:.3f}/{rep_base_oos.sharpe_annualized:.3f}, "
            f"Cal={rep_meta_oos.calmar_ratio:.3f}/{rep_base_oos.calmar_ratio:.3f}",
        ),
        ("6. Brier Skill Score > 0.0", c6_bss, f"{bss_global['brier_skill_score']:+.5f}"),
        ("7. DSR (K=15) >= 0.95", c7_dsr, f"{dsr_meta.dsr:.4f} (SR* ann={dsr_meta.sr_threshold_annualized:.3f})"),
    ]
    n_pass = sum(1 for _, ok, _ in checks if ok)

    print("\n--- VERIFICAÇÃO DO PORTÃO GO / NO-GO (FASE 2 — META-LABELING OOS) ---")
    for label, ok, detail in checks:
        print(f"{label:<32} : {'PASS' if ok else 'FAIL'} ({detail})")
    print(f"PORTÃO FASE 2: {'GO' if n_pass == len(checks) else 'NO-GO'} ({n_pass}/{len(checks)} PASS)")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
