from datetime import datetime, timezone
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.engine import PortfolioBacktestEngine
from config.settings import settings
from core.mt5_connector import MT5Connector
from core.storage import StorageClient
from data.ingestion import HistoricalDataIngestor
from strategy.primary_trend import PrimaryStrategyConfig, PrimaryTrendStrategy


def main() -> None:
    storage = StorageClient()
    connector = MT5Connector()
    if not connector.connect():
        print("[ERRO] Falha ao ligar ao MT5.")
        sys.exit(1)

    base_8 = settings.approved_symbols
    extended_10 = base_8 + ["EURJPY", "GBPJPY"]
    eligible_5 = ["EURUSD", "USDJPY", "GBPUSD", "EURJPY", "GBPJPY"]

    # Ensure EURJPY and GBPJPY are also synced in QuestDB
    ingestor = HistoricalDataIngestor(connector=connector, storage=storage)
    h4_counts = storage.count_ohlcv_rows("ohlcv_h4")
    missing_syms = [s for s in ("EURJPY", "GBPJPY") if s not in h4_counts]
    if missing_syms:
        print(f"A sincronizar pares adicionais elegíveis para QuestDB: {missing_syms}...")
        ingestor.sync_all_approved_symbols(
            start_utc=datetime(2015, 1, 1, tzinfo=timezone.utc),
            symbols=missing_syms,
        )

    specs = {}
    h4_all = {}
    d1_all = {}
    for sym in extended_10:
        spec = connector.get_symbol_spec(sym)
        if spec is None:
            print(f"[AVISO] Spec indisponível para {sym}")
            continue
        specs[sym] = spec
        h4_all[sym] = storage.load_ohlcv_dataframe("ohlcv_h4", sym)
        d1_all[sym] = storage.load_ohlcv_dataframe("ohlcv_d1", sym)
    connector.shutdown()

    configs = [
        (
            "Config A (Legada 1.0: BO+PB, TP3, BE1.5, TS24, Swap15%)",
            base_8,
            PrimaryStrategyConfig(
                use_legacy_ema_pullback=True,
                use_legacy_d1_filter=True,
                require_close_location=False,
                tp_atr_mult=3.0,
                breakeven_trigger_r=1.5,
                breakeven_lock_r=0.0,
                trailing_activation_r=99.0,
                vertical_barrier_bars_h4=24,
                max_swap_normal_ratio=0.15,
                max_swap_strong_trend_ratio=0.15,
            ),
        ),
        (
            "Config B (Sem Pullback: Donchian20, TP3, BE1.5, TS24, Swap15%)",
            base_8,
            PrimaryStrategyConfig(
                use_legacy_ema_pullback=False,
                use_legacy_d1_filter=True,
                require_close_location=False,
                tp_atr_mult=3.0,
                breakeven_trigger_r=1.5,
                breakeven_lock_r=0.0,
                trailing_activation_r=99.0,
                vertical_barrier_bars_h4=24,
                max_swap_normal_ratio=0.15,
                max_swap_strong_trend_ratio=0.15,
            ),
        ),
        (
            "Config C (+ Payoff Assimétrico: BE@1.2R, Trail@1.8R, TP4.5, TS30)",
            base_8,
            PrimaryStrategyConfig(
                use_legacy_ema_pullback=False,
                use_legacy_d1_filter=True,
                require_close_location=False,
                tp_atr_mult=4.5,
                breakeven_trigger_r=1.20,
                breakeven_lock_r=0.10,
                trailing_activation_r=1.80,
                trailing_atr_mult=2.0,
                vertical_barrier_bars_h4=30,
                max_swap_normal_ratio=0.15,
                max_swap_strong_trend_ratio=0.15,
            ),
        ),
        (
            "Config D (Canónica 8p: +DualMom KER>=0.25 CS>=0.55 Swap<=4%/10%)",
            base_8,
            PrimaryStrategyConfig(
                mode="DONCHIAN_DUAL_MOM",
                sl_atr_mult=2.0,
                tp_atr_mult=4.5,
                breakeven_trigger_r=1.20,
                breakeven_lock_r=0.10,
                trailing_activation_r=1.80,
                trailing_atr_mult=2.0,
                vertical_barrier_bars_h4=30,
            ),
        ),
        (
            "Config D+ (10 pares: 8 base + EURJPY + GBPJPY)",
            extended_10,
            PrimaryStrategyConfig(
                mode="DONCHIAN_DUAL_MOM",
                sl_atr_mult=2.0,
                tp_atr_mult=4.5,
                breakeven_trigger_r=1.20,
                breakeven_lock_r=0.10,
                trailing_activation_r=1.80,
                trailing_atr_mult=2.0,
                vertical_barrier_bars_h4=30,
            ),
        ),
        (
            "Config D5 (5 pares scanner 01b: EURUSD,USDJPY,GBPUSD,EURJPY,GBPJPY)",
            eligible_5,
            PrimaryStrategyConfig(
                mode="DONCHIAN_DUAL_MOM",
                sl_atr_mult=2.0,
                tp_atr_mult=4.5,
                breakeven_trigger_r=1.20,
                breakeven_lock_r=0.10,
                trailing_activation_r=1.80,
                trailing_atr_mult=2.0,
                vertical_barrier_bars_h4=30,
            ),
        ),
    ]

    print("\n" + "=" * 130)
    print(" ESTUDO DE ABLAÇÃO CIENTÍFICA — FASE 1 (2015–2026 | 1.000 EUR | 0.01 LOTE FIXO | CUSTOS REAIS PEPPERSTONE)")
    print("=" * 130)
    print(
        f"{'CONFIGURAÇÃO':<62} | {'N':>4} | {'WIN%':>5} | {'PF':>5} | {'EXP(R)':>8} | "
        f"{'MAXDD':>6} | {'FRIC%':>5} | {'SWAP€':>7} | {'FINAL€':>8} | {'FIM'}"
    )
    print("-" * 130)

    for label, syms, cfg in configs:
        h4_sub = {s: h4_all[s] for s in syms if s in h4_all}
        d1_sub = {s: d1_all[s] for s in syms if s in d1_all}
        spec_sub = {s: specs[s] for s in syms if s in specs}

        strat = PrimaryTrendStrategy(config=cfg)
        sigs, _ = strat.generate_signals_for_basket(h4_sub, d1_sub)
        eng = PortfolioBacktestEngine(specs=spec_sub, trend_config=cfg, initial_capital_eur=1000.0)
        rep = eng.run(h4_sub, sigs, fixed_lots=0.01)

        end_dt = str(rep.equity_curve.index[-1].date()) if not rep.equity_curve.empty else "-"
        ks_tag = f"{end_dt}(KS)" if rep.kill_switch_total_hit else end_dt
        print(
            f"{label:<62} | {rep.total_trades:>4} | {rep.win_rate_pct:>4.1f}% | {rep.profit_factor:>5.3f} | "
            f"{rep.expectancy_r:>+7.4f}R | {rep.max_drawdown_pct:>5.2f}% | {rep.friction_to_gross_risk_pct:>4.2f}% | "
            f"{rep.total_swap_eur:>+6.2f}€ | {rep.final_equity_eur:>7.2f}€ | {ks_tag}"
        )

    print("=" * 130 + "\n")


if __name__ == "__main__":
    main()
