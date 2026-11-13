import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.engine import PortfolioBacktestEngine
from config.settings import settings
from core.mt5_connector import MT5Connector
from core.storage import StorageClient
from strategy.primary_trend import PrimaryStrategyConfig, PrimaryTrendStrategy

BASKET_8 = ["EURUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "GBPUSD", "AUDJPY"]
BASKET_5_LIQUID = ["EURUSD", "USDJPY", "GBPUSD", "EURJPY", "GBPJPY"]


def build_hypothesis_configs() -> list[tuple[str, list[str], PrimaryStrategyConfig]]:
    return [
        # 1. Reversão à Média em Bandas de Bollinger (20, 2.2s) sob Regime Lateral D1
        (
            "H1_BB_RANGE_8P",
            BASKET_8,
            PrimaryStrategyConfig(
                mode="BB_RANGE_REV",
                sl_atr_mult=2.2,
                tp_atr_mult=2.8,
                breakeven_trigger_r=1.00,
                breakeven_lock_r=0.10,
                trailing_activation_r=1.30,
                trailing_atr_mult=1.6,
                vertical_barrier_bars_h4=18,
                max_swap_normal_ratio=0.03,
            ),
        ),
        (
            "H1_BB_RANGE_5L",
            BASKET_5_LIQUID,
            PrimaryStrategyConfig(
                mode="BB_RANGE_REV",
                sl_atr_mult=2.2,
                tp_atr_mult=2.8,
                breakeven_trigger_r=1.00,
                breakeven_lock_r=0.10,
                trailing_activation_r=1.30,
                trailing_atr_mult=1.6,
                vertical_barrier_bars_h4=18,
                max_swap_normal_ratio=0.03,
            ),
        ),
        # 2. Bandas de Bollinger (20, 2.0s) Pullback a favor da Tendência D1
        (
            "H2_BB_TREND_DIP_8P",
            BASKET_8,
            PrimaryStrategyConfig(
                mode="BB_TREND_DIP",
                sl_atr_mult=2.4,
                tp_atr_mult=4.2,
                breakeven_trigger_r=1.10,
                breakeven_lock_r=0.10,
                trailing_activation_r=1.50,
                trailing_atr_mult=2.0,
                vertical_barrier_bars_h4=36,
                max_swap_normal_ratio=0.04,
            ),
        ),
        (
            "H2_BB_TREND_DIP_5L",
            BASKET_5_LIQUID,
            PrimaryStrategyConfig(
                mode="BB_TREND_DIP",
                sl_atr_mult=2.4,
                tp_atr_mult=4.2,
                breakeven_trigger_r=1.10,
                breakeven_lock_r=0.10,
                trailing_activation_r=1.50,
                trailing_atr_mult=2.0,
                vertical_barrier_bars_h4=36,
                max_swap_normal_ratio=0.04,
            ),
        ),
        # 3. Trend Following + Dual Momentum + Carry Lento em D1 (SL ~80-130 pips, Holding até 10 dias úteis)
        (
            "H3_D1_TREND_CARRY_8P",
            BASKET_8,
            PrimaryStrategyConfig(
                mode="D1_TREND_CARRY",
                sl_atr_mult=2.6,
                tp_atr_mult=6.5,
                breakeven_trigger_r=1.25,
                breakeven_lock_r=0.15,
                trailing_activation_r=1.80,
                trailing_atr_mult=2.5,
                vertical_barrier_bars_h4=60,
                max_swap_normal_ratio=0.02,
                max_swap_strong_trend_ratio=0.04,
            ),
        ),
        (
            "H3_D1_TREND_CARRY_5L",
            BASKET_5_LIQUID,
            PrimaryStrategyConfig(
                mode="D1_TREND_CARRY",
                sl_atr_mult=2.6,
                tp_atr_mult=6.5,
                breakeven_trigger_r=1.25,
                breakeven_lock_r=0.15,
                trailing_activation_r=1.80,
                trailing_atr_mult=2.5,
                vertical_barrier_bars_h4=60,
                max_swap_normal_ratio=0.02,
                max_swap_strong_trend_ratio=0.04,
            ),
        ),
        # 4. Carteira Dual-Regime (Tendência D1 + Bandas de Bollinger H4)
        (
            "H4_DUAL_REGIME_8P",
            BASKET_8,
            PrimaryStrategyConfig(
                mode="DUAL_REGIME",
                sl_atr_mult=2.5,
                tp_atr_mult=4.8,
                breakeven_trigger_r=1.15,
                breakeven_lock_r=0.10,
                trailing_activation_r=1.60,
                trailing_atr_mult=2.1,
                vertical_barrier_bars_h4=42,
                max_swap_normal_ratio=0.03,
            ),
        ),
        (
            "H4_DUAL_REGIME_5L",
            BASKET_5_LIQUID,
            PrimaryStrategyConfig(
                mode="DUAL_REGIME",
                sl_atr_mult=2.5,
                tp_atr_mult=4.8,
                breakeven_trigger_r=1.15,
                breakeven_lock_r=0.10,
                trailing_activation_r=1.60,
                trailing_atr_mult=2.1,
                vertical_barrier_bars_h4=42,
                max_swap_normal_ratio=0.03,
            ),
        ),
    ]


def main() -> None:
    storage = StorageClient()
    connector = MT5Connector()
    if not connector.connect():
        print("[ERRO] Falha ao ligar ao MT5 para especificações contratuais.")
        sys.exit(1)

    all_symbols = sorted(set(BASKET_8 + BASKET_5_LIQUID))
    h4_data = {s: storage.load_ohlcv_dataframe("ohlcv_h4", s) for s in all_symbols}
    d1_data = {s: storage.load_ohlcv_dataframe("ohlcv_d1", s) for s in all_symbols}
    specs = {s: connector.get_symbol_spec(s) for s in all_symbols}
    connector.shutdown()

    if any(v is None for v in specs.values()):
        missing = [k for k, v in specs.items() if v is None]
        print(f"[ERRO] Specs indisponíveis para: {missing}")
        sys.exit(1)

    print("\n" + "=" * 134)
    print(
        " TORNEIO CIENTÍFICO DE TESES ESTRUTURAIS (2015-2026) | PORTÃO FASE 1: "
        "PF >= 1.20 | EXP >= +0.12R | DD < 14% | FRIC < 2.0%"
    )
    print("=" * 134)
    print(
        f"{'HIPÓTESE':<22} | {'TRADES':>6} | {'WIN%':>6} | {'PF':>6} | {'EXP(R)':>8} | "
        f"{'MAX DD':>7} | {'FRIC%':>6} | {'SWAP €':>8} | {'FINAL €':>9} | {'FIM':>14} | {'PORTÃO FASE 1'}"
    )
    print("-" * 134)

    for label, basket, cfg in build_hypothesis_configs():
        h4_sub = {s: h4_data[s] for s in basket}
        d1_sub = {s: d1_data[s] for s in basket}
        spec_sub = {s: specs[s] for s in basket if specs[s] is not None}

        strategy = PrimaryTrendStrategy(config=cfg)
        sigs, _ = strategy.generate_signals_for_basket(h4_sub, d1_sub)
        engine = PortfolioBacktestEngine(
            specs=spec_sub,
            trend_config=cfg,
            initial_capital_eur=settings.initial_capital_eur,
        )
        res = engine.run(h4_bars_by_symbol=h4_sub, signals_by_symbol=sigs, fixed_lots=0.01)

        gate_pass = (
            res.profit_factor >= 1.20
            and res.expectancy_r >= 0.12
            and res.max_drawdown_pct < 14.0
            and res.friction_to_gross_risk_pct < 2.0
        )
        gate_str = "GO (4/4)" if gate_pass else "NO-GO"
        end_dt = str(res.equity_curve.index[-1].date()) if not res.equity_curve.empty else "-"
        fim_tag = f"{end_dt}(KS)" if res.kill_switch_total_hit else end_dt

        print(
            f"{label:<22} | {res.total_trades:>6} | {res.win_rate_pct:>5.1f}% | {res.profit_factor:>6.3f} | "
            f"{res.expectancy_r:>+7.4f}R | {res.max_drawdown_pct:>6.2f}% | "
            f"{res.friction_to_gross_risk_pct:>5.2f}% | {res.total_swap_eur:>+7.2f}€ | "
            f"{res.final_equity_eur:>8.2f}€ | {fim_tag:>14} | {gate_str}"
        )

    print("=" * 134 + "\n")


if __name__ == "__main__":
    main()
