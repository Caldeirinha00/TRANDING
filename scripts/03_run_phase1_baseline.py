from collections import defaultdict
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.engine import PortfolioBacktestEngine
from config.settings import settings
from core.mt5_connector import MT5Connector
from core.storage import StorageClient
from strategy.primary_trend import PrimaryTrendStrategy


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

    strategy = PrimaryTrendStrategy()  # Baseline travado em H2_BB_TREND_DIP_8P
    signals_by_symbol, _ = strategy.generate_signals_for_basket(h4_bars, d1_bars)

    engine = PortfolioBacktestEngine(specs=specs, trend_config=strategy.cfg, initial_capital_eur=1000.0)
    report = engine.run(h4_bars_by_symbol=h4_bars, signals_by_symbol=signals_by_symbol, fixed_lots=0.01)

    print("\n" + "=" * 98)
    print(" RELATÓRIO FASE 1 — BASELINE OFICIAL TRAVADO (H2_BB_TREND_DIP_8P | ALTO RECALL | 0.01 LOTE FIXO)")
    print("=" * 98)
    print(f"Período analisado        : {report.equity_curve.index[0].date()} .. {report.equity_curve.index[-1].date()}")
    print(f"Capital Inicial -> Final : {report.initial_capital_eur:.2f} EUR -> {report.final_equity_eur:.2f} EUR ({report.net_return_pct:+.2f}%)")
    print(f"Total de Trades          : {report.total_trades}")
    print(f"Win Rate                 : {report.win_rate_pct:.2f}%")
    print(f"Profit Factor Líquido    : {report.profit_factor:.3f}  (Meta Alto Recall: >= 1.050)")
    print(f"Expectância Líquida (R)  : {report.expectancy_r:+.4f}R (Meta Alto Recall: >= +0.0250R)")
    print(f"Expectância Líquida (EUR): {report.expectancy_eur:+.2f} EUR/trade")
    print(f"Max Drawdown Histórico   : {report.max_drawdown_pct:.2f}%   (Meta: < 14.00%)")
    print(f"Sharpe Anualizado        : {report.sharpe_annualized:.3f}")
    print(f"Rácio de Calmar          : {report.calmar_ratio:.3f}")
    print(f"Fricção Spread+Com / 1R  : {report.friction_to_gross_risk_pct:.2f}%   (Meta: < 2.00%)")
    print(f"Swap Acumulado Líquido   : {report.total_swap_eur:+.2f} EUR")
    print(f"Kill Switches (Diár/Tot) : {report.kill_switch_daily_hits} / {report.kill_switch_total_hit}")

    exit_counts = defaultdict(int)
    for t in report.trades:
        exit_counts[t.exit_reason] += 1
    print(f"Motivos de Saída         : {dict(exit_counts)}")
    if report.rejected_counts:
        print(f"Sinais Filtrados/Vetados : {report.rejected_counts}")

    print("\n--- BREAKDOWN POR SÍMBOLO ---")
    print(f"{'SÍMBOLO':<8} | {'TRADES':>6} | {'WIN%':>6} | {'PF':>6} | {'EXP(R)':>8} | {'NET EUR':>9} | {'SWAP EUR':>9} | {'FRIC%':>6}")
    print("-" * 78)
    positive_pairs = 0
    for sym in settings.approved_symbols:
        sym_trades = [t for t in report.trades if t.symbol == sym]
        if not sym_trades:
            continue
        pnls = np.array([t.net_pnl_eur for t in sym_trades])
        rs = np.array([t.realized_r for t in sym_trades])
        gw = float(pnls[pnls > 0].sum())
        gl = float(np.abs(pnls[pnls < 0].sum()))
        pf = gw / gl if gl > 0 else 99.0
        wr = (pnls > 0).mean() * 100.0
        fric = sum(t.spread_cost_eur + t.commission_eur for t in sym_trades)
        g1r = sum(t.gross_1r_eur for t in sym_trades)
        sw = sum(t.swap_eur for t in sym_trades)
        net_sum = float(pnls.sum())
        if net_sum > 0:
            positive_pairs += 1
        print(
            f"{sym:<8} | {len(sym_trades):>6} | {wr:>5.1f}% | {pf:>6.2f} | {rs.mean():>+7.3f}R | "
            f"{net_sum:>+8.2f}€ | {sw:>+8.2f}€ | {(fric/g1r*100):>5.2f}%"
        )

    print("\n--- VERIFICAÇÃO DO PORTÃO GO / NO-GO (FASE 1 — ALTO RECALL) ---")
    c_pf = report.profit_factor >= 1.05
    c_exp = report.expectancy_r >= 0.025
    c_dd = report.max_drawdown_pct < 14.0
    c_fric = report.friction_to_gross_risk_pct < 2.0
    c_amp = positive_pairs >= 6
    passed_count = sum([c_pf, c_exp, c_dd, c_fric, c_amp])
    print(f"1. Profit Factor >= 1.05      : {'PASS' if c_pf else 'FAIL'} ({report.profit_factor:.3f})")
    print(f"2. Expectância >= +0.025R     : {'PASS' if c_exp else 'FAIL'} ({report.expectancy_r:+.4f}R)")
    print(f"3. Max Drawdown < 14.0%       : {'PASS' if c_dd else 'FAIL'} ({report.max_drawdown_pct:.2f}%)")
    print(f"4. Fricção S+C < 2.0% 1R      : {'PASS' if c_fric else 'FAIL'} ({report.friction_to_gross_risk_pct:.2f}%)")
    print(f"5. Amplitude >= 6/8 Pares (+) : {'PASS' if c_amp else 'FAIL'} ({positive_pairs}/8 pares)")
    print(f"PORTÃO FASE 1: {'GO' if passed_count == 5 else 'NO-GO'} ({passed_count}/5 PASS)")
    print("=" * 98 + "\n")


if __name__ == "__main__":
    main()
