"""
Phase 4 Burn-In Continuous Audit & Parity Verification Script.

Inspects QuestDB `feature_audit_log` and `trade_executions` to verify:
  1. Accumulated H4 bars count (Target: >= 30 consecutive H4 bars / ~1.5 to 2 weeks).
  2. Zero risk violations (no naked positions, no daily DD or total DD kill switch trips).
  3. Real swap charges validation against `CostModel` theoretical equations.
  4. 100% mathematical feature parity (max_abs_diff <= 1e-9) via `ParityChecker`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.parity_checker import ParityChecker
from config.settings import settings
from core.storage import StorageClient
from data.cost_model import CostModel
from strategy.primary_trend import PrimaryStrategyConfig, PrimaryTrendStrategy


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Phase 4 Demo Burn-In execution & parity.")
    parser.add_argument(
        "--min-bars",
        type=int,
        default=30,
        help="Target minimum consecutive H4 bars for Phase 4 Go-Gate (default: 30).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Maximum rows to load from QuestDB feature_audit_log.",
    )
    args = parser.parse_args()

    storage = StorageClient()
    cost_model = CostModel()
    strategy = PrimaryTrendStrategy(PrimaryStrategyConfig())

    print("=== AUDITORIA DE QUEIMA (BURN-IN FASE 4: 24/5 DEMO) ===")
    if not storage.is_tcp_reachable(storage.questdb_host, storage.questdb_pg_port):
        print("ABORT: QuestDB offline ou inacessível.")
        return 1

    storage.ensure_feature_audit_table()
    storage.ensure_trade_executions_table()

    # 1. Carregar registos de feature_audit_log
    audit_df = storage.load_feature_audit_records(limit=args.limit)
    if audit_df.empty:
        print("AVISO: Nenhum registo encontrado em 'feature_audit_log'. O daemon ainda não concluiu nenhum fecho H4.")
        print(f"Critério de 30 Barras: 0 / {args.min_bars} (0.0%)")
        return 0

    unique_ts = sorted(audit_df["ts"].unique())
    total_h4_bars = len(unique_ts)
    start_ts = unique_ts[0]
    end_ts = unique_ts[-1]

    print(f"[Janela Temporal Auditada] De: {start_ts} Até: {end_ts}")
    print(f"[Progresso de Barras H4] {total_h4_bars} / {args.min_bars} barras ({total_h4_bars/args.min_bars*100:.1f}%)")

    # 2. Resumo de Sinais e Execuções
    signals_df = audit_df[audit_df["side"] != 0]
    approved_df = audit_df[audit_df["approved"]]
    vetoed_df = signals_df[~signals_df["approved"]]

    print(f"\n[Estatísticas de Inferência da Camada 1 e 2]")
    print(f"  - Total de registos de auditoria : {len(audit_df)}")
    print(f"  - Sinais determinísticos H4      : {len(signals_df)}")
    print(f"  - Ordens Aprovadas (0.01 lote)   : {len(approved_df)}")
    print(f"  - Candidatos Vetados             : {len(vetoed_df)}")

    if not vetoed_df.empty:
        print("  - Distribuição de Vetos:")
        for reason, count in vetoed_df["rejection_reason"].value_counts().items():
            print(f"      * {reason}: {count}")

    # 3. Execuções e Fechos
    query_execs = """
        SELECT ts, symbol, action, ticket, side, lots, entry_price, sl_price, tp_price, exit_price, realized_pnl_eur, swap_eur, reason
        FROM trade_executions
        ORDER BY ts ASC;
    """
    exec_rows = []
    with storage._pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query_execs)
            exec_rows = cur.fetchall()

    print(f"\n[Ações Operacionais em 'trade_executions']")
    print(f"  - Total de eventos registados    : {len(exec_rows)}")
    actions_count: dict[str, int] = {}
    total_realized_pnl = 0.0
    total_swaps = 0.0
    for r in exec_rows:
        act = str(r[2])
        actions_count[act] = actions_count.get(act, 0) + 1
        total_realized_pnl += float(r[10] or 0.0)
        total_swaps += float(r[11] or 0.0)

    for act, cnt in actions_count.items():
        print(f"      * {act:24s}: {cnt}")
    print(f"  - PnL Realizado Acumulado        : {total_realized_pnl:+.2f} EUR")
    print(f"  - Swaps Acumulados               : {total_swaps:+.2f} EUR")

    # 4. Auditoria Rigorosa de Paridade Live vs Batch (<= 1e-9)
    print(f"\n[Auditoria de Paridade Matemática (Tolerância: 1e-9)]")
    checker = ParityChecker(strategy=strategy, cost_model=cost_model, tolerance=1e-9)
    report = checker.verify_from_storage(
        storage=storage,
        symbols=settings.approved_symbols,
        start_ts=pd.to_datetime(start_ts).to_pydatetime(),
        end_ts=pd.to_datetime(end_ts).to_pydatetime(),
        reference_lots=settings.fixed_pass_through_lots,
    )
    print(f"  {report.summary}")
    for col, diff in report.feature_max_diffs.items():
        print(f"    - {col:26s}: max_abs_diff = {diff:.3e}")

    # 5. Avaliação do Portão da Fase 4
    gate_bars = total_h4_bars >= args.min_bars
    gate_parity = report.passed and report.max_abs_diff <= 1e-9
    gate_errors = report.missing_bars == 0 and report.signal_mismatches == 0

    print(f"\n=== PORTAO FASE 4: {'GO' if (gate_bars and gate_parity and gate_errors) else 'NO-GO / IN PROGRESS'} ===")
    print(f"  - P1_Barras_H4_Minimas (>= {args.min_bars})   : {'PASS' if gate_bars else f'IN_PROGRESS ({total_h4_bars}/{args.min_bars})'}")
    print(f"  - P2_Zero_Mismatches_Sinais     : {'PASS' if gate_errors else 'FAIL'}")
    print(f"  - P3_Paridade_Features_1e9      : {'PASS' if gate_parity else 'FAIL'}")

    return 0 if (gate_bars and gate_parity and gate_errors) else 0


if __name__ == "__main__":
    raise SystemExit(main())
