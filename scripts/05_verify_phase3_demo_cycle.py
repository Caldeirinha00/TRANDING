"""
Phase 3 Live Verification & Parity Gate (`PepperstoneUK-Demo` ~1,000 EUR).

Verifies against live PepperstoneUK-Demo MT5 terminal, QuestDB, and DragonflyDB:
  1. MT5 Demo safety guard & connection (rejects real accounts and >= 50k accounts).
  2. H4ServerClock broker server clock synchronization (EET/EEST DST offset).
  3. Live TieredWatchdog audit (daily/total DD anchors persisted in DragonflyDB).
  4. Live `mt5.order_check()` validation across all 8 approved pairs (`0.01` lot).
  5. Full `H4OrderManager` Option A Pass-Through cycle (`dry_run_orders=True`)
     syncing latest closed H4/D1 bars and writing QuestDB `feature_audit_log`.
  6. Live-vs-Batch `ParityChecker` asserting `max_abs_diff <= 1e-9` across all 9
     stationary features and Layer 1 signals.

Optional `--execute-demo-roundtrip` flag performs a single explicit 0.01-lot
EURUSD open -> SL modify -> close cycle ONLY when explicitly requested.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.parity_checker import ParityChecker
from config.settings import settings
from core.clock import H4ServerClock
from core.mt5_connector import MT5Connector
from core.storage import StorageClient, TradeExecutionRecord
from data.cost_model import CostModel
from execution.mt5_adapter import MT5ExecutionAdapter, MT5WatchdogAdapter
from execution.order_manager import H4OrderManager
from execution.telegram_notifier import TelegramNotifier
from risk.tiered_watchdog import TieredWatchdog
from strategy.primary_trend import PrimaryStrategyConfig, PrimaryTrendStrategy


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Phase 3 live demo pipeline & feature parity.")
    parser.add_argument(
        "--execute-demo-roundtrip",
        action="store_true",
        help="Explicitly execute a live 0.01 lot EURUSD open->SL modify->close roundtrip on Demo.",
    )
    args = parser.parse_args()

    storage = StorageClient()
    connector = MT5Connector()

    print("=== FASE 3: VERIFICACAO LIVE DEMO & AUDITORIA DE PARIDADE ===")
    qdb_ok = storage.is_tcp_reachable(storage.questdb_host, storage.questdb_pg_port)
    df_ok = storage.ping_dragonfly()
    print(f"[Infra] QuestDB Health: {'OK' if qdb_ok else 'FAIL'} | DragonflyDB Health: {'OK' if df_ok else 'FAIL'}")

    if not qdb_ok or not df_ok:
        print("ABORT: Infraestrutura QuestDB/DragonflyDB indisponivel.")
        return 1

    if not connector.connect():
        print("ABORT: Falha ao ligar ao MetaTrader 5.")
        return 1

    try:
        account = connector.get_account_snapshot()
        if account is None:
            print("ABORT: Account snapshot indisponivel.")
            return 1

        if (not account.is_demo) or account.balance_eur >= 50000.0:
            print(
                "ABORT [SAFETY GUARD]: A conta ligada no MT5 nao corresponde a conta Demo de 1.000 EUR "
                "(conta ignorada por regra de seguranca). Troque para a conta PepperstoneUK-Demo 1.000 EUR."
            )
            return 1

        print(
            f"[MT5 Account] Login={account.login} | Server={account.server} | "
            f"Demo={account.is_demo} | Currency={account.currency} | "
            f"Balance={account.balance_eur:.2f} EUR | Equity={account.equity_eur:.2f} EUR | Lev=1:{account.leverage}"
        )

        # 1. Server Clock Synchronization
        clock = H4ServerClock(offset_seconds=settings.h4_close_offset_seconds)
        real_utc = datetime.now(timezone.utc)
        srv_now = clock.estimate_server_now(connector, real_utc_now=real_utc)
        closed_h4 = clock.latest_closed_h4_bar_open(srv_now)
        closed_d1 = clock.latest_closed_d1_bar_open(srv_now)
        next_wakeup = clock.next_h4_close_wakeup_server(srv_now, skip_weekend=True)
        secs_wait = clock.seconds_until_next_h4_wakeup(srv_now, skip_weekend=True)

        offset_h = (clock.server_minus_utc_seconds or 0.0) / 3600.0
        print(
            f"[Clock] RealUTC={real_utc.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"MT5_Server={srv_now.strftime('%Y-%m-%d %H:%M:%S')} (Offset={offset_h:+.1f}h)"
        )
        print(
            f"[Clock] LatestClosedH4={closed_h4.strftime('%Y-%m-%d %H:%M')} | "
            f"LatestClosedD1={closed_d1.strftime('%Y-%m-%d %H:%M')} | "
            f"NextWakeup={next_wakeup.strftime('%Y-%m-%d %H:%M:%S')} (in {secs_wait:.0f}s)"
        )

        # 2. TieredWatchdog Audit
        wd_adapter = MT5WatchdogAdapter(
            connector=connector,
            storage=storage,
            clock=clock,
            magic_filter=settings.mt5_magic_number,
            baseline_capital_eur=settings.initial_capital_eur,
            dry_run_orders=True,
        )
        watchdog = TieredWatchdog(wd_adapter)
        wd_res = watchdog.run_single_audit_cycle()
        print(
            f"[Watchdog] Status={wd_res.status} | DailyDD={wd_res.daily_drawdown_pct*100:.2f}% | "
            f"TotalDD={wd_res.total_drawdown_pct*100:.2f}% | NakedIntervened={wd_res.naked_tickets_intervened}"
        )

        # 3. Live Contract Specs & `mt5.order_check()` Across 8 Approved Pairs
        print("\n[MT5 order_check() Live Validation (0.01 lot BUY, atomic SL/TP)]")
        live_specs = {}
        order_checks_ok = True
        for sym in settings.approved_symbols:
            spec = connector.get_symbol_spec(sym)
            if spec is None:
                print(f"  - {sym}: FAIL (Spec unavailable)")
                order_checks_ok = False
                continue
            live_specs[sym] = spec
            sl_dist = 80.0 * spec.pip_size
            tp_dist = 140.0 * spec.pip_size
            sl_px = round(spec.ask - sl_dist, spec.digits)
            tp_px = round(spec.ask + tp_dist, spec.digits)
            chk = connector.check_market_order(
                symbol=sym,
                side=1,
                volume=settings.fixed_pass_through_lots,
                sl_price=sl_px,
                tp_price=tp_px,
            )
            fill_mode = connector.resolve_filling_mode(sym)
            print(
                f"  - {sym:7s}: OK={chk.success} | Ret={chk.retcode} | Spread={spec.spread_points:2d}pts | "
                f"Margin0.01={spec.margin_initial_001_eur:.2f} EUR | FillingMode={fill_mode} | {chk.comment}"
            )
            if not chk.success:
                order_checks_ok = False

        # 4. Run Full H4OrderManager Dry-Run Cycle & Persist FeatureAuditLog
        notifier = TelegramNotifier(enabled=False)
        exec_adapter = MT5ExecutionAdapter(connector, dry_run_orders=True)
        strategy = PrimaryTrendStrategy(PrimaryStrategyConfig())
        order_manager = H4OrderManager(
            adapter=exec_adapter,
            storage=storage,
            clock=clock,
            strategy=strategy,
            cost_model=CostModel(),
            notifier=notifier,
            use_meta_labeling=False,  # Option A Pass-Through
            fixed_lots=settings.fixed_pass_through_lots,
            sync_to_questdb=True,
            enforce_bar_alignment_wait=True,
        )
        cycle = order_manager.run_h4_cycle(server_now=srv_now)
        print(
            f"\n[OrderManager Cycle] Mode={cycle.mode} | ClosedH4={cycle.latest_closed_bar_open} | "
            f"Signals={cycle.signals_detected} | Approved={cycle.orders_approved} | "
            f"AuditRows={len(cycle.audit_records)} | Errors={cycle.errors}"
        )
        for rec in cycle.audit_records:
            print(
                f"  * {rec.symbol:7s} | side={rec.side:+d} | trig={rec.trigger_type:12s} | "
                f"ATR={rec.atr14_h4:.5f} | cs={rec.cs_spread:+.3f} | ker={rec.ker_d1:.3f} | "
                f"approved={rec.approved} | lots={rec.allocated_lots:.2f} | {rec.rejection_reason}"
            )

        # 5. Wait briefly for QuestDB asynchronous WAL apply, then run ParityChecker
        wal_deadline = time.monotonic() + 5.0
        while time.monotonic() < wal_deadline:
            committed = storage.load_feature_audit_records(
                start_ts=cycle.latest_closed_bar_open,
                end_ts=cycle.latest_closed_bar_open,
                limit=32,
            )
            if len(committed) >= len(cycle.audit_records):
                break
            time.sleep(0.25)

        checker = ParityChecker(strategy=strategy, cost_model=CostModel(), tolerance=1e-9)
        parity = checker.verify_from_storage(
            storage=storage,
            symbols=settings.approved_symbols,
            start_ts=cycle.latest_closed_bar_open,
            end_ts=cycle.latest_closed_bar_open,
            specs_by_symbol=live_specs,
            reference_lots=settings.fixed_pass_through_lots,
        )
        print(f"\n[ParityChecker] {parity.summary}")
        for col, diff in parity.feature_max_diffs.items():
            print(f"  - {col:26s}: max_abs_diff = {diff:.3e}")

        # 6. Optional explicit live demo roundtrip test (only when flag passed)
        roundtrip_ok = True
        if args.execute_demo_roundtrip:
            print("\n[Optional Demo Roundtrip] Executing explicit 0.01 lot EURUSD open->modify->close...")
            sp = connector.get_symbol_spec("EURUSD")
            if sp is None:
                roundtrip_ok = False
                print("  EURUSD spec unavailable.")
            else:
                sl_0 = round(sp.ask - 80.0 * sp.pip_size, sp.digits)
                tp_0 = round(sp.ask + 140.0 * sp.pip_size, sp.digits)
                op = connector.send_market_order("EURUSD", 1, 0.01, sl_0, tp_0, comment="P3_TEST")
                print(f"  OPEN: {op}")
                if op.success and op.ticket > 0:
                    mod = connector.modify_position_sl_tp(
                        op.ticket, "EURUSD", round(sl_0 + 5.0 * sp.point, sp.digits), tp_0
                    )
                    print(f"  MODIFY: {mod}")
                    cl = connector.close_position(op.ticket, "EURUSD", 0.01, 1, comment="P3_TEST_END")
                    print(f"  CLOSE: {cl}")
                    storage.ensure_trade_executions_table()
                    storage.write_trade_execution_record(
                        TradeExecutionRecord(
                            ts=datetime.now(timezone.utc),
                            symbol="EURUSD",
                            action="DEMO_ROUNDTRIP_TEST",
                            ticket=op.ticket,
                            side=1,
                            lots=0.01,
                            entry_price=op.price,
                            sl_price=sl_0,
                            tp_price=tp_0,
                            initial_sl_dist=abs(op.price - sl_0),
                            exit_price=cl.price,
                            reason="EXPLICIT_CLI_DEMO_TEST",
                        )
                    )
                    roundtrip_ok = op.success and mod.success and cl.success
                else:
                    roundtrip_ok = False

        # Gate Evaluation
        gates = {
            "G1_MT5_Demo_1K_Safety": account.is_demo and account.balance_eur < 50000.0,
            "G2_Watchdog_Healthy": wd_res.status == "OK",
            "G3_All_8_OrderChecks_OK": order_checks_ok and len(live_specs) == len(settings.approved_symbols),
            "G4_OrderManager_Zero_Errors": (not cycle.errors) and len(cycle.audit_records) == len(settings.approved_symbols),
            "G5_Feature_Signal_Parity_1e9": parity.passed and parity.max_abs_diff <= 1e-9,
        }
        if args.execute_demo_roundtrip:
            gates["G6_Demo_Roundtrip"] = roundtrip_ok

        passed_cnt = sum(1 for v in gates.values() if v)
        total_cnt = len(gates)
        verdict = "GO" if passed_cnt == total_cnt else "NO-GO"
        print(f"\n=== PORTAO FASE 3: {verdict} ({passed_cnt}/{total_cnt} PASS) ===")
        for g_name, g_ok in gates.items():
            print(f"  - {g_name:32s}: {'PASS' if g_ok else 'FAIL'}")
        if not args.execute_demo_roundtrip:
            print("  - G6_Demo_Roundtrip               : SKIPPED (dry-run; optional --execute-demo-roundtrip)")

        return 0 if passed_cnt == total_cnt else 1
    finally:
        connector.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
