"""
Autonomous Live Trading Daemon (Fase 4: Burn-In 24/5 em Demo).

Coordinates the concurrent execution of:
  1. H4 Bar-Close Execution Engine (00:00:02, 04:00:02, 08:00:02, 12:00:02, 16:00:02, 20:00:02 broker server time)
     running Option A Pass-Through (H2_BB_TREND_DIP_8P at 0.01 lot fixed) with BE@+1.10R,
     Chandelier Trail@+1.50R, 36-bar TimeStop, and QuestDB feature audit logging.
  2. Sub-5-Second Tiered Watchdog background loop protecting against naked positions,
     daily DD kill switch (-2.85%), and total DD kill switch (-19.50%).
  3. Real-time Telegram alerting and daily summary reports.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import logging
import signal
import sys
from typing import Optional

import structlog

from config.settings import settings
from core.clock import H4ServerClock
from core.mt5_connector import MT5Connector
from core.storage import StorageClient
from data.cost_model import CostModel
from execution.mt5_adapter import MT5ExecutionAdapter, MT5WatchdogAdapter
from execution.order_manager import H4OrderManager
from execution.telegram_notifier import TelegramNotifier
from risk.currency_guard import CurrencyAndHeatGuard
from risk.tiered_watchdog import TieredWatchdog
from strategy.primary_trend import PrimaryStrategyConfig, PrimaryTrendStrategy

logger = structlog.get_logger(__name__)


class TradingDaemon:
    """
    Main asynchronous trading daemon orchestrating H4 cycles, watchdog, and monitoring.
    """

    def __init__(
        self,
        mode: str = "demo",
        max_bars: Optional[int] = None,
        poll_interval_watchdog: float = 2.0,
        sync_to_questdb: bool = True,
    ) -> None:
        self.mode = mode.lower()
        self.max_bars = max_bars
        self.poll_interval_watchdog = max(0.5, float(poll_interval_watchdog))
        self.sync_to_questdb = bool(sync_to_questdb)

        self.storage = StorageClient()
        self.connector = MT5Connector()
        self.clock = H4ServerClock(offset_seconds=settings.h4_close_offset_seconds)
        self.notifier = TelegramNotifier()
        self.strategy = PrimaryTrendStrategy(PrimaryStrategyConfig())
        self.cost_model = CostModel()
        self.currency_guard = CurrencyAndHeatGuard()

        self._shutdown_event = asyncio.Event()
        self._bars_processed = 0
        self._last_reported_server_date: Optional[str] = None

    def _validate_environment_preflight(self) -> bool:
        logger.info("preflight_check_starting", mode=self.mode)

        # 1. Database infrastructure
        qdb_ok = self.storage.is_tcp_reachable(self.storage.questdb_host, self.storage.questdb_pg_port)
        df_ok = self.storage.ping_dragonfly()
        if not qdb_ok or not df_ok:
            logger.critical(
                "preflight_infrastructure_failed",
                questdb_healthy=qdb_ok,
                dragonfly_healthy=df_ok,
            )
            return False

        # 2. MT5 Broker Connection
        if not self.connector.connect():
            logger.critical("preflight_mt5_connection_failed")
            return False

        account = self.connector.get_account_snapshot()
        if account is None:
            logger.critical("preflight_account_snapshot_unavailable")
            return False

        # 3. ESMA & Demo Safety Rule
        if (not account.is_demo and not settings.allow_live_real_money) or account.balance_eur >= 50000.0:
            logger.critical(
                "preflight_safety_block_triggered",
                login=account.login,
                server=account.server,
                balance=account.balance_eur,
                is_demo=account.is_demo,
            )
            return False

        logger.info(
            "preflight_success",
            login=account.login,
            server=account.server,
            balance=account.balance_eur,
            equity=account.equity_eur,
            leverage=account.leverage,
        )
        return True

    async def _watchdog_loop(self, watchdog: TieredWatchdog) -> None:
        """Runs sub-5-second continuous watchdog audit loop."""
        logger.info("watchdog_loop_started", interval=self.poll_interval_watchdog)
        while not self._shutdown_event.is_set():
            try:
                self.connector.ensure_connected()
                audit = watchdog.run_single_audit_cycle()
                if audit.status != "OK":
                    logger.warning(
                        "watchdog_intervention_logged",
                        status=audit.status,
                        daily_dd=audit.daily_drawdown_pct,
                        total_dd=audit.total_drawdown_pct,
                        naked_intervened=audit.naked_tickets_intervened,
                    )
                    if audit.naked_tickets_intervened:
                        self.notifier.notify_risk_alert(
                            title="NAKED POSITION INTERVENED (<5s SLA)",
                            details=f"Tickets intervened: {audit.naked_tickets_intervened}",
                        )
                    if audit.all_positions_liquidated:
                        self.notifier.notify_risk_alert(
                            title="KILL SWITCH ACTIVATED",
                            details=f"Status: {audit.status} | Daily DD: {audit.daily_drawdown_pct*100:.2f}% | Total DD: {audit.total_drawdown_pct*100:.2f}%",
                        )
            except Exception as exc:
                logger.error("watchdog_loop_error", error=str(exc))

            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(), timeout=self.poll_interval_watchdog
                )
            except asyncio.TimeoutError:
                pass

        logger.info("watchdog_loop_stopped")

    async def _h4_execution_loop(self, order_manager: H4OrderManager) -> None:
        """Main H4 broker bar-close cycle execution loop."""
        logger.info("h4_execution_loop_started", max_bars=self.max_bars)

        while not self._shutdown_event.is_set():
            try:
                self.connector.ensure_connected()
                srv_now = self.clock.estimate_server_now(self.connector)
                next_wakeup = self.clock.next_h4_close_wakeup_server(srv_now, skip_weekend=True)
                secs_wait = self.clock.seconds_until_next_h4_wakeup(srv_now, skip_weekend=True)

                logger.info(
                    "h4_waiting_for_next_close",
                    server_now=srv_now.strftime("%Y-%m-%d %H:%M:%S"),
                    next_wakeup=next_wakeup.strftime("%Y-%m-%d %H:%M:%S"),
                    wait_seconds=round(secs_wait, 1),
                )

                # Asynchronously wait for the next bar close (wakes up in small chunks to handle shutdown)
                target_bar = await self.clock.wait_for_next_h4_close(
                    connector=self.connector,
                    skip_weekend=True,
                    max_sleep_chunk_seconds=15.0,
                    stop_check=self._shutdown_event.is_set,
                )

                if self._shutdown_event.is_set():
                    break

                # Bar close arrived -> Execute full H4 Cycle
                cycle_server_now = self.clock.estimate_server_now(self.connector)
                logger.info("h4_executing_cycle", closed_bar_open=target_bar.isoformat())

                cycle_res = await order_manager.run_h4_cycle_async(
                    server_now=cycle_server_now,
                    force_closed_bar_open=target_bar,
                )
                self._bars_processed += 1

                logger.info(
                    "h4_cycle_completed",
                    closed_bar=target_bar.strftime("%Y-%m-%d %H:%M"),
                    signals=cycle_res.signals_detected,
                    approved=cycle_res.orders_approved,
                    open_start=cycle_res.open_positions_count_start,
                    open_end=cycle_res.open_positions_count_end,
                    be_updates=len(cycle_res.positions_sl_updated),
                    time_stops=len(cycle_res.positions_closed_time_stop),
                    errors=cycle_res.errors,
                )

                # Send Telegram summary
                srv_after = self.clock.estimate_server_now(self.connector)
                next_target = self.clock.next_h4_close_wakeup_server(srv_after, skip_weekend=True)
                self.notifier.notify_h4_cycle_summary(
                    closed_bar_open=target_bar,
                    signals_detected=cycle_res.signals_detected,
                    approved_count=cycle_res.orders_approved,
                    open_positions_count=cycle_res.open_positions_count_end,
                    equity_eur=cycle_res.equity_eur,
                    next_wakeup_server=next_target,
                    mode=cycle_res.mode,
                )

                # Check server date rollover for daily report
                curr_date = target_bar.date().isoformat()
                if self._last_reported_server_date is None:
                    self._last_reported_server_date = curr_date
                elif curr_date != self._last_reported_server_date:
                    snap = self.connector.get_account_snapshot()
                    if snap is not None:
                        self.notifier.notify_daily_summary(
                            server_date=self._last_reported_server_date,
                            balance_eur=snap.balance_eur,
                            equity_eur=snap.equity_eur,
                            daily_pnl_eur=snap.equity_eur - snap.balance_eur,
                            open_positions_count=cycle_res.open_positions_count_end,
                            total_dd_pct=max(0.0, (1000.0 - snap.equity_eur) / 1000.0),
                        )
                    self._last_reported_server_date = curr_date

                if self.max_bars is not None and self._bars_processed >= self.max_bars:
                    logger.info("max_bars_reached", bars=self._bars_processed)
                    self._shutdown_event.set()
                    break

            except Exception as exc:
                logger.error("h4_cycle_exception", error=str(exc))
                await asyncio.sleep(2.0)

        logger.info("h4_execution_loop_stopped")

    async def run(self) -> int:
        """Starts and manages all concurrent daemon services until stopped."""
        if not self._validate_environment_preflight():
            return 1

        dry_run = self.mode != "demo"
        exec_adapter = MT5ExecutionAdapter(
            connector=self.connector,
            dry_run_orders=dry_run,
            magic_number=settings.mt5_magic_number,
        )
        wd_adapter = MT5WatchdogAdapter(
            connector=self.connector,
            storage=self.storage,
            clock=self.clock,
            magic_filter=settings.mt5_magic_number,
            baseline_capital_eur=settings.initial_capital_eur,
            dry_run_orders=dry_run,
        )
        watchdog = TieredWatchdog(wd_adapter)

        order_manager = H4OrderManager(
            adapter=exec_adapter,
            storage=self.storage,
            clock=self.clock,
            strategy=self.strategy,
            cost_model=self.cost_model,
            currency_guard=self.currency_guard,
            notifier=self.notifier,
            use_meta_labeling=False,  # Option A Pass-Through
            fixed_lots=settings.fixed_pass_through_lots,
            sync_to_questdb=self.sync_to_questdb,
            enforce_bar_alignment_wait=True,
        )

        logger.info(
            "daemon_starting_services",
            mode="LIVE_DEMO" if not dry_run else "DRY_RUN",
            watchdog_poll_sec=self.poll_interval_watchdog,
        )
        self.notifier.send_message_sync(
            f"🚀 <b>AI_TRADING Daemon Iniciado</b>\n"
            f"Modo: <code>{'LIVE DEMO' if not dry_run else 'DRY RUN'}</code>\n"
            f"Estratégia: <code>H2_BB_TREND_DIP_8P (Pass-Through 0.01 lot)</code>\n"
            f"Conta MT5: <code>PepperstoneUK-Demo (1.000 EUR)</code>"
        )

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._shutdown_event.set)
            except NotImplementedError:
                # Windows event loop fallback
                pass

        watchdog_task = asyncio.create_task(self._watchdog_loop(watchdog), name="watchdog_loop")
        h4_task = asyncio.create_task(self._h4_execution_loop(order_manager), name="h4_loop")

        try:
            await self._shutdown_event.wait()
        except KeyboardInterrupt:
            self._shutdown_event.set()

        logger.info("daemon_shutdown_initiated")
        # Await clean cancellation
        for task in (watchdog_task, h4_task):
            task.cancel()
        await asyncio.gather(watchdog_task, h4_task, return_exceptions=True)

        self.notifier.send_message_sync("🛑 <b>AI_TRADING Daemon Encerrado com Sucesso.</b>")
        self.connector.shutdown()
        logger.info("daemon_shutdown_complete")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Trading Autonomous Daemon (Fase 4).")
    parser.add_argument(
        "--mode",
        choices=["demo", "dry-run"],
        default="demo",
        help="Execution mode: 'demo' (live market orders on Demo) or 'dry-run' (order_check only).",
    )
    parser.add_argument(
        "--max-bars",
        type=int,
        default=None,
        help="Exit automatically after processing N closed H4 bars (useful for automated testing).",
    )
    parser.add_argument(
        "--poll-interval-watchdog",
        type=float,
        default=2.0,
        help="Sub-second or 2.0s interval for TieredWatchdog SLA audit.",
    )
    parser.add_argument(
        "--no-questdb-sync",
        action="store_true",
        help="Disable automatic OHLCV and feature audit log persistence to QuestDB.",
    )
    args = parser.parse_args()

    daemon = TradingDaemon(
        mode=args.mode,
        max_bars=args.max_bars,
        poll_interval_watchdog=args.poll_interval_watchdog,
        sync_to_questdb=not args.no_questdb_sync,
    )
    try:
        return asyncio.run(daemon.run())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
