"""
MetaTrader 5 Execution & Watchdog Adapters (Fase 3).

Provides:
1. `BrokerExecutionAdapter` Protocol for deterministic unit testing and live
   PepperstoneUK-Demo execution isolation.
2. `MT5ExecutionAdapter` wrapping `MT5Connector` with optional `dry_run_orders`
   (running live `mt5.order_check` instead of `mt5.order_send`).
3. `MT5WatchdogAdapter` implementing `risk.tiered_watchdog.BrokerWatchdogAdapter`
   with DragonflyDB-backed persistent daily equity & HWM equity tracking keyed
   by the MT5 broker server calendar date.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Protocol

import pandas as pd

from config.settings import settings
from core.clock import H4ServerClock
from core.mt5_connector import (
    BrokerAccountSnapshot,
    BrokerOpenPosition,
    MT5Connector,
    OrderExecutionResult,
    SymbolContractSpec,
)
from core.storage import StorageClient, TradeExecutionRecord
from risk.tiered_watchdog import BrokerWatchdogAdapter, LivePositionSnapshot

logger = logging.getLogger(__name__)


def build_fallback_contract_spec(
    symbol: str,
    spread_points: Optional[int] = None,
) -> SymbolContractSpec:
    """
    Deterministic contract spec fallback strictly for offline simulation/parity
    when live MT5 and DragonflyDB snapshots are unavailable.
    """
    sym = symbol.upper()
    is_jpy = "JPY" in sym
    digits = 3 if is_jpy else 5
    point = 10.0 ** (-digits)
    pip_size = point * 10.0
    # Approximate EUR tick value per 1.00 lot point (10x pip_val_1lot / 10)
    if sym.endswith("EUR"):
        tick_val_1lot = 1.0
    elif sym.endswith("JPY"):
        tick_val_1lot = 0.62
    elif sym.endswith("CHF"):
        tick_val_1lot = 1.05
    elif sym.endswith("CAD"):
        tick_val_1lot = 0.67
    else:
        tick_val_1lot = 0.92
    esma_lev = 20.0 if (is_jpy and not sym.startswith("USD")) or sym[:3] in {"AUD", "NZD"} else 30.0
    base_px = 152.0 if is_jpy else 1.08
    eff_spread = int(
        settings.benchmark_spread_points.get(sym, 12)
        if spread_points is None
        else spread_points
    )
    return SymbolContractSpec(
        symbol=sym,
        currency_base=sym[:3],
        currency_profit=sym[3:6],
        currency_margin=sym[:3],
        digits=digits,
        point=point,
        pip_size=pip_size,
        trade_contract_size=100_000.0,
        trade_tick_value=tick_val_1lot,
        trade_tick_value_profit=tick_val_1lot,
        trade_tick_value_loss=tick_val_1lot,
        trade_tick_size=point,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        spread_points=eff_spread,
        swap_mode=1,
        swap_long=-2.0,
        swap_short=0.5,
        swap_rollover3days=3,
        stops_level_points=0,
        margin_initial_001_eur=round(1000.0 / esma_lev, 2),
        bid=base_px,
        ask=round(base_px + eff_spread * point, digits),
    )


class BrokerExecutionAdapter(Protocol):
    """Protocol consumed by `H4OrderManager` for live or simulated brokerage."""

    def get_server_time_utc(self) -> Optional[datetime]: ...

    def get_account_snapshot(self) -> Optional[BrokerAccountSnapshot]: ...

    def get_open_positions(
        self, magic: Optional[int] = None
    ) -> List[BrokerOpenPosition]: ...

    def get_symbol_spec(self, symbol: str) -> Optional[SymbolContractSpec]: ...

    def fetch_closed_bars(
        self,
        symbol: str,
        timeframe_str: str,
        count: int = 500,
        server_now: Optional[datetime] = None,
    ) -> pd.DataFrame: ...

    def check_market_order(
        self,
        symbol: str,
        side: int,
        volume: float,
        sl_price: float,
        tp_price: float,
        deviation_points: Optional[int] = None,
        magic: Optional[int] = None,
        comment: str = "AI_TRADING_CHECK",
    ) -> OrderExecutionResult: ...

    def send_market_order(
        self,
        symbol: str,
        side: int,
        volume: float,
        sl_price: float,
        tp_price: float,
        deviation_points: Optional[int] = None,
        magic: Optional[int] = None,
        comment: str = "AI_TRADING_OP_A",
    ) -> OrderExecutionResult: ...

    def modify_position_sl_tp(
        self,
        ticket: int,
        symbol: str,
        new_sl: float,
        new_tp: float,
        magic: Optional[int] = None,
    ) -> OrderExecutionResult: ...

    def close_position(
        self,
        ticket: int,
        symbol: str,
        volume: float,
        side: int,
        deviation_points: Optional[int] = None,
        magic: Optional[int] = None,
        comment: str = "AI_TRADING_TIME_STOP",
    ) -> OrderExecutionResult: ...


class MT5ExecutionAdapter:
    """
    Production BrokerExecutionAdapter backed by MT5Connector.
    When `dry_run_orders=True`, live market reads and `mt5.order_check` still run
    against PepperstoneUK-Demo, while state-mutating calls return synthetic
    non-executed results.
    """

    def __init__(
        self,
        connector: MT5Connector,
        dry_run_orders: bool = False,
        magic_number: Optional[int] = None,
    ) -> None:
        self.connector = connector
        self.dry_run_orders = bool(dry_run_orders)
        self.magic_number = (
            settings.mt5_magic_number if magic_number is None else int(magic_number)
        )
        self._dry_ticket_seq = 900_000_000

    def get_server_time_utc(self) -> Optional[datetime]:
        return self.connector.get_server_time_utc()

    def get_account_snapshot(self) -> Optional[BrokerAccountSnapshot]:
        return self.connector.get_account_snapshot()

    def get_open_positions(
        self, magic: Optional[int] = None
    ) -> List[BrokerOpenPosition]:
        mgc = self.magic_number if magic is None else magic
        return self.connector.get_open_positions(magic=mgc)

    def get_symbol_spec(self, symbol: str) -> Optional[SymbolContractSpec]:
        return self.connector.get_symbol_spec(symbol)

    def fetch_closed_bars(
        self,
        symbol: str,
        timeframe_str: str,
        count: int = 500,
        server_now: Optional[datetime] = None,
    ) -> pd.DataFrame:
        return self.connector.fetch_closed_bars(
            symbol=symbol,
            timeframe_str=timeframe_str,
            count=count,
            server_now=server_now,
        )

    def check_market_order(
        self,
        symbol: str,
        side: int,
        volume: float,
        sl_price: float,
        tp_price: float,
        deviation_points: Optional[int] = None,
        magic: Optional[int] = None,
        comment: str = "AI_TRADING_CHECK",
    ) -> OrderExecutionResult:
        return self.connector.check_market_order(
            symbol=symbol,
            side=side,
            volume=volume,
            sl_price=sl_price,
            tp_price=tp_price,
            deviation_points=deviation_points,
            magic=self.magic_number if magic is None else magic,
            comment=comment,
        )

    def send_market_order(
        self,
        symbol: str,
        side: int,
        volume: float,
        sl_price: float,
        tp_price: float,
        deviation_points: Optional[int] = None,
        magic: Optional[int] = None,
        comment: str = "AI_TRADING_OP_A",
    ) -> OrderExecutionResult:
        mgc = self.magic_number if magic is None else magic
        if self.dry_run_orders:
            chk = self.check_market_order(
                symbol=symbol,
                side=side,
                volume=volume,
                sl_price=sl_price,
                tp_price=tp_price,
                deviation_points=deviation_points,
                magic=mgc,
                comment=comment,
            )
            if not chk.success:
                return OrderExecutionResult(
                    success=False,
                    action="DRY_RUN_OPEN",
                    symbol=symbol,
                    retcode=chk.retcode,
                    comment=f"DRY_RUN_CHECK_FAIL:{chk.comment}",
                )
            self._dry_ticket_seq += 1
            return OrderExecutionResult(
                success=True,
                action="DRY_RUN_OPEN",
                symbol=symbol,
                retcode=chk.retcode,
                comment="DRY_RUN_ORDER_CHECK_OK",
                ticket=self._dry_ticket_seq,
                order_id=self._dry_ticket_seq,
                price=chk.price,
                volume=chk.volume,
            )

        return self.connector.send_market_order(
            symbol=symbol,
            side=side,
            volume=volume,
            sl_price=sl_price,
            tp_price=tp_price,
            deviation_points=deviation_points,
            magic=mgc,
            comment=comment,
        )

    def modify_position_sl_tp(
        self,
        ticket: int,
        symbol: str,
        new_sl: float,
        new_tp: float,
        magic: Optional[int] = None,
    ) -> OrderExecutionResult:
        mgc = self.magic_number if magic is None else magic
        if self.dry_run_orders:
            return OrderExecutionResult(
                success=True,
                action="DRY_RUN_MODIFY_SLTP",
                symbol=symbol,
                retcode=0,
                comment="DRY_RUN_NOOP",
                ticket=ticket,
            )
        return self.connector.modify_position_sl_tp(
            ticket=ticket,
            symbol=symbol,
            new_sl=new_sl,
            new_tp=new_tp,
            magic=mgc,
        )

    def close_position(
        self,
        ticket: int,
        symbol: str,
        volume: float,
        side: int,
        deviation_points: Optional[int] = None,
        magic: Optional[int] = None,
        comment: str = "AI_TRADING_TIME_STOP",
    ) -> OrderExecutionResult:
        mgc = self.magic_number if magic is None else magic
        if self.dry_run_orders:
            return OrderExecutionResult(
                success=True,
                action="DRY_RUN_CLOSE",
                symbol=symbol,
                retcode=0,
                comment="DRY_RUN_NOOP",
                ticket=ticket,
                volume=volume,
            )
        return self.connector.close_position(
            ticket=ticket,
            symbol=symbol,
            volume=volume,
            side=side,
            deviation_points=deviation_points,
            magic=mgc,
            comment=comment,
        )


class MT5WatchdogAdapter(BrokerWatchdogAdapter):
    """
    BrokerWatchdogAdapter connected to MT5Connector + DragonflyDB state so daily
    DD (-2.85%) and total HWM DD (-19.50%) anchors survive process restarts.
    """

    EQUITY_STATE_KEY = "watchdog:equity_anchors"

    def __init__(
        self,
        connector: MT5Connector,
        storage: Optional[StorageClient] = None,
        clock: Optional[H4ServerClock] = None,
        magic_filter: Optional[int] = None,
        baseline_capital_eur: Optional[float] = None,
        dry_run_orders: bool = False,
    ) -> None:
        self.connector = connector
        self.storage = storage
        self.clock = clock or H4ServerClock(offset_seconds=settings.h4_close_offset_seconds)
        self.magic_filter = magic_filter
        self.baseline_capital_eur = (
            float(baseline_capital_eur) if baseline_capital_eur is not None else None
        )
        self.dry_run_orders = bool(dry_run_orders)

        self._last_known_equity: Optional[float] = None
        self._cached_state: Optional[Dict[str, object]] = None

    def _current_server_date_str(self) -> str:
        srv_now = self.clock.estimate_server_now(self.connector)
        return srv_now.date().isoformat()

    def _sync_equity_anchors(self, equity_eur: float) -> Dict[str, object]:
        srv_date = self._current_server_date_str()
        state = self._cached_state
        if state is None and self.storage is not None:
            loaded = self.storage.get_runtime_state(self.EQUITY_STATE_KEY)
            if isinstance(loaded, dict):
                state = dict(loaded)

        if state is None:
            hwm = equity_eur
            if self.baseline_capital_eur is not None:
                hwm = max(hwm, self.baseline_capital_eur)
            state = {
                "daily_date": srv_date,
                "daily_start_equity": equity_eur,
                "hwm_equity": hwm,
            }
            dirty = True
        else:
            dirty = False
            prev_hwm = float(state.get("hwm_equity", equity_eur))
            new_hwm = max(prev_hwm, equity_eur)
            if self.baseline_capital_eur is not None:
                new_hwm = max(new_hwm, self.baseline_capital_eur)
            if new_hwm != prev_hwm:
                state["hwm_equity"] = new_hwm
                dirty = True
            if state.get("daily_date") != srv_date:
                state["daily_date"] = srv_date
                state["daily_start_equity"] = equity_eur
                dirty = True

        self._cached_state = state
        if dirty and self.storage is not None:
            self.storage.set_runtime_state(self.EQUITY_STATE_KEY, state)
        return state

    def get_account_equity_eur(self) -> float:
        snap = self.connector.get_account_snapshot()
        if snap is None:
            if self._last_known_equity is not None:
                return self._last_known_equity
            raise RuntimeError("MT5 account snapshot unavailable for TieredWatchdog.")
        eq = float(snap.equity_eur)
        self._last_known_equity = eq
        self._sync_equity_anchors(eq)
        return eq

    def get_daily_starting_equity_eur(self) -> float:
        eq = self.get_account_equity_eur()
        state = self._sync_equity_anchors(eq)
        return float(state["daily_start_equity"])

    def get_high_water_mark_equity_eur(self) -> float:
        eq = self.get_account_equity_eur()
        state = self._sync_equity_anchors(eq)
        return float(state["hwm_equity"])

    def get_open_positions(self) -> List[LivePositionSnapshot]:
        positions = self.connector.get_open_positions(magic=self.magic_filter)
        return [
            LivePositionSnapshot(
                ticket=p.ticket,
                symbol=p.symbol,
                side=p.side,
                volume=p.volume,
                price_open=p.price_open,
                sl=p.sl,
                tp=p.tp,
                profit_eur=p.profit_eur,
            )
            for p in positions
        ]

    def emergency_close_position(self, ticket: int, reason: str) -> bool:
        positions = self.connector.get_open_positions(magic=self.magic_filter)
        target = next((p for p in positions if p.ticket == ticket), None)
        if target is None:
            return False
        if self.dry_run_orders:
            logger.warning(
                "Watchdog DRY_RUN emergency close skipped for ticket=%s (%s)",
                ticket,
                reason,
            )
            return True
        res = self.connector.close_position(
            ticket=target.ticket,
            symbol=target.symbol,
            volume=target.volume,
            side=target.side,
            magic=target.magic,
            comment=f"WD_{reason}"[:31],
        )
        if res.success and self.storage is not None:
            self.storage.ensure_trade_executions_table()
            self.storage.write_trade_execution_record(
                TradeExecutionRecord(
                    ts=datetime.now(timezone.utc),
                    symbol=target.symbol,
                    action="WATCHDOG_CLOSE",
                    ticket=target.ticket,
                    side=target.side,
                    lots=target.volume,
                    entry_price=target.price_open,
                    sl_price=target.sl,
                    tp_price=target.tp,
                    initial_sl_dist=abs(target.price_open - target.sl),
                    exit_price=res.price,
                    realized_pnl_eur=target.profit_eur,
                    swap_eur=target.swap_eur,
                    reason=reason,
                )
            )
        return res.success

    def emergency_close_all(self, reason: str) -> int:
        closed = 0
        for pos in self.connector.get_open_positions(magic=self.magic_filter):
            if self.emergency_close_position(pos.ticket, reason=reason):
                closed += 1
        return closed
