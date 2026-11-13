import time
from dataclasses import dataclass
from typing import List, Protocol
import structlog
from config.settings import settings

logger = structlog.get_logger(__name__)


@dataclass
class LivePositionSnapshot:
    ticket: int
    symbol: str
    side: int
    volume: float
    price_open: float
    sl: float
    tp: float
    profit_eur: float


class BrokerWatchdogAdapter(Protocol):
    """Protocolo desacoplado para permitir auditoria real no MT5 e injeção determinística em testes."""
    def get_account_equity_eur(self) -> float: ...
    def get_daily_starting_equity_eur(self) -> float: ...
    def get_high_water_mark_equity_eur(self) -> float: ...
    def get_open_positions(self) -> List[LivePositionSnapshot]: ...
    def emergency_close_position(self, ticket: int, reason: str) -> bool: ...
    def emergency_close_all(self, reason: str) -> int: ...


@dataclass(frozen=True)
class WatchdogAuditResult:
    status: str  # "OK", "NAKED_POSITION_CLOSED", "DAILY_KILL_SWITCH", "TOTAL_KILL_SWITCH"
    daily_drawdown_pct: float
    total_drawdown_pct: float
    naked_tickets_intervened: List[int]
    all_positions_liquidated: bool


class TieredWatchdog:
    """
    Daemon autónomo 24/5 de proteção patrimonial (ciclo <= 2 segundos):
    1. Kill Switch Diário Preventivo a 2,85% (limite regulatório interno: 3,00%).
    2. Kill Switch Total Preventivo a 19,50% (limite regulatório interno: 20,00%).
    3. Deteção e intervenção imediata (< 5 segundos) sobre qualquer posição aberta sem Stop Loss físico no servidor.
    """

    def __init__(self, adapter: BrokerWatchdogAdapter) -> None:
        self.adapter = adapter
        self.daily_kill_pct = settings.daily_dd_kill_switch_pct
        self.total_kill_pct = settings.total_dd_kill_switch_pct

    def run_single_audit_cycle(self) -> WatchdogAuditResult:
        equity = self.adapter.get_account_equity_eur()
        day_start_equity = max(self.adapter.get_daily_starting_equity_eur(), 1e-6)
        hwm_equity = max(self.adapter.get_high_water_mark_equity_eur(), settings.initial_capital_eur)

        daily_dd_pct = max(0.0, (day_start_equity - equity) / day_start_equity)
        total_dd_pct = max(0.0, (hwm_equity - equity) / hwm_equity)

        # 1. Verificação de Kill Switch Total (19,50%)
        if total_dd_pct >= self.total_kill_pct:
            logger.critical(
                "watchdog_total_kill_switch_triggered",
                equity=equity,
                hwm=hwm_equity,
                total_dd_pct=round(total_dd_pct * 100, 2),
            )
            self.adapter.emergency_close_all(reason=f"TOTAL_DD_KILL_SWITCH_{total_dd_pct*100:.2f}PCT")
            return WatchdogAuditResult(
                status="TOTAL_KILL_SWITCH",
                daily_drawdown_pct=round(daily_dd_pct, 4),
                total_drawdown_pct=round(total_dd_pct, 4),
                naked_tickets_intervened=[],
                all_positions_liquidated=True,
            )

        # 2. Verificação de Kill Switch Diário (2,85%)
        if daily_dd_pct >= self.daily_kill_pct:
            logger.critical(
                "watchdog_daily_kill_switch_triggered",
                equity=equity,
                day_start=day_start_equity,
                daily_dd_pct=round(daily_dd_pct * 100, 2),
            )
            self.adapter.emergency_close_all(reason=f"DAILY_DD_KILL_SWITCH_{daily_dd_pct*100:.2f}PCT")
            return WatchdogAuditResult(
                status="DAILY_KILL_SWITCH",
                daily_drawdown_pct=round(daily_dd_pct, 4),
                total_drawdown_pct=round(total_dd_pct, 4),
                naked_tickets_intervened=[],
                all_positions_liquidated=True,
            )

        # 3. Verificação de Posições sem Stop Loss Físico no Servidor (< 5s SLA)
        naked_intervened: List[int] = []
        for pos in self.adapter.get_open_positions():
            if pos.sl is None or pos.sl <= 0.0:
                logger.error(
                    "watchdog_naked_position_detected",
                    ticket=pos.ticket,
                    symbol=pos.symbol,
                    volume=pos.volume,
                )
                self.adapter.emergency_close_position(pos.ticket, reason="NAKED_POSITION_NO_SERVER_SL")
                naked_intervened.append(pos.ticket)

        status = "NAKED_POSITION_CLOSED" if naked_intervened else "OK"
        return WatchdogAuditResult(
            status=status,
            daily_drawdown_pct=round(daily_dd_pct, 4),
            total_drawdown_pct=round(total_dd_pct, 4),
            naked_tickets_intervened=naked_intervened,
            all_positions_liquidated=False,
        )

    def run_forever(self, poll_interval_sec: float = 2.0) -> None:
        logger.info("tiered_watchdog_started", poll_interval_sec=poll_interval_sec)
        while True:
            self.run_single_audit_cycle()
            time.sleep(poll_interval_sec)
