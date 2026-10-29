import time
import logging
from typing import Optional
import MetaTrader5 as mt5
import redis
from core.config import CONFIG

logger = logging.getLogger("TieredWatchdog")


class TieredRiskWatchdog:
    def __init__(
        self,
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
        max_daily_dd_pct: float = 0.0285,  # Dispara aos 2.85% para garantir teto rígido de 3.0%
        max_total_dd_pct: float = 0.1950,  # Dispara aos 19.5% para garantir teto rígido de 20.0%
        initial_capital_eur: float = 1000.0,
    ) -> None:
        host = redis_host or CONFIG.DRAGONFLY_HOST
        port = redis_port or CONFIG.DRAGONFLY_PORT
        self.r = redis.Redis(host=host, port=port, decode_responses=True)
        self.max_daily_dd_pct = max_daily_dd_pct
        self.max_total_dd_pct = max_total_dd_pct
        self.initial_capital_eur = initial_capital_eur

    def verify_and_enforce(self) -> str:
        if not mt5.initialize(path=CONFIG.MT5_PATH):
            if not mt5.initialize():
                logger.critical("MT5 Terminal inacessível pelo Watchdog!")
                return "TERMINAL_DOWN"

        acc = mt5.account_info()
        if acc is None:
            return "ACCOUNT_INFO_FAIL"

        # Regra Institucional: Ignora qualquer conta de 1M EUR e restringe a 1.000 EUR base
        if acc.balance < 50000.0:
            equity = float(acc.equity)
            balance = float(acc.balance)
        else:
            equity = 1000.0
            balance = 1000.0

        day_start_balance = float(self.r.get("account:day_start_balance") or balance)

        daily_dd = max(0.0, (day_start_balance - equity) / max(1.0, day_start_balance))
        total_dd = max(0.0, (self.initial_capital_eur - equity) / self.initial_capital_eur)

        # NÍVEL RED: Violação de Risco Financeiro -> Liquidação Imediata
        if daily_dd >= self.max_daily_dd_pct or total_dd >= self.max_total_dd_pct:
            logger.critical(
                f"[RED ALERT] Violação de Drawdown (Diário={daily_dd:.2%}, Total={total_dd:.2%}). "
                "A liquidar todas as posições!"
            )
            self._liquidate_all_positions(reason="DRAWDOWN_BREACH")
            self.r.set("system:lock", "RED_KILL_SWITCH")
            return "RED"

        # Auditoria de segurança: Garantir que NENHUMA posição aberta está sem Stop Loss no servidor
        positions = mt5.positions_get() or []
        for pos in positions:
            if pos.sl <= 0.0:
                logger.error(f"Posição {pos.ticket} detetada SEM STOP LOSS no servidor! A aplicar SL de emergência.")
                self._inject_emergency_sl(pos)

        # Verificação da idade do Heartbeat do Orchestrator Python
        last_hb = float(self.r.get("orchestrator:heartbeat_ts") or 0.0)
        silence_sec = time.time() - last_hb

        if silence_sec <= 10.0:
            self.r.set("system:state", "GREEN")
            return "GREEN"
        elif silence_sec <= 90.0:
            logger.warning(f"[YELLOW] Orchestrator silencioso há {silence_sec:.1f}s. Posições mantidas sob proteção MQL5.")
            self.r.set("system:state", "YELLOW")
            return "YELLOW"
        else:
            logger.error(f"[ORANGE] Orchestrator offline há {silence_sec:.1f}s. EA em modo AUTONOMOUS_DEFENSIVE.")
            self.r.set("system:state", "ORANGE")
            return "ORANGE"

    def _inject_emergency_sl(self, pos) -> None:
        sym_info = mt5.symbol_info(pos.symbol)
        if not sym_info:
            return
        # Aplica SL de emergência a 15 pips de distância se estiver desprotegida
        emergency_dist = 150 * sym_info.point
        sl_price = (
            round(pos.price_open - emergency_dist, sym_info.digits)
            if pos.type == mt5.POSITION_TYPE_BUY
            else round(pos.price_open + emergency_dist, sym_info.digits)
        )
        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": pos.ticket,
            "symbol": pos.symbol,
            "sl": sl_price,
            "tp": pos.tp,
        }
        mt5.order_send(req)

    def _liquidate_all_positions(self, reason: str) -> None:
        positions = mt5.positions_get() or []
        for pos in positions:
            tick = mt5.symbol_info_tick(pos.symbol)
            if not tick:
                continue
            close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": pos.ticket,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": close_type,
                "price": price,
                "deviation": 20,
                "magic": pos.magic,
                "comment": f"WD_{reason}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            mt5.order_send(req)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] TieredWatchdog: %(message)s",
    )
    watchdog = TieredRiskWatchdog()
    logger.info("TieredRiskWatchdog iniciado (monitorização ativa a cada 2s).")
    try:
        while True:
            state = watchdog.verify_and_enforce()
            time.sleep(2.0)
    except KeyboardInterrupt:
        logger.info("TieredRiskWatchdog encerrado.")

