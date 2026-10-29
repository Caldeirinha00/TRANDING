import logging
from typing import List, Dict, Any
from core.config import CONFIG

logger = logging.getLogger(__name__)


class RiskManager:
    """Pre-trade and runtime risk evaluation."""

    def __init__(self):
        self.max_daily_dd = CONFIG.MAX_DAILY_DRAWDOWN_PCT
        self.max_total_dd = CONFIG.MAX_TOTAL_DRAWDOWN_PCT
        self.max_positions = CONFIG.MAX_CONCURRENT_POSITIONS
        self.max_single_loss = CONFIG.MAX_SINGLE_TRADE_LOSS_PCT
        self.max_trades_day = CONFIG.MAX_TRADES_PER_DAY

    async def evaluate_account_risk(self, account_info: Dict[str, Any]) -> bool:
        """Returns True if account is within safe limits, False if a limit is breached."""
        from core.redis_broadcaster import broadcaster
        from datetime import datetime, timezone

        balance = account_info.get("balance", 0)
        equity = account_info.get("equity", 0)

        # Regra de Ouro: Ignora completamente conta de 1.000.000 EUR
        if balance > 50000:
            return True

        if balance <= 0:
            logger.warning("Account balance is zero or negative (possibly disconnected). Skipping drawdown check.")
            return True

        client = None
        login = account_info.get("login", CONFIG.MT5_LOGIN)
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        try:
            client = await broadcaster._get_client()
        except Exception as e:
            logger.error(f"Failed to connect to Redis client in RiskManager: {e}")

        # 1. Daily Drawdown Evaluation (Anchored at start of day balance)
        start_of_day_balance = balance
        if client:
            try:
                daily_key = f"system:daily_start_balance:{login}:{today_str}"
                saved_balance = await client.get(daily_key)
                if saved_balance is None:
                    await client.set(daily_key, balance, ex=86400 * 2)
                    start_of_day_balance = balance
                    logger.info(f"New session day for account {login}! Saved start_of_day_balance: {start_of_day_balance:.2f}")
                else:
                    start_of_day_balance = float(saved_balance)
            except Exception as e:
                logger.error(f"Failed to fetch/save daily start balance in Redis: {e}")
                start_of_day_balance = balance

        daily_drawdown_pct = ((start_of_day_balance - equity) / start_of_day_balance) * 100.0
        if 0 < self.max_daily_dd < 100.0:
            if daily_drawdown_pct > self.max_daily_dd:
                logger.critical(
                    f"DAILY DRAWDOWN BREACH! Current: {daily_drawdown_pct:.2f}% > Limit: {self.max_daily_dd:.2f}% "
                    f"(Start: {start_of_day_balance:.2f}, Equity: {equity:.2f})"
                )
                return False

        # 2. Total Drawdown Evaluation (Anchored at High-Water Mark)
        # Protects accumulated portfolio wealth across sessions against multi-day drawdowns
        base_hwm = 1000.0 if balance <= 1000.0 else balance
        hwm_balance = max(base_hwm, balance, equity)

        if client:
            try:
                hwm_key = f"system:high_water_mark:{login}"
                cached_hwm = await client.get(hwm_key)
                if cached_hwm is None:
                    await client.set(hwm_key, str(hwm_balance))
                    logger.info(f"Initialized High-Water Mark for account {login}: {hwm_balance:.2f}")
                else:
                    stored_hwm = float(cached_hwm)
                    hwm_balance = max(stored_hwm, balance, equity)
                    if hwm_balance > stored_hwm:
                        await client.set(hwm_key, str(hwm_balance))
                        logger.info(f"Updated High-Water Mark for account {login}: {hwm_balance:.2f}")
            except Exception as e:
                logger.error(f"Failed to fetch/update HWM in Redis: {e}")

        total_drawdown_pct = ((hwm_balance - equity) / hwm_balance) * 100.0
        if 0 < self.max_total_dd < 100.0:
            if total_drawdown_pct > self.max_total_dd:
                logger.critical(
                    f"TOTAL DRAWDOWN BREACH (HWM)! Current: {total_drawdown_pct:.2f}% > Limit: {self.max_total_dd:.2f}% "
                    f"(HWM: {hwm_balance:.2f}, Equity: {equity:.2f}, Floor: {hwm_balance * (1 - self.max_total_dd/100):.2f})"
                )
                return False

        return True

    def evaluate_positions_risk(self, positions: List[Dict[str, Any]]) -> bool:
        """Returns True if open positions are within safe limits."""
        num_positions = len(positions)

        if num_positions >= self.max_positions:
            logger.critical(f"MAX POSITIONS! Current: {num_positions} >= Limit: {self.max_positions}")
            return False

        for pos in positions:
            profit = pos.get("profit", 0)
            if profit < -100:
                logger.warning(f"Large loss on position {pos.get('ticket', '?')}: {profit}")

        return True
