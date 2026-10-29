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
        
        if balance <= 0:
            logger.warning("Account balance is zero or negative (possibly disconnected). Skipping drawdown check.")
            return True
            
        try:
            client = await broadcaster._get_client()
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            key = f"system:daily_start_balance:{today_str}"
            
            saved_balance = await client.get(key)
            if saved_balance is None:
                # First run of the day, save current balance
                await client.set(key, balance, ex=86400 * 2)  # Keep for 2 days
                start_of_day_balance = balance
                logger.info(f"New day! Saved start_of_day_balance: {start_of_day_balance}")
            else:
                start_of_day_balance = float(saved_balance)
        except Exception as e:
            logger.error(f"Failed to fetch start_of_day_balance from Redis: {e}")
            # Fallback to current balance if Redis fails
            start_of_day_balance = balance
            
        drawdown_pct = ((start_of_day_balance - equity) / start_of_day_balance) * 100
        
        if drawdown_pct > self.max_daily_dd:
            logger.critical(f"DAILY DRAWDOWN BREACH! Current: {drawdown_pct:.2f}% > Limit: {self.max_daily_dd:.2f}% (Start: {start_of_day_balance}, Equity: {equity})")
            return False
        
        # Total drawdown should technically be from initial deposit. Using a placeholder or 
        # a separate Redis key for Peak Equity would be better, but for now we fix daily.
        total_drawdown_pct = ((balance - equity) / balance) * 100
        if total_drawdown_pct > self.max_total_dd:
            logger.critical(f"TOTAL DRAWDOWN BREACH! Current: {total_drawdown_pct:.2f}% > Limit: {self.max_total_dd:.2f}%")
            return False
            
        return True
        
    def evaluate_positions_risk(self, positions: List[Dict[str, Any]]) -> bool:
        """Returns True if open positions are within safe limits."""
        num_positions = len(positions)
        
        if num_positions >= self.max_positions:
            logger.critical(f"MAX POSITIONS! Current: {num_positions} >= Limit: {self.max_positions}")
            return False
        
        # Check individual position losses
        for pos in positions:
            profit = pos.get("profit", 0)
            # Log large individual losses
            if profit < -100:  # arbitrary threshold for warning
                logger.warning(f"Large loss on position {pos.get('ticket', '?')}: {profit}")
            
        return True
