import asyncio
import logging
import json
import aiohttp
from datetime import datetime, timedelta
from core.mt5_connection import mt5
from redis.asyncio import Redis
from core.mt5_executor import mt5_call

from core.config import CONFIG
from memory.trade_journal import trade_journal

logger = logging.getLogger(__name__)

class TradeMonitor:
    """
    Background worker that monitors MT5 for closed trades,
    retrieves their original reasoning from Redis, and logs them to Qdrant Memory.
    """
    def __init__(self):
        self.redis_url = CONFIG.redis_url
        self._redis_client = None
        self.last_check_time = None

    async def _get_redis(self):
        if self._redis_client is None:
            self._redis_client = Redis.from_url(self.redis_url, decode_responses=True)
        return self._redis_client

    async def run(self):
        logger.info("Trade Monitor started. Watching for closed trades to build Memory...")
        while True:
            await asyncio.sleep(60) # Check every 60 seconds
            try:
                await self._check_closed_trades()
            except Exception as e:
                logger.error(f"Error in Trade Monitor: {e}")

    async def _check_closed_trades(self):
        redis = await self._get_redis()
        
        if self.last_check_time is None:
            # Try to load from Redis
            saved_time = await redis.get("trade_monitor:last_check_time")
            if saved_time:
                self.last_check_time = datetime.fromtimestamp(float(saved_time))
            else:
                self.last_check_time = datetime.now()

        now = datetime.now()
        deals = await mt5_call(mt5.history_deals_get, self.last_check_time, now)
        
        if not deals:
            # Still update the time to avoid re-querying huge windows
            self.last_check_time = now
            await redis.set("trade_monitor:last_check_time", now.timestamp())
            return

        for deal in deals:
            if deal.entry == 1:
                try:
                    ticket = deal.position_id
                    profit = deal.profit
                    action = "BUY" if deal.type == mt5.ORDER_TYPE_SELL else "SELL" 
                    if profit > 0:
                        outcome = "WIN"
                    elif profit < 0:
                        outcome = "LOSS"
                    else:
                        outcome = "BREAKEVEN"

                    key = f"trade_context:{ticket}"
                    context = await redis.get(key)
                    
                    if context:
                        try:
                            data = json.loads(context)
                            ctx_str = data.get("context", "")
                            confidence = data.get("confidence", 0)
                            setup_type = data.get("setup_type", "UNKNOWN")
                        except Exception:
                            ctx_str = context
                            confidence = 0
                            setup_type = "UNKNOWN"
                            
                        logger.info(f"Logging closed trade {ticket} to Memory (Outcome: {outcome}, Profit: {profit}, Setup: {setup_type})")
                        await trade_journal.log_trade(
                            context_summary=ctx_str,
                            action=action,
                            outcome=outcome,
                            profit=profit,
                            symbol=deal.symbol,
                            setup_type=setup_type
                        )
                        
                        asyncio.create_task(self._send_telemetry(deal.symbol, confidence, profit, outcome, setup_type))
                        await redis.delete(key)
                    else:
                        logger.debug(f"Trade {ticket} closed but no context found in memory cache.")
                except Exception as e:
                    logger.error(f"Failed to process deal {deal.ticket}: {e}")
                    
        # Update check time only after processing the batch
        self.last_check_time = now
        await redis.set("trade_monitor:last_check_time", now.timestamp())

    async def _send_telemetry(self, symbol: str, confidence: int, profit: float, outcome: str, setup_type: str = "UNKNOWN"):
        """Sends trade outcome data to QuestDB for confidence calibration."""
        if confidence is None or confidence == 0:
            return
            
        # 1 for WIN, 0 for LOSS, -1 for BREAKEVEN to easily filter in SQL if needed
        # Or better yet, queries can just 'WHERE profit != 0' for pure winrate.
        if outcome == "WIN":
            is_win = 1
        elif outcome == "LOSS":
            is_win = 0
        else:
            is_win = -1
            
        payload = f"confidence_calibration,symbol={symbol},setup_type={setup_type} predicted_confidence={confidence}i,actual_profit={profit},is_win={is_win}i"
        
        headers = {}
        if CONFIG.QUESTDB_AUTH_TOKEN:
            headers["Authorization"] = f"Bearer {CONFIG.QUESTDB_AUTH_TOKEN}"
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"http://{CONFIG.QUESTDB_HOST}:9000/write", data=payload, headers=headers, timeout=2.0) as resp:
                    if resp.status not in (200, 204):
                        logger.warning(f"QuestDB telemetry rejected: {resp.status} {await resp.text()}")
        except Exception as e:
            logger.debug(f"QuestDB telemetry failed: {e}")

trade_monitor = TradeMonitor()
