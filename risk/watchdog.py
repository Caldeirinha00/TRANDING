import asyncio
import logging
import time
from datetime import datetime
from redis.asyncio import Redis
import MetaTrader5 as mt5

from core.config import CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("Watchdog")

class Watchdog:
    def __init__(self):
        self.redis_url = CONFIG.redis_url
        self.missed_beats = 0
        self.max_misses = 3
        self.check_interval = 5  # Check every 5 seconds to match heartbeat emission
        
    async def run(self):
        logger.info("Watchdog started. Monitoring system:heartbeat...")
        redis = Redis.from_url(self.redis_url, decode_responses=True)
        
        try:
            while True:
                await asyncio.sleep(self.check_interval)
                try:
                    # 1. Self Network Validation
                    try:
                        await redis.ping()
                    except Exception as e:
                        logger.error(f"Watchdog lost connection to Redis: {e}. Skipping check.")
                        continue
                        
                    # 2. Check Heartbeat
                    try:
                        hb_str = await redis.get("system:heartbeat")
                    except Exception as e:
                        logger.error(f"Watchdog failed to read heartbeat from Redis: {e}. Skipping check.")
                        continue
                        
                    if not hb_str:
                        logger.warning("No heartbeat found in Redis!")
                        self.missed_beats += 1
                    else:
                        hb_time = float(hb_str)
                        age = time.time() - hb_time
                        if age > 15.0: # Heartbeat is older than 15s (3 missed 5s beats)
                            logger.warning(f"Heartbeat is old! Age: {age:.1f}s")
                            self.missed_beats += 1
                        else:
                            if self.missed_beats > 0:
                                logger.info("Heartbeat recovered.")
                            self.missed_beats = 0
                    
                    # 3. Action
                    if self.missed_beats >= self.max_misses:
                        logger.critical(f"WATCHDOG TRIGGERED! {self.missed_beats} missed beats.")
                        await self._take_control(redis)
                        
                except Exception as e:
                    logger.error(f"Watchdog internal error: {e}")
        except asyncio.CancelledError:
            pass
        finally:
            await redis.aclose()

    async def _take_control(self, redis: Redis):
        """Panic close only losing positions and lock the system."""
        # Lock the system so main.py doesn't wake up and trade
        await redis.set("system:lock", "WATCHDOG_ACTIVE")
        logger.critical("System Lock 'WATCHDOG_ACTIVE' injected into Redis.")
        
        # Connect to MT5
        if not mt5.initialize(path=CONFIG.MT5_PATH, login=CONFIG.MT5_LOGIN, password=CONFIG.MT5_PASSWORD, server=CONFIG.MT5_SERVER):
            logger.error("Watchdog failed to connect to MT5!")
            return
            
        positions = mt5.positions_get()
        if not positions:
            logger.info("No open positions to close.")
            mt5.shutdown()
            self.missed_beats = 0 # reset to monitor again
            return
            
        closed_count = 0
        for pos in positions:
            profit = pos.profit
            # Check if it's losing.
            if profit < 0:
                # User nuance: Don't close if it's very close to SL.
                # Distance to SL in pips:
                symbol_info = mt5.symbol_info(pos.symbol)
                point = symbol_info.point if symbol_info else 0.00001
                
                # A very rough heuristic: if SL is 0, distance is infinite.
                distance_to_sl_points = 999999
                if pos.sl > 0:
                    distance_to_sl_points = abs(pos.price_current - pos.sl) / point
                    
                # 50 points = 5 pips
                if distance_to_sl_points < 50:
                    logger.info(f"Position {pos.ticket} is losing but very close to SL ({distance_to_sl_points:.1f} points). Leaving to broker.")
                    continue
                    
                logger.critical(f"Watchdog closing losing position {pos.ticket} on {pos.symbol} (Profit: {profit})")
                
                tick = mt5.symbol_info_tick(pos.symbol)
                if not tick:
                    continue
                    
                action = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                price = tick.bid if action == mt5.ORDER_TYPE_SELL else tick.ask
                
                # Get correct filling mode dynamically
                from agents.trade_executor import _get_filling_mode
                filling_mode = _get_filling_mode(symbol_info)
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": pos.symbol,
                    "volume": pos.volume,
                    "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    "position": pos.ticket,
                    "price": price,
                    "deviation": 20,
                    "magic": 234000,
                    "comment": "Panic Close",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": filling_mode,
                }
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    closed_count += 1
                    logger.info(f"Position {pos.ticket} closed.")
                else:
                    logger.error(f"Failed to close {pos.ticket}: {result.comment if result else 'Unknown'}")
        
        logger.critical(f"Watchdog closed {closed_count} losing positions. Leaving winning positions alone.")
        mt5.shutdown()
        # Reset misses so it doesn't spam, but the lock is already in place
        self.missed_beats = 0

if __name__ == "__main__":
    wd = Watchdog()
    asyncio.run(wd.run())
