import asyncio
import logging
from core.config import CONFIG
from core.mcp_client import mcp_client
from core.dragonfly_bus import DragonflyBus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def tick_publisher_loop():
    bus = DragonflyBus()
    await bus.connect()
    
    symbols = CONFIG.symbols_list
    logger.info(f"Starting tick publisher for symbols: {symbols}")
    
    try:
        while True:
            for symbol in symbols:
                tick = await mcp_client.get_price(symbol)
                if tick:
                    channel = f"ticks:{symbol}"
                    await bus.publish(channel, tick)
                    
            # Poll every 200ms
            await asyncio.sleep(0.2)
            
    except asyncio.CancelledError:
        logger.info("Tick publisher loop cancelled")
    except Exception as e:
        logger.error(f"Tick publisher error: {e}")
    finally:
        await bus.disconnect()
        mcp_client.disconnect()

if __name__ == "__main__":
    try:
        asyncio.run(tick_publisher_loop())
    except KeyboardInterrupt:
        logger.info("Tick publisher stopped by user")
