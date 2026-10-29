import asyncio
import json
import logging
from agents.trade_executor import trade_executor
from core.dragonfly_bus import DragonflyBus
from core.mt5_connection import mt5

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dragonfly_bus = DragonflyBus()

async def main():
    logger.info("Initializing Test Trade Workflow...")
    
    # 1. Connect to MT5
    if not mt5.initialize():
        logger.error("Failed to initialize MT5")
        return
        
    # 2. Connect to DragonflyDB to broadcast to the UI
    await dragonfly_bus.connect()
    
    # 3. Create the AI Decision payload
    signal = {
        "action": "BUY",
        "symbol": "EURUSD",
        "lot_size": 0.01,
        "sl_pips": 15,
        "tp_pips": 30,
        "reasoning": "TEST WORKFLOW: Executing manual override to verify MT5 order routing and UI WebSocket broadcast latency. Testing system connectivity.",
        "confidence": 0.99
    }
    
    # 4. Broadcast to the UI first
    logger.info("Broadcasting decision to DragonflyDB (UI will see this instantly)")
    await dragonfly_bus.publish("ai_decisions", signal)
    
    # 5. Execute physically on MT5
    logger.info("Executing Trade on MT5...")
    success = await trade_executor.execute_signal(signal)
    
    if success:
        logger.info("TEST SUCCESS: Trade placed successfully on the broker!")
    else:
        logger.error("TEST FAILED: Trade failed to place.")

    mt5.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
