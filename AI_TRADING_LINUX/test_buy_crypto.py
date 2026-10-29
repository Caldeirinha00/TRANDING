import pytest
import asyncio
from core.mt5_connection import mt5
from core.config import CONFIG
from core.mcp_client import mcp_client
from agents.trade_executor import trade_executor

@pytest.mark.asyncio
async def test_buy():
    if not mcp_client.connect():
        print("Failed to connect.")
        return

    symbol = "BTCUSD"
    print(f"Using symbol: {symbol}")
    
    # Try to buy
    signal = {
        "action": "BUY",
        "symbol": symbol,
        "lot_size": 0.01,
        "sl_pips": 0,
        "tp_pips": 0
    }
    
    print(f"Executing trade: {signal}")
    result = await trade_executor.execute_signal(signal)
    
    if result:
        print(f"\nSUCCESS! Trade executed: {result}")
    else:
        print("\nFAILED to execute trade.")

if __name__ == "__main__":
    asyncio.run(test_buy())
