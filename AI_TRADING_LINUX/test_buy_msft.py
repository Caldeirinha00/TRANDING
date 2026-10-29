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

    # Find Microsoft symbol
    symbols = mt5.symbols_get()
    msft_symbols = [s.name for s in symbols if "MSFT" in s.name]
    print(f"Found MSFT symbols: {msft_symbols}")
    
    if not msft_symbols:
        print("No MSFT symbol found.")
        return
        
    symbol = msft_symbols[0]
    print(f"Using symbol: {symbol}")
    
    # Try to buy
    signal = {
        "action": "BUY",
        "symbol": symbol,
        "lot_size": 0.1,
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
