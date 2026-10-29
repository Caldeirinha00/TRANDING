import pytest
import asyncio
import logging
import sys
import MetaTrader5 as mt5

from agents.trade_executor import trade_executor
from core.mcp_client import mcp_client

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

@pytest.mark.asyncio
async def test_buy_msft():
    print("--- FORCING BUY ORDER FOR MICROSOFT (MSFT) ---")
    
    if not mcp_client.connect():
        print("ERROR: Could not connect to MT5.")
        return

    # Let's try to find the correct MSFT symbol name in this broker
    symbols = mt5.symbols_get(group="*MSFT*")
    msft_symbol = "MSFT" # Default
    
    if symbols and len(symbols) > 0:
        msft_symbol = symbols[0].name
        print(f"Found Microsoft symbol in broker: {msft_symbol}")
    else:
        print("Could not find *MSFT* in symbol groups, trying exact 'MSFT'")
        
    signal = {
        "action": "BUY",
        "symbol": msft_symbol,
        "lot_size": 0.01,
        "sl_pips": 50,
        "tp_pips": 100
    }
    
    print(f"Executing Signal: {signal}")
    success = await trade_executor.execute_signal(signal)
    
    if success:
        print(f"--- SUCCESS: BUY ORDER FOR {msft_symbol} EXECUTED! ---")
    else:
        print(f"--- FAILED: Could not execute buy order for {msft_symbol}. It might not be supported on this Demo Server, or market is closed. ---")

if __name__ == "__main__":
    asyncio.run(test_buy_msft())
