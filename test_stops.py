import pytest
import asyncio
import MetaTrader5 as mt5
from core.config import CONFIG
from core.mcp_client import mcp_client
from agents.trade_executor import trade_executor

@pytest.mark.asyncio
async def test_stops():
    if not mcp_client.connect():
        print("Failed to connect.")
        return

    symbol = "BTCUSD"
    print(f"Testing stops for: {symbol}")
    
    # Try to sell BTCUSD with 500 pip SL and 1000 pip TP
    # Based on the new logic:
    # 1 pip = 1.0 USD
    # SL = Price + 500 USD
    # TP = Price - 1000 USD
    signal = {
        "action": "SELL",
        "symbol": symbol,
        "lot_size": 0.01,
        "sl_pips": 500,
        "tp_pips": 1000
    }
    
    print(f"Executing trade: {signal}")
    result = await trade_executor.execute_signal(signal)
    
    if result:
        print(f"\nSUCCESS! Trade executed: {result}")
        # Immediately close the position to not leave a demo trade open if it succeeded
        print("Closing the position...")
        # Since close_position is an MCP tool or we can close it via MT5:
        # We can just let it run or close it. Let's close it to clean up.
        ticket = result['ticket']
        # standard MT5 close:
        position = mt5.positions_get(ticket=ticket)
        if position:
            pos = position[0]
            tick = mt5.symbol_info_tick(symbol)
            close_price = tick.ask if pos.type == mt5.POSITION_TYPE_SELL else tick.bid
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_BUY if pos.type == mt5.POSITION_TYPE_SELL else mt5.ORDER_TYPE_SELL,
                "position": ticket,
                "price": close_price,
                "deviation": 20,
                "magic": 234000,
                "comment": "Close test",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": pos.type_filling,
            }
            res = mt5.order_send(req)
            print(f"Position close result: {res.comment if res else 'failed'}")
    else:
        print("\nFAILED to execute trade.")

if __name__ == "__main__":
    asyncio.run(test_stops())
