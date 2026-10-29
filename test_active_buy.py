from core.mt5_connection import mt5
import logging
from core.config import CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_active():
    print("--- STARTING EXPLICIT BROKER TEST ---", flush=True)
    
    path = CONFIG.MT5_PATH
    print(f"Initializing MT5 with {CONFIG.MT5_SERVER} / {CONFIG.MT5_LOGIN}...", flush=True)
    
    if not mt5.initialize(path=path, login=CONFIG.MT5_LOGIN, password=CONFIG.MT5_PASSWORD, server=CONFIG.MT5_SERVER):
        print(f"ERROR: Could not initialize MT5. Error: {mt5.last_error()}", flush=True)
        return

    print("Retrieving account info...", flush=True)
    account = mt5.account_info()
    if account is None:
        print("ERROR: Could not retrieve account info.", flush=True)
        mt5.shutdown()
        return
        
    print(f"Connected to MT5 Broker: {account.company}", flush=True)
    print(f"Account: {account.login} | Server: {account.server}", flush=True)
    print(f"Balance: {account.balance} {account.currency}", flush=True)

    # Let's search for Bitcoin instead because it's the weekend!
    symbols = mt5.symbols_get(group="*BTC*")
    msft_symbol = "BTCUSD"
    
    if symbols and len(symbols) > 0:
        msft_symbol = symbols[0].name
        print(f"Found Crypto symbol: {msft_symbol}", flush=True)
    else:
        print("WARNING: Could not find MSFT in *MSFT* group.", flush=True)
        all_symbols = mt5.symbols_get()
        found = False
        if all_symbols:
            for s in all_symbols:
                if "MSFT" in s.name:
                    msft_symbol = s.name
                    print(f"Found Microsoft symbol via full search: {msft_symbol}", flush=True)
                    found = True
                    break
        if not found:
            print("Trying exact 'MSFT'", flush=True)

    print(f"\n--- ATTEMPTING TO BUY {msft_symbol} ---", flush=True)
    
    if not mt5.symbol_select(msft_symbol, True):
        print(f"ERROR: Failed to select {msft_symbol} in Market Watch.", flush=True)
        mt5.shutdown()
        return

    symbol_info = mt5.symbol_info(msft_symbol)
    if symbol_info is None:
        print(f"ERROR: Symbol {msft_symbol} not found.", flush=True)
        mt5.shutdown()
        return
        
    price = mt5.symbol_info_tick(msft_symbol).ask
    print(f"Current ASK price for {msft_symbol}: {price}", flush=True)
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": msft_symbol,
        "volume": 0.01,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "sl": 0.0,
        "tp": 0.0,
        "deviation": 20,
        "magic": 234000,
        "comment": "AI Auto Trade Test",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    
    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"FAILED to place order: retcode={result.retcode} description={result.comment}", flush=True)
    else:
        print(f"SUCCESS! Buy order placed. Ticket: {result.order}", flush=True)

    mt5.shutdown()

if __name__ == "__main__":
    test_active()
