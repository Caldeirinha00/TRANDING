import MetaTrader5 as mt5
from core.config import CONFIG

mt5.initialize(path=CONFIG.MT5_PATH, login=CONFIG.MT5_LOGIN, password=CONFIG.MT5_PASSWORD, server=CONFIG.MT5_SERVER)

try:
    # find symbols with BTC
    symbols = mt5.symbols_get()
    if symbols:
        for s in symbols:
            if "BTC" in s.name:
                tick = mt5.symbol_info_tick(s.name)
                price = tick.ask if tick else 0.0
                print(f"{s.name} - price: {price}")
    else:
        print("No symbols retrieved or MT5 not connected.")
finally:
    mt5.shutdown()
