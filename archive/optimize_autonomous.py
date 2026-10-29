import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import time

def fetch_data():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    
    print("Fetching M5 and H1 data from MT5...")
    start_dt = datetime(2026, 1, 1)
    end_dt = datetime(2026, 9, 16)
    
    rates_m5 = mt5.copy_rates_range("EURUSD", mt5.TIMEFRAME_M5, start_dt, end_dt)
    rates_h1 = mt5.copy_rates_range("EURUSD", mt5.TIMEFRAME_H1, start_dt, end_dt)
    
    symbol_info = mt5.symbol_info("EURUSD")
    tick_size = symbol_info.trade_tick_size
    tick_val = symbol_info.trade_tick_value
    point = symbol_info.point
    spread = symbol_info.spread * point
    
    mt5.shutdown()
    
    df_m5 = pd.DataFrame(rates_m5)
    df_m5['datetime'] = pd.to_datetime(df_m5['time'], unit='s')
    
    df_h1 = pd.DataFrame(rates_h1)
    df_h1['datetime'] = pd.to_datetime(df_h1['time'], unit='s')
    
    print(f"Loaded {len(df_m5)} M5 bars and {len(df_h1)} H1 bars.")
    return df_m5, df_h1, tick_size, tick_val, point, spread

if __name__ == '__main__':
    fetch_data()
