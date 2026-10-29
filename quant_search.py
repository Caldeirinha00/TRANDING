import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime

print("=== AI_TRADING QUANT BACKTEST ENGINE ===")

if not mt5.initialize():
    print("MT5 init failed")
    exit(1)

symbol = "EURUSD"
start_dt = datetime(2026, 1, 1)
end_dt = datetime(2026, 9, 16)

rates_m5 = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_dt, end_dt)
rates_h1 = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start_dt, end_dt)

sym_info = mt5.symbol_info(symbol)
tick_size = sym_info.trade_tick_size
tick_val = sym_info.trade_tick_value
point = sym_info.point
spread = sym_info.spread * point
mt5.shutdown()

print(f"Loaded: M5={len(rates_m5)}, H1={len(rates_h1)}, Spread={spread:.5f}")

df_m5 = pd.DataFrame(rates_m5)
df_h1 = pd.DataFrame(rates_h1)

def calc_py_ema(series, period):
    n = len(series)
    ema = np.zeros(n)
    if n < period:
        return ema
    k = 2.0 / (period + 1.0)
    ema[period-1] = np.mean(series[:period])
    for i in range(period, n):
        ema[i] = (series[i] - ema[i-1]) * k + ema[i-1]
    return ema

def calc_py_atr(high, low, close, period):
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, max(hc, lc))
    atr = np.zeros(n)
    if n < period:
        return tr, atr
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return tr, atr

def calc_py_adx(high, low, close, period=14):
    n = len(close)
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, max(hc, lc))
        
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down
            
    sm_tr = np.zeros(n)
    sm_plus = np.zeros(n)
    sm_minus = np.zeros(n)
    
    if n < period + 1:
        return np.zeros(n)
        
    sm_tr[period] = np.sum(tr[1:period+1])
    sm_plus[period] = np.sum(plus_dm[1:period+1])
    sm_minus[period] = np.sum(minus_dm[1:period+1])
    
    for i in range(period+1, n):
        sm_tr[i] = sm_tr[i-1] - (sm_tr[i-1]/period) + tr[i]
        sm_plus[i] = sm_plus[i-1] - (sm_plus[i-1]/period) + plus_dm[i]
        sm_minus[i] = sm_minus[i-1] - (sm_minus[i-1]/period) + minus_dm[i]
        
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    for i in range(period, n):
        if sm_tr[i] > 0:
            plus_di[i] = 100.0 * sm_plus[i] / sm_tr[i]
            minus_di[i] = 100.0 * sm_minus[i] / sm_tr[i]
        diff = abs(plus_di[i] - minus_di[i])
        summ = plus_di[i] + minus_di[i]
        if summ > 0:
            dx[i] = 100.0 * diff / summ
            
    adx = np.zeros(n)
    if n >= 2 * period:
        adx[2*period - 1] = np.mean(dx[period:2*period])
        for i in range(2*period, n):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    return adx

# M5 data
m5_close = df_m5['close'].values
m5_open = df_m5['open'].values
m5_high = df_m5['high'].values
m5_low = df_m5['low'].values
m5_time = df_m5['time'].values

m5_ema20 = calc_py_ema(m5_close, 20)
_, m5_atr14 = calc_py_atr(m5_high, m5_low, m5_close, 14)
m5_atr_med = pd.Series(m5_atr14).rolling(50, min_periods=1).median().values

# H1 data
h1_close = df_h1['close'].values
h1_high = df_h1['high'].values
h1_low = df_h1['low'].values
h1_time = df_h1['time'].values

h1_adx = calc_py_adx(h1_high, h1_low, h1_close, 14)

# Precalculate H1 EMAs for different periods
h1_emas = {}
for p in [20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100]:
    h1_emas[p] = calc_py_ema(h1_close, p)

h1_idx_for_m5 = np.searchsorted(h1_time, m5_time, side='right') - 2
h1_idx_for_m5 = np.clip(h1_idx_for_m5, 0, len(h1_time) - 1)

def simulate(fast_p, slow_p, adx_min, atr_min_mult, sl_buf, tp_rr):
    h1_fast = h1_emas[fast_p][h1_idx_for_m5]
    h1_slow = h1_emas[slow_p][h1_idx_for_m5]
    adx = h1_adx[h1_idx_for_m5]
    
    n = len(m5_close)
    pos_type = 0 # 0: none, 1: buy, -1: sell
    entry_price = 0.0
    sl = 0.0
    tp = 0.0
    volume = 0.0
    be_level = 0.0
    be_reached = False
    
    balance = 1000.0
    peak_balance = 1000.0
    max_dd = 0.0
    trades = 0
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0
    total_comm = 0.0
    
    # 7 EUR round-turn commission per lot
    comm_per_lot = 7.0
    
    for i in range(100, n):
        # Update peak & drawdown
        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance * 100.0
        if dd > max_dd:
            max_dd = dd
            
        cur_high = m5_high[i]
        cur_low = m5_low[i]
        cur_close = m5_close[i]
        
        # 1. Manage open position
        if pos_type == 1: # BUY
            # Check Break Even trigger
            r_dist = entry_price - (entry_price - (entry_price - sl) if not be_reached else 0)
            if not be_reached and (cur_high - entry_price) >= (entry_price - sl):
                be_level = entry_price + spread + (comm_per_lot * tick_size / tick_val)
                sl = max(sl, be_level)
                be_reached = True
            elif be_reached:
                # Trailing stop
                trail = cur_close - m5_atr14[i] * 1.5
                if trail > sl:
                    sl = trail
                    
            # Check TP
            if cur_high >= tp:
                pnl = (tp - entry_price) / tick_size * tick_val * volume - (comm_per_lot * volume)
                balance += pnl
                gross_profit += pnl
                wins += 1
                trades += 1
                total_comm += comm_per_lot * volume
                pos_type = 0
            # Check SL
            elif cur_low <= sl:
                pnl = (sl - entry_price) / tick_size * tick_val * volume - (comm_per_lot * volume)
                balance += pnl
                if pnl > 0:
                    gross_profit += pnl
                    wins += 1
                else:
                    gross_loss += abs(pnl)
                    losses += 1
                trades += 1
                total_comm += comm_per_lot * volume
                pos_type = 0
                
        elif pos_type == -1: # SELL
            if not be_reached and (entry_price - cur_low) >= (sl - entry_price):
                be_level = entry_price - spread - (comm_per_lot * tick_size / tick_val)
                sl = min(sl, be_level)
                be_reached = True
            elif be_reached:
                trail = cur_close + m5_atr14[i] * 1.5
                if trail < sl:
                    sl = trail
                    
            if cur_low <= tp:
                pnl = (entry_price - tp) / tick_size * tick_val * volume - (comm_per_lot * volume)
                balance += pnl
                gross_profit += pnl
                wins += 1
                trades += 1
                total_comm += comm_per_lot * volume
                pos_type = 0
            elif cur_high >= sl:
                pnl = (entry_price - sl) / tick_size * tick_val * volume - (comm_per_lot * volume)
                balance += pnl
                if pnl > 0:
                    gross_profit += pnl
                    wins += 1
                else:
                    gross_loss += abs(pnl)
                    losses += 1
                trades += 1
                total_comm += comm_per_lot * volume
                pos_type = 0
                
        # 2. If no position, evaluate entry
        if pos_type == 0:
            if adx[i-1] >= adx_min and m5_atr14[i-1] >= (m5_atr_med[i-1] * atr_min_mult):
                # Bullish
                if h1_fast[i-1] > h1_slow[i-1]:
                    if m5_low[i-1] <= m5_ema20[i-1] and m5_close[i-1] > m5_ema20[i-1]:
                        # BUY
                        pos_type = 1
                        entry_price = m5_open[i] + spread
                        sl = min(m5_low[i-1], m5_ema20[i-1]) - (m5_atr14[i-1] * sl_buf)
                        risk_pts = entry_price - sl
                        if risk_pts > 0:
                            tp = entry_price + risk_pts * tp_rr
                            risk_eur = balance * 0.01
                            volume = round((risk_eur / (risk_pts / tick_size * tick_val)), 2)
                            volume = max(0.01, min(10.0, volume))
                            be_reached = False
                        else:
                            pos_type = 0
                # Bearish
                elif h1_fast[i-1] < h1_slow[i-1]:
                    if m5_high[i-1] >= m5_ema20[i-1] and m5_close[i-1] < m5_ema20[i-1]:
                        # SELL
                        pos_type = -1
                        entry_price = m5_open[i]
                        sl = max(m5_high[i-1], m5_ema20[i-1]) + (m5_atr14[i-1] * sl_buf)
                        risk_pts = sl - entry_price
                        if risk_pts > 0:
                            tp = entry_price - risk_pts * tp_rr
                            risk_eur = balance * 0.01
                            volume = round((risk_eur / (risk_pts / tick_size * tick_val)), 2)
                            volume = max(0.01, min(10.0, volume))
                            be_reached = False
                        else:
                            pos_type = 0

    net_profit = balance - 1000.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 99.0
    win_rate = (wins / trades * 100.0) if trades > 0 else 0.0
    return {
        'fast': fast_p,
        'slow': slow_p,
        'adx_min': adx_min,
        'atr_mult': atr_min_mult,
        'sl_buf': sl_buf,
        'tp_rr': tp_rr,
        'net_profit': round(net_profit, 2),
        'trades': trades,
        'win_rate': round(win_rate, 1),
        'profit_factor': round(profit_factor, 2),
        'max_dd': round(max_dd, 1),
        'comm': round(total_comm, 2)
    }

print("Running baseline Pass 236 check...")
base = simulate(25, 70, 25.0, 1.0, 1.0, 2.75)
print("Baseline (Pass 236):", base)
