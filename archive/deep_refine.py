import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime

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

df_m5 = pd.DataFrame(rates_m5)
df_h1 = pd.DataFrame(rates_h1)

df_m5['dt'] = pd.to_datetime(df_m5['time'], unit='s')
df_m5['hour'] = df_m5['dt'].dt.hour

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

def calc_py_atr(high, low, close, period=14):
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

m5_close = df_m5['close'].values
m5_open = df_m5['open'].values
m5_high = df_m5['high'].values
m5_low = df_m5['low'].values
m5_hour = df_m5['hour'].values

m5_ema20 = calc_py_ema(m5_close, 20)
_, m5_atr14 = calc_py_atr(m5_high, m5_low, m5_close, 14)
m5_atr_med = pd.Series(m5_atr14).rolling(50, min_periods=1).median().values

h1_close = df_h1['close'].values
h1_high = df_h1['high'].values
h1_low = df_h1['low'].values
h1_time = df_h1['time'].values
h1_adx = calc_py_adx(h1_high, h1_low, h1_close, 14)

h1_emas = {}
for p in [20, 25, 30, 60, 70, 80]:
    h1_emas[p] = calc_py_ema(h1_close, p)

h1_idx_for_m5 = np.searchsorted(h1_time, df_m5['time'].values, side='right') - 2
h1_idx_for_m5 = np.clip(h1_idx_for_m5, 0, len(h1_time) - 1)

h1_adx_m5 = h1_adx[h1_idx_for_m5]

pullback_buy = (m5_low <= m5_ema20) & (m5_close > m5_ema20)
pullback_sell = (m5_high >= m5_ema20) & (m5_close < m5_ema20)

rejection_buy = m5_close > (m5_high + m5_low) / 2.0
rejection_sell = m5_close < (m5_high + m5_low) / 2.0

n_bars = len(m5_close)
comm_per_lot = 7.0

def run_sim(fast_p, slow_p, min_hour, max_hour, adx_t, atr_t, sl_buf, tp_rr, use_rej):
    h1_fast = h1_emas[fast_p][h1_idx_for_m5]
    h1_slow = h1_emas[slow_p][h1_idx_for_m5]
    bull_trend = h1_fast > h1_slow
    bear_trend = h1_fast < h1_slow
    
    balance = 1000.0
    peak_balance = 1000.0
    max_dd = 0.0
    trades = 0
    wins = 0
    gross_p = 0.0
    gross_l = 0.0
    total_comm = 0.0
    
    i = 100
    while i < n_bars - 1:
        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance * 100.0
        if dd > max_dd:
            max_dd = dd
            
        hr = m5_hour[i-1]
        if hr < min_hour or hr > max_hour:
            i += 1
            continue
            
        if h1_adx_m5[i-1] < adx_t or m5_atr14[i-1] < (m5_atr_med[i-1] * atr_t):
            i += 1
            continue
            
        is_buy = False
        is_sell = False
        
        if bull_trend[i-1] and pullback_buy[i-1]:
            if not use_rej or rejection_buy[i-1]:
                is_buy = True
        elif bear_trend[i-1] and pullback_sell[i-1]:
            if not use_rej or rejection_sell[i-1]:
                is_sell = True
                
        if not is_buy and not is_sell:
            i += 1
            continue
            
        if is_buy:
            entry = m5_open[i] + spread
            sl = min(m5_low[i-1], m5_ema20[i-1]) - (m5_atr14[i-1] * sl_buf)
            risk_pts = entry - sl
            if risk_pts <= 0:
                i += 1
                continue
            tp = entry + risk_pts * tp_rr
            risk_eur = balance * 0.01
            vol = max(0.01, min(10.0, round(risk_eur / (risk_pts / tick_size * tick_val), 2)))
            be_level = entry + spread + (comm_per_lot * tick_size / tick_val)
            be_active = False
            
            j = i
            closed = False
            while j < n_bars:
                if not be_active and (m5_high[j] - entry) >= risk_pts:
                    sl = max(sl, be_level)
                    be_active = True
                elif be_active:
                    trail = m5_close[j] - m5_atr14[j] * 1.5
                    if trail > sl:
                        sl = trail
                        
                if m5_high[j] >= tp:
                    pnl = (tp - entry) / tick_size * tick_val * vol - (comm_per_lot * vol)
                    balance += pnl
                    gross_p += pnl
                    wins += 1
                    trades += 1
                    total_comm += comm_per_lot * vol
                    closed = True
                    i = j + 1
                    break
                elif m5_low[j] <= sl:
                    pnl = (sl - entry) / tick_size * tick_val * vol - (comm_per_lot * vol)
                    balance += pnl
                    if pnl > 0:
                        gross_p += pnl
                        wins += 1
                    else:
                        gross_l += abs(pnl)
                    trades += 1
                    total_comm += comm_per_lot * vol
                    closed = True
                    i = j + 1
                    break
                j += 1
            if not closed:
                break
                
        elif is_sell:
            entry = m5_open[i]
            sl = max(m5_high[i-1], m5_ema20[i-1]) + (m5_atr14[i-1] * sl_buf)
            risk_pts = sl - entry
            if risk_pts <= 0:
                i += 1
                continue
            tp = entry - risk_pts * tp_rr
            risk_eur = balance * 0.01
            vol = max(0.01, min(10.0, round(risk_eur / (risk_pts / tick_size * tick_val), 2)))
            be_level = entry - spread - (comm_per_lot * tick_size / tick_val)
            be_active = False
            
            j = i
            closed = False
            while j < n_bars:
                if not be_active and (entry - m5_low[j]) >= risk_pts:
                    sl = min(sl, be_level)
                    be_active = True
                elif be_active:
                    trail = m5_close[j] + m5_atr14[j] * 1.5
                    if trail < sl:
                        sl = trail
                        
                if m5_low[j] <= tp:
                    pnl = (entry - tp) / tick_size * tick_val * vol - (comm_per_lot * vol)
                    balance += pnl
                    gross_p += pnl
                    wins += 1
                    trades += 1
                    total_comm += comm_per_lot * vol
                    closed = True
                    i = j + 1
                    break
                elif m5_high[j] >= sl:
                    pnl = (entry - sl) / tick_size * tick_val * vol - (comm_per_lot * vol)
                    balance += pnl
                    if pnl > 0:
                        gross_p += pnl
                        wins += 1
                    else:
                        gross_l += abs(pnl)
                    trades += 1
                    total_comm += comm_per_lot * vol
                    closed = True
                    i = j + 1
                    break
                j += 1
            if not closed:
                break

    net_p = balance - 1000.0
    pf = (gross_p / gross_l) if gross_l > 0 else 99.0
    wr = (wins / trades * 100.0) if trades > 0 else 0.0
    return {
        'fast': fast_p,
        'slow': slow_p,
        'hours': f"{min_hour:02d}-{max_hour:02d}",
        'adx': adx_t,
        'atr': atr_t,
        'sl_buf': sl_buf,
        'tp_rr': tp_rr,
        'rej': use_rej,
        'net_p': round(net_p, 2),
        'trades': trades,
        'wr': round(wr, 1),
        'pf': round(pf, 2),
        'max_dd': round(max_dd, 1),
        'comm': round(total_comm, 2)
    }

print("Running deep neighborhood refinement...")
results = []
for fast_p in [20, 25, 30]:
    for slow_p in [60, 70, 80]:
        for h_start, h_end in [(7, 17), (8, 16), (8, 17)]:
            for adx_t in [28.0, 30.0, 32.0]:
                for atr_t in [1.3, 1.4, 1.5]:
                    for sl_b in [1.2, 1.4, 1.6]:
                        for tp_r in [1.4, 1.5, 1.75]:
                            results.append(run_sim(fast_p, slow_p, h_start, h_end, adx_t, atr_t, sl_b, tp_r, True))

df = pd.DataFrame(results)
pos = df[df['net_p'] > 0]
print(f"Refinement: Tested={len(df)}, Profitable={len(pos)}")
print("\n--- TOP 15 PROFITABLE COMBINATIONS ---")
print(pos.sort_values(by=['net_p', 'pf'], ascending=[False, False]).head(15).to_string(index=False))
