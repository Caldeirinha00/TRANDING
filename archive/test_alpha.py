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
m5_time = df_m5['time'].values
m5_hour = df_m5['hour'].values

m5_ema20 = calc_py_ema(m5_close, 20)
_, m5_atr14 = calc_py_atr(m5_high, m5_low, m5_close, 14)
m5_atr_med = pd.Series(m5_atr14).rolling(50, min_periods=1).median().values

h1_close = df_h1['close'].values
h1_high = df_h1['high'].values
h1_low = df_h1['low'].values
h1_time = df_h1['time'].values
h1_adx = calc_py_adx(h1_high, h1_low, h1_close, 14)

h1_fast = calc_py_ema(h1_close, 25)
h1_slow = calc_py_ema(h1_close, 70)

h1_idx_for_m5 = np.searchsorted(h1_time, m5_time, side='right') - 2
h1_idx_for_m5 = np.clip(h1_idx_for_m5, 0, len(h1_time) - 1)

h1_fast_m5 = h1_fast[h1_idx_for_m5]
h1_slow_m5 = h1_slow[h1_idx_for_m5]
h1_adx_m5 = h1_adx[h1_idx_for_m5]

print("Indicators ready. Testing time and filter refinements...")

def run_test(min_hour, max_hour, adx_thresh, atr_thresh, sl_mult, tp_mult, require_rejection):
    n = len(m5_close)
    pos_type = 0
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
    comm_per_lot = 7.0
    
    for i in range(100, n):
        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance * 100.0
        if dd > max_dd:
            max_dd = dd
            
        cur_high = m5_high[i]
        cur_low = m5_low[i]
        cur_close = m5_close[i]
        
        if pos_type == 1: # BUY
            # BE check
            if not be_reached and (cur_high - entry_price) >= (entry_price - sl):
                be_level = entry_price + spread + (comm_per_lot * tick_size / tick_val)
                sl = max(sl, be_level)
                be_reached = True
            elif be_reached:
                trail = cur_close - m5_atr14[i] * 1.5
                if trail > sl:
                    sl = trail
                    
            if cur_high >= tp:
                pnl = (tp - entry_price) / tick_size * tick_val * volume - (comm_per_lot * volume)
                balance += pnl
                gross_profit += pnl
                wins += 1
                trades += 1
                total_comm += comm_per_lot * volume
                pos_type = 0
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
                
        if pos_type == 0:
            # Time filter
            hr = m5_hour[i-1]
            if hr < min_hour or hr > max_hour:
                continue
                
            if h1_adx_m5[i-1] >= adx_thresh and m5_atr14[i-1] >= (m5_atr_med[i-1] * atr_thresh):
                # Bullish
                if h1_fast_m5[i-1] > h1_slow_m5[i-1]:
                    pullback = (m5_low[i-1] <= m5_ema20[i-1] and m5_close[i-1] > m5_ema20[i-1])
                    if require_rejection:
                        rejection = m5_close[i-1] > (m5_high[i-1] + m5_low[i-1]) / 2.0
                    else:
                        rejection = True
                    if pullback and rejection:
                        pos_type = 1
                        entry_price = m5_open[i] + spread
                        sl = min(m5_low[i-1], m5_ema20[i-1]) - (m5_atr14[i-1] * sl_mult)
                        risk_pts = entry_price - sl
                        if risk_pts > 0:
                            tp = entry_price + risk_pts * tp_mult
                            risk_eur = balance * 0.01
                            volume = round((risk_eur / (risk_pts / tick_size * tick_val)), 2)
                            volume = max(0.01, min(10.0, volume))
                            be_reached = False
                        else:
                            pos_type = 0
                # Bearish
                elif h1_fast_m5[i-1] < h1_slow_m5[i-1]:
                    pullback = (m5_high[i-1] >= m5_ema20[i-1] and m5_close[i-1] < m5_ema20[i-1])
                    if require_rejection:
                        rejection = m5_close[i-1] < (m5_high[i-1] + m5_low[i-1]) / 2.0
                    else:
                        rejection = True
                    if pullback and rejection:
                        pos_type = -1
                        entry_price = m5_open[i]
                        sl = max(m5_high[i-1], m5_ema20[i-1]) + (m5_atr14[i-1] * sl_mult)
                        risk_pts = sl - entry_price
                        if risk_pts > 0:
                            tp = entry_price - risk_pts * tp_mult
                            risk_eur = balance * 0.01
                            volume = round((risk_eur / (risk_pts / tick_size * tick_val)), 2)
                            volume = max(0.01, min(10.0, volume))
                            be_reached = False
                        else:
                            pos_type = 0
                            
    net_p = balance - 1000.0
    pf = (gross_profit / gross_loss) if gross_loss > 0 else 99.0
    wr = (wins / trades * 100.0) if trades > 0 else 0.0
    return {
        'hours': f"{min_hour:02d}-{max_hour:02d}",
        'adx': adx_thresh,
        'atr_mult': atr_thresh,
        'sl_buf': sl_mult,
        'tp_rr': tp_mult,
        'reject': require_rejection,
        'net_p': round(net_p, 2),
        'trades': trades,
        'wr': round(wr, 1),
        'pf': round(pf, 2),
        'max_dd': round(max_dd, 1),
        'comm': round(total_comm, 2)
    }

results = []
for h_start, h_end in [(0, 23), (7, 18), (8, 17), (12, 18)]:
    for adx_t in [25.0, 27.5, 30.0, 32.5]:
        for atr_t in [1.0, 1.1, 1.2, 1.3]:
            for sl_m in [1.0, 1.2, 1.4]:
                for tp_m in [2.0, 2.5, 3.0]:
                    for rej in [False, True]:
                        res = run_test(h_start, h_end, adx_t, atr_t, sl_m, tp_m, rej)
                        results.append(res)

df_res = pd.DataFrame(results)
pos_res = df_res[df_res['net_p'] > 0]
print(f"Total configurations tested: {len(df_res)}")
print(f"Total profitable configurations: {len(pos_res)}")

if len(pos_res) > 0:
    print("\nTop 15 Most Profitable Setups:")
    print(pos_res.sort_values(by='net_p', ascending=False).head(15).to_string(index=False))
else:
    print("\nBest 10 setups:")
    print(df_res.sort_values(by='net_p', ascending=False).head(10).to_string(index=False))
