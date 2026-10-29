import sys
import os
import math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import onnxruntime as ort
import MetaTrader5 as mt5

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from risk.quant_position_sizer import InstitutionalPositionSizer, AccountState
from scripts.backtest_v2_institutional import calc_ema, calc_atr, calc_adx, calc_sevcik_fdi, calc_permutation_entropy

mt5.initialize()
symbol = "EURUSD"
start_warmup = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
start_test = datetime(2026, 8, 21, 0, 0, tzinfo=timezone.utc)
end_test = datetime(2026, 10, 21, 23, 59, tzinfo=timezone.utc)
rates_m5 = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_warmup, end_test)
rates_h1 = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start_warmup, end_test)
mt5.shutdown()

df_m5 = pd.DataFrame(rates_m5)
df_h1 = pd.DataFrame(rates_h1)

h1_close = df_h1["close"].values
h1_high = df_h1["high"].values
h1_low = df_h1["low"].values
h1_time = df_h1["time"].values
h1_ema20 = calc_ema(h1_close, 20)
h1_ema50 = calc_ema(h1_close, 50)
h1_adx = calc_adx(h1_high, h1_low, h1_close, 14)

m5_close = df_m5["close"].values
m5_open = df_m5["open"].values
m5_high = df_m5["high"].values
m5_low = df_m5["low"].values
m5_time = df_m5["time"].values
m5_ema20 = calc_ema(m5_close, 20)
m5_atr14 = calc_atr(m5_high, m5_low, m5_close, 14)
m5_atr_med = pd.Series(m5_atr14).rolling(50, min_periods=1).median().values

h1_idx_for_m5 = np.clip(np.searchsorted(h1_time, m5_time, side="right") - 2, 0, len(h1_time) - 1)

onnx_path = os.path.join(PROJECT_ROOT, "models", "meta_guard_v2.onnx")
onnx_sess = ort.InferenceSession(onnx_path)
input_name = onnx_sess.get_inputs()[0].name
sizer = InstitutionalPositionSizer(max_single_risk_pct=0.01, max_daily_dd_pct=0.03, esma_leverage=30.0, max_total_margin_usage_pct=0.45, max_concurrent_positions=3)

test_start_ts = int(start_test.timestamp())
test_start_idx = np.searchsorted(m5_time, test_start_ts)

def test_cutoff(min_p=0.41, trailing_tp=True):
    state_b = {"balance": 1000.0, "trades": [], "wins": 18, "total_trades": 32}
    active_trade = None
    x0, x1 = m5_close[0], 0.0
    p00, p01, p10, p11 = 1.0, 0.0, 0.0, 1.0

    for i in range(1, len(m5_close)):
        atr_norm = max(m5_atr14[i], 1e-5)
        pred_x0 = x0 + x1
        pred_x1 = x1
        n_p00 = p00 + p10 + p01 + p11 + 1e-6 * atr_norm
        n_p01 = p01 + p11
        n_p10 = p10 + p11
        n_p11 = p11 + 5e-7 * atr_norm
        y = m5_close[i] - pred_x0
        S = n_p00 + 1e-3 * atr_norm
        K0 = n_p00 / S
        K1 = n_p10 / S
        x0 = pred_x0 + K0 * y
        x1 = pred_x1 + K1 * y
        p00 = (1.0 - K0) * n_p00
        p01 = (1.0 - K0) * n_p01
        p10 = n_p10 - K1 * n_p00
        p11 = n_p11 - K1 * n_p01
        kalman_vel_z = x1 / math.sqrt(max(p11, 1e-12))

        if active_trade is not None:
            t = active_trade
            if trailing_tp:
                if not t["be_triggered"]:
                    fav_pips = (m5_high[i] - t["entry"]) / 0.0001 if t["dir"] == 1 else (t["entry"] - m5_low[i]) / 0.0001
                    if fav_pips >= t["r_pips"]:
                        t["be_triggered"] = True
                        t["sl"] = t["entry"] + (0.0001 if t["dir"] == 1 else -0.0001)

                if not t["partial_taken"]:
                    fav_pips = (m5_high[i] - t["entry"]) / 0.0001 if t["dir"] == 1 else (t["entry"] - m5_low[i]) / 0.0001
                    if fav_pips >= 1.5 * t["r_pips"]:
                        t["partial_taken"] = True
                        half_lots = round(t["remaining_lots"] / 2.0, 2)
                        if half_lots >= 0.01:
                            partial_pips = 1.5 * t["r_pips"]
                            partial_pnl = (partial_pips * half_lots * 9.17) - (half_lots * 6.0)
                            state_b["balance"] += partial_pnl
                            t["accumulated_pnl"] += partial_pnl
                            t["remaining_lots"] = round(t["remaining_lots"] - half_lots, 2)
                            t["sl"] = t["entry"] + (0.5 * t["r_pips"] * 0.0001 if t["dir"] == 1 else -0.5 * t["r_pips"] * 0.0001)

                if t["partial_taken"]:
                    trail_dist = 0.7 * m5_atr14[i]
                    if t["dir"] == 1:
                        c_sl = m5_high[i] - trail_dist
                        if c_sl > t["sl"]:
                            t["sl"] = c_sl
                    else:
                        c_sl = m5_low[i] + trail_dist
                        if c_sl < t["sl"]:
                            t["sl"] = c_sl

                hit_exit = False
                exit_price = 0.0
                if t["dir"] == 1:
                    if m5_low[i] <= t["sl"]:
                        hit_exit = True
                        exit_price = t["sl"]
                    elif not t["partial_taken"] and m5_high[i] >= t["tp"]:
                        hit_exit = True
                        exit_price = t["tp"]
                else:
                    if m5_high[i] >= t["sl"]:
                        hit_exit = True
                        exit_price = t["sl"]
                    elif not t["partial_taken"] and m5_low[i] <= t["tp"]:
                        hit_exit = True
                        exit_price = t["tp"]

                if hit_exit:
                    pips = (exit_price - t["entry"]) / 0.0001 if t["dir"] == 1 else (t["entry"] - exit_price) / 0.0001
                    rem_pnl = (pips * t["remaining_lots"] * 9.17) - (t["remaining_lots"] * 6.0)
                    total_pnl = t["accumulated_pnl"] + rem_pnl
                    state_b["balance"] += rem_pnl
                    is_win = 1 if total_pnl > 0 else 0
                    state_b["trades"].append({"pnl": total_pnl, "win": is_win, "r": total_pnl / max(1.0, t["risk_eur"])})
                    state_b["total_trades"] += 1
                    if is_win:
                        state_b["wins"] += 1
                    active_trade = None
            else:
                if not t["be_triggered"]:
                    fav_pips = (m5_high[i] - t["entry"]) / 0.0001 if t["dir"] == 1 else (t["entry"] - m5_low[i]) / 0.0001
                    if fav_pips >= t["r_pips"]:
                        t["be_triggered"] = True
                        t["sl"] = t["entry"] + (0.0001 if t["dir"] == 1 else -0.0001)

                hit_sl, hit_tp = False, False
                exit_price = 0.0
                if t["dir"] == 1:
                    if m5_low[i] <= t["sl"]:
                        hit_sl = True
                        exit_price = t["sl"]
                    elif m5_high[i] >= t["tp"]:
                        hit_tp = True
                        exit_price = t["tp"]
                else:
                    if m5_high[i] >= t["sl"]:
                        hit_sl = True
                        exit_price = t["sl"]
                    elif m5_low[i] <= t["tp"]:
                        hit_tp = True
                        exit_price = t["tp"]

                if hit_sl or hit_tp:
                    pips = (exit_price - t["entry"]) / 0.0001 if t["dir"] == 1 else (t["entry"] - exit_price) / 0.0001
                    pnl_eur = (pips * t["lots"] * 9.17) - (t["lots"] * 6.0)
                    state_b["balance"] += pnl_eur
                    is_win = 1 if pnl_eur > 0 else 0
                    state_b["trades"].append({"pnl": pnl_eur, "win": is_win, "r": pnl_eur / max(1.0, t["risk_eur"])})
                    state_b["total_trades"] += 1
                    if is_win:
                        state_b["wins"] += 1
                    active_trade = None

        if i < test_start_idx:
            continue

        hour_utc = (m5_time[i] % 86400) / 3600.0
        if hour_utc < 7.0 or hour_utc >= 16.0:
            continue

        h1_idx = h1_idx_for_m5[i]
        h1_e20 = h1_ema20[h1_idx]
        h1_e50 = h1_ema50[h1_idx]
        h1_a = h1_adx[h1_idx]
        if h1_a < 22.0:
            continue

        trend_bull = h1_e20 > h1_e50
        trend_bear = h1_e20 < h1_e50
        if not trend_bull and not trend_bear:
            continue

        atr = m5_atr14[i]
        atr_med = m5_atr_med[i]
        if atr < 0.70 * atr_med or atr > 2.50 * atr_med or atr <= 1e-5:
            continue

        fdi_val = calc_sevcik_fdi(m5_close[: i + 1], 30)
        if fdi_val > 1.48:
            continue

        pe_val = calc_permutation_entropy(m5_close[: i + 1], 30, 3)
        if pe_val > 0.94:
            continue

        last_low = m5_low[i]
        last_high = m5_high[i]
        last_close = m5_close[i]
        last_e20 = m5_ema20[i]

        direction = 0
        if trend_bull and last_low <= last_e20 and last_close > last_e20:
            direction = 1
        elif trend_bear and last_high >= last_e20 and last_close < last_e20:
            direction = -1

        if direction == 0:
            continue

        spread_pips = 1.2 * (1.0 + 0.3 * max(0.0, (atr / max(atr_med, 1e-5)) - 1.0))
        slippage_price = 0.00003
        exec_price = (last_close + (spread_pips * 0.0001) / 2.0 + slippage_price) if direction == 1 else (last_close - (spread_pips * 0.0001) / 2.0 - slippage_price)
        sl_buffer = 0.5 * atr
        base_sl = (min(last_low, last_e20) - sl_buffer) if direction == 1 else (max(last_high, last_e20) + sl_buffer)
        sl_dist = abs(exec_price - base_sl)
        if sl_dist < 0.00050:
            continue
        base_tp = (exec_price + 1.5 * sl_dist) if direction == 1 else (exec_price - 1.5 * sl_dist)
        r_pips = sl_dist / 0.0001

        if active_trade is None:
            hour_angle = 2.0 * math.pi * hour_utc / 24.0
            features_10d = np.array([[float(direction), atr * 10000.0, float(h1_a), float(fdi_val), float(kalman_vel_z), float(spread_pips * 10.0), 0.0, 0.0, math.sin(hour_angle), math.cos(hour_angle)]], dtype=np.float32)
            ort_out = onnx_sess.run(None, {input_name: features_10d})
            meta_prob = float(ort_out[1][0][1]) if len(ort_out) > 1 and isinstance(ort_out[1], np.ndarray) else (float(ort_out[1][0].get(1, 0.5)) if isinstance(ort_out[1], list) else float(ort_out[0][0]))

            if meta_prob < min_p:
                continue

            acc_state = AccountState(equity_eur=state_b["balance"], free_margin_eur=state_b["balance"], used_margin_eur=0.0, daily_dd_pct=0.0, open_positions_count=0, open_correlated_risk_eur=0.0)
            sizing_res = sizer.calculate_eurusd_lots(account=acc_state, eurusd_ask=exec_price, sl_distance_price=sl_dist, wins=state_b["wins"], total_trades=state_b["total_trades"], avg_win_pips=20.0, avg_loss_pips=13.0, expected_exec_cost_pips=spread_pips, atr_current=atr, atr_median=atr_med, meta_prob=meta_prob)

            if sizing_res.lots >= 0.01:
                active_trade = {
                    "time": m5_time[i], "dir": direction, "entry": exec_price, "sl": base_sl, "tp": base_tp,
                    "lots": sizing_res.lots, "initial_lots": sizing_res.lots, "remaining_lots": sizing_res.lots,
                    "risk_eur": sizing_res.risk_eur, "r_pips": r_pips, "be_triggered": False,
                    "partial_taken": False, "accumulated_pnl": 0.0
                }

    df_tr = pd.DataFrame(state_b["trades"])
    if len(df_tr) == 0:
        return None
    n = len(df_tr)
    wr = df_tr["win"].mean() * 100
    pnl = state_b["balance"] - 1000.0
    gp = df_tr[df_tr["pnl"] > 0]["pnl"].sum()
    gl = abs(df_tr[df_tr["pnl"] < 0]["pnl"].sum())
    pf = gp / gl if gl > 0 else 999.0
    cum_p = np.cumsum(df_tr["pnl"].values)
    eq = 1000.0 + np.insert(cum_p, 0, 0.0)
    pks = np.maximum.accumulate(eq)
    dd = float(np.max((pks - eq) / pks) * 100.0)
    exp_r = df_tr["r"].mean()
    return {"n": n, "wr": wr, "pnl": pnl, "pf": pf, "dd": dd, "r": exp_r}

print("=== COMPARATIVO DE CALIBRACAO DE PROBABILIDADE META-GUARD ===")
for p in [0.39, 0.40, 0.41, 0.42, 0.43, 0.44]:
    for trail in [False, True]:
        t_label = "Partial+Trail" if trail else "Classic BE"
        res = test_cutoff(min_p=p, trailing_tp=trail)
        if res:
            print(f"Cutoff={p:.2f} | {t_label:<13} -> N={res['n']:2d} | WR={res['wr']:4.1f}% | PnL={res['pnl']:+6.2f} EUR | PF={res['pf']:4.2f} | MaxDD={res['dd']:4.2f}% | Exp={res['r']:+5.2f}R")

def test_be_comparison():
    print("\n=== IMPACTO DO BREAKEVEN NA EXPECTATIVA MATEMATICA ===")
    for be in [1.0, 1.2, 99.0]:
        for p in [0.40, 0.41, 0.42, 0.43, 0.44]:
            be_str = "None (Full SL/TP)" if be == 99.0 else f"BE @ {be}R"
            # run test
            res = test_management(be_trigger_r=be, tp_multiple=1.5, min_prob=p)
            if res:
                print(f"Prob >= {p:.2f} | {be_str:<18} -> N={res['n']:2d} | WR={res['wr']:4.1f}% | PnL={res['pnl']:+6.2f} EUR | PF={res['pf']:4.2f} | MaxDD={res['dd']:4.2f}% | Exp={res['r']:+5.2f}R")

def test_management(be_trigger_r=99.0, tp_multiple=1.5, min_prob=0.41):
    state = {"balance": 1000.0, "trades": [], "wins": 18, "total_trades": 32}
    active_trade = None
    x0, x1 = m5_close[0], 0.0
    p00, p01, p10, p11 = 1.0, 0.0, 0.0, 1.0

    for i in range(1, len(m5_close)):
        atr_norm = max(m5_atr14[i], 1e-5)
        pred_x0 = x0 + x1
        pred_x1 = x1
        n_p00 = p00 + p10 + p01 + p11 + 1e-6 * atr_norm
        n_p01 = p01 + p11
        n_p10 = p10 + p11
        n_p11 = p11 + 5e-7 * atr_norm
        y = m5_close[i] - pred_x0
        S = n_p00 + 1e-3 * atr_norm
        K0 = n_p00 / S
        K1 = n_p10 / S
        x0 = pred_x0 + K0 * y
        x1 = pred_x1 + K1 * y
        p00 = (1.0 - K0) * n_p00
        p01 = (1.0 - K0) * n_p01
        p10 = n_p10 - K1 * n_p00
        p11 = n_p11 - K1 * n_p01
        kalman_vel_z = x1 / math.sqrt(max(p11, 1e-12))

        if active_trade is not None:
            t = active_trade
            if be_trigger_r < 10.0 and not t["be_triggered"]:
                fav_pips = (m5_high[i] - t["entry"]) / 0.0001 if t["dir"] == 1 else (t["entry"] - m5_low[i]) / 0.0001
                if fav_pips >= be_trigger_r * t["r_pips"]:
                    t["be_triggered"] = True
                    t["sl"] = t["entry"] + (0.0001 if t["dir"] == 1 else -0.0001)

            hit_sl, hit_tp = False, False
            exit_price = 0.0
            if t["dir"] == 1:
                if m5_low[i] <= t["sl"]:
                    hit_sl = True
                    exit_price = t["sl"]
                elif m5_high[i] >= t["tp"]:
                    hit_tp = True
                    exit_price = t["tp"]
            else:
                if m5_high[i] >= t["sl"]:
                    hit_sl = True
                    exit_price = t["sl"]
                elif m5_low[i] <= t["tp"]:
                    hit_tp = True
                    exit_price = t["tp"]

            if hit_sl or hit_tp:
                pips = (exit_price - t["entry"]) / 0.0001 if t["dir"] == 1 else (t["entry"] - exit_price) / 0.0001
                pnl_eur = (pips * t["lots"] * 9.17) - (t["lots"] * 6.0)
                state["balance"] += pnl_eur
                is_win = 1 if pnl_eur > 0 else 0
                state["trades"].append({"pnl": pnl_eur, "win": is_win, "r": pnl_eur / max(1.0, t["risk_eur"])})
                state["total_trades"] += 1
                if is_win:
                    state["wins"] += 1
                active_trade = None

        if i < test_start_idx:
            continue

        hour_utc = (m5_time[i] % 86400) / 3600.0
        if hour_utc < 7.0 or hour_utc >= 16.0:
            continue

        h1_idx = h1_idx_for_m5[i]
        h1_e20 = h1_ema20[h1_idx]
        h1_e50 = h1_ema50[h1_idx]
        h1_a = h1_adx[h1_idx]
        if h1_a < 22.0:
            continue

        trend_bull = h1_e20 > h1_e50
        trend_bear = h1_e20 < h1_e50
        if not trend_bull and not trend_bear:
            continue

        atr = m5_atr14[i]
        atr_med = m5_atr_med[i]
        if atr < 0.70 * atr_med or atr > 2.50 * atr_med or atr <= 1e-5:
            continue

        fdi_val = calc_sevcik_fdi(m5_close[: i + 1], 30)
        if fdi_val > 1.48:
            continue

        pe_val = calc_permutation_entropy(m5_close[: i + 1], 30, 3)
        if pe_val > 0.94:
            continue

        last_low = m5_low[i]
        last_high = m5_high[i]
        last_close = m5_close[i]
        last_e20 = m5_ema20[i]

        direction = 0
        if trend_bull and last_low <= last_e20 and last_close > last_e20:
            direction = 1
        elif trend_bear and last_high >= last_e20 and last_close < last_e20:
            direction = -1

        if direction == 0:
            continue

        spread_pips = 1.2 * (1.0 + 0.3 * max(0.0, (atr / max(atr_med, 1e-5)) - 1.0))
        slippage_price = 0.00003
        exec_price = (last_close + (spread_pips * 0.0001) / 2.0 + slippage_price) if direction == 1 else (last_close - (spread_pips * 0.0001) / 2.0 - slippage_price)
        sl_buffer = 0.5 * atr
        base_sl = (min(last_low, last_e20) - sl_buffer) if direction == 1 else (max(last_high, last_e20) + sl_buffer)
        sl_dist = abs(exec_price - base_sl)
        if sl_dist < 0.00050:
            continue
        base_tp = (exec_price + tp_multiple * sl_dist) if direction == 1 else (exec_price - tp_multiple * sl_dist)
        r_pips = sl_dist / 0.0001

        if active_trade is None:
            hour_angle = 2.0 * math.pi * hour_utc / 24.0
            features_10d = np.array([[float(direction), atr * 10000.0, float(h1_a), float(fdi_val), float(kalman_vel_z), float(spread_pips * 10.0), 0.0, 0.0, math.sin(hour_angle), math.cos(hour_angle)]], dtype=np.float32)
            ort_out = onnx_sess.run(None, {input_name: features_10d})
            meta_prob = float(ort_out[1][0][1]) if len(ort_out) > 1 and isinstance(ort_out[1], np.ndarray) else (float(ort_out[1][0].get(1, 0.5)) if isinstance(ort_out[1], list) else float(ort_out[0][0]))

            if meta_prob < min_prob:
                continue

            acc_state = AccountState(equity_eur=state["balance"], free_margin_eur=state["balance"], used_margin_eur=0.0, daily_dd_pct=0.0, open_positions_count=0, open_correlated_risk_eur=0.0)
            sizing_res = sizer.calculate_eurusd_lots(account=acc_state, eurusd_ask=exec_price, sl_distance_price=sl_dist, wins=state["wins"], total_trades=state["total_trades"], avg_win_pips=20.0, avg_loss_pips=13.0, expected_exec_cost_pips=spread_pips, atr_current=atr, atr_median=atr_med, meta_prob=meta_prob)

            if sizing_res.lots >= 0.01:
                active_trade = {
                    "time": m5_time[i], "dir": direction, "entry": exec_price, "sl": base_sl, "tp": base_tp,
                    "lots": sizing_res.lots, "risk_eur": sizing_res.risk_eur, "r_pips": r_pips, "be_triggered": False
                }

    df_tr = pd.DataFrame(state["trades"])
    if len(df_tr) == 0:
        return None
    n = len(df_tr)
    wr = df_tr["win"].mean() * 100
    pnl = state["balance"] - 1000.0
    gp = df_tr[df_tr["pnl"] > 0]["pnl"].sum()
    gl = abs(df_tr[df_tr["pnl"] < 0]["pnl"].sum())
    pf = gp / gl if gl > 0 else 999.0
    cum_p = np.cumsum(df_tr["pnl"].values)
    eq = 1000.0 + np.insert(cum_p, 0, 0.0)
    pks = np.maximum.accumulate(eq)
    dd = float(np.max((pks - eq) / pks) * 100.0)
    exp_r = df_tr["r"].mean()
    return {"n": n, "wr": wr, "pnl": pnl, "pf": pf, "dd": dd, "r": exp_r}

test_be_comparison()

def test_partial_at_tp1():
    print("\n=== TESTE DE PARTIAL TAKE PROFIT EM 1.5R (RUNNER TRAIL) ===")
    for runner_tp in [2.0, 2.5, 3.0]:
        for p in [0.40, 0.41, 0.42]:
            res = simulate_tp1_runner(min_prob=p, tp1_r=1.5, runner_tp_r=runner_tp)
            if res:
                print(f"Prob >= {p:.2f} | TP1=1.5R, Runner={runner_tp:.1f}R -> N={res['n']:2d} | WR={res['wr']:4.1f}% | PnL={res['pnl']:+6.2f} EUR | PF={res['pf']:4.2f} | MaxDD={res['dd']:4.2f}% | Exp={res['r']:+5.2f}R")

def simulate_tp1_runner(min_prob=0.41, tp1_r=1.5, runner_tp_r=2.5):
    state = {"balance": 1000.0, "trades": [], "wins": 18, "total_trades": 32}
    active_trade = None
    x0, x1 = m5_close[0], 0.0
    p00, p01, p10, p11 = 1.0, 0.0, 0.0, 1.0

    for i in range(1, len(m5_close)):
        atr_norm = max(m5_atr14[i], 1e-5)
        pred_x0 = x0 + x1
        pred_x1 = x1
        n_p00 = p00 + p10 + p01 + p11 + 1e-6 * atr_norm
        n_p01 = p01 + p11
        n_p10 = p10 + p11
        n_p11 = p11 + 5e-7 * atr_norm
        y = m5_close[i] - pred_x0
        S = n_p00 + 1e-3 * atr_norm
        K0 = n_p00 / S
        K1 = n_p10 / S
        x0 = pred_x0 + K0 * y
        x1 = pred_x1 + K1 * y
        p00 = (1.0 - K0) * n_p00
        p01 = (1.0 - K0) * n_p01
        p10 = n_p10 - K1 * n_p00
        p11 = n_p11 - K1 * n_p01
        kalman_vel_z = x1 / math.sqrt(max(p11, 1e-12))

        if active_trade is not None:
            t = active_trade
            # No premature BE! Only check TP1
            if not t["partial_taken"]:
                fav_pips = (m5_high[i] - t["entry"]) / 0.0001 if t["dir"] == 1 else (t["entry"] - m5_low[i]) / 0.0001
                if fav_pips >= tp1_r * t["r_pips"]:
                    t["partial_taken"] = True
                    half_lots = round(t["remaining_lots"] / 2.0, 2)
                    if half_lots >= 0.01:
                        partial_pips = tp1_r * t["r_pips"]
                        partial_pnl = (partial_pips * half_lots * 9.17) - (half_lots * 6.0)
                        state["balance"] += partial_pnl
                        t["accumulated_pnl"] += partial_pnl
                        t["remaining_lots"] = round(t["remaining_lots"] - half_lots, 2)
                        # Lock SL to +0.2R (guaranteed win on runner)
                        t["sl"] = t["entry"] + (0.2 * t["r_pips"] * 0.0001 if t["dir"] == 1 else -0.2 * t["r_pips"] * 0.0001)

            # Trail runner if partial taken
            if t["partial_taken"]:
                trail_dist = 0.8 * m5_atr14[i]
                if t["dir"] == 1:
                    c_sl = m5_high[i] - trail_dist
                    if c_sl > t["sl"]:
                        t["sl"] = c_sl
                else:
                    c_sl = m5_low[i] + trail_dist
                    if c_sl < t["sl"]:
                        t["sl"] = c_sl

            hit_exit = False
            exit_price = 0.0
            runner_tp = (t["entry"] + runner_tp_r * t["r_pips"] * 0.0001) if t["dir"] == 1 else (t["entry"] - runner_tp_r * t["r_pips"] * 0.0001)
            
            if t["dir"] == 1:
                if m5_low[i] <= t["sl"]:
                    hit_exit = True
                    exit_price = t["sl"]
                elif t["partial_taken"] and m5_high[i] >= runner_tp:
                    hit_exit = True
                    exit_price = runner_tp
                elif not t["partial_taken"] and m5_high[i] >= (t["entry"] + tp1_r * t["r_pips"] * 0.0001):
                    # Will be caught in next partial check, but if sharp gap
                    pass
            else:
                if m5_high[i] >= t["sl"]:
                    hit_exit = True
                    exit_price = t["sl"]
                elif t["partial_taken"] and m5_low[i] <= runner_tp:
                    hit_exit = True
                    exit_price = runner_tp

            if hit_exit:
                pips = (exit_price - t["entry"]) / 0.0001 if t["dir"] == 1 else (t["entry"] - exit_price) / 0.0001
                rem_pnl = (pips * t["remaining_lots"] * 9.17) - (t["remaining_lots"] * 6.0)
                total_pnl = t["accumulated_pnl"] + rem_pnl
                state["balance"] += rem_pnl
                is_win = 1 if total_pnl > 0 else 0
                state["trades"].append({"pnl": total_pnl, "win": is_win, "r": total_pnl / max(1.0, t["risk_eur"])})
                state["total_trades"] += 1
                if is_win:
                    state["wins"] += 1
                active_trade = None

        if i < test_start_idx:
            continue

        hour_utc = (m5_time[i] % 86400) / 3600.0
        if hour_utc < 7.0 or hour_utc >= 16.0:
            continue

        h1_idx = h1_idx_for_m5[i]
        h1_e20 = h1_ema20[h1_idx]
        h1_e50 = h1_ema50[h1_idx]
        h1_a = h1_adx[h1_idx]
        if h1_a < 22.0:
            continue

        trend_bull = h1_e20 > h1_e50
        trend_bear = h1_e20 < h1_e50
        if not trend_bull and not trend_bear:
            continue

        atr = m5_atr14[i]
        atr_med = m5_atr_med[i]
        if atr < 0.70 * atr_med or atr > 2.50 * atr_med or atr <= 1e-5:
            continue

        fdi_val = calc_sevcik_fdi(m5_close[: i + 1], 30)
        if fdi_val > 1.48:
            continue

        pe_val = calc_permutation_entropy(m5_close[: i + 1], 30, 3)
        if pe_val > 0.94:
            continue

        last_low = m5_low[i]
        last_high = m5_high[i]
        last_close = m5_close[i]
        last_e20 = m5_ema20[i]

        direction = 0
        if trend_bull and last_low <= last_e20 and last_close > last_e20:
            direction = 1
        elif trend_bear and last_high >= last_e20 and last_close < last_e20:
            direction = -1

        if direction == 0:
            continue

        spread_pips = 1.2 * (1.0 + 0.3 * max(0.0, (atr / max(atr_med, 1e-5)) - 1.0))
        slippage_price = 0.00003
        exec_price = (last_close + (spread_pips * 0.0001) / 2.0 + slippage_price) if direction == 1 else (last_close - (spread_pips * 0.0001) / 2.0 - slippage_price)
        sl_buffer = 0.5 * atr
        base_sl = (min(last_low, last_e20) - sl_buffer) if direction == 1 else (max(last_high, last_e20) + sl_buffer)
        sl_dist = abs(exec_price - base_sl)
        if sl_dist < 0.00050:
            continue
        r_pips = sl_dist / 0.0001

        if active_trade is None:
            hour_angle = 2.0 * math.pi * hour_utc / 24.0
            features_10d = np.array([[float(direction), atr * 10000.0, float(h1_a), float(fdi_val), float(kalman_vel_z), float(spread_pips * 10.0), 0.0, 0.0, math.sin(hour_angle), math.cos(hour_angle)]], dtype=np.float32)
            ort_out = onnx_sess.run(None, {input_name: features_10d})
            meta_prob = float(ort_out[1][0][1]) if len(ort_out) > 1 and isinstance(ort_out[1], np.ndarray) else (float(ort_out[1][0].get(1, 0.5)) if isinstance(ort_out[1], list) else float(ort_out[0][0]))

            if meta_prob < min_prob:
                continue

            acc_state = AccountState(equity_eur=state["balance"], free_margin_eur=state["balance"], used_margin_eur=0.0, daily_dd_pct=0.0, open_positions_count=0, open_correlated_risk_eur=0.0)
            sizing_res = sizer.calculate_eurusd_lots(account=acc_state, eurusd_ask=exec_price, sl_distance_price=sl_dist, wins=state["wins"], total_trades=state["total_trades"], avg_win_pips=20.0, avg_loss_pips=13.0, expected_exec_cost_pips=spread_pips, atr_current=atr, atr_median=atr_med, meta_prob=meta_prob)

            if sizing_res.lots >= 0.01:
                active_trade = {
                    "time": m5_time[i], "dir": direction, "entry": exec_price, "sl": base_sl,
                    "lots": sizing_res.lots, "initial_lots": sizing_res.lots, "remaining_lots": sizing_res.lots,
                    "risk_eur": sizing_res.risk_eur, "r_pips": r_pips, "be_triggered": False,
                    "partial_taken": False, "accumulated_pnl": 0.0
                }

    df_tr = pd.DataFrame(state["trades"])
    if len(df_tr) == 0:
        return None
    n = len(df_tr)
    wr = df_tr["win"].mean() * 100
    pnl = state["balance"] - 1000.0
    gp = df_tr[df_tr["pnl"] > 0]["pnl"].sum()
    gl = abs(df_tr[df_tr["pnl"] < 0]["pnl"].sum())
    pf = gp / gl if gl > 0 else 999.0
    cum_p = np.cumsum(df_tr["pnl"].values)
    eq = 1000.0 + np.insert(cum_p, 0, 0.0)
    pks = np.maximum.accumulate(eq)
    dd = float(np.max((pks - eq) / pks) * 100.0)
    exp_r = df_tr["r"].mean()
    return {"n": n, "wr": wr, "pnl": pnl, "pf": pf, "dd": dd, "r": exp_r}

test_partial_at_tp1()

def test_all_triggers():
    print("\n=== TESTE COM TODOS OS GATILHOS (PULLBACK + BOS + SWEEP) ===")
    for p in [0.41, 0.42, 0.43]:
        for bos in [True, False]:
            for sw in [True, False]:
                res = simulate_multitrigger(min_prob=p, enable_bos=bos, enable_sweep=sw)
                if res and res['n'] >= 5:
                    label = f"BOS={bos}, Sweep={sw}"
                    print(f"Prob >= {p:.2f} | {label:<18} -> N={res['n']:2d} | WR={res['wr']:4.1f}% | PnL={res['pnl']:+6.2f} EUR | PF={res['pf']:4.2f} | MaxDD={res['dd']:4.2f}% | Exp={res['r']:+5.2f}R")

def simulate_multitrigger(min_prob=0.42, enable_bos=True, enable_sweep=True):
    state = {"balance": 1000.0, "trades": [], "wins": 18, "total_trades": 32}
    active_trade = None
    x0, x1 = m5_close[0], 0.0
    p00, p01, p10, p11 = 1.0, 0.0, 0.0, 1.0

    for i in range(1, len(m5_close)):
        atr_norm = max(m5_atr14[i], 1e-5)
        pred_x0 = x0 + x1
        pred_x1 = x1
        n_p00 = p00 + p10 + p01 + p11 + 1e-6 * atr_norm
        n_p01 = p01 + p11
        n_p10 = p10 + p11
        n_p11 = p11 + 5e-7 * atr_norm
        y = m5_close[i] - pred_x0
        S = n_p00 + 1e-3 * atr_norm
        K0 = n_p00 / S
        K1 = n_p10 / S
        x0 = pred_x0 + K0 * y
        x1 = pred_x1 + K1 * y
        p00 = (1.0 - K0) * n_p00
        p01 = (1.0 - K0) * n_p01
        p10 = n_p10 - K1 * n_p00
        p11 = n_p11 - K1 * n_p01
        kalman_vel_z = x1 / math.sqrt(max(p11, 1e-12))

        if active_trade is not None:
            t = active_trade
            hit_sl, hit_tp = False, False
            exit_price = 0.0
            if t["dir"] == 1:
                if m5_low[i] <= t["sl"]: hit_sl = True; exit_price = t["sl"]
                elif m5_high[i] >= t["tp"]: hit_tp = True; exit_price = t["tp"]
            else:
                if m5_high[i] >= t["sl"]: hit_sl = True; exit_price = t["sl"]
                elif m5_low[i] <= t["tp"]: hit_tp = True; exit_price = t["tp"]

            if hit_sl or hit_tp:
                pips = (exit_price - t["entry"]) / 0.0001 if t["dir"] == 1 else (t["entry"] - exit_price) / 0.0001
                pnl_eur = (pips * t["lots"] * 9.17) - (t["lots"] * 6.0)
                state["balance"] += pnl_eur
                is_win = 1 if pnl_eur > 0 else 0
                state["trades"].append({"pnl": pnl_eur, "win": is_win, "r": pnl_eur / max(1.0, t["risk_eur"]), "setup": t["setup_name"]})
                state["total_trades"] += 1
                if is_win: state["wins"] += 1
                active_trade = None

        if i < test_start_idx:
            continue

        hour_utc = (m5_time[i] % 86400) / 3600.0
        if hour_utc < 7.0 or hour_utc >= 16.0:
            continue

        h1_idx = h1_idx_for_m5[i]
        h1_e20 = h1_ema20[h1_idx]
        h1_e50 = h1_ema50[h1_idx]
        h1_a = h1_adx[h1_idx]
        if h1_a < 22.0:
            continue

        trend_bull = h1_e20 > h1_e50
        trend_bear = h1_e20 < h1_e50
        if not trend_bull and not trend_bear:
            continue

        atr = m5_atr14[i]
        atr_med = m5_atr_med[i]
        if atr < 0.70 * atr_med or atr > 2.50 * atr_med or atr <= 1e-5:
            continue

        fdi_val = calc_sevcik_fdi(m5_close[: i + 1], 30)
        if fdi_val > 1.48:
            continue

        pe_val = calc_permutation_entropy(m5_close[: i + 1], 30, 3)
        if pe_val > 0.94:
            continue

        last_low = m5_low[i]
        last_high = m5_high[i]
        last_close = m5_close[i]
        last_e20 = m5_ema20[i]

        direction = 0
        setup_name = ""

        # Pullback
        if trend_bull and last_low <= last_e20 and last_close > last_e20:
            direction = 1; setup_name = "EMA_PULLBACK"
        elif trend_bear and last_high >= last_e20 and last_close < last_e20:
            direction = -1; setup_name = "EMA_PULLBACK"

        # Sweep
        if direction == 0 and enable_sweep and i >= 25:
            min_20 = np.min(m5_low[i - 20 : i])
            max_20 = np.max(m5_high[i - 20 : i])
            if trend_bull and last_low < min_20 and last_close > min_20 and kalman_vel_z > 0.3:
                direction = 1; setup_name = "LIQUIDITY_SWEEP"
            elif trend_bear and last_high > max_20 and last_close < max_20 and kalman_vel_z < -0.3:
                direction = -1; setup_name = "LIQUIDITY_SWEEP"

        # BOS
        if direction == 0 and enable_bos and i >= 20:
            prior_swing_high = np.max(m5_high[i - 15 : i])
            prior_swing_low = np.min(m5_low[i - 15 : i])
            if trend_bull and last_close > prior_swing_high and kalman_vel_z > 0.8:
                direction = 1; setup_name = "BOS_BREAKOUT"
            elif trend_bear and last_close < prior_swing_low and kalman_vel_z < -0.8:
                direction = -1; setup_name = "BOS_BREAKOUT"

        if direction == 0:
            continue

        spread_pips = 1.2 * (1.0 + 0.3 * max(0.0, (atr / max(atr_med, 1e-5)) - 1.0))
        slippage_price = 0.00003
        exec_price = (last_close + (spread_pips * 0.0001) / 2.0 + slippage_price) if direction == 1 else (last_close - (spread_pips * 0.0001) / 2.0 - slippage_price)
        sl_buffer = 0.5 * atr
        base_sl = (min(last_low, last_e20) - sl_buffer) if direction == 1 else (max(last_high, last_e20) + sl_buffer)
        sl_dist = abs(exec_price - base_sl)
        if sl_dist < 0.00050:
            continue
        base_tp = (exec_price + 1.5 * sl_dist) if direction == 1 else (exec_price - 1.5 * sl_dist)
        r_pips = sl_dist / 0.0001

        if active_trade is None:
            hour_angle = 2.0 * math.pi * hour_utc / 24.0
            features_10d = np.array([[float(direction), atr * 10000.0, float(h1_a), float(fdi_val), float(kalman_vel_z), float(spread_pips * 10.0), 0.0, 0.0, math.sin(hour_angle), math.cos(hour_angle)]], dtype=np.float32)
            ort_out = onnx_sess.run(None, {input_name: features_10d})
            meta_prob = float(ort_out[1][0][1]) if len(ort_out) > 1 and isinstance(ort_out[1], np.ndarray) else (float(ort_out[1][0].get(1, 0.5)) if isinstance(ort_out[1], list) else float(ort_out[0][0]))

            if meta_prob < min_prob:
                continue

            acc_state = AccountState(equity_eur=state["balance"], free_margin_eur=state["balance"], used_margin_eur=0.0, daily_dd_pct=0.0, open_positions_count=0, open_correlated_risk_eur=0.0)
            sizing_res = sizer.calculate_eurusd_lots(account=acc_state, eurusd_ask=exec_price, sl_distance_price=sl_dist, wins=state["wins"], total_trades=state["total_trades"], avg_win_pips=20.0, avg_loss_pips=13.0, expected_exec_cost_pips=spread_pips, atr_current=atr, atr_median=atr_med, meta_prob=meta_prob)

            if sizing_res.lots >= 0.01:
                active_trade = {
                    "time": m5_time[i], "dir": direction, "entry": exec_price, "sl": base_sl, "tp": base_tp,
                    "lots": sizing_res.lots, "risk_eur": sizing_res.risk_eur, "r_pips": r_pips, "setup_name": setup_name
                }

    df_tr = pd.DataFrame(state["trades"])
    if len(df_tr) == 0: return None
    n = len(df_tr)
    wr = df_tr["win"].mean() * 100
    pnl = state["balance"] - 1000.0
    gp = df_tr[df_tr["pnl"] > 0]["pnl"].sum()
    gl = abs(df_tr[df_tr["pnl"] < 0]["pnl"].sum())
    pf = gp / gl if gl > 0 else 999.0
    cum_p = np.cumsum(df_tr["pnl"].values)
    eq = 1000.0 + np.insert(cum_p, 0, 0.0)
    pks = np.maximum.accumulate(eq)
    dd = float(np.max((pks - eq) / pks) * 100.0)
    exp_r = df_tr["r"].mean()
    return {"n": n, "wr": wr, "pnl": pnl, "pf": pf, "dd": dd, "r": exp_r}

test_all_triggers()
