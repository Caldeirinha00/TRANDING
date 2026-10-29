"""
scripts/test_phase2_improvements.py
Simula as melhorias da Fase 2:
1. Trailing TP Parcial (50% a 1.5R, SL para +0.5R, trail de 50% a 0.7 ATR)
2. Filtro de Regime D1 (ATR Percentil >= P25)
3. Gatilhos Expandidos: Pullback EMA20 + Liquidity Sweep + Break of Structure (BOS)
Período: 2026-08-21 a 2026-10-21 | EURUSD M5 | Base: 1.000 EUR
"""

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

from core.config import CONFIG
from risk.quant_position_sizer import InstitutionalPositionSizer, AccountState, SizingResult
from scripts.backtest_v2_institutional import calc_ema, calc_atr, calc_adx, calc_sevcik_fdi, calc_permutation_entropy


def run_phase2_test():
    print("=" * 72)
    print(" AI_TRADING V2 - TESTE DE ALPHA DA FASE 2 (LAST 2 MONTHS)")
    print(" Pullback + Liquidity Sweep + BOS | Trailing TP Parcial | D1 Regime")
    print("=" * 72)

    if not mt5.initialize(path=CONFIG.MT5_PATH):
        if not mt5.initialize():
            print("[ERRO] Falha ao conectar ao MetaTrader 5.")
            return

    symbol = "EURUSD"
    start_warmup = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    start_test = datetime(2026, 8, 21, 0, 0, tzinfo=timezone.utc)
    end_test = datetime(2026, 10, 21, 23, 59, tzinfo=timezone.utc)

    rates_m5 = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_warmup, end_test)
    rates_h1 = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start_warmup, end_test)
    rates_d1 = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_D1, datetime(2025, 1, 1), end_test)
    mt5.shutdown()

    if rates_m5 is None or rates_h1 is None or rates_d1 is None:
        print("[ERRO] Falha ao extrair barras do MT5.")
        return

    df_m5 = pd.DataFrame(rates_m5)
    df_h1 = pd.DataFrame(rates_h1)
    df_d1 = pd.DataFrame(rates_d1)

    # 1. H1
    h1_close = df_h1["close"].values
    h1_high = df_h1["high"].values
    h1_low = df_h1["low"].values
    h1_time = df_h1["time"].values

    h1_ema20 = calc_ema(h1_close, 20)
    h1_ema50 = calc_ema(h1_close, 50)
    h1_adx = calc_adx(h1_high, h1_low, h1_close, 14)

    # 2. D1 ATR Percentile
    d1_close = df_d1["close"].values
    d1_high = df_d1["high"].values
    d1_low = df_d1["low"].values
    d1_time = df_d1["time"].values
    d1_atr = calc_atr(d1_high, d1_low, d1_close, 14)
    # Rolling percentile 100 barras
    d1_pct = pd.Series(d1_atr).rolling(100, min_periods=20).apply(
        lambda s: (s.iloc[-1] - s.min()) / max(1e-6, s.max() - s.min())
    ).values

    # 3. M5
    m5_close = df_m5["close"].values
    m5_open = df_m5["open"].values
    m5_high = df_m5["high"].values
    m5_low = df_m5["low"].values
    m5_time = df_m5["time"].values

    m5_ema20 = calc_ema(m5_close, 20)
    m5_atr14 = calc_atr(m5_high, m5_low, m5_close, 14)
    m5_atr_med = pd.Series(m5_atr14).rolling(50, min_periods=1).median().values

    # Mapeamentos
    h1_idx_for_m5 = np.clip(np.searchsorted(h1_time, m5_time, side="right") - 2, 0, len(h1_time) - 1)
    d1_idx_for_m5 = np.clip(np.searchsorted(d1_time, m5_time, side="right") - 2, 0, len(d1_time) - 1)

    onnx_path = os.path.join(PROJECT_ROOT, "models", "meta_guard_v2.onnx")
    onnx_sess = ort.InferenceSession(onnx_path)
    input_name = onnx_sess.get_inputs()[0].name

    sizer = InstitutionalPositionSizer(
        max_single_risk_pct=0.01,
        max_daily_dd_pct=0.03,
        esma_leverage=30.0,
        max_total_margin_usage_pct=0.45,
        max_concurrent_positions=3,
    )

    INITIAL_CAPITAL = 1000.0
    state_a = {"balance": INITIAL_CAPITAL, "trades": []}
    state_b = {
        "balance": INITIAL_CAPITAL,
        "trades": [],
        "vetoed_trades": [],
        "wins": 18,
        "total_trades": 32,
    }

    test_start_ts = int(start_test.timestamp())
    test_start_idx = np.searchsorted(m5_time, test_start_ts)

    x0, x1 = m5_close[0], 0.0
    p00, p01, p10, p11 = 1.0, 0.0, 0.0, 1.0

    active_trade_a = None
    active_trade_b = None

    for i in range(1, len(m5_close)):
        # Kalman 2D
        atr_norm = max(m5_atr14[i], 1e-5)
        q_pos = 1e-6 * atr_norm
        q_vel = 5e-7 * atr_norm
        r_obs = 1e-3 * atr_norm

        pred_x0 = x0 + x1
        pred_x1 = x1
        n_p00 = p00 + p10 + p01 + p11 + q_pos
        n_p01 = p01 + p11
        n_p10 = p10 + p11
        n_p11 = p11 + q_vel

        y = m5_close[i] - pred_x0
        S = n_p00 + r_obs
        K0 = n_p00 / S
        K1 = n_p10 / S

        x0 = pred_x0 + K0 * y
        x1 = pred_x1 + K1 * y
        p00 = (1.0 - K0) * n_p00
        p01 = (1.0 - K0) * n_p01
        p10 = n_p10 - K1 * n_p00
        p11 = n_p11 - K1 * n_p01

        kalman_vel_z = x1 / math.sqrt(max(p11, 1e-12))

        # Gestão de Trade Ativo Portfolio A (Baseline MQL5)
        if active_trade_a is not None:
            t = active_trade_a
            hit_sl = False
            hit_tp = False
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
                commission = t["lots"] * 6.0
                pnl_eur = (pips * t["lots"] * 9.17) - commission
                state_a["balance"] += pnl_eur
                state_a["trades"].append({
                    "entry_time": t["time"],
                    "exit_time": m5_time[i],
                    "dir": t["dir"],
                    "lots": t["lots"],
                    "pnl": pnl_eur,
                    "win": 1 if pnl_eur > 0 else 0,
                    "r": pnl_eur / max(1.0, t["risk_eur"]),
                })
                active_trade_a = None

        # Gestão de Trade Ativo Portfolio B (Com Trailing TP Parcial 50% @ 1.5R + Trail)
        if active_trade_b is not None:
            t = active_trade_b

            # 1. Breakeven em 1.0R
            if not t["be_triggered"]:
                fav_pips = (m5_high[i] - t["entry"]) / 0.0001 if t["dir"] == 1 else (t["entry"] - m5_low[i]) / 0.0001
                if fav_pips >= t["r_pips"]:
                    t["be_triggered"] = True
                    t["sl"] = t["entry"] + (0.0001 if t["dir"] == 1 else -0.0001)

            # 2. Take Profit Parcial (50%) ao atingir 1.5R
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
                        # Trava SL a +0.5R para os restantes 50%
                        t["sl"] = t["entry"] + (0.5 * t["r_pips"] * 0.0001 if t["dir"] == 1 else -0.5 * t["r_pips"] * 0.0001)

            # 3. Trailing Stop para o restante da posição (0.7 * ATR atrás do extremo)
            if t["partial_taken"]:
                trail_dist = 0.7 * m5_atr14[i]
                if t["dir"] == 1:
                    candidate_sl = m5_high[i] - trail_dist
                    if candidate_sl > t["sl"]:
                        t["sl"] = candidate_sl
                else:
                    candidate_sl = m5_low[i] + trail_dist
                    if candidate_sl < t["sl"]:
                        t["sl"] = candidate_sl

            # 4. Verificação de saída do restante
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
                commission = t["remaining_lots"] * 6.0
                rem_pnl = (pips * t["remaining_lots"] * 9.17) - commission
                total_pnl = t["accumulated_pnl"] + rem_pnl
                state_b["balance"] += rem_pnl
                is_win = 1 if total_pnl > 0 else 0
                state_b["trades"].append({
                    "entry_time": t["time"],
                    "exit_time": m5_time[i],
                    "dir": t["dir"],
                    "lots": t["initial_lots"],
                    "pnl": total_pnl,
                    "win": is_win,
                    "r": total_pnl / max(1.0, t["risk_eur"]),
                    "setup": t["setup_name"],
                })
                state_b["total_trades"] += 1
                if is_win:
                    state_b["wins"] += 1
                active_trade_b = None

        if i < test_start_idx:
            continue

        # Filtro de Sessão (07:00 a 16:00 UTC)
        hour_utc = (m5_time[i] % 86400) / 3600.0
        if hour_utc < 7.0 or hour_utc >= 16.0:
            continue

        # Filtro D1 Regime
        d1_idx = d1_idx_for_m5[i]
        if d1_pct[d1_idx] < 0.20:
            continue

        h1_idx = h1_idx_for_m5[i]
        h1_e20 = h1_ema20[h1_idx]
        h1_e50 = h1_ema50[h1_idx]
        h1_a = h1_adx[h1_idx]

        if h1_a < 20.0:
            continue

        trend_bull = h1_e20 > h1_e50
        trend_bear = h1_e20 < h1_e50
        if not trend_bull and not trend_bear:
            continue

        atr = m5_atr14[i]
        atr_med = m5_atr_med[i]
        if atr < 0.65 * atr_med or atr > 2.80 * atr_med or atr <= 1e-5:
            continue

        fdi_val = calc_sevcik_fdi(m5_close[: i + 1], 30)
        if fdi_val > 1.52:
            continue

        pe_val = calc_permutation_entropy(m5_close[: i + 1], 30, 3)
        if pe_val > 0.94:
            continue

        last_low = m5_low[i]
        last_high = m5_high[i]
        last_close = m5_close[i]
        last_e20 = m5_ema20[i]

        # Circuit Breaker: Máximo 2 perdas no mesmo dia
        current_day_str = datetime.fromtimestamp(m5_time[i], tz=timezone.utc).strftime("%Y-%m-%d")
        daily_losses = sum(1 for tr in state_b["trades"] if tr["win"] == 0 and datetime.fromtimestamp(tr["exit_time"], tz=timezone.utc).strftime("%Y-%m-%d") == current_day_str)
        if daily_losses >= 2:
            continue

        direction = 0
        setup_name = ""

        # GATILHO 1: Pullback à EMA(20) com confirmação cinemática de Kalman
        if trend_bull and last_low <= last_e20 and last_close > last_e20 and kalman_vel_z > 0.2:
            direction = 1
            setup_name = "EMA_PULLBACK"
        elif trend_bear and last_high >= last_e20 and last_close < last_e20 and kalman_vel_z < -0.2:
            direction = -1
            setup_name = "EMA_PULLBACK"

        # GATILHO 2: Liquidity Sweep de 20 barras (varredura com rejeição e alinhamento cinemático)
        if direction == 0 and i >= 25:
            prior_lows = m5_low[i - 20 : i]
            prior_highs = m5_high[i - 20 : i]
            min_20 = np.min(prior_lows)
            max_20 = np.max(prior_highs)

            if trend_bull and last_low < min_20 and last_close > min_20 and kalman_vel_z > 0.3:
                direction = 1
                setup_name = "LIQUIDITY_SWEEP"
            elif trend_bear and last_high > max_20 and last_close < max_20 and kalman_vel_z < -0.3:
                direction = -1
                setup_name = "LIQUIDITY_SWEEP"

        # GATILHO 3: Break of Structure (BOS) com impulso de Kalman
        if direction == 0 and i >= 20:
            prior_swing_high = np.max(m5_high[i - 15 : i])
            prior_swing_low = np.min(m5_low[i - 15 : i])

            if trend_bull and last_close > prior_swing_high and kalman_vel_z > 0.8:
                direction = 1
                setup_name = "BOS_BREAKOUT"
            elif trend_bear and last_close < prior_swing_low and kalman_vel_z < -0.8:
                direction = -1
                setup_name = "BOS_BREAKOUT"

        if direction == 0:
            continue

        spread_pips = 1.2 * (1.0 + 0.3 * max(0.0, (atr / max(atr_med, 1e-5)) - 1.0))
        slippage_price = 0.00003
        exec_price = (last_close + (spread_pips * 0.0001) / 2.0 + slippage_price) if direction == 1 else (last_close - (spread_pips * 0.0001) / 2.0 - slippage_price)

        sl_buffer = 0.5 * atr
        base_sl = (min(last_low, last_e20) - sl_buffer) if direction == 1 else (max(last_high, last_e20) + sl_buffer)
        sl_dist = abs(exec_price - base_sl)
        if sl_dist < 0.00050 or sl_dist > 0.0040:
            continue

        base_tp = (exec_price + 1.5 * sl_dist) if direction == 1 else (exec_price - 1.5 * sl_dist)
        r_pips = sl_dist / 0.0001

        # Portfolio A (Baseline)
        if active_trade_a is None:
            risk_eur = state_a["balance"] * 0.01
            lots_a = round(math.floor((risk_eur / (sl_dist * 100000.0)) / 0.01) * 0.01, 2)
            lots_a = max(0.01, min(lots_a, 0.50))
            active_trade_a = {
                "time": m5_time[i],
                "dir": direction,
                "entry": exec_price,
                "sl": base_sl,
                "tp": base_tp,
                "lots": lots_a,
                "risk_eur": risk_eur,
            }

        # Portfolio B (Hardened V2)
        if active_trade_b is None:
            t_sec = m5_time[i]
            hour_angle = 2.0 * math.pi * hour_utc / 24.0

            features_10d = np.array(
                [[
                    float(direction),
                    atr * 10000.0,
                    float(h1_a),
                    float(fdi_val),
                    float(kalman_vel_z),
                    float(spread_pips * 10.0),
                    0.0,
                    0.0,
                    math.sin(hour_angle),
                    math.cos(hour_angle),
                ]],
                dtype=np.float32,
            )
            ort_out = onnx_sess.run(None, {input_name: features_10d})
            if len(ort_out) > 1 and isinstance(ort_out[1], list):
                meta_prob = float(ort_out[1][0].get(1, 0.5))
            elif len(ort_out) > 1 and isinstance(ort_out[1], np.ndarray):
                meta_prob = float(ort_out[1][0][1])
            else:
                meta_prob = float(ort_out[0][0])

            cost_ratio = (spread_pips * 0.0001) / max(sl_dist, 1e-5)
            b_net = max(0.5, (1.5 - cost_ratio) / (1.0 + cost_ratio))
            be_threshold = 1.0 / (1.0 + b_net)
            fdi_pen = 0.15 * max(0.0, fdi_val - 1.42)
            p_crit = float(min(0.58, max(0.36, be_threshold + fdi_pen)))

            if meta_prob < p_crit:
                state_b["vetoed_trades"].append({"time": m5_time[i], "dir": direction, "meta_prob": meta_prob})
                continue

            acc_state = AccountState(
                equity_eur=state_b["balance"],
                free_margin_eur=state_b["balance"],
                used_margin_eur=0.0,
                daily_dd_pct=0.0,
                open_positions_count=0,
                open_correlated_risk_eur=0.0,
            )

            sizing_res: SizingResult = sizer.calculate_eurusd_lots(
                account=acc_state,
                eurusd_ask=exec_price,
                sl_distance_price=sl_dist,
                wins=state_b["wins"],
                total_trades=state_b["total_trades"],
                avg_win_pips=20.0,
                avg_loss_pips=13.0,
                expected_exec_cost_pips=spread_pips,
                atr_current=atr,
                atr_median=atr_med,
                meta_prob=meta_prob,
            )

            if sizing_res.lots >= 0.01:
                active_trade_b = {
                    "time": m5_time[i],
                    "dir": direction,
                    "entry": exec_price,
                    "sl": base_sl,
                    "tp": base_tp,
                    "initial_lots": sizing_res.lots,
                    "remaining_lots": sizing_res.lots,
                    "risk_eur": sizing_res.risk_eur,
                    "r_pips": r_pips,
                    "be_triggered": False,
                    "partial_taken": False,
                    "accumulated_pnl": 0.0,
                    "setup_name": setup_name,
                }

    print("=" * 72)
    print(" RESULTADOS DO TESTE DA FASE 2")
    print("=" * 72)

    df_b = pd.DataFrame(state_b["trades"])
    print(f"Total de Trades Executados V2: {len(df_b)}")
    if len(df_b) > 0:
        wins = int(df_b['win'].sum())
        total = len(df_b)
        win_rate = (wins / total) * 100.0
        pnl = state_b['balance'] - INITIAL_CAPITAL
        gross_p = float(df_b[df_b['pnl'] > 0]['pnl'].sum())
        gross_l = float(abs(df_b[df_b['pnl'] < 0]['pnl'].sum()))
        pf = (gross_p / gross_l) if gross_l > 0 else 999.0
        cum_pnl = np.cumsum(df_b['pnl'].values)
        peaks = np.maximum.accumulate(INITIAL_CAPITAL + np.insert(cum_pnl, 0, 0.0))
        eq = INITIAL_CAPITAL + np.insert(cum_pnl, 0, 0.0)
        dd = np.max((peaks - eq) / peaks) * 100.0

        print(f"Saldo Final:          {state_b['balance']:.2f} EUR")
        print(f"Lucro Líquido:        {pnl:+.2f} EUR ({pnl/10.0:+.2f}%)")
        print(f"Taxa de Acerto:       {win_rate:.1f}% ({wins}/{total})")
        print(f"Fator de Lucro (PF):  {pf:.2f}")
        print(f"Drawdown Máximo:      {dd:.2f}%")
        print(f"Expectativa Média:    {df_b['r'].mean():+.2f}R")
        print("-" * 72)
        print("DISTRIBUIÇÃO POR TIPO DE SETUP:")
        print(df_b.groupby('setup')['pnl'].agg(['count', 'sum', 'mean']))
        print("-" * 72)
        for idx, tr in enumerate(state_b["trades"], 1):
            dt_s = datetime.fromtimestamp(tr["entry_time"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            d_s = "BUY " if tr["dir"] == 1 else "SELL"
            r_s = "WIN " if tr["win"] == 1 else "LOSS"
            print(f"  {idx:02d}. [{dt_s}] {d_s} | {tr['setup']:<16} | Lotes: {tr['lots']:.2f} | PnL: {tr['pnl']:+6.2f} EUR | R: {tr['r']:+5.2f}R | {r_s}")


if __name__ == "__main__":
    run_phase2_test()
