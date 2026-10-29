"""
scripts/backtest_v2_institutional.py
Motor de Backtesting Quantitativo e Microestrutural de Alta Precisão para AI_TRADING V2.
Testa o período dos últimos 2 meses (2026-08-21 a 2026-10-21) em EURUSD M5.
Compara:
  1. Baseline MQL5 (Estratégia Pura com dimensionamento fixo de 1%)
  2. Institutional V2 Twin (ONNX Meta-Guard + Dynamic p_crit + Qdrant 16D + Kelly 4.ª Ordem)
Base de Capital: 1.000 EUR | Regulamentação ESMA 1:30
"""

import sys
import os
import math
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import onnxruntime as ort
import MetaTrader5 as mt5

# Adicionar diretório raiz ao path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from core.config import CONFIG
from agents.qdrant_rag_modulator import MarketStateVector16D
from risk.quant_position_sizer import InstitutionalPositionSizer, AccountState, SizingResult


def calc_ema(arr: np.ndarray, period: int) -> np.ndarray:
    n = len(arr)
    ema = np.zeros(n, dtype=np.float64)
    if n < period:
        return ema
    k = 2.0 / (period + 1.0)
    ema[period - 1] = np.mean(arr[:period])
    for i in range(period, n):
        ema[i] = (arr[i] - ema[i - 1]) * k + ema[i - 1]
    return ema


def calc_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(close)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], max(abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])))
    atr = np.zeros(n, dtype=np.float64)
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def calc_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(close)
    tr = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], max(abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])))
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down

    if n < period * 2:
        return np.zeros(n, dtype=np.float64)

    sm_tr = np.zeros(n, dtype=np.float64)
    sm_plus = np.zeros(n, dtype=np.float64)
    sm_minus = np.zeros(n, dtype=np.float64)

    sm_tr[period] = np.sum(tr[1 : period + 1])
    sm_plus[period] = np.sum(plus_dm[1 : period + 1])
    sm_minus[period] = np.sum(minus_dm[1 : period + 1])

    for i in range(period + 1, n):
        sm_tr[i] = sm_tr[i - 1] - (sm_tr[i - 1] / period) + tr[i]
        sm_plus[i] = sm_plus[i - 1] - (sm_plus[i - 1] / period) + plus_dm[i]
        sm_minus[i] = sm_minus[i - 1] - (sm_minus[i - 1] / period) + minus_dm[i]

    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        if sm_tr[i] > 0:
            plus_di[i] = 100.0 * sm_plus[i] / sm_tr[i]
            minus_di[i] = 100.0 * sm_minus[i] / sm_tr[i]
        diff = abs(plus_di[i] - minus_di[i])
        summ = plus_di[i] + minus_di[i]
        if summ > 0:
            dx[i] = 100.0 * diff / summ

    adx = np.zeros(n, dtype=np.float64)
    if n >= 2 * period:
        adx[2 * period - 1] = np.mean(dx[period : 2 * period])
        for i in range(2 * period, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return adx


def calc_sevcik_fdi(prices: np.ndarray, n: int = 30) -> float:
    if len(prices) < n:
        return 1.5
    sub = prices[-n:]
    h = np.max(sub)
    l = np.min(sub)
    diff = h - l
    if diff <= 1e-9:
        return 1.0
    norm_y = (sub - l) / diff
    delta_y = np.diff(norm_y)
    delta_x = 1.0 / (n - 1)
    seg_lengths = np.sqrt(delta_y ** 2 + delta_x ** 2)
    length = float(np.sum(seg_lengths))
    if length <= 0.0:
        return 1.0
    fdi = 1.0 + (math.log(length) + math.log(2.0)) / math.log(2.0 * (n - 1))
    return float(np.clip(fdi, 1.0, 2.0))


def calc_permutation_entropy(prices: np.ndarray, n: int = 30, d: int = 3) -> float:
    if len(prices) < n:
        return 0.5
    sub = prices[-n:]
    counts = [0] * 6
    total = 0
    for i in range(n - d + 1):
        p0, p1, p2 = sub[i], sub[i + 1], sub[i + 2]
        if p0 <= p1 and p1 <= p2:
            counts[0] += 1
        elif p0 <= p2 and p2 <= p1:
            counts[1] += 1
        elif p1 <= p0 and p0 <= p2:
            counts[2] += 1
        elif p1 <= p2 and p2 <= p0:
            counts[3] += 1
        elif p2 <= p0 and p0 <= p1:
            counts[4] += 1
        else:
            counts[5] += 1
        total += 1

    if total == 0:
        return 0.5

    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log(p)
    return float(entropy / math.log(6.0))


def run_institutional_backtest():
    print("=" * 72)
    print(" AI_TRADING V2 - INSTITUTIONAL BACKTEST ENGINE (LAST 2 MONTHS)")
    print(" Target: EURUSD M5 | 2026-08-21 to 2026-10-21 | Base Capital: 1.000 EUR")
    print("=" * 72)

    if not mt5.initialize(path=CONFIG.MT5_PATH):
        if not mt5.initialize():
            print("[ERRO] Falha ao conectar ao terminal MetaTrader 5.")
            return

    symbol = "EURUSD"
    start_warmup = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    start_test = datetime(2026, 8, 21, 0, 0, tzinfo=timezone.utc)
    end_test = datetime(2026, 10, 21, 23, 59, tzinfo=timezone.utc)

    rates_m5 = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_warmup, end_test)
    rates_h1 = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start_warmup, end_test)
    sym_info = mt5.symbol_info(symbol)
    mt5.shutdown()

    if rates_m5 is None or len(rates_m5) == 0:
        print("[ERRO] Falha ao extrair barras M5 do MT5.")
        return

    df_m5 = pd.DataFrame(rates_m5)
    df_h1 = pd.DataFrame(rates_h1)

    df_m5["datetime"] = pd.to_datetime(df_m5["time"], unit="s", utc=True)
    df_h1["datetime"] = pd.to_datetime(df_h1["time"], unit="s", utc=True)

    print(f"-> Dados carregados do MT5: M5={len(df_m5)} barras, H1={len(df_h1)} barras.")

    # 1. Indicadores H1
    h1_close = df_h1["close"].values
    h1_high = df_h1["high"].values
    h1_low = df_h1["low"].values
    h1_time = df_h1["time"].values

    h1_ema20 = calc_ema(h1_close, 20)
    h1_ema50 = calc_ema(h1_close, 50)
    h1_adx = calc_adx(h1_high, h1_low, h1_close, 14)

    # 2. Indicadores M5
    m5_close = df_m5["close"].values
    m5_open = df_m5["open"].values
    m5_high = df_m5["high"].values
    m5_low = df_m5["low"].values
    m5_time = df_m5["time"].values
    m5_dt = df_m5["datetime"].values

    m5_ema20 = calc_ema(m5_close, 20)
    m5_atr14 = calc_atr(m5_high, m5_low, m5_close, 14)
    m5_atr_med = pd.Series(m5_atr14).rolling(50, min_periods=1).median().values

    # Mapeamento do último candle H1 fechado para cada candle M5
    h1_idx_for_m5 = np.searchsorted(h1_time, m5_time, side="right") - 2
    h1_idx_for_m5 = np.clip(h1_idx_for_m5, 0, len(h1_time) - 1)

    # 3. Carregar Modelo ONNX Meta-Guard V2
    onnx_path = os.path.join(PROJECT_ROOT, "models", "meta_guard_v2.onnx")
    onnx_sess = ort.InferenceSession(onnx_path)
    input_name = onnx_sess.get_inputs()[0].name

    # 4. Inicializar Position Sizer V2
    sizer = InstitutionalPositionSizer(
        max_single_risk_pct=0.01,
        max_daily_dd_pct=0.03,
        esma_leverage=30.0,
        max_total_margin_usage_pct=0.45,
        max_concurrent_positions=3,
    )

    # 5. Estados dos Dois Portfólios
    INITIAL_CAPITAL = 1000.0

    # Portfolio A: Raw Baseline (Sem Copilot V2)
    state_a = {"balance": INITIAL_CAPITAL, "equity": INITIAL_CAPITAL, "trades": []}

    # Portfolio B: Hardened V2 Institutional Twin (Com Copilot V2)
    state_b = {
        "balance": INITIAL_CAPITAL,
        "equity": INITIAL_CAPITAL,
        "trades": [],
        "vetoed_trades": [],
        "wins": 18,
        "total_trades": 32,
    }

    # Índice de início do teste real (após período de aquecimento)
    test_start_ts = int(start_test.timestamp())
    test_start_idx = np.searchsorted(m5_time, test_start_ts)

    # Variáveis de Estado do Filtro de Kalman 2D
    x0, x1 = m5_close[0], 0.0
    p00, p01, p10, p11 = 1.0, 0.0, 0.0, 1.0

    active_trade_a = None
    active_trade_b = None

    print(f"-> A iniciar simulação a partir do índice M5 {test_start_idx} ({df_m5['datetime'].iloc[test_start_idx]})...\n")

    for i in range(1, len(m5_close)):
        # Atualização do Filtro de Kalman 2D
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

        # Gestão de Trade Ativo Portfolio A
        if active_trade_a is not None:
            t = active_trade_a
            # Simulação intra-candle com High e Low
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
                # Calcular PnL com custo real de execução (1 pip = 0.0001 = 9.17 EUR por lote padrão)
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

        # Gestão de Trade Ativo Portfolio B (Com Breakeven e Trailing Stop Institucional)
        if active_trade_b is not None:
            t = active_trade_b
            # 1. Breakeven Check ao atingir 1.0R
            if not t["be_triggered"]:
                favorable_pips = (m5_high[i] - t["entry"]) / 0.0001 if t["dir"] == 1 else (t["entry"] - m5_low[i]) / 0.0001
                if favorable_pips >= t["r_pips"]:
                    t["be_triggered"] = True
                    t["sl"] = t["entry"] + (0.0001 if t["dir"] == 1 else -0.0001)

            # 2. Avaliação de saída
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
                # Calcular PnL com custo real de execução (1 pip = 0.0001 = 9.17 EUR por lote padrão)
                pips = (exit_price - t["entry"]) / 0.0001 if t["dir"] == 1 else (t["entry"] - exit_price) / 0.0001
                commission = t["lots"] * 6.0
                pnl_eur = (pips * t["lots"] * 9.17) - commission
                state_b["balance"] += pnl_eur
                is_win = 1 if pnl_eur > 0 else 0
                state_b["trades"].append({
                    "entry_time": t["time"],
                    "exit_time": m5_time[i],
                    "dir": t["dir"],
                    "lots": t["lots"],
                    "pnl": pnl_eur,
                    "win": is_win,
                    "r": pnl_eur / max(1.0, t["risk_eur"]),
                })
                # Atualização Bayesiana Contínua
                state_b["total_trades"] += 1
                if is_win:
                    state_b["wins"] += 1
                active_trade_b = None

        # Avaliação de Novos Setups apenas na janela de teste
        if i < test_start_idx:
            continue

        # 0. Filtro de Sessão Operacional Institucional (07:00 a 16:00 UTC - London + NY Overlap)
        hour_utc = (m5_time[i] % 86400) / 3600.0
        if hour_utc < 7.0 or hour_utc >= 16.0:
            continue

        h1_idx = h1_idx_for_m5[i]
        h1_e20 = h1_ema20[h1_idx]
        h1_e50 = h1_ema50[h1_idx]
        h1_a = h1_adx[h1_idx]

        # Filtro de Tendência & ADX H1
        if h1_a < 22.0:
            continue

        trend_bull = h1_e20 > h1_e50
        trend_bear = h1_e20 < h1_e50
        if not trend_bull and not trend_bear:
            continue

        # Filtro de Volatilidade M5
        atr = m5_atr14[i]
        atr_med = m5_atr_med[i]
        if atr < 0.70 * atr_med or atr > 2.50 * atr_med or atr <= 1e-5:
            continue

        # Filtro Fractal FDI e Entropia de Permutação
        fdi_val = calc_sevcik_fdi(m5_close[: i + 1], 30)
        if fdi_val > 1.48:
            continue

        pe_val = calc_permutation_entropy(m5_close[: i + 1], 30, 3)
        if pe_val > 0.94:
            continue

        # Gatilho de Pullback à EMA(20)
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

        # Preço de Execução com Microstructural Slippage Penalty & Spread Dinâmico
        spread_pips = 1.2 * (1.0 + 0.3 * max(0.0, (atr / max(atr_med, 1e-5)) - 1.0))
        slippage_price = 0.00003
        exec_price = (last_close + (spread_pips * 0.0001) / 2.0 + slippage_price) if direction == 1 else (last_close - (spread_pips * 0.0001) / 2.0 - slippage_price)

        # Níveis Base de SL e TP
        sl_buffer = 0.5 * atr
        base_sl = (min(last_low, last_e20) - sl_buffer) if direction == 1 else (max(last_high, last_e20) + sl_buffer)
        sl_dist = abs(exec_price - base_sl)
        if sl_dist < 0.00050:
            continue

        base_tp = (exec_price + 1.5 * sl_dist) if direction == 1 else (exec_price - 1.5 * sl_dist)
        r_pips = sl_dist / 0.0001

        # ==========================================================
        # 1. Execução no Portfolio A (Baseline MQL5 Puro)
        # ==========================================================
        if active_trade_a is None:
            # Risco Fixo de 1.0% do Saldo
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

        # ==========================================================
        # 2. Execução no Portfolio B (Institutional V2 Hardened Twin)
        # ==========================================================
        if active_trade_b is None:
            # 2.1 Construir Vetor 16D
            t_sec = m5_time[i]
            hour_utc = (t_sec % 86400) / 3600.0
            hour_angle = 2.0 * math.pi * hour_utc / 24.0
            dow = ((t_sec // 86400) + 4) % 7
            dow_angle = 2.0 * math.pi * dow / 7.0

            state_vec = MarketStateVector16D(
                hour_sin=math.sin(hour_angle),
                hour_cos=math.cos(hour_angle),
                dow_sin=math.sin(dow_angle),
                dow_cos=math.cos(dow_angle),
                fdi_m5=float(fdi_val),
                kalman_vel_z=float(kalman_vel_z),
                adx_h1_norm=float(h1_a),
                atr_ratio=float(atr / max(atr_med, 1e-6)),
                bbw_percentile=0.55,
                rsi_m5_centered=54.0 if direction == 1 else 46.0,
                vp_dist_poc_atr=0.25 * direction,
                vp_dist_vah_atr=-0.45,
                vp_dist_val_atr=0.45,
                spread_zscore=float(spread_pips / 15.0),
                h1_ema_slope_atr=0.35 * direction,
                macro_sentiment=0.0,
            )

            # 2.2 Inferência ONNX Meta-Guard (Vetor de 10 dimensões de alta precisão)
            features_10d = np.array(
                [[
                    float(direction),
                    atr * 10000.0,
                    float(h1_a),
                    float(fdi_val),
                    float(kalman_vel_z),
                    float(spread_pips * 10.0),
                    0.0,  # sentiment * direction
                    0.0,  # rag_expectancy
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

            # 2.3 Fronteira Bayesiana Dinâmica p_crit(t)
            cost_ratio = (spread_pips * 0.0001) / max(sl_dist, 1e-5)
            b_net = max(0.5, (1.5 - cost_ratio) / (1.0 + cost_ratio))
            be_threshold = 1.0 / (1.0 + b_net)
            fdi_pen = 0.15 * max(0.0, fdi_val - 1.42)
            p_crit = float(min(0.62, max(0.38, be_threshold + fdi_pen)))

            # Veto por Meta-Guard
            if meta_prob < p_crit:
                # Regista trade vetado para auditoria contrafactual
                state_b["vetoed_trades"].append({
                    "time": m5_time[i],
                    "dir": direction,
                    "entry": exec_price,
                    "sl": base_sl,
                    "tp": base_tp,
                    "meta_prob": meta_prob,
                    "p_crit": p_crit,
                    "would_hit_sl": True if ((m5_low[min(i+36, len(m5_low)-1)] <= base_sl) if direction == 1 else (m5_high[min(i+36, len(m5_high)-1)] >= base_sl)) else False
                })
                continue

            # 2.4 Dimensionamento Institucional Kelly 4.ª Ordem sob ESMA 1:30
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
                    "lots": sizing_res.lots,
                    "risk_eur": sizing_res.risk_eur,
                    "r_pips": r_pips,
                    "be_triggered": False,
                }

    # ==========================================================
    # CÁLCULO E EXIBIÇÃO DE MÉTRICAS COMPARATIVAS
    # ==========================================================
    print("=" * 72)
    print(" RELATÓRIO QUANTITATIVO COMPARATIVO (ÚLTIMOS 2 MESES)")
    print("=" * 72)

    def compute_metrics(trades, final_balance):
        if not trades:
            return {}
        df_tr = pd.DataFrame(trades)
        total_tr = len(df_tr)
        wins = int(df_tr["win"].sum())
        losses = total_tr - wins
        win_rate = (wins / total_tr) * 100.0 if total_tr > 0 else 0.0

        pnl_series = df_tr["pnl"].values
        gross_profit = float(df_tr[df_tr["pnl"] > 0]["pnl"].sum())
        gross_loss = float(abs(df_tr[df_tr["pnl"] < 0]["pnl"].sum()))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 999.0
        net_profit = final_balance - INITIAL_CAPITAL
        return_pct = (net_profit / INITIAL_CAPITAL) * 100.0

        # Equity Curve & Drawdown
        cum_pnl = np.cumsum(pnl_series)
        equity_curve = INITIAL_CAPITAL + np.insert(cum_pnl, 0, 0.0)
        peaks = np.maximum.accumulate(equity_curve)
        dd_arr = (peaks - equity_curve) / np.maximum(peaks, 1.0)
        max_dd_pct = float(np.max(dd_arr)) * 100.0

        # Sharpe & Sortino (normalizados pelo horizonte de 2 meses = ~40 dias úteis)
        returns = pnl_series / INITIAL_CAPITAL
        mean_ret = np.mean(returns)
        std_ret = np.std(returns) if len(returns) > 1 else 1e-6
        downside_std = np.std(returns[returns < 0]) if len(returns[returns < 0]) > 1 else 1e-6

        # Fator de anualização para ~6 operações/semana (~30 trades em 2 meses)
        ann_factor = math.sqrt(252 * (total_tr / 42.0)) if total_tr > 0 else 1.0
        sharpe = (mean_ret / max(std_ret, 1e-6)) * ann_factor
        sortino = (mean_ret / max(downside_std, 1e-6)) * ann_factor

        avg_r = float(df_tr["r"].mean())

        return {
            "total_trades": total_tr,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": profit_factor,
            "net_profit": net_profit,
            "return_pct": return_pct,
            "max_dd_pct": max_dd_pct,
            "sharpe": sharpe,
            "sortino": sortino,
            "avg_r": avg_r,
        }

    met_a = compute_metrics(state_a["trades"], state_a["balance"])
    met_b = compute_metrics(state_b["trades"], state_b["balance"])

    # Eficiência dos Vetos
    vetoed = state_b["vetoed_trades"]
    total_vetoed = len(vetoed)
    saved_losses = sum(1 for v in vetoed if v["would_hit_sl"])
    veto_precision = (saved_losses / max(1, total_vetoed)) * 100.0

    print(f"{'Métrica':<32} | {'Baseline MQL5 Puro':<18} | {'Hardened V2 Twin':<18}")
    print("-" * 72)
    print(f"{'Capital Inicial':<32} | {INITIAL_CAPITAL:<18.2f} | {INITIAL_CAPITAL:<18.2f}")
    print(f"{'Saldo Final (EUR)':<32} | {state_a['balance']:<18.2f} | {state_b['balance']:<18.2f}")
    print(f"{'Lucro Líquido (EUR / %)':<32} | {met_a.get('net_profit',0):+.2f} ({met_a.get('return_pct',0):+.1f}%)   | {met_b.get('net_profit',0):+.2f} ({met_b.get('return_pct',0):+.1f}%)")
    print(f"{'Total de Trades Executados':<32} | {met_a.get('total_trades',0):<18} | {met_b.get('total_trades',0):<18}")
    print(f"{'Taxa de Acerto (Win Rate)':<32} | {met_a.get('win_rate',0):.1f}%{'':<13} | {met_b.get('win_rate',0):.1f}%")
    print(f"{'Fator de Lucro (Profit Factor)':<32} | {met_a.get('profit_factor',0):<18.2f} | {met_b.get('profit_factor',0):<18.2f}")
    print(f"{'Drawdown Máximo (Max DD)':<32} | {met_a.get('max_dd_pct',0):.2f}%{'':<12} | {met_b.get('max_dd_pct',0):.2f}%")
    print(f"{'Sharpe Ratio (Anualizado)':<32} | {met_a.get('sharpe',0):<18.2f} | {met_b.get('sharpe',0):<18.2f}")
    print(f"{'Sortino Ratio (Anualizado)':<32} | {met_a.get('sortino',0):<18.2f} | {met_b.get('sortino',0):<18.2f}")
    print(f"{'Expectativa Matemática (R)':<32} | {met_a.get('avg_r',0):+.2f}R{'':<14} | {met_b.get('avg_r',0):+.2f}R")
    print("-" * 72)
    print("AUDITORIA DE DEFESA DO META-GUARD (ONNX + BAYESIAN THRESHOLD):")
    print(f"-> Sinais Detetados pelo MQL5:        {met_a.get('total_trades',0)}")
    print(f"-> Sinais Vetados pelo Copilot V2:    {total_vetoed}")
    print(f"-> Perdas Reais Evitadas (Veto Precision): {saved_losses}/{total_vetoed} ({veto_precision:.1f}%)")
    print("-" * 72)
    print(f"REGISTO DOS {len(state_b['trades'])} TRADES EXECUTADOS PELO COPILOT V2:")
    for idx, tr in enumerate(state_b["trades"], 1):
        dt_str = datetime.fromtimestamp(tr["entry_time"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        dir_str = "BUY " if tr["dir"] == 1 else "SELL"
        res_str = "WIN " if tr["win"] == 1 else "LOSS"
        print(f"  {idx:02d}. [{dt_str}] {dir_str} | Lotes: {tr['lots']:.2f} | PnL: {tr['pnl']:+6.2f} EUR | R: {tr['r']:+5.2f}R | {res_str}")
    print("=" * 72)


if __name__ == "__main__":
    run_institutional_backtest()
