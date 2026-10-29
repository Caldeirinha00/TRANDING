"""
AI TRADING SYSTEM — 60-Day Quantitative Backtest (Last 2 Months)
Backtests EURUSD M5 over the last 60 days on a 1,000 EUR base capital.
Compares Raw MQL5 Rules vs. AI-Enhanced (34D ScalpNet/LGBM + Bayesian Half-Kelly).
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.indicators import compute_scalp_features, LOOKBACK_WINDOW
from agents.scalp_signal import ScalpSignalEngine

def calc_py_ema(series: np.ndarray, period: int) -> np.ndarray:
    n = len(series)
    ema = np.zeros(n)
    if n < period:
        return ema
    k = 2.0 / (period + 1.0)
    ema[period - 1] = np.mean(series[:period])
    for i in range(period, n):
        ema[i] = (series[i] - ema[i - 1]) * k + ema[i - 1]
    return ema

def calc_py_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14):
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, max(hc, lc))
    atr = np.zeros(n)
    if n < period:
        return tr, atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return tr, atr

def calc_py_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(close)
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, max(hc, lc))
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down

    sm_tr = np.zeros(n)
    sm_plus = np.zeros(n)
    sm_minus = np.zeros(n)
    if n < period + 1:
        return np.zeros(n)
    sm_tr[period] = np.sum(tr[1:period + 1])
    sm_plus[period] = np.sum(plus_dm[1:period + 1])
    sm_minus[period] = np.sum(minus_dm[1:period + 1])
    for i in range(period + 1, n):
        sm_tr[i] = sm_tr[i - 1] - (sm_tr[i - 1] / period) + tr[i]
        sm_plus[i] = sm_plus[i - 1] - (sm_plus[i - 1] / period) + plus_dm[i]
        sm_minus[i] = sm_minus[i - 1] - (sm_minus[i - 1] / period) + minus_dm[i]

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
        adx[2 * period - 1] = np.mean(dx[period:2 * period])
        for i in range(2 * period, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return adx


def run_60d_backtest():
    print("=" * 75)
    print("  AI TRADING SYSTEM — 60-DAY QUANTITATIVE BACKTEST (EURUSD M5)")
    print("  Capital Base: 1,000.00 EUR | Arquitetura: HWM Shield + 3% Daily DD")
    print("=" * 75)

    if not mt5.initialize():
        print("CRITICAL: Falha ao conectar ao MetaTrader 5.")
        return

    symbol = "EURUSD"
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=60)

    print(f"\nA descarregar dados historicos M5/H1 ({start_dt.strftime('%d/%m/%Y')} -> {end_dt.strftime('%d/%m/%Y')})...")
    rates_m5 = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_dt, end_dt)
    rates_h1 = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start_dt - timedelta(days=20), end_dt)
    rates_m15 = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, start_dt - timedelta(days=5), end_dt)

    sym_info = mt5.symbol_info(symbol)
    tick_size = sym_info.trade_tick_size if sym_info else 0.00001
    tick_value = sym_info.trade_tick_value if sym_info else 1.0
    point = sym_info.point if sym_info else 0.00001
    spread_pts = sym_info.spread if sym_info else 12
    spread_price = spread_pts * point
    mt5.shutdown()

    if rates_m5 is None or len(rates_m5) < 500:
        print("ERROR: Dados historicos M5 insuficientes.")
        return

    print(f"Sucesso: {len(rates_m5)} velas M5, {len(rates_h1)} velas H1 e {len(rates_m15)} velas M15.")

    df_m5 = pd.DataFrame(rates_m5)
    df_h1 = pd.DataFrame(rates_h1)
    df_m15 = pd.DataFrame(rates_m15)

    df_m5['dt'] = pd.to_datetime(df_m5['time'], unit='s')
    df_m5['hour'] = df_m5['dt'].dt.hour
    df_m5['date'] = df_m5['dt'].dt.date

    # Indicators
    m5_close = df_m5['close'].values
    m5_open = df_m5['open'].values
    m5_high = df_m5['high'].values
    m5_low = df_m5['low'].values
    m5_time = df_m5['time'].values
    m5_hour = df_m5['hour'].values

    m5_ema20 = calc_py_ema(m5_close, 20)
    _, m5_atr14 = calc_py_atr(m5_high, m5_low, m5_close, 14)
    m5_atr_median = pd.Series(m5_atr14).rolling(50, min_periods=1).median().values

    # H1 Indicators
    h1_close = df_h1['close'].values
    h1_high = df_h1['high'].values
    h1_low = df_h1['low'].values
    h1_time = df_h1['time'].values

    h1_ema25 = calc_py_ema(h1_close, 25)
    h1_ema70 = calc_py_ema(h1_close, 70)
    h1_adx = calc_py_adx(h1_high, h1_low, h1_close, 14)

    # M15 ATR for BE buffer
    _, m15_atr = calc_py_atr(df_m15['high'].values, df_m15['low'].values, df_m15['close'].values, 14)
    m15_time = df_m15['time'].values

    # Time alignments
    h1_idx_for_m5 = np.searchsorted(h1_time, m5_time, side='right') - 2
    h1_idx_for_m5 = np.clip(h1_idx_for_m5, 0, len(h1_time) - 1)

    h1_trend_bull = h1_ema25[h1_idx_for_m5] > h1_ema70[h1_idx_for_m5]
    h1_trend_bear = h1_ema25[h1_idx_for_m5] < h1_ema70[h1_idx_for_m5]
    h1_adx_m5 = h1_adx[h1_idx_for_m5]

    m15_idx_for_m5 = np.searchsorted(m15_time, m5_time, side='right') - 2
    m15_idx_for_m5 = np.clip(m15_idx_for_m5, 0, len(m15_time) - 1)
    m15_atr_m5 = m15_atr[m15_idx_for_m5]

    # Load 34D AI Engine
    print("A carregar modelo IA 34D (ScalpNet + LightGBM)...")
    engine = ScalpSignalEngine(symbol=symbol)
    has_ai = engine.scalpnet is not None or engine.lgbm_model is not None

    candles_dict_list = [
        {
            "time": int(df_m5.loc[idx, "time"]),
            "open": float(df_m5.loc[idx, "open"]),
            "high": float(df_m5.loc[idx, "high"]),
            "low": float(df_m5.loc[idx, "low"]),
            "close": float(df_m5.loc[idx, "close"]),
            "tick_volume": int(df_m5.loc[idx, "tick_volume"]),
            "spread_price": spread_price
        }
        for idx in range(len(df_m5))
    ]

    def simulate(use_ai: bool):
        initial_balance = 1000.0
        balance = initial_balance
        peak_balance = initial_balance
        max_dd = 0.0
        max_dd_pct = 0.0

        daily_start_balance = initial_balance
        current_day = None
        max_daily_dd_pct = 0.0

        trades_list = []
        vetoed_count = 0
        signals_count = 0

        wins_count = 0
        losses_count = 0
        total_win_money = 0.0
        total_loss_money = 0.0
        prior_wr = 0.50
        prior_rr = 1.25
        prior_strength = 10.0

        comm_roundtrip_per_lot = 7.0
        comm_price_offset = (comm_roundtrip_per_lot * tick_size) / tick_value if tick_value > 0 else 0.00007

        i = 100
        n_bars = len(m5_close)

        while i < n_bars - 1:
            bar_date = df_m5.loc[i, 'date']
            bar_hour = m5_hour[i]

            # Daily Anchor (00:00 rollover)
            if bar_date != current_day:
                current_day = bar_date
                daily_start_balance = balance

            # Daily Drawdown guard (3.0% daily circuit breaker)
            current_day_dd = ((daily_start_balance - balance) / daily_start_balance) * 100.0
            if current_day_dd > max_daily_dd_pct:
                max_daily_dd_pct = current_day_dd

            if current_day_dd >= 3.0:
                i += 1
                continue

            # Trading Window (08:00 to 16:00 UTC/Server)
            if bar_hour < 8 or bar_hour >= 16:
                i += 1
                continue

            atr = m5_atr14[i]
            atr_med = m5_atr_median[i]

            # ATR Volatility Filter [1.4, 2.5]
            if atr < 1.4 * atr_med or atr > 2.5 * atr_med:
                i += 1
                continue

            # ADX Trend Filter >= 30.0
            if h1_adx_m5[i] < 30.0:
                i += 1
                continue

            direction = None
            setup_type = None

            # Bullish Pullback
            if h1_trend_bull[i]:
                rejection = m5_close[i] > (m5_high[i] + m5_low[i]) * 0.5
                if m5_low[i] <= m5_ema20[i] and m5_close[i] > m5_ema20[i] and rejection:
                    direction = "BUY"
                    setup_type = "H1_BULL_PULLBACK_M5"

            # Bearish Pullback
            elif h1_trend_bear[i]:
                rejection = m5_close[i] < (m5_high[i] + m5_low[i]) * 0.5
                if m5_high[i] >= m5_ema20[i] and m5_close[i] < m5_ema20[i] and rejection:
                    direction = "SELL"
                    setup_type = "H1_BEAR_PULLBACK_M5"

            if not direction:
                i += 1
                continue

            signals_count += 1

            # AI Inference Check (34 Features)
            if use_ai and has_ai:
                window_candles = candles_dict_list[max(0, i - LOOKBACK_WINDOW + 1): i + 1]
                feats = compute_scalp_features(window_candles, spread_price)
                if feats is not None:
                    sig = engine.predict(feats, atr)
                    # Veto if AI disagrees or signals HOLD
                    if sig.action != direction or sig.confidence < 0.50:
                        vetoed_count += 1
                        i += 1
                        continue

            # Position Sizing: Bayesian Half-Kelly
            n_closed = wins_count + losses_count
            if n_closed >= 3:
                w_bayes = (wins_count + prior_strength * prior_wr) / (n_closed + prior_strength)
                avg_win = total_win_money / wins_count if wins_count > 0 else 1.0
                avg_loss = total_loss_money / losses_count if losses_count > 0 else 1.0
                r_sample = avg_win / avg_loss
                w_factor = min(n_closed / 20.0, 1.0)
                r_bayes = (1.0 - w_factor) * prior_rr + w_factor * r_sample
                kelly_raw = (w_bayes * r_bayes - (1.0 - w_bayes)) / r_bayes if r_bayes > 0 else 0.0
                kelly_risk_pct = max(0.20, min(1.0, 0.5 * kelly_raw * 100.0))
            else:
                kelly_risk_pct = 0.55

            entry_idx = i + 1
            if entry_idx >= n_bars:
                break

            entry_open = m5_open[entry_idx]
            entry_price = entry_open + spread_price if direction == "BUY" else entry_open

            # SL & TP Setup
            if direction == "BUY":
                sl = min(m5_low[i], m5_ema20[i]) - (atr * 1.2)
                sl_distance = entry_price - sl
                tp = entry_price + 1.25 * sl_distance
            else:
                sl = max(m5_high[i], m5_ema20[i]) + (atr * 1.2)
                sl_distance = sl - entry_price
                tp = entry_price - 1.25 * sl_distance

            if sl_distance <= 0:
                i += 1
                continue

            # Money risk & Lot Size
            risk_money = balance * (kelly_risk_pct / 100.0)
            sl_distance_ticks = sl_distance / tick_size
            money_lost_per_lot = sl_distance_ticks * tick_value
            if money_lost_per_lot <= 0:
                i += 1
                continue

            volume = round(max(0.01, min(0.50, risk_money / money_lost_per_lot)), 2)

            # Stage 1 BE & Stage 2 Trailing Loop
            be_active = False
            be_level = (entry_price + spread_price + comm_price_offset + (m15_atr_m5[i] * 0.15)) if direction == "BUY" else (entry_price - spread_price - comm_price_offset - (m15_atr_m5[i] * 0.15))
            current_sl = sl
            current_tp = tp
            exit_price = None
            exit_idx = entry_idx

            for k in range(entry_idx, n_bars):
                bar_h = m5_high[k]
                bar_l = m5_low[k]
                bar_c = m5_close[k]
                bar_atr = m5_atr14[k]

                if direction == "BUY":
                    if bar_l <= current_sl:
                        exit_price = current_sl
                        exit_idx = k
                        break
                    if bar_h >= current_tp:
                        exit_price = current_tp
                        exit_idx = k
                        break

                    profit_pts = bar_h - entry_price
                    if not be_active and profit_pts >= sl_distance:
                        current_sl = max(current_sl, be_level)
                        be_active = True

                    if be_active:
                        trail = bar_c - (bar_atr * 1.5)
                        if trail > current_sl:
                            current_sl = trail

                else: # SELL
                    if bar_h >= current_sl:
                        exit_price = current_sl
                        exit_idx = k
                        break
                    if bar_l <= current_tp:
                        exit_price = current_tp
                        exit_idx = k
                        break

                    profit_pts = entry_price - bar_l
                    if not be_active and profit_pts >= sl_distance:
                        current_sl = min(current_sl, be_level)
                        be_active = True

                    if be_active:
                        trail = bar_c + (bar_atr * 1.5)
                        if trail < current_sl:
                            current_sl = trail

            if exit_price is None:
                exit_price = m5_close[-1]
                exit_idx = n_bars - 1

            pnl_ticks = (exit_price - entry_price) / tick_size if direction == "BUY" else (entry_price - exit_price) / tick_size
            gross_pnl = pnl_ticks * tick_value * volume
            trade_comm = comm_roundtrip_per_lot * volume
            net_pnl = gross_pnl - trade_comm

            balance += net_pnl
            if balance > peak_balance:
                peak_balance = balance

            dd = peak_balance - balance
            dd_pct = (dd / peak_balance) * 100.0 if peak_balance > 0 else 0.0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
                max_dd = dd

            if net_pnl > 0:
                wins_count += 1
                total_win_money += net_pnl
            else:
                losses_count += 1
                total_loss_money += abs(net_pnl)

            trades_list.append({
                "entry_time": df_m5.loc[entry_idx, "dt"],
                "exit_time": df_m5.loc[exit_idx, "dt"],
                "dir": direction,
                "vol": volume,
                "entry": entry_price,
                "exit": exit_price,
                "net_pnl": net_pnl,
                "balance": balance,
                "kelly_pct": kelly_risk_pct
            })

            i = exit_idx + 1

        n_trades = len(trades_list)
        win_rate = (wins_count / n_trades * 100.0) if n_trades > 0 else 0.0
        profit_factor = (total_win_money / total_loss_money) if total_loss_money > 0 else 999.0
        net_profit = balance - initial_balance
        ret_pct = (net_profit / initial_balance) * 100.0

        return {
            "trades": n_trades,
            "wins": wins_count,
            "losses": losses_count,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "net_profit": net_profit,
            "ret_pct": ret_pct,
            "final_balance": balance,
            "max_dd_pct": max_dd_pct,
            "max_dd_eur": max_dd,
            "max_daily_dd_pct": max_daily_dd_pct,
            "signals_generated": signals_count,
            "signals_vetoed": vetoed_count,
            "trades_list": trades_list
        }

    print("\n[1/2] A simular Estrategia Base (Regras MQL5 sem IA)...")
    res_base = simulate(use_ai=False)

    print("[2/2] A simular Estrategia com IA (34D ScalpNet/LGBM + Bayesian Half-Kelly)...")
    res_ai = simulate(use_ai=True)

    print("\n" + "=" * 78)
    print("  RESULTADOS DETALHADOS DO BACKTEST — ULTIMOS 60 DIAS (EURUSD M5)")
    print("=" * 78)
    print(f"{'Metrica Quantitativa':<33} | {'Baseline (EA Puro)':<19} | {'AI-Enhanced (Twin)':<19}")
    print("-" * 78)
    print(f"{'Sinais Identificados':<33} | {res_base['signals_generated']:<19} | {res_ai['signals_generated']:<19}")
    veto_pct = (res_ai['signals_vetoed'] / max(1, res_ai['signals_generated'])) * 100.0
    print(f"{'Sinais Vetados pela IA':<33} | {'0 (0.0%)':<19} | {res_ai['signals_vetoed']} ({veto_pct:.1f}%)")
    print(f"{'Total de Operacoes':<33} | {res_base['trades']:<19} | {res_ai['trades']:<19}")
    print(f"{'Vitorias / Derrotas':<33} | {res_base['wins']}W / {res_base['losses']}L{'':<10} | {res_ai['wins']}W / {res_ai['losses']}L")
    print(f"{'Taxa de Acerto (Win Rate)':<33} | {res_base['win_rate']:.2f}%{'':<12} | {res_ai['win_rate']:.2f}%")
    print(f"{'Fator de Lucro (Profit Factor)':<33} | {res_base['profit_factor']:.2f}{'':<15} | {res_ai['profit_factor']:.2f}")
    print(f"{'Lucro Liquido (EUR)':<33} | {res_base['net_profit']:+.2f} EUR{'':<9} | {res_ai['net_profit']:+.2f} EUR")
    print(f"{'Rentabilidade Acumulada':<33} | {res_base['ret_pct']:+.2f}%{'':<12} | {res_ai['ret_pct']:+.2f}%")
    print(f"{'Saldo Final (Base 1.000 EUR)':<33} | {res_base['final_balance']:.2f} EUR{'':<7} | {res_ai['final_balance']:.2f} EUR")
    print(f"{'Max Drawdown Historico':<33} | {res_base['max_dd_pct']:.2f}% ({res_base['max_dd_eur']:.2f}E) | {res_ai['max_dd_pct']:.2f}% ({res_ai['max_dd_eur']:.2f}E)")
    print(f"{'Pico de Daily Drawdown':<33} | {res_base['max_daily_dd_pct']:.2f}% (Teto: 3.0%) | {res_ai['max_daily_dd_pct']:.2f}% (Teto: 3.0%)")
    print("=" * 78)

    return res_base, res_ai

if __name__ == "__main__":
    run_60d_backtest()
