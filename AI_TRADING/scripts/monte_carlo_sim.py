"""
scripts/monte_carlo_sim.py
Simulação de Monte Carlo para Análise de Ruína e Stress Test Institucional
Conta Base: 1.000 EUR | Alavancagem ESMA 1:30 | Amostra: Trades Reais do Hardened V2 Twin
"""

import sys
import os
import math
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

def run_monte_carlo(trades_pnl: list[float], initial_capital: float = 1000.0, num_simulations: int = 2000, num_trades: int = 100):
    print("=" * 72)
    print(" AI_TRADING V2 - INSTITUTIONAL MONTE CARLO STRESS TEST")
    print(f" Simulações: {num_simulations} | Horizonte: {num_trades} Trades | Base: {initial_capital:.2f} EUR (ESMA 1:30)")
    print("=" * 72)

    if not trades_pnl:
        print("[ERRO] Nenhuma série de trades fornecida.")
        return

    pnl_array = np.array(trades_pnl)
    n_sample = len(pnl_array)
    mean_pnl = np.mean(pnl_array)
    std_pnl = np.std(pnl_array)
    win_rate = np.mean(pnl_array > 0) * 100.0

    print(f"-> Amostra Base: {n_sample} trades reais")
    print(f"-> Média por Trade: {mean_pnl:+.2f} EUR | Desvio Padrão: {std_pnl:.2f} EUR | Win Rate: {win_rate:.1f}%\n")

    # Matriz de Reamostragem com Substituição: shape = (num_simulations, num_trades)
    np.random.seed(42)
    bootstrapped_indices = np.random.choice(n_sample, size=(num_simulations, num_trades), replace=True)
    simulated_pnls = pnl_array[bootstrapped_indices]

    # Curvas de Capital
    cum_pnls = np.cumsum(simulated_pnls, axis=1)
    equity_paths = initial_capital + cum_pnls

    # Prefixar capital inicial na barra 0
    full_equity = np.hstack([np.full((num_simulations, 1), initial_capital), equity_paths])

    # Cálculo do Max Drawdown por trajetória
    peaks = np.maximum.accumulate(full_equity, axis=1)
    drawdowns = (peaks - full_equity) / np.maximum(peaks, 1.0)
    max_dds_pct = np.max(drawdowns, axis=1) * 100.0

    # Saldos Finais
    final_equities = full_equity[:, -1]
    net_profits = final_equities - initial_capital

    # Barreiras de Risco e Ruína
    # ESMA 1:30 barreira de ruína crítica: perda de 10% (900 EUR) e 20% (800 EUR)
    ruin_10pct = np.mean(np.min(full_equity, axis=1) <= 0.90 * initial_capital) * 100.0
    ruin_20pct = np.mean(np.min(full_equity, axis=1) <= 0.80 * initial_capital) * 100.0
    ruin_50pct = np.mean(np.min(full_equity, axis=1) <= 0.50 * initial_capital) * 100.0

    # Percentis Institucionais
    p05_equity = np.percentile(final_equities, 5)
    p25_equity = np.percentile(final_equities, 25)
    p50_equity = np.percentile(final_equities, 50)
    p75_equity = np.percentile(final_equities, 75)
    p95_equity = np.percentile(final_equities, 95)

    p50_dd = np.percentile(max_dds_pct, 50)
    p95_dd = np.percentile(max_dds_pct, 95)
    p99_dd = np.percentile(max_dds_pct, 99)

    prob_profit = np.mean(final_equities > initial_capital) * 100.0

    print("=" * 72)
    print(" RELATÓRIO ESTATÍSTICO DE MONTE CARLO (2.000 ITERAÇÕES)")
    print("=" * 72)
    print(f"{'Métrica':<36} | {'Valor Estimado':<20}")
    print("-" * 72)
    print(f"{'Probabilidade de Lucro (100 Trades)':<36} | {prob_profit:.1f}%")
    print(f"{'Saldo Final Mediano (P50)':<36} | {p50_equity:.2f} EUR ({p50_equity - initial_capital:+.2f} EUR)")
    print(f"{'Pior Cenário Confiável (P05 Equity)':<36} | {p05_equity:.2f} EUR ({p05_equity - initial_capital:+.2f} EUR)")
    print(f"{'Melhor Cenário Confiável (P95 Equity)':<36} | {p95_equity:.2f} EUR ({p95_equity - initial_capital:+.2f} EUR)")
    print("-" * 72)
    print(f"{'Max Drawdown Mediano (P50)':<36} | {p50_dd:.2f}%")
    print(f"{'Max Drawdown P95 (Stress Confiável)':<36} | {p95_dd:.2f}%")
    print(f"{'Max Drawdown P99 (Cenário Extremo)':<36} | {p99_dd:.2f}%")
    print("-" * 72)
    print("PROBABILIDADES DE DRAWDOWN E RUÍNA ACUMULADA:")
    print(f"-> Risco de Atingir Drawdown >= 10% (Saldo <= 900 EUR): {ruin_10pct:.2f}%")
    print(f"-> Risco de Atingir Drawdown >= 20% (Saldo <= 800 EUR): {ruin_20pct:.2f}%")
    print(f"-> Probabilidade de Ruína (Perda >= 50% / Margin Call):  {ruin_50pct:.4f}%")
    print("=" * 72)

    return {
        "prob_profit": prob_profit,
        "p50_equity": p50_equity,
        "p05_equity": p05_equity,
        "p95_equity": p95_equity,
        "p50_dd": p50_dd,
        "p95_dd": p95_dd,
        "ruin_10pct": ruin_10pct,
        "ruin_20pct": ruin_20pct,
    }

if __name__ == "__main__":
    # 13 trades reais obtidos pelo Hardened V2 Twin no período out-of-sample (2026-08-21 a 2026-10-21)
    v2_trades = [-2.87, 2.31, 3.68, -2.18, 2.70, -3.33, 3.22, 4.79, 3.50, -2.45, 3.32, -2.43, -2.95]
    run_monte_carlo(v2_trades, initial_capital=1000.0, num_simulations=2000, num_trades=100)
