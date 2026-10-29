"""
tests/test_v2_core_units.py
Testes unitários automatizados para os componentes core da Arquitetura V2:
- InstitutionalPositionSizer (Conta 1.000 EUR, ESMA 1:30)
- Indicadores Matemáticos (FDI, Permutation Entropy)
- Telegram Notifier Fallback
"""

import math
import numpy as np
import pytest

from risk.quant_position_sizer import (
    InstitutionalPositionSizer,
    AccountState,
    SizingResult,
)
from scripts.backtest_v2_institutional import (
    calc_sevcik_fdi,
    calc_permutation_entropy,
    calc_atr,
)
from agents.telegram_notifier import TelegramNotifier


def test_position_sizer_standard_1000eur():
    sizer = InstitutionalPositionSizer(
        max_single_risk_pct=0.01,  # 1% = 10 EUR
        max_daily_dd_pct=0.03,
        esma_leverage=30.0,
        max_total_margin_usage_pct=0.45,
        max_concurrent_positions=1,
    )
    acc = AccountState(
        equity_eur=1000.0,
        free_margin_eur=1000.0,
        used_margin_eur=0.0,
        daily_dd_pct=0.0,
        open_positions_count=0,
        open_correlated_risk_eur=0.0,
    )

    # 25 pips de SL no EURUSD (0.0025)
    res: SizingResult = sizer.calculate_eurusd_lots(
        account=acc,
        eurusd_ask=1.0850,
        sl_distance_price=0.0025,
        wins=18,
        total_trades=32,
        avg_win_pips=20.0,
        avg_loss_pips=13.0,
        expected_exec_cost_pips=1.2,
        atr_current=0.0012,
        atr_median=0.0012,
        meta_prob=0.45,
    )

    assert res.lots >= 0.01
    assert res.lots <= 0.10
    # Margem consumida deve respeitar limite ESMA 1:30
    notional_margin = res.lots * 100000.0 / 30.0
    assert notional_margin <= acc.equity_eur * 0.45


def test_position_sizer_rejection_at_max_positions():
    sizer = InstitutionalPositionSizer(
        max_single_risk_pct=0.01,
        max_daily_dd_pct=0.03,
        esma_leverage=30.0,
        max_total_margin_usage_pct=0.45,
        max_concurrent_positions=1,
    )
    acc = AccountState(
        equity_eur=1000.0,
        free_margin_eur=850.0,
        used_margin_eur=150.0,
        daily_dd_pct=0.0,
        open_positions_count=1,  # Já tem 1 posição
        open_correlated_risk_eur=133.33,
    )

    res: SizingResult = sizer.calculate_eurusd_lots(
        account=acc,
        eurusd_ask=1.0850,
        sl_distance_price=0.0020,
        wins=18,
        total_trades=32,
        avg_win_pips=20.0,
        avg_loss_pips=13.0,
        expected_exec_cost_pips=1.2,
        atr_current=0.0010,
        atr_median=0.0010,
        meta_prob=0.48,
    )

    assert res.lots == 0.0
    assert res.risk_eur == 0.0


def test_sevcik_fdi_bounds():
    # Série constante / linear (D -> 1.0)
    linear_series = np.linspace(1.0, 2.0, 100)
    fdi_linear = calc_sevcik_fdi(linear_series, 30)
    assert 1.0 <= fdi_linear <= 1.30

    # Série com ruído extremo (D -> 2.0)
    np.random.seed(42)
    noise_series = np.random.randn(100)
    fdi_noise = calc_sevcik_fdi(noise_series, 30)
    assert 1.40 <= fdi_noise <= 2.0


def test_permutation_entropy_bounds():
    # Série estritamente crescente: entropia = 0
    monotonic = np.arange(100, dtype=float)
    pe_mono = calc_permutation_entropy(monotonic, 30)
    assert pe_mono == 0.0

    # Série ruidosa: entropia alta
    np.random.seed(42)
    noisy = np.random.uniform(1.0, 1.1, 100)
    pe_noisy = calc_permutation_entropy(noisy, 30)
    assert 0.70 <= pe_noisy <= 1.0


def test_telegram_notifier_no_token_safe():
    notifier = TelegramNotifier(bot_token="", chat_id="")
    assert not notifier.enabled
    # Não deve lançar exceção ao chamar _send_sync
    assert notifier._send_sync("teste") is False
