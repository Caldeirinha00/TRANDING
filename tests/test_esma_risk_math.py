from datetime import datetime, timezone
import time
import pytest
from core.mt5_connector import SymbolContractSpec
from data.cost_model import CostModel
from risk.currency_guard import CurrencyAndHeatGuard, OpenPositionRiskState
from risk.portfolio_sizer import PortfolioRiskSizer, CandidateSignal
from risk.tiered_watchdog import TieredWatchdog, LivePositionSnapshot


def make_mock_spec(
    symbol: str,
    pip_value_001_eur: float,
    digits: int = 5,
    spread_points: int = 8,
    swap_long: float = -2.0,
    swap_short: float = 0.5,
    margin_001: float = 30.0,
) -> SymbolContractSpec:
    point = 10 ** (-digits)
    pip_size = point * 10.0 if digits in (3, 5) else point
    # Como 1 pip = 10 points (em 5 dígitos), tick_value por point (em 1.00 lote) = (pip_value_001_eur * 100) / 10
    tick_value_1lot = (pip_value_001_eur * 100.0) / (pip_size / point)
    return SymbolContractSpec(
        symbol=symbol,
        currency_base=symbol[:3],
        currency_profit=symbol[3:6],
        currency_margin=symbol[:3],
        digits=digits,
        point=point,
        pip_size=pip_size,
        trade_contract_size=100000.0,
        trade_tick_value=tick_value_1lot,
        trade_tick_value_profit=tick_value_1lot,
        trade_tick_value_loss=tick_value_1lot,
        trade_tick_size=point,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        spread_points=spread_points,
        swap_mode=1,
        swap_long=swap_long,
        swap_short=swap_short,
        swap_rollover3days=3,  # Quarta-feira tripla
        stops_level_points=0,
        margin_initial_001_eur=margin_001,
        bid=1.10000,
        ask=1.10008,
    )


def test_all_8_approved_pairs_respect_10_eur_max_risk():
    """
    Critério Fase 0.1: Demonstração de que nenhum dos 8 pares com Stop Loss estrutural
    (2.0 x ATR H4 típico) excede 10,00 EUR (1,0% de 1.000 EUR) e que XAUUSD é bloqueado.
    """
    sizer = PortfolioRiskSizer()
    now_utc = datetime(2025, 3, 3, 12, 0, tzinfo=timezone.utc)  # Segunda-feira

    # Dados da tabela do Plano Diretor: (symbol, pip_val_001_eur, digits, sl_pips, sample_price)
    table_specs = [
        ("EURUSD", 0.092, 5, 56, 1.0800),
        ("USDJPY", 0.062, 3, 110, 150.00),
        ("AUDUSD", 0.092, 5, 52, 0.6500),
        ("NZDUSD", 0.092, 5, 50, 0.6000),
        ("USDCAD", 0.067, 5, 68, 1.3500),
        ("USDCHF", 0.104, 5, 56, 0.8800),
        ("GBPUSD", 0.092, 5, 80, 1.2700),
        ("AUDJPY", 0.062, 3, 104, 97.50),
    ]

    for sym, pip_val, digits, sl_pips, entry in table_specs:
        spec = make_mock_spec(sym, pip_val, digits=digits)
        sl_dist = sl_pips * spec.pip_size
        tp_dist = 1.5 * sl_dist

        sig = CandidateSignal(
            symbol=sym,
            side=+1,
            entry_price=entry,
            sl_price=entry - sl_dist,
            tp_price=entry + tp_dist,
            meta_probability=0.62,  # Entre 0.55 e 0.68 -> 0.01 lote
            signal_time_utc=now_utc,
        )
        decision = sizer.evaluate_single_candidate(sig, spec, equity_eur=1000.0, open_positions=[])
        assert decision.approved is True, f"Falha em {sym}: {decision.rejection_reason}"
        assert decision.allocated_lots == 0.01
        assert decision.risk_eur <= 10.00, f"Risco em {sym} ({decision.risk_eur}) excedeu 10.00 EUR!"

    # Validação da proibição estrita de XAUUSD abaixo de 3.000 EUR
    xau_spec = make_mock_spec("XAUUSD", pip_value_001_eur=0.92, digits=2, margin_001=130.0)
    xau_sig = CandidateSignal(
        symbol="XAUUSD",
        side=+1,
        entry_price=2850.0,
        sl_price=2815.0,  # $35 SL -> ~32.20 EUR em 0.01 lote
        tp_price=2905.0,
        meta_probability=0.75,
        signal_time_utc=now_utc,
    )
    xau_decision = sizer.evaluate_single_candidate(xau_sig, xau_spec, equity_eur=1000.0, open_positions=[])
    assert xau_decision.approved is False
    assert "VETO_UNIVERSE_CAPITAL" in xau_decision.rejection_reason


def test_discrete_meta_allocation_001_vs_002_lots():
    """
    Critério Fase 0.2: Verifica a alocação discreta guiada por p:
    - p < 0.55 -> 0.00 lotes
    - p in [0.55, 0.68) -> 0.01 lote
    - p >= 0.68 -> 0.02 lotes SE 2x risco <= 10.00 EUR (ex: AUDUSD 4.78€ x 2 = 9.56€ <= 10€),
      mas mantém 0.01 lote em GBPUSD (7.36€ x 2 = 14.72€ > 10€).
    """
    sizer = PortfolioRiskSizer()
    now_utc = datetime(2025, 3, 3, 12, 0, tzinfo=timezone.utc)

    aud_spec = make_mock_spec("AUDUSD", pip_value_001_eur=0.092, digits=5)
    sl_dist_aud = 52 * aud_spec.pip_size  # Risco 0.01 = 4.784 EUR

    # 1. p = 0.52 -> Veto
    sig_low = CandidateSignal("AUDUSD", +1, 0.6500, 0.6500 - sl_dist_aud, 0.6500 + 1.5 * sl_dist_aud, 0.52, now_utc)
    dec_low = sizer.evaluate_single_candidate(sig_low, aud_spec, 1000.0, [])
    assert not dec_low.approved and dec_low.allocated_lots == 0.0

    # 2. p = 0.60 -> 0.01 lote
    sig_mid = CandidateSignal("AUDUSD", +1, 0.6500, 0.6500 - sl_dist_aud, 0.6500 + 1.5 * sl_dist_aud, 0.60, now_utc)
    dec_mid = sizer.evaluate_single_candidate(sig_mid, aud_spec, 1000.0, [])
    assert dec_mid.approved and dec_mid.allocated_lots == 0.01

    # 3. p = 0.72 em AUDUSD -> 0.02 lotes (9.57 EUR <= 10.00 EUR)
    sig_high = CandidateSignal("AUDUSD", +1, 0.6500, 0.6500 - sl_dist_aud, 0.6500 + 1.5 * sl_dist_aud, 0.72, now_utc)
    dec_high = sizer.evaluate_single_candidate(sig_high, aud_spec, 1000.0, [])
    assert dec_high.approved and dec_high.allocated_lots == 0.02
    assert dec_high.risk_eur <= 10.00

    # 4. p = 0.72 em GBPUSD (0.01 lote arrisca 7.36 EUR -> 0.02 arriscaria 14.72 EUR > 10 EUR) -> restringe a 0.01 lote
    gbp_spec = make_mock_spec("GBPUSD", pip_value_001_eur=0.092, digits=5)
    sl_dist_gbp = 80 * gbp_spec.pip_size
    sig_gbp = CandidateSignal("GBPUSD", +1, 1.2700, 1.2700 - sl_dist_gbp, 1.2700 + 1.5 * sl_dist_gbp, 0.72, now_utc)
    dec_gbp = sizer.evaluate_single_candidate(sig_gbp, gbp_spec, 1000.0, [])
    assert dec_gbp.approved and dec_gbp.allocated_lots == 0.01


def test_portfolio_heat_4th_position_and_currency_bucket_guard():
    """
    Critério Fase 0.3:
    - Bloqueia exposição direcional repetida na mesma moeda (Long EURUSD bloqueia Long AUDUSD e Short USDCHF).
    - Liberta a moeda e o calor quando a posição transita para Break-Even.
    - Rejeita 4.ª posição em risco ativo ou qualquer ordem que eleve o Portfolio Heat acima de 21,00 EUR (2,10%).
    """
    guard = CurrencyAndHeatGuard()

    # Posição 1: Long EURUSD em risco ativo (Long EUR, Short USD) com 6.50 EUR de risco
    pos1 = OpenPositionRiskState(
        ticket=101, symbol="EURUSD", side=+1, lots=0.01, entry_price=1.0800, current_sl=1.0740, open_risk_eur=6.50
    )

    # Tentar Long AUDUSD (Short USD repetido) -> Deve ser vetado pelo Currency Bucket Guard
    ok_aud, reason_aud = guard.can_accept_new_risk("AUDUSD", +1, 4.80, 1000.0, [pos1])
    assert not ok_aud and "VETO_CURRENCY_BUCKET" in reason_aud

    # Tentar Short USDCHF (Short USD repetido) -> Deve ser vetado pelo Currency Bucket Guard
    ok_chf, reason_chf = guard.can_accept_new_risk("USDCHF", -1, 5.50, 1000.0, [pos1])
    assert not ok_chf and "VETO_CURRENCY_BUCKET" in reason_chf

    # Tentar Long AUDJPY (Long AUD, Short JPY -> sem EUR nem USD) -> Deve ser aprovado!
    ok_audjpy, _ = guard.can_accept_new_risk("AUDJPY", +1, 6.45, 1000.0, [pos1])
    assert ok_audjpy is True

    # Adicionar Posição 2 (Long AUDJPY, risco 7.50€) e Posição 3 (Long GBPCHF, risco 6.80€)
    pos2 = OpenPositionRiskState(
        ticket=102, symbol="AUDJPY", side=+1, lots=0.01, entry_price=97.00, current_sl=96.00, open_risk_eur=7.50
    )
    # Testar violação de Portfolio Heat (> 21.00 EUR): 6.50 + 7.50 + 7.50 = 21.50 EUR > 21.00 EUR
    ok_heat, reason_heat = guard.can_accept_new_risk("NZDCAD", +1, 7.50, 1000.0, [pos1, pos2])
    assert not ok_heat and "VETO_PORTFOLIO_HEAT" in reason_heat

    # Com risco menor (6.00 EUR -> total 20.00 EUR <= 21.00 EUR), a 3.ª posição é aprovada
    pos3 = OpenPositionRiskState(
        ticket=103, symbol="NZDCAD", side=+1, lots=0.01, entry_price=0.8200, current_sl=0.8140, open_risk_eur=6.00
    )
    # Tentar abrir 4.ª posição em risco ativo -> Vetado pelo limite de 3 posições
    ok_4th, reason_4th = guard.can_accept_new_risk("GBPCHF", +1, 0.50, 1000.0, [pos1, pos2, pos3])
    assert not ok_4th and "VETO_MAX_CONCURRENT_POSITIONS" in reason_4th

    # Se a Posição 1 (EURUSD) mover o SL para Break-Even (current_sl >= entry_price), liberta a vaga e a moeda USD!
    pos1_be = OpenPositionRiskState(
        ticket=101, symbol="EURUSD", side=+1, lots=0.01, entry_price=1.0800, current_sl=1.0802, open_risk_eur=6.50
    )
    ok_after_be, _ = guard.can_accept_new_risk("USDCHF", -1, 5.50, 1000.0, [pos1_be, pos2, pos3])
    assert ok_after_be is True


class MockWatchdogAdapter:
    def __init__(self, equity: float, day_start: float, hwm: float, positions: list[LivePositionSnapshot]):
        self.equity = equity
        self.day_start = day_start
        self.hwm = hwm
        self.positions = positions
        self.closed_tickets: list[int] = []
        self.liquidated_all = False

    def get_account_equity_eur(self) -> float:
        return self.equity

    def get_daily_starting_equity_eur(self) -> float:
        return self.day_start

    def get_high_water_mark_equity_eur(self) -> float:
        return self.hwm

    def get_open_positions(self) -> list[LivePositionSnapshot]:
        return self.positions

    def emergency_close_position(self, ticket: int, reason: str) -> bool:
        self.closed_tickets.append(ticket)
        return True

    def emergency_close_all(self, reason: str) -> int:
        self.liquidated_all = True
        return len(self.positions)


def test_watchdog_naked_sl_under_5_seconds_and_kill_switches():
    """
    Critério Fase 0.4: O Watchdog deteta e encerra qualquer posição sem SL no servidor em < 5 segundos
    e dispara os kill switches a 2,85% diário e 19,5% total.
    """
    naked_pos = LivePositionSnapshot(
        ticket=55501, symbol="EURUSD", side=+1, volume=0.01, price_open=1.0800, sl=0.0, tp=1.0950, profit_eur=-0.20
    )
    adapter = MockWatchdogAdapter(equity=995.0, day_start=1000.0, hwm=1000.0, positions=[naked_pos])
    watchdog = TieredWatchdog(adapter)

    t0 = time.perf_counter()
    res = watchdog.run_single_audit_cycle()
    elapsed = time.perf_counter() - t0

    assert elapsed < 5.0
    assert res.status == "NAKED_POSITION_CLOSED"
    assert 55501 in adapter.closed_tickets

    # Teste Kill Switch Diário (2,85% = 971.50 EUR num início de dia de 1.000 EUR)
    adapter_daily = MockWatchdogAdapter(equity=971.0, day_start=1000.0, hwm=1000.0, positions=[naked_pos])
    res_daily = TieredWatchdog(adapter_daily).run_single_audit_cycle()
    assert res_daily.status == "DAILY_KILL_SWITCH" and adapter_daily.liquidated_all is True

    # Teste Kill Switch Total (19,5% = 805.00 EUR num HWM de 1.000 EUR)
    adapter_total = MockWatchdogAdapter(equity=804.0, day_start=820.0, hwm=1000.0, positions=[naked_pos])
    res_total = TieredWatchdog(adapter_total).run_single_audit_cycle()
    assert res_total.status == "TOTAL_KILL_SWITCH" and adapter_total.liquidated_all is True
