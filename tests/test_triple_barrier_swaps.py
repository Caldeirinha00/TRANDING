from datetime import datetime, timezone
from data.cost_model import CostModel
from tests.test_esma_risk_math import make_mock_spec


def test_wednesday_triple_rollover_accounting():
    """
    Valida que uma posição mantida por 24 barras H4 (4 dias úteis) atravessando a noite de
    quarta para quinta-feira contabiliza exatamente 6 dias de swap (3 na quarta + 3 nos restantes 3 dias)
    e aciona o Veto de Financiamento Hostil quando o swap excede 15% de 1R.
    """
    cost_model = CostModel()
    # Segunda-feira às 12:00 UTC -> 24 barras H4 (96h) termina na Sexta-feira às 12:00 UTC
    # Passa pelas 00:00 de: Terça (Seg=1x), Quarta (Ter=1x), Quinta (Qua=3x), Sexta (Qui=1x) -> Total = 6x swap diário
    entry_monday = datetime(2025, 3, 3, 12, 0, tzinfo=timezone.utc)

    spec = make_mock_spec("EURUSD", pip_value_001_eur=0.10, digits=5, swap_long=-1.5, swap_short=0.4)
    daily_long_swap = cost_model.calculate_daily_swap_eur(spec, side=+1, lots=0.01)
    projected_4d_swap = cost_model.project_holding_swap_eur(
        spec, side=+1, lots=0.01, entry_time_utc=entry_monday, holding_bars_h4=24
    )

    assert round(projected_4d_swap, 4) == round(daily_long_swap * 6, 4)

    # Se o swap negativo for extremo (ex: -20 points/dia -> -1.20 EUR em 6 dias de swap para um 1R de 5.00 EUR = 24% > 15%)
    hostile_spec = make_mock_spec("EURUSD", pip_value_001_eur=0.10, digits=5, swap_long=-20.0)
    friction = cost_model.evaluate_trade_friction(
        spec=hostile_spec,
        side=+1,
        lots=0.01,
        sl_distance_price=50 * hostile_spec.pip_size,  # 5.00 EUR de risco 1R
        entry_time_utc=entry_monday,
        holding_bars_h4=24,
    )
    assert friction.swap_to_1r_ratio > 0.15
    assert friction.is_hostile_swap_vetoed is True
