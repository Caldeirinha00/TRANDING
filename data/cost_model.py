from dataclasses import dataclass
from datetime import datetime, timedelta
import structlog
from core.mt5_connector import SymbolContractSpec
from config.settings import settings

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class TradeFrictionEstimate:
    symbol: str
    lots: float
    spread_cost_eur: float
    commission_roundtrip_eur: float
    expected_swap_eur: float
    total_friction_eur: float
    gross_1r_risk_eur: float
    friction_to_risk_ratio: float
    swap_to_1r_ratio: float
    is_hostile_swap_vetoed: bool


class CostModel:
    """
    Implementa a equação contabilística exata do Plano Diretor:
    C_total = S_entry + K_round-trip + sum_{d=1..N} (Phi_swap(s, d) * M(d))
    Incluindo o multiplicador triplo M(d) = 3 na noite de quarta-feira (rollover T+2).
    """

    def __init__(self, commission_per_lot_roundtrip_eur: float | None = None) -> None:
        # Pepperstone Standard embeds fee in spread (0.0 EUR commission); Razor is ~6.50 EUR/lot round-trip.
        self.commission_per_lot_roundtrip_eur = (
            settings.commission_per_lot_roundtrip_eur
            if commission_per_lot_roundtrip_eur is None
            else commission_per_lot_roundtrip_eur
        )

    @staticmethod
    def value_per_price_unit_for_lots(spec: SymbolContractSpec, lots: float, for_loss: bool = True) -> float:
        """
        Converte uma distância de preço (ex.: 0.0050 em EURUSD) diretamente para EUR
        usando o trade_tick_value_loss/profit reportado pelo servidor MT5 para a conta em EUR.
        """
        if spec.trade_tick_size <= 0:
            raise ValueError(f"trade_tick_size inválido para {spec.symbol}")
        tick_val = spec.trade_tick_value_loss if for_loss else spec.trade_tick_value_profit
        if tick_val <= 0:
            tick_val = spec.trade_tick_value
        ticks_per_unit = 1.0 / spec.trade_tick_size
        return ticks_per_unit * tick_val * lots

    def calculate_sl_risk_eur(self, spec: SymbolContractSpec, sl_distance_price: float, lots: float) -> float:
        """Risco nominal em EUR caso o Stop Loss seja atingido (antes de custos)."""
        if sl_distance_price <= 0 or lots <= 0:
            return 0.0
        val_per_unit = self.value_per_price_unit_for_lots(spec, lots, for_loss=True)
        return round(sl_distance_price * val_per_unit, 4)

    def calculate_daily_swap_eur(self, spec: SymbolContractSpec, side: int, lots: float) -> float:
        """
        Retorna o impacto em EUR de 1 dia de swap (positivo = crédito, negativo = custo).
        Em MT5 Forex (SYMBOL_SWAP_MODE_POINTS = 1), swap_long/short é expresso em points.
        """
        if spec.swap_mode == 0:  # SYMBOL_SWAP_MODE_DISABLED
            return 0.0

        raw_swap = spec.swap_long if side > 0 else spec.swap_short
        if spec.swap_mode == 1:  # SYMBOL_SWAP_MODE_POINTS
            # No modo 1 (points), 1 ponto = spec.point de variação de preço
            point_value_eur = self.value_per_price_unit_for_lots(spec, lots, for_loss=(raw_swap < 0)) * spec.point
            return raw_swap * point_value_eur

        if spec.swap_mode == 4:  # SYMBOL_SWAP_MODE_CURRENCY_DEPOSIT (already in account currency per 1.0 lot)
            return raw_swap * lots

        logger.warning("unsupported_mt5_swap_mode", symbol=spec.symbol, swap_mode=spec.swap_mode)
        return 0.0

    @staticmethod
    def rollover_multiplier_for_weekday(weekday_iso: int, rollover3days_mt5: int = 3) -> int:
        """
        Devolve M(d) = 3 na quarta-feira (weekday() == 2 em Python / 3 no MT5),
        1 nos restantes dias úteis, e 0 ao fim de semana.
        """
        # Em Python datetime.weekday(): Seg=0, Ter=1, Qua=2, Qui=3, Sex=4, Sáb=5, Dom=6
        # Em MT5 swap_rollover3days: Dom=0, Seg=1, Ter=2, Qua=3, Qui=4, Sex=5
        mt5_day = weekday_iso + 1 if weekday_iso < 6 else 0
        if weekday_iso in (5, 6):
            return 0
        return 3 if mt5_day == rollover3days_mt5 else 1

    def compute_rollover_swap_between(
        self,
        spec: SymbolContractSpec,
        side: int,
        lots: float,
        prev_time: datetime,
        curr_time: datetime,
    ) -> float:
        """
        Compute realized swap EUR across all 00:00 server rollover boundaries strictly in (prev_time, curr_time].
        Reused identically by holding projections and event-driven backtesting.
        """
        if curr_time <= prev_time:
            return 0.0
        daily_swap_eur = self.calculate_daily_swap_eur(spec, side, lots)
        if daily_swap_eur == 0.0:
            return 0.0

        total_swap_eur = 0.0
        cursor = datetime(prev_time.year, prev_time.month, prev_time.day, tzinfo=prev_time.tzinfo) + timedelta(days=1)
        while cursor <= curr_time:
            session_day = (cursor - timedelta(hours=1)).weekday()
            mult = self.rollover_multiplier_for_weekday(session_day, spec.swap_rollover3days)
            total_swap_eur += daily_swap_eur * mult
            cursor += timedelta(days=1)
        return round(total_swap_eur, 6)

    def project_holding_swap_eur(
        self,
        spec: SymbolContractSpec,
        side: int,
        lots: float,
        entry_time_utc: datetime,
        holding_bars_h4: int = 24,
    ) -> float:
        """
        Calcula o swap acumulado projetado para as próximas `holding_bars_h4` barras H4
        (padrão: 24 barras = 4 dias úteis, incluindo uma quarta-feira de triplo rollover).
        Retorna valor negativo se for um custo líquido para a posição.
        """
        end_dt = entry_time_utc + timedelta(hours=holding_bars_h4 * 4)
        return round(self.compute_rollover_swap_between(spec, side, lots, entry_time_utc, end_dt), 4)

    def evaluate_trade_friction(
        self,
        spec: SymbolContractSpec,
        side: int,
        lots: float,
        sl_distance_price: float,
        entry_time_utc: datetime,
        holding_bars_h4: int = 24,
        max_swap_ratio_override: float | None = None,
    ) -> TradeFrictionEstimate:
        gross_1r_eur = self.calculate_sl_risk_eur(spec, sl_distance_price, lots)
        spread_price = spec.spread_points * spec.point
        spread_cost_eur = self.calculate_sl_risk_eur(spec, spread_price, lots)
        commission_eur = round(self.commission_per_lot_roundtrip_eur * lots, 4)

        projected_swap_eur = self.project_holding_swap_eur(
            spec=spec,
            side=side,
            lots=lots,
            entry_time_utc=entry_time_utc,
            holding_bars_h4=holding_bars_h4,
        )
        # Custo de swap (se projected_swap_eur < 0, representa um encargo positivo de custo)
        adverse_swap_cost_eur = max(0.0, -projected_swap_eur)
        total_friction_eur = round(spread_cost_eur + commission_eur + adverse_swap_cost_eur, 4)

        friction_ratio = ((spread_cost_eur + commission_eur) / gross_1r_eur) if gross_1r_eur > 0 else 1.0
        swap_to_1r_ratio = (adverse_swap_cost_eur / gross_1r_eur) if gross_1r_eur > 0 else 1.0

        # Veto de Financiamento Hostil: Custo de swap projetado > limite de 1R (default settings ou regime override)
        max_ratio = settings.max_swap_to_1r_ratio if max_swap_ratio_override is None else max_swap_ratio_override
        is_vetoed = swap_to_1r_ratio > max_ratio

        return TradeFrictionEstimate(
            symbol=spec.symbol,
            lots=lots,
            spread_cost_eur=round(spread_cost_eur, 4),
            commission_roundtrip_eur=commission_eur,
            expected_swap_eur=projected_swap_eur,
            total_friction_eur=total_friction_eur,
            gross_1r_risk_eur=gross_1r_eur,
            friction_to_risk_ratio=round(friction_ratio, 4),
            swap_to_1r_ratio=round(swap_to_1r_ratio, 4),
            is_hostile_swap_vetoed=is_vetoed,
        )
