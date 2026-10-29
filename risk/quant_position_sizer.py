import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AccountState:
    equity_eur: float
    free_margin_eur: float
    used_margin_eur: float
    daily_dd_pct: float
    open_positions_count: int
    open_correlated_risk_eur: float  # Soma de Σ |ρ_ij| * risco_aberto_j


@dataclass(frozen=True)
class SizingResult:
    lots: float
    risk_eur: float
    risk_pct: float
    required_margin_eur: float
    capped_by_margin: bool


class InstitutionalPositionSizer:
    def __init__(
        self,
        max_single_risk_pct: float = 0.01,        # 1.0% (10 EUR em conta de 1.000 EUR)
        max_daily_dd_pct: float = 0.03,           # 3.0% (30 EUR)
        esma_leverage: float = 30.0,              # 1:30 ESMA Retail
        max_total_margin_usage_pct: float = 0.45, # Máx 45% da equity retida em margem
        max_concurrent_positions: int = 3,
        prior_win_rate: float = 0.53,
        prior_strength: float = 12.0,
    ) -> None:
        self.max_single_risk_pct = max_single_risk_pct
        self.max_daily_dd_pct = max_daily_dd_pct
        self.esma_leverage = esma_leverage
        self.max_total_margin_usage_pct = max_total_margin_usage_pct
        self.max_concurrent_positions = max_concurrent_positions
        self.prior_win_rate = prior_win_rate
        self.prior_strength = prior_strength

    @staticmethod
    def apply_higher_moment_kelly_penalty(
        half_kelly_raw: float,
        skewness: float = -0.35,
        kurtosis: float = 5.20,
        sigma_r: float = 1.15,
    ) -> float:
        """
        Ajusta a fração de Half-Kelly pelo 3.º momento (Skewness) e 4.º momento (Kurtosis)
        da distribuição empírica de retornos em unidades R via expansão de Taylor de 4.ª ordem.
        """
        if half_kelly_raw <= 0.0:
            return 0.0

        f0 = half_kelly_raw * 2.0  # Kelly completo base
        excess_kurt = max(0.0, kurtosis - 3.0)

        # Penalização analítica de Taylor de 4.ª ordem
        tail_denominator = (
            1.0
            - (f0 * skewness * sigma_r)
            + (0.3333 * (f0 ** 2) * excess_kurt * (sigma_r ** 2))
        )
        tail_multiplier = 1.0 / max(0.65, min(2.50, tail_denominator))
        return half_kelly_raw * tail_multiplier

    def calculate_eurusd_lots(
        self,
        account: AccountState,
        eurusd_ask: float,
        sl_distance_price: float,
        wins: int,
        total_trades: int,
        avg_win_pips: float,
        avg_loss_pips: float,
        expected_exec_cost_pips: float,
        atr_current: float,
        atr_median: float,
        meta_prob: float,
        skewness: float = -0.35,
        kurtosis: float = 5.20,
        sigma_r: float = 1.15,
    ) -> SizingResult:
        if account.open_positions_count >= self.max_concurrent_positions:
            return SizingResult(0.0, 0.0, 0.0, 0.0, False)

        # 1. Buffer de Drawdown Diário Restante (Desaceleração quadrática perto do limite de 3%)
        remaining_daily_dd = self.max_daily_dd_pct - account.daily_dd_pct
        if remaining_daily_dd <= 0.0025:  # Menos de 0.25% restante -> Bloqueia
            return SizingResult(0.0, 0.0, 0.0, 0.0, False)

        # 2. Posterior Bayesiano + Fusão com Probabilidade do Meta-Modelo ONNX
        p_bayes = (wins + self.prior_strength * self.prior_win_rate) / (total_trades + self.prior_strength)
        p_effective = 0.60 * p_bayes + 0.40 * meta_prob

        net_win = max(0.1, avg_win_pips - expected_exec_cost_pips)
        net_loss = max(0.1, avg_loss_pips + expected_exec_cost_pips)
        b_eff = net_win / net_loss

        kelly_raw = max(0.0, (p_effective * b_eff - (1.0 - p_effective)) / b_eff)
        half_kelly_base = 0.50 * kelly_raw
        half_kelly = self.apply_higher_moment_kelly_penalty(
            half_kelly_base, skewness=skewness, kurtosis=kurtosis, sigma_r=sigma_r
        )

        # 3. Penalização de Covariância e Escalonamento de Amostra
        sample_scale = min(1.0, max(0.25, total_trades / 20.0))
        max_base_risk_eur = account.equity_eur * self.max_single_risk_pct
        cov_penalty = 1.0 / (1.0 + (account.open_correlated_risk_eur / max(1.0, max_base_risk_eur)))

        # 4. Volatility Targeting
        vol_scale = min(1.25, max(0.50, atr_median / max(atr_current, 1e-6)))

        target_risk_pct = min(
            self.max_single_risk_pct,
            remaining_daily_dd * 0.50,  # Nunca arrisca mais de metade do DD diário disponível
            half_kelly * sample_scale * cov_penalty * vol_scale,
        )

        risk_budget_eur = account.equity_eur * target_risk_pct
        if risk_budget_eur <= 0.50 or sl_distance_price <= 1e-5 or eurusd_ask <= 0.0:
            return SizingResult(0.0, 0.0, 0.0, 0.0, False)

        # 5. Conversão Exata de Risco em EUR para Lotes de EURUSD
        # Perda em EUR por 1.00 lote = (100,000 * sl_distance_price) / eurusd_ask
        loss_per_std_lot_eur = (100_000.0 * sl_distance_price) / eurusd_ask
        raw_lots_by_risk = risk_budget_eur / loss_per_std_lot_eur

        # 6. Barreira de Margem ESMA 1:30 (1.00 lote EURUSD requer 3.333,33 EUR de margem)
        margin_per_std_lot_eur = 100_000.0 / self.esma_leverage
        max_allowed_margin_eur = max(
            0.0,
            (account.equity_eur * self.max_total_margin_usage_pct) - account.used_margin_eur,
        )
        # Limite por slot disponível
        slot_margin_cap_eur = min(max_allowed_margin_eur, (account.equity_eur * 0.15))

        max_lots_by_margin = slot_margin_cap_eur / margin_per_std_lot_eur
        capped_by_margin = raw_lots_by_risk > max_lots_by_margin

        final_raw_lots = min(raw_lots_by_risk, max_lots_by_margin)

        # 7. Quantização Estrita por Defeito (Floor ao step de 0.01 lote)
        lot_step = 0.01
        min_lot = 0.01
        max_lot = 10.0

        quantized_lots = math.floor(final_raw_lots / lot_step) * lot_step
        quantized_lots = round(quantized_lots, 2)

        if quantized_lots < min_lot:
            return SizingResult(0.0, 0.0, 0.0, 0.0, capped_by_margin)

        quantized_lots = min(quantized_lots, max_lot)
        actual_risk_eur = quantized_lots * loss_per_std_lot_eur
        actual_margin_eur = quantized_lots * margin_per_std_lot_eur

        # Verificação final de segurança: se mesmo 0.01 lote violar 1.25x do risco máximo, aborta
        if actual_risk_eur > (account.equity_eur * self.max_single_risk_pct * 1.25):
            return SizingResult(0.0, 0.0, 0.0, 0.0, False)

        return SizingResult(
            lots=quantized_lots,
            risk_eur=round(actual_risk_eur, 2),
            risk_pct=round(actual_risk_eur / account.equity_eur, 5),
            required_margin_eur=round(actual_margin_eur, 2),
            capped_by_margin=capped_by_margin,
        )
