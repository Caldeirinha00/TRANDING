from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from config.settings import settings
from core.mt5_connector import SymbolContractSpec
from data.cost_model import CostModel
from risk.currency_guard import CurrencyAndHeatGuard, OpenPositionRiskState
from strategy.primary_trend import PrimaryStrategyConfig


@dataclass(frozen=True)
class CandidateSignal:
    symbol: str
    side: int                   # +1 (Buy) ou -1 (Sell)
    entry_price: float
    sl_price: float
    tp_price: float
    meta_probability: float     # p in [0, 1] proveniente da Camada 2 (Meta-Labeling)
    signal_time_utc: datetime
    ker_d1: float = 0.0
    priority_score: Optional[float] = None
    holding_bars_h4: Optional[int] = None


@dataclass(frozen=True)
class SizingDecision:
    symbol: str
    side: int
    approved: bool
    allocated_lots: float       # 0.00, 0.01 ou 0.02
    risk_eur: float
    risk_pct_equity: float
    expected_net_r: float       # E[R_net] = p * R_win_net - (1 - p) * R_loss_net
    margin_required_eur: float
    rejection_reason: str


class PortfolioRiskSizer:
    """
    Motor de dimensionamento discreto sob regulação ESMA para conta de 1.000 EUR.
    Garante:
    - Exclusão estrita de Ouro/Índices enquanto Equity < 3.000 EUR.
    - Risco máximo por trade <= 1,00% da Equity (<= 10,00 EUR em 1.000 EUR).
    - Alocação discreta guiada por Meta-Labeling (0.00, 0.01 ou 0.02 lotes).
    - Ranqueamento Cross-Seccional por Valor Esperado Líquido E[R_net].
    """

    def __init__(
        self,
        cost_model: Optional[CostModel] = None,
        guard: Optional[CurrencyAndHeatGuard] = None,
        use_meta_labeling: bool = True,
        strategy_cfg: Optional[PrimaryStrategyConfig] = None,
    ) -> None:
        self.cost_model = cost_model or CostModel()
        self.guard = guard or CurrencyAndHeatGuard()
        self.use_meta_labeling = use_meta_labeling
        self.strategy_cfg = strategy_cfg or PrimaryStrategyConfig()
        universe = settings.load_universe()
        self.approved_symbols = set(universe["approved_symbols"].keys())
        self.forbidden_under_3k = set(universe.get("forbidden_under_3000_eur", []))
        self.min_equity_for_metals = float(universe["account_constraints"]["min_equity_for_metals_eur"])

    def evaluate_single_candidate(
        self,
        signal: CandidateSignal,
        spec: SymbolContractSpec,
        equity_eur: float,
        open_positions: List[OpenPositionRiskState],
    ) -> SizingDecision:
        # 1. Proteção de Universo (Proibição de Ouro/Índices abaixo de 3.000 EUR)
        if signal.symbol in self.forbidden_under_3k and equity_eur < self.min_equity_for_metals:
            return SizingDecision(
                symbol=signal.symbol,
                side=signal.side,
                approved=False,
                allocated_lots=0.0,
                risk_eur=0.0,
                risk_pct_equity=0.0,
                expected_net_r=0.0,
                margin_required_eur=0.0,
                rejection_reason=f"VETO_UNIVERSE_CAPITAL: {signal.symbol} proibido com equity {equity_eur:.2f} EUR < {self.min_equity_for_metals:.2f} EUR.",
            )

        if signal.symbol not in self.approved_symbols:
            return SizingDecision(
                symbol=signal.symbol,
                side=signal.side,
                approved=False,
                allocated_lots=0.0,
                risk_eur=0.0,
                risk_pct_equity=0.0,
                expected_net_r=0.0,
                margin_required_eur=0.0,
                rejection_reason=f"VETO_UNAPPROVED_SYMBOL: {signal.symbol} fora da cesta aprovada.",
            )

        # 2. Filtro de Confiança Mínima do Meta-Modelo (apenas quando use_meta_labeling=True)
        p = signal.meta_probability
        if self.use_meta_labeling and p < settings.meta_min_probability:
            return SizingDecision(
                symbol=signal.symbol,
                side=signal.side,
                approved=False,
                allocated_lots=0.0,
                risk_eur=0.0,
                risk_pct_equity=0.0,
                expected_net_r=0.0,
                margin_required_eur=0.0,
                rejection_reason=f"VETO_META_PROBABILITY: p={p:.3f} < limiar {settings.meta_min_probability:.2f}.",
            )

        # 3. Distâncias estruturais de SL e TP
        sl_dist = abs(signal.entry_price - signal.sl_price)
        tp_dist = abs(signal.tp_price - signal.entry_price)
        if sl_dist <= 0 or tp_dist <= 0:
            return SizingDecision(
                symbol=signal.symbol,
                side=signal.side,
                approved=False,
                allocated_lots=0.0,
                risk_eur=0.0,
                risk_pct_equity=0.0,
                expected_net_r=0.0,
                margin_required_eur=0.0,
                rejection_reason="VETO_INVALID_STOPS: Distância SL ou TP nula.",
            )

        # 4. Avaliação de Custos, Swap (com teto adaptativo em modo Opção A) e Risco para lote base 0,01
        base_lots = settings.fixed_pass_through_lots if not self.use_meta_labeling else 0.01
        hold_bars = (
            signal.holding_bars_h4
            if signal.holding_bars_h4 is not None
            else (self.strategy_cfg.vertical_barrier_bars_h4 if not self.use_meta_labeling else 24)
        )
        swap_cap_override = None
        if not self.use_meta_labeling:
            swap_cap_override = (
                self.strategy_cfg.max_swap_strong_trend_ratio
                if signal.ker_d1 >= self.strategy_cfg.ker_strong_trend_threshold
                else self.strategy_cfg.max_swap_normal_ratio
            )

        friction_001 = self.cost_model.evaluate_trade_friction(
            spec=spec,
            side=signal.side,
            lots=base_lots,
            sl_distance_price=sl_dist,
            entry_time_utc=signal.signal_time_utc,
            holding_bars_h4=hold_bars,
            max_swap_ratio_override=swap_cap_override,
        )

        if friction_001.is_hostile_swap_vetoed:
            active_cap = settings.max_swap_to_1r_ratio if swap_cap_override is None else swap_cap_override
            return SizingDecision(
                symbol=signal.symbol,
                side=signal.side,
                approved=False,
                allocated_lots=0.0,
                risk_eur=friction_001.gross_1r_risk_eur,
                risk_pct_equity=round(friction_001.gross_1r_risk_eur / equity_eur, 4),
                expected_net_r=0.0,
                margin_required_eur=0.0,
                rejection_reason=(
                    f"VETO_HOSTILE_SWAP: Swap projetado consome {friction_001.swap_to_1r_ratio*100:.1f}% de 1R "
                    f"(máx permitido {active_cap*100:.1f}%)."
                ),
            )

        # 5. Valor Esperado Líquido / Prioridade (em Pass-Through não veta por E[R] ML)
        gross_rr = tp_dist / sl_dist
        total_friction_r = (
            friction_001.total_friction_eur / friction_001.gross_1r_risk_eur
            if friction_001.gross_1r_risk_eur > 0
            else 1.0
        )
        r_win_net = gross_rr - total_friction_r
        r_loss_net = 1.0 + total_friction_r
        expected_net_r = round((p * r_win_net) - ((1.0 - p) * r_loss_net), 4)

        if self.use_meta_labeling and expected_net_r < settings.meta_min_expected_net_r:
            return SizingDecision(
                symbol=signal.symbol,
                side=signal.side,
                approved=False,
                allocated_lots=0.0,
                risk_eur=friction_001.gross_1r_risk_eur,
                risk_pct_equity=round(friction_001.gross_1r_risk_eur / equity_eur, 4),
                expected_net_r=expected_net_r,
                margin_required_eur=0.0,
                rejection_reason=(
                    f"VETO_LOW_EXPECTED_R: E[R_net]={expected_net_r:.3f}R < {settings.meta_min_expected_net_r:.2f}R."
                ),
            )

        # 6. Verificação do Teto Unitário de 1,00% da Equity (<= 10,00 EUR em 1.000 EUR)
        max_unit_risk_eur = round(equity_eur * settings.max_risk_per_trade_pct, 4)
        risk_001_eur = friction_001.gross_1r_risk_eur

        if risk_001_eur > max_unit_risk_eur:
            return SizingDecision(
                symbol=signal.symbol,
                side=signal.side,
                approved=False,
                allocated_lots=0.0,
                risk_eur=risk_001_eur,
                risk_pct_equity=round(risk_001_eur / equity_eur, 4),
                expected_net_r=expected_net_r,
                margin_required_eur=0.0,
                rejection_reason=(
                    f"VETO_UNIT_RISK_EXCEEDED: {base_lots:.2f} lote arrisca {risk_001_eur:.2f} EUR > teto 1.0% ({max_unit_risk_eur:.2f} EUR)."
                ),
            )

        # 7. Alocador Discreto (0,01 fixo em Opção A Pass-Through; 0,01 vs 0,02 quando ML ativo)
        target_lots = base_lots
        target_risk_eur = risk_001_eur

        if self.use_meta_labeling and p >= settings.meta_high_conviction_probability:
            risk_002_eur = round(risk_001_eur * 2.0, 4)
            # Só escala para 0,02 lotes se 0,02 lotes respeitar simultaneamente o teto unitário de 1,0% E o Portfolio Heat de 2,10%
            can_fit_002, _ = self.guard.can_accept_new_risk(
                candidate_symbol=signal.symbol,
                candidate_side=signal.side,
                candidate_risk_eur=risk_002_eur,
                equity_eur=equity_eur,
                open_positions=open_positions,
            )
            if risk_002_eur <= max_unit_risk_eur and can_fit_002:
                target_lots = 0.02
                target_risk_eur = risk_002_eur

        # 8. Validação Final na Carteira (Portfolio Heat <= 2.10%, Max 3 Posições, Currency Bucket Guard)
        approved_by_guard, guard_reason = self.guard.can_accept_new_risk(
            candidate_symbol=signal.symbol,
            candidate_side=signal.side,
            candidate_risk_eur=target_risk_eur,
            equity_eur=equity_eur,
            open_positions=open_positions,
        )

        if not approved_by_guard:
            return SizingDecision(
                symbol=signal.symbol,
                side=signal.side,
                approved=False,
                allocated_lots=0.0,
                risk_eur=target_risk_eur,
                risk_pct_equity=round(target_risk_eur / equity_eur, 4),
                expected_net_r=expected_net_r,
                margin_required_eur=0.0,
                rejection_reason=guard_reason,
            )

        margin_req = round(spec.margin_initial_001_eur * (target_lots / 0.01), 2)
        return SizingDecision(
            symbol=signal.symbol,
            side=signal.side,
            approved=True,
            allocated_lots=target_lots,
            risk_eur=target_risk_eur,
            risk_pct_equity=round(target_risk_eur / equity_eur, 4),
            expected_net_r=expected_net_r,
            margin_required_eur=margin_req,
            rejection_reason="APPROVED",
        )

    def rank_and_allocate_batch(
        self,
        candidates: List[CandidateSignal],
        specs: dict[str, SymbolContractSpec],
        equity_eur: float,
        open_positions: List[OpenPositionRiskState],
    ) -> List[SizingDecision]:
        """
        Resolução de Conflitos Cross-Seccionais:
        Ordena todos os candidatos da barra H4 atual por E[R_net] decrescente e aloca
        sequencialmente atualizando o estado virtual da carteira a cada aprovação.
        """
        pre_evals = []
        for sig in candidates:
            spec = specs.get(sig.symbol)
            if spec is None:
                continue
            dec = self.evaluate_single_candidate(sig, spec, equity_eur, open_positions)
            rank_score = sig.priority_score if (not self.use_meta_labeling and sig.priority_score is not None) else dec.expected_net_r
            pre_evals.append((rank_score, sig, spec))

        # Ordenar por maior Valor Esperado Líquido E[R_net]
        pre_evals.sort(key=lambda item: item[0], reverse=True)

        simulated_open = list(open_positions)
        final_decisions: List[SizingDecision] = []

        for idx, (_, sig, spec) in enumerate(pre_evals):
            decision = self.evaluate_single_candidate(sig, spec, equity_eur, simulated_open)
            final_decisions.append(decision)
            if decision.approved:
                # Regista imediatamente a nova posição no estado simulado para bloquear conflitos de moeda/calor nos candidatos seguintes
                simulated_open.append(
                    OpenPositionRiskState(
                        ticket=900000 + idx,
                        symbol=sig.symbol,
                        side=sig.side,
                        lots=decision.allocated_lots,
                        entry_price=sig.entry_price,
                        current_sl=sig.sl_price,
                        open_risk_eur=decision.risk_eur,
                    )
                )

        return final_decisions
