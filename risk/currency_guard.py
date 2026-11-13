from dataclasses import dataclass
from typing import List, Set, Tuple
from config.settings import settings


@dataclass(frozen=True)
class OpenPositionRiskState:
    ticket: int
    symbol: str
    side: int  # +1 (Long) ou -1 (Short)
    lots: float
    entry_price: float
    current_sl: float
    open_risk_eur: float  # Risco financeiro até ao SL atual

    @property
    def is_at_breakeven(self) -> bool:
        """
        Verifica se o Stop Loss já protege o ponto de equilíbrio (Break-Even ou lucro travado).
        Posições em Break-Even têm risco residual nulo e libertam o orçamento de Portfolio Heat e Moeda.
        """
        if self.current_sl <= 0:
            return False
        if self.side > 0:
            return self.current_sl >= self.entry_price
        return self.current_sl <= self.entry_price

    @property
    def effective_active_risk_eur(self) -> float:
        return 0.0 if self.is_at_breakeven else max(0.0, self.open_risk_eur)


class CurrencyAndHeatGuard:
    """
    Implementa as 3 barreiras de carteira para 1.000 EUR sob regulação ESMA:
    1. Máximo de 3 posições concorrentes em risco ativo (fora de Break-Even).
    2. Max Portfolio Heat <= 2,10% da equity (~21,00 EUR).
    3. Currency Bucket Guard: Proíbe deter mais de 1 posição em risco ativo com exposição
       direcional idêntica à mesma divisa (ex.: Long EURUSD = Short USD bloqueia Long AUDUSD e Short USDCHF).
    """

    def __init__(self) -> None:
        self.max_active_positions = settings.max_active_risk_positions
        self.max_heat_pct = settings.max_portfolio_heat_pct

    @staticmethod
    def extract_currency_exposures(symbol: str, side: int) -> Tuple[Tuple[str, int], Tuple[str, int]]:
        """
        Decompõe um par cambial (ex.: 'EURUSD', side=+1) nas exposições direcionais das suas divisas:
        (('EUR', +1), ('USD', -1)).
        Se for Short USDCHF (side=-1): (('USD', -1), ('CHF', +1)).
        """
        clean = symbol.strip().upper()
        if len(clean) < 6:
            raise ValueError(f"Símbolo cambial inválido para decomposição: {symbol}")
        base_ccy = clean[:3]
        quote_ccy = clean[3:6]
        return ((base_ccy, side), (quote_ccy, -side))

    def get_active_currency_buckets(self, open_positions: List[OpenPositionRiskState]) -> Set[Tuple[str, int]]:
        """Retorna o conjunto de (divisa, direção) atualmente ocupados por posições em risco ativo."""
        occupied: Set[Tuple[str, int]] = set()
        for pos in open_positions:
            if not pos.is_at_breakeven:
                base_exp, quote_exp = self.extract_currency_exposures(pos.symbol, pos.side)
                occupied.add(base_exp)
                occupied.add(quote_exp)
        return occupied

    def current_portfolio_heat_eur(self, open_positions: List[OpenPositionRiskState]) -> float:
        return round(sum(pos.effective_active_risk_eur for pos in open_positions), 4)

    def active_risk_positions_count(self, open_positions: List[OpenPositionRiskState]) -> int:
        return sum(1 for pos in open_positions if not pos.is_at_breakeven)

    def can_accept_new_risk(
        self,
        candidate_symbol: str,
        candidate_side: int,
        candidate_risk_eur: float,
        equity_eur: float,
        open_positions: List[OpenPositionRiskState],
    ) -> Tuple[bool, str]:
        # 1. Evitar duplicação no mesmo símbolo em risco ativo
        for pos in open_positions:
            if pos.symbol == candidate_symbol and not pos.is_at_breakeven:
                return False, f"VETO_DUPLICATE_SYMBOL: {candidate_symbol} já possui posição em risco ativo."

        # 2. Limite de 3 posições concorrentes em risco ativo
        active_count = self.active_risk_positions_count(open_positions)
        if active_count >= self.max_active_positions:
            return (
                False,
                f"VETO_MAX_CONCURRENT_POSITIONS: {active_count}/{self.max_active_positions} posições ativas.",
            )

        # 3. Limite de Calor da Carteira (Max Portfolio Heat <= 2,10% da equity)
        current_heat = self.current_portfolio_heat_eur(open_positions)
        max_heat_eur = round(equity_eur * self.max_heat_pct, 4)
        projected_heat = round(current_heat + candidate_risk_eur, 4)
        if projected_heat > max_heat_eur:
            return (
                False,
                f"VETO_PORTFOLIO_HEAT: Calor projetado {projected_heat:.2f} EUR excede teto de {max_heat_eur:.2f} EUR (2.10%).",
            )

        # 4. Currency Bucket Guard (Exposição direcional idêntica por moeda)
        occupied_buckets = self.get_active_currency_buckets(open_positions)
        cand_base_exp, cand_quote_exp = self.extract_currency_exposures(candidate_symbol, candidate_side)

        if cand_base_exp in occupied_buckets:
            dir_str = "LONG" if cand_base_exp[1] > 0 else "SHORT"
            return (
                False,
                f"VETO_CURRENCY_BUCKET: Exposição direcional {dir_str} {cand_base_exp[0]} já ativa na carteira.",
            )
        if cand_quote_exp in occupied_buckets:
            dir_str = "LONG" if cand_quote_exp[1] > 0 else "SHORT"
            return (
                False,
                f"VETO_CURRENCY_BUCKET: Exposição direcional {dir_str} {cand_quote_exp[0]} já ativa na carteira.",
            )

        return True, "APPROVED"
