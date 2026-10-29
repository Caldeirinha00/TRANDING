"""
AI Trading System — Dynamic Portfolio Risk Manager
Implements EWMA Covariance, Ledoit-Wolf Shrinkage, and Online PCA (Effective Number of Bets).
"""
import numpy as np
import logging

logger = logging.getLogger("dynamic_risk")


class DynamicPortfolioRiskManager:
    def __init__(self, max_portfolio_risk_pct: float = 2.0, min_effective_bets: float = 1.35):
        self.max_portfolio_risk_pct = max_portfolio_risk_pct
        self.min_effective_bets = min_effective_bets

    def evaluate_new_trade_scaling(
        self,
        returns_matrix: np.ndarray,  # Shape (T, N_assets) with M5 returns of open + candidate trades
        risk_weights: np.ndarray,    # Shape (N_assets,) with % risk (+ for Long, - for Short)
        ewma_lambda: float = 0.94,
        shrinkage: float = 0.10
    ) -> float:
        """
        Returns a scaling multiplier [0.0, 1.0] to scale down candidate lot size
        if portfolio correlation risk or factor concentration (ENB) exceeds threshold.
        """
        if returns_matrix is None or returns_matrix.size == 0 or len(risk_weights) == 0:
            return 1.0

        T, N = returns_matrix.shape
        if N <= 1:
            candidate_risk = abs(float(risk_weights[0]))
            if candidate_risk <= self.max_portfolio_risk_pct:
                return 1.0
            return float(self.max_portfolio_risk_pct / candidate_risk)

        # 1. EWMA Weighted Covariance Matrix + Shrinkage to Identity
        weights_t = (1.0 - ewma_lambda) * (ewma_lambda ** np.arange(T - 1, -1, -1))
        sum_w = weights_t.sum()
        if sum_w <= 0:
            return 1.0
        weights_t /= sum_w

        centered = returns_matrix - np.average(returns_matrix, axis=0, weights=weights_t)
        cov = (centered.T * weights_t) @ centered
        std = np.sqrt(np.diag(cov)) + 1e-12
        corr = cov / np.outer(std, std)
        corr_shrunk = (1.0 - shrinkage) * corr + shrinkage * np.eye(N)

        # 2. Correlation-Adjusted Portfolio Equivalent Risk: sqrt(w^T * Corr * w)
        portfolio_equiv_risk_pct = float(np.sqrt(np.maximum(0.0, risk_weights.T @ corr_shrunk @ risk_weights)))

        # 3. Online PCA: Effective Number of Bets (Shannon Entropy of Eigenvalues)
        eigvals = np.linalg.eigvalsh(corr_shrunk)
        eigvals = np.maximum(eigvals, 1e-10)
        p = eigvals / eigvals.sum()
        enb = float(np.exp(-np.sum(p * np.log(p))))

        scale = 1.0
        if portfolio_equiv_risk_pct > self.max_portfolio_risk_pct:
            scale *= (self.max_portfolio_risk_pct / portfolio_equiv_risk_pct)

        # Penalize if portfolio is heavily collinear (all betting on the same common factor like USD)
        if enb < self.min_effective_bets:
            enb_penalty = enb / self.min_effective_bets
            scale *= enb_penalty
            logger.info(f"Dynamic Risk: ENB={enb:.2f} < {self.min_effective_bets:.2f}. Scaling factor by {enb_penalty:.2f}")

        return round(max(0.0, min(1.0, scale)), 4)


dynamic_risk_manager = DynamicPortfolioRiskManager()
