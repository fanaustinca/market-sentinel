"""Portfolios built from a covariance forecast alone, with no view on returns.

Every strategy added to this project so far has been a bet, in some disguise, on
being able to say something about future returns -- trend, regime, sentiment,
momentum. Each was measured and each came back at a significance the project
declined to act on. The strategies here make no such bet. They are functions of
the covariance matrix and nothing else, so the only thing they can be wrong about
is risk, which is the thing this project can actually estimate.

`RiskParity` equalises how much each asset contributes to portfolio risk. With
the correlation structure ignored this reduces to inverse-volatility weighting,
which is deliberately the version implemented: it needs only the diagonal, the
best-measured part of the matrix, and it cannot be destabilised by a spurious
correlation estimate.

`MinimumVariance` uses the whole matrix, solving for the lowest-variance long-only
portfolio. It is strictly more ambitious and strictly more fragile -- the
optimiser is drawn to whichever pair of assets happened to look most mutually
hedging, and on short samples that is usually an artefact. Shrinkage in the
covariance estimate is what makes it usable at all, and comparing the two
strategies is how the value of the off-diagonal terms gets measured rather than
assumed.

Both are then scaled to a volatility target, for a reason that is easy to skip
past: without it a comparison between them is meaningless. A minimum-variance
portfolio is, by construction, lower-volatility than an equal-weight one, so it
will show a lower return and a higher Sharpe on almost any data. Scaling both to
the same forecast risk removes that tautology and asks the question actually worth
asking -- at the *same* risk, which allocates it better?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.risk.covariance import EWMACovariance
from sentinel.sandbox.market import MarketData
from sentinel.strategies.base import Strategy

TRADING_DAYS_PER_YEAR = 252


class CovariancePortfolio(Strategy):
    """Shared machinery: forecast the matrix, choose weights, scale to a risk budget.

    Args:
        target_volatility: annualised portfolio risk to aim for. Matches
            `VolatilityTarget`'s 12% so the two are directly comparable.
        max_gross: hard cap on total exposure. 1.0 means no leverage, ever --
            a standing policy in this project rather than a parameter.
        rebalance_days: how often weights are recomputed. Daily reoptimisation of
            a noisy matrix produces enormous turnover for changes that are mostly
            estimation noise; monthly is both cheaper and more stable.
        floor_volatility: the portfolio risk estimate is floored before dividing,
            so an unusually calm stretch cannot produce an enormous position.
    """

    def __init__(
        self,
        target_volatility: float = 0.12,
        max_gross: float = 1.0,
        rebalance_days: int = 21,
        floor_volatility: float = 0.04,
        covariance: EWMACovariance | None = None,
    ) -> None:
        if target_volatility <= 0:
            raise ValueError("target_volatility must be positive")
        if not 0.0 < max_gross <= 1.0:
            raise ValueError("max_gross must be in (0, 1]; this project does not use leverage")
        if rebalance_days < 1:
            raise ValueError("rebalance_days must be at least 1")
        if floor_volatility <= 0:
            raise ValueError("floor_volatility must be positive")

        self.target_volatility = float(target_volatility)
        self.max_gross = float(max_gross)
        self.rebalance_days = int(rebalance_days)
        self.floor_volatility = float(floor_volatility)
        self.covariance = covariance or EWMACovariance()

    def allocate(self, matrix: np.ndarray) -> np.ndarray:
        """Unscaled long-only weights summing to 1. Overridden by each subclass."""
        raise NotImplementedError

    def compute_weights(self, data: MarketData) -> pd.DataFrame:
        prices = data.prices
        matrices = self.covariance.estimate(prices)
        n_assets = prices.shape[1]

        weights = np.zeros((len(prices), n_assets))
        current = np.zeros(n_assets)
        last_rebalance: int | None = None

        for t, matrix in enumerate(matrices):
            if matrix is None:
                continue

            due = last_rebalance is None or (t - last_rebalance) >= self.rebalance_days
            if due:
                raw = self.allocate(matrix)
                # Forecast risk of the chosen mix, annualised. This is the number
                # the budget divides, and it uses the same matrix the weights came
                # from, so the two cannot disagree.
                variance = float(raw @ matrix @ raw)
                risk = np.sqrt(max(variance, 0.0) * TRADING_DAYS_PER_YEAR)
                scale = self.target_volatility / max(risk, self.floor_volatility)
                current = raw * min(scale, self.max_gross / max(raw.sum(), 1e-12))
                last_rebalance = t

            weights[t] = current

        return pd.DataFrame(weights, index=prices.index, columns=prices.columns)


class RiskParity(CovariancePortfolio):
    """Inverse-volatility weights: every asset contributes equal risk.

    Uses only the diagonal of the covariance matrix. That is a real limitation and
    also the source of its robustness -- individual volatilities are the part of
    the matrix estimated from `n` numbers rather than `n squared`, and they are
    what this project has repeatedly shown to be forecastable.
    """

    name = "risk_parity"

    def allocate(self, matrix: np.ndarray) -> np.ndarray:
        volatilities = np.sqrt(np.diag(matrix))
        inverse = np.where(volatilities > 1e-12, 1.0 / np.maximum(volatilities, 1e-12), 0.0)
        total = inverse.sum()
        return inverse / total if total > 0 else np.full(len(inverse), 1.0 / len(inverse))


class MinimumVariance(CovariancePortfolio):
    """The lowest-variance long-only portfolio the forecast matrix allows.

    The long-only constraint is doing more work than it appears to. Unconstrained
    minimum variance takes large offsetting long and short positions in assets
    whose estimated correlation is near one, which is exactly where estimation
    error is largest, and is the standard way this portfolio blows up. Forbidding
    shorts caps the damage any single bad correlation estimate can do, and it is
    already this project's policy for unrelated reasons.
    """

    name = "minimum_variance"

    def allocate(self, matrix: np.ndarray) -> np.ndarray:
        from scipy.optimize import minimize

        n = matrix.shape[0]
        start = np.full(n, 1.0 / n)
        if n == 1:
            return start

        result = minimize(
            lambda w: float(w @ matrix @ w),
            start,
            jac=lambda w: 2.0 * matrix @ w,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
            options={"maxiter": 200, "ftol": 1e-12},
        )
        # A failed solve falls back to equal weight rather than to whatever the
        # optimiser reached. A partially converged minimum-variance solution is
        # not a conservative answer; it is an arbitrary one.
        if not result.success or not np.all(np.isfinite(result.x)):
            return start
        weights = np.clip(result.x, 0.0, 1.0)
        total = weights.sum()
        return weights / total if total > 0 else start
