"""Geometric Brownian motion -- the null market.

This is the most important generator in the project, and the reason is that it
contains **nothing**. Prices wander with a fixed drift and a fixed volatility, and
each day's move is drawn independently of every day before it. There is no trend
to ride, no level to revert to, no regime to detect. The past tells you nothing
about the future beyond today's price.

That emptiness is the point. It is the control group for every experiment we run.
An AI turned loose on this market *must* fail to make money, and when it does not
fail, we have found a bug in our own code rather than an edge in the market.

The model
---------
The exact solution of the GBM stochastic differential equation gives log returns
that are independent draws from

    log(S_t / S_t-1) ~ Normal( (mu - sigma^2/2) * dt,  sigma^2 * dt )

The `- sigma^2/2` correction is easy to miss and matters. `mu` is the drift of the
*price*, but we simulate *log* prices, and because volatility drags on compounded
growth the log-space drift is lower than mu by exactly sigma^2/2. Omit it and the
generator quietly produces higher returns than requested -- the kind of small bias
that would flow into every downstream result.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sentinel.sandbox.generators.base import TRADING_DAYS_PER_YEAR, Generator, Simulation


class GBMGenerator(Generator):
    """Correlated random-walk markets with known parameters.

    Args:
        mu: annualised drift, scalar or per asset. 0.08 is a reasonable stand-in
            for long-run equity returns.
        sigma: annualised volatility, scalar or per asset. 0.16 is roughly the
            historical volatility of the S&P 500.
        correlation: asset correlation matrix, defaulting to independence. Real
            equity markets are far more correlated than that.
    """

    model_name = "gbm"
    has_exploitable_signal = False

    def __init__(
        self,
        mu: float | np.ndarray = 0.08,
        sigma: float | np.ndarray = 0.16,
        correlation: np.ndarray | None = None,
        initial_price: float = 100.0,
        trading_days: int = TRADING_DAYS_PER_YEAR,
    ) -> None:
        super().__init__(correlation, initial_price, trading_days)
        self.mu = mu
        self.sigma = sigma

    def _simulate(self, n_steps: int, n_assets: int, rng: np.random.Generator) -> Simulation:
        mu = self._broadcast(self.mu, n_assets, "mu")
        sigma = self._broadcast(self.sigma, n_assets, "sigma")
        if (sigma < 0).any():
            raise ValueError("volatility cannot be negative")

        log_drift = (mu - 0.5 * sigma**2) * self.dt
        log_vol = sigma * np.sqrt(self.dt)
        shocks = self._correlated_shocks(n_steps - 1, n_assets, rng)

        return Simulation(
            log_returns=log_drift + log_vol * shocks,
            extra={
                "mu": mu.tolist(),
                "sigma": sigma.tolist(),
                "correlation": self._resolve_correlation(n_assets).tolist(),
            },
        )

    def _params(self, n_assets: int) -> dict[str, Any]:
        return {}
