"""Ornstein-Uhlenbeck -- mean reversion.

Log price is pulled back toward a fixed level with strength `theta`:

    d(log S) = theta * (level - log S) dt + sigma dW

The opposite of momentum. Prices that run up tend to come back down, and the
signal lives in the *level* rather than in recent returns.

This generator exists to stop the AI from learning one trick. A model that only
ever sees trending markets will happily conclude "buy what went up" is a law of
nature; pointed at a mean-reverting market it will lose money steadily while
remaining perfectly confident. Testing against both directions is the only way to
tell a model that detects structure from one that has memorised a direction.

Half-life
---------
The intuitive parameter is not theta but the half-life, ln(2) / theta: how long a
deviation from the level takes to decay by half. A half-life of 0.25 years means
shocks wash out over roughly three months. Real mean reversion in equities, where
it exists at all, is slow and unreliable.

The path always starts exactly at the mean level, since the price frame is
anchored at `initial_price`. Expect roughly one half-life of burn-in before the
process reaches its stationary distribution.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import lfilter

from sentinel.sandbox.generators.base import TRADING_DAYS_PER_YEAR, Generator, Simulation


class OUGenerator(Generator):
    """Mean-reverting markets.

    Args:
        theta: annualised reversion speed. Higher pulls back harder.
        sigma: annualised volatility of the driving noise. Note the *stationary*
            volatility of log price is sigma / sqrt(2 * theta), not sigma.
        half_life: convenience alternative to theta, in years. If given, theta is
            derived as ln(2) / half_life.
    """

    model_name = "ou"
    has_exploitable_signal = True

    def __init__(
        self,
        theta: float = 3.0,
        sigma: float = 0.16,
        half_life: float | None = None,
        correlation: np.ndarray | None = None,
        initial_price: float = 100.0,
        trading_days: int = TRADING_DAYS_PER_YEAR,
    ) -> None:
        super().__init__(correlation, initial_price, trading_days)
        if half_life is not None:
            if half_life <= 0:
                raise ValueError("half_life must be positive")
            theta = np.log(2.0) / float(half_life)
        if float(theta) <= 0:
            raise ValueError("theta must be positive; use GBMGenerator for no reversion")
        if float(sigma) < 0:
            raise ValueError("volatility cannot be negative")
        self.theta = float(theta)
        self.sigma = float(sigma)

    @property
    def half_life_years(self) -> float:
        return float(np.log(2.0) / self.theta)

    def _simulate(self, n_steps: int, n_assets: int, rng: np.random.Generator) -> Simulation:
        n = n_steps - 1
        dt = self.dt

        # Exact discretisation, not an Euler approximation. The OU process has a
        # closed-form transition density, so there is no reason to accept
        # discretisation error -- and at large theta the Euler scheme is visibly
        # wrong, which would show up as the wrong stationary volatility.
        decay = np.exp(-self.theta * dt)
        shock_scale = self.sigma * np.sqrt((1.0 - decay**2) / (2.0 * self.theta))

        innovations = shock_scale * self._correlated_shocks(n, n_assets, rng)

        # Deviations from the mean level follow an AR(1) with coefficient `decay`,
        # starting at zero because the price frame is anchored at initial_price.
        deviations, _ = lfilter(
            [1.0], [1.0, -decay], innovations, axis=0, zi=np.zeros((1, n_assets))
        )

        # Log returns are the *changes* in deviation, with the first step measured
        # from the starting level of zero.
        log_returns = np.diff(deviations, axis=0, prepend=np.zeros((1, n_assets)))

        return Simulation(
            log_returns=log_returns,
            extra={
                "theta": self.theta,
                "sigma": self.sigma,
                "half_life_years": self.half_life_years,
                "stationary_sd_log_price": float(self.sigma / np.sqrt(2 * self.theta)),
            },
        )

    def _params(self, n_assets: int) -> dict[str, Any]:
        return {}
