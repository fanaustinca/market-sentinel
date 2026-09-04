"""AR(1) momentum -- the market with a signal of known strength.

This is the Recovery Test's instrument. Returns follow

    r_t = m + phi * (r_t-1 - m) + eps_t

so `phi` is exactly the lag-1 autocorrelation of returns: a dial that sets how
predictable the market is, from `phi = 0` (identical to a random walk) up to
values no real market has ever shown.

Turn the dial down until the AI loses the scent, and the value where it does is
the AI's sensitivity. That number is the whole point, because real equity
autocorrelations are tiny -- on daily data, roughly 0.01 to 0.05 and unstable. If
the AI needs phi = 0.15 to find anything, it cannot work on real markets, and we
learn that from a measurement rather than a hunch.

Holding volatility constant
---------------------------
An AR(1) process has unconditional variance sigma_eps^2 / (1 - phi^2), so raising
phi would raise volatility too unless it is compensated for. That would ruin the
experiment: any change in results could be attributed either to the signal or to
the volatility, with no way to separate them.

So the innovation scale is set to sigma_daily * sqrt(1 - phi^2), which holds total
volatility fixed at `sigma` for every value of phi. The dial changes *only*
predictability -- which is what makes the Recovery Test a clean measurement.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import lfilter

from sentinel.sandbox.generators.base import TRADING_DAYS_PER_YEAR, Generator, Simulation


class AR1Generator(Generator):
    """Markets whose returns are autocorrelated at a strength you choose.

    Args:
        mu: annualised drift.
        sigma: annualised volatility, held constant regardless of phi.
        phi: lag-1 autocorrelation of returns. Positive is momentum (moves
            continue), negative is short-term reversal (moves bounce back).
            Must be strictly between -1 and 1 or the process explodes.
    """

    model_name = "ar1"

    def __init__(
        self,
        mu: float = 0.08,
        sigma: float = 0.16,
        phi: float = 0.05,
        correlation: np.ndarray | None = None,
        initial_price: float = 100.0,
        trading_days: int = TRADING_DAYS_PER_YEAR,
    ) -> None:
        super().__init__(correlation, initial_price, trading_days)
        if not -1.0 < float(phi) < 1.0:
            raise ValueError(f"phi must be strictly between -1 and 1, got {phi}")
        if float(sigma) < 0:
            raise ValueError("volatility cannot be negative")
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.phi = float(phi)

    @property
    def has_exploitable_signal(self) -> bool:
        """At phi = 0 this generator *is* a random walk, and says so."""
        return self.phi != 0.0

    def _simulate(self, n_steps: int, n_assets: int, rng: np.random.Generator) -> Simulation:
        n = n_steps - 1
        phi = self.phi

        mean_log_return = (self.mu - 0.5 * self.sigma**2) * self.dt
        sigma_daily = self.sigma * np.sqrt(self.dt)
        # The variance compensation described in the module docstring.
        innovation_scale = sigma_daily * np.sqrt(1.0 - phi**2)

        innovations = innovation_scale * self._correlated_shocks(n, n_assets, rng)

        # Start from the stationary distribution rather than from zero, so there
        # is no burn-in period during which volatility is wrong.
        initial_deviation = sigma_daily * rng.standard_normal((1, n_assets))

        # AR(1) is a one-pole IIR filter: y_t = phi * y_t-1 + eps_t. lfilter's
        # initial state for this form is phi * y_-1.
        deviations, _ = lfilter(
            [1.0], [1.0, -phi], innovations, axis=0, zi=phi * initial_deviation
        )

        return Simulation(
            log_returns=mean_log_return + deviations,
            extra={
                "mu": self.mu,
                "sigma": self.sigma,
                "phi": phi,
                "implied_lag1_autocorrelation": phi,
            },
        )

    def _params(self, n_assets: int) -> dict[str, Any]:
        return {}
