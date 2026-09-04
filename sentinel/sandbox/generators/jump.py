"""Merton jump diffusion -- a null market with fat tails.

A random walk with sudden discontinuous drops layered on top. Jumps arrive as a
Poisson process, so their timing is completely unpredictable, and their sizes are
drawn independently of everything else.

This is the sandbox's **second null market**, and it is arguably a harder test
than plain GBM.

The reason is that a random walk produces normally distributed returns, which real
markets emphatically do not. Real returns have fat tails: crashes happen far more
often than a normal distribution permits, and a model validated only on GBM has
never seen a day that should have been impossible. Here it will -- while the
market still contains no predictable signal whatsoever.

So an AI that made money on this market would be doing one of two things, both
worth catching early. It might have a lookahead bug, exactly as with GBM. Or it
might be pattern-matching on volatility clustering that the crashes create in
hindsight, "predicting" crashes it can only see once they have happened. The
second failure is subtler and more seductive, and plain GBM cannot expose it.

The drift compensator
---------------------
Jumps with a negative average size drag the price down, so leaving `mu` alone
would produce a market with lower drift than requested. Subtracting `lambda * k`
from the drift -- where `k` is the expected proportional jump size -- keeps the
total expected return at `mu` regardless of jump settings. Without it, changing
the jump parameters would silently change the drift too, and any comparison
between jump and non-jump markets would be confounded.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sentinel.sandbox.generators.base import TRADING_DAYS_PER_YEAR, Generator, Simulation


class JumpDiffusionGenerator(Generator):
    """Random walk plus unpredictable crashes.

    Args:
        mu: annualised drift, held fixed by the compensator regardless of jumps.
        sigma: annualised volatility of the continuous part.
        jump_intensity: expected number of jumps per year. 2.0 means a couple of
            sharp dislocations a year.
        jump_mean: average jump size in log terms. Negative for crashes; -0.05 is
            about a 5% gap down.
        jump_sd: dispersion of jump sizes.
    """

    model_name = "jump"
    # Jump timing is Poisson, so nothing about the past predicts the next jump.
    has_exploitable_signal = False

    def __init__(
        self,
        mu: float = 0.08,
        sigma: float = 0.14,
        jump_intensity: float = 2.0,
        jump_mean: float = -0.05,
        jump_sd: float = 0.04,
        correlation: np.ndarray | None = None,
        initial_price: float = 100.0,
        trading_days: int = TRADING_DAYS_PER_YEAR,
    ) -> None:
        super().__init__(correlation, initial_price, trading_days)
        if float(sigma) < 0 or float(jump_sd) < 0:
            raise ValueError("volatility cannot be negative")
        if float(jump_intensity) < 0:
            raise ValueError("jump_intensity cannot be negative")
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.jump_intensity = float(jump_intensity)
        self.jump_mean = float(jump_mean)
        self.jump_sd = float(jump_sd)

    @property
    def expected_jump_size(self) -> float:
        """k = E[e^Y - 1], the expected proportional price change from one jump."""
        return float(np.exp(self.jump_mean + 0.5 * self.jump_sd**2) - 1.0)

    def _simulate(self, n_steps: int, n_assets: int, rng: np.random.Generator) -> Simulation:
        n = n_steps - 1
        dt = self.dt

        drift = (self.mu - 0.5 * self.sigma**2 - self.jump_intensity * self.expected_jump_size) * dt
        diffusion = self.sigma * np.sqrt(dt) * self._correlated_shocks(n, n_assets, rng)

        # Number of jumps in each step. The sum of N independent normal jump sizes
        # is itself normal with mean N * jump_mean and variance N * jump_sd^2,
        # which lets the whole jump component be drawn in one vectorised step
        # instead of looping over individual jumps.
        counts = rng.poisson(self.jump_intensity * dt, size=(n, n_assets))
        jumps = counts * self.jump_mean + np.sqrt(counts) * self.jump_sd * rng.standard_normal((n, n_assets))

        return Simulation(
            log_returns=drift + diffusion + jumps,
            extra={
                "mu": self.mu,
                "sigma": self.sigma,
                "jump_intensity": self.jump_intensity,
                "jump_mean": self.jump_mean,
                "jump_sd": self.jump_sd,
                "expected_jump_size": self.expected_jump_size,
                "realised_jump_count": int(counts.sum()),
            },
        )

    def _params(self, n_assets: int) -> dict[str, Any]:
        return {}
