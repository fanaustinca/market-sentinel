"""Heston stochastic volatility -- unpredictable direction, predictable risk.

Volatility is itself a mean-reverting random process rather than a constant. This
produces **volatility clustering**: calm stretches followed by stormy ones, which
is one of the most robust empirical facts about real markets and something none of
the constant-volatility generators can reproduce.

The distinction this generator is here to enforce
-------------------------------------------------
This market has `has_exploitable_signal = False` but `has_predictable_volatility
= True`, and keeping those two apart is one of the more valuable ideas in the
project.

The *direction* of returns is unforecastable -- knowing tomorrow's volatility
tells you nothing about tomorrow's sign. But the *risk level* is highly
forecastable, because volatility is persistent: today being turbulent makes
tomorrow likely to be turbulent too.

Those permit completely different things. No amount of volatility forecasting
creates a directional edge, so a model claiming to predict returns here is wrong.
But a strategy can still use the forecast to hold less when risk is high, which
improves return *per unit of risk* without predicting direction at all. Conflating
the two is a common way to overstate what a model has found, and this generator
makes the confusion detectable.

The leverage effect
-------------------
`rho` is negative by default, meaning price shocks and volatility shocks move
oppositely: markets fall and volatility spikes together. This is why crashes are
violent and rallies are calm, and it is strongly present in real equity data.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sentinel.sandbox.generators.base import TRADING_DAYS_PER_YEAR, Generator, Simulation


class HestonGenerator(Generator):
    """Markets with clustering, mean-reverting volatility.

    Args:
        mu: annualised drift.
        long_run_variance: the level variance reverts to. 0.0256 is a 16%
            annualised volatility, since 0.16^2 = 0.0256.
        kappa: speed of variance mean reversion, per year.
        vol_of_vol: volatility of the variance process itself.
        rho: correlation between price and variance shocks. Negative gives the
            leverage effect seen in real equities.
        initial_variance: starting variance, defaulting to the long-run level.
    """

    model_name = "heston"
    has_exploitable_signal = False
    has_predictable_volatility = True

    def __init__(
        self,
        mu: float = 0.08,
        long_run_variance: float = 0.0256,
        kappa: float = 3.0,
        vol_of_vol: float = 0.3,
        rho: float = -0.7,
        initial_variance: float | None = None,
        correlation: np.ndarray | None = None,
        initial_price: float = 100.0,
        trading_days: int = TRADING_DAYS_PER_YEAR,
    ) -> None:
        super().__init__(correlation, initial_price, trading_days)
        if float(long_run_variance) <= 0:
            raise ValueError("long_run_variance must be positive")
        if float(kappa) <= 0:
            raise ValueError("kappa must be positive")
        if float(vol_of_vol) < 0:
            raise ValueError("vol_of_vol cannot be negative")
        if not -1.0 <= float(rho) <= 1.0:
            raise ValueError(f"rho must be between -1 and 1, got {rho}")

        self.mu = float(mu)
        self.long_run_variance = float(long_run_variance)
        self.kappa = float(kappa)
        self.vol_of_vol = float(vol_of_vol)
        self.rho = float(rho)
        self.initial_variance = (
            float(long_run_variance) if initial_variance is None else float(initial_variance)
        )
        if self.initial_variance < 0:
            raise ValueError("initial_variance cannot be negative")

    @property
    def satisfies_feller(self) -> bool:
        """Feller condition: 2 * kappa * theta > xi^2 keeps variance away from zero.

        When it fails the variance process hits zero regularly, which is not
        wrong exactly -- the truncation scheme handles it -- but it produces
        markets with implausible dead-calm stretches. Worth knowing about.
        """
        return 2 * self.kappa * self.long_run_variance > self.vol_of_vol**2

    def _simulate(self, n_steps: int, n_assets: int, rng: np.random.Generator) -> Simulation:
        n = n_steps - 1
        dt = self.dt

        price_shocks = self._correlated_shocks(n, n_assets, rng)
        independent = rng.standard_normal((n, n_assets))
        # Impose the leverage correlation between price and variance shocks.
        variance_shocks = self.rho * price_shocks + np.sqrt(1 - self.rho**2) * independent

        log_returns = np.empty((n, n_assets))
        variance_path = np.empty((n, n_assets))
        variance = np.full(n_assets, self.initial_variance)

        # Full truncation Euler. The variance process can step negative under
        # discretisation even though the true process cannot, so every use of it
        # is floored at zero while the state itself is allowed to go negative.
        # Simply clipping the state instead introduces a systematic upward bias.
        for t in range(n):
            positive = np.maximum(variance, 0.0)
            root = np.sqrt(positive * dt)
            variance_path[t] = positive
            log_returns[t] = (self.mu - 0.5 * positive) * dt + root * price_shocks[t]
            variance = variance + self.kappa * (self.long_run_variance - positive) * dt + self.vol_of_vol * root * variance_shocks[t]

        return Simulation(
            log_returns=log_returns,
            # variance_path holds annualised variance -- dt is applied separately
            # in the return equation -- so the annualised volatility is its
            # square root with no further scaling. For a single asset it is
            # flattened, matching how `regimes` is shaped.
            volatility=np.sqrt(variance_path[:, 0]) if n_assets == 1 else np.sqrt(variance_path),
            extra={
                "mu": self.mu,
                "long_run_variance": self.long_run_variance,
                "long_run_volatility": float(np.sqrt(self.long_run_variance)),
                "kappa": self.kappa,
                "vol_of_vol": self.vol_of_vol,
                "rho": self.rho,
                "satisfies_feller": self.satisfies_feller,
                "realised_mean_volatility": float(np.sqrt(variance_path.mean())),
            },
        )

    def _params(self, n_assets: int) -> dict[str, Any]:
        return {}
