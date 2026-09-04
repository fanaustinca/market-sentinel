"""Markov regime switching -- the market the AI is actually meant to detect.

The market flips between a calm state (upward drift, low volatility) and a
stressed one (negative drift, high volatility), at random times, with the state
persisting once entered. This is the closest thing in the sandbox to what real
markets appear to do, and detecting the switch is the AI's actual job -- far more
tractable than predicting tomorrow's return, and the source of nearly all the
defensive value in the strategy.

Why this generator matters most
-------------------------------
It is the only one that ships an **answer key for the model itself**. Every step
carries its true regime label, so a classifier can be scored directly: on day 900,
was the market really in the bear state, and did the model say so?

On real data that comparison is impossible. Nobody knows what regime the market
was "really" in on a given day -- the label does not exist, and any label a human
assigns is drawn with hindsight. Here the label is ground truth, so we can measure
not just profit but *accuracy*, and separate a model that genuinely detects
regimes from one that merely got lucky in a backtest.

The hard part, and the honest one: the switch is only visible after the fact.
Volatility rises, drift turns negative, but any single day looks like noise. The
model must accumulate evidence, which takes time, which costs money. Measuring
that delay -- how many days after a switch before the model notices -- is one of
the most useful things the sandbox can tell us.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sentinel.sandbox.generators.base import TRADING_DAYS_PER_YEAR, Generator, Simulation


class RegimeSwitchingGenerator(Generator):
    """A market that alternates between calm and stressed states.

    Args:
        mu: annualised drift in each state. Default is +12% in the calm state,
            -15% in the stressed one.
        sigma: annualised volatility in each state. Default 12% calm, 32%
            stressed -- roughly the ratio real equity markets show.
        persistence: probability of *staying* in each state on any given day.
            0.99 means an average calm run of 100 trading days; 0.97 an average
            stressed run of about 33. High persistence is what makes regimes
            detectable at all: a state that flipped daily would be pure noise.
    """

    model_name = "regime"
    has_exploitable_signal = True
    has_predictable_volatility = True

    def __init__(
        self,
        mu: tuple[float, float] = (0.12, -0.15),
        sigma: tuple[float, float] = (0.12, 0.32),
        persistence: tuple[float, float] = (0.99, 0.97),
        correlation: np.ndarray | None = None,
        initial_price: float = 100.0,
        trading_days: int = TRADING_DAYS_PER_YEAR,
    ) -> None:
        super().__init__(correlation, initial_price, trading_days)
        if len(mu) != 2 or len(sigma) != 2 or len(persistence) != 2:
            raise ValueError("mu, sigma and persistence must each have two entries")
        if any(not 0.0 < p < 1.0 for p in persistence):
            raise ValueError("persistence values must be strictly between 0 and 1")
        if any(s < 0 for s in sigma):
            raise ValueError("volatility cannot be negative")
        self.mu = tuple(float(m) for m in mu)
        self.sigma = tuple(float(s) for s in sigma)
        self.persistence = tuple(float(p) for p in persistence)

    @property
    def transition_matrix(self) -> np.ndarray:
        p_calm, p_stress = self.persistence
        return np.array([[p_calm, 1 - p_calm], [1 - p_stress, p_stress]])

    @property
    def stationary_distribution(self) -> np.ndarray:
        """Long-run share of time spent in each state."""
        p_calm, p_stress = self.persistence
        calm_share = (1 - p_stress) / ((1 - p_calm) + (1 - p_stress))
        return np.array([calm_share, 1 - calm_share])

    @property
    def expected_durations(self) -> tuple[float, float]:
        """Average consecutive days spent in each state before switching."""
        return tuple(1.0 / (1.0 - p) for p in self.persistence)

    def _simulate(self, n_steps: int, n_assets: int, rng: np.random.Generator) -> Simulation:
        n = n_steps - 1

        # The regime is a property of the market, so every asset shares it. That
        # is both realistic and what makes the classification problem well posed.
        states = np.empty(n, dtype=np.int8)
        stay = np.array(self.persistence)
        draws = rng.random(n)

        # Start from the stationary distribution rather than always in the calm
        # state, so the series does not begin with a systematic bias.
        state = 0 if rng.random() < self.stationary_distribution[0] else 1
        for t in range(n):
            states[t] = state
            if draws[t] >= stay[state]:
                state = 1 - state

        mu = np.array(self.mu)[states][:, None]
        sigma = np.array(self.sigma)[states][:, None]

        shocks = self._correlated_shocks(n, n_assets, rng)
        log_returns = (mu - 0.5 * sigma**2) * self.dt + sigma * np.sqrt(self.dt) * shocks

        return Simulation(
            log_returns=log_returns,
            regimes=states,
            extra={
                "mu": list(self.mu),
                "sigma": list(self.sigma),
                "persistence": list(self.persistence),
                "expected_durations_days": list(self.expected_durations),
                "stationary_distribution": self.stationary_distribution.tolist(),
                "state_names": ["calm", "stressed"],
                "realised_stressed_fraction": float(states.mean()),
            },
        )

    @classmethod
    def equity_like(cls, **kwargs) -> "RegimeSwitchingGenerator":
        """A regime market whose states are calibrated to what SPY actually does.

        The default parameters, `mu=(0.12, -0.15)` with `sigma=(0.12, 0.32)`,
        weld two properties together: calm means rising *and* quiet, stressed
        means falling *and* volatile. That coupling is not a fact about markets;
        it was a modelling choice, and it silently taught every strategy built in
        this sandbox that fleeing volatility means fleeing losses.

        Measured on SPY from 1993, conditioning on the classifier's own states
        (see `experiments/simulator_gap.py`):

            calm       +9.0% a year at 13.4% volatility   Sharpe 0.67
            stressed  +17.1% a year at 27.8% volatility   Sharpe 0.62

        Real equity volatility is *compensated*. The high-volatility state has
        the higher return, and the risk-adjusted return is close to identical in
        both -- the market is not offering a better deal in one state, it is
        offering the same deal at two different sizes.

        This preset reproduces that. It is a harder and more honest sandbox: a
        strategy that gets its edge from sitting out volatility will earn nothing
        here, which is what it earns in reality. Use it to check whether a result
        obtained with the default parameters survives removing the assumption
        that produced it.

        The class was always able to express this. Nothing needed changing except
        the numbers, and every call site used the defaults.
        """
        kwargs.setdefault("mu", (0.09, 0.17))
        kwargs.setdefault("sigma", (0.134, 0.278))
        kwargs.setdefault("persistence", (0.985, 0.968))
        return cls(**kwargs)

    def _params(self, n_assets: int) -> dict[str, Any]:
        return {}
