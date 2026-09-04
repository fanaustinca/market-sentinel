"""Markets built to break things.

Every other generator asks "can the strategy find what is there?". These ask the
opposite question: **what happens when something arrives that the strategy has
never seen?** That is Phase 3, and it exists because the failures that actually
destroy accounts are not gradual underperformance. They are single events the
system had no representation of.

Two are modelled here because they are the two the plan names, and because both
are absent from every generator built so far.

**Scripted crashes.** A fall of a chosen depth over a chosen number of days, with
no warning in the preceding returns. The point is not that the strategy avoids it
-- it cannot, and a strategy that appeared to would be predicting the
unpredictable. The point is that the risk layer fires, the loss stays inside the
limit it promised, and nothing fails silently.

**Correlation breakdown.** The nastiest behaviour real markets have: correlations
between assets rush toward one during a crash, so diversification evaporates
exactly when it is needed. A simulator with a fixed correlation matrix -- which
is every generator in this project until now -- makes every portfolio look far
safer than it is, because it quietly promises that the assets will keep behaving
differently at the worst possible moment.

Not for measuring noise floors
------------------------------
These markets contain a scripted event, so they are neither signal-free nor
stationary. They are stress tests, and their output is a pass or fail against the
risk limits, not a Sharpe ratio to be compared against a floor.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sentinel.sandbox.generators.base import Generator, Simulation

TRADING_DAYS_PER_YEAR = 252


class CrashGenerator(Generator):
    """A quiet market interrupted by a fall of known depth and duration.

    Args:
        crash_depth: the *deterministic* part of the fall, as a fraction. The
            realised drawdown is deeper -- typically by a third at the default
            settings -- because `crash_volatility` noise and the surrounding
            market are layered on top. That is deliberate: a crash of exactly the
            depth you specified would be a crash you could have planned for.
        crash_days: trading days the fall is spread over. A 35% fall in five days
            and the same fall over six months are completely different problems
            for a system that reacts to trailing data -- the first is over before
            any trailing window notices, and no risk layer can help.
        crash_volatility: annualised volatility *during* the crash. Real crashes
            are violent as well as negative, and a smoothly sloping decline would
            be far easier to survive than anything that has ever happened.
        recovery: fraction of the fall regained afterwards, over `crash_days * 3`.
            Included because selling the bottom is the expensive half of a badly
            timed exit, and a scenario that ends at the low would never reveal it.
        timing: where the crash starts, as a fraction of the series. `None`
            randomises it per seed, which is the honest default -- a fixed
            position is a date a strategy could in principle learn.
    """

    model_name = "crash"
    has_exploitable_signal = False
    has_predictable_volatility = False

    def __init__(
        self,
        mu: float = 0.08,
        sigma: float = 0.14,
        crash_depth: float = 0.35,
        crash_days: int = 20,
        crash_volatility: float = 0.60,
        recovery: float = 0.5,
        timing: float | None = None,
        correlation: np.ndarray | None = None,
        initial_price: float = 100.0,
        trading_days: int = TRADING_DAYS_PER_YEAR,
    ) -> None:
        super().__init__(correlation, initial_price, trading_days)
        if not 0.0 < crash_depth < 1.0:
            raise ValueError("crash_depth must be between 0 and 1")
        if crash_days < 1:
            raise ValueError("crash_days must be at least 1")
        if not 0.0 <= recovery <= 1.0:
            raise ValueError("recovery must be between 0 and 1")
        if timing is not None and not 0.0 < timing < 1.0:
            raise ValueError("timing must be strictly between 0 and 1")

        self.mu = float(mu)
        self.sigma = float(sigma)
        self.crash_depth = float(crash_depth)
        self.crash_days = int(crash_days)
        self.crash_volatility = float(crash_volatility)
        self.recovery = float(recovery)
        self.timing = timing

    def _simulate(self, n_steps: int, n_assets: int, rng: np.random.Generator) -> Simulation:
        n = n_steps - 1
        dt = self.dt

        drift = (self.mu - 0.5 * self.sigma**2) * dt
        returns = drift + self.sigma * np.sqrt(dt) * self._correlated_shocks(n, n_assets, rng)

        recovery_days = self.crash_days * 3
        latest_start = n - self.crash_days - recovery_days
        if latest_start <= self.crash_days:
            raise ValueError(
                f"a {self.crash_days}-day crash with recovery needs at least "
                f"{2 * self.crash_days + recovery_days + 2} steps, got {n_steps}"
            )

        if self.timing is None:
            # Keep it clear of the warmup so the strategy has actually formed a
            # view before being hit.
            start = int(rng.integers(max(self.crash_days, n // 5), latest_start))
        else:
            start = min(int(self.timing * n), latest_start)

        # The crash is added on top of the ordinary process rather than replacing
        # it, so it retains the market's own noise and does not arrive as an
        # implausibly smooth slide.
        total_log_fall = np.log(1.0 - self.crash_depth)
        per_day = total_log_fall / self.crash_days
        crash_noise = self.crash_volatility * np.sqrt(dt) * rng.standard_normal(
            (self.crash_days, n_assets)
        )
        returns[start : start + self.crash_days] += per_day + crash_noise

        if self.recovery > 0:
            regained = -total_log_fall * self.recovery / recovery_days
            returns[start + self.crash_days : start + self.crash_days + recovery_days] += regained

        return Simulation(
            log_returns=returns,
            extra={
                "crash_start_index": int(start),
                "crash_depth": self.crash_depth,
                "crash_days": self.crash_days,
                "crash_volatility": self.crash_volatility,
                "recovery_fraction": self.recovery,
            },
        )

    def _params(self, n_assets: int) -> dict[str, Any]:
        return {"mu": self.mu, "sigma": self.sigma}


class CorrelationBreakdownGenerator(Generator):
    """Assets that diversify in calm markets and move as one in a crisis.

    The failure this models is not that a portfolio loses money in a crash. It is
    that it loses **far more than its own backtest said it could**, because the
    backtest measured diversification during calm periods and assumed it would
    still be there.

    Every other generator in this sandbox uses a single fixed correlation matrix
    and therefore cannot express this at all. A multi-asset strategy validated
    only against those is validated against a market that has promised, in
    advance, never to do the one thing that would hurt it most.

    Args:
        calm_correlation: pairwise correlation in normal conditions.
        stress_correlation: pairwise correlation during stress. 0.95 rather than
            1.0 because a perfectly correlated matrix is singular and Cholesky
            refuses it -- and because real crash correlations, while brutal, stop
            short of exact.
        stress_volatility_multiple: how much volatility rises in stress. Both
            effects arrive together, which is what makes the combination so much
            worse than either alone.
        stress_probability: chance of entering stress on any calm day.
        stress_persistence: chance of staying in stress once there. Together
            these fix how much of the sample is stressed and how long an episode
            lasts -- see `stationary_stress_share` and `expected_stress_days`.
            The defaults give roughly 14% of days in episodes averaging 33 days,
            which is about what post-war equity markets have delivered. Setting
            both to 0.05/0.95, an easy mistake, puts the market in crisis half
            the time, which is not a stress test of anything -- it is a different
            market.
    """

    model_name = "correlation_breakdown"
    has_exploitable_signal = False
    has_predictable_volatility = True

    def __init__(
        self,
        mu: float = 0.08,
        sigma: float = 0.14,
        calm_correlation: float = 0.2,
        stress_correlation: float = 0.95,
        stress_volatility_multiple: float = 2.5,
        stress_probability: float = 0.005,
        stress_persistence: float = 0.97,
        initial_price: float = 100.0,
        trading_days: int = TRADING_DAYS_PER_YEAR,
    ) -> None:
        super().__init__(None, initial_price, trading_days)
        for name, value in (
            ("calm_correlation", calm_correlation),
            ("stress_correlation", stress_correlation),
        ):
            if not -1.0 < value < 1.0:
                raise ValueError(f"{name} must be strictly between -1 and 1")
        if stress_correlation <= calm_correlation:
            raise ValueError(
                "stress_correlation must exceed calm_correlation; "
                "this generator exists to model diversification getting worse"
            )
        if not 0.0 < stress_probability < 1.0:
            raise ValueError("stress_probability must be strictly between 0 and 1")
        if not 0.0 < stress_persistence < 1.0:
            raise ValueError("stress_persistence must be strictly between 0 and 1")

        self.mu = float(mu)
        self.sigma = float(sigma)
        self.calm_correlation = float(calm_correlation)
        self.stress_correlation = float(stress_correlation)
        self.stress_volatility_multiple = float(stress_volatility_multiple)
        self.stress_probability = float(stress_probability)
        self.stress_persistence = float(stress_persistence)

    @property
    def stationary_stress_share(self) -> float:
        """Long-run fraction of days spent in the correlated-stress state."""
        enter, exit_ = self.stress_probability, 1.0 - self.stress_persistence
        return enter / (enter + exit_)

    @property
    def expected_stress_days(self) -> float:
        """Average length of one stress episode, in trading days."""
        return 1.0 / (1.0 - self.stress_persistence)

    def _matrix(self, rho: float, n_assets: int) -> np.ndarray:
        matrix = np.full((n_assets, n_assets), rho)
        np.fill_diagonal(matrix, 1.0)
        return matrix

    def _simulate(self, n_steps: int, n_assets: int, rng: np.random.Generator) -> Simulation:
        if n_assets < 2:
            raise ValueError(
                "correlation breakdown needs at least two assets; with one there "
                "is no diversification to lose"
            )

        n = n_steps - 1
        dt = self.dt

        # A two-state chain over the correlation environment, so stress arrives in
        # stretches rather than as isolated days. Isolated days would be absorbed
        # by any trailing window and the scenario would be far too easy.
        states = np.empty(n, dtype=np.int8)
        state = 0
        for t in range(n):
            states[t] = state
            if state == 0:
                state = 1 if rng.random() < self.stress_probability else 0
            else:
                state = 1 if rng.random() < self.stress_persistence else 0

        calm_factor = np.linalg.cholesky(self._matrix(self.calm_correlation, n_assets))
        stress_factor = np.linalg.cholesky(self._matrix(self.stress_correlation, n_assets))

        raw = rng.standard_normal((n, n_assets))
        shocks = np.where(
            states[:, None] == 1, raw @ stress_factor.T, raw @ calm_factor.T
        )

        volatility = np.where(
            states == 1, self.sigma * self.stress_volatility_multiple, self.sigma
        )[:, None]
        drift = (self.mu - 0.5 * volatility**2) * dt

        return Simulation(
            log_returns=drift + volatility * np.sqrt(dt) * shocks,
            regimes=states,
            extra={
                "calm_correlation": self.calm_correlation,
                "stress_correlation": self.stress_correlation,
                "stress_volatility_multiple": self.stress_volatility_multiple,
                "realised_stress_fraction": float(states.mean()),
                "stationary_stress_share": self.stationary_stress_share,
                "expected_stress_days": self.expected_stress_days,
                "state_names": ["calm", "correlated_stress"],
            },
        )

    def _params(self, n_assets: int) -> dict[str, Any]:
        return {"mu": self.mu, "sigma": self.sigma, "n_assets": n_assets}
