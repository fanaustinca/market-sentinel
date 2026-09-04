"""Tests for the adversarial generators.

These markets exist to break the system, so the tests are mostly about whether
they are actually hard. A stress scenario that turns out to be mild is worse than
no stress scenario, because it produces a passing result that means nothing.

The correlation-breakdown generator matters most. Every other generator in this
project uses a single fixed correlation matrix, which quietly promises that
assets will keep behaving differently during a crash. They do not, and a
multi-asset strategy validated only against that promise has been validated
against a market that agreed in advance not to hurt it.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.engine.backtest import UNLIMITED, RiskLimits, run_backtest
from sentinel.sandbox.generators.adversarial import (
    CorrelationBreakdownGenerator,
    CrashGenerator,
)
from sentinel.strategies.baseline import BuyAndHold, FixedWeights


def mean_offdiagonal(returns: np.ndarray) -> float:
    matrix = np.corrcoef(returns.T)
    return float(matrix[np.triu_indices_from(matrix, k=1)].mean())


class TestCrashes:
    def test_the_crash_is_at_least_as_deep_as_requested(self) -> None:
        """Noise is layered on top, so realised depth exceeds the deterministic part.

        Deliberate: a crash of exactly the depth you specified is a crash you
        could have planned for, and no real one has ever been that obliging.
        """
        depths = []
        for seed in range(8):
            scenario = CrashGenerator(crash_depth=0.35, crash_days=20).generate(
                n_steps=1500, n_assets=1, seed=seed
            )
            prices = scenario.data.prices.iloc[:, 0]
            depths.append((prices / prices.cummax() - 1).min())
        assert np.mean(depths) < -0.35

    def test_the_crash_lands_where_the_answer_key_says(self) -> None:
        scenario = CrashGenerator(crash_depth=0.40, crash_days=15, timing=0.5).generate(
            n_steps=2000, n_assets=1, seed=3
        )
        start = scenario.truth.params["crash_start_index"]
        returns = scenario.data.log_returns().to_numpy().ravel()
        during = returns[start : start + 15].sum()
        assert during < np.log(1 - 0.40) * 0.5, "the fall must actually happen there"

    def test_timing_is_randomised_by_default(self) -> None:
        """A fixed crash date is a date a strategy could in principle learn."""
        starts = {
            CrashGenerator().generate(n_steps=2000, seed=seed).truth.params["crash_start_index"]
            for seed in range(8)
        }
        assert len(starts) > 5

    def test_nothing_warns_of_the_crash_beforehand(self) -> None:
        """The scenario is only meaningful if it is genuinely unforecastable.

        If pre-crash returns drifted or grew volatile, the test would be measuring
        whether a strategy notices a telegraphed crash, which is a different and
        much easier question than the one being asked.
        """
        from sentinel.stats.randomwalk import ljung_box

        scenario = CrashGenerator(timing=0.7).generate(n_steps=3000, n_assets=1, seed=6)
        start = scenario.truth.params["crash_start_index"]
        before = scenario.data.log_returns().to_numpy().ravel()[:start]
        assert ljung_box(before, lags=10).p_value > 0.01

    def test_a_series_too_short_for_the_crash_is_refused(self) -> None:
        with pytest.raises(ValueError, match="needs at least"):
            CrashGenerator(crash_days=100).generate(n_steps=200, seed=1)

    @pytest.mark.parametrize(
        "kwargs,message",
        [
            ({"crash_depth": 1.0}, "crash_depth must be between"),
            ({"crash_days": 0}, "crash_days must be at least"),
            ({"recovery": 1.5}, "recovery must be between"),
            ({"timing": 1.0}, "timing must be strictly between"),
        ],
    )
    def test_rejects_impossible_parameters(self, kwargs, message) -> None:
        with pytest.raises(ValueError, match=message):
            CrashGenerator(**kwargs)

    def test_the_breaker_helps_except_when_the_crash_is_instantaneous(self) -> None:
        """The one case where the risk control is net harmful, pinned.

        Measured across crash speeds on a 45% fall, the drawdown the breaker
        saves against having none:

            1 day    -3.0%   <- it makes things WORSE
            2 days  +13.3%
            10 days +21.5%
            90 days +18.4%

        A drawdown breaker reacts to losses already realised. When the whole fall
        lands in a single day it cannot act until the damage is done, and what it
        then does is sell at the bottom and sit in cash through the rebound. So it
        converts an unavoidable loss into a permanent one and gives up 3.4% a year
        for the privilege.

        This is gap risk -- 1987, a currency peg breaking, an overnight halt -- and
        it is the scenario a trailing risk rule is structurally unable to help
        with. It is pinned here because it would otherwise be discovered the
        expensive way, and because the honest response is to size positions so a
        gap is survivable rather than to tune the breaker.
        """
        def saved(days: int) -> float:
            gaps = []
            for seed in range(8):
                data = CrashGenerator(crash_depth=0.45, crash_days=days).generate(
                    n_steps=2520, n_assets=1, seed=seed
                ).data
                guarded = run_backtest(data, BuyAndHold(), limits=RiskLimits())
                unguarded = run_backtest(data, BuyAndHold(), limits=UNLIMITED)
                gaps.append(
                    guarded.performance.max_drawdown - unguarded.performance.max_drawdown
                )
            return float(np.mean(gaps))

        assert saved(1) < 0.0, "an overnight gap is not something a trailing rule can help with"
        assert saved(10) > 0.10
        assert saved(90) > 0.10

class TestCorrelationBreakdown:
    def test_correlation_really_does_break(self) -> None:
        scenario = CorrelationBreakdownGenerator().generate(n_steps=6000, n_assets=4, seed=2)
        returns = scenario.data.log_returns().to_numpy()
        states = scenario.truth.regimes

        assert mean_offdiagonal(returns[states == 0]) == pytest.approx(0.2, abs=0.06)
        assert mean_offdiagonal(returns[states == 1]) == pytest.approx(0.95, abs=0.05)

    def test_volatility_rises_at_the_same_time(self) -> None:
        """Both effects arrive together, which is what makes it so much worse
        than either alone."""
        scenario = CorrelationBreakdownGenerator().generate(n_steps=6000, n_assets=4, seed=2)
        returns = scenario.data.log_returns().to_numpy()
        states = scenario.truth.regimes
        assert returns[states == 1].std() > 2.0 * returns[states == 0].std()

    def test_diversification_fails_when_it_is_needed(self) -> None:
        """The finding the generator exists to produce.

        The same portfolio, the same weights, the same assets: a fixed-correlation
        market understates the drawdown. Every multi-asset result measured on the
        other generators carries this as an unstated caveat.
        """
        from sentinel.sandbox.generators.gbm import GBMGenerator

        correlation = np.full((4, 4), 0.2)
        np.fill_diagonal(correlation, 1.0)
        weights = {f"SYN{i}": 0.25 for i in range(4)}

        fixed, breaking = [], []
        for seed in range(10):
            calm = GBMGenerator(mu=0.08, sigma=0.14, correlation=correlation).generate(
                n_steps=2520, n_assets=4, seed=seed
            )
            broken = CorrelationBreakdownGenerator().generate(
                n_steps=2520, n_assets=4, seed=seed
            )
            fixed.append(
                run_backtest(calm.data, FixedWeights(weights), limits=UNLIMITED)
                .performance.max_drawdown
            )
            breaking.append(
                run_backtest(broken.data, FixedWeights(weights), limits=UNLIMITED)
                .performance.max_drawdown
            )
        assert np.mean(breaking) < np.mean(fixed) - 0.05

    def test_stress_arrives_in_stretches_not_isolated_days(self) -> None:
        """Isolated bad days are absorbed by any trailing window.

        A scenario made of them would be far too easy and would certify a system
        that has never been tested.
        """
        generator = CorrelationBreakdownGenerator()
        assert generator.expected_stress_days > 20
        scenario = generator.generate(n_steps=6000, n_assets=3, seed=4)
        states = scenario.truth.regimes
        runs = np.diff(np.flatnonzero(np.diff(np.concatenate([[0], states, [0]]))))[::2]
        assert np.mean(runs) > 10

    def test_the_sample_is_mostly_calm(self) -> None:
        """A market in crisis half the time is not a stress test of anything.

        The defaults are chosen so stress is roughly 14% of days in episodes of
        about a month, which is broadly what post-war equity markets delivered.
        """
        generator = CorrelationBreakdownGenerator()
        assert 0.08 < generator.stationary_stress_share < 0.22

    def test_one_asset_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least two assets"):
            CorrelationBreakdownGenerator().generate(n_steps=500, n_assets=1, seed=1)

    def test_stress_correlation_must_exceed_calm(self) -> None:
        """The generator exists to model diversification getting worse."""
        with pytest.raises(ValueError, match="must exceed calm_correlation"):
            CorrelationBreakdownGenerator(calm_correlation=0.8, stress_correlation=0.3)

    def test_the_answer_key_records_the_stress_state(self) -> None:
        scenario = CorrelationBreakdownGenerator().generate(n_steps=1000, n_assets=3, seed=5)
        assert scenario.truth.regimes is not None
        assert scenario.truth.regimes.shape == (999,)
        assert scenario.truth.has_predictable_volatility is True
