"""Tests for volatility targeting.

This strategy exists because a measurement contradicted the assumption the regime
strategies were built on. In the sandbox, high volatility means falling prices --
because `RegimeSwitchingGenerator` was told to make it so. On real SPY it does
not: the classifier's stressed state has a *higher* forward return than its calm
state.

So the tests here are largely about the thing the strategy must NOT do, which is
treat volatility as a reason to leave. It sizes down and stays in.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.engine.backtest import UNLIMITED, run_backtest
from sentinel.evaluation.causality import check_causality
from sentinel.evaluation.null_test import run_null_test
from sentinel.sandbox.generators.gbm import GBMGenerator
from sentinel.sandbox.generators.heston import HestonGenerator
from sentinel.sandbox.generators.regime import RegimeSwitchingGenerator
from sentinel.strategies.baseline import BuyAndHold
from sentinel.strategies.volatility import RegimeVolatilityTarget, VolatilityTarget


@pytest.fixture(scope="module")
def clustered():
    """Heston: volatility clusters, direction is unforecastable.

    The right market for this strategy. It has exactly the property volatility
    targeting exploits -- risk is predictable -- and none of the property it must
    not rely on, since Heston's direction is pure noise.
    """
    return HestonGenerator(mu=0.08).generate(n_steps=2520, n_assets=1, seed=31).data


@pytest.fixture(scope="module")
def calm_market():
    return GBMGenerator(mu=0.06, sigma=0.16).generate(n_steps=1600, seed=13).data


class TestCausality:
    @pytest.mark.parametrize(
        "strategy", [VolatilityTarget(), RegimeVolatilityTarget()], ids=lambda s: s.name
    )
    def test_is_causal(self, strategy, calm_market) -> None:
        report = check_causality(strategy, calm_market)
        assert report.is_causal, str(report)

    def test_the_regime_variant_does_not_use_the_final_refit(self, calm_market) -> None:
        """The bug this test exists for was real and was caught by the detector.

        The first implementation read `classifier.last_parameters` -- the
        parameters from the most recent refit -- and applied them to every row,
        including rows from years earlier. `check_causality` reported LOOKAHEAD
        DETECTED at row 756, drift 2.67e-02, within seconds of it being written.

        Nothing about the code looked wrong. `last_parameters` reads like an
        accessor and the volatility forecast it produced was entirely plausible.
        That is the whole argument for testing causality mechanically rather than
        by review.
        """
        report = check_causality(RegimeVolatilityTarget(), calm_market, tolerance=1e-12)
        assert report.max_discrepancy == 0.0, str(report)


class TestSizing:
    def test_exposure_falls_as_volatility_rises(self, clustered) -> None:
        strategy = VolatilityTarget(target_volatility=0.12, band=0.0)
        weights = strategy.compute_weights(clustered).to_numpy().ravel()
        volatility = strategy.forecast_volatility(clustered, clustered.tickers[0])

        usable = np.isfinite(volatility) & (weights > 0) & (weights < 1.0)
        assert usable.sum() > 100
        correlation = np.corrcoef(volatility[usable], weights[usable])[0, 1]
        assert correlation < -0.9, "weight must be inversely proportional to volatility"

    def test_never_leaves_the_market_entirely_after_warmup(self, clustered) -> None:
        """The point of the whole strategy.

        On real equities the high-volatility state is *compensated* -- it has a
        higher forward return than the calm state -- so going flat sells exactly
        the periods you are being paid to hold. Sizing down keeps the exposure.
        """
        weights = VolatilityTarget().compute_weights(clustered).to_numpy().ravel()
        after_warmup = weights[100:]
        assert (after_warmup > 0).all()

    def test_never_borrows(self, clustered) -> None:
        """No leverage, permanently. Without the cap, volatility targeting quietly
        becomes a leveraged strategy in calm markets, which is how it blows up."""
        weights = VolatilityTarget(target_volatility=0.40).compute_weights(clustered)
        assert weights.to_numpy().max() <= 1.0

    def test_a_quiet_stretch_cannot_produce_an_enormous_position(self) -> None:
        """Dividing by a small estimated volatility is the standard failure mode."""
        strategy = VolatilityTarget(target_volatility=0.12, floor_volatility=0.04)
        market = GBMGenerator(mu=0.0, sigma=0.005).generate(n_steps=800, seed=2).data
        weights = strategy.compute_weights(market).to_numpy()
        assert weights.max() <= 1.0

    def test_delivers_roughly_the_volatility_it_targets(self, clustered) -> None:
        """The test of whether the mechanism does what it claims."""
        result = run_backtest(clustered, VolatilityTarget(target_volatility=0.10), limits=UNLIMITED)
        realised = result.returns.iloc[100:].std() * np.sqrt(252)
        assert 0.06 < realised < 0.15

    def test_reduces_volatility_relative_to_holding_the_market(self, clustered) -> None:
        targeted = run_backtest(clustered, VolatilityTarget(), limits=UNLIMITED)
        held = run_backtest(clustered, BuyAndHold(), limits=UNLIMITED)
        assert targeted.performance.volatility < held.performance.volatility

    def test_the_no_trade_band_cuts_turnover(self, clustered) -> None:
        wide = run_backtest(clustered, VolatilityTarget(band=0.20), limits=UNLIMITED)
        tight = run_backtest(clustered, VolatilityTarget(band=0.0), limits=UNLIMITED)
        assert wide.annual_turnover < tight.annual_turnover


class TestItIsNotACrashDefence:
    def test_it_is_still_invested_going_into_the_first_leg_down(self) -> None:
        """Stated as a test so the claim cannot quietly inflate.

        Volatility targeting reduces exposure *after* volatility rises. It cuts
        the depth and length of a drawdown; it does not avoid one. A version that
        did would be predicting the crash, which this project has already
        measured itself unable to do.
        """
        scenario = RegimeSwitchingGenerator().generate(n_steps=2520, n_assets=1, seed=5)
        weights = VolatilityTarget().compute_weights(scenario.data).to_numpy().ravel()
        regimes = scenario.truth.regimes

        switches = np.flatnonzero(np.diff(regimes) == 1) + 1  # calm -> stressed
        switches = switches[(switches > 100) & (switches < len(weights) - 1)]
        assert len(switches) > 3
        # On the day stress begins, the strategy is still holding a real position.
        assert np.median(weights[switches]) > 0.3


class TestNullBehaviour:
    def test_does_not_profit_on_noise(self) -> None:
        result = run_null_test(
            VolatilityTarget(),
            GBMGenerator(mu=0.0, sigma=0.16),
            n_markets=60,
            n_steps=1260,
            workers=1,
        )
        assert result.passed()

    def test_does_not_profit_on_forecastable_volatility_alone(self) -> None:
        """Heston at mu = 0 has predictable *risk* and unpredictable direction.

        The sharpest test of this strategy, and the one it could plausibly fail.
        Volatility targeting genuinely exploits predictable volatility, so it
        might look like it is finding something. It must not: knowing how large
        tomorrow's move will be says nothing about its sign, and a market with
        zero drift pays nothing for holding it at any size.
        """
        result = run_null_test(
            VolatilityTarget(),
            HestonGenerator(mu=0.0),
            n_markets=60,
            n_steps=1260,
            workers=1,
        )
        assert result.passed(), result.report()


class TestGuards:
    @pytest.mark.parametrize(
        "kwargs,message",
        [
            ({"target_volatility": 0.0}, "target_volatility must be positive"),
            ({"window": 1}, "window must be at least 2"),
            ({"max_weight": 1.5}, "does not use leverage"),
            ({"band": 1.0}, "band must be"),
            ({"floor_volatility": 0.0}, "floor_volatility must be positive"),
        ],
    )
    def test_rejects_impossible_parameters(self, kwargs, message) -> None:
        with pytest.raises(ValueError, match=message):
            VolatilityTarget(**kwargs)
