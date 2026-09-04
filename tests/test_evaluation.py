"""Tests for the evaluation harness -- the instruments, not the strategies.

The Null Test and the Recovery Test are the two measurements the whole project
rests on. Every claim it will ever make is of the form "this beats the null
floor" or "this detects a signal of strength X", so an error in either instrument
silently corrupts every downstream result while leaving all of them looking fine.

Two of the tests here matter more than the rest:

- `test_null_test_catches_a_strategy_that_peeks` proves the Null Test is not
  blind. A test that has never caught anything is not evidence of correctness.
- `test_recovery_curve_is_calibrated_at_zero_signal` proves the Recovery Test's
  detection threshold means what it claims, by checking the one point on the
  curve whose value is known in advance from the definition of a percentile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.evaluation.null_test import NullTestResult, run_null_test
from sentinel.evaluation.recovery_test import RecoveryCurve, RecoveryPoint, run_recovery_test
from sentinel.evaluation.sweep import sweep_markets
from sentinel.sandbox.generators.ar1 import AR1Generator
from sentinel.sandbox.generators.gbm import GBMGenerator
from sentinel.sandbox.generators.ou import OUGenerator
from sentinel.sandbox.market import MarketData
from sentinel.strategies.base import Strategy
from sentinel.strategies.baseline import (
    AbsoluteMomentum,
    AlwaysCash,
    BuyAndHold,
    ShortHorizonMomentum,
)

# Small and single-process throughout: these test the harness's logic, not its
# statistical power, and a process pool inside a test suite is slower than the
# work it parallelises at this size.
MARKETS = 40
STEPS = 400


class TomorrowPeeker(Strategy):
    """Deliberately broken: holds the asset exactly when tomorrow is an up day.

    It is here for the same reason the cheat strategies in `test_no_lookahead.py`
    are there. The Null Test's entire value is that it fails when a strategy
    profits from noise, and the only way to know it does is to hand it a strategy
    that certainly does. Do not delete it.
    """

    name = "tomorrow_peeker"

    def compute_weights(self, data: MarketData) -> pd.DataFrame:
        forward = data.prices.pct_change().shift(-1)
        return (forward > 0).astype(float)


class HalfInvested(Strategy):
    """Holds a constant half position. No timing, no trading after day one."""

    name = "half_invested"

    def compute_weights(self, data: MarketData) -> pd.DataFrame:
        return pd.DataFrame(0.5, index=data.prices.index, columns=data.tickers)


@pytest.fixture(scope="module")
def null_market() -> GBMGenerator:
    return GBMGenerator(mu=0.0, sigma=0.16)


# --------------------------------------------------------------------------
# The shared sweep runner
# --------------------------------------------------------------------------

def test_sweep_is_reproducible(null_market: GBMGenerator) -> None:
    """Same seeds, same answers. A sweep that redraws its own data proves nothing."""
    first = sweep_markets(BuyAndHold(), null_market, n_markets=8, n_steps=300, workers=1)
    second = sweep_markets(BuyAndHold(), null_market, n_markets=8, n_steps=300, workers=1)
    np.testing.assert_array_equal(first[0], second[0])


def test_sweep_seeds_produce_genuinely_different_markets(null_market: GBMGenerator) -> None:
    """Each market must be an independent draw, or `n_markets` overstates the sample."""
    sharpes, _, _ = sweep_markets(BuyAndHold(), null_market, n_markets=8, n_steps=300, workers=1)
    assert len(np.unique(sharpes)) == len(sharpes)


def test_sweep_shifting_the_offset_changes_the_sample(null_market: GBMGenerator) -> None:
    sharpes, _, _ = sweep_markets(BuyAndHold(), null_market, n_markets=8, n_steps=300, workers=1)
    shifted, _, _ = sweep_markets(
        BuyAndHold(), null_market, n_markets=8, n_steps=300, seed_offset=999, workers=1
    )
    assert not np.allclose(sharpes, shifted)


def test_sweep_rejects_an_empty_request(null_market: GBMGenerator) -> None:
    with pytest.raises(ValueError, match="at least one market"):
        sweep_markets(BuyAndHold(), null_market, n_markets=0, n_steps=300, workers=1)


# --------------------------------------------------------------------------
# The Null Test
# --------------------------------------------------------------------------

def test_null_test_refuses_a_market_containing_a_signal() -> None:
    """The control arm must not run on a market that secretly has something in it.

    Allowing it would raise the measured noise floor, which would then quietly
    excuse a genuinely broken strategy -- the failure would be invisible and
    would make everything downstream look better than it is.
    """
    with pytest.raises(ValueError, match="declares an exploitable signal"):
        run_null_test(BuyAndHold(), AR1Generator(phi=0.2), n_markets=2, n_steps=300, workers=1)


def test_null_test_accepts_ar1_only_when_phi_is_zero() -> None:
    """At phi = 0 the AR(1) generator *is* a random walk, and is allowed through."""
    result = run_null_test(
        AlwaysCash(), AR1Generator(mu=0.0, phi=0.0), n_markets=4, n_steps=300, workers=1
    )
    assert result.n_markets == 4


def test_null_test_catches_a_strategy_that_peeks(null_market: GBMGenerator) -> None:
    """THE test. A strategy reading tomorrow's return must fail loudly.

    If this ever passes, the Null Test has gone blind and every "PASS" the
    project has recorded is worthless.
    """
    result = run_null_test(
        TomorrowPeeker(), null_market, n_markets=MARKETS, n_steps=STEPS, workers=1
    )
    assert result.mean_sharpe > 1.0
    assert result.t_statistic > 3.0
    assert not result.passed()
    assert "FAIL" in result.report()


def test_an_honest_strategy_passes_the_null_test(null_market: GBMGenerator) -> None:
    result = run_null_test(
        AbsoluteMomentum(lookback=60, rebalance_days=21),
        null_market,
        n_markets=MARKETS,
        n_steps=STEPS,
        workers=1,
    )
    assert result.passed()
    assert result.mean_sharpe < 0.5


def test_holding_cash_scores_exactly_zero(null_market: GBMGenerator) -> None:
    """The floor of the floor: no position, no trades, no cost, no result."""
    result = run_null_test(AlwaysCash(), null_market, n_markets=8, n_steps=300, workers=1)
    assert np.all(result.sharpes == 0.0)
    assert result.profitable_fraction == 0.0
    assert result.passed()


def test_a_zero_drift_market_pays_a_holder_nothing(null_market: GBMGenerator) -> None:
    """Why null markets use mu = 0.

    With drift, buy-and-hold profits from exposure rather than skill and the
    Null Test would measure the wrong thing entirely. At mu = 0 it must be a
    coin flip, so roughly half the markets are profitable.
    """
    result = run_null_test(HalfInvested(), null_market, n_markets=MARKETS, n_steps=STEPS, workers=1)
    assert result.passed()
    assert 0.25 < result.profitable_fraction < 0.75


def test_trading_costs_push_a_busy_strategy_below_a_calm_one(null_market: GBMGenerator) -> None:
    """Turnover shifts the null distribution down, and with it the noise floor.

    This pins the corrected claim in the module docstring. The intuitive story --
    a busier strategy gets more chances to be lucky, so its null distribution is
    wider and its bar higher -- is wrong, and the next test shows why. What
    turnover actually does is pay costs on markets that offer nothing back, which
    moves the whole distribution left.

    The consequence is a trap worth stating: a busy strategy clears its own noise
    floor *more* easily, purely because it starts out losing. Beating the floor
    is necessary, not sufficient. Beating zero is the other half.
    """
    busy = run_null_test(
        AbsoluteMomentum(lookback=5, rebalance_days=1),
        null_market,
        n_markets=MARKETS,
        n_steps=STEPS,
        workers=1,
    )
    calm = run_null_test(HalfInvested(), null_market, n_markets=MARKETS, n_steps=STEPS, workers=1)
    assert busy.mean_sharpe < calm.mean_sharpe
    assert busy.noise_floor < calm.noise_floor


def test_null_spread_is_set_by_track_length_not_by_turnover(null_market: GBMGenerator) -> None:
    """The width of the null distribution is 1/sqrt(years), whoever is trading.

    An estimated Sharpe has standard error 1/sqrt(T) in years, and that dominates
    everything a strategy does -- scaling exposure scales mean and volatility
    together, leaving the ratio's spread untouched.

    Which is why a noise floor quoted without an evaluation window is not a
    number anyone can use. Measured across turnover from 0.8 to 84 round trips a
    year, the spread stayed at 0.31 on ten-year markets against a predicted
    0.316.
    """
    years = STEPS / 252
    expected = 1.0 / np.sqrt(years)

    busy = run_null_test(
        AbsoluteMomentum(lookback=5, rebalance_days=1),
        null_market,
        n_markets=MARKETS,
        n_steps=STEPS,
        workers=1,
    )
    calm = run_null_test(HalfInvested(), null_market, n_markets=MARKETS, n_steps=STEPS, workers=1)

    for result in (busy, calm):
        assert result.sharpes.std(ddof=1) == pytest.approx(expected, rel=0.35)


class TestNullTestResult:
    """The arithmetic behind the verdict, on data with known answers."""

    def _result(self, sharpes: np.ndarray) -> NullTestResult:
        zeros = np.zeros_like(sharpes)
        return NullTestResult("s", "m", len(sharpes), sharpes, zeros, zeros)

    def test_noise_floor_is_the_95th_percentile(self) -> None:
        sharpes = np.linspace(-1.0, 1.0, 101)
        assert self._result(sharpes).noise_floor == pytest.approx(0.9)
        assert self._result(sharpes).noise_floor_99 == pytest.approx(0.98)

    def test_t_statistic_counts_standard_errors_above_zero(self) -> None:
        sharpes = np.array([1.0, 1.0, 1.0, 1.0, 2.0])
        result = self._result(sharpes)
        expected = np.mean(sharpes) / (np.std(sharpes, ddof=1) / np.sqrt(5))
        assert result.t_statistic == pytest.approx(expected)

    def test_a_negative_mean_passes_comfortably(self) -> None:
        """Losing money on noise is the correct behaviour: trading costs something."""
        assert self._result(np.linspace(-0.5, -0.4, 30)).passed()

    def test_an_identical_sample_does_not_divide_by_zero(self) -> None:
        assert self._result(np.full(10, 0.4)).t_statistic == 0.0


# --------------------------------------------------------------------------
# The Recovery Test
# --------------------------------------------------------------------------

def test_recovery_requires_a_zero_signal_arm() -> None:
    """Every point on the curve is measured against the zero arm. It cannot be optional."""
    with pytest.raises(ValueError, match="strength 0.0"):
        run_recovery_test(
            BuyAndHold(),
            lambda phi: AR1Generator(mu=0.0, phi=phi),
            strengths=[0.05, 0.1],
            n_markets=4,
            n_steps=300,
            workers=1,
        )


def test_recovery_rejects_a_null_arm_that_still_contains_a_signal() -> None:
    """A builder that ignores its argument would compare a signal against itself."""
    with pytest.raises(ValueError, match="still declares a signal"):
        run_recovery_test(
            BuyAndHold(),
            lambda phi: AR1Generator(mu=0.0, phi=0.2),
            strengths=[0.0, 0.2],
            n_markets=4,
            n_steps=300,
            workers=1,
        )


def test_recovery_curve_is_calibrated_at_zero_signal() -> None:
    """The zero-signal point must sit at 5% detection, by the definition of p95.

    This is the curve's self-check. If it lands anywhere else, the floor being
    compared against does not describe the markets being run, and no threshold
    read off the curve means anything.
    """
    curve = run_recovery_test(
        AbsoluteMomentum(lookback=60, rebalance_days=21),
        lambda phi: AR1Generator(mu=0.0, sigma=0.16, phi=phi),
        strengths=[0.0, 0.1],
        n_markets=MARKETS,
        n_steps=STEPS,
        workers=1,
    )
    zero = next(p for p in curve.points if p.strength == 0.0)
    # With n markets the empirical p95 puts ceil(0.05n) points above it at most.
    assert zero.detection_power <= 0.05 + 1.0 / MARKETS


def test_a_strong_signal_is_detected_more_often_than_none() -> None:
    """The curve must actually rise. A flat curve means the plant is not landing."""
    curve = run_recovery_test(
        ShortHorizonMomentum(),
        lambda phi: AR1Generator(mu=0.0, sigma=0.16, phi=phi),
        strengths=[0.0, 0.30],
        n_markets=MARKETS,
        n_steps=1260,
        workers=1,
    )
    strong = next(p for p in curve.points if p.strength == 0.30)
    assert strong.detection_power > 0.50
    assert strong.lift > 0.0


def test_a_rule_on_the_wrong_horizon_stays_at_the_null_rate() -> None:
    """A twelve-month rule cannot see a one-day signal, however strong it is.

    phi = 0.3 is an order of magnitude beyond anything real markets offer, and
    `AbsoluteMomentum` at its conventional 252-day horizon still detects it at
    roughly the 5-in-100 rate chance alone produces. It is not weak here, it is
    blind: the signal acts at a one-day lag and a 252-day average cannot
    represent it.

    This is why `ShortHorizonMomentum` is in the control arm. Without a rule
    matched to the signal's timescale, the Recovery Test would report "no
    strategy detects this" when the truth is "nothing we ran could have", and
    that mistake reads as evidence to abandon the project.
    """
    curve = run_recovery_test(
        AbsoluteMomentum(lookback=252, rebalance_days=21),
        lambda phi: AR1Generator(mu=0.0, sigma=0.16, phi=phi),
        strengths=[0.0, 0.30],
        n_markets=MARKETS,
        n_steps=1260,
        workers=1,
    )
    strong = next(p for p in curve.points if p.strength == 0.30)
    assert strong.detection_power < 0.35


def test_every_strength_level_reuses_the_same_seeds() -> None:
    """Common random numbers: levels must differ by the signal, not by luck.

    Verified from the outside -- an unpaired sweep would not reproduce exactly
    when re-run against a hand-built pair of arms using the same offset.
    """
    builder = lambda phi: AR1Generator(mu=0.0, sigma=0.16, phi=phi)
    curve = run_recovery_test(
        BuyAndHold(), builder, strengths=[0.0, 0.2], n_markets=6, n_steps=300, workers=1
    )
    for strength in (0.0, 0.2):
        direct, _, _ = sweep_markets(
            BuyAndHold(), builder(strength), n_markets=6, n_steps=300, workers=1
        )
        point = next(p for p in curve.points if p.strength == strength)
        np.testing.assert_allclose(point.sharpes, direct)


def test_an_external_noise_floor_is_used_when_supplied() -> None:
    """Later phases pass the floor measured by the larger Null Test sweep."""
    curve = run_recovery_test(
        BuyAndHold(),
        lambda phi: AR1Generator(mu=0.0, phi=phi),
        strengths=[0.0, 0.1],
        null_p95=99.0,
        n_markets=6,
        n_steps=300,
        workers=1,
    )
    assert all(point.detection_power == 0.0 for point in curve.points)
    assert curve.metadata["null_p95_source"] == "supplied"


def test_ou_null_arm_may_be_a_different_generator() -> None:
    """Theta = 0 is not a valid OU process, so the null arm is a matched GBM."""
    curve = run_recovery_test(
        BuyAndHold(),
        lambda theta: GBMGenerator(mu=0.0, sigma=0.16) if theta == 0 else OUGenerator(theta=theta),
        strengths=[0.0, 4.0],
        n_markets=6,
        n_steps=300,
        workers=1,
        parameter="theta",
    )
    assert curve.generator == "gbm"
    assert curve.parameter == "theta"


class TestDetectionThreshold:
    """Interpolation, on curves built by hand so the right answer is known."""

    def _curve(self, pairs: list[tuple[float, float]]) -> RecoveryCurve:
        """Build a curve whose detection power at each strength is exactly `power`."""
        points = []
        for strength, power in pairs:
            n = 100
            above = int(round(power * n))
            sharpes = np.concatenate([np.full(above, 2.0), np.full(n - above, -2.0)])
            points.append(
                RecoveryPoint(strength, sharpes, np.zeros(n), null_p95=0.0, null_mean=0.0)
            )
        return RecoveryCurve("s", "g", "phi", points)

    def test_interpolates_between_the_bracketing_levels(self) -> None:
        curve = self._curve([(0.0, 0.05), (0.10, 0.30), (0.20, 0.70)])
        # 50% sits halfway between 30% and 70%, so halfway between 0.10 and 0.20.
        assert curve.detection_threshold(0.5) == pytest.approx(0.15)

    def test_returns_none_when_the_curve_never_gets_there(self) -> None:
        """A strategy that never detects the signal is a result, not an error."""
        assert self._curve([(0.0, 0.05), (0.3, 0.2)]).detection_threshold(0.5) is None

    def test_reports_the_first_level_that_already_clears_the_bar(self) -> None:
        assert self._curve([(0.0, 0.9), (0.3, 0.95)]).detection_threshold(0.5) == 0.0

    def test_calibration_error_measures_distance_from_five_percent(self) -> None:
        assert self._curve([(0.0, 0.05), (0.2, 0.6)]).calibration_error == pytest.approx(0.0)
        assert self._curve([(0.0, 0.25), (0.2, 0.6)]).calibration_error == pytest.approx(0.20)
