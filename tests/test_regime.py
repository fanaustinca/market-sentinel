"""Tests for regime detection -- the plan's centrepiece.

Three things are being protected here, in order of how expensive they would be to
get wrong:

**The filtered/smoothed distinction.** Every HMM library's default inference uses
the whole series to estimate each day's state, which is the right answer for
describing history and pure lookahead for trading. It is invisible in the output:
a smoothed state sequence looks entirely plausible and makes regime detection
appear far easier than it is. `test_smoothing_is_not_causal` deliberately proves
`smooth()` fails the causality check, so the distinction stays a fact rather than
a comment.

**Alignment.** `regimes[t]` is the state governing the return from `t` to `t+1`.
Off-by-one here would score the model against the wrong day and produce a small,
believable, entirely wrong accuracy.

**Detection lag.** The metric that decides whether a classifier is usable, tested
against sequences built by hand so the answer is known rather than plausible.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.ai.regime.classifier import WalkForwardRegimeClassifier
from sentinel.ai.regime.hmm import GaussianHMM2State
from sentinel.evaluation.causality import check_causality
from sentinel.evaluation.oracle import DelayedRegimeOracle, RegimeOracle
from sentinel.evaluation.regime_score import score_regimes
from sentinel.engine.backtest import UNLIMITED, run_backtest
from sentinel.sandbox.generators.gbm import GBMGenerator
from sentinel.sandbox.generators.regime import RegimeSwitchingGenerator
from sentinel.sandbox.market import MarketData
from sentinel.strategies.base import Strategy
from sentinel.strategies.regime import RegimeAwareStrategy, RegimeGate


@pytest.fixture(scope="module")
def scenario():
    return RegimeSwitchingGenerator().generate(n_steps=2520, n_assets=1, seed=7)


@pytest.fixture(scope="module")
def returns(scenario) -> np.ndarray:
    return np.log(scenario.data.prices.iloc[:, 0]).diff().dropna().to_numpy()


@pytest.fixture(scope="module")
def fitted(returns) -> GaussianHMM2State:
    return GaussianHMM2State().fit(returns)


# --------------------------------------------------------------------------
# The HMM itself
# --------------------------------------------------------------------------

class TestParameterRecovery:
    """The model must recover parameters it was never told, from data alone.

    Same standard the generators were held to in Phase 0. A regime model that
    cannot recover a regime process it is looking straight at has no chance on
    anything subtler.
    """

    def test_recovers_the_volatility_of_each_state(self, fitted: GaussianHMM2State) -> None:
        annualised = fitted.params.volatilities * np.sqrt(252)
        assert annualised[0] == pytest.approx(0.12, abs=0.02)
        assert annualised[1] == pytest.approx(0.32, abs=0.04)

    def test_recovers_how_long_each_state_lasts(self, fitted: GaussianHMM2State) -> None:
        """True durations are 100 and 33 days, set by persistence 0.99 and 0.97."""
        durations = fitted.params.expected_durations
        assert durations[0] == pytest.approx(100, rel=0.35)
        assert durations[1] == pytest.approx(33, rel=0.35)

    def test_transition_rows_are_probabilities(self, fitted: GaussianHMM2State) -> None:
        np.testing.assert_allclose(fitted.params.transition.sum(axis=1), 1.0)
        assert (fitted.params.transition >= 0).all()

    def test_calm_state_is_always_first(self) -> None:
        """EM has no idea which state is which, so the ordering is imposed.

        Without it roughly half of all fits come out with the labels swapped and
        every accuracy score becomes a coin flip laid over the real answer.
        """
        for seed in range(6):
            scenario = RegimeSwitchingGenerator().generate(n_steps=1500, seed=seed)
            data = np.log(scenario.data.prices.iloc[:, 0]).diff().dropna().to_numpy()
            params = GaussianHMM2State().fit(data).params
            assert params.variances[0] <= params.variances[1]


class TestFilteringIsCausal:
    """The distinction the whole module turns on."""

    def test_filtering_never_revises_the_past(self, fitted, returns) -> None:
        full = fitted.filter(returns)
        for cut in (500, 1200, 2000):
            partial = fitted.filter(returns[:cut])
            np.testing.assert_allclose(full[:cut], partial, atol=1e-12)

    def test_smoothing_is_not_causal(self, fitted, returns) -> None:
        """Deliberately proves `smooth()` fails, so the distinction stays real.

        A test suite that only ever confirms good behaviour cannot show its
        checks work. If this test ever passes -- if smoothing turns out to be
        causal -- then the causality check is broken, not the mathematics.
        """
        full = fitted.smooth(returns)
        partial = fitted.smooth(returns[:1200])
        assert not np.allclose(full[:1200], partial, atol=1e-6)

    def test_hindsight_makes_regimes_look_easier_than_they_are(
        self, fitted, returns, scenario
    ) -> None:
        """Smoothed accuracy beats filtered accuracy. That gap is the hindsight bonus.

        Quoting a smoothed number as if it were achievable live is one of the
        most common ways regime models are oversold, and the size of the gap is
        the measure of how misleading it would be.
        """
        truth = scenario.truth.regimes
        filtered = (fitted.filter(returns)[:, 1] > 0.5).astype(int)
        smoothed = (fitted.smooth(returns)[:, 1] > 0.5).astype(int)
        assert (smoothed == truth).mean() > (filtered == truth).mean()

    def test_one_step_ahead_differs_from_the_filtered_state(self, fitted, returns) -> None:
        """Sizing needs the state governing the *next* return, not the last one."""
        assert not np.allclose(fitted.filter(returns), fitted.predict_next(returns))


class TestHMMGuards:
    def test_refuses_a_sample_too_short_to_identify_two_states(self) -> None:
        with pytest.raises(ValueError, match="at least 20 observations"):
            GaussianHMM2State().fit(np.zeros(10))

    def test_refuses_non_finite_observations(self) -> None:
        data = np.random.default_rng(0).normal(size=100)
        data[7] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            GaussianHMM2State().fit(data)

    def test_inference_before_fitting_is_an_error(self) -> None:
        with pytest.raises(RuntimeError, match="fit\\(\\) must be called"):
            GaussianHMM2State().filter(np.zeros(50))

    def test_a_single_regime_does_not_produce_a_degenerate_fit(self) -> None:
        """On plain GBM there is only one state. EM must not collapse to a spike.

        The classic failure of EM on Gaussian mixtures is a state that claims one
        observation and drives its variance to zero, producing infinite
        likelihood and nonsense probabilities.
        """
        data = np.log(GBMGenerator(mu=0.0).generate(n_steps=1500, seed=3).data.prices.iloc[:, 0])
        params = GaussianHMM2State().fit(data.diff().dropna().to_numpy()).params
        assert (params.variances > 1e-10).all()
        assert np.isfinite(params.transition).all()


# --------------------------------------------------------------------------
# The walk-forward classifier
# --------------------------------------------------------------------------

class TestWalkForwardClassifier:
    def test_is_causal(self, scenario) -> None:
        strategy = RegimeAwareStrategy()
        assert check_causality(strategy, scenario.data).is_causal

    def test_says_nothing_before_it_has_enough_history(self, scenario) -> None:
        """Warmup rows are NaN, not a guess. Filling them would be lookahead."""
        classifier = WalkForwardRegimeClassifier(min_train=756)
        probabilities = classifier.probabilities(scenario.data)
        assert probabilities.iloc[:756].isna().all().all()
        assert probabilities.iloc[756:].notna().all().all()

    def test_probabilities_are_a_distribution(self, scenario) -> None:
        frame = WalkForwardRegimeClassifier().probabilities(scenario.data).dropna()
        states = frame[["p_calm", "p_stressed"]]
        np.testing.assert_allclose(states.sum(axis=1), 1.0, atol=1e-9)
        assert ((states >= 0) & (states <= 1)).all().all()

    def test_the_variance_reported_is_the_one_in_force_at_that_row(self, scenario) -> None:
        """Not the one the final refit eventually settles on.

        `probabilities` carries the per-state variances alongside the state
        probabilities so that a volatility forecast can be built causally. The
        obvious alternative -- read `last_parameters` after the fact -- is
        lookahead, and it is invisible: the accessor reads like an accessor and
        the forecast it produces looks entirely sensible. `check_causality`
        reported LOOKAHEAD DETECTED at row 756 the first time it was written that
        way, which is the only reason it is not still there.
        """
        classifier = WalkForwardRegimeClassifier(min_train=756, retrain_every=126)
        frame = classifier.probabilities(scenario.data)
        variances = frame["variance_stressed"].dropna()
        # Refits happen every 126 rows, so the series must be piecewise constant
        # and must actually change -- a constant column would mean one model was
        # applied to the whole series.
        assert variances.nunique() > 1
        assert variances.nunique() <= len(variances) // 100

    def test_the_variance_forecast_is_causal(self, scenario) -> None:
        classifier = WalkForwardRegimeClassifier()
        full = classifier.forecast_variance(scenario.data)
        partial = classifier.forecast_variance(
            type(scenario.data)(prices=scenario.data.prices.iloc[:1800], name="cut")
        )
        np.testing.assert_allclose(
            full.to_numpy()[:1800], partial.to_numpy(), atol=1e-12, equal_nan=True
        )

    def test_beats_chance_on_a_market_that_really_has_regimes(self, scenario) -> None:
        stressed = (
            WalkForwardRegimeClassifier()
            .probabilities(scenario.data)["p_stressed"]
            .to_numpy()[:-1]
        )
        score = score_regimes(stressed, scenario.truth.regimes)
        assert score.balanced_accuracy > 0.75
        assert score.auc > 0.85

    def test_incremental_filtering_matches_refitting_from_scratch(self, scenario) -> None:
        """Between refits the state advances one step at a time, for speed.

        That shortcut must be exactly equivalent to re-running the filter over
        the whole window, or the classifier drifts away from the model it claims
        to be using.
        """
        classifier = WalkForwardRegimeClassifier(min_train=400, retrain_every=10_000)
        probabilities = classifier.probabilities(scenario.data)

        returns = np.log(scenario.data.prices.iloc[:, 0]).diff().to_numpy()[1:]
        model = GaussianHMM2State(seed=0).fit(returns[max(0, 400 - 1260) : 400])
        for t in (400, 450, 700, 1500):
            expected = model.predict_next(returns[:t])[-1]
            assert probabilities["p_stressed"].to_numpy()[t] == pytest.approx(expected[1], abs=1e-9)

    def test_rejects_a_training_window_too_short_for_two_states(self) -> None:
        with pytest.raises(ValueError, match="min_train below 100"):
            WalkForwardRegimeClassifier(min_train=50)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

class TestRegimeScore:
    def test_detection_lag_on_a_sequence_with_a_known_answer(self) -> None:
        """Truth switches at index 5; the model agrees at index 8. Lag is 3."""
        truth = np.array([0] * 5 + [1] * 15)
        probabilities = np.array([0.0] * 8 + [1.0] * 12)
        score = score_regimes(probabilities, truth)
        assert score.n_switches == 1
        assert list(score.detection_lags) == [3]
        assert score.missed_switches == 0

    def test_a_switch_recognised_only_after_it_ended_counts_as_missed(self) -> None:
        """Being right about a crash after the recovery is not being late. It is being wrong."""
        truth = np.array([0] * 5 + [1] * 5 + [0] * 10)
        probabilities = np.array([0.0] * 12 + [1.0] * 8)
        score = score_regimes(probabilities, truth)
        assert score.missed_switches == 1
        assert score.detected_fraction == pytest.approx(0.5)

    def test_instant_detection_scores_zero_lag(self) -> None:
        truth = np.array([0] * 10 + [1] * 10)
        score = score_regimes(truth.astype(float), truth)
        assert list(score.detection_lags) == [0]
        assert score.accuracy == 1.0

    def test_accuracy_alone_flatters_a_model_that_never_predicts_stress(self) -> None:
        """Why balanced accuracy is reported beside it, and read first."""
        truth = np.array([0] * 80 + [1] * 20)
        never = np.zeros(100)
        score = score_regimes(never, truth)
        assert score.accuracy == pytest.approx(0.80)
        assert score.balanced_accuracy == pytest.approx(0.50)

    def test_warmup_is_dropped_rather_than_scored_as_wrong(self) -> None:
        """A model that has not spoken yet is silent, not incorrect."""
        truth = np.array([0] * 10 + [1] * 10)
        probabilities = np.concatenate([np.full(10, np.nan), np.ones(10)])
        score = score_regimes(probabilities, truth)
        assert score.n_scored == 10
        assert score.accuracy == 1.0

    def test_misalignment_is_refused_rather_than_scored(self) -> None:
        with pytest.raises(ValueError, match="alignment error"):
            score_regimes(np.zeros(50), np.zeros(49, dtype=int))

    def test_auc_of_a_perfect_ranking_is_one(self) -> None:
        truth = np.array([0] * 10 + [1] * 10)
        assert score_regimes(np.linspace(0, 1, 20), truth).auc == pytest.approx(1.0)

    def test_a_constant_prediction_scores_auc_of_a_half(self) -> None:
        """Ties must be averaged, or a model with no view scores 1.0 or 0.0 by sort order."""
        truth = np.array([0] * 10 + [1] * 10)
        assert score_regimes(np.full(20, 0.3), truth).auc == pytest.approx(0.5)

    def test_false_alarms_count_only_calm_days(self) -> None:
        truth = np.array([0] * 10 + [1] * 10)
        probabilities = np.concatenate([np.array([1.0, 1.0]), np.zeros(8), np.ones(10)])
        assert score_regimes(probabilities, truth).false_alarm_rate == pytest.approx(0.2)


# --------------------------------------------------------------------------
# The oracles
# --------------------------------------------------------------------------

class TestOracles:
    def test_oracle_holds_exactly_during_calm_periods(self, scenario) -> None:
        weights = RegimeOracle(scenario.truth.regimes).compute_weights(scenario.data)
        column = weights.to_numpy().ravel()
        np.testing.assert_array_equal(
            column[:-1], (scenario.truth.regimes == 0).astype(float)
        )

    def test_oracle_refuses_a_mismatched_answer_key(self, scenario) -> None:
        """Alignment is the one mistake that would silently corrupt the ceiling."""
        with pytest.raises(ValueError, match="regime labels"):
            RegimeOracle(scenario.truth.regimes[:-5]).compute_weights(scenario.data)

    def test_the_oracle_beats_holding_the_market(self, scenario) -> None:
        """Sanity: perfect regime knowledge must be worth something, or the
        generator does not contain what it claims to."""
        oracle = run_backtest(scenario.data, RegimeOracle(scenario.truth.regimes), limits=UNLIMITED)
        from sentinel.strategies.baseline import BuyAndHold

        held = run_backtest(scenario.data, BuyAndHold(), limits=UNLIMITED)
        assert oracle.performance.sharpe > held.performance.sharpe

    def test_delay_costs_the_oracle_money(self, scenario) -> None:
        """Prices the lag alone, holding everything else perfect."""
        instant = run_backtest(
            scenario.data, RegimeOracle(scenario.truth.regimes), limits=UNLIMITED
        )
        late = run_backtest(
            scenario.data, DelayedRegimeOracle(scenario.truth.regimes, lag=20), limits=UNLIMITED
        )
        assert late.performance.sharpe < instant.performance.sharpe

    def test_zero_lag_is_the_instant_oracle(self, scenario) -> None:
        a = RegimeOracle(scenario.truth.regimes).compute_weights(scenario.data)
        b = DelayedRegimeOracle(scenario.truth.regimes, lag=0).compute_weights(scenario.data)
        np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())


# --------------------------------------------------------------------------
# The strategies built on it
# --------------------------------------------------------------------------

class TestRegimeStrategies:
    @pytest.mark.parametrize("strategy", [RegimeAwareStrategy(), RegimeGate()], ids=lambda s: s.name)
    def test_are_causal(self, strategy: Strategy) -> None:
        data = GBMGenerator(mu=0.06).generate(n_steps=1400, seed=11).data
        report = check_causality(strategy, data)
        assert report.is_causal, str(report)

    def test_hold_cash_through_the_warmup(self, scenario) -> None:
        weights = RegimeAwareStrategy().compute_weights(scenario.data)
        assert (weights.iloc[:756] == 0.0).all().all()

    def test_the_no_trade_band_cuts_turnover(self, scenario) -> None:
        """Following the probability exactly rebalances daily and pays for it."""
        banded = run_backtest(scenario.data, RegimeAwareStrategy(band=0.10), limits=UNLIMITED)
        raw = run_backtest(scenario.data, RegimeAwareStrategy(band=0.0), limits=UNLIMITED)
        assert banded.annual_turnover < raw.annual_turnover

    def test_reject_impossible_parameters(self) -> None:
        with pytest.raises(ValueError, match="band must be"):
            RegimeAwareStrategy(band=1.5)
        with pytest.raises(ValueError, match="floor must be"):
            RegimeAwareStrategy(floor=-0.1)

    def test_reduce_drawdown_relative_to_holding_the_market(self, scenario) -> None:
        """The point of the whole exercise: capital preservation, not return."""
        from sentinel.strategies.baseline import BuyAndHold

        timed = run_backtest(scenario.data, RegimeAwareStrategy(), limits=UNLIMITED)
        held = run_backtest(scenario.data, BuyAndHold(), limits=UNLIMITED)
        assert timed.performance.max_drawdown > held.performance.max_drawdown
