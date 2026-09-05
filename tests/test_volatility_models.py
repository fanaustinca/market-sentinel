"""Tests for volatility forecasting — the one question this data can answer.

Return forecasting failed here for a reason that is about information rather
than effort: a Sharpe ratio over 33 years has a standard error near 0.18, so two
strategies differing by 0.1 cannot be separated at all. Volatility forecasts are
scored on every day, so they can.

Three things are protected. **Causality**, as everywhere. **Alignment**, because
a forecast compared against the previous day's return would look far better than
it is — volatility is persistent, so yesterday's move is already inside the
estimate. And **level calibration**, because two bugs in this module were exactly
that and neither looked like a bug.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.ai.volatility import (
    EWMAVolatility,
    GARCHVolatility,
    HARVolatility,
    RollingVolatility,
)
from sentinel.evaluation.volatility_score import (
    common_sample,
    diebold_mariano,
    score_against_truth,
    score_all,
    score_forecast,
)
from sentinel.sandbox.generators.gbm import GBMGenerator
from sentinel.sandbox.generators.heston import HestonGenerator

ALL = [RollingVolatility(), EWMAVolatility(), HARVolatility(), GARCHVolatility()]
FAST = [RollingVolatility(), EWMAVolatility(), HARVolatility()]


@pytest.fixture(scope="module")
def clustered():
    """Heston: volatility clusters and the true path is known."""
    return HestonGenerator(mu=0.0).generate(n_steps=2520, n_assets=1, seed=17)


@pytest.fixture(scope="module")
def prices(clustered) -> pd.Series:
    return clustered.data.prices.iloc[:, 0]


class TestCausality:
    @pytest.mark.parametrize("forecaster", ALL, ids=lambda f: f.name)
    def test_a_forecast_never_changes_when_the_future_arrives(self, forecaster, prices) -> None:
        """The property every honest forecast must have.

        Row t must be final the moment it is computed. A model that revises it
        when later data arrives used that data, whatever the code appeared to say.
        """
        full = forecaster.forecast(prices)
        for cut in (1200, 1800, 2400):
            partial = forecaster.forecast(prices.iloc[:cut])
            a = full.to_numpy()[:cut]
            b = partial.to_numpy()
            both = np.isfinite(a) & np.isfinite(b)
            assert both.sum() > 50
            np.testing.assert_allclose(a[both], b[both], rtol=1e-9)

    @pytest.mark.parametrize("forecaster", ALL, ids=lambda f: f.name)
    def test_warmup_rows_are_nan_rather_than_guessed(self, forecaster, prices) -> None:
        """Back-filling a forecast imports the future into the past.

        It is the easiest way to break causality in this module and it looks like
        tidying up.
        """
        values = forecaster.forecast(prices)
        assert values.iloc[:20].isna().all()
        assert values.iloc[-1] == values.iloc[-1]  # not NaN at the end


class TestLevelCalibration:
    """Both bugs found in this module were level bias, and neither looked wrong."""

    @pytest.mark.parametrize("forecaster", ALL, ids=lambda f: f.name)
    def test_the_forecast_is_close_to_the_truth_on_average(self, forecaster, clustered) -> None:
        """Within 10% of the volatility that actually generated the returns.

        HAR failed this twice on the way in. Predicting log volatility and
        exponentiating gives the geometric mean, which for a right-skewed
        quantity sits below the arithmetic mean that is wanted — Jensen's
        inequality — and it under-forecast by 9.7%. The textbook smearing fix,
        exp(residual variance / 2), then overshot by 12%, because it assumes the
        residuals are normal in logs and log|z| is sharply left-skewed.

        Both errors were constant multiplicative biases. Nothing looked wrong:
        the forecasts tracked volatility perfectly well and only their level was
        off, so a scoring rule charged the mistake to the model as poor
        forecasting rather than to the arithmetic.
        """
        forecast = forecaster.forecast(clustered.data.prices.iloc[:, 0])
        graded = score_against_truth(forecast, clustered.truth.volatility)
        assert abs(graded["bias"]) < 0.10, f"{forecaster.name} bias {graded['bias']:+.1%}"

    @pytest.mark.parametrize("forecaster", ALL, ids=lambda f: f.name)
    def test_the_forecast_tracks_the_true_volatility(self, forecaster, clustered) -> None:
        forecast = forecaster.forecast(clustered.data.prices.iloc[:, 0])
        graded = score_against_truth(forecast, clustered.truth.volatility)
        assert graded["correlation"] > 0.6

    def test_a_constant_volatility_market_is_forecast_at_its_level(self) -> None:
        """GBM has one volatility forever. Every model must find it."""
        data = GBMGenerator(mu=0.0, sigma=0.20).generate(n_steps=3000, seed=4).data
        prices = data.prices.iloc[:, 0]
        for forecaster in FAST:
            values = forecaster.forecast(prices).dropna()
            assert values.mean() == pytest.approx(0.20, rel=0.15), forecaster.name


class TestAlignment:
    def test_a_forecast_is_scored_against_the_next_day_not_the_last(self, prices) -> None:
        """The error that would silently flatter every model in this file.

        Volatility is persistent, so yesterday's squared return is already inside
        any trailing estimate. Scoring against it instead of tomorrow's would
        make every forecaster look far better than it is, and the numbers would
        remain entirely plausible.
        """
        forecast = RollingVolatility().forecast(prices)
        honest = score_forecast(forecast, prices)

        # Deliberately misaligned by one day, in the flattering direction.
        cheating = score_forecast(forecast.shift(-1).rename("cheat"), prices)
        assert cheating.qlike < honest.qlike

    def test_scoring_needs_enough_observations(self, prices) -> None:
        with pytest.raises(ValueError, match="too few"):
            score_forecast(RollingVolatility().forecast(prices.iloc[:40]), prices.iloc[:40])


class TestRankingModels:
    def test_all_three_alternatives_beat_the_rolling_window(self, prices) -> None:
        """The result that motivated changing the default.

        Measured on eight real indices with Diebold-Mariano p < 0.05 in all
        eight; here it is checked once on synthetic data so the suite does not
        depend on the network.
        """
        built = {f.name: f.forecast(prices).rename(f.name) for f in ALL}
        scores, shared = score_all(built, prices)
        incumbent = scores["rolling_21d"].qlike
        for name, score in scores.items():
            if name == "rolling_21d":
                continue
            assert score.qlike < incumbent, f"{name} did not beat the rolling window"

    def test_models_are_ranked_on_the_days_they_all_share(self, prices) -> None:
        """Otherwise the comparison is confounded by warmup.

        A rolling window needs 21 days and a walk-forward regression needs two
        years. Scoring each over its own full range compares them on different
        periods, and on SPY that difference covered a bear market and reversed
        the ranking.
        """
        built = {f.name: f.forecast(prices).rename(f.name) for f in ALL}
        shared = common_sample(built)
        for series in built.values():
            assert series.reindex(shared).notna().all()

    def test_diebold_mariano_finds_a_real_difference(self, prices) -> None:
        """The power argument, made concrete.

        On the same data where two strategies' Sharpe ratios cannot be told
        apart, two volatility forecasters are separated decisively.
        """
        ewma = EWMAVolatility().forecast(prices).rename("ewma")
        rolling = RollingVolatility().forecast(prices).rename("rolling")
        result = diebold_mariano(ewma, rolling, prices)
        assert result["better"] == "ewma"
        assert result["p_value"] < 0.05

    def test_diebold_mariano_finds_no_difference_between_a_model_and_itself(self, prices) -> None:
        """The calibration check. Identical forecasts must be a dead heat."""
        forecast = RollingVolatility().forecast(prices).rename("a")
        result = diebold_mariano(forecast, forecast.rename("b"), prices)
        assert result["mean_difference"] == pytest.approx(0.0, abs=1e-12)

    def test_an_unknown_loss_is_refused(self, prices) -> None:
        forecast = RollingVolatility().forecast(prices).rename("a")
        with pytest.raises(ValueError, match="unknown loss"):
            diebold_mariano(forecast, forecast, prices, loss="rmse")


class TestGuards:
    @pytest.mark.parametrize(
        "call,message",
        [
            (lambda: RollingVolatility(window=1), "window must be at least 2"),
            (lambda: EWMAVolatility(lambda_=1.0), "lambda_ must be strictly"),
            (lambda: EWMAVolatility(lambda_=0.0), "lambda_ must be strictly"),
            (lambda: HARVolatility(min_train=50), "min_train below 100"),
            (lambda: HARVolatility(retrain_every=0), "retrain_every must be"),
            (lambda: GARCHVolatility(min_train=10), "min_train below 100"),
        ],
    )
    def test_impossible_parameters_are_refused(self, call, message) -> None:
        with pytest.raises(ValueError, match=message):
            call()

    def test_ewma_reports_its_half_life(self) -> None:
        """0.94 is the published RiskMetrics daily decay, not something fitted
        here — which is what keeps it out of sample with respect to this data."""
        assert EWMAVolatility(lambda_=0.94).half_life_days == pytest.approx(11.2, rel=0.02)

    def test_garch_stays_stationary(self) -> None:
        """alpha + beta must be below 1, or the model has no long-run variance
        and its forecasts diverge."""
        returns = np.log(
            GBMGenerator(mu=0.0, sigma=0.18).generate(n_steps=1500, seed=8).data.prices.iloc[:, 0]
        ).diff().dropna().to_numpy()
        omega, alpha, beta = GARCHVolatility()._fit(returns)
        assert omega > 0
        assert 0 <= alpha and 0 <= beta
        assert alpha + beta < 1.0
