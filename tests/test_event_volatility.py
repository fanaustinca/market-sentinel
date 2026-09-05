"""Tests for the earnings-aware volatility forecaster.

The claim this forecaster makes is narrow and testable: on the ~1% of days that
carry a scheduled release it should predict a larger move, on every other day it
should be its base model exactly, and at no point should a later observation
change an earlier forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.ai.volatility.events import MAX_MULTIPLIER, MIN_HISTORY, EarningsAwareVolatility
from sentinel.ai.volatility.forecasters import EWMAVolatility


@pytest.fixture
def market() -> tuple[pd.Series, pd.Series]:
    """A calm series with a large move every 63rd day, and flags marking them.

    The flag sits on the row *before* the jump, matching `announcement_flag`:
    row t forecasts the return from t to t+1, so a jump landing in that window is
    the thing row t is supposed to have predicted.
    """
    rng = np.random.default_rng(11)
    n = 1500
    returns = rng.normal(0.0, 0.01, n)
    flags = np.zeros(n)
    for day in range(63, n - 1, 63):
        returns[day + 1] = rng.normal(0.0, 0.06)
        flags[day] = 1.0

    index = pd.bdate_range("2010-01-01", periods=n)
    prices = pd.Series(100.0 * np.exp(np.cumsum(returns)), index=index, name="A")
    return prices, pd.Series(flags, index=index)


class TestEarningsAwareVolatility:
    def test_it_predicts_larger_moves_on_event_days(self, market):
        prices, flags = market
        base = EWMAVolatility()
        adjusted = EarningsAwareVolatility(base, flags).forecast(prices)
        plain = base.forecast(prices)

        mask = (flags.to_numpy() > 0) & np.isfinite(plain.to_numpy())
        warm = np.flatnonzero(mask)[MIN_HISTORY:]
        assert (adjusted.to_numpy()[warm] > plain.to_numpy()[warm]).all()

    def test_ordinary_days_are_left_exactly_alone(self, market):
        """The claim is confined to event days; anything else is scope creep."""
        prices, flags = market
        base = EWMAVolatility()
        adjusted = EarningsAwareVolatility(base, flags).forecast(prices)
        plain = base.forecast(prices)

        quiet = flags.to_numpy() == 0
        np.testing.assert_allclose(
            adjusted.to_numpy()[quiet], plain.to_numpy()[quiet], rtol=0, atol=0
        )

    def test_no_multiplier_before_enough_history(self, market):
        prices, flags = market
        base = EWMAVolatility()
        adjusted = EarningsAwareVolatility(base, flags).forecast(prices)
        plain = base.forecast(prices)

        early = np.flatnonzero(flags.to_numpy() > 0)[:MIN_HISTORY]
        np.testing.assert_allclose(adjusted.to_numpy()[early], plain.to_numpy()[early])

    def test_truncating_the_future_does_not_change_the_past(self, market):
        """The truncation test, applied to a forecaster instead of a strategy.

        This is the check that would have caught the bug that made an earlier
        version of this module read its own final estimate on every row.
        """
        prices, flags = market
        for cut in (900, 1100, 1300):
            full = EarningsAwareVolatility(EWMAVolatility(), flags).forecast(prices)
            short = EarningsAwareVolatility(
                EWMAVolatility(), flags.iloc[:cut]
            ).forecast(prices.iloc[:cut])
            np.testing.assert_allclose(
                full.to_numpy()[:cut], short.to_numpy(), rtol=1e-10, atol=1e-12
            )

    def test_a_single_shock_cannot_blow_up_the_multiplier(self):
        """One 40% day in a thin sample must not produce a wild forecast."""
        rng = np.random.default_rng(3)
        n = 900
        returns = rng.normal(0.0, 0.01, n)
        flags = np.zeros(n)
        for day in range(63, n - 1, 63):
            flags[day] = 1.0
        returns[63 * 9 + 1] = 0.40

        index = pd.bdate_range("2010-01-01", periods=n)
        prices = pd.Series(100.0 * np.exp(np.cumsum(returns)), index=index)
        flags = pd.Series(flags, index=index)

        base = EWMAVolatility()
        ratio = (
            EarningsAwareVolatility(base, flags).forecast(prices) / base.forecast(prices)
        ).dropna()
        assert ratio.max() <= MAX_MULTIPLIER + 1e-9

    def test_it_beats_its_base_model_on_qlike(self, market):
        """The whole point, stated as a test rather than left to an experiment."""
        from sentinel.evaluation.volatility_score import score_forecast

        prices, flags = market
        base = EWMAVolatility()
        plain = base.forecast(prices)
        plain.name = "ewma"
        adjusted = EarningsAwareVolatility(base, flags).forecast(prices)
        assert score_forecast(adjusted, prices).qlike < score_forecast(plain, prices).qlike
