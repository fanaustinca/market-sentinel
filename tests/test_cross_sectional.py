"""Tests for cross-sectional ranking strategies.

The two implementation details that carry the literature -- skipping the last
month, and rebalancing monthly -- are the ones most likely to be silently lost in
a refactor, so both are asserted directly rather than left to the docstring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.evaluation.causality import check_causality
from sentinel.sandbox.market import MarketData
from sentinel.strategies.cross_sectional import CrossSectionalMomentum, LowVolatility


def ramps(slopes: dict[str, float], n: int = 700, noise: float = 0.0) -> MarketData:
    """Assets rising at fixed, different rates. The ranking is then known."""
    rng = np.random.default_rng(0)
    index = pd.bdate_range("2020-01-01", periods=n)
    frame = {}
    for name, slope in slopes.items():
        steps = np.full(n, slope) + rng.normal(0.0, noise, n)
        frame[name] = 100.0 * np.exp(np.cumsum(steps))
    return MarketData(prices=pd.DataFrame(frame, index=index))


class TestCrossSectionalMomentum:
    def test_it_holds_the_strongest_performers(self):
        data = ramps({"best": 0.0008, "mid": 0.0004, "worst": -0.0002})
        weights = CrossSectionalMomentum(n_hold=1).compute_weights(data).iloc[-1]
        assert weights["best"] == pytest.approx(1.0)
        assert weights["worst"] == pytest.approx(0.0)

    def test_it_splits_evenly_among_the_chosen(self):
        data = ramps({"a": 0.0009, "b": 0.0007, "c": 0.0002, "d": -0.0003})
        weights = CrossSectionalMomentum(n_hold=2).compute_weights(data).iloc[-1]
        assert sorted(weights.to_numpy(), reverse=True)[:2] == pytest.approx([0.5, 0.5])
        assert weights.sum() == pytest.approx(1.0)

    def test_the_skip_window_is_actually_skipped(self):
        """A recent crash inside the skip window must not change the ranking."""
        data = ramps({"a": 0.0008, "b": 0.0002})
        crashed = data.prices.copy()
        crashed.iloc[-10:, crashed.columns.get_loc("a")] *= 0.65

        strategy = CrossSectionalMomentum(n_hold=1, skip=21)
        before = strategy.compute_weights(data).iloc[-1]
        after = strategy.compute_weights(MarketData(prices=crashed)).iloc[-1]
        assert before["a"] == pytest.approx(1.0)
        assert after["a"] == pytest.approx(1.0)

    def test_no_skip_lets_the_recent_crash_flip_the_ranking(self):
        """The mirror of the test above: without skip, the crash is visible."""
        data = ramps({"a": 0.0006, "b": 0.0005})
        crashed = data.prices.copy()
        crashed.iloc[-10:, crashed.columns.get_loc("a")] *= 0.5
        strategy = CrossSectionalMomentum(n_hold=1, skip=1)
        assert strategy.compute_weights(MarketData(prices=crashed)).iloc[-1]["b"] == pytest.approx(1.0)

    def test_positive_filter_goes_to_cash_when_everything_falls(self):
        data = ramps({"a": -0.0003, "b": -0.0006})
        weights = CrossSectionalMomentum(n_hold=2, long_only_positive=True).compute_weights(data)
        assert weights.iloc[-1].sum() == pytest.approx(0.0)

    def test_without_the_filter_it_stays_invested_in_a_falling_market(self):
        """The defining difference from time-series momentum, asserted."""
        data = ramps({"a": -0.0003, "b": -0.0006})
        weights = CrossSectionalMomentum(n_hold=1).compute_weights(data)
        assert weights.iloc[-1].sum() == pytest.approx(1.0)

    def test_rebalance_schedule_is_respected(self):
        data = ramps({"a": 0.0006, "b": 0.0005, "c": 0.0001}, noise=0.02)
        weights = CrossSectionalMomentum(n_hold=1, rebalance_days=21).compute_weights(data)
        live = weights.loc[weights.sum(axis=1) > 0]
        changed = (live.diff().abs().sum(axis=1) > 1e-12).sum()
        assert changed <= len(live) / 15

    def test_never_leveraged_or_short(self):
        data = ramps({"a": 0.0006, "b": 0.0002, "c": -0.0004}, noise=0.01)
        weights = CrossSectionalMomentum(n_hold=2).compute_weights(data)
        assert (weights.to_numpy() >= -1e-12).all()
        assert weights.sum(axis=1).max() <= 1.0 + 1e-9

    def test_is_causal(self):
        data = ramps({"a": 0.0006, "b": 0.0002, "c": -0.0004}, noise=0.01)
        assert check_causality(CrossSectionalMomentum(n_hold=2), data).is_causal

    def test_skip_not_smaller_than_lookback(self):
        with pytest.raises(ValueError, match="exceed skip"):
            CrossSectionalMomentum(lookback=21, skip=21)


class TestLowVolatility:
    def test_it_holds_the_calmest(self):
        rng = np.random.default_rng(1)
        n = 700
        index = pd.bdate_range("2020-01-01", periods=n)
        frame = pd.DataFrame({
            "calm": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.003, n))),
            "wild": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.030, n))),
        }, index=index)
        weights = LowVolatility(n_hold=1).compute_weights(MarketData(prices=frame)).iloc[-1]
        assert weights["calm"] == pytest.approx(1.0)

    def test_is_causal(self):
        data = ramps({"a": 0.0006, "b": 0.0002, "c": -0.0004}, noise=0.01)
        assert check_causality(LowVolatility(n_hold=2), data).is_causal
