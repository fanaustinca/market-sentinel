"""Tests for the long-short book.

Dollar neutrality is the property the whole design rests on -- if the two sides
are not equal, the strategy is a market bet wearing a hedge and its results mean
something different. That is asserted directly, along with the borrow cost
actually being charged, since a short book that forgets its financing looks far
better than it is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.evaluation.causality import check_causality
from sentinel.sandbox.market import MarketData
from sentinel.strategies.market_neutral import (
    MarketNeutralRanking,
    rank_normalise,
    momentum_signal,
    reversal_signal,
)


@pytest.fixture
def panel() -> MarketData:
    """Ten assets with steadily different trends, so the ranking is knowable."""
    rng = np.random.default_rng(2)
    n = 900
    slopes = np.linspace(-0.0006, 0.0006, 10)
    steps = slopes + rng.normal(0.0, 0.004, (n, 10))
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(steps, axis=0)),
        index=pd.bdate_range("2018-01-01", periods=n),
        columns=[f"S{i}" for i in range(10)],
    )
    return MarketData(prices=prices)


class TestNeutrality:
    def test_the_book_is_dollar_neutral(self, panel):
        """The property everything else depends on."""
        weights = MarketNeutralRanking(quantile=0.2).compute_weights(panel)
        live = weights.loc[weights.abs().sum(axis=1) > 0]
        assert live.sum(axis=1).abs().max() < 1e-12

    def test_gross_exposure_is_respected(self, panel):
        weights = MarketNeutralRanking(quantile=0.2, gross=1.0).compute_weights(panel)
        assert weights.abs().sum(axis=1).max() <= 1.0 + 1e-9

    def test_both_sides_are_populated(self, panel):
        weights = MarketNeutralRanking(quantile=0.2).compute_weights(panel).iloc[-1]
        assert (weights > 0).sum() == 2
        assert (weights < 0).sum() == 2

    def test_quantile_controls_breadth(self, panel):
        narrow = MarketNeutralRanking(quantile=0.1).compute_weights(panel).iloc[-1]
        wide = MarketNeutralRanking(quantile=0.3).compute_weights(panel).iloc[-1]
        assert (wide != 0).sum() > (narrow != 0).sum()


class TestSignals:
    def test_momentum_goes_long_the_strongest(self, panel):
        """Assert the mechanism, not which ticker wins.

        An earlier version named the asset with the steepest trend and expected
        it long. Noise displaced it -- the fixture's second-steepest name had the
        stronger trailing return over that particular window, and the strategy
        correctly picked it. Naming the expected winner tested the fixture's
        random seed rather than the ranking.
        """
        signal = momentum_signal(panel.prices).iloc[-1]
        weights = MarketNeutralRanking(signals=("momentum",), quantile=0.1).compute_weights(panel)
        held = weights.iloc[-1]

        long_leg = held[held > 0].index
        short_leg = held[held < 0].index
        assert len(long_leg) == 1 and len(short_leg) == 1
        assert signal[long_leg[0]] == signal.max()
        assert signal[short_leg[0]] == signal.min()

    def test_reversal_points_the_other_way_from_momentum(self):
        """The two must disagree, or combining them buys nothing."""
        rng = np.random.default_rng(5)
        n = 400
        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, 6)), axis=0)),
            index=pd.bdate_range("2020-01-01", periods=n),
            columns=list("ABCDEF"),
        )
        recent = momentum_signal(prices, lookback=42, skip=1).iloc[-1]
        flipped = reversal_signal(prices, lookback=42).iloc[-1]
        assert np.corrcoef(recent.dropna(), flipped.dropna())[0, 1] < -0.9

    def test_rank_normalise_spans_minus_one_to_one(self, panel):
        ranked = rank_normalise(momentum_signal(panel.prices)).dropna()
        assert ranked.to_numpy().min() >= -1.0
        assert ranked.to_numpy().max() <= 1.0
        np.testing.assert_allclose(ranked.mean(axis=1).to_numpy(), 0.0, atol=1e-12)

    def test_unknown_signal_is_refused(self):
        with pytest.raises(ValueError, match="unknown signals"):
            MarketNeutralRanking(signals=("crystal_ball",))


class TestBorrowCost:
    def test_borrow_is_charged_on_the_short_side_only(self, panel):
        strategy = MarketNeutralRanking(quantile=0.2, borrow_cost=0.05)
        weights = strategy.compute_weights(panel)
        drag = strategy.borrow_drag(weights)
        live = weights.abs().sum(axis=1) > 0
        expected = 0.5 * 0.05 / 252.0  # half the gross is short
        np.testing.assert_allclose(drag[live].to_numpy(), expected, rtol=1e-9)

    def test_zero_borrow_costs_nothing(self, panel):
        strategy = MarketNeutralRanking(borrow_cost=0.0)
        assert strategy.borrow_drag(strategy.compute_weights(panel)).abs().max() == 0.0

    def test_borrow_drag_is_a_pure_function_of_weights(self, panel):
        """It must not depend on hidden state left over from a previous call."""
        strategy = MarketNeutralRanking(quantile=0.2)
        weights = strategy.compute_weights(panel)
        first = strategy.borrow_drag(weights)
        strategy.compute_weights(MarketData(prices=panel.prices.iloc[:500]))
        np.testing.assert_allclose(first.to_numpy(), strategy.borrow_drag(weights).to_numpy())


class TestContract:
    def test_is_causal(self, panel):
        assert check_causality(MarketNeutralRanking(quantile=0.2), panel).is_causal

    def test_rebalance_schedule_is_respected(self, panel):
        weights = MarketNeutralRanking(quantile=0.2, rebalance_days=21).compute_weights(panel)
        live = weights.loc[weights.abs().sum(axis=1) > 0]
        changed = (live.diff().abs().sum(axis=1) > 1e-12).sum()
        assert changed <= len(live) / 15

    def test_bad_quantile_is_refused(self):
        with pytest.raises(ValueError, match="quantile"):
            MarketNeutralRanking(quantile=0.7)
