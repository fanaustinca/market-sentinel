"""Tests for the resampler that decides whether a portfolio result is real.

The instrument has to preserve the two things the strategies under test react to
-- cross-sectional correlation and volatility clustering -- or it silently rigs
the comparison against the strategies that use them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.evaluation.panel_bootstrap import (
    PanelComparison,
    compare_on_panel,
    resample_panel,
)
from sentinel.strategies.baseline import BuyAndHold
from sentinel.strategies.volatility import VolatilityTarget


@pytest.fixture
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 1200
    a = rng.normal(0.0, 0.015, n)
    b = 0.85 * a + rng.normal(0.0, 0.008, n)
    c = rng.normal(0.0, 0.005, n)
    returns = np.column_stack([a, b, c])
    return pd.DataFrame(
        100.0 * np.exp(np.cumsum(returns, axis=0)),
        index=pd.bdate_range("2015-01-01", periods=n),
        columns=["A", "B", "C"],
    )


class TestResamplePanel:
    def test_shape_and_index_are_preserved(self, panel):
        got = resample_panel(panel, np.random.default_rng(0))
        assert got.shape == panel.shape
        assert got.index.equals(panel.index)

    def test_cross_sectional_correlation_is_preserved(self, panel):
        """Resampling assets independently would destroy the input under test."""
        original = np.log(panel).diff().dropna().corr().to_numpy()
        drawn = np.log(resample_panel(panel, np.random.default_rng(1))).diff().dropna()
        resampled = drawn.corr().to_numpy()
        triangle = np.triu_indices(3, 1)
        np.testing.assert_allclose(
            original[triangle], resampled[triangle], atol=0.10
        )

    def test_volatilities_are_approximately_preserved(self, panel):
        original = np.log(panel).diff().dropna().std().to_numpy()
        drawn = np.log(resample_panel(panel, np.random.default_rng(2))).diff().dropna()
        np.testing.assert_allclose(original, drawn.std().to_numpy(), rtol=0.35)

    def test_different_seeds_give_different_paths(self, panel):
        first = resample_panel(panel, np.random.default_rng(0))
        second = resample_panel(panel, np.random.default_rng(1))
        assert not np.allclose(first.to_numpy(), second.to_numpy())

    def test_the_same_seed_reproduces_a_path(self, panel):
        first = resample_panel(panel, np.random.default_rng(5))
        second = resample_panel(panel, np.random.default_rng(5))
        np.testing.assert_allclose(first.to_numpy(), second.to_numpy())

    def test_positive_prices(self, panel):
        got = resample_panel(panel, np.random.default_rng(9))
        assert (got.to_numpy() > 0).all()


class TestComparison:
    def test_every_strategy_sees_the_same_paths(self, panel):
        """Pairing is what makes a 0.05 difference detectable; it must be real."""
        strategies = {"hold": BuyAndHold(), "hold_again": BuyAndHold()}
        result = compare_on_panel(strategies, panel, n_paths=6, seed=0)
        np.testing.assert_allclose(
            result.sharpes["hold"].to_numpy(), result.sharpes["hold_again"].to_numpy()
        )

    def test_summary_is_ordered_best_first(self, panel):
        result = compare_on_panel(
            {"hold": BuyAndHold(), "target": VolatilityTarget()}, panel, n_paths=6
        )
        means = result.summary()["mean"].to_numpy()
        assert (np.diff(means) <= 0).all()

    def test_against_excludes_the_benchmark_itself(self, panel):
        result = compare_on_panel(
            {"hold": BuyAndHold(), "target": VolatilityTarget()}, panel, n_paths=6
        )
        assert "hold" not in set(result.against("hold")["strategy"])

    def test_a_strategy_has_no_edge_over_itself(self):
        frame = pd.DataFrame({"a": [0.5, 0.7, 0.9], "b": [0.5, 0.7, 0.9]})
        row = PanelComparison(sharpes=frame).against("a").iloc[0]
        assert row["mean_edge"] == pytest.approx(0.0)
