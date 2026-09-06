"""Tests for the RL trader.

The most important test here is the recovery test: an agent that cannot find a
signal that is definitely present tells you nothing when it fails on real data.
Without it, a null result is indistinguishable from a bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.ai.rl import ACTIONS, PolicyGradientTrader


@pytest.fixture
def planted():
    """Feature 0 genuinely predicts the return. The agent should find it."""
    rng = np.random.default_rng(0)
    n = 30_000
    features = rng.normal(0, 1, (n, 4))
    returns = 0.01 * features[:, 0] + rng.normal(0, 0.005, n)
    return features, returns


class TestItCanLearn:
    def test_it_recovers_a_planted_signal(self, planted):
        """The control that makes every null result in this file interpretable."""
        features, returns = planted
        agent = PolicyGradientTrader(n_features=4, seed=0)
        agent.fit(features, returns, epochs=15)
        positions = agent.act(features[:3000])
        assert np.corrcoef(positions, features[:3000, 0])[0, 1] > 0.3

    def test_training_reward_improves_on_a_learnable_problem(self, planted):
        features, returns = planted
        agent = PolicyGradientTrader(n_features=4, seed=0)
        log = agent.fit(features, returns, epochs=15)
        assert log.rewards[-1] > log.rewards[0]

    def test_it_does_not_invent_a_signal_that_is_absent(self):
        """Pure noise must not produce a confident directional policy."""
        rng = np.random.default_rng(3)
        n = 30_000
        features = rng.normal(0, 1, (n, 4))
        returns = rng.normal(0, 0.01, n)
        agent = PolicyGradientTrader(n_features=4, seed=0)
        agent.fit(features, returns, epochs=15)
        positions = agent.act(features[:3000])
        assert abs(np.corrcoef(positions, features[:3000, 0])[0, 1]) < 0.3


class TestContract:
    def test_positions_are_always_legal(self, planted):
        """Long-only and unlevered, the standing policy for every strategy here."""
        features, returns = planted
        agent = PolicyGradientTrader(n_features=4, seed=0)
        agent.fit(features, returns, epochs=3)
        positions = agent.act(features[:500])
        assert set(np.unique(positions)).issubset(set(ACTIONS))
        assert positions.min() >= 0.0
        assert positions.max() <= 1.0

    def test_act_returns_one_position_per_row(self, planted):
        features, returns = planted
        agent = PolicyGradientTrader(n_features=4, seed=0)
        agent.fit(features, returns, epochs=2)
        assert len(agent.act(features[:250])) == 250

    def test_same_seed_reproduces_training(self, planted):
        features, returns = planted
        first = PolicyGradientTrader(n_features=4, seed=7).fit(features, returns, epochs=3)
        second = PolicyGradientTrader(n_features=4, seed=7).fit(features, returns, epochs=3)
        np.testing.assert_allclose(first.rewards, second.rewards, rtol=1e-6)

    def test_turnover_cost_discourages_churn(self, planted):
        """A punitive cost must reduce trading, or the cost term is not wired in."""
        features, returns = planted
        cheap = PolicyGradientTrader(n_features=4, seed=0, cost_bps=0.0)
        cheap.fit(features, returns, epochs=10)
        dear = PolicyGradientTrader(n_features=4, seed=0, cost_bps=500.0)
        dear.fit(features, returns, epochs=10)

        def churn(agent):
            return float(np.abs(np.diff(agent.act(features[:2000]))).mean())

        assert churn(dear) <= churn(cheap) + 1e-9

    def test_the_log_records_every_epoch(self, planted):
        features, returns = planted
        log = PolicyGradientTrader(n_features=4, seed=0).fit(features, returns, epochs=6)
        assert len(log.rewards) == len(log.entropies) == len(log.mean_positions) == 6
