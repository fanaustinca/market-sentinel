"""Tests for the bootstrap generator -- the bridge from sandbox to real markets.

Its job is to produce a market that is realistic in every respect except that
there is nothing left to predict, so a real-data backtest can be judged against a
floor built from the right return distribution rather than a Gaussian one.

Two properties carry the weight. It must **keep** the marginal distribution --
the fat tails that make real markets what they are, and the cross-sectional
correlation that makes diversification fail exactly when it is needed. And it must
**destroy** the ordering, or the floor it produces is inflated by structure it was
supposed to remove, which would quietly excuse a broken strategy.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.evaluation.null_test import run_null_test
from sentinel.sandbox.generators.bootstrap import BootstrapGenerator
from sentinel.sandbox.generators.gbm import GBMGenerator
from sentinel.stats.randomwalk import ljung_box
from sentinel.strategies.baseline import BuyAndHold


@pytest.fixture(scope="module")
def fat_tailed_returns() -> np.ndarray:
    """A history with real-market character: fat tails and volatility clustering."""
    rng = np.random.default_rng(11)
    volatility = 0.01 * np.exp(rng.normal(0, 0.4, size=4000).cumsum() * 0.02)
    return rng.standard_t(df=4, size=4000) * volatility


class TestPreservesTheDistribution:
    def test_keeps_the_fat_tails(self, fat_tailed_returns) -> None:
        """A Gaussian floor is the wrong floor for a market with kurtosis of 11."""
        from scipy import stats

        generator = BootstrapGenerator(fat_tailed_returns, demean=True)
        resampled = generator.generate(n_steps=4000, seed=3).data.log_returns().to_numpy().ravel()

        source_kurtosis = stats.kurtosis(fat_tailed_returns)
        assert source_kurtosis > 2.0, "the fixture must actually be fat-tailed"
        assert stats.kurtosis(resampled) == pytest.approx(source_kurtosis, rel=0.5)

    def test_keeps_the_volatility(self, fat_tailed_returns) -> None:
        generator = BootstrapGenerator(fat_tailed_returns)
        resampled = generator.generate(n_steps=4000, seed=3).data.log_returns().to_numpy()
        assert resampled.std() == pytest.approx(fat_tailed_returns.std(), rel=0.15)

    def test_every_drawn_return_really_happened(self, fat_tailed_returns) -> None:
        """Resampling, not simulating: each value is an observation from history."""
        generator = BootstrapGenerator(fat_tailed_returns, demean=False)
        resampled = generator.generate(n_steps=500, seed=5).data.log_returns().to_numpy().ravel()
        assert np.isin(np.round(resampled, 12), np.round(fat_tailed_returns, 12)).all()

    def test_keeps_the_correlation_between_assets(self) -> None:
        """Rows are drawn whole, so a crash still hits everything at once.

        Resampling each asset independently would destroy this and make
        diversification look far more reliable than it is -- correlations spiking
        toward one during a crash is the single nastiest behaviour in real
        markets, and a simulator that misses it flatters every portfolio.
        """
        rng = np.random.default_rng(2)
        base = rng.normal(0, 0.01, size=(3000, 1))
        source = np.hstack([base, base * 0.8 + rng.normal(0, 0.004, size=(3000, 1))])
        expected = np.corrcoef(source.T)[0, 1]

        resampled = (
            BootstrapGenerator(source)
            .generate(n_steps=3000, n_assets=2, seed=4)
            .data.log_returns()
            .to_numpy()
        )
        assert np.corrcoef(resampled.T)[0, 1] == pytest.approx(expected, abs=0.06)


class TestDestroysTheOrdering:
    def test_single_day_sampling_leaves_no_autocorrelation(self) -> None:
        """The property that makes it a valid null market."""
        rng = np.random.default_rng(7)
        # A source with strong momentum, so the test would fail if ordering survived.
        source = np.zeros(3000)
        for i in range(1, 3000):
            source[i] = 0.5 * source[i - 1] + rng.normal(0, 0.01)
        assert ljung_box(source, lags=10).p_value < 0.01, "the fixture must be autocorrelated"

        resampled = (
            BootstrapGenerator(source).generate(n_steps=3000, seed=9).data.log_returns().to_numpy().ravel()
        )
        assert ljung_box(resampled, lags=10).p_value > 0.01

    def test_demeaning_removes_the_drift(self, fat_tailed_returns) -> None:
        """Null markets have no drift, or holding the asset earns from beta.

        Judged against the standard error of the resample rather than against
        zero. Demeaning removes the drift in *expectation*; any single 4000-day
        draw still lands a standard error or so away from zero, and a test that
        demanded otherwise would be asserting that random sampling is not random.
        """
        source = fat_tailed_returns + 0.001
        generator = BootstrapGenerator(source, demean=True)
        assert generator.source.mean() == pytest.approx(0.0, abs=1e-15)

        means = [
            generator.generate(n_steps=4000, seed=seed).data.log_returns().to_numpy().mean()
            for seed in range(8)
        ]
        standard_error = source.std() / np.sqrt(3999)
        assert abs(np.mean(means)) < 1.5 * standard_error / np.sqrt(8)
        assert abs(np.mean(means)) < abs(source.mean()) / 2


class TestBlocksAreNotNullMarkets:
    """A blocked bootstrap keeps serial structure, and must say so.

    Blocks are more realistic -- volatility clustering survives inside them --
    and that is precisely why they cannot measure a noise floor. The floor would
    be inflated by the structure the blocks preserved, and an inflated floor
    excuses a broken strategy instead of catching it.
    """

    def test_a_blocked_bootstrap_declares_a_signal(self, fat_tailed_returns) -> None:
        assert BootstrapGenerator(fat_tailed_returns, block_size=21).has_exploitable_signal
        assert not BootstrapGenerator(fat_tailed_returns, block_size=1).has_exploitable_signal

    def test_the_null_test_refuses_a_blocked_bootstrap(self, fat_tailed_returns) -> None:
        generator = BootstrapGenerator(fat_tailed_returns, block_size=21)
        with pytest.raises(ValueError, match="declares an exploitable signal"):
            run_null_test(BuyAndHold(), generator, n_markets=2, n_steps=300, workers=1)

    def test_the_null_test_accepts_a_single_day_bootstrap(self, fat_tailed_returns) -> None:
        result = run_null_test(
            BuyAndHold(),
            BootstrapGenerator(fat_tailed_returns),
            n_markets=30,
            n_steps=500,
            workers=1,
        )
        assert result.passed()

    def test_blocks_wrap_around_the_end_of_history(self) -> None:
        """Without wrapping the last few days are systematically under-sampled.

        That silently reweights the history -- the most recent observations,
        usually the ones a user cares most about, would appear less often than
        every other day.
        """
        source = np.arange(200, dtype=float) / 10_000
        generator = BootstrapGenerator(source, block_size=10, demean=False)
        drawn = generator.generate(n_steps=20_001, seed=1).data.log_returns().to_numpy().ravel()
        counts = np.array([np.isclose(drawn, value).sum() for value in source])
        # Every observation should appear at a comparable rate.
        assert counts.min() > 0
        assert counts.max() / counts.min() < 2.0


class TestGuards:
    def test_refuses_a_history_too_short_to_resample(self) -> None:
        with pytest.raises(ValueError, match="only 50 historical returns"):
            BootstrapGenerator(np.zeros(50))

    def test_refuses_non_finite_history(self) -> None:
        source = np.random.default_rng(0).normal(size=200)
        source[3] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            BootstrapGenerator(source)

    def test_refuses_more_assets_than_the_history_has(self, fat_tailed_returns) -> None:
        with pytest.raises(ValueError, match="asked for 3 assets"):
            BootstrapGenerator(fat_tailed_returns).generate(n_steps=300, n_assets=3, seed=1)

    def test_refuses_a_block_size_below_one(self, fat_tailed_returns) -> None:
        with pytest.raises(ValueError, match="block_size must be"):
            BootstrapGenerator(fat_tailed_returns, block_size=0)

    def test_from_market_reads_a_market_s_own_history(self) -> None:
        data = GBMGenerator(mu=0.05).generate(n_steps=1000, seed=2).data
        generator = BootstrapGenerator.from_market(data)
        assert generator.source.shape == (999, 1)

    def test_the_answer_key_records_what_it_resampled(self, fat_tailed_returns) -> None:
        truth = BootstrapGenerator(fat_tailed_returns, block_size=5).generate(
            n_steps=500, seed=1
        ).truth
        assert truth.params["block_size"] == 5
        assert truth.params["n_source_observations"] == len(fat_tailed_returns)
        assert truth.has_exploitable_signal is True


class TestTickersTravelWithTheReturns:
    """A bootstrap of SPY and IEF must still be called SPY and IEF.

    This was a real bug. Synthetic generators name their columns SYN0, SYN1 --
    correct for them, and part of keeping the model unable to tell which rung it
    is on. The bootstrap inherited that, so a multi-asset strategy asked to trade
    `SPY` could not run on a bootstrap of SPY, which meant it silently could not
    be given a noise floor.

    It surfaced as a crash, which is the good case. A strategy that addressed
    columns by position instead of by name would have computed its floor against
    whichever assets happened to line up, and the number would have looked
    entirely reasonable.
    """

    def test_a_bootstrap_of_a_named_market_keeps_the_names(self) -> None:
        from sentinel.sandbox.market import MarketData
        import pandas as pd

        index = pd.bdate_range("2000-01-03", periods=600, name="date")
        prices = pd.DataFrame(
            {"SPY": np.linspace(100, 180, 600), "IEF": np.linspace(100, 130, 600)}, index=index
        )
        data = MarketData(prices=prices, name="pair")

        scenario = BootstrapGenerator.from_market(data).generate(
            n_steps=400, n_assets=2, seed=1
        )
        assert scenario.data.tickers == ["SPY", "IEF"]

    def test_synthetic_generators_keep_neutral_names(self) -> None:
        """They must stay anonymous; a model must not learn what it is trading."""
        scenario = GBMGenerator().generate(n_steps=300, n_assets=2, seed=1)
        assert scenario.data.tickers == ["SYN0", "SYN1"]

    def test_a_ticker_count_mismatch_is_refused(self, fat_tailed_returns) -> None:
        with pytest.raises(ValueError, match="got 2 tickers for 1 return columns"):
            BootstrapGenerator(fat_tailed_returns, tickers=["SPY", "IEF"])
