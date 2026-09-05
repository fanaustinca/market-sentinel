"""Tests for the multi-horizon ensemble and the trend x volatility composite.

Both were registered in DECISIONS.md before they were run, with predictions
attached, because the temptation at this point in the project is specific: there
is now a precise p-value to tune against, and `HANDOFF.md` names tuning it as the
one dishonest continuation.

Neither change fits a parameter. The ensemble removes a parameter by averaging
over horizons chosen in advance; the composite multiplies two existing strategies
and has no free parameter of its own. What is tested here is that they do what
they claim mechanically — the question of whether they help is answered by
measurement across eight markets, not by a unit test.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.engine.backtest import UNLIMITED, run_backtest
from sentinel.evaluation.causality import check_causality
from sentinel.evaluation.null_test import run_null_test
from sentinel.sandbox.generators.gbm import GBMGenerator
from sentinel.sandbox.generators.regime import RegimeSwitchingGenerator
from sentinel.strategies.base import Strategy
from sentinel.strategies.baseline import AbsoluteMomentum, BuyAndHold, EnsembleMomentum
from sentinel.strategies.composite import TrendScaledVolatility
from sentinel.strategies.volatility import VolatilityTarget


@pytest.fixture(scope="module")
def trending():
    return GBMGenerator(mu=0.10, sigma=0.16).generate(n_steps=2000, seed=41).data


@pytest.fixture(scope="module")
def multi_asset():
    return GBMGenerator(mu=0.08).generate(n_steps=1600, n_assets=3, seed=42).data


class TestEnsembleMomentum:
    def test_is_causal(self, trending) -> None:
        assert check_causality(EnsembleMomentum(), trending).is_causal

    def test_is_causal_across_assets(self, multi_asset) -> None:
        report = check_causality(EnsembleMomentum(), multi_asset)
        assert report.is_causal, str(report)

    def test_exposure_is_the_fraction_of_horizons_that_are_positive(self, trending) -> None:
        """Four horizons means five possible positions, not two.

        That graduation is a side effect rather than the goal -- the point is
        that no single lookback determines the answer -- but it is the visible
        consequence and it is what reduces whipsaw.
        """
        weights = EnsembleMomentum(lookbacks=(21, 63, 126, 252), rebalance_days=1).compute_weights(
            trending
        )
        settled = np.unique(weights.iloc[300:].to_numpy().round(6))
        assert set(settled) <= {0.0, 0.25, 0.5, 0.75, 1.0}
        assert len(settled) > 2, "a binary result means the horizons never disagree"

    def test_a_horizon_that_has_not_filled_does_not_vote(self) -> None:
        """Treating "no opinion" as bearish would make the strategy
        systematically defensive for its first year — an artefact of warmup
        rather than a view about the market."""
        data = GBMGenerator(mu=0.15, sigma=0.10).generate(n_steps=400, seed=9).data
        weights = EnsembleMomentum(rebalance_days=1).compute_weights(data)
        # Only the 21- and 63-day horizons have filled by day 100, and in a
        # strongly rising market both are positive, so exposure must be full.
        assert weights.iloc[100:120].to_numpy().max() == pytest.approx(1.0)

    def test_a_single_lookback_reduces_to_absolute_momentum(self, trending) -> None:
        """The ensemble must generalise the rule it replaces, not differ from it."""
        ensemble = EnsembleMomentum(lookbacks=(252,), rebalance_days=21).compute_weights(trending)
        single = AbsoluteMomentum(lookback=252, rebalance_days=21).compute_weights(trending)
        np.testing.assert_allclose(ensemble.to_numpy(), single.to_numpy())

    def test_does_not_profit_on_noise(self) -> None:
        result = run_null_test(
            EnsembleMomentum(), GBMGenerator(mu=0.0), n_markets=40, n_steps=1260, workers=1
        )
        assert result.passed(), result.report()

    @pytest.mark.parametrize(
        "kwargs,message",
        [
            ({"lookbacks": ()}, "at least one lookback"),
            ({"lookbacks": (1, 63)}, "at least 2 days"),
            ({"rebalance_days": 0}, "rebalance_days must be"),
        ],
    )
    def test_rejects_impossible_parameters(self, kwargs, message) -> None:
        with pytest.raises(ValueError, match=message):
            EnsembleMomentum(**kwargs)


class TestTrendScaledVolatility:
    def test_is_causal(self, trending) -> None:
        report = check_causality(TrendScaledVolatility(), trending)
        assert report.is_causal, str(report)

    def test_either_component_can_shrink_the_position_alone(self, trending) -> None:
        """Multiplication rather than an average, and deliberately so.

        A falling market is a reason to stand aside however calm it is, and a
        violent market is a reason to hold less however strong the trend. An
        average would let one component override the other, which is the opposite
        of what a defensive system wants.
        """
        composite = TrendScaledVolatility(band=0.0)
        combined = composite.compute_weights(trending).to_numpy().ravel()
        trend = composite.trend.compute_weights(trending).to_numpy().ravel()
        volatility = composite.volatility.compute_weights(trending).to_numpy().ravel()

        assert (combined <= trend + 1e-9).all()
        assert (combined <= volatility + 1e-9).all()

    def test_the_trend_going_flat_takes_the_position_to_zero(self, trending) -> None:
        composite = TrendScaledVolatility(band=0.0)
        combined = composite.compute_weights(trending).to_numpy().ravel()
        trend = composite.trend.compute_weights(trending).to_numpy().ravel()
        flat = trend == 0.0
        assert flat.sum() > 20, "the fixture must contain some out-of-market stretches"
        assert (combined[flat] == 0.0).all()

    def test_never_borrows(self, trending) -> None:
        weights = TrendScaledVolatility().compute_weights(trending)
        assert weights.to_numpy().max() <= 1.0
        assert weights.to_numpy().min() >= 0.0

    def test_holds_less_than_either_component_on_average(self, trending) -> None:
        """Which is the cost. It will trail badly in a long calm bull run, and
        the drawdown result has to be worth that."""
        composite = TrendScaledVolatility(band=0.0)
        combined = composite.compute_weights(trending).to_numpy().mean()
        trend = composite.trend.compute_weights(trending).to_numpy().mean()
        assert combined < trend

    def test_reduces_drawdown_against_both_components(self) -> None:
        """The claim the strategy exists to make, judged across paths not on one.

        Measured across eight national indices it was shallower than
        buy-and-hold in 8 of 8, by a median of 37 points, and shallower than
        either component alone. Here it is checked on twelve regime-switching
        markets so the suite needs no network.

        Twelve rather than one because the first version of this test used a
        single seed and failed: on that path the composite drew down 27.2%
        against volatility targeting's 22.2%. Nothing was wrong with the
        strategy. A single path is exactly what this project's founding argument
        says proves nothing, and asserting a distributional claim on one is the
        same error one level down.
        """
        drawdowns: dict[str, list] = {"hold": [], "trend": [], "volatility": [], "composite": []}
        for seed in range(12):
            scenario = RegimeSwitchingGenerator().generate(n_steps=2520, n_assets=1, seed=seed)
            for name, strategy in (
                ("hold", BuyAndHold()),
                ("trend", AbsoluteMomentum()),
                ("volatility", VolatilityTarget()),
                ("composite", TrendScaledVolatility()),
            ):
                drawdowns[name].append(
                    run_backtest(
                        scenario.data, strategy, limits=UNLIMITED
                    ).performance.max_drawdown
                )

        mean = {name: float(np.mean(values)) for name, values in drawdowns.items()}
        # Drawdowns are negative, so "greater" means shallower.
        assert mean["composite"] > mean["hold"], mean
        assert mean["composite"] > mean["trend"], mean
        assert mean["composite"] > mean["volatility"], mean

    def test_does_not_profit_on_noise(self) -> None:
        result = run_null_test(
            TrendScaledVolatility(), GBMGenerator(mu=0.0), n_markets=40, n_steps=1260, workers=1
        )
        assert result.passed(), result.report()

    def test_rejects_an_impossible_band(self) -> None:
        with pytest.raises(ValueError, match="band must be"):
            TrendScaledVolatility(band=1.0)

    def test_accepts_the_ensemble_as_its_trend(self, trending) -> None:
        strategy = TrendScaledVolatility(trend=EnsembleMomentum())
        assert check_causality(strategy, trending).is_causal


class TestMultiAssetSizing:
    """The bug here was silent, plausible, and produced a whole report.

    `VolatilityTarget` used to return a single column -- always `tickers[0]` --
    and anything multiplying it against a multi-asset weight matrix got numpy
    broadcasting rather than an error. A six-asset portfolio of equities, bonds
    and gold was therefore sized entirely by SPY's volatility. Nothing failed,
    no shape mismatch was raised, and the resulting Sharpe and drawdown looked
    entirely reasonable.

    Fixing it exposed a second one immediately: both components are complete
    allocators and both divide their conviction across the available assets, so
    multiplying their raw weights divides twice. That version returned 0.43% a
    year against buy-and-hold's 8.09% -- low enough to notice, but not obviously
    a bug rather than a conservative strategy.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def panel(cls):
        return GBMGenerator(mu=0.08, sigma=0.16).generate(
            n_steps=2000, n_assets=4, seed=77
        ).data

    def test_volatility_target_sizes_every_asset(self, panel) -> None:
        weights = VolatilityTarget().compute_weights(panel)
        assert list(weights.columns) == panel.tickers
        assert weights.shape[1] == 4

    def test_each_asset_is_sized_by_its_own_volatility(self) -> None:
        """Gold does not have equity's volatility, and sizing it as though it
        did is not a conservative approximation -- it is a different strategy."""
        from sentinel.sandbox.market import MarketData
        import pandas as pd

        rng = np.random.default_rng(3)
        n = 1500
        index = pd.bdate_range("2000-01-03", periods=n, name="date")
        # One calm asset and one wild one, same drift.
        calm = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
        wild = 100 * np.exp(np.cumsum(rng.normal(0, 0.020, n)))
        data = MarketData(prices=pd.DataFrame({"CALM": calm, "WILD": wild}, index=index))

        weights = VolatilityTarget(band=0.0).compute_weights(data).iloc[200:]
        assert weights["CALM"].mean() > 2 * weights["WILD"].mean()

    def test_the_composite_refuses_mismatched_columns(self, panel) -> None:
        """Rather than letting numpy broadcast one column over four."""

        class SingleColumn(Strategy):
            name = "single_column"

            def compute_weights(self, data):
                import pandas as pd

                return pd.DataFrame(
                    {data.tickers[0]: np.ones(len(data.prices))}, index=data.prices.index
                )

        composite = TrendScaledVolatility()
        composite.volatility = SingleColumn()
        with pytest.raises(ValueError, match="silently broadcast"):
            composite.compute_weights(panel)

    def test_the_composite_does_not_divide_the_exposure_twice(self, panel) -> None:
        """Aggregate views multiply; the split across assets comes from the product.

        If trend wants to be fully invested and volatility wants half exposure,
        the answer is half exposure -- not a twenty-fourth of it.
        """
        composite = TrendScaledVolatility(trend=EnsembleMomentum(), band=0.0)
        combined = composite.compute_weights(panel).to_numpy().sum(axis=1)
        trend = composite.trend.compute_weights(panel).to_numpy().sum(axis=1)
        volatility = composite.volatility.compute_weights(panel).to_numpy().sum(axis=1)

        settled = slice(400, None)
        np.testing.assert_allclose(
            combined[settled], (trend * volatility)[settled], atol=1e-9
        )

    def test_a_single_asset_is_unaffected_by_the_multi_asset_rule(self) -> None:
        """The regression guard. For one asset the composition is exactly
        trend x volatility, as it always was."""
        data = GBMGenerator(mu=0.08).generate(n_steps=1600, n_assets=1, seed=5).data
        composite = TrendScaledVolatility(band=0.0)
        combined = composite.compute_weights(data).to_numpy().ravel()
        trend = composite.trend.compute_weights(data).to_numpy().ravel()
        volatility = composite.volatility.compute_weights(data).to_numpy().ravel()
        np.testing.assert_allclose(combined, trend * volatility, atol=1e-9)

    def test_a_basket_of_quiet_assets_cannot_become_leverage(self, panel) -> None:
        """Six assets each sized to a 12% target must not add up to 72%."""
        weights = VolatilityTarget(target_volatility=0.30).compute_weights(panel)
        assert weights.to_numpy().sum(axis=1).max() <= 1.0 + 1e-9

    def test_stays_causal_across_assets(self, panel) -> None:
        for strategy in (VolatilityTarget(), TrendScaledVolatility(trend=EnsembleMomentum())):
            report = check_causality(strategy, panel)
            assert report.is_causal, str(report)
