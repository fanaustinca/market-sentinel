"""The most important test file in the repository.

Lookahead bias makes a broken strategy look excellent, which is why it survives
code review and destroys real money. Every test here asks the same question in a
different place:

    Does a decision made on day t change when data from after day t arrives?

If it does, the code used the future, no matter what it appeared to say.

The file also contains deliberately broken strategies whose only purpose is to
prove the detector works. A test that has never caught anything is not evidence
of correctness -- it may simply be blind. These make sure that if lookahead is
ever introduced, this file fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.ai.model import SizingRule, WalkForwardClassifier, WalkForwardModel
from sentinel.evaluation.causality import check_causality, truncate
from sentinel.features.build import build_features
from sentinel.sandbox.generators import GBMGenerator, RegimeSwitchingGenerator
from sentinel.sandbox.market import MarketData
from sentinel.strategies.base import Strategy
from sentinel.strategies.baseline import (
    AbsoluteMomentum,
    AlwaysCash,
    BuyAndHold,
    DualMomentum,
    ShortHorizonMomentum,
)


@pytest.fixture(scope="module")
def market() -> MarketData:
    return GBMGenerator(mu=0.06).generate(n_steps=1600, seed=77).data


@pytest.fixture(scope="module")
def multi_asset_market() -> MarketData:
    return GBMGenerator(mu=0.06).generate(n_steps=1600, n_assets=3, seed=78).data


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

def test_features_do_not_change_when_the_future_arrives(market: MarketData) -> None:
    """Every feature value must be final the moment it is computed.

    This is the layer where lookahead most often enters, usually through
    something that does not feel like modelling at all -- standardising a column
    by its full-sample mean, or a rolling window that centres instead of trailing.
    """
    full = build_features(market)
    for cut in (800, 1100, 1400):
        partial = build_features(truncate(market, cut))
        pd.testing.assert_frame_equal(full.iloc[:cut], partial, check_exact=False, rtol=1e-12)


def test_drawdown_uses_an_expanding_peak(market: MarketData) -> None:
    """Drawdown must be measured against the highest price *so far*.

    Using the full-sample maximum is a textbook leak, and a seductive one: the
    resulting feature looks sensible, correlates with real drawdown, and tells the
    model on day 200 how high the market will get on day 1400.
    """
    features = build_features(market)
    prices = market.prices.iloc[:, 0]
    expected = prices / prices.cummax() - 1.0
    pd.testing.assert_series_equal(features["drawdown"], expected, check_names=False)
    # The first observation is always at its own running peak.
    assert features["drawdown"].iloc[0] == 0.0


def test_no_feature_is_standardised_over_the_whole_sample(market: MarketData) -> None:
    """A leaked feature would be suspiciously well behaved end to end.

    Standardising over the full sample leaves a column whose overall mean is ~0
    and standard deviation ~1 by construction. Trailing statistics almost never
    produce that, so a column that looks perfectly normalised is a red flag.
    """
    features = build_features(market).dropna()
    for column in features.columns:
        values = features[column]
        looks_normalised = abs(values.mean()) < 1e-9 and abs(values.std(ddof=0) - 1.0) < 1e-9
        assert not looks_normalised, f"{column} appears standardised over the full sample"


# --------------------------------------------------------------------------
# Strategies, including the AI
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "strategy",
    [
        BuyAndHold(),
        AlwaysCash(),
        AbsoluteMomentum(lookback=252, rebalance_days=21),
        AbsoluteMomentum(lookback=63, rebalance_days=5),
        ShortHorizonMomentum(),
    ],
    ids=lambda s: f"{s.name}",
)
def test_baseline_strategies_are_causal(strategy: Strategy, market: MarketData) -> None:
    report = check_causality(strategy, market)
    assert report.is_causal, str(report)


def test_dual_momentum_is_causal(multi_asset_market: MarketData) -> None:
    report = check_causality(DualMomentum(lookback=252, rebalance_days=21), multi_asset_market)
    assert report.is_causal, str(report)


@pytest.mark.parametrize(
    "strategy",
    [
        WalkForwardModel(min_train=400, retrain_every=100, sizing=SizingRule(aggression=0.5)),
        WalkForwardClassifier(min_train=400, retrain_every=100),
    ],
    ids=lambda s: s.name,
)
def test_the_ai_is_causal(strategy: Strategy, market: MarketData) -> None:
    """The one that matters most.

    The AI has the most places to get this wrong: the label is the *next*
    period's return, so the training window must stop one row earlier than
    instinct suggests. A model used on day t may train on labels up to y[t-1],
    because y[t-1] is the return from t-1 to t and is first observable on day t.
    Off by one in the other direction and the model trains on the answer to the
    question it is being asked.
    """
    report = check_causality(strategy, market)
    assert report.is_causal, str(report)


# --------------------------------------------------------------------------
# Proving the detector actually detects
# --------------------------------------------------------------------------

class TomorrowPeeker(Strategy):
    """Deliberately broken: holds the asset exactly when tomorrow is an up day.

    The purest possible cheat, and a useful reference point -- run through the
    backtester it produces a Sharpe ratio in the tens, which is what impossible
    results look like.
    """

    name = "tomorrow_peeker"

    def compute_weights(self, data: MarketData) -> pd.DataFrame:
        forward = data.prices.pct_change().shift(-1)
        return (forward > 0).astype(float)


class FullSampleNormaliser(Strategy):
    """Deliberately broken in the subtle way: scales by full-sample statistics.

    Far more realistic than `TomorrowPeeker`, and far more dangerous. Nothing here
    looks like cheating -- it looks like ordinary preprocessing, the kind that
    appears in a thousand tutorials -- yet every row silently carries information
    about the entire series, including its future.
    """

    name = "full_sample_normaliser"

    def compute_weights(self, data: MarketData) -> pd.DataFrame:
        returns = np.log(data.prices).diff(21)
        standardised = (returns - returns.mean()) / returns.std()
        return (standardised > 0).astype(float)


@pytest.mark.parametrize(
    "cheat", [TomorrowPeeker(), FullSampleNormaliser()], ids=lambda s: s.name
)
def test_detector_catches_known_cheats(cheat: Strategy, market: MarketData) -> None:
    """If this ever passes, the detector has gone blind and every other test in
    this file is worthless."""
    report = check_causality(cheat, market)
    assert not report.is_causal, f"detector failed to catch {cheat.name}"


def test_peeking_produces_impossible_results(market: MarketData) -> None:
    """Calibration for what "too good to be true" actually looks like.

    A strategy with one day of foresight earns a Sharpe far beyond anything real.
    Worth having written down: when a backtest ever produces numbers in this
    range, the answer is a leak, not a discovery.
    """
    from sentinel.engine.backtest import UNLIMITED, run_backtest

    result = run_backtest(market, TomorrowPeeker(), limits=UNLIMITED)
    assert result.performance.sharpe > 5.0


# --------------------------------------------------------------------------
# The engine's own timing
# --------------------------------------------------------------------------

def test_engine_applies_yesterdays_decision_to_todays_return() -> None:
    """The engine's core alignment, checked against a hand-computed answer.

    A strategy that holds nothing until a known date, then everything, must earn
    exactly the return of the period *after* that decision -- not the period
    before, and not the period of the decision itself.
    """
    from sentinel.engine.backtest import CostModel, UNLIMITED, run_backtest

    index = pd.bdate_range("2020-01-01", periods=5)
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0, 133.1, 146.41]}, index=index)
    data = MarketData(prices=prices, name="handmade")

    class InvestOnDayTwo(Strategy):
        name = "invest_on_day_two"

        def compute_weights(self, market_data: MarketData) -> pd.DataFrame:
            weights = pd.DataFrame(0.0, index=market_data.prices.index, columns=["A"])
            weights.iloc[2] = 1.0  # decided on day 2, applies to the day 2 -> 3 move
            return weights

    free = CostModel(commission_bps=0, spread_bps=0, slippage_bps=0)
    result = run_backtest(data, InvestOnDayTwo(), costs=free, limits=UNLIMITED)

    # Prices rise 10% every period, so the single held period earns exactly 10%.
    assert result.returns.iloc[0] == pytest.approx(0.0)
    assert result.returns.iloc[1] == pytest.approx(0.0)
    assert result.returns.iloc[2] == pytest.approx(0.10)
    assert result.returns.iloc[3] == pytest.approx(0.0)
    assert result.equity.iloc[-1] == pytest.approx(1.10)


def test_regime_labels_are_not_reachable_from_market_data() -> None:
    """The answer key must not be an attribute away from the model.

    `Scenario` holds prices and truth together for the evaluation harness, but a
    strategy receives `MarketData` alone. If a route ever opened from one to the
    other, a model could read the regime labels it is supposed to infer.
    """
    scenario = RegimeSwitchingGenerator().generate(n_steps=300, seed=1)
    assert scenario.truth.regimes is not None
    assert not hasattr(scenario.data, "truth")
    assert not hasattr(scenario.data, "regimes")
    assert set(vars(scenario.data)) == {"prices", "name"}
