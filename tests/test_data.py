"""Tests for the real-data adapter.

These skip when the parquet cache is empty, because a test suite that reaches the
network is a test suite that fails when the network does -- and worse, one whose
result depends on what Yahoo returned that morning. Populate the cache once by
running `python experiments/real_data.py`, and thereafter these run offline
against fixed bytes.

What is checked is the interface contract, not the data. Real prices must arrive
as a `MarketData` indistinguishable from a synthetic one, because that
interchangeability is the property the entire Reality Ladder rests on: climbing a
rung must change the data source and nothing else.
"""

from __future__ import annotations

import pytest

from sentinel.data.yahoo import CACHE_DIR, fingerprint, load_prices
from sentinel.engine.backtest import run_backtest
from sentinel.evaluation.causality import check_causality
from sentinel.sandbox.generators.bootstrap import BootstrapGenerator
from sentinel.sandbox.market import MarketData
from sentinel.strategies.baseline import AbsoluteMomentum, BuyAndHold

cached = pytest.mark.skipif(
    not CACHE_DIR.exists() or not list(CACHE_DIR.glob("SPY_*.parquet")),
    reason="no cached price data; run experiments/real_data.py once to populate it",
)


@pytest.fixture(scope="module")
def spy() -> MarketData:
    return load_prices("SPY")


@cached
def test_real_prices_arrive_as_ordinary_market_data(spy: MarketData) -> None:
    """The interface the whole Reality Ladder depends on."""
    assert isinstance(spy, MarketData)
    assert spy.prices.index.is_monotonic_increasing
    assert (spy.prices > 0).all().all()
    assert spy.tickers == ["SPY"]
    assert not hasattr(spy, "truth"), "MarketData must have no route to an answer key"


@cached
def test_prices_are_total_return_adjusted(spy: MarketData) -> None:
    """Dividends must be baked in, or every timing strategy is flattered.

    Raw closing prices ignore roughly 1.5-2% a year of dividends for broad equity
    ETFs, which is credited to any strategy that was in cash when they were paid.
    SPY's price return since 1993 is far below its total return, and the gap is
    the test: a price-only series would show materially less growth.
    """
    growth = spy.prices["SPY"].iloc[-1] / spy.prices["SPY"].iloc[0]
    years = spy.n_steps / 252
    annualised = growth ** (1 / years) - 1
    assert annualised > 0.085, f"{annualised:.2%} a year is too low to be total return"


@cached
def test_the_same_strategies_run_unchanged_on_real_data(spy: MarketData) -> None:
    """No branch anywhere says "if this is real". That is the point of rung 2."""
    result = run_backtest(spy, BuyAndHold())
    assert result.performance.n_periods == spy.n_steps - 1
    assert result.equity.iloc[-1] > 0


@cached
def test_strategies_stay_causal_on_real_prices(spy: MarketData) -> None:
    """Real data has gaps, holidays and fat tails that synthetic data does not.

    Causality is a property of the code rather than of the data, so this should
    pass -- but "should" is exactly what the truncation check exists to replace.
    """
    window = MarketData(prices=spy.prices.iloc[-2000:], name="spy-recent")
    assert check_causality(AbsoluteMomentum(), window).is_causal


@cached
def test_a_bootstrap_of_real_returns_is_a_valid_null_market(spy: MarketData) -> None:
    """The floor a real-data result is judged against."""
    generator = BootstrapGenerator.from_market(spy, block_size=1, demean=True)
    assert not generator.has_exploitable_signal
    scenario = generator.generate(n_steps=2520, seed=1)
    assert scenario.data.n_steps == 2520
    assert scenario.truth.params["n_source_observations"] == spy.n_steps - 1


@cached
def test_fingerprint_pins_a_result_to_its_input(spy: MarketData) -> None:
    """Yahoo restates history. A changed fingerprint means results are not comparable."""
    assert fingerprint(spy) == fingerprint(spy)
    shortened = MarketData(prices=spy.prices.iloc[:-1], name=spy.name)
    assert fingerprint(spy) != fingerprint(shortened)


@cached
def test_multiple_tickers_align_by_intersection() -> None:
    """Not by union with a forward fill.

    A forward-filled holiday looks like a zero-return day, which biases
    volatility downward on precisely the asset that was not trading.
    """
    pair = load_prices(["SPY", "IEF"])
    assert pair.prices.notna().all().all()
    assert pair.prices.index.is_unique
    # IEF starts in 2002, so the pair cannot begin at SPY's 1993 start.
    assert pair.prices.index[0].year >= 2002


def test_an_empty_ticker_list_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one ticker"):
        load_prices([])
