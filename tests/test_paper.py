"""Tests for the paper portfolio.

The point of paper trading is that its record cannot be edited after the outcome
is known. So most of these tests are about the ledger refusing things, and about
the accounting closing exactly -- an equity figure that quietly loses a few
dollars a day would look like a strategy underperforming rather than a bug.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from sentinel.engine.backtest import CostModel
from sentinel.paper.portfolio import PaperEntry, PaperPortfolio
from sentinel.sandbox.market import MarketData
from sentinel.strategies.baseline import BuyAndHold


@pytest.fixture
def data() -> MarketData:
    rng = np.random.default_rng(4)
    n = 400
    returns = rng.normal(0.0003, 0.01, (n, 2))
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(returns, axis=0)),
        index=pd.bdate_range("2024-01-01", periods=n),
        columns=["A", "B"],
    )
    return MarketData(prices=prices, name="test")


@pytest.fixture
def portfolio(tmp_path) -> PaperPortfolio:
    return PaperPortfolio(BuyAndHold(), ledger=tmp_path)


class TestLedgerDiscipline:
    def test_first_step_writes_an_entry(self, portfolio, data):
        entry = portfolio.step(data)
        assert portfolio.path.exists()
        assert len(portfolio.history()) == 1
        assert entry.date == str(date.today())

    def test_a_second_step_on_the_same_day_is_refused(self, portfolio, data):
        """Rewriting a prediction after the fact destroys the only useful property."""
        portfolio.step(data)
        with pytest.raises(ValueError, match="append-only"):
            portfolio.step(data)

    def test_record_refuses_to_replace_an_existing_day(self, portfolio, data):
        portfolio.step(data)
        clone = portfolio.history()[0]
        with pytest.raises(ValueError, match="append-only"):
            portfolio.record(clone)

    def test_the_entry_pins_the_data_it_used(self, portfolio, data):
        entry = portfolio.step(data)
        assert len(entry.fingerprint) == 12
        assert entry.as_of == str(data.prices.index[-1].date())

    def test_history_survives_a_round_trip_through_disk(self, portfolio, data):
        entry = portfolio.step(data)
        reloaded = PaperPortfolio(BuyAndHold(), ledger=portfolio.ledger).history()[0]
        assert reloaded == entry


class TestAccounting:
    def test_equity_equals_holdings_plus_cash(self, portfolio, data):
        entry = portfolio.step(data)
        held = sum(entry.shares[t] * entry.prices[t] for t in entry.shares)
        assert held + entry.cash == pytest.approx(entry.equity, rel=1e-9)

    def test_first_trade_costs_the_engine_rate(self, tmp_path, data):
        """Paper and backtest must charge identically or one is validating nothing."""
        costs = CostModel()
        book = PaperPortfolio(BuyAndHold(), ledger=tmp_path, starting_cash=10_000.0, costs=costs)
        entry = book.step(data)
        # BuyAndHold goes from all cash to fully invested: turnover 1.0.
        assert entry.turnover == pytest.approx(1.0, abs=1e-9)
        assert entry.cost == pytest.approx(10_000.0 * costs.one_way_cost, rel=1e-9)
        assert entry.equity == pytest.approx(10_000.0 - entry.cost, rel=1e-9)

    def test_weights_match_the_strategy(self, portfolio, data):
        entry = portfolio.step(data)
        for ticker, weight in entry.target_weights.items():
            realised = entry.shares[ticker] * entry.prices[ticker] / entry.equity
            assert realised == pytest.approx(weight, abs=1e-9)

    def test_holding_still_costs_nothing(self, tmp_path, data):
        """Second day with unchanged targets must not be charged turnover."""
        book = PaperPortfolio(BuyAndHold(), ledger=tmp_path)
        first = book.step(data)
        # Simulate the next day by writing directly, then stepping on the same
        # prices: an unchanged target against unchanged prices is zero turnover.
        rows = json.loads(book.path.read_text())
        rows[0]["date"] = "1999-01-01"
        book.path.write_text(json.dumps(rows))
        second = book.step(data)
        assert second.turnover == pytest.approx(0.0, abs=1e-9)
        assert second.cost == pytest.approx(0.0, abs=1e-9)

    def test_a_wiped_out_account_refuses_to_continue(self, tmp_path, data):
        book = PaperPortfolio(BuyAndHold(), ledger=tmp_path)
        book.record(PaperEntry(
            date="1999-01-01", strategy=book.strategy.name, as_of="1999-01-01",
            fingerprint="x" * 12, target_weights={"A": 0.0, "B": 0.0},
            prices={"A": 1.0, "B": 1.0}, shares={"A": 0.0, "B": 0.0},
            cash=0.0, equity=0.0, turnover=0.0, cost=0.0,
        ))
        with pytest.raises(ValueError, match="wiped out"):
            book.step(data)


class TestEquityCurve:
    def test_empty_before_any_steps(self, portfolio):
        assert portfolio.equity_curve().empty

    def test_curve_is_dated_and_named(self, portfolio, data):
        portfolio.step(data)
        curve = portfolio.equity_curve()
        assert isinstance(curve.index, pd.DatetimeIndex)
        assert curve.name == portfolio.strategy.name
