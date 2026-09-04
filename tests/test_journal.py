"""Tests for the decision journal.

The journal's only value is that it records a prediction *before* the outcome is
known. Everything protected here follows from that: the reading must come from
the same code the backtests ran, it must refuse to act on stale data, and it must
refuse to overwrite a prediction that has already been made.

Most of these run on synthetic markets, so they work with no network and no
cached prices.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sentinel.engine.backtest import UNLIMITED, run_backtest
from sentinel.journal.signals import (
    MAX_STALENESS_DAYS,
    current_signals,
    write_journal_entry,
)
from sentinel.sandbox.generators.gbm import GBMGenerator
from sentinel.sandbox.market import MarketData
from sentinel.strategies.baseline import AbsoluteMomentum, AlwaysCash, BuyAndHold
from sentinel.strategies.volatility import VolatilityTarget


def market_ending_today(n_steps: int = 1600, seed: int = 5) -> MarketData:
    """A synthetic market whose last bar is today, so staleness checks pass."""
    data = GBMGenerator(mu=0.08).generate(n_steps=n_steps, seed=seed).data
    index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_steps, name="date")
    return MarketData(prices=data.prices.set_axis(index), name=data.name)


@pytest.fixture(scope="module")
def today() -> MarketData:
    return market_ending_today()


class TestTheReadingMatchesTheBacktest:
    def test_the_signal_is_the_backtest_s_last_decision(self, today: MarketData) -> None:
        """There is no separate live code path, and this is what pins that.

        Most retail systems have subtly different backtest and production code,
        and the gap is where undetected bugs live -- the backtest exercises one
        path and production runs the other, so a discrepancy can survive
        indefinitely with nothing looking wrong.
        """
        strategy = AbsoluteMomentum()
        report = current_signals(today, [strategy])[0]
        expected = strategy.compute_weights(today).iloc[-1]
        for ticker, weight in report.weights.items():
            assert weight == pytest.approx(float(expected[ticker]))

    def test_previous_weights_are_the_row_before(self, today: MarketData) -> None:
        """Which is what makes "did anything change?" answerable with no stored state."""
        strategy = AbsoluteMomentum()
        report = current_signals(today, [strategy])[0]
        expected = strategy.compute_weights(today).iloc[-2]
        for ticker, weight in report.previous_weights.items():
            assert weight == pytest.approx(float(expected[ticker]))

    def test_cash_is_whatever_is_not_invested(self, today: MarketData) -> None:
        report = current_signals(today, [VolatilityTarget()])[0]
        assert report.cash == pytest.approx(1.0 - sum(report.weights.values()))

    def test_a_cash_position_is_described_as_such(self, today: MarketData) -> None:
        report = current_signals(today, [AlwaysCash()])[0]
        assert "100% cash" in report.describe()

    def test_every_reading_carries_a_reason(self, today: MarketData) -> None:
        """plan.md section 4 makes interpretability a requirement.

        A system whose decisions cannot be interrogated cannot be maintained,
        debugged, or held on to during a drawdown -- and the drawdown is exactly
        when the explanation is needed.
        """
        for report in current_signals(today, [AbsoluteMomentum(), VolatilityTarget()]):
            assert len(report.reason) > 20
            assert report.reason == report.reason.strip()

    def test_the_momentum_reason_states_the_actual_trailing_return(self, today: MarketData) -> None:
        report = current_signals(today, [AbsoluteMomentum(lookback=252)])[0]
        prices = today.prices.iloc[:, 0]
        trailing = float(np.log(prices.iloc[-1] / prices.iloc[-253]))
        assert f"{trailing:+.1%}" in report.reason
        assert "252-day" in report.reason


class TestStaleness:
    def test_stale_data_is_refused_not_flagged(self) -> None:
        """plan.md section 8 requires blocking rather than warning.

        A warning nobody reads is not a control, and a decision taken on a stale
        feed is a decision about the wrong world.
        """
        old = GBMGenerator().generate(n_steps=800, seed=1).data
        with pytest.raises(ValueError, match="Refusing to produce"):
            current_signals(old, [BuyAndHold()])

    def test_a_normal_weekend_gap_is_accepted(self) -> None:
        """Markets close. A few days stale is ordinary, not a fault."""
        n = 900
        index = pd.bdate_range(
            end=pd.Timestamp.today().normalize() - pd.Timedelta(days=MAX_STALENESS_DAYS - 2),
            periods=n,
            name="date",
        )
        data = GBMGenerator().generate(n_steps=n, seed=2).data
        recent = MarketData(prices=data.prices.set_axis(index), name="recent")
        assert current_signals(recent, [BuyAndHold()])


class TestChangeDetection:
    def test_an_unchanged_position_is_not_reported_as_a_change(self, today: MarketData) -> None:
        report = current_signals(today, [BuyAndHold()])[0]
        assert not report.is_change
        assert "CHANGED" not in report.describe()

    def test_a_change_is_detected_and_marked(self, today: MarketData) -> None:
        class Flipper(BuyAndHold):
            name = "flipper"

            def compute_weights(self, data):
                frame = super().compute_weights(data)
                frame.iloc[-1] = 0.0
                return frame

        report = current_signals(today, [Flipper()])[0]
        assert report.is_change
        assert "CHANGED" in report.describe()


class TestJournalEntries:
    def test_writes_a_dated_entry(self, today: MarketData, tmp_path) -> None:
        reports = current_signals(today, [AbsoluteMomentum(), BuyAndHold()])
        path = write_journal_entry(reports, today, directory=tmp_path)

        payload = json.loads(path.read_text())
        assert payload["data_as_of"] == str(today.prices.index[-1].date())
        assert len(payload["signals"]) == 2
        assert {s["strategy"] for s in payload["signals"]} == {"absolute_momentum", "buy_and_hold"}
        assert "data_fingerprint" in payload

    def test_refuses_to_overwrite_an_existing_prediction(self, today: MarketData, tmp_path) -> None:
        """The one rule that makes the journal worth keeping.

        Rewriting a prediction after the fact -- even innocently, even to fix a
        typo -- removes the only property that made it evidence. A decision
        recalled after the outcome is not a decision anyone can check.
        """
        reports = current_signals(today, [BuyAndHold()])
        write_journal_entry(reports, today, directory=tmp_path)
        with pytest.raises(FileExistsError, match="before the outcome was known"):
            write_journal_entry(reports, today, directory=tmp_path)

    def test_the_entry_records_the_reason_and_the_floor(self, today: MarketData, tmp_path) -> None:
        reports = current_signals(
            today, [AbsoluteMomentum()], noise_floors={"absolute_momentum": 0.506}
        )
        path = write_journal_entry(reports, today, directory=tmp_path)
        signal = json.loads(path.read_text())["signals"][0]
        assert signal["noise_floor"] == 0.506
        assert "trailing" in signal["reason"]
