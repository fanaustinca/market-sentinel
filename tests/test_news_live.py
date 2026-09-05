"""Tests for the live event feed, with the network stubbed out.

These assert the *contract* rather than the data: that a missing date is omitted
rather than guessed, that a broken ticker cannot take down a whole universe, and
that the frame shape is stable. Anything stronger would be testing Yahoo.
"""

from __future__ import annotations

import pandas as pd
import pytest

from sentinel.data.news import live


class _Ticker:
    def __init__(self, dates=None, news=None, fail=False):
        self._dates, self._news, self._fail = dates, news or [], fail

    def get_earnings_dates(self, limit=12):
        if self._fail:
            raise RuntimeError("delisted")
        return self._dates

    @property
    def news(self):
        return self._news


@pytest.fixture
def stub(monkeypatch):
    def install(mapping):
        module = type("yf", (), {"Ticker": staticmethod(lambda t: mapping[t])})
        monkeypatch.setitem(__import__("sys").modules, "yfinance", module)
    return install


def _frame(when: str) -> pd.DataFrame:
    return pd.DataFrame({"EPS Estimate": [1.0]}, index=pd.DatetimeIndex([when], tz="UTC"))


class TestUpcomingEarnings:
    def test_it_reports_the_soonest_future_date(self, stub):
        future = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=20)
        stub({"A": _Ticker(_frame(str(future)))})
        got = live.upcoming_earnings(["A"])
        assert list(got.columns) == ["ticker", "date", "days_away"]
        assert got.loc[0, "ticker"] == "A"
        assert 18 <= got.loc[0, "days_away"] <= 21

    def test_past_dates_are_not_reported_as_upcoming(self, stub):
        past = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=10)
        stub({"A": _Ticker(_frame(str(past)))})
        assert live.upcoming_earnings(["A"]).empty

    def test_a_ticker_with_no_date_is_omitted_not_guessed(self, stub):
        """A guessed date would be worse than none: cadence is too imprecise."""
        stub({"A": _Ticker(None)})
        assert live.upcoming_earnings(["A"]).empty

    def test_one_broken_ticker_does_not_lose_the_others(self, stub):
        future = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=5)
        stub({"A": _Ticker(fail=True), "B": _Ticker(_frame(str(future)))})
        got = live.upcoming_earnings(["A", "B"])
        assert list(got["ticker"]) == ["B"]

    def test_results_are_sorted_soonest_first(self, stub):
        now = pd.Timestamp.now(tz="UTC")
        stub({
            "A": _Ticker(_frame(str(now + pd.Timedelta(days=40)))),
            "B": _Ticker(_frame(str(now + pd.Timedelta(days=3)))),
        })
        assert list(live.upcoming_earnings(["A", "B"])["ticker"]) == ["B", "A"]


class TestRecentHeadlines:
    def test_it_flattens_the_nested_content_payload(self, stub):
        stub({"A": _Ticker(news=[{"content": {
            "title": "A beats estimates", "pubDate": "2026-01-02T21:00:00Z",
            "provider": {"displayName": "Reuters"}}}])})
        got = live.recent_headlines("A")
        assert got.loc[0, "title"] == "A beats estimates"
        assert got.loc[0, "publisher"] == "Reuters"

    def test_no_news_gives_an_empty_frame_not_an_error(self, stub):
        stub({"A": _Ticker(news=[])})
        assert live.recent_headlines("A").empty
