"""Tests for event timing, which is the only thing that can invalidate the rest.

Every assertion here is about *which row* a piece of news lands on. That is the
whole game: a news feature off by one row in the permissive direction produces a
backtest that trades on information it could not have had, and the resulting
Sharpe looks like a discovery rather than a bug.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.data.news.align import tradable_dates
from sentinel.features.events import (
    announcement_flag,
    event_flag,
    predicted_announcement_flag,
    reaction,
)


@pytest.fixture
def week() -> pd.DatetimeIndex:
    """A Tue-Fri trading week plus the following Monday. Jan 1 2024 is a holiday."""
    return pd.DatetimeIndex(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    )


def _events(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [t for t, _ in pairs],
            "accepted": pd.to_datetime([s for _, s in pairs], utc=True),
        }
    )


class TestTradableDates:
    def test_morning_event_trades_same_day(self, week):
        got = tradable_dates(pd.to_datetime(["2024-01-03 14:30"], utc=True), week)
        assert got.iloc[0] == pd.Timestamp("2024-01-03")

    def test_after_close_event_rolls_to_next_day(self, week):
        # 21:21 UTC is 16:21 New York: the classic after-hours earnings release.
        got = tradable_dates(pd.to_datetime(["2024-01-03 21:21"], utc=True), week)
        assert got.iloc[0] == pd.Timestamp("2024-01-04")

    def test_event_inside_the_buffer_rolls_forward(self, week):
        # 20:50 UTC is 15:50 New York, inside the default 15-minute buffer.
        got = tradable_dates(pd.to_datetime(["2024-01-03 20:50"], utc=True), week)
        assert got.iloc[0] == pd.Timestamp("2024-01-04")

    def test_weekend_and_holiday_events_roll_to_the_next_session(self, week):
        got = tradable_dates(
            pd.to_datetime(["2024-01-06 16:00", "2024-01-01 15:00"], utc=True), week
        )
        assert list(got) == [pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-02")]

    def test_events_past_the_end_are_dropped_not_clamped(self, week):
        """Clamping would pile every future event onto the final row."""
        got = tradable_dates(pd.to_datetime(["2024-06-01 14:00"], utc=True), week)
        assert pd.isna(got.iloc[0])

    def test_widening_the_buffer_can_only_delay(self, week):
        stamps = pd.to_datetime(
            ["2024-01-03 14:00", "2024-01-03 18:00", "2024-01-04 15:00"], utc=True
        )
        tight = tradable_dates(stamps, week, buffer_minutes=0)
        loose = tradable_dates(stamps, week, buffer_minutes=240)
        assert (loose.to_numpy() >= tight.to_numpy()).all()

    def test_unsorted_index_is_refused(self):
        bad = pd.DatetimeIndex(["2024-01-05", "2024-01-02"])
        with pytest.raises(ValueError, match="sorted"):
            tradable_dates(pd.to_datetime(["2024-01-03"], utc=True), bad)


class TestAnnouncementFlag:
    """The volatility feature, which marks a *different* row from `event_flag`."""

    def test_after_hours_release_marks_the_day_it_was_filed(self, week):
        """Row X forecasts the session X->X+1, which is when the repricing happens."""
        panel = announcement_flag(_events([("A", "2024-01-03 21:21")]), week, ["A"])
        assert panel.loc["2024-01-03", "A"] == 1.0
        assert panel["A"].sum() == 1.0

    def test_premarket_release_marks_the_previous_close(self, week):
        """A 09:00 release is priced in the session that began at the prior close."""
        panel = announcement_flag(_events([("A", "2024-01-04 14:00")]), week, ["A"])
        assert panel.loc["2024-01-03", "A"] == 1.0

    def test_it_marks_the_row_before_the_one_event_flag_marks(self, week):
        """The two features exist to answer different questions and must differ."""
        events = _events([("A", "2024-01-03 21:21")])
        vol = announcement_flag(events, week, ["A"])
        direction = event_flag(events, week, ["A"])
        assert vol.loc["2024-01-03", "A"] == 1.0
        assert direction.loc["2024-01-04", "A"] == 1.0

    def test_unknown_tickers_are_ignored_not_crashed_on(self, week):
        panel = announcement_flag(
            _events([("A", "2024-01-03 21:21"), ("ZZZ", "2024-01-04 21:00")]), week, ["A"]
        )
        assert panel["A"].sum() == 1.0


class TestPredictedAnnouncementFlag:
    def test_it_uses_only_prior_events(self):
        """The estimate for event i must not move when later events change."""
        index = pd.bdate_range("2015-01-01", periods=1400)
        days = pd.to_datetime([f"{y}-{m:02d}-15" for y in range(2015, 2021) for m in (1, 4, 7, 10)])
        events = pd.DataFrame({"ticker": "A", "accepted": days.tz_localize("UTC")})

        full = predicted_announcement_flag(events, index, ["A"])
        truncated = predicted_announcement_flag(events.iloc[:-4], index, ["A"])

        # Where the shorter history has already made predictions, the longer
        # history must agree -- otherwise later events are informing earlier rows.
        marked = truncated["A"].to_numpy() > 0
        cut = np.flatnonzero(marked).max()
        assert (full["A"].to_numpy()[: cut + 1] == truncated["A"].to_numpy()[: cut + 1]).all()

    def test_nothing_is_predicted_before_enough_history(self):
        index = pd.bdate_range("2015-01-01", periods=1000)
        days = pd.to_datetime(
            [f"{y}-{m:02d}-15" for y in (2015, 2016, 2017) for m in (1, 4, 7, 10)]
        )
        events = pd.DataFrame({"ticker": "A", "accepted": days.tz_localize("UTC")})
        panel = predicted_announcement_flag(events, index, ["A"])

        marked = np.flatnonzero(panel["A"].to_numpy() > 0)
        assert len(marked) > 0, "twelve events should be enough to predict something"
        # The first prediction can only be made once eight releases have happened,
        # so nothing may be marked before the eighth.
        assert index[marked[0]] > days[7]

    def test_exactly_the_minimum_history_predicts_nothing(self):
        """The boundary is the interesting case: eight priors, zero predictions."""
        index = pd.bdate_range("2015-01-01", periods=800)
        days = pd.to_datetime([f"{y}-{m:02d}-15" for y in (2015, 2016) for m in (1, 4, 7, 10)])
        events = pd.DataFrame({"ticker": "A", "accepted": days.tz_localize("UTC")})
        panel = predicted_announcement_flag(events, index, ["A"])
        assert panel["A"].sum() == 0.0


class TestReaction:
    def test_reaction_is_stamped_on_the_day_the_move_completed(self, week):
        prices = pd.DataFrame({"A": [100.0, 100.0, 110.0, 110.0, 110.0]}, index=week)
        got = reaction(_events([("A", "2024-01-03 21:21")]), prices)
        # Released after the close of the 3rd; the market prices it on the 4th.
        assert got.loc["2024-01-04", "A"] == pytest.approx(0.10)

    def test_days_without_events_are_nan_not_zero(self, week):
        """Zero would mean 'an event happened and nothing moved', a different claim."""
        prices = pd.DataFrame({"A": [100.0, 101.0, 110.0, 111.0, 112.0]}, index=week)
        got = reaction(_events([("A", "2024-01-03 21:21")]), prices)
        assert got["A"].notna().sum() == 1
        assert got.loc["2024-01-02", "A"] != 0.0 or pd.isna(got.loc["2024-01-02", "A"])
