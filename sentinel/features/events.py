"""Turning a list of timestamped events into something a strategy may read.

The output is a panel shaped exactly like `prices`: rows are trading dates,
columns are tickers. That shape is not cosmetic. It means an event feature can be
handed to the same backtest engine, scored by the same null test, and checked by
the same `check_causality` truncation test as any price feature, with no separate
code path that could quietly use a different timing rule.

Two features are built here and they answer different questions.

`event_flag` marks the days on which an event became actionable. It knows nothing
about content -- only that something happened. This is the feature for volatility
questions ("is this a day to be smaller?"), which need no view on direction.

`reaction` records how the stock moved when the market first got to price the
event in. It is the closest thing to a free surprise measure: no analyst
estimates, no vendor, just the move. Crucially it is stamped on the day the move
*completed*, so a strategy reading row *t* learns about a reaction that is already
over, and can only bet on what comes after it. That is the post-earnings drift
question, and it is the honest version of "trade the news".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.data.news.align import (
    DEFAULT_BUFFER_MINUTES,
    MARKET_CLOSE_HOUR,
    MARKET_TIMEZONE,
    tradable_dates,
)


def event_flag(
    events: pd.DataFrame,
    index: pd.DatetimeIndex,
    tickers: list[str],
    buffer_minutes: int = DEFAULT_BUFFER_MINUTES,
) -> pd.DataFrame:
    """A 0/1 panel: did an event become actionable for this ticker on this day?

    Args:
        events: needs columns `ticker` and `accepted` (timezone-aware).
    """
    panel = pd.DataFrame(0.0, index=index, columns=tickers)
    dates = tradable_dates(events["accepted"], index, buffer_minutes=buffer_minutes)
    live = events.assign(date=dates.to_numpy()).dropna(subset=["date"])
    for ticker, group in live.groupby("ticker"):
        if ticker in panel.columns:
            panel.loc[panel.index.isin(group["date"]), ticker] = 1.0
    return panel


def reaction(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    buffer_minutes: int = DEFAULT_BUFFER_MINUTES,
) -> pd.DataFrame:
    """The move the market made when it first priced each event in.

    Stamped on the day the reaction finished, so reading row *t* tells a strategy
    what already happened rather than what is about to. The value is a simple
    return over the window from the last close before the event was actionable to
    the close of the actionable day itself -- for a typical after-hours earnings
    release that is the overnight gap plus the next full session, which is where
    essentially all of the repricing occurs.

    Days with no event are `NaN`, not zero. Zero would mean "an event happened and
    the stock did not move", which is a different and much rarer thing, and
    averaging the two together understates the size of real reactions.
    """
    index = prices.index
    dates = tradable_dates(events["accepted"], index, buffer_minutes=buffer_minutes)
    live = events.assign(date=dates.to_numpy()).dropna(subset=["date"])

    position = pd.Series(np.arange(len(index)), index=index)
    panel = pd.DataFrame(np.nan, index=index, columns=prices.columns)

    for ticker, group in live.groupby("ticker"):
        if ticker not in panel.columns:
            continue
        rows = position.reindex(group["date"]).dropna().astype(int)
        rows = rows[rows > 0]
        if rows.empty:
            continue
        series = prices[ticker].to_numpy()
        moves = series[rows.to_numpy()] / series[rows.to_numpy() - 1] - 1.0
        panel.iloc[rows.to_numpy(), panel.columns.get_loc(ticker)] = moves
    return panel


def announcement_flag(
    events: pd.DataFrame,
    index: pd.DatetimeIndex,
    tickers: list[str],
) -> pd.DataFrame:
    """Mark the row whose forecast window contains the release itself.

    This is the feature for volatility, and it is a different row from
    `event_flag`, which is the feature for direction. Getting the two confused
    silently destroys both, so the distinction is worth stating plainly.

    An earnings release filed at 16:21 on day *X* is priced in the session that
    runs from *X*'s close to *X+1*'s close. Row *X* is the row that forecasts that
    session, so row *X* is what gets marked. By the close of *X+1* the repricing
    is finished -- measured across this universe, variance on *X+1* is back to
    ordinary -- which is why `event_flag`, marking *X+1*, is useful for asking
    what drifts afterwards and useless for asking what moves.

    Why marking *X* is not lookahead
    --------------------------------
    Row *X*'s forecast is formed at 16:00 and the filing arrives at 16:21, so the
    *content* is genuinely unavailable and nothing here reads it. Only the date is
    used, and earnings dates are published two to four weeks ahead by the company
    and carried by every broker and data feed -- `yfinance`'s `get_earnings_dates`
    returns them with the 16:00 timestamp attached. So a trader at *X*'s close
    knows a release is imminent and does not know what it says, which is exactly
    the information set this feature encodes.

    The residual cheat is small but real: companies occasionally move a date after
    announcing it, and using the filing that actually occurred quietly assumes
    they never did. `predicted_announcement_flag` removes even that assumption at
    the cost of precision, and the two together bracket the honest answer.
    """
    panel = pd.DataFrame(0.0, index=index, columns=tickers)
    # The close of each trading day, which is when a forecast for the following
    # session is formed. An event belongs to the last row whose close precedes it.
    closes = index + pd.Timedelta(hours=MARKET_CLOSE_HOUR)
    stamps = pd.DatetimeIndex(events["accepted"]).tz_convert(MARKET_TIMEZONE).tz_localize(None)
    rows = closes.searchsorted(stamps, side="left") - 1

    valid = (rows >= 0) & (rows < len(index))
    for ticker, group in pd.DataFrame(
        {"ticker": events["ticker"].to_numpy(), "row": rows, "valid": valid}
    ).groupby("ticker"):
        if ticker not in panel.columns:
            continue
        hits = group.loc[group["valid"], "row"].to_numpy()
        panel.iloc[hits, panel.columns.get_loc(ticker)] = 1.0
    return panel


#: Quarterly gaps outside this range are reporting-schedule changes, not cadence,
#: and including them drags the median estimate away from the true rhythm.
_MIN_GAP_DAYS, _MAX_GAP_DAYS = 45, 130

#: Past releases needed before a cadence estimate is formed.
_MIN_PRIOR_EVENTS = 8


def predicted_announcement_flag(
    events: pd.DataFrame,
    index: pd.DatetimeIndex,
    tickers: list[str],
    window: int = 3,
) -> pd.DataFrame:
    """`announcement_flag` with the date guessed from cadence rather than known.

    Companies report on a quarterly rhythm, so the next release can be estimated
    as the previous one plus the median gap so far -- using only filings that had
    already happened. Measured over this universe that lands within one day 56% of
    the time and within a week 84%, so a window is marked rather than a single day.

    This arm assumes no external calendar at all. It is strictly worse than
    knowing the announced date and it is strictly honest, which makes it the floor
    under any claim the known-date version produces.
    """
    panel = pd.DataFrame(0.0, index=index, columns=tickers)
    stamps = pd.DatetimeIndex(events["accepted"]).tz_convert(MARKET_TIMEZONE).tz_localize(None)
    frame = pd.DataFrame({"ticker": events["ticker"].to_numpy(), "day": stamps.normalize()})

    for ticker, group in frame.groupby("ticker"):
        if ticker not in panel.columns:
            continue
        days = group["day"].sort_values().reset_index(drop=True)
        column = panel.columns.get_loc(ticker)
        for i in range(_MIN_PRIOR_EVENTS, len(days)):
            gaps = days[:i].diff().dropna().dt.days
            gaps = gaps[(gaps > _MIN_GAP_DAYS) & (gaps < _MAX_GAP_DAYS)]
            if len(gaps) < 4:
                continue
            centre = days[i - 1] + pd.Timedelta(days=int(np.median(gaps)))
            lo = index.searchsorted(centre - pd.Timedelta(days=window), side="left")
            hi = index.searchsorted(centre + pd.Timedelta(days=window), side="right")
            if lo < len(index):
                panel.iloc[lo:min(hi, len(index)), column] = 1.0
    return panel


def event_reactions(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    buffer_minutes: int = DEFAULT_BUFFER_MINUTES,
) -> pd.DataFrame:
    """One row per event, carrying the move the market made when it repriced it.

    The panel form (`reaction`) is what a strategy reads. This per-event form is
    what an analysis reads, because it keeps every column the caller attached to
    the event -- a sentiment score, an item code -- beside the outcome, without
    the positional re-pairing that panel-to-event joins invite and get wrong.

    Events that fall outside the price index, or on its first row, are dropped:
    there is no prior close to measure a move against.
    """
    index = prices.index
    dates = tradable_dates(events["accepted"], index, buffer_minutes=buffer_minutes)
    frame = events.assign(date=dates.to_numpy()).dropna(subset=["date"])

    position = pd.Series(np.arange(len(index)), index=index)
    rows = position.reindex(frame["date"]).to_numpy()
    keep = np.isfinite(rows) & (rows > 0)
    frame = frame.loc[keep]
    rows = rows[keep].astype(int)

    moves = np.full(len(frame), np.nan)
    for ticker in frame["ticker"].unique():
        if ticker not in prices.columns:
            continue
        mask = (frame["ticker"] == ticker).to_numpy()
        series = prices[ticker].to_numpy()
        moves[mask] = series[rows[mask]] / series[rows[mask] - 1] - 1.0

    return frame.assign(move=moves).dropna(subset=["move"]).reset_index(drop=True)
