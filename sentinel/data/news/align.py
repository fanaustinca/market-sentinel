"""When a piece of news becomes tradable, which is the only question that matters.

Every news-driven backtest dies in the same place. A headline is stamped with the
day it happened, the strategy is given that day's row, and the result looks
extraordinary -- because a quarterly earnings release that crossed the wire at
4:21pm was handed to a model that traded the 4:00pm close.

The engine's convention (see `strategies/base.py`) is that row *t* holds from *t*
to *t+1* and may use information up to and including *t*. For a price that is
unambiguous: the close of *t* is known at *t*. For an event it is not, because
events carry a wall-clock time and the market carries a closing bell.

The rule implemented here
-------------------------
An event is usable on row *t* only if it was published strictly before the
execution moment of row *t* -- the closing bell of day *t*, less a safety buffer.
Anything later rolls forward to the next trading day in the index. Events on
weekends, holidays, and after-hours all roll the same way.

This is deliberately pessimistic. Real execution at the closing auction on the
same second the news lands is not achievable, so the buffer defaults to fifteen
minutes and can be widened to test how much of any result depends on speed. A
signal that survives a one-day lag is a different and much stronger claim than
one that needs to trade the closing bell, and `tradable_dates` makes the
difference a parameter rather than an assumption.
"""

from __future__ import annotations

import pandas as pd

#: The closing bell, in exchange-local time. US equity markets close at 16:00 ET.
MARKET_CLOSE_HOUR = 16
MARKET_TIMEZONE = "America/New_York"

#: Minutes before the close after which an event is treated as tomorrow's news.
#: Fifteen minutes is a guess at how long it takes a human or a batch job to see
#: a filing, score it, and place an order. It is not a measurement.
DEFAULT_BUFFER_MINUTES = 15


def tradable_dates(
    timestamps: pd.Series | pd.DatetimeIndex,
    index: pd.DatetimeIndex,
    buffer_minutes: int = DEFAULT_BUFFER_MINUTES,
) -> pd.Series:
    """Map event timestamps to the first index date whose close may act on them.

    Args:
        timestamps: when each event was published. Timezone-aware values are
            converted to exchange time; naive values are *assumed* to already be
            exchange time, which is the convention SEC acceptance times follow
            once converted.
        index: the trading calendar to snap onto, normally `data.prices.index`.
        buffer_minutes: how long before the close an event must arrive to be
            actionable that same day.

    Returns:
        A series parallel to `timestamps` holding the trading date each event
        may first influence. Events after the last date in `index` map to `NaT`
        and must be dropped by the caller rather than clamped -- clamping them
        onto the final row would concentrate all future news on one day.

    Raises:
        ValueError: if `index` is not sorted, since the roll-forward search
            depends on order.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex")
    if not index.is_monotonic_increasing:
        raise ValueError("index must be sorted oldest to newest")

    stamps = pd.DatetimeIndex(pd.Series(timestamps).values)
    if stamps.tz is not None:
        stamps = stamps.tz_convert(MARKET_TIMEZONE).tz_localize(None)

    # The moment on each event's own calendar day at which it would be too late
    # to act. Comparing the event to this threshold decides same-day or next-day.
    deadline = (
        stamps.normalize()
        + pd.Timedelta(hours=MARKET_CLOSE_HOUR)
        - pd.Timedelta(minutes=buffer_minutes)
    )
    late = stamps >= deadline

    # searchsorted with "left" finds the first index date >= the event's day, which
    # is the same day when it trades and the next trading day otherwise. Late
    # events use "right" instead, pushing them past their own day.
    day = stamps.normalize()
    idx = index.normalize()
    positions = idx.searchsorted(day, side="left")
    positions_late = idx.searchsorted(day, side="right")
    positions = pd.Series(positions).where(~late, pd.Series(positions_late))

    out_of_range = positions >= len(index)
    positions = positions.where(~out_of_range, 0)
    result = pd.Series(index[positions.to_numpy()], index=pd.Series(timestamps).index)
    return result.where(~out_of_range.to_numpy(), pd.NaT)
