"""Live event data: what a running system can actually fetch today.

The backtest above runs on SEC filings, which are perfect for history and useless
for the future -- a filing exists only once the event has happened. A live system
needs the opposite: the *schedule*, ahead of time. This module is the answer to
"where do the events come from when the system is actually running", and the
finding it serves is narrow enough that the answer is genuinely easy.

What the result needs, and what each source provides
----------------------------------------------------
The volatility result needs one thing per stock: the date of the next earnings
release. Not the content, not a sentiment score, not a news feed -- FinBERT over
1,423 press releases called direction at 50.6%, a coin flip, so the text is
measured to be worth nothing here. Only the date matters, and the date is
published weeks ahead.

`upcoming_earnings` reads it from Yahoo, free and without a key. Yahoo carries
the company-announced date with a 16:00 timestamp attached, which is the same
after-the-close convention the historical filings follow.

`recent_headlines` is included because headlines are what people ask for, and
because it is the honest place to record what they cost. Yahoo serves the last
few days per ticker and keeps no archive, so headlines can drive a live system
but can never be backtested from this source -- and a signal that cannot be
backtested cannot be trusted with money. GDELT has an archive back to 2015 and
throttles hard enough to be impractical from a single machine. Paid vendors
(Benzinga via Alpaca, Finnhub) sell archives from about 2015 and are the route
worth paying for *if* a text signal ever justifies the cost. This project's own
measurement says it does not.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


def upcoming_earnings(tickers: list[str]) -> pd.DataFrame:
    """The next announced earnings date for each ticker, as published by Yahoo.

    Returns:
        Columns `ticker`, `date`, `days_away`, sorted soonest first. Tickers with
        no published date are omitted rather than given a guess -- the cadence
        estimator exists for that and is measured to be too imprecise to help.
    """
    import yfinance as yf

    now = pd.Timestamp.now(tz="UTC")
    rows = []
    for ticker in tickers:
        try:
            dates = yf.Ticker(ticker).get_earnings_dates(limit=12)
        except Exception:
            continue
        if dates is None or dates.empty:
            continue
        future = dates.index[dates.index > now]
        if len(future) == 0:
            continue
        when = future.min()
        rows.append({"ticker": ticker, "date": when,
                     "days_away": (when - now).days})

    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "days_away"])
    return pd.DataFrame(rows).sort_values("days_away").reset_index(drop=True)


def recent_headlines(ticker: str, limit: int = 10) -> pd.DataFrame:
    """Yahoo's current headlines for one ticker. Live only -- there is no archive.

    Provided for a live advisory pass, and deliberately not wired into any
    backtest: this source cannot reproduce what it served last week, so anything
    fitted to it would be unfalsifiable.
    """
    import yfinance as yf

    rows = []
    for item in (yf.Ticker(ticker).news or [])[:limit]:
        content = item.get("content", item)
        rows.append({
            "ticker": ticker,
            "title": content.get("title"),
            "published": content.get("pubDate") or content.get("displayTime"),
            "publisher": (content.get("provider") or {}).get("displayName"),
        })
    return pd.DataFrame(rows)
