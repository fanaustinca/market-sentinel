"""Daily total-return prices for real ETFs, cached so results stay reproducible.

Three decisions here shape every number this project will ever report about real
markets, and each is a place where the standard easy choice quietly corrupts the
result.

**Total return, not price.** `auto_adjust=True` folds dividends and splits into
the price series. Raw closing prices ignore dividends, which for broad equity
ETFs is one and a half to two percent a year -- so a strategy that sits in cash
during a dividend-paying stretch is credited for return it never gave up, and
every timing strategy looks better than it is. On a twenty-year backtest the
error compounds to more than a third of terminal wealth.

**Cached to disk.** Yahoo restates adjusted history when corporate actions are
reprocessed, so the "same" download can differ between runs. Silently changing
input data would make every result irreproducible and every comparison between
two runs meaningless. The first download is written to parquet and reused, with
a recorded row count and checksum, so a change is visible rather than invisible.

**Failures are loud.** A missing ticker, a short history, or a gap is raised
rather than filled or dropped. Forward-filling a stale price makes a market look
calm on exactly the days it was closed or broken, which flatters volatility
estimates and understates drawdown.

What this cannot fix
--------------------
The universe is chosen today, from funds that exist today. Broad-market ETFs
rarely close, so survivorship bias is far smaller here than for single stocks,
but it is not zero and it is not measurable from inside this file. It belongs in
the caveats of any result, and it is one reason the plan restricts the universe
to broad index funds rather than sector or thematic funds, where closures are
common.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from sentinel.sandbox.market import MarketData

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data_cache"

#: A deliberately boring universe: broad, liquid, long-lived, and covering the
#: asset classes a defensive strategy needs to rotate between. Chosen for
#: inception date as much as for exposure -- a fund launched in 2015 cannot tell
#: you anything about 2008, which is the period that matters most for a system
#: whose entire purpose is not losing money in a crash.
DEFAULT_UNIVERSE = {
    "SPY": "US large cap (1993)",
    "IWM": "US small cap (2000)",
    "EFA": "developed international (2001)",
    "TLT": "long US treasuries (2002)",
    "IEF": "intermediate US treasuries (2002)",
    "GLD": "gold (2004)",
}

#: Longest continuous daily history available from one ticker in the universe.
LONGEST_HISTORY_TICKER = "SPY"


def _cache_path(ticker: str, start: str, end: str) -> Path:
    return CACHE_DIR / f"{ticker}_{start}_{end}.parquet"


def _download(ticker: str, start: str, end: str) -> pd.Series:
    import yfinance as yf

    frame = yf.download(
        ticker, start=start, end=end, auto_adjust=True, progress=False, actions=False
    )
    if frame is None or frame.empty:
        raise ValueError(f"no data returned for {ticker} between {start} and {end}")

    # yfinance returns a column MultiIndex when given a list and a flat one for a
    # single string, and has changed which it does between versions. Normalising
    # here means a library update cannot silently reindex the data.
    close = frame["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.astype(float)


def load_prices(
    tickers: list[str] | str,
    start: str = "1993-02-01",
    end: str = "2026-01-01",
    cache: bool = True,
    name: str | None = None,
) -> MarketData:
    """Total-return daily prices, as a `MarketData` indistinguishable from synthetic.

    Args:
        tickers: one ticker or several. Several are aligned on their shared
            trading days, which is an intersection rather than a union -- a
            forward-filled holiday shows up as a zero-return day and biases
            volatility downward on exactly the assets that were not trading.
        cache: reuse a previous download when one exists. Turning this off makes
            results depend on when they were run.

    Raises:
        ValueError: if a ticker is missing, empty, or the aligned frame has fewer
            than 252 rows -- a backtest on under a year of data cannot produce a
            number worth reading, as the noise-floor scaling makes explicit.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    if not tickers:
        raise ValueError("need at least one ticker")

    CACHE_DIR.mkdir(exist_ok=True)
    series = {}
    for ticker in tickers:
        path = _cache_path(ticker, start, end)
        if cache and path.exists():
            series[ticker] = pd.read_parquet(path)["close"]
        else:
            close = _download(ticker, start, end)
            close.name = "close"
            if cache:
                close.to_frame().to_parquet(path)
            series[ticker] = close

    frame = pd.concat(series, axis=1).dropna(how="any")
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
    frame.index.name = "date"
    frame = frame.sort_index()

    if len(frame) < 252:
        raise ValueError(
            f"only {len(frame)} shared trading days across {tickers}; "
            "the shortest history in the set is the binding constraint, and "
            "under a year cannot support a result"
        )
    if (frame <= 0).to_numpy().any():
        raise ValueError("non-positive prices in the downloaded data")

    return MarketData(prices=frame, name=name or "+".join(tickers))


def universe_history(
    tickers: list[str] | None = None, start: str = "1993-02-01", end: str = "2026-01-01"
) -> MarketData:
    """The default multi-asset universe, aligned to its shortest member.

    Note what alignment costs: adding GLD, which begins in 2004, truncates the
    whole panel to 2004 onward and throws away the 2000-2002 bear market. That
    trade-off is real and should be made deliberately -- for a system whose
    purpose is surviving crashes, two extra crashes in the sample is usually
    worth more than an extra asset to rotate into.
    """
    return load_prices(tickers or list(DEFAULT_UNIVERSE), start=start, end=end, name="universe")


def fingerprint(data: MarketData) -> str:
    """A short checksum of the price data, for pinning a result to its inputs.

    Reported beside every real-data result. If a rerun produces a different
    fingerprint, the data changed underneath -- which happens when Yahoo
    reprocesses corporate actions -- and the two results are not comparable.
    """
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(data.prices.round(6), index=True).values.tobytes()
    )
    return digest.hexdigest()[:12]
