"""Corporate events from SEC EDGAR, stamped to the second.

Why filings rather than headlines
---------------------------------
The obvious way to give a model "knowledge of world events" is a news feed. The
problem is not finding one, it is finding one whose archive is honest. Most news
APIs expose a rolling window of recent articles; the ones with deep archives
restate them, re-tag them, and silently drop dead links. A backtest built on a
feed that was assembled after the fact is measuring what a publisher chose to
keep, and there is no way to detect that from inside the backtest.

EDGAR has the properties a backtest actually needs:

* **An acceptance timestamp to the second.** `acceptanceDateTime` is when the SEC
  received the document, which is when it became public. Not the day, the second.
* **A complete archive back to 1993**, never restated -- accession numbers are
  immutable, and the filing you fetch today is byte-identical to the one filed.
* **Item codes**, so an earnings release can be separated from a bylaw amendment
  without reading a word of text.
* **Free, keyless, and permitted**, at ten requests a second with a declared
  user agent.

What it is not
--------------
It is not a news feed. It contains what companies were legally required to
disclose, which is a strict and somewhat arbitrary subset of what mattered. A
chip launch covered by every outlet on earth may generate no filing at all, and
the item that does get filed is often a dry contract amendment. So EDGAR is the
*point-in-time skeleton* -- exact dates for real corporate events -- and headline
text from elsewhere is the flesh. When the two disagree about whether something
happened, EDGAR is the one that can be trusted about *when*.

The most useful codes, and roughly what they mean:

    2.02  results of operations -- this is the earnings release
    1.01  entry into a material definitive agreement
    2.01  completion of an acquisition or disposition
    5.02  departure or election of directors and officers
    7.01  Regulation FD disclosure -- a deliberate public statement
    8.01  other events, the catch-all companies use for announcements
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path(__file__).resolve().parents[3] / "data_cache" / "edgar"

#: SEC requires a declared identity on every request and throttles at 10/second.
#: Exceeding it earns a block that lasts ten minutes, so the delay is generous.
USER_AGENT = "market-sentinel research fanaustinca@gmail.com"
REQUEST_DELAY_SECONDS = 0.15

#: 8-K item codes worth naming. Anything not listed still comes through; this is
#: for reporting, not filtering.
ITEM_MEANINGS = {
    "1.01": "material agreement",
    "2.01": "acquisition completed",
    "2.02": "earnings release",
    "5.02": "officer or director change",
    "7.01": "Regulation FD disclosure",
    "8.01": "other announcement",
}

#: The item that behaves most like a scheduled, market-moving event.
EARNINGS_ITEM = "2.02"


def _get(url: str) -> dict:
    time.sleep(REQUEST_DELAY_SECONDS)
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    return response.json()


def ticker_to_cik(tickers: list[str], cache: bool = True) -> dict[str, str]:
    """Resolve tickers to zero-padded SEC central index keys.

    Raises:
        KeyError: if a ticker has no CIK. This is loud on purpose -- a silently
            dropped ticker becomes a silently missing event stream, and the
            backtest that results looks fine.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / "company_tickers.json"
    if cache and path.exists():
        table = json.loads(path.read_text())
    else:
        table = _get("https://www.sec.gov/files/company_tickers.json")
        if cache:
            path.write_text(json.dumps(table))

    lookup = {row["ticker"].upper(): f"{int(row['cik_str']):010d}" for row in table.values()}
    missing = [t for t in tickers if t.upper() not in lookup]
    if missing:
        raise KeyError(f"no SEC CIK for {missing}; these cannot contribute events")
    return {t: lookup[t.upper()] for t in tickers}


def _submissions(cik: str, cache: bool = True) -> list[dict]:
    """Every filing for one company, recent page plus the older archive pages."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"submissions_{cik}.json"
    if cache and path.exists():
        return json.loads(path.read_text())

    root = _get(f"https://data.sec.gov/submissions/CIK{cik}.json")
    pages = [root["filings"]["recent"]]
    # Companies with more than 1000 filings spill into dated archive files. Missing
    # these silently truncates history to the last few years for exactly the large,
    # long-lived companies a tech backtest cares about most.
    for extra in root["filings"].get("files", []):
        pages.append(_get(f"https://data.sec.gov/submissions/{extra['name']}"))

    if cache:
        path.write_text(json.dumps(pages))
    return pages


def filings(
    tickers: list[str],
    forms: tuple[str, ...] = ("8-K",),
    cache: bool = True,
) -> pd.DataFrame:
    """Every matching filing for each ticker, with its acceptance timestamp.

    Returns:
        Columns `ticker`, `form`, `items`, `accepted` (UTC), `accession`, sorted
        by acceptance time. `accepted` is the field to align on -- `filingDate`
        is only a date and rounds an after-hours release back onto the day it
        could not have been traded.
    """
    ciks = ticker_to_cik(tickers, cache=cache)
    frames = []
    for ticker, cik in ciks.items():
        rows = []
        for page in _submissions(cik, cache=cache):
            for i, form in enumerate(page["form"]):
                if form in forms:
                    rows.append(
                        {
                            "ticker": ticker,
                            "form": form,
                            "items": page["items"][i],
                            "accepted": page["acceptanceDateTime"][i],
                            "accession": page["accessionNumber"][i],
                        }
                    )
        if not rows:
            raise ValueError(f"{ticker} returned no {forms} filings, which is implausible")
        frames.append(pd.DataFrame(rows))

    table = pd.concat(frames, ignore_index=True)
    table["accepted"] = pd.to_datetime(table["accepted"], utc=True, format="mixed")
    return table.sort_values("accepted").reset_index(drop=True)


def has_item(table: pd.DataFrame, item: str) -> pd.Series:
    """Whether each filing carries a given 8-K item code.

    Item strings are comma-separated and unpadded (`"2.02,9.01"`), so a substring
    test would match `"2.02"` inside `"12.02"`. Splitting avoids that.
    """
    return table["items"].fillna("").apply(lambda s: item in [p.strip() for p in s.split(",")])
