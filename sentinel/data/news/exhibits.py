"""The text of an event, fetched from the filing that announced it.

An 8-K reporting results carries the earnings press release as an exhibit, and
that exhibit is the closest thing to a point-in-time headline that exists for
free. It is the company's own words, published at the accession timestamp, never
revised. Where a news archive can quietly rewrite a headline years later, an
exhibit cannot: the accession is immutable and the SEC serves the original bytes.

Only the opening of each document is kept. A press release front-loads its
news -- the title and first paragraph carry the beat-or-miss and the guidance,
and the remainder is tables of segment revenue that a sentence classifier reads
as neutral noise. Truncating also keeps the download to something a rate-limited
API can serve in minutes rather than hours, and FinBERT's 512-token window would
discard the tail regardless.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

from sentinel.data.news.edgar import USER_AGENT, CACHE_DIR

#: Bytes of each exhibit to retrieve. Enough for a title and several paragraphs
#: once HTML markup is stripped, which is where a press release puts its news.
FETCH_BYTES = 120_000

#: Characters of extracted text to keep per document.
KEEP_CHARS = 3_000

REQUEST_DELAY_SECONDS = 0.12

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
#: Exhibit 99 is the conventional slot for a press release, but issuers name the
#: file freely -- "q226earningsrelease.htm", "a2026q3ex991-pressrelease.htm" and
#: "q2fy27pr.htm" are all the same thing. The alternatives below were derived by
#: listing what the tech universe actually files rather than guessed at.
_PRESS_RELEASE = re.compile(
    r"(ex.?99|press.?rel|earnings|[-_]?pr\.(htm|html|txt)$|release)", re.IGNORECASE
)

#: Files that are never the press release, used to filter the fallback scan.
_NOT_TEXT = re.compile(r"(index|filingsummary|metalinks|\.xsd|_lab|_pre|_cal|_def|"
                       r"^r\d+\.htm|report\.css|show\.js|\.xml|\.zip|\.jpg|\.gif)",
                       re.IGNORECASE)


def _strip_html(raw: str) -> str:
    text = re.sub(r"(?is)<(script|style|table)\b.*?</\1>", " ", raw)
    text = _TAG.sub(" ", text)
    for entity, char in (("&nbsp;", " "), ("&amp;", "&"), ("&#8217;", "'"), ("&quot;", '"'),
                         ("&#146;", "'"), ("&rsquo;", "'"), ("&#39;", "'"), ("&lt;", "<")):
        text = text.replace(entity, char)
    return _WS.sub(" ", text).strip()


def _get(url: str, byte_limit: int | None = None) -> str:
    time.sleep(REQUEST_DELAY_SECONDS)
    headers = {"User-Agent": USER_AGENT}
    if byte_limit:
        headers["Range"] = f"bytes=0-{byte_limit}"
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


def press_release_text(cik: str, accession: str) -> str | None:
    """The opening text of a filing's press release exhibit, or None if it has none.

    Returning None rather than falling back to the main document matters. The
    primary 8-K document is usually three sentences of boilerplate incorporating
    the exhibit by reference, and scoring that as sentiment produces a confident
    neutral for every filing -- a signal that looks well-behaved and is empty.
    """
    folder = accession.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{folder}"
    try:
        listing = json.loads(_get(f"{base}/index.json"))
    except Exception:
        return None

    names = [item["name"] for item in listing.get("directory", {}).get("item", [])]
    htm = [n for n in names if n.lower().endswith((".htm", ".html", ".txt"))]
    candidates = [n for n in htm if _PRESS_RELEASE.search(n)]
    if not candidates:
        # Fall back to the largest remaining document. The press release is
        # reliably the longest prose file in an earnings 8-K, and the primary
        # document that would otherwise win is excluded by name above.
        sizes = {i["name"]: int(i.get("size") or 0)
                 for i in listing.get("directory", {}).get("item", [])}
        rest = [n for n in htm if not _NOT_TEXT.search(n) and n != accession + ".txt"]
        if not rest:
            return None
        candidates = [max(rest, key=lambda n: sizes.get(n, 0))]

    try:
        raw = _get(f"{base}/{candidates[0]}", byte_limit=FETCH_BYTES)
    except Exception:
        return None
    return _strip_html(raw)[:KEEP_CHARS] or None


def collect(events: pd.DataFrame, ciks: dict[str, str], cache_name: str) -> pd.DataFrame:
    """Fetch text for each event, resuming from cache so a stall costs nothing.

    Args:
        events: rows needing `ticker` and `accession`.
        cache_name: parquet file under the EDGAR cache to accumulate into.
    """
    path = CACHE_DIR / cache_name
    done: dict[str, str] = {}
    if path.exists():
        prior = pd.read_parquet(path)
        done = dict(zip(prior["accession"], prior["text"]))

    rows = []
    for n, (_, event) in enumerate(events.iterrows(), 1):
        accession = event["accession"]
        if accession in done:
            text = done[accession]
        else:
            text = press_release_text(ciks[event["ticker"]], accession)
            if n % 50 == 0:
                # Checkpoint everything seen so far, cached and new alike, so an
                # interrupted run resumes instead of restarting a 1400-request job.
                pd.DataFrame(rows).to_parquet(path)
                found = sum(r["text"] is not None for r in rows)
                print(f"  {n}/{len(events)} processed, {found} with text", flush=True)
        rows.append({"ticker": event["ticker"], "accession": accession,
                     "accepted": event["accepted"], "text": text})

    table = pd.DataFrame(rows)
    table.to_parquet(path)
    return table
