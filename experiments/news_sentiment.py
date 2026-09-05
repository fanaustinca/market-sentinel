"""Does reading the earnings release help? Graded against the oracle ceiling.

Run: python experiments/news_sentiment.py

Three questions, in the order that makes the third one interpretable:

1. What could a *perfect* reader earn? An oracle that knows the sign of every
   earnings reaction, aligned to a day it could actually trade. This is the
   ceiling, and nothing that reads text can exceed it.
2. Where does FinBERT land against that ceiling, on direction?
3. Does FinBERT's *intensity* -- how strongly the release commits, in either
   direction -- say anything about how far the stock moves? Direction and
   magnitude have come apart everywhere else in this project, and there is no
   reason to expect news to be the exception.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel.ai.sentiment.finbert import FinBERTScorer
from sentinel.data.news.edgar import CACHE_DIR, has_item
from sentinel.data.news.universe import LONG_LIVED_TECH
from sentinel.data.yahoo import load_prices
from sentinel.features.events import event_reactions

START = "1999-01-25"
TICKERS = [t for t in LONG_LIVED_TECH if t != "DELL"]


def main() -> None:
    text = pd.read_parquet(CACHE_DIR / "tech_earnings_text.parquet")
    text = text[text["text"].notna() & text["ticker"].isin(TICKERS)].copy()
    print(f"press releases with text: {len(text)}")

    scorer = FinBERTScorer(cache_path=CACHE_DIR / "finbert_cache.parquet")
    scores = [scorer.score_document(t) for t in text["text"]]
    text = pd.concat([text.reset_index(drop=True), pd.DataFrame(scores)], axis=1)
    text = text[text["tone"].notna()]
    print(f"scored: {len(text)}   mean tone {text['tone'].mean():+.3f}   "
          f"mean intensity {text['intensity'].mean():.3f}")

    prices = load_prices(TICKERS, start=START).prices
    frame = event_reactions(text, prices)
    print(f"paired with the reaction that followed: {len(frame)}")

    print("\n" + "=" * 68)
    print("Q1  DIRECTION -- can the text predict which way the stock moved?")
    r, p = stats.pearsonr(frame["tone"], frame["move"])
    hit = (np.sign(frame["tone"]) == np.sign(frame["move"])).mean()
    print(f"  correlation(tone, reaction)  {r:+.4f}   p = {p:.4f}")
    print(f"  direction called correctly   {hit:.1%}   (a coin flip is 50%)")
    print(f"  oracle, same alignment       100%, and worth Sharpe +0.25")
    print(f"  -> FinBERT captures {max(0.0, 2 * hit - 1):.1%} of the oracle's edge,")
    print(f"     so at most Sharpe {max(0.0, 2 * hit - 1) * 0.25:+.3f} before costs.")

    print("\n" + "=" * 68)
    print("Q2  MAGNITUDE -- does a strongly-worded release move the stock further?")
    r2, p2 = stats.pearsonr(frame["intensity"], frame["move"].abs())
    print(f"  correlation(intensity, |reaction|)  {r2:+.4f}   p = {p2:.4f}")
    quartile = pd.qcut(frame["intensity"], 4, labels=["calmest", "2nd", "3rd", "strongest"])
    table = frame.groupby(quartile, observed=True)["move"].apply(lambda s: s.abs().mean())
    for label, value in table.items():
        print(f"    {label:10s} mean |move| {value:6.2%}")

    print("\n" + "=" * 68)
    print("READING IT")
    if p < 0.05 and hit > 0.55:
        print("  Tone carries some directional information.")
    else:
        print("  Tone does not predict direction at a useful level. That is the")
        print("  expected result: the release is public before anyone can trade it,")
        print("  and the oracle experiment showed the ceiling is +0.25 Sharpe even")
        print("  with perfect knowledge. A model reading the same text later cannot")
        print("  do better than the model that already knows the answer.")


if __name__ == "__main__":
    main()
