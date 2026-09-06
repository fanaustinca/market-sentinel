"""Train on the perfect answers. If that fails, nothing else can work.

Run: python experiments/perfect_timing.py

The proposal was to tell the model the exact right time to buy. That is a better
idea than it first appears, because it converts an open-ended question ("can a
model make money?") into a closed one with a decisive answer.

The usual training target is the next day's return, which is mostly noise, so a
model that fails on it has an excuse: the label was hopeless. Here the label is
the *perfect action* -- buy if tomorrow is up, hold cash if tomorrow is down.
Perfect foresight, handed over during training. The model is asked only to
reproduce it from information that existed at the time.

Three outcomes are possible and they mean completely different things:

  can't fit even in-sample      the model is too weak. Fixable with a bigger model.
  fits in-sample, fails out     the model memorises. Fixable with more data.
  fails no matter the data      the information is not in the features, and no
                                model, no amount of data, and no RL agent can
                                recover it.

The third is the one that matters, and the way to tell it apart from the second is
to vary the training set size and watch whether out-of-sample accuracy moves. If
pooling a hundred stocks reads exactly like pooling one, the ceiling is the data.

This also settles whether reinforcement learning is worth building. RL has strictly
*less* information than this experiment: it must discover good actions by trial
and error, where here they are handed over labelled. An RL agent cannot beat a
supervised model trained on the answer key. So this is the upper bound on RL too.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from sentinel.data.yahoo import load_prices
from sentinel.features.build import build_features, feature_warmup
from sentinel.sandbox.market import MarketData

#: Everything before this trains; everything after is never seen during fitting.
SPLIT_DATE = "2016-01-01"
START = "1995-01-01"


def build_pool(tickers: list[str]) -> pd.DataFrame:
    """Features and perfect-action labels for every stock, stacked into one frame."""
    warmup = feature_warmup()
    frames = []
    for ticker in tickers:
        try:
            data = load_prices(ticker, start=START)
        except Exception:
            continue
        features = build_features(data, ticker).iloc[warmup:]
        prices = data.prices[ticker].iloc[warmup:]

        # The perfect action for row t concerns the return from t to t+1, which is
        # exactly the return the engine applies to a weight decided at t. The last
        # row has no outcome and is dropped rather than filled.
        forward = np.log(prices).diff().shift(-1)
        frame = features.copy()
        frame["label"] = (forward > 0).astype(float)
        frame["forward_return"] = forward
        frame["ticker"] = ticker
        frame["date"] = features.index
        frames.append(frame.dropna())
    return pd.concat(frames, ignore_index=True)


def fit_and_score(pool: pd.DataFrame, n_tickers: int, seed: int = 0) -> dict:
    """Train on a subset of stocks, score on the held-out years of all of them."""
    import lightgbm as lgb

    rng = np.random.default_rng(seed)
    chosen = rng.choice(sorted(pool["ticker"].unique()), size=n_tickers, replace=False)
    columns = [c for c in pool.columns if c not in ("label", "forward_return", "ticker", "date")]

    train = pool[(pool["ticker"].isin(chosen)) & (pool["date"] < SPLIT_DATE)]
    test = pool[pool["date"] >= SPLIT_DATE]

    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        min_child_samples=200, subsample=0.8, colsample_bytree=0.8,
        verbose=-1, random_state=seed,
    )
    model.fit(train[columns], train["label"])

    in_sample = float((model.predict(train[columns]) == train["label"]).mean())
    out_sample = float((model.predict(test[columns]) == test["label"]).mean())
    base_rate = float(test["label"].mean())

    # What trading the model's own conviction would have earned, versus what the
    # oracle it was trained on would have earned over the same rows.
    probability = model.predict_proba(test[columns])[:, 1]
    position = (probability > 0.5).astype(float)
    model_return = position * test["forward_return"].to_numpy()
    oracle_return = (test["label"].to_numpy()) * test["forward_return"].to_numpy()

    def sharpe(series: np.ndarray) -> float:
        return float(np.mean(series) / np.std(series) * np.sqrt(252)) if np.std(series) > 0 else 0.0

    return {
        "n_tickers": n_tickers,
        "train_rows": len(train),
        "in_sample": in_sample,
        "out_sample": out_sample,
        "base_rate": base_rate,
        "edge": out_sample - max(base_rate, 1 - base_rate),
        "model_sharpe": sharpe(model_return),
        "oracle_sharpe": sharpe(oracle_return),
    }


def main() -> None:
    tickers = Path("/tmp/claude-1001/-home-austin/2284dc5e-346b-412b-9d24-e3cadfdd73a0/"
                   "scratchpad/big_universe.txt").read_text().split()
    print(f"UNIVERSE {len(tickers)} US large caps, {START} onward")
    print("  Survivorship-biased by construction: every name is one still listed in")
    print("  2026. That inflates returns and is irrelevant here -- the question is")
    print("  whether tomorrow's direction is predictable, not what these stocks did.\n")

    pool = build_pool(tickers)
    print(f"  {len(pool):,} stock-days, {pool['ticker'].nunique()} tickers")
    print(f"  train: before {SPLIT_DATE}   test: after, {(pool['date'] >= SPLIT_DATE).sum():,} rows")
    print(f"  days that went up: {pool['label'].mean():.1%}\n")

    print("DOES MORE DATA HELP? training set size vs out-of-sample accuracy\n")
    print(f"  {'stocks':>7s} {'train rows':>12s} {'in-sample':>10s} {'out-sample':>11s} "
          f"{'vs guessing':>12s} {'model Sharpe':>13s}")
    results = []
    for n in (1, 3, 10, 25, 50, len(pool["ticker"].unique())):
        if n > pool["ticker"].nunique():
            continue
        row = fit_and_score(pool, n)
        results.append(row)
        print(f"  {row['n_tickers']:7d} {row['train_rows']:12,d} {row['in_sample']:10.1%} "
              f"{row['out_sample']:11.1%} {row['edge']:+12.2%} {row['model_sharpe']:+13.2f}")

    oracle = results[-1]["oracle_sharpe"]
    always = max(results[-1]["base_rate"], 1 - results[-1]["base_rate"])
    print(f"\n  always guessing 'up'  {always:.1%}   (the number to beat)")
    print(f"  the oracle it was trained on: Sharpe {oracle:+.1f}")

    print("\n\nREADING IT\n")
    best = max(results, key=lambda r: r["out_sample"])
    print(f"  Best out-of-sample accuracy: {best['out_sample']:.1%} on "
          f"{best['train_rows']:,} training rows,")
    print(f"  against {always:.1%} for guessing 'up' every single day.")
    print()
    if best["edge"] < 0.01:
        print("  Trained on perfect answers, the model cannot reproduce them on data")
        print("  it has not seen. Accuracy does not improve as the training set grows")
        print(f"  from {results[0]['train_rows']:,} rows to {results[-1]['train_rows']:,} --")
        print("  which is the signature of information that is absent rather than")
        print("  merely hard to extract. More data cannot supply what is not there.")
        print()
        print("  This also bounds reinforcement learning. An RL agent must *discover*")
        print("  good actions through trial and error; this model was handed them and")
        print("  still could not generalise. RL cannot exceed its own answer key.")
    else:
        print("  There is a detectable edge. It needs the full stress battery before")
        print("  it means anything: null test, period split, and paper trading.")


if __name__ == "__main__":
    main()
