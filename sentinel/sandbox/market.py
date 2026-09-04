"""The single interface every market source presents.

The most important rule in this project lives here: **the AI must not be able to
tell whether it is looking at a synthetic market, real history, or a live feed.**

That is why `MarketData` carries prices and nothing else. A generated market knows
its own true parameters -- the drift it was built with, whether it contains a
signal, which regime it was in on each day -- but that knowledge is deliberately
kept out of `MarketData` and put in a separate `GroundTruth` object that only the
evaluation harness receives.

Keeping them in separate objects means leaking the answer into the model is not a
matter of remembering not to. There is no field to leak it through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketData:
    """Prices, and only prices. This is all the AI ever sees.

    Attributes:
        prices: rows are dates, columns are tickers. Total-return adjusted, so
            dividends and splits are already baked in -- see DECISIONS.md for why
            raw prices would quietly bias every backtest downward.
        name: a neutral label for reporting. Never says "synthetic" or "real";
            it exists for tearsheet titles, not for the model to condition on.
    """

    prices: pd.DataFrame
    name: str = "unnamed"

    def __post_init__(self) -> None:
        if not isinstance(self.prices.index, pd.DatetimeIndex):
            raise TypeError("prices must be indexed by dates")
        if not self.prices.index.is_monotonic_increasing:
            raise ValueError("prices must be sorted oldest to newest")
        if (self.prices <= 0).to_numpy().any():
            raise ValueError("prices must be positive (log returns are undefined at zero)")

    @property
    def tickers(self) -> list[str]:
        return list(self.prices.columns)

    @property
    def n_steps(self) -> int:
        """Number of price observations (one more than the number of returns)."""
        return len(self.prices)

    def log_returns(self) -> pd.DataFrame:
        """Log returns, log(P_t / P_t-1). One row shorter than `prices`.

        Log returns are used throughout because they add over time, which makes
        the statistical tests in `sentinel.stats` straightforward.
        """
        return np.log(self.prices).diff().dropna()

    def simple_returns(self) -> pd.DataFrame:
        """Percent returns. Used for portfolio arithmetic, where these add across
        assets (log returns do not)."""
        return self.prices.pct_change().dropna()


@dataclass(frozen=True)
class GroundTruth:
    """What a generated market is really made of. Never shown to the AI.

    Attributes:
        model: which generator produced this, e.g. "gbm".
        params: the exact parameters it was built with.
        has_exploitable_signal: whether any pattern exists that a strategy could
            profit from. This is the field the Null Test asserts against -- when
            it is False, an AI that makes money has found a bug in our code, not
            an edge in the market.
        regimes: for regime-switching models, the true state on each day. Lets us
            score a regime classifier directly against the right answer, which is
            impossible on real data where nobody knows the true label.
    """

    model: str
    params: dict[str, Any]
    has_exploitable_signal: bool
    regimes: np.ndarray | None = None


@dataclass(frozen=True)
class Scenario:
    """A generated market plus its answer key, held apart.

    `data` goes to the AI. `truth` goes to the evaluation harness. Nothing in the
    modelling code should ever accept a `Scenario`; it should accept `MarketData`.
    """

    data: MarketData
    truth: GroundTruth
    metadata: dict[str, Any] = field(default_factory=dict)
