"""The strategy contract.

A strategy is a **pure function from price history to target weights**. Same
input, same output, no hidden state, no network calls, no clock.

That purity is what allows the identical code to run in a backtest and in live
trading. Most retail systems have subtly different code paths for the two, and
that gap is where undetected bugs live -- the backtest exercises one path and
production runs the other, so a discrepancy can survive indefinitely without
anything looking wrong.

The timing convention, which everything depends on
--------------------------------------------------
`compute_weights(data)` returns a frame indexed exactly like `data.prices`, where

    row t = the weights to hold from t until t + 1

So row t may use prices up to and including t, and never beyond. The engine
applies row t to the return earned between t and t + 1. Getting this off by one
day in the wrong direction produces a strategy that trades on tomorrow's news --
which looks like genius and is worth nothing.

Weights need not sum to 1. The remainder is cash, and holding cash is a real
position: it is where nearly all of this project's downside protection comes from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from sentinel.sandbox.market import MarketData


class Strategy(ABC):
    """Base class for anything that decides what to hold."""

    name: str = "unnamed"

    @abstractmethod
    def compute_weights(self, data: MarketData) -> pd.DataFrame:
        """Target weights, indexed like `data.prices`. Row t uses data up to t."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"
