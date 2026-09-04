"""Strategies that cheat, for measuring ceilings.

These live in `evaluation/` and not in `strategies/` on purpose. They are handed
the generator's answer key directly, so they are not strategies at all -- they are
**instruments for measuring how much money a given market actually contains**.

The question they answer cannot be answered any other way, and it is the question
that should be asked before blaming a model for a poor result:

    If the classifier were perfect -- no lag, no false alarms, no estimation
    error -- how much would that be worth on this market?

If the answer is "not much", then a disappointing result is a fact about the
market's parameters rather than a failure of the model, and no amount of work on
the classifier will change it. Without this measurement it is impossible to tell
those two cases apart, and the natural response to a weak result is to tune the
model -- which `plan.md` section 10 identifies as how strategies die.

Never import these from `sentinel.strategies`, never run them on real data (there
is no answer key to run them with), and never quote their results as achievable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.sandbox.market import MarketData
from sentinel.strategies.base import Strategy


class RegimeOracle(Strategy):
    """Holds the market exactly when the true regime is calm. Impossible to build.

    The upper bound on any regime-timing strategy: perfect knowledge of the state
    governing the very next return, with no lag and no error.

    Note what it still cannot do. It knows the *state*, not the return, so it
    still loses money on unlucky calm periods and still sits out profitable
    stressed ones. It is the ceiling for regime timing specifically, not for
    clairvoyance -- which is the comparison that matters, since regime timing is
    what the project is actually attempting.

    Args:
        regimes: the true state per return, from `GroundTruth.regimes`. Length is
            one less than the number of prices, matching the return series.
    """

    name = "regime_oracle"

    def __init__(self, regimes: np.ndarray, calm_state: int = 0) -> None:
        self.regimes = np.asarray(regimes, dtype=int).ravel()
        self.calm_state = int(calm_state)

    def compute_weights(self, data: MarketData) -> pd.DataFrame:
        n = data.n_steps
        if len(self.regimes) != n - 1:
            raise ValueError(
                f"got {len(self.regimes)} regime labels for {n} prices; "
                f"expected {n - 1}, one per return"
            )
        # regimes[t] governs the return from t to t+1, which is exactly the
        # period row t's weight is exposed to. The final row decides a period
        # outside the data and is never applied.
        weights = np.zeros(n)
        weights[:-1] = (self.regimes == self.calm_state).astype(float)
        return pd.DataFrame({data.tickers[0]: weights}, index=data.prices.index)


class DelayedRegimeOracle(RegimeOracle):
    """A perfect classifier that reacts `lag` days late. Everything else exact.

    Isolates the cost of detection lag from every other source of error. The
    classifier measured on the sandbox has a median lag of about four days, and
    comparing this at lag 4 against the instant oracle prices that delay directly
    -- which tells you whether working on lag is worth anything before doing it.
    """

    name = "regime_oracle_delayed"

    def __init__(self, regimes: np.ndarray, lag: int = 5, calm_state: int = 0) -> None:
        super().__init__(regimes, calm_state=calm_state)
        if lag < 0:
            raise ValueError("lag cannot be negative")
        self.lag = int(lag)
        self.name = f"regime_oracle_lag{lag}"

    def compute_weights(self, data: MarketData) -> pd.DataFrame:
        weights = super().compute_weights(data)
        if self.lag == 0:
            return weights
        # Shift the decisions forward in time: the state of day t-lag drives the
        # position on day t. Warmup rows default to cash.
        return weights.shift(self.lag).fillna(0.0)
