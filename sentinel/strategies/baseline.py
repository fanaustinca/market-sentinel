"""The control arm.

These are the strategies the AI has to beat. They are not a warm-up exercise or a
stepping stone -- they run in every experiment, permanently, beside whatever the
model produces.

The reason is simple and it is the heart of the project's method. A backtest
result on its own cannot be judged. "The AI returned 9% a year" is uninterpretable
until you know what a two-line rule returned on the same data over the same period
with the same costs. If a moving-average crossover matches a gradient-boosted
ensemble, the ensemble has learned nothing, and it is strictly worse than the rule
because it carries far more ways to break silently.

Every one of these is simple enough to verify by hand in a spreadsheet, which
matters more than it sounds: you should never run a system whose decisions you
cannot check manually.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.sandbox.market import MarketData
from sentinel.strategies.base import Strategy


class BuyAndHold(Strategy):
    """Own the market, equally weighted, and never trade again.

    The benchmark that matters most. If a system cannot beat this on a
    risk-adjusted basis, the honest response is to buy the index fund and delete
    the code -- and the plan explicitly permits that conclusion.
    """

    name = "buy_and_hold"

    def compute_weights(self, data: MarketData) -> pd.DataFrame:
        weight = 1.0 / len(data.tickers)
        return pd.DataFrame(weight, index=data.prices.index, columns=data.tickers)


class AlwaysCash(Strategy):
    """Hold nothing. The floor: zero return, zero risk, zero cost.

    Included because a strategy that fails to beat *this* is not merely
    unprofitable, it is actively destroying money, and that is worth seeing
    stated plainly in a results table.
    """

    name = "always_cash"

    def compute_weights(self, data: MarketData) -> pd.DataFrame:
        return pd.DataFrame(0.0, index=data.prices.index, columns=data.tickers)


class AbsoluteMomentum(Strategy):
    """Hold the asset while its trailing return is positive, otherwise hold cash.

    One parameter, one comparison, and the whole of the defensive behaviour the
    project is built around: the ability to step aside. In the sandbox's
    regime-switching markets this is what a well-functioning system should
    approximate, and in the null markets it should earn nothing.

    Args:
        lookback: trailing window in trading days. 252 is the conventional
            twelve-month horizon.
        rebalance_days: how often the rule is allowed to act. Checking daily
            produces whipsaw and heavy costs; monthly is the usual compromise.
    """

    name = "absolute_momentum"

    def __init__(self, lookback: int = 252, rebalance_days: int = 21) -> None:
        if lookback < 2:
            raise ValueError("lookback must be at least 2 days")
        if rebalance_days < 1:
            raise ValueError("rebalance_days must be at least 1")
        self.lookback = int(lookback)
        self.rebalance_days = int(rebalance_days)

    def compute_weights(self, data: MarketData) -> pd.DataFrame:
        prices = data.prices
        # `diff(lookback)` at row t compares today with `lookback` days ago. Both
        # are in the past, so this is causal.
        trailing = np.log(prices).diff(self.lookback)
        desired = (trailing > 0).astype(float) / len(data.tickers)

        # Act only on rebalance dates, holding the previous decision in between.
        # This is what keeps turnover -- and therefore cost -- under control.
        mask = np.zeros(len(prices), dtype=bool)
        mask[:: self.rebalance_days] = True
        desired = desired.where(pd.Series(mask, index=prices.index), other=np.nan).ffill()

        return desired.fillna(0.0)


class DualMomentum(Strategy):
    """Absolute momentum plus a choice of which asset to own.

    First the defensive question: has the best-performing asset actually made
    money over the lookback? If not, hold cash. Only then the selection question:
    among the assets, which has done best?

    The order matters. The absolute test is what caps drawdown, and a version that
    only ranked assets against each other would always be fully invested -- owning
    the least-bad thing all the way down.
    """

    name = "dual_momentum"

    def __init__(self, lookback: int = 252, rebalance_days: int = 21) -> None:
        self.lookback = int(lookback)
        self.rebalance_days = int(rebalance_days)

    def compute_weights(self, data: MarketData) -> pd.DataFrame:
        prices = data.prices
        trailing = np.log(prices).diff(self.lookback)

        weights = pd.DataFrame(0.0, index=prices.index, columns=data.tickers)

        # Rows inside the warmup have no trailing return for any asset yet.
        # Asking for the best of nothing raises, so they are skipped entirely and
        # left holding cash -- which is the right default for "we do not know".
        ready = trailing.notna().all(axis=1)
        known = trailing[ready]
        if not known.empty:
            best = known.idxmax(axis=1)
            for date in known.index[known.max(axis=1) > 0]:
                weights.loc[date, best[date]] = 1.0

        mask = np.zeros(len(prices), dtype=bool)
        mask[:: self.rebalance_days] = True
        weights = weights.where(pd.Series(mask, index=prices.index), other=np.nan).ffill()
        return weights.fillna(0.0)
