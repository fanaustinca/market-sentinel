"""Ranking assets against each other, which is a different bet from ranking one against itself.

Everything this project has tested so far is *time-series* momentum: is this asset
above its own trailing average? That is a question about one asset at a time, and
it was measured across eight countries at p = 0.145 and set aside.

Cross-sectional momentum asks something else -- of the things I could hold, which
have gone up the most relative to each other? -- and the two are not the same
effect. Time-series momentum is a market-timing signal that goes to cash in a
general decline. Cross-sectional momentum is a *relative* signal that stays fully
invested and merely rotates, so it is unaffected by whether the market as a whole
is rising. Jegadeesh and Titman (1993) is the cross-sectional result and it has a
stronger and longer replication record than the time-series version, which is why
it is worth a separate test rather than an assumption that it fails too.

Two implementation details carry most of the literature's weight and neither is
optional.

**Skip the most recent month.** A twelve-month ranking that includes the last
twenty-one days mixes momentum with short-term reversal, which points the other
way and partially cancels it. The convention is 12-1: rank on months 2 through 12.

**Rebalance monthly, not daily.** The effect operates on a horizon of months. Daily
reranking of a noisy signal produces turnover that costs more than the effect
pays, which this project has already seen happen with `short_momentum`.

`LowVolatility` is included as a second documented anomaly using machinery that
already exists. Low-volatility assets have historically delivered better returns
per unit of risk than high-volatility ones -- the opposite of what the textbook
predicts -- and testing it costs almost nothing given the forecasters here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.sandbox.market import MarketData
from sentinel.strategies.base import Strategy

#: Days skipped at the end of the ranking window, to keep short-term reversal out.
SKIP_DAYS = 21


class CrossSectionalMomentum(Strategy):
    """Hold the assets that have outperformed their peers, rotating monthly.

    Args:
        lookback: total ranking window in trading days. 252 with `skip=21` gives
            the conventional 12-1 formation period.
        skip: days at the end of the window to ignore.
        n_hold: how many of the ranked assets to hold. Equal-weighted among them.
            With a small universe this is the parameter that matters most: holding
            the top 3 of 6 is a real selection, holding the top 5 of 6 is not.
        long_only_positive: if True, an asset is held only when its own trailing
            return is also positive. This grafts the time-series filter onto the
            cross-sectional one; it is off by default so the two effects can be
            measured apart rather than confounded.
    """

    def __init__(
        self,
        lookback: int = 252,
        skip: int = SKIP_DAYS,
        n_hold: int = 3,
        rebalance_days: int = 21,
        long_only_positive: bool = False,
    ) -> None:
        if lookback <= skip:
            raise ValueError("lookback must exceed skip, or the window is empty")
        if n_hold < 1:
            raise ValueError("n_hold must be at least 1")
        if rebalance_days < 1:
            raise ValueError("rebalance_days must be at least 1")

        self.lookback = int(lookback)
        self.skip = int(skip)
        self.n_hold = int(n_hold)
        self.rebalance_days = int(rebalance_days)
        self.long_only_positive = bool(long_only_positive)
        self.name = f"xsec_mom_{lookback}_{skip}_top{n_hold}" + (
            "_pos" if long_only_positive else ""
        )

    def compute_weights(self, data: MarketData) -> pd.DataFrame:
        prices = data.prices
        logged = np.log(prices)

        # Return from t-lookback to t-skip. Both endpoints are in the past at row
        # t, so this is causal; shifting by `skip` is what removes the reversal
        # window rather than merely shortening the lookback.
        formation = logged.shift(self.skip).diff(self.lookback - self.skip)

        weights = np.zeros(prices.shape)
        current = np.zeros(prices.shape[1])
        last_rebalance: int | None = None

        for t in range(len(prices)):
            row = formation.iloc[t]
            if row.isna().any():
                continue

            due = last_rebalance is None or (t - last_rebalance) >= self.rebalance_days
            if due:
                order = np.argsort(row.to_numpy())[::-1][: self.n_hold]
                chosen = np.zeros(prices.shape[1])
                if self.long_only_positive:
                    order = [i for i in order if row.to_numpy()[i] > 0]
                if len(order) > 0:
                    chosen[list(order)] = 1.0 / len(order)
                current = chosen
                last_rebalance = t

            weights[t] = current

        return pd.DataFrame(weights, index=prices.index, columns=prices.columns)


class LowVolatility(Strategy):
    """Hold the calmest assets. A documented anomaly the textbook does not predict.

    Ranks on realised volatility over a trailing window and holds the quietest
    `n_hold`. This is distinct from `VolatilityTarget`, which sizes every asset by
    its own volatility but holds them all: here the high-volatility assets are
    *excluded*, which is the actual claim in the low-volatility literature.
    """

    def __init__(self, lookback: int = 252, n_hold: int = 3, rebalance_days: int = 21) -> None:
        if lookback < 21:
            raise ValueError("lookback must be at least 21 days")
        if n_hold < 1:
            raise ValueError("n_hold must be at least 1")
        self.lookback = int(lookback)
        self.n_hold = int(n_hold)
        self.rebalance_days = int(rebalance_days)
        self.name = f"low_vol_{lookback}_top{n_hold}"

    def compute_weights(self, data: MarketData) -> pd.DataFrame:
        prices = data.prices
        realised = np.log(prices).diff().rolling(self.lookback).std()

        weights = np.zeros(prices.shape)
        current = np.zeros(prices.shape[1])
        last_rebalance: int | None = None

        for t in range(len(prices)):
            row = realised.iloc[t]
            if row.isna().any():
                continue
            due = last_rebalance is None or (t - last_rebalance) >= self.rebalance_days
            if due:
                order = np.argsort(row.to_numpy())[: self.n_hold]
                chosen = np.zeros(prices.shape[1])
                chosen[list(order)] = 1.0 / len(order)
                current = chosen
                last_rebalance = t
            weights[t] = current

        return pd.DataFrame(weights, index=prices.index, columns=prices.columns)
