"""Combining a trend signal with volatility sizing.

The two answer genuinely different questions, which is the only reason combining
them is worth trying rather than being another parameter to tune:

    Trend:      *whether* to be exposed at all.
    Volatility: *how much*, given that you are.

Neither can substitute for the other. Trend following says nothing about size, so
it holds a full position into a calm market and an identical full position into a
turbulent one. Volatility targeting says nothing about direction, so it holds a
carefully sized position all the way down a bear market.

Measured separately across eight national indices, each helped drawdown and
neither produced a return edge distinguishable from luck. The prediction
registered in DECISIONS.md before this was written is that the combination
improves drawdown more than return, consistent with everything else here.

The honesty constraint
----------------------
Both components have already been measured on this data, so any combination
chosen now is informed by results already seen. That is unavoidable, and the
response is not to pretend otherwise but to fix the form in advance and test it
once. The multiplication below is the simplest composition there is, has no free
parameters of its own, and was written before it was run. No variant of it will
be tried and reported afterwards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.sandbox.market import MarketData
from sentinel.strategies.base import Strategy
from sentinel.strategies.baseline import AbsoluteMomentum
from sentinel.strategies.volatility import VolatilityTarget


class TrendScaledVolatility(Strategy):
    """Hold `trend x volatility-target`, capped at fully invested.

    The trend component contributes a number between 0 and 1 — for
    `EnsembleMomentum` that is the fraction of horizons currently positive, and
    for `AbsoluteMomentum` it is simply 0 or 1. The volatility component
    contributes `target / forecast`, also capped at 1. Their product is the
    position.

    The multiplication is deliberate rather than an average. Either component
    saying "no" should be able to shrink the position on its own: a falling
    market is a reason to stand aside however calm it is, and a violent market is
    a reason to hold less however strong the trend. An average would let one
    override the other, which is the opposite of what a defensive system wants.

    Args:
        trend: any strategy returning weights in [0, 1]. Its opinion is read as
            a scaling factor, not as a position.
        volatility: the sizing rule. Its own no-trade band is bypassed, since the
            band is applied once here to the combined target -- applying it twice
            would produce a strategy whose turnover depends on the order the two
            components happened to be composed in.
        band: no-trade band on the final position.
    """

    name = "trend_scaled_volatility"

    def __init__(
        self,
        trend: Strategy | None = None,
        volatility: VolatilityTarget | None = None,
        band: float = 0.10,
    ) -> None:
        if not 0.0 <= band < 1.0:
            raise ValueError("band must be in [0, 1)")
        self.trend = trend or AbsoluteMomentum()
        self.volatility = volatility or VolatilityTarget(band=0.0)
        self.band = float(band)

    def compute_weights(self, data: MarketData) -> pd.DataFrame:
        trend_weights = self.trend.compute_weights(data).to_numpy(dtype=float)
        volatility_weights = self.volatility.compute_weights(data).to_numpy(dtype=float)

        # Both components produce cash during their own warmups, and cash is the
        # correct answer for "not enough is known yet". Multiplying preserves
        # that: the combination is invested only once both have a view.
        desired = np.nan_to_num(trend_weights, nan=0.0) * np.nan_to_num(
            volatility_weights, nan=0.0
        )
        desired = np.clip(desired, 0.0, 1.0)

        applied = np.zeros_like(desired)
        held = np.zeros(desired.shape[1])
        for t in range(len(desired)):
            if np.abs(desired[t] - held).sum() > self.band:
                held = desired[t].copy()
            applied[t] = held

        return pd.DataFrame(applied, index=data.prices.index, columns=data.tickers)
