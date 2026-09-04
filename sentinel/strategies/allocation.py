"""Multi-asset allocation: what to hold instead of the market, not just when to leave it.

Every strategy so far answers one question -- own the market, or hold cash. That
is the honest single-asset version of defence and it has a real cost: cash earns
nothing, so a system that spends a third of its life in cash gives up a third of
its compounding to avoid a fraction of the drawdowns.

The plan's actual design allocates *across* assets. When equities are stressed,
the alternative to cash is a defensive asset -- treasuries, gold -- that is not
merely uncorrelated but has historically risen during equity crises.

The trap in that sentence, stated plainly
-----------------------------------------
"Has historically risen during equity crises" is a fact about 2000, 2008 and 2020,
and it is exactly the kind of fact that gets fitted to without anyone deciding to
fit it. Treasuries fell *with* equities through 2022, and a system built on the
assumption that they would not was wrong about the most recent bear market it
faced.

So the defensive sleeve is a declared parameter rather than a discovery, and the
2022 result is reported alongside the others rather than averaged into them. Any
version of this strategy that looks good only because it skips 2022 has not been
tested; it has been selected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.ai.regime.classifier import WalkForwardRegimeClassifier
from sentinel.sandbox.market import MarketData
from sentinel.strategies.base import Strategy


class RegimeRotation(Strategy):
    """Hold risk assets while the market is calm, defensive assets while it is not.

    The regime is read from the risk asset alone, because "is the market
    stressed" is a question about equities. Averaging the state across an
    equity/bond panel would blur the signal with the behaviour of the very assets
    the strategy rotates *into*.

    Args:
        risk_assets: tickers held when calm. Split equally, or selected by
            trailing momentum when `select_by_momentum` is set.
        defensive_assets: tickers held when stressed. Empty means cash, which
            makes this a multi-asset version of `RegimeGate` and is the right
            control to compare the rotation against.
        regime_ticker: which asset the regime is read from. Defaults to the first
            risk asset.
        band: no-trade band, as in `RegimeAwareStrategy`.
        select_by_momentum: among the risk assets, hold only the best performer
            over `lookback` days rather than an equal split.
    """

    name = "regime_rotation"

    def __init__(
        self,
        risk_assets: list[str],
        defensive_assets: list[str] | None = None,
        regime_ticker: str | None = None,
        classifier: WalkForwardRegimeClassifier | None = None,
        band: float = 0.15,
        select_by_momentum: bool = False,
        lookback: int = 126,
    ) -> None:
        if not risk_assets:
            raise ValueError("need at least one risk asset")
        if not 0.0 <= band < 1.0:
            raise ValueError("band must be in [0, 1)")
        self.risk_assets = list(risk_assets)
        self.defensive_assets = list(defensive_assets or [])
        self.regime_ticker = regime_ticker or self.risk_assets[0]
        self.classifier = classifier or WalkForwardRegimeClassifier()
        self.band = float(band)
        self.select_by_momentum = bool(select_by_momentum)
        self.lookback = int(lookback)

        # The name encodes the configuration, because every report keys its
        # results by strategy name and several rotations differing only in their
        # sleeves are exactly the comparison worth running. With a shared class
        # name they collapse into one row and the last one silently wins -- which
        # is not a crash, it is a table that looks fine and is wrong.
        risk = "+".join(self.risk_assets)
        defensive = "+".join(self.defensive_assets) if self.defensive_assets else "cash"
        self.name = f"rotate_{risk}_to_{defensive}" + ("_mom" if self.select_by_momentum else "")

    def compute_weights(self, data: MarketData) -> pd.DataFrame:
        tickers = data.tickers
        missing = [t for t in self.risk_assets + self.defensive_assets if t not in tickers]
        if missing:
            raise ValueError(f"{missing} not in the market's tickers {tickers}")
        if self.regime_ticker not in tickers:
            raise ValueError(f"regime ticker {self.regime_ticker} not in {tickers}")

        calm = self.classifier.probabilities(data, ticker=self.regime_ticker)[
            "p_calm"
        ].to_numpy(dtype=float)

        risk_target = self._sleeve_weights(data, self.risk_assets)
        defensive_target = self._sleeve_weights(data, self.defensive_assets)

        n, n_assets = len(calm), len(tickers)
        desired = np.zeros((n, n_assets))
        known = np.isfinite(calm)

        # A blend rather than a switch: at 70% calm the portfolio is 70% risk and
        # 30% defensive. This is the same "uncertainty shrinks positions" rule as
        # the single-asset version -- an unsure model ends up near half and half,
        # which is the portfolio that regrets least whichever state turns out to
        # hold.
        weight = np.where(known, calm, 0.0)[:, None]
        desired = weight * risk_target + (1.0 - weight) * defensive_target
        desired[~known] = 0.0  # warmup: cash, because nothing is known yet

        applied = np.zeros_like(desired)
        held = np.zeros(n_assets)
        for t in range(n):
            if np.abs(desired[t] - held).sum() > self.band:
                held = desired[t].copy()
            applied[t] = held

        return pd.DataFrame(applied, index=data.prices.index, columns=tickers)

    def _sleeve_weights(self, data: MarketData, sleeve: list[str]) -> np.ndarray:
        """Target weights within one sleeve, as a (n_rows, n_assets) array.

        An empty sleeve is all cash -- a row of zeros, which is a real position
        and the source of most of this project's downside protection.
        """
        tickers = data.tickers
        rows, n_assets = len(data.prices), len(tickers)
        target = np.zeros((rows, n_assets))
        if not sleeve:
            return target

        columns = [tickers.index(t) for t in sleeve]

        if not self.select_by_momentum or len(sleeve) == 1:
            target[:, columns] = 1.0 / len(sleeve)
            return target

        # Trailing return over the lookback, comparing today with the past. Both
        # ends are in the past, so this is causal.
        trailing = np.log(data.prices[sleeve]).diff(self.lookback).to_numpy(dtype=float)
        usable = np.isfinite(trailing).all(axis=1)
        best = np.argmax(np.where(usable[:, None], trailing, -np.inf), axis=1)

        for row in np.flatnonzero(usable):
            target[row, columns[best[row]]] = 1.0
        # Before the lookback has filled, split the sleeve equally rather than
        # guessing which member is best from incomplete data.
        for row in np.flatnonzero(~usable):
            target[row, columns] = 1.0 / len(sleeve)
        return target
