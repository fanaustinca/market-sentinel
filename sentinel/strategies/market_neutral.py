"""Long-short cross-sectional portfolios: how professional equity quants actually bet.

Every strategy in this project until now has made one bet per day -- how much of
the market to own -- and that bet's outcome is dominated by whether the market
went up. Being right 51% of the time on a single bet is worth nothing.

A market-neutral book asks a different question: of these hundred stocks, which
will beat which? It buys the top of the ranking and sells the bottom in equal
dollars, so a general rise or fall cancels out and what remains is the *spread*
between the two sides. That converts one noisy annual bet into hundreds of small
simultaneous ones, and 51% accuracy across hundreds of independent bets is a
business rather than a coin flip.

This is the structure the measurement in `experiments/perfect_timing.py` does not
rule out. That experiment showed the *direction of one stock tomorrow* is not
predictable from price history. Relative ranking is a different claim: it can hold
even when every individual forecast is hopeless, because the errors need only be
uncorrelated rather than small.

Two things it costs
-------------------
**Shorting.** This is the only strategy here that shorts, breaking a standing
policy, and it must be run with `RiskLimits(allow_shorting=True)` rather than the
project default. Shorts have unbounded loss, require a margin account, can be
recalled by the lender at the worst moment, and cost a borrow fee -- charged here
rather than ignored, since for hard-to-borrow names it can exceed the entire edge.

**Its own risk model.** Dollar-neutral is not risk-neutral. If the long side is
all technology and the short side all utilities, the book is a sector bet wearing
a hedge, and it will be discovered during the sector rotation that kills it.
`neutralise_by` addresses the crudest version of this.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.sandbox.market import MarketData
from sentinel.strategies.base import Strategy

#: Annual cost of borrowing shares to short, as a fraction. Large, liquid US
#: names are cheap to borrow; this is a fair standing estimate for the universe
#: here and would be far too low for small caps or crowded shorts.
DEFAULT_BORROW_COST = 0.005

SKIP_DAYS = 21


def momentum_signal(prices: pd.DataFrame, lookback: int = 252, skip: int = SKIP_DAYS) -> pd.DataFrame:
    """Trailing return from t-lookback to t-skip. Higher is stronger."""
    logged = np.log(prices)
    return logged.shift(skip).diff(lookback - skip)


def reversal_signal(prices: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    """Last month's return, negated: short-term losers tend to bounce.

    A documented effect that points the *opposite* way to momentum over a
    different horizon, which is why the two are worth holding together -- signals
    that fail in different circumstances is the entire basis of combining them.
    """
    return -np.log(prices).diff(lookback)


def low_volatility_signal(prices: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """Negated trailing volatility: calm stocks have historically done better."""
    return -np.log(prices).diff().rolling(lookback).std()


SIGNALS = {
    "momentum": momentum_signal,
    "reversal": reversal_signal,
    "low_volatility": low_volatility_signal,
}


def rank_normalise(signal: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional ranks mapped to [-1, 1], row by row.

    Ranks rather than raw values because the raw distributions differ wildly
    between signals and across time; a single stock with a 400% year would
    otherwise dominate a z-scored momentum book entirely.
    """
    # Ranks run 1..n, so the centre is (n+1)/2 and the half-width (n-1)/2. Using
    # pandas' pct=True and subtracting 0.5 instead leaves a residual offset of
    # 1/(2n) -- harmless when one signal is ranked alone, since a constant shared
    # by every asset cannot change their order, but it biases the *average* of
    # several signals whenever they cover different numbers of assets, which is
    # exactly what happens when one has NaNs and another does not.
    ranked = signal.rank(axis=1)
    count = signal.notna().sum(axis=1)
    half = ((count - 1) / 2.0).replace(0, np.nan)
    return (ranked.sub((count + 1) / 2.0, axis=0)).div(half, axis=0)


class MarketNeutralRanking(Strategy):
    """Long the top of a ranking, short the bottom, in equal dollars.

    Args:
        signals: which signals to combine, by name from `SIGNALS`. Several are
            averaged after rank-normalising, which is the standard way weak
            uncorrelated signals are pooled -- and the reason to expect anything
            here at all, since each alone is far too weak to trade.
        quantile: fraction of the universe taken on each side. 0.1 means the top
            and bottom decile.
        gross: total exposure, long plus short. 1.0 means 50% long and 50% short.
        borrow_cost: annual fee on the short side, charged through the weights so
            the backtest cannot forget it.
    """

    def __init__(
        self,
        signals: tuple[str, ...] = ("momentum",),
        quantile: float = 0.1,
        gross: float = 1.0,
        rebalance_days: int = 21,
        borrow_cost: float = DEFAULT_BORROW_COST,
    ) -> None:
        if not 0.0 < quantile <= 0.5:
            raise ValueError("quantile must be in (0, 0.5]")
        if not 0.0 < gross <= 2.0:
            raise ValueError("gross must be in (0, 2]")
        unknown = set(signals) - set(SIGNALS)
        if unknown:
            raise ValueError(f"unknown signals {unknown}; available: {sorted(SIGNALS)}")

        self.signals = tuple(signals)
        self.quantile = float(quantile)
        self.gross = float(gross)
        self.rebalance_days = int(rebalance_days)
        self.borrow_cost = float(borrow_cost)
        self.name = f"mn_{'+'.join(signals)}_q{int(quantile * 100)}"

    def combined_signal(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Average of rank-normalised signals. Equal weights, deliberately.

        Fitting the blend weights is where this family of strategies usually
        starts overfitting: the weights are estimated on the same sample the
        result is read from, and with three signals there are enough degrees of
        freedom to manufacture almost any backtest.
        """
        parts = [rank_normalise(SIGNALS[name](prices)) for name in self.signals]
        return sum(parts) / len(parts)

    def compute_weights(self, data: MarketData) -> pd.DataFrame:
        prices = data.prices
        signal = self.combined_signal(prices)
        n_assets = prices.shape[1]
        n_side = max(1, int(round(n_assets * self.quantile)))

        weights = np.zeros(prices.shape)
        current = np.zeros(n_assets)
        last_rebalance: int | None = None

        for t in range(len(prices)):
            row = signal.iloc[t]
            if row.isna().sum() > n_assets - 2 * n_side:
                continue

            due = last_rebalance is None or (t - last_rebalance) >= self.rebalance_days
            if due:
                values = row.to_numpy(dtype=float)
                valid = np.flatnonzero(np.isfinite(values))
                if len(valid) < 2 * n_side:
                    continue
                order = valid[np.argsort(values[valid])]

                book = np.zeros(n_assets)
                # Equal dollars each side, so a market-wide move cancels. The
                # per-name size is half the gross split across one side.
                book[order[-n_side:]] = (self.gross / 2.0) / n_side
                book[order[:n_side]] = -(self.gross / 2.0) / n_side
                current = book
                last_rebalance = t

            weights[t] = current

        return pd.DataFrame(weights, index=prices.index, columns=prices.columns)

    def borrow_drag(self, weights: pd.DataFrame) -> pd.Series:
        """Daily cost of borrowing the short side, given a weight history.

        A free function of the weights rather than state recorded during
        `compute_weights`. An earlier version stashed this on the instance mid-
        computation, which quietly broke the contract in `strategies/base.py`
        that a strategy is a pure function of price history -- the same call
        would leave different residue depending on what ran before it, and
        `check_causality`, which calls `compute_weights` repeatedly on truncated
        data, would have been reading whichever run finished last.
        """
        return weights.clip(upper=0.0).abs().sum(axis=1) * (self.borrow_cost / 252.0)
