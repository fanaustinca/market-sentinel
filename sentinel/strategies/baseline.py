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


class ShortHorizonMomentum(AbsoluteMomentum):
    """Absolute momentum on a one-week horizon, acting daily.

    The same rule as `AbsoluteMomentum` with the horizon shortened, and it exists
    because a rule can only see signals that live on its own timescale. Measured
    on AR(1) markets with a very strong planted signal (phi = 0.3, an order of
    magnitude beyond anything real):

        lookback 252, monthly   detects 14% of the time -- the null rate
        lookback  60, monthly   detects 14%
        lookback  20, weekly    detects 16%
        lookback   5, daily     detects 100%
        lookback   2, daily     detects 100%

    The conventional twelve-month rule is not weak at finding this signal. It is
    **blind** to it, at the null rate, at any strength. AR(1) momentum acts at a
    one-day lag, and a rule averaging over 252 days cannot represent it.

    So the control arm needs both horizons. Without a short-horizon rule the
    Recovery Test would report "no strategy detects this signal" when the truth
    is "no strategy we ran was capable of detecting it", and that mistake would
    read as evidence the project should be abandoned.

    The cost of the short horizon is turnover -- roughly 50 round trips a year
    against 0.8 for the twelve-month rule, which by itself costs about 0.24 of
    Sharpe. It must find a real signal simply to break even.
    """

    name = "short_momentum"

    def __init__(self, lookback: int = 5, rebalance_days: int = 1) -> None:
        super().__init__(lookback=lookback, rebalance_days=rebalance_days)


class FixedWeights(Strategy):
    """A constant allocation, rebalanced on a schedule. The portfolio control arm.

    60/40 stocks and bonds is the allocation most defensive systems are
    implicitly competing with, and comparing against it is far more informative
    than comparing against all-equity buy-and-hold. A timing system that beats
    100% SPY on a risk-adjusted basis has often only discovered that holding less
    equity reduces risk -- which a fixed 60/40 does for free, with one decision
    made once, and no model to break.

    Args:
        weights: ticker to weight. Need not sum to 1; the remainder is cash.
        rebalance_days: how often to return to target. Never rebalancing lets the
            best performer take over the portfolio, which quietly turns a
            balanced allocation into a concentrated one over a long backtest.
    """

    name = "fixed_weights"

    def __init__(self, weights: dict[str, float], rebalance_days: int = 63) -> None:
        if not weights:
            raise ValueError("need at least one weight")
        if any(w < 0 for w in weights.values()):
            raise ValueError("weights cannot be negative; this project does not short")
        total = sum(weights.values())
        if total > 1.0 + 1e-9:
            raise ValueError(f"weights sum to {total:.3f}; leverage is not permitted")
        if rebalance_days < 1:
            raise ValueError("rebalance_days must be at least 1")
        self.weights = dict(weights)
        self.rebalance_days = int(rebalance_days)
        self.name = "fixed_" + "_".join(f"{t}{int(w * 100)}" for t, w in weights.items())

    def compute_weights(self, data: MarketData) -> pd.DataFrame:
        missing = [t for t in self.weights if t not in data.tickers]
        if missing:
            raise ValueError(f"{missing} not in the market's tickers {data.tickers}")

        target = pd.DataFrame(0.0, index=data.prices.index, columns=data.tickers)
        for ticker, weight in self.weights.items():
            target[ticker] = weight

        # Between rebalances the engine lets positions drift with prices, so
        # holding the target constant here would understate turnover and
        # therefore understate cost.
        mask = np.zeros(len(target), dtype=bool)
        mask[:: self.rebalance_days] = True
        return target.where(pd.Series(mask, index=target.index), other=np.nan).ffill().fillna(0.0)


class EnsembleMomentum(Strategy):
    """Absolute momentum averaged over several horizons at once.

    Not an optimisation. `AbsoluteMomentum` uses a 252-day lookback because
    twelve months is conventional, and there is no evidence anywhere in this
    project that 252 is better than 189 or 315 — the number was inherited, not
    measured. A single unjustified parameter is a hidden bet, and the honest
    response is not to search for the best value on data already seen, which is
    how a backtest stops meaning anything. It is to stop betting on the choice.

    So the signal is the *fraction* of horizons currently positive:

        1 of 4 positive -> hold 25%
        4 of 4 positive -> hold 100%

    That has two effects, and only the second is the point. It makes exposure
    graduated rather than binary, which happens to reduce whipsaw. And it makes
    the result insensitive to any individual horizon, which means the number
    reported is no longer partly a statement about which lookback happened to
    suit the sample.

    Judge it on **dispersion across markets**, not on mean return. If mean
    performance improves noticeably, that is a warning rather than a discovery:
    it would mean a horizon was picked with hindsight somewhere.

    Args:
        lookbacks: horizons in trading days. One month, one quarter, six months
            and a year — the standard spread, chosen before any of them was
            tested here.
        rebalance_days: how often the rule may act.
    """

    name = "ensemble_momentum"

    def __init__(
        self,
        lookbacks: tuple[int, ...] = (21, 63, 126, 252),
        rebalance_days: int = 21,
    ) -> None:
        if not lookbacks:
            raise ValueError("need at least one lookback")
        if any(lookback < 2 for lookback in lookbacks):
            raise ValueError("every lookback must be at least 2 days")
        if rebalance_days < 1:
            raise ValueError("rebalance_days must be at least 1")
        self.lookbacks = tuple(int(lookback) for lookback in lookbacks)
        self.rebalance_days = int(rebalance_days)

    def compute_weights(self, data: MarketData) -> pd.DataFrame:
        log_prices = np.log(data.prices)

        # Each horizon votes. A horizon that has not filled yet does not vote,
        # rather than voting zero -- treating "no opinion" as "bearish" would
        # make the strategy systematically defensive for its first year, which
        # is an artefact of the warmup rather than a view about the market.
        votes = None
        counted = None
        for lookback in self.lookbacks:
            trailing = log_prices.diff(lookback)
            positive = (trailing > 0).astype(float).where(trailing.notna())
            votes = positive if votes is None else votes.add(positive, fill_value=0.0)
            available = trailing.notna().astype(float)
            counted = available if counted is None else counted.add(available, fill_value=0.0)

        desired = (votes / counted.replace(0, np.nan)) / len(data.tickers)

        mask = np.zeros(len(data.prices), dtype=bool)
        mask[:: self.rebalance_days] = True
        desired = desired.where(pd.Series(mask, index=data.prices.index), other=np.nan).ffill()
        return desired.fillna(0.0)
