"""The backtest engine.

One loop, walked forward in time, with a strict rule about what is known when:
the weights applied to the move from day t to day t+1 were decided using data up
to day t and nothing after it.

The loop is deliberately explicit rather than vectorised. A vectorised
implementation would be faster and would also make an off-by-one alignment error
nearly invisible; here, the fact that `target_weights` is read from the previous
row is a line you can point at. At a few thousand steps the speed difference does
not matter, and correctness here is worth far more than speed.

The engine also carries costs and the risk layer, because both change results
enough that a backtest without them is fiction. Costs turn many apparently
profitable high-turnover strategies into losers, and the risk layer is what caps
the drawdown the strategy is allowed to inflict.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sentinel.engine.metrics import Performance, summarise
from sentinel.sandbox.market import MarketData
from sentinel.strategies.base import Strategy


@dataclass(frozen=True)
class CostModel:
    """What trading actually costs, in basis points of traded value.

    Defaults are deliberately realistic-to-pessimistic for retail ETF trading.
    Optimistic cost assumptions are one of the standard ways a backtest flatters
    a strategy that does not work: at 100% turnover a month, a 5bp underestimate
    compounds into more than half a percent a year of imaginary return.
    """

    commission_bps: float = 1.0
    spread_bps: float = 2.0
    slippage_bps: float = 2.0

    @property
    def one_way_cost(self) -> float:
        """Fraction of traded notional lost per unit bought or sold."""
        return (self.commission_bps + self.spread_bps + self.slippage_bps) / 10_000.0


@dataclass(frozen=True)
class RiskLimits:
    """Hard limits the risk layer enforces, regardless of what a strategy asks for.

    The risk layer can only ever *shrink* a position. A bug in a model therefore
    cannot produce an oversized trade -- the failure mode is being too cautious,
    which costs return, rather than too aggressive, which costs capital.
    """

    max_position: float = 1.0
    max_gross_exposure: float = 1.0
    #: Fall this far below the reference peak and go flat. `None` disables the
    #: breaker entirely, which is what measurement experiments want -- the Null
    #: Test needs to see a strategy's raw behaviour, not the risk layer's.
    drawdown_stop: float | None = 0.12
    #: Once tripped, stay in cash this many days before allowing risk again.
    cooldown_days: int = 21
    allow_shorting: bool = False


#: Limits with every constraint switched off, for measurement runs where the risk
#: layer would otherwise mask the thing being measured.
UNLIMITED = RiskLimits(max_position=1.0, max_gross_exposure=1.0, drawdown_stop=None)


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    performance: Performance
    breaker_days: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def total_cost(self) -> float:
        return float(self.costs.sum())

    @property
    def annual_turnover(self) -> float:
        years = len(self.turnover) / 252
        return float(self.turnover.sum() / years) if years > 0 else 0.0


def _apply_risk_limits(weights: np.ndarray, limits: RiskLimits) -> np.ndarray:
    """Shrink a weight vector to satisfy the limits. Never enlarges it."""
    if not limits.allow_shorting:
        weights = np.maximum(weights, 0.0)
    weights = np.clip(weights, -limits.max_position, limits.max_position)

    gross = np.abs(weights).sum()
    if gross > limits.max_gross_exposure and gross > 0:
        weights = weights * (limits.max_gross_exposure / gross)
    return weights


def run_backtest(
    data: MarketData,
    strategy: Strategy,
    costs: CostModel | None = None,
    limits: RiskLimits | None = None,
    risk_free_rate: float = 0.0,
) -> BacktestResult:
    """Run `strategy` over `data` and measure what happened.

    Args:
        risk_free_rate: annual rate earned on the cash portion. Defaults to zero,
            which is the right choice inside the sandbox: a positive rate would
            let a strategy earn money simply by holding cash, and the Null Test
            needs to measure skill, not interest.
    """
    costs = costs or CostModel()
    limits = limits or RiskLimits()

    prices = data.prices
    simple_returns = data.simple_returns()
    raw_weights = strategy.compute_weights(data)

    if not raw_weights.index.equals(prices.index):
        raise ValueError(
            f"{strategy.name} returned weights indexed differently from prices; "
            "row t must correspond to price t"
        )
    if list(raw_weights.columns) != data.tickers:
        raise ValueError(f"{strategy.name} returned columns {list(raw_weights.columns)}, expected {data.tickers}")

    target_matrix = raw_weights.to_numpy(dtype=float)
    if not np.isfinite(target_matrix[np.isfinite(target_matrix)]).all():
        raise ValueError(f"{strategy.name} produced non-finite weights")
    target_matrix = np.nan_to_num(target_matrix, nan=0.0)

    return_matrix = simple_returns.to_numpy(dtype=float)
    n_periods, n_assets = return_matrix.shape

    daily_risk_free = risk_free_rate / 252
    cost_rate = costs.one_way_cost

    equity = np.empty(n_periods)
    period_returns = np.empty(n_periods)
    turnover_series = np.empty(n_periods)
    cost_series = np.empty(n_periods)
    applied = np.zeros((n_periods, n_assets))

    value = 1.0
    peak = 1.0
    held = np.zeros(n_assets)
    cooldown_remaining = 0
    breaker_days = 0

    for i in range(n_periods):
        # Row i of the *price* index is day i; returns row i is the move from day
        # i to day i+1. So the weights that apply are the ones decided on day i --
        # which is `target_matrix[i]`, computed from prices up to and including i.
        target = _apply_risk_limits(target_matrix[i], limits)

        # The drawdown circuit breaker, using only equity observed so far.
        if limits.drawdown_stop is not None:
            if cooldown_remaining > 0:
                target = np.zeros(n_assets)
                cooldown_remaining -= 1
                breaker_days += 1
                # Re-entering resets the reference peak to the current value.
                # Without this the breaker measures against an all-time high the
                # strategy may never see again, so it re-trips on the very next
                # bar and the system is locked in cash permanently after one bad
                # stretch. Each re-entry instead gets its own drawdown budget.
                if cooldown_remaining == 0:
                    peak = value
            elif value / peak - 1.0 <= -limits.drawdown_stop:
                target = np.zeros(n_assets)
                cooldown_remaining = limits.cooldown_days
                breaker_days += 1

        traded = np.abs(target - held).sum()
        cost = traded * cost_rate

        asset_return = float(target @ return_matrix[i])
        cash_weight = 1.0 - target.sum()
        period_return = asset_return + cash_weight * daily_risk_free - cost

        value *= 1.0 + period_return
        peak = max(peak, value)

        # Positions drift with prices between rebalances, so tomorrow's starting
        # weights are not today's targets. Ignoring this would understate turnover
        # and therefore understate costs.
        grown = target * (1.0 + return_matrix[i])
        denominator = grown.sum() + cash_weight * (1.0 + daily_risk_free)
        held = grown / denominator if denominator > 0 else np.zeros(n_assets)

        equity[i] = value
        period_returns[i] = period_return
        turnover_series[i] = traded
        cost_series[i] = cost
        applied[i] = target

    index = simple_returns.index
    returns = pd.Series(period_returns, index=index, name="return")

    return BacktestResult(
        equity=pd.Series(equity, index=index, name="equity"),
        returns=returns,
        weights=pd.DataFrame(applied, index=index, columns=data.tickers),
        turnover=pd.Series(turnover_series, index=index, name="turnover"),
        costs=pd.Series(cost_series, index=index, name="cost"),
        performance=summarise(returns),
        breaker_days=breaker_days,
        metadata={"strategy": strategy.name, "market": data.name},
    )
