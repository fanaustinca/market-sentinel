"""Performance measurement.

Every number here appears beside a benchmark elsewhere in the project, because a
return figure alone means nothing. "12% a year" is not a result until you know
what the market did over the same period and how much risk was taken to get it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class Performance:
    """Standard summary of one equity curve."""

    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    max_drawdown_days: int
    hit_rate: float
    n_periods: int

    def __str__(self) -> str:
        return (
            f"CAGR {self.cagr:+7.2%}   vol {self.volatility:6.2%}   "
            f"Sharpe {self.sharpe:+5.2f}   maxDD {self.max_drawdown:7.2%}"
        )


def max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """Worst peak-to-trough fall, and the longest time spent below a peak.

    The duration matters at least as much as the depth. It is rarely the size of
    a loss that makes people abandon a working system -- it is being underwater
    for eighteen months while everyone else is making money.
    """
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    worst = float(drawdown.min())

    underwater = drawdown < 0
    longest = current = 0
    for is_under in underwater:
        current = current + 1 if is_under else 0
        longest = max(longest, current)
    return worst, longest


def summarise(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> Performance:
    """Summarise a series of periodic simple returns."""
    returns = returns.dropna()
    n = len(returns)
    if n == 0:
        return Performance(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0)

    equity = (1 + returns).cumprod()
    total = float(equity.iloc[-1] - 1)
    years = n / periods_per_year
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 and equity.iloc[-1] > 0 else -1.0

    volatility = float(returns.std(ddof=1) * np.sqrt(periods_per_year))
    mean_annual = float(returns.mean() * periods_per_year)
    sharpe = mean_annual / volatility if volatility > 0 else 0.0

    # Sortino penalises only downside deviation, on the argument that upside
    # volatility is not a risk anyone objects to.
    downside = returns[returns < 0]
    downside_vol = float(downside.std(ddof=1) * np.sqrt(periods_per_year)) if len(downside) > 1 else 0.0
    sortino = mean_annual / downside_vol if downside_vol > 0 else 0.0

    worst, longest = max_drawdown(equity)
    traded = returns[returns != 0]
    hit_rate = float((traded > 0).mean()) if len(traded) else 0.0

    return Performance(
        total_return=total,
        cagr=cagr,
        volatility=volatility,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=worst,
        max_drawdown_days=longest,
        hit_rate=hit_rate,
        n_periods=n,
    )
