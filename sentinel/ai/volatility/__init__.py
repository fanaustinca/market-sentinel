"""Forecasting risk, which — unlike direction — is a question the data can answer.

The project's return-forecasting attempt failed, and the Recovery Test explained
why rather than leaving it a mystery: real return signals are roughly three times
weaker than the method can detect. That is a fact about the signal-to-noise ratio
of returns, and no amount of modelling changes it.

Volatility is a different problem, and the difference is one of statistical power
rather than of opinion about markets:

    A Sharpe ratio estimated over 33 years has a standard error near 0.18, so two
    strategies differing by 0.1 cannot be told apart. A volatility forecast is
    scored on every one of those 8,000 days with a proper scoring rule, and two
    forecasters differing by a few percent are separated decisively.

Every negative result this project has produced came from asking a question the
available data had too little power to answer. This is the part of the problem
where that is not true, and the sandbox has said so from the beginning:
`GroundTruth` has carried `has_predictable_volatility` as a field separate from
`has_exploitable_signal` since day one, and `HestonGenerator` exists to produce
markets where risk is forecastable and direction is not.

What a better forecast is worth is a separate question from whether it is better,
and the two are deliberately measured separately — scoring by QLIKE first, and
only then asking what it does to a strategy.
"""

from sentinel.ai.volatility.forecasters import (
    EWMAVolatility,
    GARCHVolatility,
    HARVolatility,
    RollingVolatility,
    VolatilityForecaster,
)

__all__ = [
    "VolatilityForecaster",
    "RollingVolatility",
    "EWMAVolatility",
    "HARVolatility",
    "GARCHVolatility",
]
