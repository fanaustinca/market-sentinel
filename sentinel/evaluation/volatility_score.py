"""Scoring volatility forecasts — where this project finally has enough power.

Every negative result so far came from asking a question the data could not
answer. A Sharpe ratio estimated over 33 years has a standard error near 0.18, so
two strategies differing by 0.1 are indistinguishable no matter how carefully the
comparison is set up. That is not a flaw in the comparison; it is the amount of
information a single return path contains.

Forecast accuracy is different. Every one of those 8,000 days is an observation,
so two forecasters differing by a few percent are separated decisively. The
statistical machinery below exists to take advantage of that, and to say by how
much rather than merely which is ahead.

The proxy problem, and why QLIKE
--------------------------------
Tomorrow's true volatility is never observed. All that can be seen is tomorrow's
return, and `r^2` is an unbiased but extremely noisy estimate of `sigma^2` -- its
own standard deviation is larger than its mean.

Most loss functions break under a noisy proxy: they rank forecasters differently
depending on the noise, so the winner is partly an artefact. Patton (2011) showed
only two common losses are **robust** in the sense of preserving the true ranking
in expectation despite the proxy, and both are used here:

    MSE     (r^2 - sigma^2)^2
    QLIKE   log(sigma^2) + r^2 / sigma^2

QLIKE is reported first because it penalises *under*-forecasting far more heavily
than over-forecasting, which is the correct asymmetry for anything that sizes a
position by dividing by a volatility estimate. A forecast that is half the truth
doubles the position; one that is double the truth halves it. Those are not
equally bad mistakes, and MSE treats them as if they were.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class VolatilityScore:
    """How well one forecaster predicted the next period's volatility."""

    name: str
    n: int
    qlike: float
    mse: float
    bias: float
    correlation: float
    mz_r_squared: float
    mz_slope: float

    def report(self) -> str:
        return (
            f"{self.name:<16}QLIKE {self.qlike:>8.4f}   MSE {self.mse:>10.3e}   "
            f"bias {self.bias:>+7.1%}   R2 {self.mz_r_squared:>5.1%}   "
            f"slope {self.mz_slope:>5.2f}"
        )


def score_forecast(
    forecast: pd.Series, prices: pd.Series, annualised: bool = True
) -> VolatilityScore:
    """Score an annualised volatility forecast against what actually happened.

    Args:
        forecast: row t is the predicted volatility of the return from t to t+1,
            matching the project's strategy timing convention exactly.

    The alignment is the one thing that would silently invalidate everything: a
    forecast compared against the *previous* day's return would look far better
    than it is, because volatility is persistent and yesterday's move is already
    inside the estimate.
    """
    returns = np.log(prices).diff()
    # Row t of the forecast predicts the return from t to t+1, which is
    # returns[t+1] in a diff-based series. Hence shift(-1).
    realised = returns.shift(-1)

    scale = TRADING_DAYS_PER_YEAR if annualised else 1.0
    variance = (forecast.to_numpy(dtype=float) ** 2) / scale
    squared = realised.to_numpy(dtype=float) ** 2

    usable = np.isfinite(variance) & np.isfinite(squared) & (variance > 0)
    variance, squared = variance[usable], squared[usable]
    if len(variance) < 30:
        raise ValueError(f"only {len(variance)} scoreable observations; too few to rank models")

    qlike = float(np.mean(np.log(variance) + squared / variance))
    mse = float(np.mean((squared - variance) ** 2))

    # Mincer-Zarnowitz: regress what happened on what was predicted. A perfect
    # forecast gives slope 1 and intercept 0. The R-squared is low for everything
    # -- around 10-20% is normal and expected -- because the target is a single
    # day's squared return, which is mostly noise even when the forecast is good.
    design = np.column_stack([np.ones(len(variance)), variance])
    intercept, slope = np.linalg.lstsq(design, squared, rcond=None)[0]
    predicted = design @ np.array([intercept, slope])
    total = float(np.sum((squared - squared.mean()) ** 2))
    r_squared = float(1.0 - np.sum((squared - predicted) ** 2) / total) if total > 0 else 0.0

    return VolatilityScore(
        name=str(forecast.name or "unnamed"),
        n=int(len(variance)),
        qlike=qlike,
        mse=mse,
        bias=float(np.sqrt(variance.mean()) / np.sqrt(squared.mean()) - 1.0),
        correlation=float(np.corrcoef(np.sqrt(variance), np.abs(np.sqrt(squared)))[0, 1]),
        mz_r_squared=r_squared,
        mz_slope=float(slope),
    )


def diebold_mariano(
    forecast_a: pd.Series,
    forecast_b: pd.Series,
    prices: pd.Series,
    loss: str = "qlike",
    lags: int = 10,
) -> dict:
    """Is A's advantage over B larger than sampling noise? Diebold-Mariano (1995).

    Compares the two forecasters day by day and asks whether the mean difference
    in loss is distinguishable from zero. This is the test that makes the power
    argument concrete: on the same data where two strategies' Sharpe ratios cannot
    be separated at all, two volatility forecasters can be separated at t = 10 or
    more.

    Standard errors are Newey-West, because daily losses are strongly
    autocorrelated -- a volatile week produces a run of large losses for every
    model at once. Ignoring that would overstate significance by a factor of two
    or three, which would be the same mistake this project spent a whole session
    correcting elsewhere.

    Returns the mean loss difference, the t-statistic, and a two-sided p-value.
    Negative `mean_difference` means A is better, since these are losses.
    """
    returns = np.log(prices).diff()
    realised = returns.shift(-1).to_numpy(dtype=float) ** 2

    variance_a = (forecast_a.to_numpy(dtype=float) ** 2) / TRADING_DAYS_PER_YEAR
    variance_b = (forecast_b.to_numpy(dtype=float) ** 2) / TRADING_DAYS_PER_YEAR

    usable = (
        np.isfinite(variance_a)
        & np.isfinite(variance_b)
        & np.isfinite(realised)
        & (variance_a > 0)
        & (variance_b > 0)
    )
    variance_a, variance_b, realised = variance_a[usable], variance_b[usable], realised[usable]

    if loss == "qlike":
        loss_a = np.log(variance_a) + realised / variance_a
        loss_b = np.log(variance_b) + realised / variance_b
    elif loss == "mse":
        loss_a = (realised - variance_a) ** 2
        loss_b = (realised - variance_b) ** 2
    else:
        raise ValueError(f"unknown loss {loss!r}; use 'qlike' or 'mse'")

    difference = loss_a - loss_b
    n = len(difference)
    mean = float(difference.mean())

    # Newey-West long-run variance with Bartlett weights.
    centred = difference - mean
    variance = float(np.dot(centred, centred) / n)
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        covariance = float(np.dot(centred[lag:], centred[:-lag]) / n)
        variance += 2.0 * weight * covariance
    variance = max(variance, 1e-300)

    t_statistic = mean / np.sqrt(variance / n)
    return {
        "n": int(n),
        "mean_difference": mean,
        "t_statistic": float(t_statistic),
        "p_value": float(2.0 * stats.norm.sf(abs(t_statistic))),
        "better": forecast_a.name if mean < 0 else forecast_b.name,
    }


def common_sample(forecasts: dict[str, pd.Series]) -> pd.Index:
    """Rows where every forecaster has produced a number.

    Models warm up at different times -- a rolling window needs 21 days, a
    walk-forward regression needs two years -- so scoring each on its own full
    range compares them on different periods. On SPY that difference was 1993-95,
    which contains a bear market, and it reversed the ranking between the
    per-model table and the pairwise tests.

    A ranking is only a ranking if every model faced the same days.
    """
    index = None
    for series in forecasts.values():
        valid = series.dropna().index
        index = valid if index is None else index.intersection(valid)
    return index if index is not None else pd.Index([])


def score_all(
    forecasts: dict[str, pd.Series], prices: pd.Series
) -> tuple[dict[str, VolatilityScore], pd.Index]:
    """Score every forecaster on the sample they all share."""
    shared = common_sample(forecasts)
    if len(shared) < 30:
        raise ValueError(
            f"only {len(shared)} rows are common to all forecasters; "
            "they cannot be compared"
        )
    scores = {
        name: score_forecast(series.reindex(prices.index).where(prices.index.isin(shared)), prices)
        for name, series in forecasts.items()
    }
    return scores, shared


def score_against_truth(forecast: pd.Series, true_volatility: np.ndarray) -> dict:
    """Grade a forecast against the volatility that actually governed each return.

    Only possible in the sandbox, and it is what the sandbox is for. On real data
    the only evidence about a day's volatility is that day's squared return --
    unbiased, but with a standard deviation larger than its mean -- so every
    real-data ranking is made through a very noisy lens.

    Here the answer is known, which allows the more valuable question: **does the
    noisy proxy rank forecasters the same way the exact measure does?** If it
    does, the real-data rankings can be trusted. If it does not, every
    volatility comparison anyone has ever published on real data is suspect,
    including the ones in this repository.

    Args:
        forecast: annualised, row t predicting the return from t to t+1.
        true_volatility: annualised, entry i governing the return from i to i+1 --
            the same alignment `GroundTruth.regimes` uses.
    """
    predicted = forecast.to_numpy(dtype=float)[: len(true_volatility)]
    truth = np.asarray(true_volatility, dtype=float)[: len(predicted)]

    usable = np.isfinite(predicted) & np.isfinite(truth) & (predicted > 0) & (truth > 0)
    predicted, truth = predicted[usable], truth[usable]
    if len(predicted) < 30:
        raise ValueError(f"only {len(predicted)} scoreable observations")

    log_error = np.log(predicted) - np.log(truth)
    return {
        "name": str(forecast.name or "unnamed"),
        "n": int(len(predicted)),
        # RMSE in logs: scale-free, symmetric in over- and under-forecasting, and
        # readable as an average proportional error.
        "rmse_log": float(np.sqrt(np.mean(log_error**2))),
        "bias": float(np.exp(np.mean(log_error)) - 1.0),
        "correlation": float(np.corrcoef(predicted, truth)[0, 1]),
        # QLIKE against the truth rather than a proxy, so the two are directly
        # comparable and the proxy's distortion is visible.
        "qlike_exact": float(
            np.mean(np.log(predicted**2) + truth**2 / predicted**2)
        ),
    }
