"""Tests for whether a price series is a random walk.

These answer the question "is there anything in here?" -- which is the question
the entire project turns on. They are used two ways:

1.  To verify the GBM generator really produces the empty market it claims to.
    If our null market is not actually null, every result built on it is void.
2.  Later, to check whether a *real* market shows detectable structure, using the
    identical code. Same instrument, both places.

Each function returns a `TestResult` carrying the statistic, its p-value, and a
plain-language reading, because a bare p-value is easy to misread -- and misreading
one in this domain is expensive.

A note on what a p-value means here, since it is the most misused number in
statistics: it is the probability of seeing a result at least this extreme *if the
series really is a random walk*. A small p-value (below 0.05, say) is evidence
against randomness. A large one is **not** proof of randomness -- it only means we
failed to find structure, which may mean there is none, or may mean our test was
not sensitive enough to see it. That asymmetry is exactly why the Recovery Test in
Phase 2 exists: it measures how much structure this instrument can actually see.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class TestResult:
    """Outcome of a single statistical test."""

    name: str
    statistic: float
    p_value: float
    reading: str

    def rejects_random_walk(self, alpha: float = 0.05) -> bool:
        """True if the series shows structure a random walk would not produce."""
        return self.p_value < alpha

    def __str__(self) -> str:
        return f"{self.name}: stat={self.statistic:.4f} p={self.p_value:.4f} -- {self.reading}"


def autocorrelation(x: np.ndarray, lag: int) -> float:
    """Correlation of a series with itself `lag` steps earlier.

    In a random walk this is zero at every lag: today's return says nothing about
    tomorrow's. A reliably positive value at short lags is momentum; a negative one
    is mean reversion. Both are tradeable, which is why finding one in our null
    market would mean the generator is broken.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if lag >= n:
        raise ValueError(f"lag {lag} needs more than {n} observations")
    centred = x - x.mean()
    denominator = np.dot(centred, centred)
    if denominator == 0:
        return 0.0
    return float(np.dot(centred[lag:], centred[:-lag]) / denominator)


def ljung_box(x: np.ndarray, lags: int = 10) -> TestResult:
    """Ljung-Box test: is there autocorrelation at *any* lag up to `lags`?

    Checking each lag separately would be a trap -- test twenty lags at the 5%
    level and you expect one false alarm by chance alone. Ljung-Box pools them
    into a single statistic with a single p-value, so the false-alarm rate stays
    where you set it.

        Q = n(n+2) * sum over k of  rho_k^2 / (n - k)

    Under the null of no autocorrelation, Q follows a chi-squared distribution
    with `lags` degrees of freedom. The n(n+2)/(n-k) weighting is a small-sample
    correction to the simpler Box-Pierce form.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if lags >= n:
        raise ValueError(f"{lags} lags needs more than {n} observations")

    q = 0.0
    for k in range(1, lags + 1):
        rho_k = autocorrelation(x, k)
        q += rho_k**2 / (n - k)
    q *= n * (n + 2)

    p_value = float(stats.chi2.sf(q, df=lags))
    if p_value < 0.05:
        reading = f"autocorrelation detected within {lags} lags -- not a random walk"
    else:
        reading = f"no autocorrelation found within {lags} lags -- consistent with a random walk"
    return TestResult("ljung_box", float(q), p_value, reading)


def variance_ratio(log_prices: np.ndarray, q: int = 5) -> TestResult:
    """Lo-MacKinlay variance ratio test -- the classic random-walk test.

    The idea is elegant. If returns are independent, variance grows linearly with
    time: two-day moves have twice the variance of one-day moves, ten-day moves
    ten times. So the ratio

        VR(q) = Var(q-day returns) / (q * Var(1-day returns))

    is 1 for a random walk. Above 1 means moves reinforce each other -- trends,
    momentum. Below 1 means they partly cancel -- mean reversion.

    This catches things Ljung-Box can miss, because it responds to weak
    correlation spread across many lags that is individually invisible at each one
    but accumulates over longer horizons. Running both is deliberate.

    Args:
        log_prices: log of the price series, not returns and not raw prices.
        q: the aggregation horizon in periods.
    """
    p = np.asarray(log_prices, dtype=float).ravel()
    if q < 2:
        raise ValueError("q must be at least 2")
    t = len(p) - 1  # number of one-period returns
    if t < q * 2:
        raise ValueError(f"need at least {q * 2} returns for q={q}, have {t}")

    mu = (p[-1] - p[0]) / t

    # One-period variance, with the mean removed.
    one_period = np.diff(p) - mu
    var_1 = np.dot(one_period, one_period) / (t - 1)

    # q-period variance from *overlapping* windows -- overlapping uses far more of
    # the data than chopping the series into disjoint blocks, which matters a lot
    # at longer horizons where disjoint windows are few. The unbiasing constant
    # below is what corrects for the resulting dependence between windows.
    q_period = p[q:] - p[:-q] - q * mu
    m = q * (t - q + 1) * (1 - q / t)
    var_q = np.dot(q_period, q_period) / m

    if var_1 == 0:
        raise ValueError("series has zero variance")
    vr = var_q / var_1

    # Asymptotic standard error under the homoscedastic null.
    phi = 2 * (2 * q - 1) * (q - 1) / (3 * q)
    z = np.sqrt(t) * (vr - 1) / np.sqrt(phi)
    p_value = float(2 * stats.norm.sf(abs(z)))

    if p_value >= 0.05:
        reading = f"VR({q})={vr:.3f}, indistinguishable from 1 -- consistent with a random walk"
    elif vr > 1:
        reading = f"VR({q})={vr:.3f} above 1 -- moves reinforce each other (trending)"
    else:
        reading = f"VR({q})={vr:.3f} below 1 -- moves partly cancel (mean reverting)"
    return TestResult(f"variance_ratio_q{q}", float(vr), p_value, reading)


def normality(x: np.ndarray) -> TestResult:
    """Jarque-Bera test for normally distributed returns.

    GBM produces normal log returns by construction, so this checks the generator.
    Pointed at real markets the same test fails resoundingly: real returns have fat
    tails, meaning crashes happen far more often than a normal distribution allows.
    That failure is one of the most important facts in finance -- models assuming
    normality badly underestimate how bad the bad days get -- and it is a gap
    between our simulator and reality that later generators exist to close.
    """
    x = np.asarray(x, dtype=float).ravel()
    result = stats.jarque_bera(x)
    statistic, p_value = float(result.statistic), float(result.pvalue)
    if p_value < 0.05:
        reading = (
            f"returns are not normal (skew={stats.skew(x):.3f}, "
            f"excess kurtosis={stats.kurtosis(x):.3f})"
        )
    else:
        reading = "returns are consistent with a normal distribution"
    return TestResult("jarque_bera", statistic, p_value, reading)


def full_battery(log_prices: np.ndarray, lags: int = 10, horizons: tuple[int, ...] = (2, 5, 10)) -> list[TestResult]:
    """Run every random-walk test and return the results together."""
    p = np.asarray(log_prices, dtype=float).ravel()
    returns = np.diff(p)
    results = [ljung_box(returns, lags=lags), normality(returns)]
    results.extend(variance_ratio(p, q=q) for q in horizons)
    return results
