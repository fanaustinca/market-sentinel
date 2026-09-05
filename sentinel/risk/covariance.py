"""Forecasting the covariance matrix, which is the part of a portfolio that is knowable.

The case for spending effort here rather than on returns
--------------------------------------------------------
This project has measured the same thing from six directions: expected returns
are not forecastable at any level it can detect, and volatility is. Across eight
national markets the best return edge reached p = 0.145, while volatility
forecasters separate at t = 3 to 5 on a single stock, and the earnings-calendar
result reached p = 0.000031.

Mean-variance portfolio construction needs two inputs, expected returns and a
covariance matrix. The received wisdom -- and it is right -- is that the return
estimates are so noisy they wreck the optimisation, which is why unconstrained
Markowitz portfolios famously behave worse out of sample than equal weights.

But two classical portfolios need *no return forecast at all*. Minimum variance
minimises `w' S w` outright, and risk parity equalises each asset's contribution
to portfolio risk. Both are functions of `S` alone. So they ask only for the
input this project can actually estimate, and ignore the one it cannot. That is
the entire argument for what follows, and it is why this is worth trying after so
many additions have made things worse: every previous addition tried to predict
returns in some disguise. This one does not.

Why the sample covariance is not enough
---------------------------------------
With `n` assets there are `n(n+1)/2` numbers to estimate. For six ETFs that is 21
from a year of data, which is tolerable; for fifteen stocks it is 120, which is
not. The sample estimate's largest eigenvalues are biased upward and its smallest
downward, and a minimum-variance optimiser is drawn precisely to the smallest --
it will happily pour weight into whatever pair of assets *looked* most mutually
hedging, which is usually an estimation artefact.

Shrinkage (Ledoit and Wolf, 2003) pulls the estimate toward a structured target
with few parameters. The target here is constant-correlation: every asset keeps
its own estimated volatility, and all pairwise correlations are replaced by their
average. It preserves the thing that is well estimated (individual volatilities)
and disciplines the thing that is not (the correlation structure).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: RiskMetrics' daily decay. The same constant `EWMAVolatility` uses, so a
#: covariance diagonal and a standalone volatility forecast agree by construction.
DEFAULT_LAMBDA = 0.94

#: Rows required before an estimate is emitted at all. Below this the matrix is
#: singular or nearly so, and a minimum-variance optimiser handed a singular
#: matrix returns a confident, meaningless answer rather than an error.
MIN_OBSERVATIONS = 60


def constant_correlation_target(sample: np.ndarray) -> np.ndarray:
    """The shrinkage target: own volatilities, one shared correlation.

    Chosen over the two usual alternatives on purpose. A scaled identity target
    discards the volatility differences between assets, which for a universe
    holding both gold and treasuries is throwing away the best-measured thing
    present. A single-factor target assumes a market factor exists, which is a
    claim about the assets rather than a piece of statistical discipline.
    """
    deviations = np.sqrt(np.diag(sample))
    safe = np.where(deviations > 0, deviations, 1.0)
    correlation = sample / np.outer(safe, safe)

    n = sample.shape[0]
    if n < 2:
        return sample.copy()
    off_diagonal = correlation[~np.eye(n, dtype=bool)]
    mean_correlation = float(np.mean(off_diagonal))

    target_correlation = np.full((n, n), mean_correlation)
    np.fill_diagonal(target_correlation, 1.0)
    return target_correlation * np.outer(deviations, deviations)


def shrink(sample: np.ndarray, intensity: float) -> np.ndarray:
    """Blend the sample estimate toward the constant-correlation target.

    Args:
        intensity: 0 keeps the sample estimate, 1 uses the target alone. A fixed
            intensity is used rather than Ledoit-Wolf's analytic optimum because
            the optimum is itself estimated from the same window, and on the
            sample sizes here it is unstable enough to add more noise than it
            removes. 0.3 is a standard default and is not tuned on the results.
    """
    if not 0.0 <= intensity <= 1.0:
        raise ValueError("intensity must be in [0, 1]")
    if intensity == 0.0:
        return sample
    return (1.0 - intensity) * sample + intensity * constant_correlation_target(sample)


class EWMACovariance:
    """Exponentially weighted covariance, one matrix per row, strictly causal.

    Row *t* of the output is estimated from returns up to and including *t*, and
    is the forecast for the covariance of the return from *t* to *t+1* -- the same
    convention every forecaster and strategy in this project uses. Rows before
    `MIN_OBSERVATIONS` are `None`, which callers must handle; emitting a plausible
    matrix there would let a strategy trade on an estimate built from ten days.
    """

    def __init__(
        self,
        lambda_: float = DEFAULT_LAMBDA,
        shrinkage: float = 0.3,
        min_observations: int = MIN_OBSERVATIONS,
    ) -> None:
        if not 0.0 < lambda_ < 1.0:
            raise ValueError("lambda_ must be in (0, 1)")
        self.lambda_ = float(lambda_)
        self.shrinkage = float(shrinkage)
        self.min_observations = int(min_observations)
        self.name = f"ewma_cov_l{lambda_}_s{shrinkage}"

    def estimate(self, prices: pd.DataFrame) -> list[np.ndarray | None]:
        """A covariance matrix per row of `prices`, in *daily* (not annualised) units."""
        returns = np.log(prices).diff().to_numpy(dtype=float)
        n_rows, n_assets = returns.shape

        out: list[np.ndarray | None] = [None] * n_rows
        state = np.zeros((n_assets, n_assets))
        seen = 0

        for t in range(1, n_rows):
            row = returns[t]
            if not np.all(np.isfinite(row)):
                out[t] = out[t - 1]
                continue

            # Update with day t's return, then emit. Row t may use day t's own
            # return because row t forecasts the return from t to t+1, which has
            # not happened yet.
            outer = np.outer(row, row)
            state = self.lambda_ * state + (1.0 - self.lambda_) * outer if seen else outer
            seen += 1

            if seen >= self.min_observations:
                out[t] = shrink(state, self.shrinkage)
        return out
