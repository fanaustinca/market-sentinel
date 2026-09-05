"""Volatility forecasters, all causal, all producing one number per day.

Each answers the same question: **given everything up to and including day t, how
volatile will the return from t to t+1 be?** Annualised, so the numbers are
comparable and readable.

The contract is deliberately narrow. A forecaster sees prices and returns a
series; it cannot see the future, cannot see ground truth, and has no opinion
about direction. That narrowness is what makes them comparable — the scoring in
`sentinel/evaluation/volatility_score.py` can then rank them on a proper scoring
rule with thousands of observations, which is the whole reason this is worth
doing at all.

Row t is a forecast for the period t to t+1, matching the strategy timing
convention exactly: `compute_weights` row t is the weight held from t to t+1, so
a sizing rule can divide by row t of a forecast without any realignment. An
off-by-one here would be invisible — the forecast would still look sensible — and
`tests/test_volatility_models.py` checks it against hand-computed values.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

#: -E[log|Z|] for a standard normal Z, equal to (gamma + log 2)/2. Converts a
#: prediction of E[log|return|] into a prediction of log(sigma). Exact under
#: conditional normality.
LOG_ABS_NORMAL_CORRECTION = 0.6351814227307392

#: Forecasts are floored here before any strategy divides by one. An unusually
#: quiet stretch would otherwise produce an enormous position, which is the
#: standard way this family of strategies fails.
MIN_ANNUAL_VOLATILITY = 1e-4


class VolatilityForecaster(ABC):
    """Predicts the annualised volatility of the next period's return."""

    name: str = "unnamed"

    @abstractmethod
    def forecast(self, prices: pd.Series) -> pd.Series:
        """Annualised volatility for each row, using data up to and including it.

        Returns a series indexed like `prices`, with NaN wherever the model has
        not yet seen enough history. Those NaNs are deliberate and must not be
        filled: back-filling a forecast imports future information into the past,
        and it is the single easiest way to break causality here.
        """

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


class RollingVolatility(VolatilityForecaster):
    """Standard deviation of the last `window` returns. The incumbent.

    This is what `VolatilityTarget` currently uses and what everything else has
    to beat. It is worth stating why it is hard to beat: it is unbiased, has no
    parameters to fit, cannot break, and captures most of what is knowable about
    tomorrow's volatility from the fact that volatility clusters.

    Its weakness is equal weighting. A return from twenty days ago counts exactly
    as much as yesterday's, which makes it slow at a genuine change in regime and
    leaves a visible artefact when a large move drops out of the window — the
    forecast falls sharply on a day when nothing happened.
    """

    def __init__(self, window: int = 21) -> None:
        if window < 2:
            raise ValueError("window must be at least 2 days")
        self.window = int(window)
        self.name = f"rolling_{window}d"

    def forecast(self, prices: pd.Series) -> pd.Series:
        returns = np.log(prices).diff()
        # Right-aligned by default in pandas. `center=True` would straddle t and
        # silently break causality while looking like smoothing.
        return returns.rolling(self.window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)


class EWMAVolatility(VolatilityForecaster):
    """Exponentially weighted volatility — RiskMetrics.

    Recent returns count for more, decaying geometrically. `lambda_ = 0.94` is
    the RiskMetrics daily parameter, and it is used here *as published* rather
    than fitted, which matters: a decay fitted on the same data it is evaluated
    on would make this comparison meaningless, and 0.94 has been the industry
    default since 1994, so it is genuinely out of sample with respect to
    everything here.

    It fixes the incumbent's worst artefact. Nothing ever drops out of a window,
    so the forecast cannot fall off a cliff on a quiet day merely because an old
    shock aged out.
    """

    def __init__(self, lambda_: float = 0.94, min_periods: int = 21) -> None:
        if not 0.0 < lambda_ < 1.0:
            raise ValueError("lambda_ must be strictly between 0 and 1")
        self.lambda_ = float(lambda_)
        self.min_periods = int(min_periods)
        self.name = f"ewma_{lambda_}"

    @property
    def half_life_days(self) -> float:
        return float(np.log(0.5) / np.log(self.lambda_))

    def forecast(self, prices: pd.Series) -> pd.Series:
        returns = np.log(prices).diff()
        # RiskMetrics is a variance recursion on squared returns with zero mean,
        # not a variance about a rolling mean. At daily frequency the mean is
        # negligible against the volatility and estimating it adds noise for
        # nothing.
        variance = (returns**2).ewm(alpha=1.0 - self.lambda_, min_periods=self.min_periods).mean()
        return np.sqrt(variance * TRADING_DAYS_PER_YEAR)


class HARVolatility(VolatilityForecaster):
    """Heterogeneous AutoRegressive realised volatility — Corsi (2009).

    Regresses tomorrow's volatility on realised volatility over three horizons at
    once: yesterday, the last week, and the last month. The idea it encodes is
    that different participants act on different horizons — a day trader, a
    portfolio manager, a pension fund — and each leaves a trace at its own scale,
    so a single window necessarily discards information the others carry.

    It is the standard benchmark in the volatility-forecasting literature and it
    is a linear regression on three columns, which makes it a fair comparison for
    a project that has just finished measuring gradient boosting into the ground.
    If three regressors beat a rolling standard deviation, the gain is
    attributable to the horizons and not to model capacity.

    Fitted walk-forward, refitting periodically, on log volatility — volatility
    is right-skewed and strictly positive, so a regression in logs is better
    behaved and cannot predict a negative one.
    """

    def __init__(
        self,
        min_train: int = 504,
        retrain_every: int = 63,
        horizons: tuple[int, int, int] = (1, 5, 22),
    ) -> None:
        if min_train < 100:
            raise ValueError("min_train below 100 rows cannot fit a useful regression")
        if retrain_every < 1:
            raise ValueError("retrain_every must be at least 1")
        self.min_train = int(min_train)
        self.retrain_every = int(retrain_every)
        self.horizons = tuple(int(h) for h in horizons)
        self.name = "har"

    def _features(self, returns: pd.Series) -> np.ndarray:
        """Mean absolute return over each horizon, ending today.

        Absolute return rather than squared: at daily frequency squared returns
        are dominated by a handful of extreme days, and a regression on them ends
        up fitting those days rather than the process. Mean absolute deviation is
        the robust equivalent and is scaled to a standard deviation below.
        """
        absolute = returns.abs()
        columns = [absolute.rolling(h).mean().to_numpy(dtype=float) for h in self.horizons]
        return np.column_stack(columns)

    def forecast(self, prices: pd.Series) -> pd.Series:
        returns = np.log(prices).diff()
        features = self._features(returns)

        # Target: tomorrow's absolute return, the one-observation estimate of
        # tomorrow's volatility. Noisy per row and unbiased in aggregate, which
        # is all a regression needs.
        target = returns.abs().shift(-1).to_numpy(dtype=float)

        n = len(prices)
        warmup = max(self.horizons)
        start = max(self.min_train, warmup + 2)

        out = np.full(n, np.nan)
        calibrations = np.ones(n)
        coefficients = None
        calibration = 1.0

        for t in range(start, n):
            if coefficients is None or (t - start) % self.retrain_every == 0:
                # The label for row i is observed on day i+1, so a model used on
                # day t may train on labels up to y[t-1] and no further. The same
                # off-by-one the return model documents, in a different place.
                train_end = t - 1
                x = features[warmup:train_end]
                y = target[warmup:train_end]
                usable = np.isfinite(x).all(axis=1) & np.isfinite(y) & (y > 0)
                if usable.sum() >= 100:
                    design = np.column_stack([np.ones(usable.sum()), np.log(x[usable] + 1e-12)])
                    coefficients = np.linalg.lstsq(
                        design, np.log(y[usable]), rcond=None
                    )[0]
                    # Calibrate the level on the training window rather than
                    # assuming conditional normality. The analytic constant
                    # below is exact for Gaussian returns and real returns are
                    # not Gaussian -- with the constant alone this model
                    # under-forecast by 9.7% on SPY, which QLIKE charges heavily
                    # because halving a forecast doubles a position sized off it.
                    #
                    # `calibration` scales the forecast so that predicted and
                    # realised second moments match in-sample. Fitted only on
                    # data up to t-1, like the coefficients, so it is causal.
                    in_sample = np.exp(
                        design @ coefficients + LOG_ABS_NORMAL_CORRECTION
                    )
                    calibration = float(
                        np.sqrt(np.mean(y[usable] ** 2) / np.mean(in_sample**2))
                    )

            if coefficients is None or not np.isfinite(features[t]).all():
                continue
            row = np.concatenate([[1.0], np.log(features[t] + 1e-12)])
            out[t] = float(row @ coefficients)
            calibrations[t] = calibration

        # The regression predicts E[log|r|], and the wanted quantity is sigma.
        # For r = sigma * z with z standard normal,
        #     log|r| = log(sigma) + log|z|,   E[log|z|] = -(gamma + log 2)/2
        # so log(sigma) = E[log|r|] + 0.63518, exactly. The constant is added
        # here rather than left to the intercept because it makes the assumption
        # visible: it is exact under conditional normality and approximate for
        # real fat-tailed returns.
        #
        # This replaced a smearing correction, exp(residual_variance / 2), which
        # is the textbook fix for retransformation bias and is wrong here. It
        # assumes the residuals are normal in logs; log|z| is not, being sharply
        # left-skewed, and its residual variance is large (about 1.23), so
        # smearing overshot by 12%. The uncorrected version undershot by a
        # similar amount in the other direction. Both errors were constant
        # multiplicative biases, which a scoring rule would have charged to the
        # model as poor forecasting rather than to the arithmetic.
        sigma = (
            np.exp(out + LOG_ABS_NORMAL_CORRECTION)
            * calibrations
            * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
        return pd.Series(sigma, index=prices.index, name=self.name)


class GARCHVolatility(VolatilityForecaster):
    """GARCH(1,1), fitted by maximum likelihood, walked forward.

    The classical model of volatility clustering: tomorrow's variance is a blend
    of a long-run level, yesterday's variance, and yesterday's squared shock.
    Three parameters, and it has been the reference point for thirty years.

    Fitted here rather than imported for the same reason the HMM and the
    random-walk tests are hand-written: it is the instrument a claim rests on,
    it is short, and understanding it fully costs less than trusting it blindly.
    The optimiser is a plain grid-refined search rather than a gradient method,
    because the likelihood is cheap, the parameter space is three-dimensional and
    bounded, and a robust search that always returns something sensible is worth
    more here than a fast one that occasionally does not converge.
    """

    def __init__(
        self,
        min_train: int = 504,
        retrain_every: int = 126,
        max_fit_window: int = 2000,
    ) -> None:
        if min_train < 100:
            raise ValueError("min_train below 100 rows cannot fit GARCH")
        self.min_train = int(min_train)
        self.retrain_every = int(retrain_every)
        # Three parameters do not need thirty years of data, and the likelihood
        # recursion is inherently sequential, so cost is linear in window length
        # times the number of optimiser evaluations. Capping the window at eight
        # years keeps a refit to well under a second on a century-long series
        # without measurably changing the estimates, and it has the side benefit
        # of letting the parameters drift as the market does.
        self.max_fit_window = int(max_fit_window)
        self.name = "garch11"

    @staticmethod
    def _negative_log_likelihood(params: np.ndarray, returns: np.ndarray) -> float:
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1.0:
            return 1e12  # non-stationary or impossible; finite so the optimiser can escape

        variance = np.empty(len(returns))
        variance[0] = returns.var()
        for i in range(1, len(returns)):
            variance[i] = omega + alpha * returns[i - 1] ** 2 + beta * variance[i - 1]
        variance = np.maximum(variance, 1e-16)
        return float(0.5 * np.sum(np.log(variance) + returns**2 / variance))

    def _fit(self, returns: np.ndarray) -> tuple[float, float, float]:
        """Maximum likelihood over (alpha, beta), with omega implied.

        omega is pinned by requiring the model's long-run variance to equal the
        sample variance: omega = var * (1 - alpha - beta). That removes a
        dimension and guarantees the fitted model has the right unconditional
        level, which matters more for forecasting than the last of the likelihood.

        Nelder-Mead rather than a gradient method: the likelihood is cheap but
        its gradient is not available in closed form here, the space is two
        bounded dimensions, and a derivative-free search that always returns
        something sensible is worth more than a fast one that occasionally fails
        to converge. It replaced a coarse-to-fine grid search that needed about a
        thousand likelihood evaluations per fit -- fine on a ten-year series and
        far too slow on a century of daily data.
        """
        from scipy.optimize import minimize

        sample_variance = float(np.var(returns))
        if sample_variance <= 0:
            return 1e-12, 0.05, 0.90

        def objective(free: np.ndarray) -> float:
            # Optimise in an unconstrained space and squash into the simplex, so
            # the search can never propose a non-stationary model and stall.
            alpha = 0.5 / (1.0 + np.exp(-free[0]))
            beta = (1.0 - alpha) * 0.999 / (1.0 + np.exp(-free[1]))
            omega = sample_variance * (1.0 - alpha - beta)
            return self._negative_log_likelihood(np.array([omega, alpha, beta]), returns)

        # Start from the RiskMetrics-like corner, which is close to the optimum
        # for almost every equity series and makes convergence quick.
        best = minimize(
            objective,
            x0=np.array([-2.2, 2.2]),  # ~ alpha 0.10, beta 0.88
            method="Nelder-Mead",
            options={"maxiter": 200, "xatol": 1e-4, "fatol": 1e-4},
        )
        alpha = 0.5 / (1.0 + np.exp(-best.x[0]))
        beta = (1.0 - alpha) * 0.999 / (1.0 + np.exp(-best.x[1]))
        return sample_variance * (1.0 - alpha - beta), alpha, beta

    def forecast(self, prices: pd.Series) -> pd.Series:
        returns = np.log(prices).diff().to_numpy(dtype=float)
        n = len(prices)
        out = np.full(n, np.nan)

        params = None
        variance = None

        for t in range(self.min_train, n):
            if params is None or (t - self.min_train) % self.retrain_every == 0:
                window = returns[1:t]
                window = window[np.isfinite(window)]
                if len(window) < 100:
                    continue
                window = window[-self.max_fit_window :]
                params = self._fit(window)
                # Re-run the recursion under the new parameters so the current
                # variance is consistent with them. Carrying the old state
                # forward would mix two models' beliefs, the same error the
                # regime classifier documents.
                omega, alpha, beta = params
                variance = float(np.var(window))
                for r in window:
                    variance = omega + alpha * r**2 + beta * variance

            if params is None or variance is None:
                continue

            omega, alpha, beta = params
            if np.isfinite(returns[t]):
                variance = omega + alpha * returns[t] ** 2 + beta * variance
            out[t] = np.sqrt(max(variance, 1e-16) * TRADING_DAYS_PER_YEAR)

        return pd.Series(out, index=prices.index, name=self.name)
