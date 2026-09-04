"""Geometric Brownian motion -- the null market.

This is the most important generator in the project, and the reason is that it
contains **nothing**. Prices wander with a fixed drift and a fixed volatility, and
each day's move is drawn independently of every day before it. There is no trend
to ride, no level to revert to, no regime to detect. The past tells you nothing
about the future beyond today's price.

That emptiness is the point. It is the control group for every experiment we run.
An AI turned loose on this market *must* fail to make money, and when it does not
fail, we have found a bug in our own code rather than an edge in the market.

The model
---------
Prices follow the standard GBM stochastic differential equation, whose exact
solution over a timestep dt gives log returns that are independent draws from

    log(S_t / S_t-1) ~ Normal( (mu - sigma^2/2) * dt,  sigma^2 * dt )

The `- sigma^2/2` correction is easy to miss and matters. `mu` is the drift of the
*price*, but we are simulating *log* prices, and because volatility drags on
compounded growth, the log-space drift is lower than mu by exactly sigma^2/2. Omit
it and the generator quietly produces higher returns than requested -- the kind of
small bias that would flow into every downstream result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.sandbox.market import GroundTruth, MarketData, Scenario

TRADING_DAYS_PER_YEAR = 252


class GBMGenerator:
    """Generates correlated random-walk markets with known parameters.

    Args:
        mu: annualised drift, per asset or one value for all. 0.08 is a
            reasonable stand-in for long-run equity returns.
        sigma: annualised volatility, per asset or one value for all. 0.16 is
            roughly the historical volatility of the S&P 500.
        correlation: asset correlation matrix. Defaults to the identity matrix
            (independent assets). Real equity markets are far more correlated
            than that, which is why later work will not leave this at default.
        initial_price: starting price level for every asset. Arbitrary -- returns
            are what matter -- but 100.0 makes results easy to read as percentages.
        trading_days: bars per year, used to convert annual parameters into the
            per-step values the simulation actually uses.
    """

    def __init__(
        self,
        mu: float | np.ndarray = 0.08,
        sigma: float | np.ndarray = 0.16,
        correlation: np.ndarray | None = None,
        initial_price: float = 100.0,
        trading_days: int = TRADING_DAYS_PER_YEAR,
    ) -> None:
        self.mu = mu
        self.sigma = sigma
        self.correlation = correlation
        self.initial_price = float(initial_price)
        self.trading_days = int(trading_days)

    def generate(
        self,
        n_steps: int = 2520,
        n_assets: int = 1,
        seed: int | None = None,
        tickers: list[str] | None = None,
        start_date: str = "2000-01-03",
    ) -> Scenario:
        """Produce one market path.

        Args:
            n_steps: number of price observations. The default 2520 is ten years
                of trading days.
            n_assets: how many correlated assets to simulate.
            seed: fixes the random draw. Always pass one in tests -- a statistical
                test that silently changes its own data on every run is not a test.
            tickers: column names. Defaults to SYN0, SYN1, ...
            start_date: first date on the index. Cosmetic; only the spacing matters.

        Returns:
            A `Scenario` whose `.data` holds prices and whose `.truth` records the
            parameters used and, critically, that no exploitable signal exists.
        """
        if n_steps < 2:
            raise ValueError("need at least 2 steps to produce one return")
        if n_assets < 1:
            raise ValueError("need at least one asset")

        mu = np.broadcast_to(np.asarray(self.mu, dtype=float), (n_assets,)).copy()
        sigma = np.broadcast_to(np.asarray(self.sigma, dtype=float), (n_assets,)).copy()
        if (sigma < 0).any():
            raise ValueError("volatility cannot be negative")

        corr = self._resolve_correlation(n_assets)
        rng = np.random.default_rng(seed)

        dt = 1.0 / self.trading_days
        # The volatility drag described in the module docstring.
        log_drift = (mu - 0.5 * sigma**2) * dt
        log_vol = sigma * np.sqrt(dt)

        # Draw independent shocks, then impose the correlation structure with the
        # Cholesky factor: if Z has identity covariance, Z @ L.T has covariance
        # L @ L.T, which is the correlation matrix we asked for.
        shocks = rng.standard_normal((n_steps - 1, n_assets))
        if n_assets > 1:
            shocks = shocks @ np.linalg.cholesky(corr).T

        log_returns = log_drift + log_vol * shocks

        # Prepend a zero so the first row is the initial price rather than the
        # price after one step already happened -- an off-by-one here would shift
        # every series by a day, which is exactly the kind of silent misalignment
        # that turns into apparent predictive power downstream.
        log_paths = np.vstack([np.zeros((1, n_assets)), np.cumsum(log_returns, axis=0)])
        prices = self.initial_price * np.exp(log_paths)

        if tickers is None:
            tickers = [f"SYN{i}" for i in range(n_assets)]
        elif len(tickers) != n_assets:
            raise ValueError(f"got {len(tickers)} tickers for {n_assets} assets")

        index = pd.bdate_range(start=start_date, periods=n_steps, name="date")
        frame = pd.DataFrame(prices, index=index, columns=tickers)

        truth = GroundTruth(
            model="gbm",
            params={
                "mu": mu.tolist(),
                "sigma": sigma.tolist(),
                "correlation": corr.tolist(),
                "trading_days": self.trading_days,
                "initial_price": self.initial_price,
                "seed": seed,
            },
            # The whole reason this generator exists.
            has_exploitable_signal=False,
        )
        return Scenario(
            data=MarketData(prices=frame, name="synthetic-gbm"),
            truth=truth,
            metadata={"n_steps": n_steps, "n_assets": n_assets},
        )

    def _resolve_correlation(self, n_assets: int) -> np.ndarray:
        if self.correlation is None:
            return np.eye(n_assets)

        corr = np.asarray(self.correlation, dtype=float)
        if corr.shape != (n_assets, n_assets):
            raise ValueError(f"correlation must be {n_assets}x{n_assets}, got {corr.shape}")
        if not np.allclose(corr, corr.T):
            raise ValueError("correlation matrix must be symmetric")
        if not np.allclose(np.diag(corr), 1.0):
            raise ValueError("correlation matrix must have ones on the diagonal")
        # Cholesky requires positive definiteness, and a correlation matrix typed
        # in by hand very easily is not. Failing here with an explanation beats
        # failing inside numpy later.
        eigenvalues = np.linalg.eigvalsh(corr)
        if eigenvalues.min() <= 0:
            raise ValueError(
                "correlation matrix is not positive definite "
                f"(smallest eigenvalue {eigenvalues.min():.4g}); "
                "the correlations you specified are mutually impossible"
            )
        return corr
