"""Shared plumbing for every market generator.

Each generator differs only in how it produces log returns. Everything else --
seeding, correlation between assets, turning returns into a price frame, packaging
the answer key -- is identical, and lives here.

That uniformity is not just tidiness. It means every market in the sandbox comes
out the far end shaped exactly the same way, so the AI genuinely cannot tell a
random walk from a regime-switching market from real history by anything other
than the numbers themselves.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sentinel.sandbox.market import GroundTruth, MarketData, Scenario

TRADING_DAYS_PER_YEAR = 252


@dataclass
class Simulation:
    """Raw output of a generator's inner loop.

    Attributes:
        log_returns: shape (n_steps - 1, n_assets).
        regimes: true market state per step, when the model has states. This is
            the answer key a regime classifier gets scored against -- something
            no real market can ever provide.
        extra: resolved parameters to record in the ground truth.
    """

    log_returns: np.ndarray
    regimes: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Generator(ABC):
    """Base class for synthetic market generators.

    Subclasses implement `_simulate` and `_params`, and declare what they contain
    via `has_exploitable_signal` / `has_predictable_volatility`.
    """

    model_name: str = "unnamed"

    #: Whether the *direction* of returns is predictable from the past. This is
    #: the field the Null Test asserts against.
    has_exploitable_signal: bool = False

    #: Whether *volatility* is predictable. Deliberately separate: a market can
    #: have entirely unpredictable direction while its risk level is highly
    #: forecastable. Real markets are exactly like this, and conflating the two
    #: is a common way to overstate what a model has found.
    has_predictable_volatility: bool = False

    def __init__(
        self,
        correlation: np.ndarray | None = None,
        initial_price: float = 100.0,
        trading_days: int = TRADING_DAYS_PER_YEAR,
    ) -> None:
        self.correlation = correlation
        self.initial_price = float(initial_price)
        self.trading_days = int(trading_days)

    @property
    def dt(self) -> float:
        """Length of one step in years."""
        return 1.0 / self.trading_days

    # -- subclass contract ---------------------------------------------------

    @abstractmethod
    def _simulate(self, n_steps: int, n_assets: int, rng: np.random.Generator) -> Simulation:
        """Produce log returns of shape (n_steps - 1, n_assets)."""

    @abstractmethod
    def _params(self, n_assets: int) -> dict[str, Any]:
        """Parameters to record in the ground truth."""

    # -- shared implementation ----------------------------------------------

    def generate(
        self,
        n_steps: int = 2520,
        n_assets: int = 1,
        seed: int | None = None,
        tickers: list[str] | None = None,
        start_date: str = "2000-01-03",
    ) -> Scenario:
        """Produce one market.

        Args:
            n_steps: number of price observations. 2520 is ten years of trading days.
            n_assets: how many assets to simulate.
            seed: fixes the random draw. Always pass one in tests -- a statistical
                test that silently changes its own data each run is not a test.
            tickers: column names, defaulting to SYN0, SYN1, ...
            start_date: first date on the index. Only the spacing matters.
        """
        if n_steps < 2:
            raise ValueError("need at least 2 steps to produce one return")
        if n_assets < 1:
            raise ValueError("need at least one asset")

        rng = np.random.default_rng(seed)
        simulation = self._simulate(n_steps, n_assets, rng)

        log_returns = np.asarray(simulation.log_returns, dtype=float)
        expected = (n_steps - 1, n_assets)
        if log_returns.shape != expected:
            raise AssertionError(
                f"{type(self).__name__} produced {log_returns.shape}, expected {expected}"
            )

        # Prepend a zero row so the series starts at the initial price rather than
        # at the price after one step has already happened. An off-by-one here
        # would shift every series relative to its dates, which is precisely how
        # lookahead bias gets in.
        log_paths = np.vstack([np.zeros((1, n_assets)), np.cumsum(log_returns, axis=0)])
        prices = self.initial_price * np.exp(log_paths)

        if tickers is None:
            tickers = [f"SYN{i}" for i in range(n_assets)]
        elif len(tickers) != n_assets:
            raise ValueError(f"got {len(tickers)} tickers for {n_assets} assets")

        index = pd.bdate_range(start=start_date, periods=n_steps, name="date")
        frame = pd.DataFrame(prices, index=index, columns=tickers)

        params = dict(self._params(n_assets))
        params.update(simulation.extra)
        params.update(
            {
                "trading_days": self.trading_days,
                "initial_price": self.initial_price,
                "seed": seed,
            }
        )

        truth = GroundTruth(
            model=self.model_name,
            params=params,
            has_exploitable_signal=self.has_exploitable_signal,
            has_predictable_volatility=self.has_predictable_volatility,
            regimes=simulation.regimes,
        )
        return Scenario(
            data=MarketData(prices=frame, name=f"synthetic-{self.model_name}"),
            truth=truth,
            metadata={"n_steps": n_steps, "n_assets": n_assets},
        )

    # -- helpers shared by subclasses ---------------------------------------

    def _broadcast(self, value: float | np.ndarray, n_assets: int, name: str) -> np.ndarray:
        """Turn a scalar or per-asset value into an array of length n_assets."""
        array = np.asarray(value, dtype=float)
        try:
            return np.broadcast_to(array, (n_assets,)).copy()
        except ValueError as exc:
            raise ValueError(f"{name} has shape {array.shape}, cannot use for {n_assets} assets") from exc

    def _resolve_correlation(self, n_assets: int) -> np.ndarray:
        """Validate the correlation matrix, defaulting to independence."""
        if self.correlation is None:
            return np.eye(n_assets)

        corr = np.asarray(self.correlation, dtype=float)
        if corr.shape != (n_assets, n_assets):
            raise ValueError(f"correlation must be {n_assets}x{n_assets}, got {corr.shape}")
        if not np.allclose(corr, corr.T):
            raise ValueError("correlation matrix must be symmetric")
        if not np.allclose(np.diag(corr), 1.0):
            raise ValueError("correlation matrix must have ones on the diagonal")
        # Cholesky needs positive definiteness, which a hand-written correlation
        # matrix very easily lacks. Failing here with an explanation beats failing
        # opaquely inside numpy.
        eigenvalues = np.linalg.eigvalsh(corr)
        if eigenvalues.min() <= 0:
            raise ValueError(
                "correlation matrix is not positive definite "
                f"(smallest eigenvalue {eigenvalues.min():.4g}); "
                "the correlations you specified are mutually impossible"
            )
        return corr

    def _correlated_shocks(self, n_rows: int, n_assets: int, rng: np.random.Generator) -> np.ndarray:
        """Standard normal shocks carrying the requested correlation structure.

        If Z has identity covariance, then Z @ L.T has covariance L @ L.T, which
        is the correlation matrix -- where L is its Cholesky factor.
        """
        shocks = rng.standard_normal((n_rows, n_assets))
        if n_assets == 1:
            return shocks
        corr = self._resolve_correlation(n_assets)
        return shocks @ np.linalg.cholesky(corr).T
