"""Markets built by resampling real returns -- the bridge from sandbox to reality.

Every other generator in the sandbox produces returns from a formula, which is
what makes their ground truth exact. It is also their limitation: real returns are
not normal, not independent, and not stationary, and a noise floor measured on
Gaussian markets may not describe the floor on real ones.

This generator closes that gap from the other side. It takes an actual return
history and resamples it, keeping the real marginal distribution -- the fat tails,
the skew, the occasional impossible day -- while destroying the time ordering. The
result is a market that is realistic in every respect except the one that matters:
there is nothing left to predict.

That makes it the right instrument for judging a real-data backtest. A Sharpe of
0.6 on real SPY history means little against a floor measured on Gaussian
simulations; it means a great deal against a floor measured on SPY's own returns,
reshuffled. `plan.md` section 3 lists this as the bridge to rung 2, and this is
what it is for.

Block size, and why the default is 1
------------------------------------
Resampling one day at a time destroys all serial structure, which is exactly what
a null market requires. Resampling in blocks of consecutive days preserves
whatever happens inside a block -- volatility clustering, short-horizon momentum,
the shape of a crash -- which is more realistic and no longer signal-free.

Both are useful and they answer different questions, so the generator refuses to
be confused about which it is: `has_exploitable_signal` is True whenever
`block_size > 1`, and `run_null_test` rejects any generator that declares one. A
blocked bootstrap can be used to test robustness. It cannot be used to compute a
noise floor, because the floor it produced would be inflated by structure the
blocks preserved, and an inflated floor quietly excuses a broken strategy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sentinel.sandbox.generators.base import Generator, Simulation
from sentinel.sandbox.market import MarketData


class BootstrapGenerator(Generator):
    """Resamples a real return history into new markets of any length.

    Args:
        log_returns: shape (n_observations,) or (n_observations, n_assets). Rows
            are days. Multi-asset histories are resampled by *row*, so the
            cross-sectional correlation between assets on any given day survives
            intact -- including the part that matters most, correlations spiking
            together during a crash. Resampling each asset independently would
            destroy that and make diversification look far more reliable than it
            is.
        block_size: consecutive days drawn together. 1 destroys all serial
            structure and is the only setting valid for a null market.
        demean: subtract each asset's mean return, so the market has zero drift.
            On by default for the same reason every other null market uses
            `mu = 0`: with drift, holding the asset makes money from exposure and
            the Null Test measures the wrong thing. Turn it off to study
            realistic markets rather than to measure a floor.
    """

    model_name = "bootstrap"

    def __init__(
        self,
        log_returns: np.ndarray,
        block_size: int = 1,
        demean: bool = True,
        initial_price: float = 100.0,
        trading_days: int = 252,
        tickers: list[str] | None = None,
    ) -> None:
        super().__init__(None, initial_price, trading_days)

        returns = np.asarray(log_returns, dtype=float)
        if returns.ndim == 1:
            returns = returns[:, None]
        if returns.ndim != 2:
            raise ValueError(f"log_returns must be 1- or 2-dimensional, got {returns.ndim}")
        if not np.isfinite(returns).all():
            raise ValueError("log_returns contain non-finite values")
        if len(returns) < 100:
            raise ValueError(
                f"only {len(returns)} historical returns; resampling from a sample "
                "that small reproduces the same few days over and over and the "
                "resulting distribution describes the sample, not the market"
            )
        if block_size < 1:
            raise ValueError("block_size must be at least 1")

        if tickers is not None and len(tickers) != returns.shape[1]:
            raise ValueError(
                f"got {len(tickers)} tickers for {returns.shape[1]} return columns"
            )

        self.source = returns - returns.mean(axis=0) if demean else returns
        self.block_size = int(block_size)
        self.demean = bool(demean)
        self.tickers = list(tickers) if tickers else None

    @classmethod
    def from_market(cls, data: MarketData, **kwargs) -> BootstrapGenerator:
        """Build a bootstrap from a real market's own history, names included.

        The names travel with the returns. A strategy that asks for `SPY` must be
        able to run on a bootstrap of SPY, or it cannot be given a noise floor --
        and a strategy without a floor cannot be judged at all.
        """
        kwargs.setdefault("tickers", list(data.tickers))
        return cls(data.log_returns().to_numpy(dtype=float), **kwargs)

    def default_tickers(self, n_assets: int) -> list[str] | None:
        return self.tickers[:n_assets] if self.tickers else None

    @property
    def has_exploitable_signal(self) -> bool:
        """Blocks longer than a day preserve serial structure, so they may."""
        return self.block_size > 1

    @property
    def has_predictable_volatility(self) -> bool:
        """Volatility clustering survives inside a block and nowhere else."""
        return self.block_size > 1

    @property
    def n_source_assets(self) -> int:
        return self.source.shape[1]

    def _simulate(self, n_steps: int, n_assets: int, rng: np.random.Generator) -> Simulation:
        if n_assets > self.n_source_assets:
            raise ValueError(
                f"asked for {n_assets} assets but the source history has "
                f"{self.n_source_assets}"
            )

        n = n_steps - 1
        source = self.source[:, :n_assets]
        n_source = len(source)

        if self.block_size == 1:
            indices = rng.integers(0, n_source, size=n)
        else:
            # Circular blocks: a block starting near the end wraps to the
            # beginning, so every observation is equally likely to be drawn.
            # Without wrapping, the last `block_size` days are systematically
            # under-sampled, which quietly reweights the history.
            n_blocks = int(np.ceil(n / self.block_size))
            starts = rng.integers(0, n_source, size=n_blocks)
            offsets = np.arange(self.block_size)
            indices = ((starts[:, None] + offsets[None, :]) % n_source).ravel()[:n]

        return Simulation(
            log_returns=source[indices],
            extra={
                "block_size": self.block_size,
                "demeaned": self.demean,
                "n_source_observations": int(n_source),
                "source_annual_volatility": (
                    source.std(axis=0) * np.sqrt(self.trading_days)
                ).tolist(),
                "source_excess_kurtosis": [
                    float(((column - column.mean()) ** 4).mean() / column.var() ** 2 - 3.0)
                    for column in source.T
                ],
            },
        )

    def _params(self, n_assets: int) -> dict[str, Any]:
        return {"n_assets_available": self.n_source_assets}
