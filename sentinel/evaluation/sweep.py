"""Running one strategy across many generated markets, in parallel.

Both headline experiments -- the Null Test and the Recovery Test -- do the same
mechanical thing: generate hundreds of markets from one generator, backtest one
strategy on each, and collect the resulting performance distribution. Only the
question differs. The Null Test asks whether the distribution is centred on zero;
the Recovery Test asks how far a planted signal pushes it away from where the
null put it.

Keeping the execution in one place means the two experiments cannot drift apart
in ways that would make their numbers incomparable -- and they are compared
directly: a recovery result is only meaningful measured against the noise floor
the Null Test produced with the identical machinery.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from sentinel.engine.backtest import UNLIMITED, CostModel, RiskLimits, run_backtest
from sentinel.sandbox.generators.base import Generator
from sentinel.strategies.base import Strategy


def _run_one(args: tuple) -> tuple[float, float, float]:
    strategy, generator, n_steps, seed, costs, limits, n_assets, periods = args
    scenario = generator.generate(n_steps=n_steps, n_assets=n_assets, seed=seed)
    result = run_backtest(
        scenario.data, strategy, costs=costs, limits=limits, periods_per_year=periods
    )
    performance = result.performance
    return performance.sharpe, performance.cagr, performance.max_drawdown


def sweep_markets(
    strategy: Strategy,
    generator: Generator,
    n_markets: int,
    n_steps: int,
    seed_offset: int = 100_000,
    costs: CostModel | None = None,
    limits: RiskLimits | None = None,
    workers: int | None = None,
    n_assets: int = 1,
    periods_per_year: int = 252,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backtest `strategy` on `n_markets` markets from `generator`.

    Args:
        seed_offset: markets use seeds `seed_offset .. seed_offset + n_markets`.
            Holding this fixed across a sweep of generator settings gives every
            setting the *same underlying random draws*, so a difference between
            two settings is caused by the setting rather than by luck. The
            Recovery Test depends on this: without it, the signal being measured
            would be buried under sampling noise several times its size.
        limits: defaults to `UNLIMITED`. Measurement runs switch the risk layer
            off deliberately -- it would mask the behaviour being measured.
        periods_per_year: rows per year in the generated markets. Must match the
            sampling frequency or every annualised figure is off by a constant
            factor, silently -- see `run_backtest`.
        n_assets: how many assets each generated market carries. Must match what
            the strategy expects: a multi-asset strategy given a one-asset market
            cannot find the tickers it trades, and a noise floor computed on the
            wrong number of assets would describe a different strategy than the
            one being judged.

    Returns:
        Arrays of Sharpe, CAGR and max drawdown, one entry per market.
    """
    if n_markets < 1:
        raise ValueError("need at least one market")

    costs = costs or CostModel()
    limits = limits if limits is not None else UNLIMITED

    jobs = [
        (strategy, generator, n_steps, seed_offset + i, costs, limits, n_assets, periods_per_year)
        for i in range(n_markets)
    ]

    if workers == 1:
        outcomes = [_run_one(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(_run_one, jobs, chunksize=4))

    sharpes, cagrs, drawdowns = (np.array(column) for column in zip(*outcomes))
    return sharpes, cagrs, drawdowns
