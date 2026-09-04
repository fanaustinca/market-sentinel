"""The Null Test -- the most important experiment in the project.

Run a strategy across many markets that provably contain no exploitable signal.
It must fail. If it makes money, we have not found an edge; we have found a bug.

Why one run is not enough
-------------------------
A single null market proves nothing. Any strategy will make money on *some* random
markets by luck, and lose on others. Judging by one run is exactly the error the
project exists to avoid -- the same error as validating the generator on a single
path, one level up.

So the test asks a distributional question instead:

    Across hundreds of markets containing nothing, what does this strategy score?

The answer is a distribution, and it gives two things that no single backtest can.

**A verdict.** The mean Sharpe across null markets should be zero or slightly
negative -- slightly negative because trading costs money and there is nothing to
pay for it. A mean that is significantly positive means information is leaking
from the future into the model, and the leak must be found before anything else
happens.

**A noise floor.** The 95th percentile of that distribution is the score this
strategy reaches by pure chance one time in twenty. If it is 0.8, then a Sharpe
of 0.6 on real data is *not evidence of skill* -- it is below the noise floor of
the instrument that measured it. Almost nobody computes this number, and without
it a backtest result cannot be interpreted at all.

The floor is a property of the strategy, not of the market. A strategy that trades
constantly has many more chances to get lucky and therefore a much wider null
distribution -- which is a rigorous version of the intuition that a busier
strategy needs a higher bar.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from sentinel.engine.backtest import UNLIMITED, CostModel, RiskLimits, run_backtest
from sentinel.sandbox.generators.base import Generator
from sentinel.strategies.base import Strategy


@dataclass
class NullTestResult:
    """The distribution of a strategy's performance on markets containing nothing."""

    strategy: str
    market: str
    n_markets: int
    sharpes: np.ndarray
    cagrs: np.ndarray
    max_drawdowns: np.ndarray
    metadata: dict = field(default_factory=dict)

    @property
    def mean_sharpe(self) -> float:
        return float(np.mean(self.sharpes))

    @property
    def standard_error(self) -> float:
        return float(np.std(self.sharpes, ddof=1) / np.sqrt(len(self.sharpes)))

    @property
    def t_statistic(self) -> float:
        """How many standard errors the mean Sharpe sits above zero."""
        return self.mean_sharpe / self.standard_error if self.standard_error > 0 else 0.0

    @property
    def noise_floor(self) -> float:
        """The Sharpe this strategy reaches by luck one time in twenty.

        Any real-market result at or below this is indistinguishable from chance.
        """
        return float(np.percentile(self.sharpes, 95))

    @property
    def noise_floor_99(self) -> float:
        return float(np.percentile(self.sharpes, 99))

    @property
    def profitable_fraction(self) -> float:
        """Share of null markets on which the strategy made money.

        Expected to be near half. This is the number that should be shown to
        anyone who reports a single profitable backtest as evidence.
        """
        return float(np.mean(self.cagrs > 0))

    def passed(self, threshold: float = 3.0) -> bool:
        """True when the strategy correctly fails to profit from noise.

        The bar is that the mean Sharpe is not significantly positive. A negative
        mean passes comfortably -- losing money on noise is the correct behaviour,
        because trading costs something and there is nothing to earn.
        """
        return self.t_statistic < threshold

    def report(self) -> str:
        verdict = "PASS" if self.passed() else "FAIL -- INVESTIGATE FOR LOOKAHEAD"
        return (
            f"{self.strategy} on {self.market} ({self.n_markets} markets)\n"
            f"  mean Sharpe      {self.mean_sharpe:+.4f}  (s.e. {self.standard_error:.4f}, "
            f"t = {self.t_statistic:+.2f})\n"
            f"  noise floor p95  {self.noise_floor:+.3f}   p99 {self.noise_floor_99:+.3f}\n"
            f"  profitable on    {self.profitable_fraction:.1%} of markets\n"
            f"  verdict          {verdict}"
        )


def _run_one(args: tuple) -> tuple[float, float, float]:
    strategy, generator, n_steps, seed, costs, limits = args
    scenario = generator.generate(n_steps=n_steps, n_assets=1, seed=seed)
    if scenario.truth.has_exploitable_signal:
        raise ValueError(
            f"{generator.model_name} declares an exploitable signal; "
            "the Null Test requires a market that provably contains none"
        )
    result = run_backtest(scenario.data, strategy, costs=costs, limits=limits)
    performance = result.performance
    return performance.sharpe, performance.cagr, performance.max_drawdown


def run_null_test(
    strategy: Strategy,
    generator: Generator,
    n_markets: int = 200,
    n_steps: int = 2520,
    seed_offset: int = 100_000,
    costs: CostModel | None = None,
    limits: RiskLimits | None = None,
    workers: int | None = None,
) -> NullTestResult:
    """Run `strategy` across `n_markets` signal-free markets.

    Args:
        generator: must declare `has_exploitable_signal = False`. This is checked,
            not assumed -- running the Null Test on a market that secretly
            contains a signal would produce a noise floor that is far too high and
            would quietly excuse a genuinely broken strategy.
        limits: defaults to `UNLIMITED`. The risk layer is switched off on
            purpose: it would mask the strategy's raw behaviour, and here we want
            to measure the strategy, not the safety net around it.
    """
    if generator.has_exploitable_signal:
        raise ValueError(
            f"{generator.model_name} declares an exploitable signal and cannot serve as a null market"
        )

    costs = costs or CostModel()
    limits = limits if limits is not None else UNLIMITED

    jobs = [
        (strategy, generator, n_steps, seed_offset + i, costs, limits)
        for i in range(n_markets)
    ]

    if workers == 1:
        outcomes = [_run_one(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(_run_one, jobs, chunksize=4))

    sharpes, cagrs, drawdowns = (np.array(column) for column in zip(*outcomes))

    return NullTestResult(
        strategy=strategy.name,
        market=generator.model_name,
        n_markets=n_markets,
        sharpes=sharpes,
        cagrs=cagrs,
        max_drawdowns=drawdowns,
        metadata={"n_steps": n_steps, "one_way_cost_bps": costs.one_way_cost * 10_000},
    )
