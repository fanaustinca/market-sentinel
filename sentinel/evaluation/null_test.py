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

What sets the floor, measured rather than assumed
-------------------------------------------------
An earlier version of this docstring claimed a busier strategy has a *wider* null
distribution and therefore a higher bar. Measurement says otherwise, and the
correction matters because it changes how a threshold should be set.

The spread of null Sharpes is almost entirely fixed by the **length of the track
record**, not by the strategy. Across constant-weight strategies at exposures
from 0.1 to 1.0, and momentum rules turning over anywhere from 0.8 to 84 times a
year, the standard deviation stayed at 0.31 +/- 0.01 on ten-year markets -- which
is the textbook standard error of an estimated Sharpe, 1/sqrt(years) = 0.316.

Trading frequency moves the *mean* instead. Costs drag it down roughly in
proportion to turnover: 0.8 turns a year cost 0.04 Sharpe, 84 turns cost 0.375.
Since the floor is mean + 1.645 sd, a busier strategy ends up with a **lower**
floor, not a higher one.

That is the opposite of the original intuition and it is a trap. A busy strategy
clears its own floor more easily only because it starts out losing money to
costs, so "beat your noise floor" is a necessary test and not a sufficient one --
a strategy must also clear zero. Both checks are reported.

The practical consequence is that a noise floor is meaningless without the
evaluation window attached. At ten years the floor is about 0.52; at six years,
1.645/sqrt(6) = 0.67. A target Sharpe quoted without a window length is not a
target. See `experiments/noise_floor_scaling.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sentinel.engine.backtest import CostModel, RiskLimits
from sentinel.evaluation.sweep import sweep_markets
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

    sharpes, cagrs, drawdowns = sweep_markets(
        strategy,
        generator,
        n_markets=n_markets,
        n_steps=n_steps,
        seed_offset=seed_offset,
        costs=costs,
        limits=limits,
        workers=workers,
    )

    return NullTestResult(
        strategy=strategy.name,
        market=generator.model_name,
        n_markets=n_markets,
        sharpes=sharpes,
        cagrs=cagrs,
        max_drawdowns=drawdowns,
        metadata={"n_steps": n_steps, "one_way_cost_bps": costs.one_way_cost * 10_000},
    )
