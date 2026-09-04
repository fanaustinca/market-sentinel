"""The Recovery Test -- how weak a signal can this strategy actually find?

The Null Test answers "does it correctly find nothing?" This answers the question
that decides whether the project is viable at all: **given a real signal, how
strong must it be before the strategy notices?**

Plant a pattern of known strength, sweep the strength down, and find where the
strategy stops detecting it. That number is the strategy's sensitivity, and it is
brutally informative, because real market signals are extremely weak. Daily
equity-return autocorrelation is roughly 0.01-0.05 and unstable. If a strategy
needs phi = 0.15 to see anything, it cannot work on real markets -- and that is
known from a measurement, before any money is involved.

What counts as "detected"
-------------------------
Not "made money". A strategy makes money on plenty of markets containing nothing;
that is exactly what the Null Test measured. Detection is defined against that
measured noise floor:

    detection power = the fraction of markets at this signal strength on which
                      the strategy beats its own 95th-percentile null Sharpe

At zero signal strength this is 5% by construction -- that is what a 95th
percentile means -- which gives the curve a built-in calibration check. If the
zero-strength point does not come out near 5%, the comparison is broken and no
other point on the curve can be trusted.

Common random numbers
---------------------
Every strength level uses the *same seeds*, so the same underlying random shocks
are reused as the planted signal is turned up. Each level therefore differs from
the last by the signal and nothing else. Without this the sampling noise between
levels is several times larger than the effect being measured, and the curve is
unreadable at any affordable number of markets.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from sentinel.engine.backtest import CostModel, RiskLimits
from sentinel.evaluation.sweep import sweep_markets
from sentinel.sandbox.generators.base import Generator
from sentinel.strategies.base import Strategy


@dataclass(frozen=True)
class RecoveryPoint:
    """One strength level: what the strategy scored, and whether that is a signal."""

    strength: float
    sharpes: np.ndarray
    cagrs: np.ndarray
    null_p95: float
    null_mean: float

    @property
    def mean_sharpe(self) -> float:
        return float(np.mean(self.sharpes))

    @property
    def standard_error(self) -> float:
        return float(np.std(self.sharpes, ddof=1) / np.sqrt(len(self.sharpes)))

    @property
    def detection_power(self) -> float:
        """Fraction of markets scoring above the null 95th percentile.

        The natural reading: run this strategy once on a market with this much
        signal in it, and this is the chance the result is strong enough to be
        distinguishable from luck.
        """
        return float(np.mean(self.sharpes > self.null_p95))

    @property
    def lift(self) -> float:
        """How far the mean Sharpe moved above where the null put it."""
        return self.mean_sharpe - self.null_mean

    @property
    def t_versus_null(self) -> float:
        """Standard errors between this mean and the null mean.

        Paired across the shared seeds, so this is a within-market comparison and
        far tighter than comparing two independent samples would be.
        """
        return self.lift / self.standard_error if self.standard_error > 0 else 0.0


@dataclass
class RecoveryCurve:
    """A strategy's sensitivity: detection power as a function of signal strength."""

    strategy: str
    generator: str
    parameter: str
    points: list[RecoveryPoint]
    metadata: dict = field(default_factory=dict)

    def detection_threshold(self, power: float = 0.5) -> float | None:
        """Smallest signal strength reaching `power`, linearly interpolated.

        `None` means the curve never gets there -- the strategy cannot reliably
        find this signal at any strength tested, which is itself a result.
        """
        ordered = sorted(self.points, key=lambda p: p.strength)
        previous = None
        for point in ordered:
            if point.detection_power >= power:
                if previous is None:
                    return point.strength
                span = point.detection_power - previous.detection_power
                if span <= 0:
                    return point.strength
                fraction = (power - previous.detection_power) / span
                return previous.strength + fraction * (point.strength - previous.strength)
            previous = point
        return None

    @property
    def calibration_error(self) -> float:
        """How far the zero-signal point sits from the 5% it must equal.

        The curve's self-check. A large value means the null floor being compared
        against does not match the markets being run, and every threshold read
        off this curve is wrong.
        """
        zero = [p for p in self.points if p.strength == 0.0]
        if not zero:
            return float("nan")
        return abs(zero[0].detection_power - 0.05)

    def report(self) -> str:
        lines = [
            f"{self.strategy} vs {self.generator} ({self.parameter} swept, "
            f"{len(self.points[0].sharpes)} markets each)",
            f"  null floor p95 {self.points[0].null_p95:+.3f}   "
            f"null mean {self.points[0].null_mean:+.3f}",
            "",
            f"  {self.parameter:>8}  {'mean Sharpe':>12}  {'lift':>8}  {'t':>7}  {'detected':>9}",
        ]
        for point in sorted(self.points, key=lambda p: p.strength):
            lines.append(
                f"  {point.strength:>8.3f}  {point.mean_sharpe:>+12.3f}  "
                f"{point.lift:>+8.3f}  {point.t_versus_null:>+7.1f}  "
                f"{point.detection_power:>8.0%}"
            )

        threshold = self.detection_threshold(0.5)
        lines.append("")
        if threshold is None:
            lines.append(
                f"  sensitivity   never reaches 50% detection over the range tested"
            )
        else:
            lines.append(f"  sensitivity   {self.parameter} = {threshold:.3f} for 50% detection")
        lines.append(f"  calibration   zero-signal detection {self.calibration_error:+.1%} from the 5% it must equal")
        return "\n".join(lines)


def run_recovery_test(
    strategy: Strategy,
    build_generator: Callable[[float], Generator],
    strengths: list[float] | tuple[float, ...],
    null_p95: float | None = None,
    n_markets: int = 200,
    n_steps: int = 2520,
    seed_offset: int = 100_000,
    costs: CostModel | None = None,
    limits: RiskLimits | None = None,
    workers: int | None = None,
    parameter: str = "phi",
) -> RecoveryCurve:
    """Sweep signal strength and measure where the strategy stops seeing it.

    Args:
        build_generator: maps a strength to a generator. Strength 0 must produce
            a market with no signal -- it is the arm the null floor is measured
            from, and every other point is judged against it.
        null_p95: the strategy's noise floor, if already measured. Left as `None`
            it is derived from this run's own zero-strength arm, which keeps the
            comparison exactly self-consistent at the cost of estimating the
            percentile from `n_markets` samples rather than from the larger Null
            Test sweep.
        seed_offset: shared by every strength level on purpose -- see the module
            docstring on common random numbers.
    """
    strengths = sorted(set(float(s) for s in strengths))
    if 0.0 not in strengths:
        raise ValueError(
            "the sweep must include strength 0.0: it is the null arm every other "
            "point is measured against, and the curve's calibration check"
        )

    baseline = build_generator(0.0)
    if baseline.has_exploitable_signal:
        raise ValueError(
            f"{baseline.model_name} still declares a signal at strength 0; "
            "the null arm of the sweep must be genuinely empty"
        )

    results: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for strength in strengths:
        sharpes, cagrs, _ = sweep_markets(
            strategy,
            build_generator(strength),
            n_markets=n_markets,
            n_steps=n_steps,
            seed_offset=seed_offset,
            costs=costs,
            limits=limits,
            workers=workers,
        )
        results[strength] = (sharpes, cagrs)

    null_sharpes = results[0.0][0]
    floor = float(np.percentile(null_sharpes, 95)) if null_p95 is None else float(null_p95)
    null_mean = float(np.mean(null_sharpes))

    points = [
        RecoveryPoint(
            strength=strength,
            sharpes=sharpes,
            cagrs=cagrs,
            null_p95=floor,
            null_mean=null_mean,
        )
        for strength, (sharpes, cagrs) in results.items()
    ]

    return RecoveryCurve(
        strategy=strategy.name,
        generator=baseline.model_name,
        parameter=parameter,
        points=points,
        metadata={
            "n_markets": n_markets,
            "n_steps": n_steps,
            "null_p95_source": "measured here" if null_p95 is None else "supplied",
        },
    )
