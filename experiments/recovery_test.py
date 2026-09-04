"""The Recovery Test sweep -- the Phase 2 gate.

Run with:  python experiments/recovery_test.py [--quick] [--markets N]

The Null Test established that nothing here profits from noise. This asks the
question that decides whether the project is worth continuing: **how strong must
a real signal be before the strategy finds it?**

An AR(1) momentum signal is planted at a known strength `phi` and swept from zero
upward. For each level, detection power is the fraction of markets on which the
strategy beats its own 95th-percentile null Sharpe -- the floor measured by the
Null Test, using the same machinery, so the two numbers are directly comparable.

The number this produces is then held against reality:

    Daily autocorrelation in real broad-equity returns is roughly 0.01 to 0.05,
    it is unstable, and it is not reliably of one sign.

If a strategy needs phi = 0.15 to reach 50% detection, it needs a signal an order
of magnitude stronger than real markets plausibly offer, and it cannot work. That
finding costs nothing and arrives before any money is at risk, which is the entire
point of building the sandbox first. **Phase 2 is explicitly allowed to kill the
project**, and reporting a discouraging threshold honestly is a success, not a
failure.

A second sweep runs the Ornstein-Uhlenbeck generator, whose signal lives in the
price *level* rather than in the sequence of returns. A returns-only feature set
should be substantially blind to it, and confirming that blindness is a real
finding about the feature set rather than about the market.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

from sentinel.ai.model import WalkForwardClassifier, WalkForwardModel
from sentinel.evaluation.recovery_test import RecoveryCurve, run_recovery_test
from sentinel.sandbox.generators.ar1 import AR1Generator
from sentinel.sandbox.generators.ou import OUGenerator
from sentinel.strategies.baseline import (
    AbsoluteMomentum,
    BuyAndHold,
    DualMomentum,
    ShortHorizonMomentum,
)

REPORTS = Path(__file__).resolve().parent.parent / "reports"

#: What real daily equity autocorrelation plausibly looks like. The comparison
#: that turns a sensitivity number into a verdict.
REAL_PHI_RANGE = (0.01, 0.05)

#: Signal strengths to sweep. Dense at the bottom, because that is where the
#: answer lives -- the difference between needing phi = 0.02 and phi = 0.10 is
#: the difference between a viable project and a dead one, while everything
#: above 0.15 is equally irrelevant.
PHI_LEVELS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)

#: Mean-reversion speeds for the OU blindness check, as annualised theta. Zero is
#: not a valid OU parameter -- theta must be positive -- so the null arm is a
#: GBM with matched volatility instead.
THETA_LEVELS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)


def strategies() -> list:
    """The AI and the control arm it has to beat.

    `BuyAndHold` is the degenerate case: it has no timing ability at all, so its
    detection power must stay near 5% at every signal strength. A curve that rises
    for buy-and-hold would mean the planted signal is leaking into returns as
    drift rather than as timing structure, and would invalidate every other curve
    in the sweep.

    Both momentum horizons are here because a rule can only see signals on its own
    timescale. `AbsoluteMomentum` at 252 days is blind to a one-day AR(1) signal
    at any strength, while `ShortHorizonMomentum` detects it easily. Running only
    the conventional horizon would produce a flat curve and the false conclusion
    that the signal is undetectable.
    """
    return [
        BuyAndHold(),
        AbsoluteMomentum(),
        ShortHorizonMomentum(),
        WalkForwardModel(),
        WalkForwardClassifier(),
    ]


def ar1_builder(phi: float):
    """Momentum of strength `phi`, with zero drift and volatility held constant.

    `mu=0` for the same reason the Null Test uses it: with positive drift, simply
    holding the asset makes money and the sweep would measure exposure rather
    than signal detection. The AR(1) generator internally compensates its
    innovation variance so that raising phi changes the *predictability* of
    returns without changing their volatility -- otherwise a strategy could
    appear to detect the signal while actually just reacting to a calmer market.
    """
    return AR1Generator(mu=0.0, sigma=0.16, phi=phi)


def ou_builder(theta: float):
    """Mean reversion of speed `theta`; theta = 0 falls back to a matched GBM."""
    if theta == 0.0:
        from sentinel.sandbox.generators.gbm import GBMGenerator

        return GBMGenerator(mu=0.0, sigma=0.16)
    return OUGenerator(theta=theta, sigma=0.16)


def verdict(curve: RecoveryCurve) -> str:
    """Translate a sensitivity threshold into a statement about real markets."""
    threshold = curve.detection_threshold(0.5)
    low, high = REAL_PHI_RANGE
    if threshold is None:
        return "cannot detect this signal at any strength tested"
    if threshold <= high:
        return f"threshold {threshold:.3f} is inside the plausible real range {low}-{high}"
    ratio = threshold / high
    return f"threshold {threshold:.3f} is {ratio:.0f}x the top of the plausible real range"


def run_sweep(builder, levels, parameter, n_markets, n_steps, workers) -> list[RecoveryCurve]:
    curves = []
    for strategy in strategies():
        started = time.time()
        curve = run_recovery_test(
            strategy,
            builder,
            strengths=levels,
            n_markets=n_markets,
            n_steps=n_steps,
            workers=workers,
            parameter=parameter,
        )
        curve.metadata["seconds"] = round(time.time() - started, 1)
        print(curve.report())
        print(f"  verdict       {verdict(curve)}")
        print(f"  elapsed       {curve.metadata['seconds']}s\n", flush=True)
        curves.append(curve)
    return curves


def summarise(ar1_curves: list[RecoveryCurve], ou_curves: list[RecoveryCurve]) -> str:
    low, high = REAL_PHI_RANGE
    lines = [
        "",
        "SENSITIVITY: signal strength needed for 50% detection",
        "",
        f"  real markets plausibly offer phi in {low}-{high}, unstably and without a reliable sign",
        "",
        f"  {'strategy':<28}{'phi @ 50%':>12}{'phi @ 80%':>12}   verdict",
    ]
    for curve in ar1_curves:
        fifty = curve.detection_threshold(0.5)
        eighty = curve.detection_threshold(0.8)
        lines.append(
            f"  {curve.strategy:<28}"
            f"{('none' if fifty is None else f'{fifty:.3f}'):>12}"
            f"{('none' if eighty is None else f'{eighty:.3f}'):>12}   {verdict(curve)}"
        )

    lines += [
        "",
        "OU MEAN REVERSION: the same measurement on a signal held in the price level",
        "",
        "  The feature set is built from returns. A signal that lives in where the",
        "  price sits relative to its own history should be much harder for it to",
        "  see, and how much harder is a fact about the features, not the market.",
        "",
        f"  {'strategy':<28}{'theta @ 50%':>14}",
    ]
    for curve in ou_curves:
        fifty = curve.detection_threshold(0.5)
        lines.append(
            f"  {curve.strategy:<28}{('none' if fifty is None else f'{fifty:.2f}'):>14}"
        )

    calibration = max(c.calibration_error for c in ar1_curves + ou_curves)
    lines += [
        "",
        f"  calibration check: worst zero-signal detection rate is {calibration:+.1%} from 5%",
    ]
    if calibration > 0.06:
        lines.append("  WARNING: the zero-signal arm is not landing at 5%. Thresholds above are unreliable.")
    return "\n".join(lines)


def write_reports(ar1_curves, ou_curves, n_markets, n_steps, text) -> Path:
    REPORTS.mkdir(exist_ok=True)

    def serialise(curves, levels_name):
        return {
            curve.strategy: {
                "generator": curve.generator,
                "parameter": curve.parameter,
                "threshold_50": curve.detection_threshold(0.5),
                "threshold_80": curve.detection_threshold(0.8),
                "calibration_error": round(curve.calibration_error, 4),
                "curve": [
                    {
                        "strength": point.strength,
                        "mean_sharpe": round(point.mean_sharpe, 4),
                        "lift": round(point.lift, 4),
                        "t_versus_null": round(point.t_versus_null, 3),
                        "detection_power": round(point.detection_power, 4),
                    }
                    for point in sorted(curve.points, key=lambda p: p.strength)
                ],
            }
            for curve in curves
        }

    payload = {
        "generated": date.today().isoformat(),
        "n_markets_per_point": n_markets,
        "n_steps": n_steps,
        "real_phi_range": list(REAL_PHI_RANGE),
        "ar1_momentum": serialise(ar1_curves, "phi"),
        "ou_mean_reversion": serialise(ou_curves, "theta"),
    }
    (REPORTS / "sensitivity.json").write_text(json.dumps(payload, indent=2) + "\n")

    dated = REPORTS / f"{date.today().isoformat()}-recovery-test.txt"
    dated.write_text(text + "\n")
    return dated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markets", type=int, default=200)
    parser.add_argument("--steps", type=int, default=2520)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="30 markets, 1260 days, coarse grid")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--skip-ou", action="store_true")
    args = parser.parse_args()

    n_markets, n_steps = (30, 1260) if args.quick else (args.markets, args.steps)
    phi_levels = (0.0, 0.02, 0.05, 0.10, 0.20) if args.quick else PHI_LEVELS
    theta_levels = (0.0, 1.0, 4.0) if args.quick else THETA_LEVELS

    print(f"Recovery Test: {n_markets} markets x {n_steps} days per point\n")
    print("AR(1) momentum -- a signal in the sequence of returns\n", flush=True)

    started = time.time()
    ar1_curves = run_sweep(ar1_builder, phi_levels, "phi", n_markets, n_steps, args.workers)

    ou_curves = []
    if not args.skip_ou:
        print("Ornstein-Uhlenbeck mean reversion -- a signal in the price level\n", flush=True)
        ou_curves = run_sweep(ou_builder, theta_levels, "theta", n_markets, n_steps, args.workers)

    text = summarise(ar1_curves, ou_curves)
    print(text)
    print(f"\ntotal {time.time() - started:.0f}s")

    if not args.no_write:
        path = write_reports(ar1_curves, ou_curves, n_markets, n_steps, text)
        print(f"written: {path.name} and reports/sensitivity.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
