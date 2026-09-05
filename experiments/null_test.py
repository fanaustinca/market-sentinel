"""The Null Test sweep -- the Phase 1 gate.

Run with:  python experiments/null_test.py [--quick] [--markets N] [--steps N]

Every strategy in the project is run across hundreds of markets that provably
contain no exploitable signal. Each one must fail to profit. A strategy whose
mean Sharpe is significantly positive on nothing has not found an edge; it has a
lookahead bug, and finding it is more urgent than anything else in the project.

The sweep produces two artefacts:

**A verdict per (strategy, market) pair.** Mean Sharpe must not be significantly
positive. The t-statistic bar is 3.0 rather than the conventional 2.0 because the
sweep runs many combinations at once, and at t > 2 a fifteen-cell grid would be
expected to show a false alarm reasonably often.

**A noise floor per strategy.** The 95th and 99th percentiles of the null Sharpe
distribution -- what this strategy scores by luck alone on markets containing
nothing. Every later result in this project is compared against these numbers.
A real-market Sharpe below its strategy's floor is not a finding, and without
this table there is no way to know that.

The floors are written to `reports/noise_floor.json` so later phases can load
them rather than re-deriving them by hand, and to a dated text report for the
record.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

from sentinel.ai.model import WalkForwardClassifier, WalkForwardModel
from sentinel.evaluation.null_test import NullTestResult, run_null_test
from sentinel.sandbox.generators.gbm import GBMGenerator
from sentinel.sandbox.generators.heston import HestonGenerator
from sentinel.sandbox.generators.jump import JumpDiffusionGenerator
from sentinel.strategies.composite import TrendScaledVolatility
from sentinel.strategies.regime import RegimeAwareStrategy, RegimeGate
from sentinel.strategies.volatility import RegimeVolatilityTarget, VolatilityTarget
from sentinel.strategies.baseline import (
    AbsoluteMomentum,
    AlwaysCash,
    BuyAndHold,
    DualMomentum,
    EnsembleMomentum,
    ShortHorizonMomentum,
)

REPORTS = Path(__file__).resolve().parent.parent / "reports"

#: The t-statistic above which a mean Sharpe on noise counts as evidence of a
#: leak. Higher than the usual 2.0 because this is a multi-cell grid.
T_THRESHOLD = 3.0


def strategies() -> list:
    """Everything that decides what to hold, baselines included.

    The baselines are not here as filler. They are the control arm every later
    result is reported against, and each one needs its own floor: turnover shifts
    a strategy's null distribution down by the cost it pays on markets that offer
    nothing back, so the floors differ by more than a third across this list even
    though the markets are identical.
    """
    return [
        AlwaysCash(),
        BuyAndHold(),
        AbsoluteMomentum(),
        ShortHorizonMomentum(),
        EnsembleMomentum(),
        DualMomentum(),
        RegimeAwareStrategy(),
        RegimeGate(),
        VolatilityTarget(),
        RegimeVolatilityTarget(),
        TrendScaledVolatility(trend=EnsembleMomentum()),
        WalkForwardModel(),
        WalkForwardClassifier(),
    ]


def null_markets() -> list:
    """Generators that provably contain no exploitable direction signal.

    All three are set to `mu=0`. With positive drift any strategy that holds the
    asset earns money from beta rather than skill, and the test would measure
    exposure instead of leakage.

    They differ in what else they contain, which is the point: GBM is the clean
    null, jump-diffusion adds fat tails, and Heston adds forecastable
    *volatility* while leaving direction unforecastable. The last is the
    interesting one -- it is the shape real markets actually have, and a strategy
    that converts predictable risk into predictable return is making an error
    that GBM alone would never expose.
    """
    return [
        GBMGenerator(mu=0.0, sigma=0.16),
        JumpDiffusionGenerator(mu=0.0, sigma=0.14),
        HestonGenerator(mu=0.0),
    ]


def run_sweep(n_markets: int, n_steps: int, workers: int | None = None) -> list[NullTestResult]:
    results: list[NullTestResult] = []
    for strategy in strategies():
        for generator in null_markets():
            started = time.time()
            result = run_null_test(
                strategy, generator, n_markets=n_markets, n_steps=n_steps, workers=workers
            )
            result.metadata["seconds"] = round(time.time() - started, 1)
            print(result.report())
            print(f"  elapsed          {result.metadata['seconds']}s\n", flush=True)
            results.append(result)
    return results


def summarise(results: list[NullTestResult]) -> str:
    """The grid, as a table, with the noise floor per strategy at the bottom."""
    lines = ["", "Grid: mean Sharpe on markets containing nothing (t-statistic)", ""]
    markets = sorted({r.market for r in results})
    width = max(len(r.strategy) for r in results) + 2

    lines.append(" " * width + "".join(f"{m:>22}" for m in markets))
    for name in dict.fromkeys(r.strategy for r in results):
        row = f"{name:<{width}}"
        for market in markets:
            cell = next(r for r in results if r.strategy == name and r.market == market)
            row += f"{cell.mean_sharpe:>+13.3f} (t{cell.t_statistic:+5.1f})"
        lines.append(row)

    lines += ["", "Noise floor: the Sharpe luck alone hands each strategy on empty markets", ""]
    lines.append(f"{'strategy':<{width}}{'p95':>10}{'p99':>10}{'worst cell mean':>20}")
    for name in dict.fromkeys(r.strategy for r in results):
        cells = [r for r in results if r.strategy == name]
        pooled = np.concatenate([c.sharpes for c in cells])
        lines.append(
            f"{name:<{width}}{np.percentile(pooled, 95):>+10.3f}"
            f"{np.percentile(pooled, 99):>+10.3f}"
            f"{max(c.mean_sharpe for c in cells):>+20.3f}"
        )

    failures = [r for r in results if not r.passed(T_THRESHOLD)]
    lines.append("")
    if failures:
        lines.append(f"VERDICT: FAIL -- {len(failures)} of {len(results)} cells profit from noise.")
        for r in failures:
            lines.append(f"  {r.strategy} on {r.market}: t = {r.t_statistic:+.2f}")
        lines.append("")
        lines.append("Find the leak before doing anything else. The most likely location is")
        lines.append("label alignment in the walk-forward models: a model used on day t may")
        lines.append("train on labels up to y[t-1] and no further.")
    else:
        lines.append(f"VERDICT: PASS -- all {len(results)} cells correctly fail to profit from noise.")
    return "\n".join(lines)


def write_reports(results: list[NullTestResult], n_markets: int, n_steps: int, text: str) -> Path:
    REPORTS.mkdir(exist_ok=True)

    floors = {}
    for name in dict.fromkeys(r.strategy for r in results):
        cells = [r for r in results if r.strategy == name]
        pooled = np.concatenate([c.sharpes for c in cells])
        floors[name] = {
            "p95": round(float(np.percentile(pooled, 95)), 4),
            "p99": round(float(np.percentile(pooled, 99)), 4),
            "mean": round(float(np.mean(pooled)), 4),
            "sd": round(float(np.std(pooled, ddof=1)), 4),
            "n": int(pooled.size),
            "per_market": {
                c.market: {
                    "mean_sharpe": round(c.mean_sharpe, 4),
                    "t": round(c.t_statistic, 3),
                    "p95": round(c.noise_floor, 4),
                    "p99": round(c.noise_floor_99, 4),
                    "profitable_fraction": round(c.profitable_fraction, 4),
                    "passed": c.passed(T_THRESHOLD),
                }
                for c in cells
            },
        }

    payload = {
        "generated": date.today().isoformat(),
        "n_markets_per_cell": n_markets,
        "n_steps": n_steps,
        "t_threshold": T_THRESHOLD,
        "passed": all(r.passed(T_THRESHOLD) for r in results),
        "platform": f"{platform.python_version()} / {platform.machine()}",
        "noise_floor": floors,
    }
    (REPORTS / "noise_floor.json").write_text(json.dumps(payload, indent=2) + "\n")

    dated = REPORTS / f"{date.today().isoformat()}-null-test.txt"
    dated.write_text(text + "\n")
    return dated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markets", type=int, default=200, help="null markets per cell")
    parser.add_argument("--steps", type=int, default=2520, help="trading days per market")
    parser.add_argument("--workers", type=int, default=None, help="processes (default: all cores)")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="30 markets of 1260 days -- a smoke test, not evidence",
    )
    parser.add_argument("--no-write", action="store_true", help="print only, write nothing")
    args = parser.parse_args()

    n_markets, n_steps = (30, 1260) if args.quick else (args.markets, args.steps)

    if n_markets < 100 and not args.no_write:
        parser.error(
            "refusing to write a report from fewer than 100 markets per cell: a "
            "95th percentile estimated from that few samples is too noisy to "
            "publish, and every later phase compares against these numbers. "
            "Add --no-write for a smaller check."
        )

    print(f"Null Test sweep: {n_markets} markets x {n_steps} days per cell")
    print("Every cell below must FAIL to make money. A pass here is the AI")
    print("proving it can find nothing when there is nothing to find.\n", flush=True)

    started = time.time()
    results = run_sweep(n_markets, n_steps, workers=args.workers)
    text = summarise(results)
    print(text)
    print(f"\ntotal {time.time() - started:.0f}s")

    if not args.no_write:
        path = write_reports(results, n_markets, n_steps, text)
        print(f"written: {path.relative_to(Path.cwd())} and reports/noise_floor.json")

    return 0 if all(r.passed(T_THRESHOLD) for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
