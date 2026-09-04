"""How the noise floor depends on the length of the track record.

Run with:  python experiments/noise_floor_scaling.py [--quick]

This experiment exists because of a question left open in the project's own
handoff notes. An early 100-market run measured a noise floor of 0.688 for
absolute momentum; the full Phase 1 sweep measured 0.506 for the same strategy.
Neither was wrong. The first ran on six-year markets and the second on ten-year
markets, and **the noise floor is mostly a function of how long the track record
is**, not of the strategy or the market.

The reason is textbook and worth stating plainly, because the project's goals are
written in Sharpe ratios. An estimated Sharpe has standard error roughly
`1/sqrt(T)` with T in years. The 95th percentile of the null distribution is
therefore about

    floor ~= (cost drag) + 1.645 / sqrt(years)

which is 0.52 at ten years, 0.67 at six, 0.95 at three, and 1.64 at one. A
strategy showing Sharpe 0.9 over three years has shown nothing at all. The same
0.9 over twenty years is a genuine result.

This has a direct consequence for `plan.md` section 2, which sets a target of
"Sharpe > 0.7". That target is only meaningful with an evaluation window attached
-- it is a demanding bar over twenty years, roughly a coin flip over six, and
literally unachievable-as-evidence over two. The experiment measures the curve so
the target can be restated in terms that mean something.

It also separates the two effects that move a floor, which the earlier reasoning
in `null_test.py` conflated:

  * **Track length** sets the *spread*. Nothing a strategy does changes it much.
  * **Turnover** sets the *centre*, by paying costs on markets that offer nothing
    back. A busy strategy therefore has a *lower* floor -- which is a trap, since
    it is lower only because the strategy is already losing money.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

from sentinel.evaluation.null_test import run_null_test
from sentinel.sandbox.generators.gbm import GBMGenerator
from sentinel.strategies.baseline import (
    AbsoluteMomentum,
    BuyAndHold,
    ShortHorizonMomentum,
)

REPORTS = Path(__file__).resolve().parent.parent / "reports"

#: Track-record lengths in trading days: 1, 2, 3, 5, 6, 10 and 20 years.
LENGTHS = (252, 504, 756, 1260, 1512, 2520, 5040)

#: 1.645 is the standard normal 95th percentile. The prediction being tested is
#: that the null Sharpe distribution is normal with standard deviation
#: 1/sqrt(years), so its 95th percentile sits this many standard errors up.
Z95 = 1.6448536269514722


def strategies() -> list:
    """One strategy per turnover regime, to separate spread from centre.

    Buy-and-hold trades once and pays nothing after that, so its floor is the
    pure statistical effect. The two momentum rules add roughly 1 and 50 round
    trips a year on top of it.
    """
    return [BuyAndHold(), AbsoluteMomentum(), ShortHorizonMomentum()]


def run(n_markets: int, lengths: tuple[int, ...], workers: int | None) -> dict:
    generator = GBMGenerator(mu=0.0, sigma=0.16)
    table: dict[str, dict[int, dict]] = {}

    for strategy in strategies():
        table[strategy.name] = {}
        print(f"\n{strategy.name}")
        print(
            f"  {'years':>6}{'floor p95':>12}{'predicted':>12}"
            f"{'sd':>9}{'1/sqrt(T)':>11}{'mean':>9}"
        )
        for n_steps in lengths:
            # A strategy needs its warmup before it holds anything, so very short
            # markets would measure a strategy that spent most of its life in
            # cash. Momentum's 252-day lookback makes one-year markets useless
            # for it, and the run is skipped rather than reported as a floor.
            if n_steps <= getattr(strategy, "lookback", 0) + 21:
                print(f"  {n_steps / 252:>6.1f}{'-- shorter than warmup --':>44}")
                continue

            result = run_null_test(
                strategy, generator, n_markets=n_markets, n_steps=n_steps, workers=workers
            )
            years = n_steps / 252
            sd = float(result.sharpes.std(ddof=1))
            predicted = result.mean_sharpe + Z95 / np.sqrt(years)
            print(
                f"  {years:>6.1f}{result.noise_floor:>+12.3f}{predicted:>+12.3f}"
                f"{sd:>9.3f}{1 / np.sqrt(years):>11.3f}{result.mean_sharpe:>+9.3f}"
            )
            table[strategy.name][n_steps] = {
                "years": round(years, 2),
                "floor_p95": round(result.noise_floor, 4),
                "predicted_p95": round(float(predicted), 4),
                "sd": round(sd, 4),
                "sd_predicted": round(float(1 / np.sqrt(years)), 4),
                "mean_sharpe": round(result.mean_sharpe, 4),
            }
    return table


def summarise(table: dict) -> str:
    lines = [
        "",
        "NOISE FLOOR vs TRACK-RECORD LENGTH",
        "",
        "  The 95th-percentile Sharpe that markets containing nothing hand each",
        "  strategy by luck alone. A real result must clear its own row.",
        "",
    ]

    lengths = sorted({int(k) for s in table.values() for k in s})
    header = f"  {'strategy':<22}" + "".join(f"{n / 252:>9.0f}y" for n in lengths)
    lines.append(header)
    for name, rows in table.items():
        line = f"  {name:<22}"
        for n in lengths:
            line += f"{rows[n]['floor_p95']:>+10.2f}" if n in rows else f"{'--':>10}"
        lines.append(line)

    lines += [
        "",
        "  Theory says the spread is 1/sqrt(years) regardless of strategy, so the",
        "  floor is (cost drag) + 1.645/sqrt(years). Measured against predicted:",
        "",
    ]
    errors = []
    for name, rows in table.items():
        for n, row in rows.items():
            errors.append(abs(row["floor_p95"] - row["predicted_p95"]))
    lines.append(f"  worst deviation from prediction: {max(errors):.3f} Sharpe")
    lines.append(f"  mean deviation:                  {np.mean(errors):.3f} Sharpe")

    lines += [
        "",
        "  WHAT THIS MEANS FOR THE PROJECT'S TARGET",
        "",
        "  plan.md sets 'Sharpe > 0.7'. Against the buy-and-hold floor that target is:",
        "",
    ]
    bh = table.get("buy_and_hold", {})
    for n in lengths:
        if n not in bh:
            continue
        floor = bh[n]["floor_p95"]
        years = bh[n]["years"]
        if 0.7 <= floor:
            reading = "BELOW the floor -- indistinguishable from luck"
        elif 0.7 < floor * 1.3:
            reading = "barely above the floor -- weak evidence"
        else:
            reading = "clearly above the floor -- a real result"
        lines.append(f"    over {years:>5.1f} years (floor {floor:+.2f}):  {reading}")

    lines += [
        "",
        "  A Sharpe target quoted without an evaluation window is not a target.",
        "  Recommended restatement: 'Sharpe > 0.7 measured over at least ten years,",
        "  and above the measured null floor for that same window.'",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markets", type=int, default=400)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    n_markets = 60 if args.quick else args.markets
    lengths = (504, 1260, 2520) if args.quick else LENGTHS

    print(f"Noise floor scaling: {n_markets} null markets at each track length")
    started = time.time()
    table = run(n_markets, lengths, args.workers)
    text = summarise(table)
    print(text)
    print(f"\ntotal {time.time() - started:.0f}s")

    if not args.no_write:
        REPORTS.mkdir(exist_ok=True)
        (REPORTS / "noise_floor_scaling.json").write_text(
            json.dumps(
                {
                    "generated": date.today().isoformat(),
                    "n_markets": n_markets,
                    "z95": Z95,
                    "table": table,
                },
                indent=2,
            )
            + "\n"
        )
        path = REPORTS / f"{date.today().isoformat()}-noise-floor-scaling.txt"
        path.write_text(text + "\n")
        print(f"written: {path.name} and reports/noise_floor_scaling.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
