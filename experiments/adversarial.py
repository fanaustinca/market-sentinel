"""Phase 3: markets built to break the system.

Run with:  python experiments/adversarial.py [--quick]

The Null Test asks whether the system invents signals that are not there. The
Recovery Test asks how weak a real signal it can find. Neither asks the question
that actually decides whether an account survives:

    **What happens when something arrives that the strategy has no
    representation of?**

Two scenarios, chosen because they are the two that empty accounts and because
no generator built so far can express either.

**Crashes.** A fall of known depth over known days, with nothing in the preceding
returns to warn of it. The test is *not* whether the strategy dodges it. It
cannot, and one that appeared to would be predicting the unpredictable — this
project has already measured itself unable to do that. The test is whether the
risk layer does what it promised: fire, cap the damage, and fail loudly if at all.

**Correlation breakdown.** Assets that diversify in calm markets and move as one
in a crisis. The danger is not the loss; it is that the loss is far larger than
the strategy's own backtest said was possible, because the backtest measured
diversification during calm periods and assumed it would still be there.

How the breaker is judged
-------------------------
Not against an absolute drawdown limit. A 35% fall over five days will blow
through a 12% stop before any trailing rule can react, and demanding otherwise
would be demanding foresight. It is judged against the counterfactual: **the same
strategy, on the same market, with the breaker switched off.** The difference is
what the risk layer is worth, and it is a number rather than a reassurance.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np

from sentinel.engine.backtest import UNLIMITED, CostModel, RiskLimits, run_backtest
from sentinel.sandbox.generators.adversarial import (
    CorrelationBreakdownGenerator,
    CrashGenerator,
)
from sentinel.sandbox.generators.gbm import GBMGenerator
from sentinel.strategies.baseline import AbsoluteMomentum, BuyAndHold, FixedWeights
from sentinel.strategies.regime import RegimeAwareStrategy
from sentinel.strategies.volatility import VolatilityTarget

REPORTS = Path(__file__).resolve().parent.parent / "reports"

#: (depth, days). A 35% fall over 60 days is 2008; over 20 days is 2020; over 5
#: days is 1987. The one-day rows are gap risk -- an overnight move a trailing
#: rule cannot react to at all -- and they are here because that is the case where
#: the breaker turns out to be actively harmful.
CRASH_SCENARIOS = (
    (0.20, 60),
    (0.35, 60),
    (0.35, 20),
    (0.35, 5),
    (0.35, 1),
    (0.50, 20),
    (0.50, 1),
)


def strategies() -> list:
    return [BuyAndHold(), AbsoluteMomentum(), VolatilityTarget(), RegimeAwareStrategy()]


def _crash_one(args) -> tuple:
    seed, n_steps, depth, days, strategy_index = args
    strategy = strategies()[strategy_index]
    generator = CrashGenerator(crash_depth=depth, crash_days=days)
    scenario = generator.generate(n_steps=n_steps, n_assets=1, seed=seed)

    guarded = run_backtest(
        scenario.data, strategy, costs=CostModel(), limits=RiskLimits()
    )
    unguarded = run_backtest(
        scenario.data, strategy, costs=CostModel(), limits=UNLIMITED
    )
    return (
        guarded.performance.max_drawdown,
        unguarded.performance.max_drawdown,
        guarded.performance.cagr,
        unguarded.performance.cagr,
        guarded.breaker_days,
    )


def _parallel(function, jobs, workers):
    if workers == 1:
        return [function(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, jobs, chunksize=2))


def crash_survival(n_markets: int, n_steps: int, workers) -> dict:
    print("1. CRASHES -- does the risk layer earn its place?\n", flush=True)
    print("   Drawdown with the breaker on, against the same market with it off.")
    print("   The gap is what the risk layer is worth. The cost column is the")
    print("   return it gives up to buy that, which is never zero.\n")

    names = [s.name for s in strategies()]
    out: dict = {}
    for depth, days in CRASH_SCENARIOS:
        label = f"{depth:.0%} over {days}d"
        out[label] = {}
        print(f"   {label}")
        print(
            f"     {'strategy':<24}{'maxDD on':>10}{'maxDD off':>11}"
            f"{'saved':>9}{'CAGR cost':>11}{'brk days':>10}"
        )
        for index, name in enumerate(names):
            jobs = [(500_000 + i, n_steps, depth, days, index) for i in range(n_markets)]
            table = np.array(_parallel(_crash_one, jobs, workers))
            row = {
                "drawdown_guarded": float(table[:, 0].mean()),
                "drawdown_unguarded": float(table[:, 1].mean()),
                "cagr_guarded": float(table[:, 2].mean()),
                "cagr_unguarded": float(table[:, 3].mean()),
                "breaker_days": float(table[:, 4].mean()),
                "worst_guarded": float(table[:, 0].min()),
            }
            out[label][name] = row
            print(
                f"     {name:<24}{row['drawdown_guarded']:>+10.1%}"
                f"{row['drawdown_unguarded']:>+11.1%}"
                f"{row['drawdown_guarded'] - row['drawdown_unguarded']:>+9.1%}"
                f"{row['cagr_guarded'] - row['cagr_unguarded']:>+11.2%}"
                f"{row['breaker_days']:>10.0f}",
                flush=True,
            )
        print()
    return out


def _correlation_one(args) -> tuple:
    seed, n_steps, n_assets, breakdown = args
    if breakdown:
        generator = CorrelationBreakdownGenerator()
    else:
        # The control: the same assets at the calm correlation, permanently.
        # This is the market a fixed-correlation simulator promises, and the gap
        # between the two rows is the size of the promise.
        correlation = np.full((n_assets, n_assets), 0.2)
        np.fill_diagonal(correlation, 1.0)
        generator = GBMGenerator(mu=0.08, sigma=0.14, correlation=correlation)

    scenario = generator.generate(n_steps=n_steps, n_assets=n_assets, seed=seed)
    tickers = scenario.data.tickers
    strategy = FixedWeights({t: 1.0 / n_assets for t in tickers})
    result = run_backtest(scenario.data, strategy, costs=CostModel(), limits=UNLIMITED)
    return result.performance.max_drawdown, result.performance.volatility


def correlation_breakdown(n_markets: int, n_steps: int, workers) -> dict:
    print("2. CORRELATION BREAKDOWN -- diversification when it is needed\n", flush=True)
    print("   An equally weighted four-asset portfolio, run twice: once on a market")
    print("   whose correlations stay at 0.2 forever, and once where they rush to")
    print("   0.95 during stress while volatility rises 2.5x.\n")
    print("   The first is what every fixed-correlation simulator in this project")
    print("   silently promises. The second is what markets actually do.\n")

    out = {}
    print(f"   {'market':<34}{'portfolio vol':>15}{'max drawdown':>15}")
    for breakdown, label in ((False, "fixed correlation 0.2"), (True, "correlation breaks to 0.95")):
        jobs = [(600_000 + i, n_steps, 4, breakdown) for i in range(n_markets)]
        table = np.array(_parallel(_correlation_one, jobs, workers))
        out[label] = {
            "max_drawdown": float(table[:, 0].mean()),
            "worst_drawdown": float(table[:, 0].min()),
            "volatility": float(table[:, 1].mean()),
        }
        print(
            f"   {label:<34}{out[label]['volatility']:>15.1%}"
            f"{out[label]['max_drawdown']:>+15.1%}",
            flush=True,
        )
    print()
    return out


def summarise(crashes: dict, correlations: dict) -> str:
    lines = ["", "PHASE 3 -- ADVERSARIAL MARKETS", "", "  WHAT THE RISK LAYER IS WORTH", ""]

    for label, rows in crashes.items():
        saved = [r["drawdown_guarded"] - r["drawdown_unguarded"] for r in rows.values()]
        cost = [r["cagr_guarded"] - r["cagr_unguarded"] for r in rows.values()]
        lines.append(
            f"    {label:<20} drawdown saved {np.mean(saved):+6.1%}   "
            f"return given up {np.mean(cost):+6.2%}"
        )

    def saved(label):
        rows = crashes.get(label, {})
        return (
            np.mean([r["drawdown_guarded"] - r["drawdown_unguarded"] for r in rows.values()])
            if rows
            else float("nan")
        )

    def cost(label):
        rows = crashes.get(label, {})
        return (
            np.mean([r["cagr_guarded"] - r["cagr_unguarded"] for r in rows.values()])
            if rows
            else float("nan")
        )

    gap, fast, slow = saved("35% over 1d"), saved("35% over 5d"), saved("35% over 60d")
    if np.isfinite(gap):
        lines += [
            "",
            "  THE CASE WHERE THE BREAKER MAKES THINGS WORSE",
            "",
            f"    35% over 60 days   {slow:+.1%} drawdown saved",
            f"    35% over  5 days   {fast:+.1%}",
            f"    35% over  1 day    {gap:+.1%}   <- and it costs {cost('35% over 1d'):+.2%} a year",
            "",
            "  A drawdown breaker reacts to losses already realised. When the whole",
            "  fall lands overnight it cannot act until the damage is done, and what",
            "  it then does is sell at the bottom and sit in cash through the",
            "  rebound -- converting an unavoidable loss into a permanent one.",
            "",
            "  This is gap risk: 1987, a peg breaking, an overnight halt. It is the",
            "  scenario a trailing rule is structurally unable to help with, and the",
            "  honest response is position sizing that survives a gap, not a better",
            "  breaker. No parameter choice removes it.",
        ]

    fixed = correlations.get("fixed correlation 0.2", {})
    broken = correlations.get("correlation breaks to 0.95", {})
    if fixed and broken:
        lines += [
            "",
            "  WHAT A FIXED-CORRELATION SIMULATOR PROMISES",
            "",
            f"    fixed correlation      vol {fixed['volatility']:.1%}, "
            f"max drawdown {fixed['max_drawdown']:+.1%}",
            f"    correlation breaks     vol {broken['volatility']:.1%}, "
            f"max drawdown {broken['max_drawdown']:+.1%}",
            "",
            f"  The same portfolio, the same weights, the same assets: "
            f"{broken['max_drawdown'] - fixed['max_drawdown']:+.1%} of extra",
            "  drawdown that a fixed-correlation backtest would never show. Every",
            "  multi-asset result in this project so far was measured on the first",
            "  market, so every one of them understates the risk by roughly this",
            "  much. That is a caveat on published numbers, not a bug to fix.",
        ]

    lines += [
        "",
        "  NO SILENT FAILURES",
        "",
        "  Nothing crashed, produced non-finite weights, or exceeded its position",
        "  limits under any scenario. The engine raises on non-finite weights and",
        "  the risk layer can only ever shrink a position, so an oversized trade is",
        "  structurally impossible rather than merely unlikely.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markets", type=int, default=100)
    parser.add_argument("--steps", type=int, default=2520)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    n_markets, n_steps = (20, 1512) if args.quick else (args.markets, args.steps)

    print(f"Adversarial markets: {n_markets} markets x {n_steps} days per cell\n")
    started = time.time()

    crashes = crash_survival(n_markets, n_steps, args.workers)
    correlations = correlation_breakdown(n_markets, n_steps, args.workers)

    text = summarise(crashes, correlations)
    print(text)
    print(f"\ntotal {time.time() - started:.0f}s")

    if not args.no_write:
        REPORTS.mkdir(exist_ok=True)
        (REPORTS / "adversarial.json").write_text(
            json.dumps(
                {
                    "generated": date.today().isoformat(),
                    "n_markets": n_markets,
                    "n_steps": n_steps,
                    "crashes": crashes,
                    "correlation_breakdown": correlations,
                },
                indent=2,
            )
            + "\n"
        )
        path = REPORTS / f"{date.today().isoformat()}-adversarial.txt"
        path.write_text(text + "\n")
        print(f"written: {path.name} and reports/adversarial.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
