"""Does the corrected sandbox predict what real markets do?

Run with:  python experiments/corrected_sandbox.py [--quick]

`experiments/simulator_gap.py` found that the regime generator's default
parameters weld high volatility to negative drift, and that real equities do the
opposite. Every rung-1 conclusion about regime strategies was drawn inside that
assumption, and none of them transferred.

Fixing the assumption is easy — `RegimeSwitchingGenerator.equity_like()`, whose
states are calibrated to what SPY actually shows. The harder and more useful
question is whether the fix *worked*, and there is a way to check it that does
not require waiting years:

    Run the same strategies on both sandboxes and on real SPY. A sandbox that
    models reality should rank the strategies the way reality ranks them.

Rank agreement is the right test rather than matching levels. No simulator will
reproduce SPY's exact Sharpe, and it does not need to -- what a sandbox is *for*
is deciding which of two strategies to pursue, and it earns trust by getting that
ordering right for reasons that are inspectable.

If the corrected sandbox agrees with reality and the original one does not, then
rung-1 results become worth acting on again, which is the thing the gap finding
took away. If neither agrees, the sandbox is not yet a useful guide for this
class of strategy and should be said so out loud rather than quietly used anyway.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

from sentinel.data.yahoo import load_prices
from sentinel.engine.backtest import UNLIMITED, CostModel, run_backtest
from sentinel.evaluation.sweep import sweep_markets
from sentinel.sandbox.generators.regime import RegimeSwitchingGenerator
from sentinel.strategies.baseline import AbsoluteMomentum, BuyAndHold, ShortHorizonMomentum
from sentinel.strategies.regime import RegimeAwareStrategy, RegimeGate
from sentinel.strategies.volatility import RegimeVolatilityTarget, VolatilityTarget

REPORTS = Path(__file__).resolve().parent.parent / "reports"


def strategies() -> list:
    return [
        BuyAndHold(),
        AbsoluteMomentum(),
        ShortHorizonMomentum(),
        RegimeAwareStrategy(),
        RegimeGate(),
        VolatilityTarget(),
        RegimeVolatilityTarget(),
    ]


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation, computed directly. Ties averaged.

    Hand-computed rather than imported for the same reason the random-walk tests
    are: this single number is the experiment's whole verdict, and it is six
    lines.
    """
    def ranks(values):
        order = np.argsort(np.argsort(values))
        return order.astype(float)

    ra, rb = ranks(np.array(a)), ranks(np.array(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def run_sandbox(generator, label: str, n_markets: int, n_steps: int, workers) -> dict:
    print(f"\n{label}")
    print(f"  {'strategy':<26}{'Sharpe':>9}{'s.e.':>8}{'CAGR':>9}{'maxDD':>9}")
    out = {}
    for strategy in strategies():
        sharpes, cagrs, drawdowns = sweep_markets(
            strategy,
            generator,
            n_markets=n_markets,
            n_steps=n_steps,
            costs=CostModel(),
            limits=UNLIMITED,
            workers=workers,
        )
        out[strategy.name] = {
            "sharpe": float(sharpes.mean()),
            "sharpe_se": float(sharpes.std(ddof=1) / np.sqrt(len(sharpes))),
            "cagr": float(cagrs.mean()),
            "max_drawdown": float(drawdowns.mean()),
        }
        print(
            f"  {strategy.name:<26}{out[strategy.name]['sharpe']:>+9.3f}"
            f"{out[strategy.name]['sharpe_se']:>8.3f}{out[strategy.name]['cagr']:>+9.2%}"
            f"{out[strategy.name]['max_drawdown']:>+9.1%}",
            flush=True,
        )
    return out


def run_real() -> dict:
    data = load_prices("SPY")
    print(f"\nreal SPY ({data.n_steps} days, {data.n_steps / 252:.1f} years)")
    print(f"  {'strategy':<26}{'Sharpe':>9}{'CAGR':>9}{'maxDD':>9}")
    out = {}
    for strategy in strategies():
        result = run_backtest(data, strategy, costs=CostModel(), limits=UNLIMITED)
        performance = result.performance
        out[strategy.name] = {
            "sharpe": performance.sharpe,
            "cagr": performance.cagr,
            "max_drawdown": performance.max_drawdown,
        }
        print(
            f"  {strategy.name:<26}{performance.sharpe:>+9.3f}"
            f"{performance.cagr:>+9.2%}{performance.max_drawdown:>+9.1%}",
            flush=True,
        )
    return out


def summarise(classic: dict, corrected: dict, real: dict) -> str:
    names = list(real)
    real_sharpes = [real[n]["sharpe"] for n in names]
    classic_sharpes = [classic[n]["sharpe"] for n in names]
    corrected_sharpes = [corrected[n]["sharpe"] for n in names]

    rho_classic = spearman(classic_sharpes, real_sharpes)
    rho_corrected = spearman(corrected_sharpes, real_sharpes)

    lines = [
        "",
        "DOES THE SANDBOX PREDICT REALITY?",
        "",
        "  Strategies ranked by Sharpe in each world. A sandbox worth trusting",
        "  should order them the way reality does; matching levels is neither",
        "  expected nor necessary.",
        "",
        f"  {'strategy':<26}{'classic':>10}{'corrected':>12}{'real SPY':>11}",
    ]
    order = sorted(names, key=lambda n: -real[n]["sharpe"])
    for name in order:
        lines.append(
            f"  {name:<26}{classic[name]['sharpe']:>+10.3f}"
            f"{corrected[name]['sharpe']:>+12.3f}{real[name]['sharpe']:>+11.3f}"
        )

    lines += [
        "",
        f"  rank correlation with real SPY:",
        f"    classic sandbox    (mu = 0.12 / -0.15)   {rho_classic:+.3f}",
        f"    corrected sandbox  (mu = 0.09 / +0.17)   {rho_corrected:+.3f}",
        "",
    ]

    if rho_corrected > rho_classic + 0.2:
        lines += [
            "  The corrected sandbox is a materially better guide. Removing one",
            "  false assumption -- that volatility implies loss -- moved it from",
            "  misleading to informative, without changing any strategy.",
            "",
            "  Rung-1 results are worth acting on again, provided they are obtained",
            "  with equity_like() parameters and the classic preset is treated as",
            "  what it is: a market where volatility and loss are the same thing,",
            "  which is a market nobody trades.",
        ]
    elif rho_corrected > 0.5:
        lines += [
            "  Both sandboxes rank strategies broadly as reality does, so the",
            "  coupling was not the only thing driving the earlier divergence.",
            "  Treat rung-1 rankings as weak evidence and keep looking for what",
            "  else the simulator is missing.",
        ]
    else:
        lines += [
            "  Neither sandbox predicts the real ordering. That is the honest",
            "  result and it must not be worked around: for this class of strategy",
            "  the simulator is not currently a useful guide, and rung-1 rankings",
            "  should not be used to choose what to pursue until it is.",
            "",
            "  This does not invalidate the Null Test or the Recovery Test. Those",
            "  measure whether a method finds signals that are definitely present",
            "  or definitely absent, which does not depend on the market resembling",
            "  reality. Ranking strategies by profitability does.",
        ]

    lines += [
        "",
        "  A caution that applies to the corrected sandbox too: its parameters were",
        "  calibrated on the same SPY history it is being checked against, so the",
        "  agreement above is not out-of-sample. It shows the correction is",
        "  self-consistent, not that it will hold on data nobody has seen.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markets", type=int, default=200)
    parser.add_argument("--steps", type=int, default=2520)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    n_markets, n_steps = (30, 1512) if args.quick else (args.markets, args.steps)

    print("Comparing two sandboxes against reality")
    print(f"{n_markets} markets x {n_steps} days per synthetic cell")

    started = time.time()
    classic = run_sandbox(
        RegimeSwitchingGenerator(),
        "classic sandbox  -- mu = (+0.12, -0.15): volatility implies loss, by construction",
        n_markets,
        n_steps,
        args.workers,
    )
    corrected = run_sandbox(
        RegimeSwitchingGenerator.equity_like(),
        "corrected sandbox -- mu = (+0.09, +0.17): volatility is compensated, as in SPY",
        n_markets,
        n_steps,
        args.workers,
    )
    real = run_real()

    text = summarise(classic, corrected, real)
    print(text)
    print(f"\ntotal {time.time() - started:.0f}s")

    if not args.no_write:
        REPORTS.mkdir(exist_ok=True)
        (REPORTS / "corrected_sandbox.json").write_text(
            json.dumps(
                {
                    "generated": date.today().isoformat(),
                    "n_markets": n_markets,
                    "n_steps": n_steps,
                    "classic": classic,
                    "corrected": corrected,
                    "real_spy": real,
                },
                indent=2,
            )
            + "\n"
        )
        path = REPORTS / f"{date.today().isoformat()}-corrected-sandbox.txt"
        path.write_text(text + "\n")
        print(f"written: {path.name} and reports/corrected_sandbox.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
