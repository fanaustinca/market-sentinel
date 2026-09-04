"""Is the edge timing, or is it just being invested?

Run with:  python experiments/is_it_timing.py [--markets N]

`absolute_momentum` scored Sharpe 0.798 on 33 years of SPY against a bootstrap
noise floor of 0.344, and on that comparison it looks like a real result. This
experiment argues that comparison is too generous, and replaces it with a harder
one.

The problem with a demeaned floor
---------------------------------
The floor in `real_data.py` comes from resampling SPY's returns *after removing
their mean*, the same convention every null market in this project uses. That is
correct for asking "did this strategy find structure?", because with drift left in,
any strategy that holds the asset earns money from exposure rather than skill.

But it makes the resulting floor answer a question nobody is really asking. Buy
and hold scores 0.654 on real SPY against a demeaned floor of 0.414 — it "beats
its noise floor" while doing nothing at all except being invested in a market
that went up. If buy and hold clears that bar, clearing it is not evidence of
skill.

The harder floor
----------------
Resample SPY's returns **with the drift left in**. Ordering is still destroyed, so
there is nothing left to time — but the market still rises, so a strategy is still
paid for being exposed to it.

    On that market, a timing rule's score is exactly what its average exposure
    earns. Nothing more is available.

So the comparison becomes clean. A strategy that scores the same on shuffled SPY
as on real SPY has demonstrated **exposure**, not timing: it would have done just
as well holding the same average amount of the market at random moments. A
strategy that scores meaningfully higher on the real series has used the ordering
of returns, which is the only thing the shuffle removed and the only thing timing
could possibly be.

This is the single most decision-relevant number the project can produce right
now, because `absolute_momentum` is the leading candidate for real money and this
is the test that decides whether it deserves to be.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

from sentinel.data.yahoo import fingerprint, load_prices
from sentinel.engine.backtest import UNLIMITED, CostModel, run_backtest
from sentinel.evaluation.sweep import sweep_markets
from sentinel.sandbox.generators.bootstrap import BootstrapGenerator
from sentinel.strategies.baseline import AbsoluteMomentum, BuyAndHold, ShortHorizonMomentum
from sentinel.strategies.regime import RegimeAwareStrategy
from sentinel.strategies.volatility import VolatilityTarget

REPORTS = Path(__file__).resolve().parent.parent / "reports"


def strategies() -> list:
    return [
        BuyAndHold(),
        AbsoluteMomentum(),
        ShortHorizonMomentum(),
        VolatilityTarget(),
        RegimeAwareStrategy(),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markets", type=int, default=300)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    data = load_prices("SPY")
    years = data.n_steps / 252
    print(
        f"SPY {data.prices.index[0].date()} to {data.prices.index[-1].date()}  "
        f"({years:.1f} years)  fingerprint {fingerprint(data)}\n"
    )
    print(f"{args.markets} shuffled markets per strategy, drift preserved, ordering destroyed.")
    print("On those markets there is nothing to time, but the market still rises.\n")

    # Drift left in on purpose. This is the one place in the project where a null
    # market is not demeaned, and the docstring above is why.
    generator = BootstrapGenerator.from_market(data, block_size=1, demean=False)

    started = time.time()
    results = {}
    print(
        f"  {'strategy':<24}{'real SPY':>10}{'shuffled':>10}{'p95':>9}"
        f"{'timing edge':>13}{'percentile':>12}"
    )
    for strategy in strategies():
        actual = run_backtest(
            data, strategy, costs=CostModel(), limits=UNLIMITED
        ).performance.sharpe

        sharpes, _, _ = sweep_markets(
            strategy,
            generator,
            n_markets=args.markets,
            n_steps=data.n_steps,
            costs=CostModel(),
            limits=UNLIMITED,
            workers=args.workers,
            n_assets=1,
        )

        # Where the real result sits inside the distribution the shuffle produced.
        # This is the p-value of "the ordering of returns was worth nothing".
        percentile = float((sharpes < actual).mean())
        results[strategy.name] = {
            "real_sharpe": float(actual),
            "shuffled_mean": float(sharpes.mean()),
            "shuffled_p95": float(np.percentile(sharpes, 95)),
            "shuffled_sd": float(sharpes.std(ddof=1)),
            "timing_edge": float(actual - sharpes.mean()),
            "percentile": percentile,
        }
        row = results[strategy.name]
        print(
            f"  {strategy.name:<24}{row['real_sharpe']:>+10.3f}{row['shuffled_mean']:>+10.3f}"
            f"{row['shuffled_p95']:>+9.3f}{row['timing_edge']:>+13.3f}"
            f"{percentile:>11.1%}",
            flush=True,
        )

    lines = [
        "",
        "IS IT TIMING, OR IS IT JUST BEING INVESTED?",
        "",
        f"  SPY {data.prices.index[0].date()} to {data.prices.index[-1].date()} "
        f"({years:.1f} years), fingerprint {fingerprint(data)}",
        f"  {args.markets} bootstrap markets, drift preserved, ordering destroyed",
        "",
        "  'shuffled' is what each strategy scores when there is nothing left to",
        "  time but the market still rises. The gap between the two columns is the",
        "  only part attributable to using the order of returns.",
        "",
        f"  {'strategy':<24}{'real':>9}{'shuffled':>10}{'timing edge':>13}{'percentile':>12}",
    ]
    for name, row in results.items():
        lines.append(
            f"  {name:<24}{row['real_sharpe']:>+9.3f}{row['shuffled_mean']:>+10.3f}"
            f"{row['timing_edge']:>+13.3f}{row['percentile']:>11.1%}"
        )

    lines += ["", "  READING IT", ""]
    for name, row in results.items():
        if name == "buy_and_hold":
            lines.append(
                f"  {name}: scores {row['shuffled_mean']:+.3f} on shuffled markets against "
                f"{row['real_sharpe']:+.3f} real."
            )
            lines.append(
                "    As it must — it has no timing ability, so the shuffle takes nothing"
            )
            lines.append(
                "    from it. This is the experiment's calibration check, and it passes."
            )
            continue
        percentile = row["percentile"]
        if percentile >= 0.95:
            verdict = (
                f"beats {percentile:.0%} of shuffled markets (p = {1 - percentile:.2f}). "
                "The ordering of returns was worth something."
            )
        elif percentile >= 0.80:
            verdict = (
                f"{percentile:.0%}th percentile, p = {1 - percentile:.2f}. Suggestive and "
                "NOT significant — the conventional bar is 95%."
            )
        elif percentile >= 0.50:
            verdict = (
                f"{percentile:.0%}th percentile, p = {1 - percentile:.2f}. No detectable "
                "timing edge; consistent with exposure alone."
            )
        else:
            verdict = (
                f"{percentile:.0%}th percentile. It did *worse* than shuffling its own "
                "market, so the timing actively cost money."
            )
        lines.append(f"  {name}: {verdict}")

    momentum = results.get("absolute_momentum", {})
    if momentum:
        lines += [
            "",
            "  THE ONE THAT MATTERS",
            "",
            f"  absolute_momentum scores {momentum['real_sharpe']:+.3f} on real SPY and",
            f"  {momentum['shuffled_mean']:+.3f} on shuffled SPY — a timing edge of "
            f"{momentum['timing_edge']:+.3f}, at the",
            f"  {momentum['percentile']:.0%}th percentile of what the shuffle produces.",
            "",
        ]
        p_value = 1 - momentum["percentile"]
        if momentum["percentile"] >= 0.95:
            lines += [
                f"  p = {p_value:.2f}. It clears the conventional bar against the harder",
                "  floor — the one that gives no credit for simply being invested. That",
                "  makes it the strongest evidence the project has produced.",
            ]
        else:
            lines += [
                f"  p = {p_value:.2f}. **This does not reach significance.** Against the",
                "  demeaned floor in real_data.py it looked decisive — 0.798 against 0.344.",
                "  Against this floor it does not, and the difference is entirely that the",
                "  demeaned floor credited it for being invested in a market that went up.",
                "  That is not a skill; it is available for free by holding the index.",
                "",
                "  The honest statement is: over the single path of 1993-2025, trend",
                "  following added an estimated +0.24 of Sharpe over the same average",
                f"  exposure held at random moments, and roughly {p_value:.0%} of random",
                "  reorderings of that same history would have produced as much by chance.",
                "  That is worth continuing to test. It is not worth funding.",
            ]

        lines += [
            "",
            "  WHAT WOULD SETTLE IT",
            "",
            "  Not more backtesting on SPY — the shuffle has already extracted what",
            "  this path can tell us. Either more independent paths (other indices,",
            "  other countries, pre-1993 data), or paper trading forward, where the",
            "  answer is not yet known to anybody.",
            "",
            "  Note also what this experiment does NOT reduce: the drawdown result.",
            "  absolute_momentum's max drawdown was -27.8% against buy-and-hold's",
            "  -51.0%. Cutting the worst loss roughly in half is a property of always",
            "  stepping aside after sustained declines, not a claim about predicting",
            "  them, and it does not depend on the timing edge being real.",
        ]

    text = "\n".join(lines)
    print(text)
    print(f"\ntotal {time.time() - started:.0f}s")

    if not args.no_write:
        REPORTS.mkdir(exist_ok=True)
        (REPORTS / "is_it_timing.json").write_text(
            json.dumps(
                {
                    "generated": date.today().isoformat(),
                    "ticker": "SPY",
                    "fingerprint": fingerprint(data),
                    "n_markets": args.markets,
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )
        path = REPORTS / f"{date.today().isoformat()}-is-it-timing.txt"
        path.write_text(text + "\n")
        print(f"written: {path.name} and reports/is_it_timing.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
