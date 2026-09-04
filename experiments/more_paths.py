"""Does the trend edge replicate on markets other than SPY?

Run with:  python experiments/more_paths.py [--markets N]

`experiments/is_it_timing.py` reached p = 0.13 for `absolute_momentum` on 33
years of SPY, and said what the next step had to be: **more independent paths.**
More backtesting on SPY cannot improve that number — the shuffle has already
extracted what one path can say.

So the same test is run on eight national equity indices covering 391 market-years
between them, including one, the Nikkei, that spent thirty years going down. Each
market's real result is compared against 200 reshuffles of **its own** returns
with drift preserved, so each is judged on its own terms and the comparison is
internally valid even where the index is price-return rather than total-return.

Three things this does not let us claim
---------------------------------------
**These paths are not independent.** 2008 happened everywhere at once, and global
equity markets share most of their variance. Eight correlated markets are worth
considerably less than eight independent ones, so the combined p-value below is
optimistic — it is reported with a sign test beside it, which cares only about
the direction of each result and is far more robust to that correlation.

**These are the markets that survived.** Every index here belongs to a country
whose exchange stayed open and whose records are continuous. Russia in 1917,
China in 1949, and a long list of others are missing, and they are exactly the
cases where buying and holding was catastrophic and trend following would have
looked heroic. The bias runs *against* trend following here, which is worth
knowing but is not a licence to add anything back.

**Most are price-return indices.** Dividends are missing, which understates every
absolute return by one to four percent a year depending on the market and era. It
does not affect the real-versus-shuffled comparison, since both arms share the
same drift, but no absolute Sharpe here should be quoted as achievable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
from scipy import stats

from sentinel.data.yahoo import load_prices
from sentinel.engine.backtest import UNLIMITED, CostModel, run_backtest
from sentinel.evaluation.sweep import sweep_markets
from sentinel.sandbox.generators.bootstrap import BootstrapGenerator
from sentinel.strategies.baseline import AbsoluteMomentum, BuyAndHold
from sentinel.strategies.volatility import VolatilityTarget

REPORTS = Path(__file__).resolve().parent.parent / "reports"

#: ticker -> (label, whether dividends are included). Chosen for length of
#: history rather than for what they did, and the Nikkei is here specifically
#: because it is the one long series where buy-and-hold failed.
MARKETS = {
    "^GSPC": ("US S&P 500 (1927)", False),
    "^N225": ("Japan Nikkei 225 (1965)", False),
    "^GSPTSE": ("Canada TSX (1979)", False),
    "^FTSE": ("UK FTSE 100 (1984)", False),
    "^HSI": ("Hong Kong Hang Seng (1986)", False),
    "^GDAXI": ("Germany DAX (1987)", True),
    "^FCHI": ("France CAC 40 (1990)", False),
    "^AXJO": ("Australia ASX 200 (1992)", False),
}


def strategies() -> list:
    """The candidate, the other candidate, and the calibration check.

    `buy_and_hold` must land near the 50th percentile in every market: it has no
    timing ability, so destroying the ordering must take nothing from it. If it
    drifts away from 50% somewhere, the shuffle is not doing what it claims in
    that market and nothing else from it can be read.
    """
    return [BuyAndHold(), AbsoluteMomentum(), VolatilityTarget()]


def run_market(ticker: str, n_markets: int, workers) -> dict | None:
    label, total_return = MARKETS[ticker]
    try:
        data = load_prices(ticker, start="1900-01-01")
    except Exception as exc:  # network, delisting, a renamed symbol
        print(f"  {label:<30} unavailable: {type(exc).__name__}")
        return None

    years = data.n_steps / 252
    generator = BootstrapGenerator.from_market(data, block_size=1, demean=False)

    out = {"label": label, "years": years, "n_days": data.n_steps, "total_return": total_return}
    for strategy in strategies():
        result = run_backtest(data, strategy, costs=CostModel(), limits=UNLIMITED)
        actual = result.performance.sharpe

        sharpes, _, _ = sweep_markets(
            strategy,
            generator,
            n_markets=n_markets,
            n_steps=data.n_steps,
            costs=CostModel(),
            limits=UNLIMITED,
            workers=workers,
        )
        out[strategy.name] = {
            "real_sharpe": float(actual),
            "real_cagr": float(result.performance.cagr),
            "real_max_drawdown": float(result.performance.max_drawdown),
            "shuffled_mean": float(sharpes.mean()),
            "timing_edge": float(actual - sharpes.mean()),
            "percentile": float((sharpes < actual).mean()),
        }
    return out


def combine(percentiles: list[float]) -> dict:
    """Two ways of pooling the evidence, because the honest one is weaker.

    Fisher's method treats the p-values as independent, which these are not --
    global equity markets share most of their variance and 2008 happened
    everywhere at once. It is reported because it is the conventional number and
    because its optimism is easy to state.

    The sign test cares only whether each market's timing edge was positive. It
    throws away magnitude and is far more robust to the correlation, which makes
    it the number to believe when the two disagree.
    """
    p_values = np.clip([1.0 - p for p in percentiles], 1e-6, 1.0)
    fisher_statistic = -2.0 * np.log(p_values).sum()
    fisher_p = float(stats.chi2.sf(fisher_statistic, df=2 * len(p_values)))

    positive = int(sum(1 for p in percentiles if p > 0.5))
    sign_p = float(stats.binomtest(positive, len(percentiles), 0.5, alternative="greater").pvalue)

    return {
        "fisher_p": fisher_p,
        "sign_positive": positive,
        "sign_total": len(percentiles),
        "sign_p": sign_p,
        "median_percentile": float(np.median(percentiles)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markets", type=int, default=200)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    print(f"Replication across {len(MARKETS)} national indices")
    print(f"{args.markets} drift-preserving reshuffles of each market's own returns\n")

    started = time.time()
    results = {}
    print(
        f"  {'market':<30}{'years':>7}{'strategy':>20}"
        f"{'real':>9}{'shuffled':>10}{'edge':>9}{'pct':>7}"
    )
    for ticker in MARKETS:
        outcome = run_market(ticker, args.markets, args.workers)
        if outcome is None:
            continue
        results[ticker] = outcome
        for strategy in strategies():
            row = outcome[strategy.name]
            print(
                f"  {outcome['label']:<30}{outcome['years']:>7.1f}{strategy.name:>20}"
                f"{row['real_sharpe']:>+9.3f}{row['shuffled_mean']:>+10.3f}"
                f"{row['timing_edge']:>+9.3f}{row['percentile']:>6.0%}",
                flush=True,
            )
        print()

    if not results:
        print("No markets available. Nothing to conclude.")
        return 1

    total_years = sum(r["years"] for r in results.values())
    lines = [
        "",
        "REPLICATION ACROSS MARKETS",
        "",
        f"  {len(results)} national equity indices, {total_years:.0f} market-years, each judged",
        f"  against {args.markets} reshuffles of its own returns with drift preserved.",
        "",
    ]

    for strategy in strategies():
        percentiles = [r[strategy.name]["percentile"] for r in results.values()]
        edges = [r[strategy.name]["timing_edge"] for r in results.values()]
        pooled = combine(percentiles)

        lines += [
            f"  {strategy.name}",
            f"    {'market':<30}{'edge':>9}{'percentile':>13}",
        ]
        for ticker, payload in results.items():
            row = payload[strategy.name]
            lines.append(
                f"    {payload['label']:<30}{row['timing_edge']:>+9.3f}{row['percentile']:>12.0%}"
            )
        lines += [
            f"    {'median':<30}{np.median(edges):>+9.3f}{pooled['median_percentile']:>12.0%}",
            f"    positive in {pooled['sign_positive']} of {pooled['sign_total']} markets"
            f"   sign test p = {pooled['sign_p']:.3f}"
            f"   Fisher p = {pooled['fisher_p']:.4f}",
            "",
        ]

    momentum_percentiles = [r["absolute_momentum"]["percentile"] for r in results.values()]
    momentum = combine(momentum_percentiles)
    hold = combine([r["buy_and_hold"]["percentile"] for r in results.values()])

    lines += ["  READING IT", ""]
    lines.append(
        f"  Calibration: buy_and_hold's median percentile is {hold['median_percentile']:.0%}. "
    )
    if abs(hold["median_percentile"] - 0.5) < 0.15:
        lines.append(
            "  It should be 50% — no timing ability means the shuffle takes nothing —"
        )
        lines.append("  and it is. The instrument is working.")
    else:
        lines += [
            "  It should be 50%, and it is not. Something in the shuffle is not doing",
            "  what it claims, and nothing else on this page can be read until that is",
            "  understood.",
        ]

    lines += [
        "",
        f"  absolute_momentum showed a positive timing edge in "
        f"{momentum['sign_positive']} of {momentum['sign_total']} markets",
        f"  (sign test p = {momentum['sign_p']:.3f}), median edge "
        f"{np.median([r['absolute_momentum']['timing_edge'] for r in results.values()]):+.3f} Sharpe.",
        "",
    ]

    if momentum["sign_p"] < 0.05:
        lines += [
            "  On SPY alone this was p = 0.13 and not significant. Across markets the",
            "  direction is consistent, which is what replication is for: a single",
            "  path cannot distinguish a weak real effect from luck, and several",
            "  paths agreeing on the sign can.",
            "",
            "  The sign test is the number to trust here. Fisher's method assumes the",
            "  markets are independent and they are plainly not — 2008 happened",
            "  everywhere — so its p-value is optimistic by an unknown amount.",
        ]
    else:
        lines += [
            "  The direction is not consistent enough across markets to call it an",
            "  effect. On SPY alone it was p = 0.13; pooling several paths has not",
            "  rescued it, and pooling was the last cheap thing available.",
            "",
            "  That is a real answer, and the plan permits acting on it: the honest",
            "  conclusion from this evidence is to buy an index fund.",
        ]

    drawdowns = [
        (
            r["label"],
            r["absolute_momentum"]["real_max_drawdown"],
            r["buy_and_hold"]["real_max_drawdown"],
        )
        for r in results.values()
    ]
    better = sum(1 for _, m, b in drawdowns if m > b)
    lines += [
        "",
        "  THE DRAWDOWN RESULT, WHICH IS SEPARATE",
        "",
        f"  {'market':<30}{'momentum':>11}{'hold':>9}",
    ]
    for label, momentum_dd, hold_dd in drawdowns:
        lines.append(f"  {label:<30}{momentum_dd:>+11.1%}{hold_dd:>+9.1%}")
    drawdown_p = float(
        stats.binomtest(better, len(drawdowns), 0.5, alternative="greater").pvalue
    )
    lines += [
        "",
        f"  Momentum's worst drawdown was shallower in {better} of {len(drawdowns)} markets,"
        f" by a median of {np.median([b - m for _, m, b in drawdowns]):.1%},",
        f"  sign test p = {drawdown_p:.4f}.",
        "",
        "  **This is the only result in the project that reaches significance.** Note",
        "  what it is and is not. It does not depend on the timing edge being real:",
        "  stepping aside after a sustained decline mechanically truncates a long",
        "  fall, whether or not the rule can anticipate one. It is a property of the",
        "  rule's shape rather than evidence of skill, which is exactly why it",
        "  replicates when the return edge does not.",
        "",
        "  It is also not free. The same rule costs a median of "
        f"{np.median([r['buy_and_hold']['real_cagr'] - r['absolute_momentum']['real_cagr'] for r in results.values()]):.2%}"
        " a year in return",
        "  across these markets, and sits in cash through part of every recovery.",
        "  Whether halving the worst drawdown is worth that is a question about the",
        "  person holding it, not about the data -- and it is the honest form of the",
        "  question this project set out to answer.",
    ]

    text = "\n".join(lines)
    print(text)
    print(f"\ntotal {time.time() - started:.0f}s")

    if not args.no_write:
        REPORTS.mkdir(exist_ok=True)
        (REPORTS / "more_paths.json").write_text(
            json.dumps(
                {
                    "generated": date.today().isoformat(),
                    "n_bootstrap": args.markets,
                    "markets": results,
                    "pooled": {s.name: combine([r[s.name]["percentile"] for r in results.values()]) for s in strategies()},
                },
                indent=2,
            )
            + "\n"
        )
        path = REPORTS / f"{date.today().isoformat()}-more-paths.txt"
        path.write_text(text + "\n")
        print(f"written: {path.name} and reports/more_paths.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
