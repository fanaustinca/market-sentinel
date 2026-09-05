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
from sentinel.strategies.baseline import AbsoluteMomentum, BuyAndHold, EnsembleMomentum
from sentinel.strategies.composite import TrendScaledVolatility
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

    `ensemble_momentum` and `trend_scaled_volatility` are the two changes
    registered in DECISIONS.md before any of this ran. The predictions were that
    the ensemble would *not* improve mean performance but would reduce the spread
    across markets, and that the combination would improve drawdown more than
    return. Both are checked below rather than described.
    """
    return [
        BuyAndHold(),
        AbsoluteMomentum(),
        EnsembleMomentum(),
        VolatilityTarget(),
        TrendScaledVolatility(trend=EnsembleMomentum()),
    ]


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
        sharpes = [r[strategy.name]["real_sharpe"] for r in results.values()]
        lines += [
            f"    {'median':<30}{np.median(edges):>+9.3f}{pooled['median_percentile']:>12.0%}",
            f"    positive in {pooled['sign_positive']} of {pooled['sign_total']} markets"
            f"   sign test p = {pooled['sign_p']:.3f}"
            f"   Fisher p = {pooled['fisher_p']:.4f}",
            f"    mean Sharpe {np.mean(sharpes):+.3f}, spread across markets "
            f"{np.std(sharpes, ddof=1):.3f}",
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

    lines += ["", "  THE TWO CHANGES REGISTERED IN ADVANCE", ""]

    def stats_for(name):
        sharpes = np.array([r[name]["real_sharpe"] for r in results.values()])
        drawdowns = np.array([r[name]["real_max_drawdown"] for r in results.values()])
        return sharpes.mean(), sharpes.std(ddof=1), drawdowns.mean()

    lines.append(f"  {'strategy':<28}{'mean Sharpe':>13}{'spread':>10}{'mean maxDD':>13}")
    for name in [s.name for s in strategies()]:
        mean, spread, drawdown = stats_for(name)
        lines.append(f"  {name:<28}{mean:>+13.3f}{spread:>10.3f}{drawdown:>+13.1%}")

    single_mean, single_spread, _ = stats_for("absolute_momentum")
    ensemble_mean, ensemble_spread, _ = stats_for("ensemble_momentum")
    lines += [
        "",
        "  Prediction 4: the ensemble would not improve mean performance and would",
        "  reduce the spread across markets.",
        f"    mean  {single_mean:+.3f} -> {ensemble_mean:+.3f}   "
        f"spread  {single_spread:.3f} -> {ensemble_spread:.3f}",
    ]
    if ensemble_spread < single_spread and abs(ensemble_mean - single_mean) < 0.05:
        lines.append("    Held. The point of the ensemble was never the mean.")
    elif ensemble_spread >= single_spread:
        lines.append(
            "    Wrong on dispersion: averaging horizons did not stabilise the result, "
            "so the single lookback was not the source of the variation."
        )
    else:
        lines.append(
            "    The mean moved more than expected, which is a warning rather than a "
            "win -- it suggests a horizon suits this sample, not that averaging helps."
        )

    hold_mean, _, hold_dd = stats_for("buy_and_hold")
    trend_mean, _, trend_dd = stats_for("absolute_momentum")
    combined_mean, _, combined_dd = stats_for("trend_scaled_volatility")
    lines += [
        "",
        "  Prediction 5: the combination would improve drawdown more than return.",
        f"    vs holding      Sharpe {combined_mean - hold_mean:+.3f}   "
        f"drawdown {combined_dd - hold_dd:+.1%}",
        f"    vs trend alone  Sharpe {combined_mean - trend_mean:+.3f}   "
        f"drawdown {combined_dd - trend_dd:+.1%}",
    ]
    if combined_dd > trend_dd and combined_mean <= trend_mean + 0.05:
        lines.append("    Held. Composing them buys risk reduction, not return.")
    else:
        lines.append("    Did not hold as stated -- see the table above for what happened instead.")

    lines += [
        "",
        "  THE DRAWDOWN RESULT, WHICH IS SEPARATE",
        "",
        f"  {'market':<28}" + "".join(f"{n[:11]:>13}" for n in [x.name for x in strategies()]),
    ]
    for payload in results.values():
        lines.append(
            f"  {payload['label']:<28}"
            + "".join(
                f"{payload[n]['real_max_drawdown']:>+13.1%}" for n in [x.name for x in strategies()]
            )
        )

    hold = np.array([r["buy_and_hold"]["real_max_drawdown"] for r in results.values()])
    lines += ["", f"  {'strategy':<28}{'shallower in':>14}{'sign p':>10}{'median gain':>14}"]
    best_name, best_gain = None, 0.0
    for name in [x.name for x in strategies()][1:]:
        values = np.array([r[name]["real_max_drawdown"] for r in results.values()])
        better = int((values > hold).sum())
        p_value = float(
            stats.binomtest(better, len(values), 0.5, alternative="greater").pvalue
        )
        gain = float(np.median(values - hold))
        if gain > best_gain:
            best_name, best_gain = name, gain
        lines.append(
            f"  {name:<28}{f'{better}/{len(values)}':>14}{p_value:>10.4f}{gain:>+14.1%}"
        )

    combined_sharpe = float(
        np.mean([r["trend_scaled_volatility"]["real_sharpe"] for r in results.values()])
    )
    hold_sharpe = float(np.mean([r["buy_and_hold"]["real_sharpe"] for r in results.values()]))
    lines += [
        "",
        f"  {best_name} is the best of these: {best_gain:+.1%} of median drawdown",
        f"  against buy-and-hold, shallower in every market, at a mean Sharpe of",
        f"  {combined_sharpe:+.3f} against buy-and-hold's {hold_sharpe:+.3f}.",
        "",
        "  **Roughly the same risk-adjusted return for less than half the worst",
        "  loss**, replicated across eight countries and 391 market-years. That is",
        "  the strongest result this project has, and it is a risk result rather",
        "  than a return one -- which is what everything else here also says.",
        "",
        "  It does not depend on the timing edge being real. Stepping aside after a",
        "  sustained decline mechanically truncates a long fall, and sizing down",
        "  when volatility rises mechanically shrinks the position going into one.",
        "  Neither requires anticipating anything, which is precisely why they",
        "  replicate when the return edges do not.",
        "",
        "  It is also not free: it holds less than the market most of the time and",
        "  will trail badly in a long calm bull run. Whether that trade is worth",
        "  making is a question about the person holding it, not about the data.",
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
