"""Rung 2: first contact with reality.

Run with:  python experiments/real_data.py [--no-write]

Everything until now happened inside markets this project built itself. This runs
the identical strategies, engine, costs and risk layer against real ETF history --
the same code paths, with only the data source changed, which is what makes any
difference in behaviour attributable to reality rather than to a second code path.

How a real-market result is judged
----------------------------------
Not by whether it made money. A strategy makes money on plenty of markets
containing nothing, and the Null Test measured exactly how often. The comparison
that means something is against a floor built from **this market's own returns**.

The synthetic noise floor came from Gaussian simulations. Real returns are not
Gaussian -- SPY's daily returns have an excess kurtosis around 11 against a normal
distribution's 0 -- and a floor measured on the wrong distribution is the wrong
floor. So the floor here is recomputed by resampling SPY's own daily returns,
one day at a time, which keeps the real fat tails and destroys the ordering. The
result is a market that is realistic in every way except that there is nothing
left to predict, and whatever a strategy scores on it is what luck alone pays.

Three numbers are therefore reported for every strategy, and all three matter:

    Sharpe on real history
    the bootstrap floor from the same returns, at the same length
    buy and hold over the same period

A strategy must clear all three to have shown anything. Clearing only the floor
means it beat luck but not the index, and the honest response to that is to buy
the index.

What this is not
----------------
It is not evidence the system will make money in future. The strategies' shapes
and parameters -- a 252-day momentum window, a two-state regime model, 756 days
of warmup -- were chosen from convention and from synthetic experiments, not
fitted to this data, so the result is closer to out-of-sample than most published
backtests. But the *choice to test on SPY at all* is itself informed by knowing
SPY went up, and no amount of care inside this file removes that.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.ai.model import WalkForwardClassifier, WalkForwardModel
from sentinel.data.yahoo import fingerprint, load_prices
from sentinel.engine.backtest import UNLIMITED, CostModel, RiskLimits, run_backtest
from sentinel.evaluation.sweep import sweep_markets
from sentinel.sandbox.generators.bootstrap import BootstrapGenerator
from sentinel.sandbox.market import MarketData
from sentinel.data.yahoo import universe_history
from sentinel.strategies.allocation import RegimeRotation
from sentinel.strategies.baseline import (
    AbsoluteMomentum,
    AlwaysCash,
    BuyAndHold,
    FixedWeights,
    ShortHorizonMomentum,
)
from sentinel.strategies.regime import RegimeAwareStrategy, RegimeGate
from sentinel.strategies.volatility import RegimeVolatilityTarget, VolatilityTarget

REPORTS = Path(__file__).resolve().parent.parent / "reports"

#: Periods a defensive system exists for. Reported individually because an
#: average across thirty years hides the only stretches anyone cares about.
CRISES = {
    "dot-com bust": ("2000-03-24", "2002-10-09"),
    "financial crisis": ("2007-10-09", "2009-03-09"),
    "covid crash": ("2020-02-19", "2020-03-23"),
    "2022 bear": ("2022-01-03", "2022-10-12"),
}


def strategies() -> list:
    return [
        BuyAndHold(),
        AlwaysCash(),
        AbsoluteMomentum(),
        ShortHorizonMomentum(),
        RegimeAwareStrategy(),
        RegimeGate(),
        VolatilityTarget(),
        RegimeVolatilityTarget(),
        WalkForwardModel(),
        WalkForwardClassifier(),
    ]


def bootstrap_floor(data: MarketData, strategy, n_markets: int, workers) -> dict:
    """The Sharpe luck alone pays this strategy on this market's own returns.

    Resampled one day at a time and demeaned, so the marginal distribution is
    SPY's and the drift and the ordering are gone.
    """
    generator = BootstrapGenerator.from_market(data, block_size=1, demean=True)
    sharpes, cagrs, _ = sweep_markets(
        strategy,
        generator,
        n_markets=n_markets,
        n_steps=data.n_steps,
        costs=CostModel(),
        limits=UNLIMITED,
        workers=workers,
        n_assets=len(data.tickers),
    )
    return {
        "p95": float(np.percentile(sharpes, 95)),
        "p99": float(np.percentile(sharpes, 99)),
        "mean": float(sharpes.mean()),
        "sd": float(sharpes.std(ddof=1)),
    }


def slice_market(data: MarketData, start: str, end: str) -> MarketData | None:
    window = data.prices.loc[start:end]
    if len(window) < 30:
        return None
    return MarketData(prices=window, name=data.name)


def crisis_table(data: MarketData, results: dict) -> dict:
    """Return through each crisis, computed from the weights already decided.

    The strategies are *not* re-run on the slice. Re-running would give each one
    a fresh warmup starting at the crisis, which is precisely the information it
    would not have had. Instead the equity curve from the full run is sliced,
    which is what the strategy actually experienced.
    """
    out: dict[str, dict[str, float]] = {}
    for label, (start, end) in CRISES.items():
        out[label] = {}
        for name, payload in results.items():
            equity = payload["equity"]
            window = equity.loc[start:end]
            if len(window) < 5:
                continue
            out[label][name] = float(window.iloc[-1] / window.iloc[0] - 1.0)
    return out


def run(n_bootstrap: int, workers, start: str, end: str) -> tuple[dict, dict, MarketData]:
    data = load_prices("SPY", start=start, end=end)
    years = data.n_steps / 252
    print(
        f"SPY {data.prices.index[0].date()} to {data.prices.index[-1].date()}  "
        f"({data.n_steps} days, {years:.1f} years)  fingerprint {fingerprint(data)}\n",
        flush=True,
    )

    # The risk layer is ON here, unlike every measurement run so far. Those runs
    # switched it off to see raw strategy behaviour; this one is meant to
    # represent what would actually have been traded, and the drawdown breaker
    # is part of that.
    limits = RiskLimits()

    results = {}
    print(
        f"  {'strategy':<26}{'Sharpe':>9}{'floor':>9}{'CAGR':>9}"
        f"{'maxDD':>9}{'ddDays':>8}{'turn':>7}{'brk':>6}"
    )
    for strategy in strategies():
        started = time.time()
        outcome = run_backtest(data, strategy, costs=CostModel(), limits=limits)
        floor = bootstrap_floor(data, strategy, n_bootstrap, workers)
        performance = outcome.performance

        results[strategy.name] = {
            "sharpe": performance.sharpe,
            "cagr": performance.cagr,
            "volatility": performance.volatility,
            "max_drawdown": performance.max_drawdown,
            "max_drawdown_days": performance.max_drawdown_days,
            "turnover": outcome.annual_turnover,
            "breaker_days": outcome.breaker_days,
            "bootstrap_floor": floor,
            "beats_floor": performance.sharpe > floor["p95"],
            "equity": outcome.equity,
            "seconds": round(time.time() - started, 1),
        }
        print(
            f"  {strategy.name:<26}{performance.sharpe:>+9.3f}{floor['p95']:>+9.3f}"
            f"{performance.cagr:>+9.2%}{performance.max_drawdown:>+9.1%}"
            f"{performance.max_drawdown_days:>8}{outcome.annual_turnover:>7.1f}"
            f"{outcome.breaker_days:>6}",
            flush=True,
        )

    return results, crisis_table(data, results), data


def summarise(results: dict, crises: dict, data: MarketData) -> str:
    hold = results["buy_and_hold"]
    years = data.n_steps / 252

    lines = [
        "",
        f"RUNG 2 -- REAL HISTORY  (SPY, {data.prices.index[0].date()} to "
        f"{data.prices.index[-1].date()}, {years:.1f} years)",
        f"data fingerprint {fingerprint(data)}",
        "",
        "  Sharpe against two bars: the bootstrap floor from SPY's own returns,",
        "  and buy-and-hold over the same period. Both must be cleared.",
        "",
        f"  {'strategy':<26}{'Sharpe':>9}{'floor':>9}{'vs hold':>9}{'verdict':>26}",
    ]
    for name, payload in results.items():
        if name == "always_cash":
            continue
        floor = payload["bootstrap_floor"]["p95"]
        beats_floor = payload["sharpe"] > floor
        beats_hold = payload["sharpe"] > hold["sharpe"]
        if name == "buy_and_hold":
            verdict = "the bar everything else must clear"
        elif beats_floor and beats_hold:
            verdict = "clears both bars"
        elif beats_floor:
            verdict = "beats luck, not the index"
        else:
            verdict = "indistinguishable from luck"
        lines.append(
            f"  {name:<26}{payload['sharpe']:>+9.3f}{floor:>+9.3f}"
            f"{payload['sharpe'] - hold['sharpe']:>+9.3f}{verdict:>26}"
        )

    lines += [
        "",
        "  DRAWDOWN -- what the system is actually for",
        "",
        f"  {'strategy':<26}{'maxDD':>9}{'underwater':>12}{'CAGR':>9}",
    ]
    for name, payload in results.items():
        lines.append(
            f"  {name:<26}{payload['max_drawdown']:>+9.1%}"
            f"{payload['max_drawdown_days']:>10}d{payload['cagr']:>+9.2%}"
        )

    lines += ["", "  THE PERIODS THAT MATTER", "", "  Return through each crisis:", ""]
    names = [n for n in results if n != "always_cash"]
    lines.append("  " + f"{'crisis':<20}" + "".join(f"{n[:11]:>13}" for n in names))
    for label, row in crises.items():
        line = f"  {label:<20}"
        for name in names:
            line += f"{row.get(name, float('nan')):>+13.1%}"
        lines.append(line)

    best = max(
        (n for n in results if n not in ("always_cash", "buy_and_hold")),
        key=lambda n: results[n]["sharpe"],
    )
    payload = results[best]
    lines += [
        "",
        "  READING THIS HONESTLY",
        "",
        f"  Best non-trivial strategy: {best}, Sharpe {payload['sharpe']:+.3f} against a",
        f"  bootstrap floor of {payload['bootstrap_floor']['p95']:+.3f} and buy-and-hold's "
        f"{hold['sharpe']:+.3f}.",
        "",
    ]
    if payload["sharpe"] <= payload["bootstrap_floor"]["p95"]:
        lines += [
            "  It does not clear its own noise floor. On this evidence the result is",
            "  indistinguishable from luck, and plan.md's kill switch applies: the",
            "  correct action is to buy an index fund, not to tune the strategy.",
        ]
    elif payload["sharpe"] <= hold["sharpe"]:
        lines += [
            "  It beats luck but not the index. A defensive system that trails",
            "  buy-and-hold on a risk-adjusted basis has bought nothing that could",
            "  not be had for free, however much better its drawdowns look.",
        ]
    else:
        lines += [
            "  It clears both bars. That is a genuine result and it is also one",
            "  sample: thirty years of SPY is a single path, and this project's",
            "  founding argument is that a single path proves very little. The",
            "  bootstrap floor is what keeps the claim honest, and the next test is",
            "  paper trading, where the outcome is not yet known to anyone.",
        ]

    lines += [
        "",
        "  Caveats that do not go away:",
        "    - one market, one path, and it is the path everyone already knows went up",
        "    - the universe was chosen today from funds that exist today",
        "    - taxes are not modelled here and are a real drag on turnover",
        f"    - a {years:.0f}-year record still carries a noise floor around "
        f"{1.645 / np.sqrt(years):+.2f} from length alone",
    ]
    return "\n".join(lines)



# ---------------------------------------------------------------------------
# The multi-asset question: what to hold *instead* of equities, not just when to
# leave them.
# ---------------------------------------------------------------------------

def universe_strategies() -> list:
    """Rotation candidates and the controls that make them interpretable.

    The controls are the point. "Rotation beat buy-and-hold" is two claims stuck
    together -- the timing worked, and the defensive asset went up -- and they
    have to be separated or a lucky decade in bonds gets recorded as a working
    model. So the ladder runs:

      SPY alone                 what doing nothing earned
      fixed 60/40               what a single decision, made once, earned
      rotation into cash        the timing alone, with no defensive asset
      rotation into treasuries  the timing plus the rotation
      rotation, momentum-picked adding asset selection on top

    Each rung adds one thing. If the treasury rung is the only one that beats
    SPY, the finding is about treasuries, not about the classifier.
    """
    return [
        BuyAndHold(),
        FixedWeights({"SPY": 0.6, "IEF": 0.4}),
        RegimeRotation(risk_assets=["SPY"], defensive_assets=[]),
        RegimeRotation(risk_assets=["SPY"], defensive_assets=["IEF"]),
        RegimeRotation(risk_assets=["SPY"], defensive_assets=["IEF", "GLD"]),
        RegimeRotation(
            risk_assets=["SPY", "IWM", "EFA"],
            defensive_assets=["IEF", "GLD"],
            regime_ticker="SPY",
            select_by_momentum=True,
        ),
    ]


def run_universe(n_bootstrap: int, workers) -> tuple[dict, dict, MarketData]:
    data = universe_history()
    years = data.n_steps / 252
    print(
        f"\nUNIVERSE {sorted(data.tickers)}\n"
        f"  {data.prices.index[0].date()} to {data.prices.index[-1].date()} "
        f"({data.n_steps} days, {years:.1f} years)  fingerprint {fingerprint(data)}\n"
        "  Note the start date: adding GLD (2004) truncates the panel and throws\n"
        "  away the dot-com bear market. That is a real cost of the extra asset.\n",
        flush=True,
    )

    limits = RiskLimits()
    results = {}
    print(f"  {'strategy':<34}{'Sharpe':>9}{'floor':>9}{'CAGR':>9}{'maxDD':>9}{'turn':>7}")
    for strategy in universe_strategies():
        outcome = run_backtest(data, strategy, costs=CostModel(), limits=limits)
        floor = bootstrap_floor(data, strategy, n_bootstrap, workers)
        performance = outcome.performance
        results[strategy.name] = {
            "sharpe": performance.sharpe,
            "cagr": performance.cagr,
            "volatility": performance.volatility,
            "max_drawdown": performance.max_drawdown,
            "max_drawdown_days": performance.max_drawdown_days,
            "turnover": outcome.annual_turnover,
            "breaker_days": outcome.breaker_days,
            "bootstrap_floor": floor,
            "beats_floor": performance.sharpe > floor["p95"],
            "equity": outcome.equity,
        }
        print(
            f"  {strategy.name:<34}{performance.sharpe:>+9.3f}{floor['p95']:>+9.3f}"
            f"{performance.cagr:>+9.2%}{performance.max_drawdown:>+9.1%}"
            f"{outcome.annual_turnover:>7.1f}",
            flush=True,
        )
    return results, crisis_table(data, results), data


def summarise_universe(results: dict, crises: dict, data: MarketData) -> str:
    hold = results["buy_and_hold"]
    lines = [
        "",
        "MULTI-ASSET ROTATION  "
        f"({data.prices.index[0].date()} to {data.prices.index[-1].date()}, "
        f"{data.n_steps / 252:.1f} years)",
        f"data fingerprint {fingerprint(data)}",
        "",
        "  Each rung adds one thing to the one above it, so a gain can be",
        "  attributed rather than guessed at.",
        "",
        f"  {'strategy':<34}{'Sharpe':>9}{'floor':>9}{'CAGR':>9}{'maxDD':>9}",
    ]
    for name, payload in results.items():
        lines.append(
            f"  {name:<34}{payload['sharpe']:>+9.3f}{payload['bootstrap_floor']['p95']:>+9.3f}"
            f"{payload['cagr']:>+9.2%}{payload['max_drawdown']:>+9.1%}"
        )

    names = list(results)
    lines += ["", "  Return through each crisis:", ""]
    lines.append("  " + f"{'crisis':<20}" + "".join(f"{n[:15]:>17}" for n in names))
    for label, row in crises.items():
        line = f"  {label:<20}"
        for name in names:
            value = row.get(name)
            line += f"{value:>+17.1%}" if value is not None else f"{'--':>17}"
        lines.append(line)

    lines += [
        "",
        "  2022 is the row to read carefully. Treasuries fell alongside equities",
        "  that year, so any rotation that leans on bonds had nowhere defensive to",
        "  go. A version of this strategy that looks good only because the sample",
        "  stops before 2022 has not been tested, it has been selected.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=200, help="bootstrap markets per floor")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--start", default="1993-02-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    started = time.time()
    results, crises, data = run(args.bootstrap, args.workers, args.start, args.end)
    text = summarise(results, crises, data)
    print(text)

    universe_results, universe_crises, universe_data = run_universe(args.bootstrap, args.workers)
    universe_text = summarise_universe(universe_results, universe_crises, universe_data)
    print(universe_text)
    text = text + "\n" + universe_text
    print(f"\ntotal {time.time() - started:.0f}s")

    if not args.no_write:
        REPORTS.mkdir(exist_ok=True)
        def strip(payload_map):
            return {
                name: {k: v for k, v in payload.items() if not isinstance(v, pd.Series)}
                for name, payload in payload_map.items()
            }

        serialisable = strip(results)
        (REPORTS / "real_data.json").write_text(
            json.dumps(
                {
                    "generated": date.today().isoformat(),
                    "ticker": "SPY",
                    "start": str(data.prices.index[0].date()),
                    "end": str(data.prices.index[-1].date()),
                    "n_days": data.n_steps,
                    "fingerprint": fingerprint(data),
                    "n_bootstrap": args.bootstrap,
                    "results": serialisable,
                    "crises": crises,
                    "universe": {
                        "tickers": sorted(universe_data.tickers),
                        "start": str(universe_data.prices.index[0].date()),
                        "end": str(universe_data.prices.index[-1].date()),
                        "fingerprint": fingerprint(universe_data),
                        "results": strip(universe_results),
                        "crises": universe_crises,
                    },
                },
                indent=2,
            )
            + "\n"
        )
        path = REPORTS / f"{date.today().isoformat()}-real-data.txt"
        path.write_text(text + "\n")
        print(f"written: {path.name} and reports/real_data.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
