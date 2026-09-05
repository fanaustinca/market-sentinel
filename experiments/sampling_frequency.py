"""Would hourly bars help? Two answers, and they point opposite ways.

Run with:  python experiments/sampling_frequency.py

The intuition is reasonable and very common: daily bars give 250 numbers a year,
hourly bars give about 1,600, so hourly must carry more information and let a
model learn more. It is worth testing rather than asserting, because the
arithmetic disagrees for one question and agrees for the other.

**For predicting returns, sampling faster adds almost nothing.** The precision of
an estimated Sharpe ratio is set by the *calendar span* of the record, not by how
finely it is chopped. Its standard error is roughly `1/sqrt(years)` whether those
years are sliced into days, weeks or months -- because slicing does not create new
market cycles, new recessions, or new crashes. It re-describes the ones already
there. Section 1 measures this on real SPY by going the other way, sampling
weekly and monthly, where the effect is easy to see and does not need data nobody
has.

**For estimating volatility, sampling faster helps a great deal.** A day's
squared return is one very noisy observation of that day's variance. Chopping the
day into `M` pieces and summing their squares gives an estimate roughly `sqrt(M)`
times more precise. This is the standard realised-volatility result and section 2
confirms it by simulation, where the true variance is known.

Which is a neat split, because it lines up exactly with what this project has
already found the hard way: **direction is not forecastable and risk is.** Faster
data helps the half that works and not the half that does not.

Then there is the practical problem, which settles it for now. Free hourly
history runs to about two years. Two years of hourly bars is *fewer* observations
than the daily series already holds, over a sixteenth of the calendar span, and
the noise floor at two years is 1.12 against 0.49 at ten. It would convert a
question that is barely answerable into one that is not answerable at all.
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

from sentinel.data.yahoo import load_prices
from sentinel.evaluation.sweep import sweep_markets
from sentinel.sandbox.generators.bootstrap import BootstrapGenerator
from sentinel.sandbox.market import MarketData
from sentinel.strategies.baseline import AbsoluteMomentum

REPORTS = Path(__file__).resolve().parent.parent / "reports"

#: Bars per year at each sampling frequency, and how many daily bars to skip to
#: produce it. Going coarser rather than finer, because the effect is symmetric
#: and coarse data is available for the full 33 years while hourly is not.
FREQUENCIES = {
    "daily": (1, 252),
    "every 2 days": (2, 126),
    "weekly": (5, 50),
    "fortnightly": (10, 25),
    "monthly": (21, 12),
}


def resample(data: MarketData, step: int) -> MarketData:
    """Keep every `step`-th price. Same calendar span, fewer bars."""
    return MarketData(prices=data.prices.iloc[::step], name=f"{data.name}-{step}")


def noise_floor_by_frequency(n_markets: int, workers) -> dict:
    """The Sharpe luck alone pays, at each sampling frequency, over the same years."""
    print("1. DOES SAMPLING FASTER MAKE THE ANSWER SHARPER?\n", flush=True)
    print("   The same 33 years of SPY, chopped more or less finely. If bar count")
    print("   were what mattered, the daily row would have a much tighter noise")
    print("   floor than the monthly one.\n")

    spy = load_prices("SPY")
    years = spy.n_steps / 252

    print(f"   {'sampling':<16}{'bars':>8}{'bars/yr':>10}{'null sd':>10}{'floor p95':>12}{'1.645/sqrt(yr)':>16}")
    out = {}
    for label, (step, per_year) in FREQUENCIES.items():
        sampled = resample(spy, step)
        # Costs off and the risk layer off: this measures the statistics of the
        # estimate, not the strategy's behaviour, and a cost drag that scales
        # with turnover would confound exactly the comparison being made.
        from sentinel.engine.backtest import CostModel

        generator = BootstrapGenerator.from_market(sampled, block_size=1, demean=True)
        # The lookback is held at one year of *calendar* time at every frequency,
        # so the rule is looking at the same history rather than a different one.
        lookback = max(2, round(252 / step))
        sharpes, _, _ = sweep_markets(
            AbsoluteMomentum(lookback=lookback, rebalance_days=max(1, round(21 / step))),
            generator,
            n_markets=n_markets,
            n_steps=sampled.n_steps,
            costs=CostModel(commission_bps=0, spread_bps=0, slippage_bps=0),
            workers=workers,
            # Must match the sampling frequency. Getting this wrong inflates the
            # reported Sharpe by sqrt(252 / periods) and would make coarse
            # sampling look far noisier than it is -- which is precisely the
            # false result the first version of this experiment produced.
            periods_per_year=per_year,
        )
        out[label] = {
            "bars": int(sampled.n_steps),
            "bars_per_year": per_year,
            "null_sd": float(sharpes.std(ddof=1)),
            "floor_p95": float(np.percentile(sharpes, 95)),
        }
        print(
            f"   {label:<16}{sampled.n_steps:>8}{per_year:>10}"
            f"{out[label]['null_sd']:>10.3f}{out[label]['floor_p95']:>12.3f}"
            f"{1.645 / np.sqrt(years):>16.3f}",
            flush=True,
        )

    spread = max(v["null_sd"] for v in out.values()) - min(v["null_sd"] for v in out.values())
    ratio = max(v["bars"] for v in out.values()) / min(v["bars"] for v in out.values())
    print(f"\n   Bar count varies {ratio:.0f}-fold. The null spread varies by {spread:.3f}.")
    print("   Precision is set by the calendar span, not by how finely it is sliced.\n", flush=True)
    return {"frequencies": out, "years": years, "bar_ratio": float(ratio), "sd_spread": float(spread)}


def volatility_precision(n_days: int = 4000, seed: int = 0) -> dict:
    """How much does chopping the day up help estimate the day's volatility?

    Simulated, because it needs the true variance to compare against. A day is
    generated as `M` equal sub-periods with a known volatility; the estimator is
    the sum of squared sub-returns, and the question is how far it lands from the
    truth as `M` grows.
    """
    print("2. DOES SAMPLING FASTER MAKE THE *RISK* ESTIMATE SHARPER?\n", flush=True)
    print("   Yes, and this is the standard realised-volatility result. A day's")
    print("   squared return is one very noisy look at that day's variance;")
    print("   chopping the day into M pieces is about sqrt(M) times more precise.\n")

    rng = np.random.default_rng(seed)
    true_daily_volatility = 0.16 / np.sqrt(252)

    print(f"   {'bars per day':<16}{'error vs truth':>16}{'improvement':>14}{'sqrt(M) predicts':>18}")
    out = {}
    baseline = None
    for bars in (1, 2, 7, 13, 26, 78):
        # Each sub-period carries variance / bars, so the day's total variance is
        # unchanged and only the estimator's precision differs.
        shocks = rng.standard_normal((n_days, bars)) * true_daily_volatility / np.sqrt(bars)
        estimated = np.sqrt((shocks**2).sum(axis=1))
        error = float(np.std(estimated / true_daily_volatility - 1.0))
        baseline = baseline if baseline is not None else error
        out[bars] = {"relative_error": error, "improvement": baseline / error}
        print(
            f"   {bars:<16}{error:>16.3f}{baseline / error:>14.2f}x{np.sqrt(bars):>17.2f}x"
        )
    print(flush=True)
    return out


def data_reality() -> dict:
    """What hourly history actually exists, which settles the practical question."""
    print("3. WHAT HOURLY DATA ACTUALLY EXISTS\n", flush=True)
    import yfinance as yf

    rows = {}
    for interval, period in (("1d", "max"), ("1h", "730d"), ("30m", "60d")):
        try:
            frame = yf.download(
                "SPY", interval=interval, period=period, auto_adjust=True, progress=False
            )
            if frame is None or frame.empty:
                continue
            span_years = (frame.index[-1] - frame.index[0]).days / 365.25
            rows[interval] = {
                "bars": int(len(frame)),
                "span_years": float(span_years),
                "floor_at_that_span": float(1.645 / np.sqrt(max(span_years, 0.01))),
            }
        except Exception as exc:
            print(f"   {interval}: unavailable ({type(exc).__name__})")

    print(f"   {'interval':<12}{'bars':>10}{'calendar span':>16}{'noise floor':>14}")
    for interval, row in rows.items():
        print(
            f"   {interval:<12}{row['bars']:>10}{row['span_years']:>14.1f}y"
            f"{row['floor_at_that_span']:>14.2f}"
        )
    print(flush=True)
    return rows


def summarise(floors: dict, precision: dict, availability: dict) -> str:
    lines = [
        "",
        "WOULD HOURLY BARS HELP?",
        "",
        "  1. For predicting returns: no.",
        "",
        f"     {'sampling':<16}{'bars':>8}{'null sd':>10}{'floor p95':>12}",
    ]
    for label, row in floors["frequencies"].items():
        lines.append(
            f"     {label:<16}{row['bars']:>8}{row['null_sd']:>10.3f}{row['floor_p95']:>12.3f}"
        )
    lines += [
        "",
        f"     Bar count varies {floors['bar_ratio']:.0f}-fold across those rows. The spread of",
        f"     the null distribution varies by {floors['sd_spread']:.3f}.",
        "",
        "     The precision of an estimated Sharpe is set by how many years the",
        "     record covers, not by how finely those years are sliced. Slicing does",
        "     not create new recessions or new crashes; it re-describes the ones",
        "     already there. Going from daily to hourly multiplies the rows by seven",
        "     and the information about market cycles by roughly nothing.",
        "",
        "  2. For estimating risk: yes, substantially.",
        "",
        f"     {'bars per day':<16}{'error vs truth':>16}{'improvement':>14}",
    ]
    for bars, row in precision.items():
        lines.append(f"     {bars:<16}{row['relative_error']:>16.3f}{row['improvement']:>13.2f}x")
    lines += [
        "",
        "     A day's squared return is one very noisy look at that day's variance.",
        "     Chopping the day into M pieces improves the estimate by about sqrt(M),",
        "     which is the standard realised-volatility result.",
        "",
        "     This matters here more than it might elsewhere, because this project",
        "     has already established that direction is not forecastable and risk is.",
        "     Faster data helps the half that works and not the half that does not.",
        "",
        "  3. And then the practical problem, which settles it for now.",
        "",
        f"     {'interval':<12}{'bars':>10}{'span':>10}{'noise floor':>14}",
    ]
    for interval, row in availability.items():
        lines.append(
            f"     {interval:<12}{row['bars']:>10}{row['span_years']:>9.1f}y"
            f"{row['floor_at_that_span']:>14.2f}"
        )

    daily = availability.get("1d", {})
    hourly = availability.get("1h", {})
    if daily and hourly:
        lines += [
            "",
            f"     Free hourly history is {hourly['span_years']:.1f} years -- "
            f"{hourly['bars']:,} bars against the daily",
            f"     series' {daily['bars']:,} over {daily['span_years']:.0f} years. Switching to hourly would mean",
            "     **fewer observations, over a sixteenth of the calendar span.**",
            "",
            f"     The noise floor at {hourly['span_years']:.1f} years is "
            f"{hourly['floor_at_that_span']:.2f}. Nothing this project has",
            f"     measured comes close to that, so the answer would be 'cannot tell'",
            "     for every strategy, including the ones that work.",
        ]

    lines += [
        "",
        "  WHAT THIS DOES NOT RULE OUT",
        "",
        "  Buying intraday history, or recording it going forward from now, and",
        "  using it *only* to sharpen the volatility forecast while the trend signal",
        "  continues to run on daily bars. That is the version consistent with",
        "  everything measured here, and it is a real improvement rather than a",
        "  hopeful one -- section 2 says roughly how large.",
        "",
        "  It would not make the strategy trade faster. The drawdown result comes",
        "  from stepping aside after a *sustained* decline, which is a slow signal",
        "  by construction. A sharper volatility estimate changes position size, not",
        "  trading frequency.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markets", type=int, default=300)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    started = time.time()
    floors = noise_floor_by_frequency(args.markets, args.workers)
    precision = volatility_precision()
    availability = data_reality()

    text = summarise(floors, precision, availability)
    print(text)
    print(f"\ntotal {time.time() - started:.0f}s")

    if not args.no_write:
        REPORTS.mkdir(exist_ok=True)
        (REPORTS / "sampling_frequency.json").write_text(
            json.dumps(
                {
                    "generated": date.today().isoformat(),
                    "noise_floor_by_frequency": floors,
                    "volatility_precision": {str(k): v for k, v in precision.items()},
                    "data_availability": availability,
                },
                indent=2,
            )
            + "\n"
        )
        path = REPORTS / f"{date.today().isoformat()}-sampling-frequency.txt"
        path.write_text(text + "\n")
        print(f"written: {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
