"""Advance the paper portfolios one day.  Run:  python -m sentinel.paper

The horse race is pre-registered here rather than chosen later. Four strategies
run, and each is included because it is the best candidate under a *different*
definition of best -- which is the honest way to run this, because three days of
measurement established that the definitions disagree:

    buy_and_hold             highest median terminal wealth across 200
                             resampled histories. The benchmark that keeps winning.
    volatility_target        highest Sharpe measured anywhere in the project,
                             and ends richer than holding in 2% of histories.
    xsec_mom_top3            highest *mean* terminal wealth. A lottery ticket:
                             usually behind, occasionally far ahead.
    trend_scaled_volatility  shallowest drawdown, replicated 8/8 across countries.

Committing to all four in advance is what stops the winner being chosen after the
fact. In a year one of them will be ahead, and without this file it would be
tempting to describe whichever it is as the one that was always intended.

Nothing here places an order. It records what each strategy would have done, at
prices that existed, on a date that cannot be edited afterwards.
"""

from __future__ import annotations

import argparse
import sys

from sentinel.data.yahoo import fingerprint, load_prices
from sentinel.paper.portfolio import PaperPortfolio
from sentinel.strategies.baseline import BuyAndHold
from sentinel.strategies.composite import TrendScaledVolatility
from sentinel.strategies.cross_sectional import CrossSectionalMomentum
from sentinel.strategies.volatility import VolatilityTarget

UNIVERSE = ["SPY", "IWM", "EFA", "TLT", "IEF", "GLD"]

#: Refuse to act on a feed older than this. A decision taken on stale prices is a
#: decision about the wrong world; the journal applies the same rule.
MAX_STALENESS_DAYS = 5


def candidates() -> list:
    return [
        BuyAndHold(),
        VolatilityTarget(),
        CrossSectionalMomentum(n_hold=3),
        TrendScaledVolatility(),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Advance the paper portfolios one day.")
    parser.add_argument("--write", action="store_true",
                        help="record the step. Without this, nothing is written.")
    args = parser.parse_args(argv)

    import pandas as pd
    from datetime import date, timedelta

    # `load_prices` defaults to an end date fixed for reproducible backtests.
    # Paper trading is the one caller that must ask for today, and it must skip
    # the cache -- a cached file ending last week would pass every check here
    # while quietly making the decision a week late.
    today = date.today()
    data = load_prices(
        UNIVERSE, start="2004-11-18",
        end=str(today + timedelta(days=1)), cache=False,
    )
    latest = data.prices.index[-1]
    age = (pd.Timestamp(date.today()) - pd.Timestamp(latest)).days
    if age > MAX_STALENESS_DAYS:
        print(f"REFUSING: prices end {latest.date()}, {age} days ago.", file=sys.stderr)
        return 1

    print(f"PAPER PORTFOLIOS  prices through {latest.date()}  "
          f"fingerprint {fingerprint(data)}")
    print(f"{'(dry run -- pass --write to record)' if not args.write else '(recording)'}\n")
    print(f"  {'strategy':26s} {'equity':>10s} {'turnover':>9s}  holdings")

    for strategy in candidates():
        portfolio = PaperPortfolio(strategy)
        frame = strategy.compute_weights(data)
        targets = frame.iloc[-1]
        held = ", ".join(f"{t} {w:.0%}" for t, w in targets.items() if w > 0.005) or "all cash"

        if args.write:
            try:
                entry = portfolio.step(data)
                print(f"  {strategy.name:26s} ${entry.equity:>9,.2f} {entry.turnover:9.2f}  {held}")
            except ValueError as error:
                print(f"  {strategy.name:26s} {'skipped':>10s} {'':9s}  {error}")
        else:
            history = portfolio.history()
            equity = history[-1].equity if history else 10_000.0
            print(f"  {strategy.name:26s} ${equity:>9,.2f} {'-':>9s}  {held}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
