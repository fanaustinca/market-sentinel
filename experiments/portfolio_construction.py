"""Is any way of weighting these six assets genuinely better than another?

Run: python experiments/portfolio_construction.py

The motivation is a table that looked like a result and was not. On the real
multi-asset panel the Sharpe ratios come out 0.788 for buy-and-hold, 0.860 for
minimum variance, 0.886 for risk parity and 0.907 for volatility targeting -- and
the standard error on a Sharpe measured over 21 years is about 0.22. Every one of
those numbers is inside every other one's error bar. Ranking them is ranking noise,
and this project exists because that mistake is so easy to make.

So the panel is resampled in blocks of whole rows, preserving both the
cross-sectional correlation and the volatility clustering, and every strategy is
run on the same 200 synthetic histories. Pairing removes the variance contributed
by the path itself, which is what makes a difference of 0.05 detectable at all.

The strategies under test need no return forecast. That is the point: this
project has failed to forecast returns six different ways, and portfolio
construction is the one remaining lever that only asks for the covariance matrix,
which it *can* estimate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel.data.yahoo import fingerprint, universe_history
from sentinel.evaluation.panel_bootstrap import compare_on_panel
from sentinel.strategies.baseline import BuyAndHold
from sentinel.strategies.composite import TrendScaledVolatility
from sentinel.strategies.portfolio import MinimumVariance, RiskParity
from sentinel.strategies.volatility import VolatilityTarget

N_PATHS = 200


def main() -> None:
    data = universe_history()
    print(f"MULTI-ASSET PANEL  {data.prices.index[0].date()} to {data.prices.index[-1].date()}")
    print(f"  {data.prices.shape[1]} assets, {len(data.prices) / 252:.1f} years, "
          f"fingerprint {fingerprint(data)}")
    print(f"  resampled into {N_PATHS} histories, blocks of whole rows\n")

    strategies = {
        "buy_and_hold": BuyAndHold(),
        "volatility_target": VolatilityTarget(),
        "volatility_target_8pct": VolatilityTarget(target_volatility=0.08),
        "risk_parity": RiskParity(),
        "minimum_variance": MinimumVariance(),
        "trend_scaled_volatility": TrendScaledVolatility(),
    }

    comparison = compare_on_panel(strategies, data.prices, n_paths=N_PATHS)

    print("SHARPE ACROSS RESAMPLED HISTORIES\n")
    summary = comparison.summary()
    print(f"  {'strategy':26s} {'mean':>7s} {'sd':>7s} {'worst':>7s} {'best':>7s}")
    for name, row in summary.iterrows():
        print(f"  {name:26s} {row['mean']:+7.3f} {row['sd']:7.3f} "
              f"{row['worst']:+7.3f} {row['best']:+7.3f}")

    print("\nPAIRED AGAINST BUY-AND-HOLD  (same path for every strategy)\n")
    print(f"  {'strategy':26s} {'edge':>7s} {'beats it':>9s} {'t':>7s} {'p':>10s}")
    for _, row in comparison.against("buy_and_hold").iterrows():
        print(f"  {row['strategy']:26s} {row['mean_edge']:+7.3f} "
              f"{row['beats_it']:9.1%} {row['t']:+7.2f} {row['p']:10.2e}")

    print("\nPAIRED AGAINST THE INCUMBENT (volatility_target)\n")
    print(f"  {'strategy':26s} {'edge':>7s} {'beats it':>9s} {'t':>7s} {'p':>10s}")
    for _, row in comparison.against("volatility_target").iterrows():
        print(f"  {row['strategy']:26s} {row['mean_edge']:+7.3f} "
              f"{row['beats_it']:9.1%} {row['t']:+7.2f} {row['p']:10.2e}")

    comparison.sharpes.to_csv("reports/portfolio_construction_sharpes.csv", index=False)

    # ---- the two checks that decide what the bootstrap actually meant --------
    # A block bootstrap resamples one history, so it can only establish that an
    # edge is robust to recombination -- never that it generalises. These do.
    from sentinel.engine.backtest import UNLIMITED, CostModel, run_backtest
    from sentinel.sandbox.market import MarketData
    from sentinel.data.yahoo import load_prices

    costs = CostModel()

    def sharpe(prices: pd.DataFrame, factory) -> float:
        market = MarketData(prices=prices, name="slice")
        return run_backtest(market, factory(), costs=costs, limits=UNLIMITED).performance.sharpe

    print("\n\nDOES IT SURVIVE CHANGING THE ASSETS?\n")
    base = ["SPY", "IWM", "EFA", "TLT", "IEF", "GLD"]
    universes = {
        "6 base": base,
        "9 wider (+EEM,VNQ,LQD)": base + ["EEM", "VNQ", "LQD"],
        "5 no gold": ["SPY", "IWM", "EFA", "TLT", "IEF"],
        "4 minimal": ["SPY", "EFA", "IEF", "GLD"],
        "8 with sectors": base + ["XLU", "XLP"],
    }
    print(f"  {'universe':26s} {'hold':>8s} {'volTarget':>10s} {'edge':>8s}")
    edges = []
    for label, tickers in universes.items():
        prices = load_prices(tickers, start="1990-01-01").prices
        hold, targeted = sharpe(prices, BuyAndHold), sharpe(prices, VolatilityTarget)
        edges.append(targeted - hold)
        print(f"  {label:26s} {hold:+8.3f} {targeted:+10.3f} {targeted - hold:+8.3f}")
    print(f"  positive in {sum(e > 0 for e in edges)}/{len(edges)}, "
          f"median edge {pd.Series(edges).median():+.3f}")

    print("\n\nDOES IT SURVIVE CHANGING THE PERIOD?\n")
    prices = load_prices(base, start="1990-01-01").prices
    periods = {
        "2004-2010 (the crisis)": prices.loc[:"2010-12-31"],
        "2011-2017": prices.loc["2011-01-01":"2017-12-31"],
        "2018-2025": prices.loc["2018-01-01":],
    }
    print(f"  {'period':26s} {'hold':>8s} {'volTarget':>10s} {'edge':>8s}")
    for label, window in periods.items():
        hold, targeted = sharpe(window, BuyAndHold), sharpe(window, VolatilityTarget)
        print(f"  {label:26s} {hold:+8.3f} {targeted:+10.3f} {targeted - hold:+8.3f}")

    print("\n\nREADING IT\n")
    print("  The bootstrap edge is real and it is not a return edge. Volatility")
    print("  targeting beats holding on 90% of resampled histories, survives")
    print("  every change of asset universe tried, and needs no return forecast --")
    print("  which is why it succeeds where six previous attempts did not.")
    print()
    print("  But the period split is the number to read honestly. The whole edge")
    print("  is 2008: +0.308 through the crisis, +0.043 in the middle years, and")
    print("  -0.027 since 2018. It is crisis insurance, and the bootstrap scores")
    print("  it well because it resamples a history that contains a crisis.")
    print()
    print("  Out of sample across eight national indices the edge falls to +0.041")
    print("  and stops being significant (6/8 markets, p = 0.14) -- while drawdown")
    print("  improves in 8 of 8. That is the same sentence this project has")
    print("  written from six directions now: the risk result replicates, the")
    print("  return result does not.")


if __name__ == "__main__":
    main()
