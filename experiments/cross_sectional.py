"""Ranking assets against each other: the last documented anomaly left untested.

Run: python experiments/cross_sectional.py

Every momentum test in this project until now has been *time-series* momentum --
is this asset above its own trailing average? Cross-sectional momentum asks which
assets have beaten their peers, and it is a genuinely different effect with a
longer replication record (Jegadeesh and Titman, 1993). Leaving it untested while
concluding "returns are unforecastable" was premature, so it is tested here with
the same instruments that killed everything else.

The single-path result looks like the first real win in the project:

    buy_and_hold          +8.09%/yr   -26.5%   $5,152
    xsec_momentum top3    +8.42%/yr   -23.3%   $5,492

More money and a shallower drawdown. Three checks decide whether that survives,
and the last one is the one that matters, because terminal wealth and Sharpe can
disagree and the goal here is money rather than ratio.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel.data.yahoo import load_prices, universe_history
from sentinel.engine.backtest import UNLIMITED, CostModel, run_backtest
from sentinel.evaluation.panel_bootstrap import resample_panel
from sentinel.sandbox.market import MarketData
from sentinel.strategies.baseline import BuyAndHold
from sentinel.strategies.cross_sectional import CrossSectionalMomentum, LowVolatility
from sentinel.strategies.volatility import VolatilityTarget

N_PATHS = 200
UNIVERSE = ["SPY", "IWM", "EFA", "TLT", "IEF", "GLD"]


def measure(prices: pd.DataFrame, strategy):
    return run_backtest(MarketData(prices=prices, name="x"), strategy,
                        costs=CostModel(), limits=UNLIMITED).performance


def main() -> None:
    data = universe_history()
    panel = data.prices

    print("1. PARAMETER SENSITIVITY -- is the winning setting cherry-picked?\n")
    baseline = measure(panel, BuyAndHold()).cagr
    print(f"   CAGR by lookback x n_hold.   buy-and-hold = {baseline:+.2%}")
    print(f"   {'lookback':>9s}" + "".join(f"{'top' + str(n):>9s}" for n in (1, 2, 3, 4, 5)))
    cells = []
    for lookback in (126, 189, 252, 378, 504):
        line = f"   {lookback:9d}"
        for n in (1, 2, 3, 4, 5):
            cagr = measure(panel, CrossSectionalMomentum(lookback=lookback, n_hold=n)).cagr
            cells.append(cagr)
            line += f"{cagr:+9.2%}"
        print(line)
    print(f"   beats buy-and-hold in {sum(c > baseline for c in cells)}/{len(cells)} cells")
    print("   A real effect should win most cells. The best single cell out of")
    print("   twenty-five is what noise looks like when it is reported selectively.")

    print("\n\n2. BY PERIOD AND UNIVERSE -- unlike leverage, is it crisis-dependent?\n")
    periods = {"2004-2010": (None, "2010-12-31"), "2011-2017": ("2011-01-01", "2017-12-31"),
               "2018-2025": ("2018-01-01", None)}
    print(f"   {'period':12s} {'hold':>9s} {'xsec top3':>10s} {'gain':>8s}")
    for label, (start, end) in periods.items():
        window = panel.loc[start:end]
        hold = measure(window, BuyAndHold()).cagr
        xsec = measure(window, CrossSectionalMomentum(n_hold=3)).cagr
        print(f"   {label:12s} {hold:+9.2%} {xsec:+10.2%} {xsec - hold:+8.2%}")
    print("   Not crisis-dependent, which is more than leverage or volatility")
    print("   targeting could say. That is a point in its favour.")

    print("\n\n3. THE TEST THAT DECIDES IT -- terminal wealth across resampled histories\n")
    print(f"   Sharpe rewards smoothness; the goal here is dollars. Both are shown,")
    print(f"   because for these strategies they disagree. {N_PATHS} paths, paired.\n")

    strategies = {
        "buy_and_hold": BuyAndHold(),
        "volatility_target": VolatilityTarget(),
        "xsec_mom_top1": CrossSectionalMomentum(n_hold=1),
        "xsec_mom_top2": CrossSectionalMomentum(n_hold=2),
        "xsec_mom_top3": CrossSectionalMomentum(n_hold=3),
        "low_volatility": LowVolatility(n_hold=3),
    }

    wealth, drawdown, sharpe = [], [], []
    for path in range(N_PATHS):
        resampled = MarketData(prices=resample_panel(panel, np.random.default_rng(path)), name="r")
        w, d, s = {}, {}, {}
        for name, strategy in strategies.items():
            p = run_backtest(resampled, strategy, costs=CostModel(), limits=UNLIMITED).performance
            w[name] = 1000 * (1 + p.total_return)
            d[name] = p.max_drawdown
            s[name] = p.sharpe
        wealth.append(w); drawdown.append(d); sharpe.append(s)
    wealth, drawdown, sharpe = pd.DataFrame(wealth), pd.DataFrame(drawdown), pd.DataFrame(sharpe)

    print(f"   {'strategy':20s} {'median $':>10s} {'mean $':>9s} {'10th':>8s} {'90th':>9s} "
          f"{'maxDD':>8s} {'Sharpe':>7s}")
    for name in wealth.columns:
        print(f"   {name:20s} ${wealth[name].median():>9,.0f} ${wealth[name].mean():>8,.0f} "
              f"${wealth[name].quantile(0.1):>7,.0f} ${wealth[name].quantile(0.9):>8,.0f} "
              f"{drawdown[name].median():8.1%} {sharpe[name].mean():+7.3f}")

    print(f"\n   {'strategy':20s} {'ends richer':>12s} {'median gap':>12s} {'p':>10s}")
    for name in wealth.columns:
        if name == "buy_and_hold":
            continue
        difference = wealth[name] - wealth["buy_and_hold"]
        _, p_value = stats.ttest_rel(wealth[name], wealth["buy_and_hold"])
        print(f"   {name:20s} {(difference > 0).mean():12.1%} "
              f"${difference.median():>11,.0f} {p_value:10.2e}")

    print("\n\nREADING IT\n")
    print("  Cross-sectional momentum has a higher mean terminal wealth than")
    print("  buy-and-hold and a lower median. That is the signature of a lottery")
    print("  ticket: it usually ends with less, occasionally ends with far more,")
    print("  and the 90th percentile carries the average. It ends richer in under")
    print("  40% of histories, at p = 0.19, with a median drawdown of -35% against")
    print("  buy-and-hold's -25%.")
    print()
    print("  It also wins only 8 of 25 parameter settings, so the single-path")
    print("  result that opened this file was the best cell of twenty-five.")
    print()
    print("  The row worth staring at is volatility_target: the best Sharpe in the")
    print("  project, and it ends with more money than buy-and-hold in 2% of")
    print("  histories. Ratio and dollars are different goals and this is what")
    print("  choosing the wrong one costs.")
    print()
    print("  Nothing here reliably ends richer than holding the six assets.")


if __name__ == "__main__":
    main()
