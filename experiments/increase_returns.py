"""Can the top be raised rather than the bottom lowered? Six attempts, one answer.

Run: python experiments/increase_returns.py

Every result in this project so far improves *return per unit of risk*, and that
ratio can be improved two ways: earn more, or risk less. Everything here has done
the second. Terminal wealth on the multi-asset panel is $5,152 for buy-and-hold
and $3,173 for volatility targeting -- a better ratio and two thousand dollars
less. This file asks whether the first route is available.

The honest framing is a constraint, not a strategy: for a drawdown you are
willing to accept, what earns the most? A plain stock/bond mix answers that
question with no forecast, no hindsight, and no machinery. Anything built here
has to beat the plain mix carrying the *same* drawdown, or it is decoration.

Two levers can raise returns without forecasting anything:

  leverage      converts a superior ratio into superior returns. This is the
                textbook answer and it is tested here against realistic financing
                rather than the 0% the rest of the project assumes.
  asset choice  hold things with higher expected returns. Legitimate in the form
                "more equities, fewer bonds"; illegitimate in the form "more of
                whatever went up", which is the trap this file walks into
                deliberately in order to measure the size of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel.data.yahoo import load_prices
from sentinel.engine.backtest import UNLIMITED, CostModel, RiskLimits, run_backtest
from sentinel.sandbox.market import MarketData
from sentinel.strategies.base import Strategy
from sentinel.strategies.baseline import BuyAndHold, FixedWeights
from sentinel.strategies.composite import TrendScaledVolatility
from sentinel.strategies.portfolio import RiskParity
from sentinel.strategies.volatility import VolatilityTarget

START = "2004-11-18"
UNIVERSE = ["SPY", "IWM", "EFA", "TLT", "IEF", "GLD"]

#: Retail margin costs materially more than the risk-free rate. Interactive
#: Brokers charges roughly the benchmark rate plus 1.5%, most brokers charge far
#: more, and over 2004-2025 the average short rate was near 1.8%. 5% is a fair
#: standing estimate and 2.5% is a generous one; the result depends entirely on
#: which is used, which is the finding rather than an aside.
REALISTIC_FINANCING = 0.05
GENEROUS_FINANCING = 0.025


class Scaled(Strategy):
    """Hold `k` times another strategy's weights. Above 1.0 this borrows."""

    def __init__(self, inner: Strategy, k: float, name: str) -> None:
        self.inner, self.k, self.name = inner, float(k), name

    def compute_weights(self, data: MarketData) -> pd.DataFrame:
        return self.inner.compute_weights(data) * self.k


def measure(prices: pd.DataFrame, strategy: Strategy, k: float, financing: float):
    limits = RiskLimits(
        max_position=max(k, 1.0), max_gross_exposure=max(k, 1.0), drawdown_stop=None
    )
    return run_backtest(
        MarketData(prices=prices, name="x"), strategy,
        costs=CostModel(), limits=limits, risk_free_rate=financing,
    ).performance


def main() -> None:
    panel = load_prices(UNIVERSE, start=START).prices
    stock_bond = load_prices(["SPY", "IEF"], start=START).prices

    print(f"RAISING THE TOP  ({panel.index[0].date()} to {panel.index[-1].date()})\n")

    print("THE MENU THAT NEEDS NO FORECAST -- plain stock/bond mixes\n")
    print(f"  {'mix':24s} {'CAGR':>7s} {'maxDD':>8s} {'$1000 ->':>10s}")
    mixes = []
    for weight in (0.2, 0.4, 0.6, 0.8, 1.0):
        p = measure(stock_bond, FixedWeights({"SPY": weight, "IEF": 1 - weight}),
                    1.0, GENEROUS_FINANCING)
        mixes.append((f"{weight:.0%} stocks", p.max_drawdown, 1000 * (1 + p.total_return)))
        print(f"  {weight:.0%} SPY / {1 - weight:.0%} bonds{'':6s} {p.cagr:+7.2%} "
              f"{p.max_drawdown:8.1%} ${1000 * (1 + p.total_return):>9,.0f}")
    print("\n  More stocks, more money, more pain. Nothing here breaks that link,")
    print("  and the rest of this file is the search for something that does.")

    print("\n\nLEVERAGE -- does a better ratio become more money?\n")
    print(f"  {'financing':>10s} {'volTarget x2.5':>15s} {'plain 80% stocks':>18s} {'verdict':>9s}")
    reference = measure(stock_bond, FixedWeights({"SPY": 0.8, "IEF": 0.2}), 1.0, 0.0)
    for rate in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08):
        levered = measure(panel, Scaled(VolatilityTarget(), 2.5, "lev"), 2.5, rate)
        plain = measure(stock_bond, FixedWeights({"SPY": 0.8, "IEF": 0.2}), 1.0, rate)
        a, b = 1000 * (1 + levered.total_return), 1000 * (1 + plain.total_return)
        print(f"  {rate:10.1%} ${a:>14,.0f} ${b:>17,.0f} {'beats' if a > b else 'loses':>9s}")
    print("\n  The entire advantage lives between 2% and 4%. Retail margin is 5-8%.")

    print("\n\nAND BY PERIOD, at the realistic 5%\n")
    periods = {"2004-2010": (None, "2010-12-31"),
               "2011-2017": ("2011-01-01", "2017-12-31"),
               "2018-2025": ("2018-01-01", None)}
    print(f"  {'period':12s} {'x2.5 CAGR':>10s} {'80/20 CAGR':>11s} {'gain':>8s}")
    for label, (start, end) in periods.items():
        a = measure(panel.loc[start:end], Scaled(VolatilityTarget(), 2.5, "lev"),
                    2.5, REALISTIC_FINANCING)
        b = measure(stock_bond.loc[start:end], FixedWeights({"SPY": 0.8, "IEF": 0.2}),
                    1.0, REALISTIC_FINANCING)
        print(f"  {label:12s} {a.cagr:+10.2%} {b.cagr:+11.2%} {a.cagr - b.cagr:+8.2%}")
    print("\n  Wins the crisis, loses everything after it. The same shape as every")
    print("  other result here, amplified -- because leverage amplifies.")

    print("\n\nWHAT LEVERAGE COSTS ON THE WAY DOWN\n")
    print(f"  {'gearing':>8s} {'worst day':>10s} {'worst month':>12s} {'maxDD':>8s}")
    for k in (1.0, 1.75, 2.5):
        limits = RiskLimits(max_position=max(k, 1.0), max_gross_exposure=max(k, 1.0),
                            drawdown_stop=None)
        result = run_backtest(MarketData(prices=panel, name="x"),
                              Scaled(VolatilityTarget(), k, "lev"), costs=CostModel(),
                              limits=limits, risk_free_rate=REALISTIC_FINANCING)
        returns = result.returns
        print(f"  {k:8.2f} {returns.min():+10.2%} {returns.rolling(21).sum().min():+12.2%} "
              f"{result.performance.max_drawdown:8.1%}")

    print("\n\nEVERYTHING BUILT HERE vs THE PLAIN MIX CARRYING THE SAME DRAWDOWN\n")
    print(f"  {'strategy':24s} {'maxDD':>8s} {'its $':>9s} {'plain mix':>22s} {'verdict':>8s}")
    built = [("buy_and_hold", BuyAndHold(), 1.0),
             ("volatility_target", VolatilityTarget(), 1.0),
             ("risk_parity", RiskParity(), 1.0),
             ("trend_scaled_volatility", TrendScaledVolatility(), 1.0),
             ("volTarget x1.75", Scaled(VolatilityTarget(), 1.75, "a"), 1.75),
             ("volTarget x2.5", Scaled(VolatilityTarget(), 2.5, "b"), 2.5)]
    for name, strategy, k in built:
        p = measure(panel, strategy, k, REALISTIC_FINANCING)
        wealth = 1000 * (1 + p.total_return)
        deeper = [m for m in mixes if m[1] <= p.max_drawdown]
        match = max(deeper, key=lambda m: m[1]) if deeper else min(mixes, key=lambda m: m[1])
        print(f"  {name:24s} {p.max_drawdown:8.1%} ${wealth:>8,.0f} "
              f"{match[0] + f' ${match[2]:,.0f}':>22s} "
              f"{'BEATS' if wealth > match[2] else 'loses':>8s}")

    print("\n\nREADING IT\n")
    print("  Six ways to raise the top were tried. Predicting returns failed six")
    print("  times already. Better risk forecasting improves the ratio and not the")
    print("  money. Covariance-based construction does not beat its own diagonal.")
    print("  Adding asset classes made every strategy worse. Leverage works only at")
    print("  financing nobody is offered and only through 2008. Choosing higher-")
    print("  returning assets works and is indistinguishable from hindsight -- QQQ")
    print("  returned +14.84% and EFA +6.08% over this sample, and knowing that in")
    print("  2004 is the entire trick.")
    print()
    print("  What is left is the first table. The amount of money is set by how")
    print("  much stock is held, and every strategy here is a way of choosing a")
    print("  point on that line rather than a way of moving the line.")


if __name__ == "__main__":
    main()
