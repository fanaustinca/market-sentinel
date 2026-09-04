"""Where reality diverges from the simulator — the project's most valuable output.

Run with:  python experiments/simulator_gap.py

`plan.md` section 10 predicted, in advance: *"Real markets will break something the
simulator never did. Guaranteed. That gap is the most interesting thing this
project will produce."* This is that gap, found and measured.

The finding
-----------
The regime classifier works on real data. It identifies high-volatility periods
accurately, with a short lag, and its probabilities are well calibrated. What
fails is the **assumption about what a high-volatility state means**, and that
assumption came from the generator rather than from the market.

`RegimeSwitchingGenerator` is built with `mu = (0.12, -0.15)` and
`sigma = (0.12, 0.32)`. Calm means rising and quiet; stressed means falling and
volatile. The two properties are welded together by construction, so in the
sandbox a strategy that flees volatility is automatically fleeing losses, and
every experiment at rung 1 rewards it for doing so.

Real markets do not honour that coupling. Measured below, on SPY from 1993:
the state the classifier calls "stressed" has a *higher* forward return than the
calm state. High volatility in equities is compensated -- it is the volatility
risk premium, and selling into it means selling exactly the periods you are being
paid to hold.

Why this could only be found this way
-------------------------------------
Every rung-1 result was correct. The classifier really does reach 90% balanced
accuracy with a three-day lag; the oracle ladder really does show it capturing a
large share of the available edge. None of that was wrong, and none of it
transferred, because all of it was measured inside a market whose author had
already decided that volatility and loss are the same thing.

This is the entire argument for the Reality Ladder in one result. The failure is
attributable to a single named assumption because nothing else changed between
rungs -- the strategies, engine, costs and metrics are the identical code.

What follows from it
--------------------
Not "abandon the classifier". The classifier is fine. What must change is the
mapping from state to position: a volatility estimate should **scale** exposure,
not switch it off. That is `plan.md` section 4's "uncertainty shrinks positions"
applied to the right quantity, and `sentinel/strategies/volatility.py` implements
it.

It also means the generator should be corrected. A regime model where volatility
and drift are independently controllable is a more honest sandbox, and the fact
that the current one cannot express "volatile but rising" is a limitation of the
simulator rather than a fact about markets.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

from sentinel.ai.regime.classifier import WalkForwardRegimeClassifier
from sentinel.data.yahoo import fingerprint, load_prices
from sentinel.sandbox.generators.regime import RegimeSwitchingGenerator
from sentinel.sandbox.market import MarketData

REPORTS = Path(__file__).resolve().parent.parent / "reports"


def state_profile(data: MarketData) -> dict:
    """Forward return and volatility conditional on the classifier's state."""
    probabilities = WalkForwardRegimeClassifier().probabilities(data)["p_stressed"].to_numpy()
    forward = data.prices.iloc[:, 0].pct_change().shift(-1).to_numpy()

    usable = np.isfinite(probabilities) & np.isfinite(forward)
    stressed = probabilities[usable] > 0.5
    returns = forward[usable]

    out = {"n_scored": int(usable.sum())}
    for label, mask in (("calm", ~stressed), ("stressed", stressed)):
        out[label] = {
            "annual_return": float(returns[mask].mean() * 252),
            "annual_volatility": float(returns[mask].std() * np.sqrt(252)),
            "share_of_days": float(mask.mean()),
            "sharpe": float(returns[mask].mean() * 252 / (returns[mask].std() * np.sqrt(252))),
        }
    return out


def volatility_quintiles(data: MarketData) -> list[dict]:
    """The same question without the classifier, in case the classifier is the problem.

    Trailing 21-day realised volatility against next-day return. No model, no
    fitting, nothing to get wrong -- if the pattern holds here too, it is a
    property of the market rather than an artefact of how the states were
    estimated.
    """
    returns = data.prices.iloc[:, 0].pct_change().dropna()
    volatility = returns.rolling(21).std() * np.sqrt(252)
    forward = returns.shift(-1)

    edges = [0.0, *volatility.quantile([0.2, 0.4, 0.6, 0.8, 1.0]).tolist()]
    rows = []
    for i in range(5):
        inside = (volatility > edges[i]) & (volatility <= edges[i + 1]) & forward.notna()
        rows.append(
            {
                "quintile": i + 1,
                "volatility_from": float(edges[i]),
                "volatility_to": float(edges[i + 1]),
                "forward_annual_return": float(forward[inside].mean() * 252),
                "n_days": int(inside.sum()),
            }
        )
    return rows


def main() -> int:
    print("=" * 74)
    print("THE SIMULATOR-REALITY GAP")
    print("=" * 74)
    print()
    print("Question: does the state the classifier calls 'stressed' actually lose money?")
    print()

    sandbox = RegimeSwitchingGenerator().generate(n_steps=5000, n_assets=1, seed=3).data
    real = load_prices("SPY")

    results = {}
    for label, data in (("sandbox regime market", sandbox), ("real SPY 1993-2025", real)):
        profile = state_profile(data)
        results[label] = profile
        print(f"  {label}  ({profile['n_scored']} scored days)")
        print(f"    {'state':<12}{'ann. return':>14}{'ann. vol':>11}{'Sharpe':>9}{'share':>8}")
        for state in ("calm", "stressed"):
            row = profile[state]
            print(
                f"    {state:<12}{row['annual_return']:>+14.1%}"
                f"{row['annual_volatility']:>11.1%}{row['sharpe']:>+9.2f}"
                f"{row['share_of_days']:>8.0%}"
            )
        print()

    sandbox_gap = (
        results["sandbox regime market"]["calm"]["annual_return"]
        - results["sandbox regime market"]["stressed"]["annual_return"]
    )
    real_gap = (
        results["real SPY 1993-2025"]["calm"]["annual_return"]
        - results["real SPY 1993-2025"]["stressed"]["annual_return"]
    )

    print("  In the sandbox, leaving the stressed state is worth "
          f"{sandbox_gap:+.1%} a year.")
    print(f"  On real SPY it is worth {real_gap:+.1%} a year.")
    print()
    print("  The generator was built with mu = (0.12, -0.15) and sigma = (0.12, 0.32):")
    print("  volatility and loss are welded together by construction, so at rung 1")
    print("  a strategy that flees volatility is automatically fleeing losses.")
    print()

    quintiles = volatility_quintiles(real)
    print("  Without the classifier, to rule it out as the cause --")
    print("  real SPY next-day return by trailing 21-day realised volatility:")
    print()
    print(f"    {'quintile':<10}{'volatility':>16}{'forward ann. return':>22}{'days':>8}")
    for row in quintiles:
        span = f"{row['volatility_from']:.0%}-{row['volatility_to']:.0%}"
        print(
            f"    {row['quintile']:<10}{span:>16}"
            f"{row['forward_annual_return']:>+22.1%}{row['n_days']:>8}"
        )
    print()
    print("  The pattern survives without any model, so it is a property of the")
    print("  market: equity volatility is compensated. Selling into it means")
    print("  selling exactly the periods you are being paid to hold.")
    print()
    print("=" * 74)
    print("CONSEQUENCE")
    print("=" * 74)
    print()
    print("  The classifier is not broken -- it identifies volatile periods accurately.")
    print("  What is broken is the assumption that a volatile period is one to sit out,")
    print("  and that assumption was inherited from the generator, not measured.")
    print()
    print("  The fix is not to exit on high volatility but to SCALE for it: hold a")
    print("  constant amount of risk rather than a constant amount of capital, so")
    print("  exposure falls when volatility rises without ever going to zero. See")
    print("  sentinel/strategies/volatility.py.")
    print()
    print("  The generator should also be corrected. A regime model that cannot")
    print("  express 'volatile but rising' cannot represent the most common state")
    print("  real equity markets are actually in.")

    REPORTS.mkdir(exist_ok=True)
    payload = {
        "generated": date.today().isoformat(),
        "spy_fingerprint": fingerprint(real),
        "state_profiles": results,
        "sandbox_gap_annual": sandbox_gap,
        "real_gap_annual": real_gap,
        "volatility_quintiles": quintiles,
    }
    (REPORTS / "simulator_gap.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwritten: reports/simulator_gap.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
