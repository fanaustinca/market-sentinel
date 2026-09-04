"""Today's reading.  Run with:  python -m sentinel.journal [--write]

Advisory output, which is what `plan.md` section 12 recommends for a long time
yet -- recommend, and let a person click. Safer, simpler, and it keeps the person
learning what each position means, which matters more than it sounds when the
alternative is holding an automated position through a drawdown you do not
understand.

Every strategy in the project is shown, including the ones the evidence has
already ruled out. They are the control arm, and the control arm is permanent: a
day when the retired models agree with the candidate is uninformative, and a day
when they disagree is worth looking at.

Each reading is compared against the strategy's measured noise floor from
`reports/noise_floor.json`, so nothing here can be read as more meaningful than
it is.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

from sentinel.data.yahoo import fingerprint, load_prices
from sentinel.journal.signals import current_signals, write_journal_entry
from sentinel.strategies.baseline import AbsoluteMomentum, BuyAndHold, ShortHorizonMomentum
from sentinel.strategies.regime import RegimeAwareStrategy
from sentinel.strategies.volatility import VolatilityTarget

REPORTS = Path(__file__).resolve().parent.parent.parent / "reports"


def load_noise_floors() -> dict[str, float]:
    path = REPORTS / "noise_floor.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {name: row["p95"] for name, row in payload.get("noise_floor", {}).items()}


def strategies() -> list:
    """The candidate first, then the control arm, then the ruled-out.

    Ordered so the reading a person acts on is at the top and the comparisons are
    beneath it, rather than mixed in where a disagreement might be read as a
    committee vote. It is not a vote. `absolute_momentum` is the only one whose
    behaviour replicated across markets, and the rest are there to be checked
    against, not averaged with.
    """
    return [
        AbsoluteMomentum(),
        BuyAndHold(),
        VolatilityTarget(),
        ShortHorizonMomentum(),
        RegimeAwareStrategy(),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--write", action="store_true", help="record the reading to journal/")
    parser.add_argument("--start", default="1993-02-01")
    args = parser.parse_args()

    # Live readings always fetch, never cache. The committed cache exists so a
    # dated report can be reproduced exactly; a signal needs today's prices, and
    # reusing yesterday's would be the staleness failure this project blocks.
    end = (date.today() + timedelta(days=1)).isoformat()
    data = load_prices(args.ticker, start=args.start, end=end, cache=False)

    print(f"{args.ticker} through {data.prices.index[-1].date()}  "
          f"close {float(data.prices.iloc[-1, 0]):,.2f}  fingerprint {fingerprint(data)}")
    print(f"{data.n_steps} days of history\n")

    reports = current_signals(data, strategies(), noise_floors=load_noise_floors())
    for report in reports:
        print(report.describe())
        if report.noise_floor is not None:
            print(f"    (this strategy's null floor is Sharpe {report.noise_floor:+.2f})")
        print()

    changed = [r.strategy for r in reports if r.is_change]
    if changed:
        print(f"CHANGED TODAY: {', '.join(changed)}")
    else:
        print("No changes today. Most days there will be none, which is the point --")
        print("the rule turns over about once a year.")

    print()
    print("Read this as advice to check, not as a result. The measured evidence is")
    print("that this rule roughly halves the worst drawdown (replicated in 8 of 8")
    print("markets) and that its return edge is not statistically distinguishable")
    print("from luck (p = 0.145). See reports/ for both numbers in full.")

    if args.write:
        try:
            path = write_journal_entry(reports, data)
            print(f"\nrecorded: {path}")
        except FileExistsError as exc:
            print(f"\n{exc}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
