"""Rung 3 of the Reality Ladder: a portfolio that holds positions over real time.

What this adds that a backtest cannot
--------------------------------------
A backtest and a paper portfolio can produce identical numbers and mean entirely
different things. In a backtest the whole price series exists before the first
decision is taken, so every safeguard against lookahead is a safeguard against
*my own code*, and the project has caught four such bugs already. In a paper
portfolio the future genuinely has not happened. There is no lookahead to guard
against because there is nothing to look ahead at.

That is the only thing this buys, and it is the thing nothing else can buy. It
does not make a strategy better. It makes a result *unfakeable*, which after
three days of finding that promising results were artefacts is worth more than
another backtest.

The discipline that makes it worth keeping
-------------------------------------------
**Entries are append-only and never rewritten.** A ledger that can be edited
after the outcome is known is not evidence. `record` refuses to overwrite a day
that already exists, exactly as the decision journal does.

**The same strategy objects run here as in the backtest.** No parallel code path
for live trading; that gap is where undetected bugs live and it is why
`Strategy.compute_weights` was defined as a pure function of price history.

**Costs are charged.** A paper portfolio that fills at the close for free is a
backtest with extra steps. The same `CostModel` applies, and fills are assumed at
the close of the day the decision was taken -- optimistic, and stated rather than
hidden.

**Every entry carries the data fingerprint.** Yahoo restates history; if the
prices underlying yesterday's decision have changed, that must be visible rather
than silently absorbed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.engine.backtest import CostModel
from sentinel.sandbox.market import MarketData
from sentinel.strategies.base import Strategy

DEFAULT_LEDGER = Path(__file__).resolve().parents[2] / "paper"

#: Starting capital for a paper account. Notional -- it only sets the units.
STARTING_CASH = 10_000.0


@dataclass
class PaperEntry:
    """One day's decision and its consequences, written before the next is known."""

    date: str
    strategy: str
    as_of: str
    fingerprint: str
    target_weights: dict[str, float]
    prices: dict[str, float]
    shares: dict[str, float]
    cash: float
    equity: float
    turnover: float
    cost: float
    note: str = ""
    metadata: dict = field(default_factory=dict)


class PaperPortfolio:
    """A single strategy's paper account, persisted as an append-only ledger.

    Args:
        strategy: the same object a backtest would use.
        ledger: directory holding one JSON file per strategy.
    """

    def __init__(
        self,
        strategy: Strategy,
        ledger: Path | None = None,
        starting_cash: float = STARTING_CASH,
        costs: CostModel | None = None,
    ) -> None:
        self.strategy = strategy
        self.ledger = ledger or DEFAULT_LEDGER
        self.starting_cash = float(starting_cash)
        self.costs = costs or CostModel()

    @property
    def path(self) -> Path:
        return self.ledger / f"{self.strategy.name}.json"

    def history(self) -> list[PaperEntry]:
        if not self.path.exists():
            return []
        return [PaperEntry(**row) for row in json.loads(self.path.read_text())]

    def _cost_of(self, turnover: float, equity: float) -> float:
        """Charge the engine's own cost rate on traded notional.

        Deliberately reads `CostModel.one_way_cost` rather than recomputing from
        the basis-point fields. An earlier version of this method summed
        commission and spread and silently omitted slippage, making paper trading
        two basis points cheaper per unit traded than the backtest it is supposed
        to be validating -- the exact class of divergence this module's docstring
        warns about, introduced inside the module itself.
        """
        return float(turnover * equity * self.costs.one_way_cost)

    def step(self, data: MarketData, note: str = "") -> PaperEntry:
        """Compute today's target, rebalance into it, and return the new entry.

        The decision uses row -1 of the strategy's own output, which by the
        project's timing convention is the weight to hold *from now onward* -- so
        it is formed from prices up to today's close and applied to a return that
        has not happened.
        """
        from sentinel.data.yahoo import fingerprint

        prior = self.history()
        as_of = data.prices.index[-1]
        today = str(date.today())
        if any(entry.date == today for entry in prior):
            raise ValueError(
                f"{self.strategy.name} already has an entry for {today}. "
                "The ledger is append-only; a rewritten prediction is not evidence."
            )

        frame = self.strategy.compute_weights(data)
        targets = np.nan_to_num(frame.to_numpy(dtype=float)[-1], nan=0.0)
        tickers = list(frame.columns)
        prices = data.prices.iloc[-1]

        if prior:
            last = prior[-1]
            shares = {t: float(last.shares.get(t, 0.0)) for t in tickers}
            cash = float(last.cash)
        else:
            shares = {t: 0.0 for t in tickers}
            cash = self.starting_cash

        held = sum(shares[t] * float(prices[t]) for t in tickers)
        equity = held + cash
        if equity <= 0:
            raise ValueError("paper account is wiped out; refusing to continue")

        # Turnover is measured in weight space so it matches the backtest's
        # definition exactly, rather than being redefined here.
        current_weights = {t: shares[t] * float(prices[t]) / equity for t in tickers}
        turnover = float(sum(abs(targets[i] - current_weights[t]) for i, t in enumerate(tickers)))
        cost = self._cost_of(turnover, equity)

        equity_after = equity - cost
        new_shares = {
            t: float(targets[i] * equity_after / float(prices[t])) for i, t in enumerate(tickers)
        }
        new_cash = equity_after - sum(new_shares[t] * float(prices[t]) for t in tickers)

        entry = PaperEntry(
            date=today,
            strategy=self.strategy.name,
            as_of=str(as_of.date()),
            fingerprint=fingerprint(data),
            target_weights={t: float(targets[i]) for i, t in enumerate(tickers)},
            prices={t: float(prices[t]) for t in tickers},
            shares=new_shares,
            cash=float(new_cash),
            equity=float(equity_after),
            turnover=turnover,
            cost=cost,
            note=note,
            metadata={"n_days": int(data.n_steps)},
        )
        self.record(entry)
        return entry

    def record(self, entry: PaperEntry) -> Path:
        """Append one entry. Refuses to replace a day that already exists."""
        self.ledger.mkdir(parents=True, exist_ok=True)
        rows = [asdict(e) for e in self.history()]
        if any(row["date"] == entry.date for row in rows):
            raise ValueError(f"entry for {entry.date} already exists; ledger is append-only")
        rows.append(asdict(entry))
        self.path.write_text(json.dumps(rows, indent=2))
        return self.path

    def equity_curve(self) -> pd.Series:
        entries = self.history()
        if not entries:
            return pd.Series(dtype=float)
        return pd.Series(
            [e.equity for e in entries],
            index=pd.DatetimeIndex([e.date for e in entries]),
            name=self.strategy.name,
        )
