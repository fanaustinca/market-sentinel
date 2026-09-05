"""What the system says to hold today, and why.

`plan.md` section 12 recommends staying advisory well into Phase 6 -- recommend
and let a person click -- on the grounds that it is safer, simpler, and keeps the
person learning what each trade means. This module is that advisory output.

The strategies here are the same objects the backtests ran. There is no separate
"live" code path, because a live path that differs from the backtest path is
where undetected bugs live, and the difference can survive indefinitely with
nothing looking wrong. `compute_weights` is called on the current history and the
last row is read; that is the whole implementation, and it is deliberately too
small to hide anything.

Every reading is written to a dated file before the outcome is known. See the
package docstring for why that is a requirement rather than a nicety.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.data.yahoo import fingerprint
from sentinel.sandbox.market import MarketData
from sentinel.strategies.base import Strategy

JOURNAL_DIR = Path(__file__).resolve().parent.parent.parent / "journal"

#: How stale the price data may be before a signal is refused. Markets close for
#: weekends and holidays, so a few days is normal; a fortnight means the feed is
#: broken and a decision taken on it would be taken on the wrong world.
MAX_STALENESS_DAYS = 10


@dataclass
class SignalReport:
    """One strategy's current recommendation, with the evidence behind it."""

    strategy: str
    weights: dict[str, float]
    previous_weights: dict[str, float]
    as_of: str
    reason: str
    noise_floor: float | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def cash(self) -> float:
        return 1.0 - sum(self.weights.values())

    @property
    def is_change(self) -> bool:
        """Whether this differs from yesterday's reading.

        Reported because most days it will be `False`, and a system that only
        speaks when something changes is far easier to keep watching for six
        months than one that produces a full report daily.
        """
        keys = set(self.weights) | set(self.previous_weights)
        return any(
            abs(self.weights.get(k, 0.0) - self.previous_weights.get(k, 0.0)) > 1e-9
            for k in keys
        )

    def describe(self) -> str:
        held = {k: v for k, v in self.weights.items() if v > 1e-6}
        if not held:
            position = "100% cash"
        else:
            position = ", ".join(f"{k} {v:.0%}" for k, v in sorted(held.items()))
            if self.cash > 1e-6:
                position += f", cash {self.cash:.0%}"
        marker = "  <- CHANGED" if self.is_change else ""
        return f"{self.strategy:<24} {position}{marker}\n    {self.reason}"


def _explain(strategy: Strategy, data: MarketData, weights: np.ndarray) -> str:
    """A readable reason for today's position.

    `plan.md` section 4 makes interpretability a requirement, not a preference:
    a system whose decisions cannot be interrogated cannot be maintained,
    debugged, or held on to during a drawdown -- and the drawdown is exactly when
    the explanation is needed.
    """
    ticker = data.tickers[0]
    prices = data.prices[ticker]
    invested = float(np.sum(weights))

    lookback = getattr(strategy, "lookback", None)
    if lookback and len(prices) > lookback:
        trailing = float(np.log(prices.iloc[-1] / prices.iloc[-1 - lookback]))
        direction = "positive" if trailing > 0 else "negative"
        return (
            f"trailing {lookback}-day return is {direction} ({trailing:+.1%}), "
            f"so {'hold' if invested > 0 else 'stand aside'}"
        )

    if hasattr(strategy, "target_volatility"):
        forecast = strategy.forecast_volatility(data, ticker)
        current = float(forecast[-1]) if np.isfinite(forecast[-1]) else float("nan")
        return (
            f"{strategy.forecaster.name} puts volatility at {current:.1%} against a "
            f"{strategy.target_volatility:.0%} target, so hold {invested:.0%}"
        )

    if hasattr(strategy, "classifier"):
        probabilities = strategy.classifier.probabilities(data, ticker=ticker)
        stressed = float(probabilities["p_stressed"].iloc[-1])
        return f"regime model puts P(stressed) at {stressed:.0%}, so hold {invested:.0%}"

    return f"holds {invested:.0%} invested"


def current_signals(
    data: MarketData,
    strategies: list[Strategy],
    noise_floors: dict[str, float] | None = None,
) -> list[SignalReport]:
    """Read each strategy's recommendation for the next period.

    Raises:
        ValueError: if the data is stale. A decision taken on a stale feed is a
            decision about the wrong world, and `plan.md` section 8 requires
            staleness to be *blocked* rather than flagged -- a warning nobody
            reads is not a control.
    """
    latest = data.prices.index[-1]
    age = (pd.Timestamp(date.today()) - pd.Timestamp(latest)).days
    if age > MAX_STALENESS_DAYS:
        raise ValueError(
            f"price data ends {latest.date()}, {age} days ago. Refusing to produce "
            "a signal from a stale feed -- fix the data before trading on it."
        )

    floors = noise_floors or {}
    reports = []
    for strategy in strategies:
        frame = strategy.compute_weights(data)
        matrix = np.nan_to_num(frame.to_numpy(dtype=float), nan=0.0)

        # Row -1 is the decision for the period starting now. Row -2 is what the
        # same call said yesterday, which is what makes "did anything change?"
        # answerable without storing state between runs.
        today, yesterday = matrix[-1], matrix[-2]

        reports.append(
            SignalReport(
                strategy=strategy.name,
                weights={t: float(w) for t, w in zip(frame.columns, today)},
                previous_weights={t: float(w) for t, w in zip(frame.columns, yesterday)},
                as_of=str(latest.date()),
                reason=_explain(strategy, data, today),
                noise_floor=floors.get(strategy.name),
                metadata={"data_age_days": int(age), "n_days": int(data.n_steps)},
            )
        )
    return reports


def write_journal_entry(
    reports: list[SignalReport],
    data: MarketData,
    directory: Path | None = None,
) -> Path:
    """Write today's readings to a dated file, before the outcome is known.

    Refuses to overwrite an existing entry for the same day. Rewriting a
    prediction after the fact -- even innocently, even to fix a typo -- destroys
    the only property that makes the journal worth keeping.
    """
    directory = directory or JOURNAL_DIR
    directory.mkdir(exist_ok=True)
    path = directory / f"{date.today().isoformat()}.json"

    if path.exists():
        raise FileExistsError(
            f"{path.name} already exists. A journal entry is a prediction made "
            "before the outcome was known; rewriting one is not an edit, it is "
            "the removal of the only thing that made it evidence. Delete it "
            "deliberately if you must."
        )

    payload = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "data_as_of": str(data.prices.index[-1].date()),
        "data_fingerprint": fingerprint(data),
        "tickers": data.tickers,
        "signals": [
            {
                "strategy": r.strategy,
                "weights": r.weights,
                "previous_weights": r.previous_weights,
                "cash": r.cash,
                "changed": r.is_change,
                "reason": r.reason,
                "noise_floor": r.noise_floor,
            }
            for r in reports
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
