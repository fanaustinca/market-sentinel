"""Automated detection of lookahead bias.

Lookahead bias is the most expensive bug in quantitative finance, and the reason
is that it does not look like a bug. It makes results *better*. A strategy that
can see one day into the future produces a beautiful equity curve, passes every
check a careful person would think to run, and then loses money immediately in
live trading.

Reviewing code for it does not work reliably. It hides in a `shift` of the wrong
sign, a `rolling(center=True)`, a full-sample mean used to standardise a feature,
a join that silently aligns on the wrong day.

So it is tested mechanically instead, by exploiting the one property every honest
strategy must have:

    **A decision made on day t cannot change when data after day t arrives.**

Feed a strategy history up to day k, record what it decided. Feed it more history
and look at day k again. If the answer moved, the strategy used information from
after day k -- whatever the code appeared to say.

This catches lookahead in any strategy, including ones nobody has written yet,
without needing to understand how they work.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sentinel.sandbox.market import MarketData
from sentinel.strategies.base import Strategy


@dataclass(frozen=True)
class CausalityReport:
    """Result of a causality check."""

    strategy: str
    n_checks: int
    max_discrepancy: float
    first_bad_row: int | None
    tolerance: float

    @property
    def is_causal(self) -> bool:
        return self.first_bad_row is None

    def __str__(self) -> str:
        if self.is_causal:
            return (
                f"{self.strategy}: causal across {self.n_checks} truncations "
                f"(max drift {self.max_discrepancy:.2e})"
            )
        return (
            f"{self.strategy}: LOOKAHEAD DETECTED at row {self.first_bad_row} "
            f"-- past decisions changed by {self.max_discrepancy:.2e} when future data arrived"
        )


def truncate(data: MarketData, n_rows: int) -> MarketData:
    """The market as it would have looked after `n_rows` observations."""
    return MarketData(prices=data.prices.iloc[:n_rows], name=data.name)


def check_causality(
    strategy: Strategy,
    data: MarketData,
    cut_points: tuple[int, ...] | None = None,
    tolerance: float = 1e-10,
) -> CausalityReport:
    """Verify that revealing future data never changes a past decision.

    Args:
        cut_points: how many rows to reveal in each trial. Defaults to a spread
            across the second half of the series, since most strategies need a
            warmup before they decide anything at all.
        tolerance: allowed floating-point drift. Deliberately tiny -- a genuine
            leak moves weights by far more than rounding does, and a loose
            tolerance here would defeat the purpose of the check.

    Returns:
        A report naming the first row where a decision changed, if any.
    """
    n = data.n_steps
    if cut_points is None:
        cut_points = tuple(int(n * fraction) for fraction in (0.55, 0.7, 0.85, 1.0))

    full = strategy.compute_weights(data).to_numpy(dtype=float)

    max_discrepancy = 0.0
    first_bad_row: int | None = None

    for cut in cut_points:
        if cut < 2 or cut > n:
            continue
        partial = strategy.compute_weights(truncate(data, cut)).to_numpy(dtype=float)

        # Compare EVERY row the shorter run produced, including its last.
        #
        # An earlier version of this function excluded the final row, reasoning
        # that it decides a period whose outcome the shorter run cannot yet see.
        # That reasoning is wrong and the exclusion was dangerous: a strategy
        # peeking exactly one day ahead differs from an honest one *only* in that
        # final row, so skipping it blinded the detector to the single most
        # important case. `TomorrowPeeker` in the test suite exists to keep it
        # from coming back.
        #
        # There is no legitimate reason for the last row to differ. Row k-1
        # computed from k rows of history uses prices up to k-1, which both runs
        # can see.
        rows = min(len(partial), cut)
        if rows <= 0:
            continue

        a = np.nan_to_num(full[:rows], nan=0.0)
        b = np.nan_to_num(partial[:rows], nan=0.0)
        difference = np.abs(a - b)

        if difference.size:
            max_discrepancy = max(max_discrepancy, float(difference.max()))
            offending = np.flatnonzero(difference.max(axis=1) > tolerance)
            if offending.size and first_bad_row is None:
                first_bad_row = int(offending[0])

    return CausalityReport(
        strategy=strategy.name,
        n_checks=len(cut_points),
        max_discrepancy=max_discrepancy,
        first_bad_row=first_bad_row,
        tolerance=tolerance,
    )
