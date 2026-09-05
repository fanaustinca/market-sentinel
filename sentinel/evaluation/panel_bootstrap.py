"""Comparing multi-asset strategies across many resampled histories.

The problem this exists to solve
--------------------------------
A twenty-one year multi-asset backtest is one path, and the standard error on a
Sharpe ratio measured over `y` years is roughly `1/sqrt(y)` -- about 0.22 here.
So a table showing 0.788, 0.860, 0.886 and 0.907 is showing four numbers that are
statistically indistinguishable, and picking the largest is picking noise. This
project's founding argument applies to its own results, and applying it is the
difference between a finding and a story.

The single-market version of this already exists (`evaluation/null_test.py` and
the eight-country replication). Neither transfers: a national equity index is one
column, and a strategy that allocates *between* six assets cannot run on it.

What is resampled, and what is preserved
----------------------------------------
Blocks of **entire rows** are drawn, so all six assets move together on any
resampled day. That preserves the cross-sectional correlation structure, which is
the thing a portfolio strategy is reacting to -- resampling each asset
independently would destroy exactly the input under test and guarantee that
correlation-aware strategies looked useless.

Blocks rather than single rows preserve volatility clustering, which is the other
input under test. A strategy that sizes down after a volatile week has nothing to
react to if the resampling makes every day independent.

Drift is preserved. The alternative -- demeaning first -- answers "can it time?",
which the project asks elsewhere. The question here is narrower and more
practical: given that these assets are held, does one way of weighting them
deliver more return per unit of risk than another? Both arms are invested, so
being invested is not what is being rewarded.

Every strategy sees the **same** resampled paths, so the comparison is paired and
the noise common to all of them cancels rather than being counted against them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from sentinel.engine.backtest import UNLIMITED, CostModel, run_backtest
from sentinel.sandbox.market import MarketData
from sentinel.strategies.base import Strategy

#: Expected block length in days. Roughly a quarter: long enough to carry a
#: volatility cluster and a correlation shift, short enough that a 21-year sample
#: yields many independent blocks.
DEFAULT_BLOCK = 63


@dataclass(frozen=True)
class PanelComparison:
    """Paired Sharpe ratios for several strategies across resampled histories."""

    sharpes: pd.DataFrame  # rows are paths, columns are strategies

    def summary(self) -> pd.DataFrame:
        frame = self.sharpes
        return pd.DataFrame({
            "mean": frame.mean(),
            "sd": frame.std(),
            "worst": frame.min(),
            "best": frame.max(),
        }).sort_values("mean", ascending=False)

    def against(self, benchmark: str) -> pd.DataFrame:
        """Paired test of every strategy against one named benchmark.

        Paired is what makes this sensitive. The paths themselves dominate the
        variance -- a resampled history containing two crashes scores everything
        badly -- and pairing removes that shared component, which is the same
        reason the Diebold-Mariano test is paired.
        """
        rows = []
        for name in self.sharpes.columns:
            if name == benchmark:
                continue
            difference = self.sharpes[name] - self.sharpes[benchmark]
            t_statistic, p_value = stats.ttest_rel(self.sharpes[name], self.sharpes[benchmark])
            rows.append({
                "strategy": name,
                "mean_edge": float(difference.mean()),
                "sd_edge": float(difference.std()),
                "beats_it": float((difference > 0).mean()),
                "t": float(t_statistic),
                "p": float(p_value),
            })
        return pd.DataFrame(rows).sort_values("mean_edge", ascending=False).reset_index(drop=True)


def resample_panel(
    prices: pd.DataFrame, rng: np.random.Generator, block: int = DEFAULT_BLOCK
) -> pd.DataFrame:
    """One synthetic history of the same length, by drawing blocks of whole rows."""
    returns = np.log(prices).diff().dropna().to_numpy(dtype=float)
    n_rows = len(returns)

    picked = []
    while sum(len(p) for p in picked) < n_rows:
        length = max(1, int(rng.geometric(1.0 / block)))
        start = int(rng.integers(0, n_rows))
        # Wrap around the end rather than truncating, so late rows are drawn as
        # often as early ones. Without this the final block-length of history is
        # systematically under-represented in every path.
        index = (np.arange(start, start + length)) % n_rows
        picked.append(returns[index])

    drawn = np.concatenate(picked)[:n_rows]
    levels = prices.iloc[0].to_numpy(dtype=float) * np.exp(np.vstack([np.zeros(drawn.shape[1]),
                                                                     np.cumsum(drawn, axis=0)]))
    return pd.DataFrame(levels, index=prices.index[: len(levels)], columns=prices.columns)


def compare_on_panel(
    strategies: dict[str, Strategy],
    prices: pd.DataFrame,
    n_paths: int = 200,
    block: int = DEFAULT_BLOCK,
    seed: int = 0,
    costs: CostModel | None = None,
) -> PanelComparison:
    """Run every strategy on `n_paths` resampled histories of the same panel."""
    costs = costs or CostModel()
    records: list[dict[str, float]] = []

    for path in range(n_paths):
        rng = np.random.default_rng(seed + path)
        data = MarketData(prices=resample_panel(prices, rng, block=block), name="resampled")
        row = {}
        for name, strategy in strategies.items():
            result = run_backtest(data, strategy, costs=costs, limits=UNLIMITED)
            row[name] = result.performance.sharpe
        records.append(row)

    return PanelComparison(sharpes=pd.DataFrame(records))
