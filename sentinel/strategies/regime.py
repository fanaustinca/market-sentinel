"""Trading the regime estimate: own the market while it is calm, hold cash otherwise.

This is the strategy the project was designed around, and it is deliberately the
simplest possible use of the classifier. It makes no return forecast at all. It
does not decide *how much* the market will rise -- only whether conditions are
the kind it wants exposure to. Forecasting direction is close to impossible;
recognising a stressed market is merely hard.

Position size is the probability of calm, which gives the behaviour `plan.md`
section 4 asks for without any extra machinery: when the classifier is unsure,
the probability sits near a half and the position is halved automatically. A
model that says "I don't know" and sizes down is worth more than one that is
confidently wrong.

The no-trade band
-----------------
A probability moves a little every day, and following it exactly would rebalance
daily forever. At 5bp each way that is expensive enough to consume the entire
edge -- the Null Test measured a daily-acting momentum rule paying 0.24 of Sharpe
in costs on markets containing nothing.

So the position only moves when the target has drifted more than `band` from what
is currently held. This is a standard no-trade band, and it makes the strategy
path-dependent: today's position depends on the whole history of positions. That
is still perfectly causal -- every input is in the past -- and `check_causality`
verifies it rather than taking the argument's word for it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.ai.regime.classifier import WalkForwardRegimeClassifier
from sentinel.sandbox.market import MarketData
from sentinel.strategies.base import Strategy


class RegimeAwareStrategy(Strategy):
    """Exposure proportional to the probability that the market is calm.

    Args:
        classifier: the regime model. Defaults to the walk-forward HMM.
        band: how far the target must drift from the held position before
            trading. Zero rebalances daily and pays for it.
        max_weight: cap on exposure. The risk layer enforces its own cap
            regardless; this one keeps the strategy's *intent* inside the limit
            rather than relying on being clipped.
        floor: probabilities of calm below this produce no position at all.
            Deliberately asymmetric -- a small position in a market the model
            thinks is probably stressed earns little and costs the trade.
    """

    name = "regime_aware"

    def __init__(
        self,
        classifier: WalkForwardRegimeClassifier | None = None,
        band: float = 0.10,
        max_weight: float = 1.0,
        floor: float = 0.35,
    ) -> None:
        if not 0.0 <= band < 1.0:
            raise ValueError("band must be in [0, 1)")
        if not 0.0 <= floor < 1.0:
            raise ValueError("floor must be in [0, 1)")
        self.classifier = classifier or WalkForwardRegimeClassifier()
        self.band = float(band)
        self.max_weight = float(max_weight)
        self.floor = float(floor)

    def compute_weights(self, data: MarketData) -> pd.DataFrame:
        ticker = data.tickers[0]
        probabilities = self.classifier.probabilities(data, ticker=ticker)
        calm = probabilities["p_calm"].to_numpy(dtype=float)

        desired = np.where(calm >= self.floor, np.minimum(calm, self.max_weight), 0.0)
        # Warmup rows have no estimate. Cash is the right answer for "we do not
        # know yet", and it is the only answer that does not require inventing
        # information the model does not have.
        desired = np.where(np.isfinite(calm), desired, 0.0)

        weights = np.empty(len(desired))
        held = 0.0
        for t, target in enumerate(desired):
            if abs(target - held) > self.band:
                held = float(target)
            weights[t] = held

        return pd.DataFrame({ticker: weights}, index=data.prices.index)


class RegimeGate(RegimeAwareStrategy):
    """All in when calm, all out when stressed. No partial sizing.

    The binary version, kept as a comparison. If probability-weighted sizing does
    not beat a threshold rule, the calibration work behind the probabilities has
    bought nothing and the simpler rule should win on the project's standing
    principle that added machinery must earn its place.
    """

    name = "regime_gate"

    def __init__(
        self,
        classifier: WalkForwardRegimeClassifier | None = None,
        threshold: float = 0.5,
        band: float = 0.10,
    ) -> None:
        super().__init__(classifier=classifier, band=band)
        self.threshold = float(threshold)

    def compute_weights(self, data: MarketData) -> pd.DataFrame:
        ticker = data.tickers[0]
        calm = self.classifier.probabilities(data, ticker=ticker)["p_calm"].to_numpy(dtype=float)
        desired = np.where(np.isfinite(calm) & (calm > self.threshold), 1.0, 0.0)

        weights = np.empty(len(desired))
        held = 0.0
        for t, target in enumerate(desired):
            if abs(target - held) > self.band:
                held = float(target)
            weights[t] = held
        return pd.DataFrame({ticker: weights}, index=data.prices.index)
