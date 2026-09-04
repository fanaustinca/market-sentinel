"""Regime detection walked forward in time, with no unsupervised cheating.

The classifier is deliberately **unsupervised**. It never sees a regime label,
even though the sandbox has them sitting right there. Two reasons, and the second
is the important one:

1.  Real markets have no labels, so a classifier trained on them could never
    leave the sandbox. It would score wonderfully at rung 1 and be unbuildable at
    rung 2, which is the most expensive possible time to find out.
2.  Labels are what the model is *graded against*. A model trained on the answer
    key cannot be tested with it. Keeping the labels strictly on the evaluation
    side means the accuracy and lag numbers in `regime_score` mean what they say.

So the labels flow one way only: `GroundTruth` reaches the scorer and never the
model, which is exactly the separation `MarketData` was built to enforce.

Timing
------
Row `t` is `P(state governing the return from t to t+1 | prices up to and
including t)`. That is one step ahead of the last observed return, because the
state that matters for sizing a position is the one the position will be exposed
to -- not the one that has already happened. It is produced by filtering to `t`
and pushing the result once through the transition matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.ai.regime.hmm import GaussianHMM2State
from sentinel.sandbox.market import MarketData

CALM, STRESSED = 0, 1


class WalkForwardRegimeClassifier:
    """Refits an HMM periodically and filters forward between refits.

    Args:
        min_train: returns required before the first estimate. Regime parameters
            are estimated from the handful of switches present in the sample, so
            a short window sees one or two and fits them as if they were the law.
        retrain_every: days between refits. Far less frequent than the AI's
            return model, because the *parameters* of a regime process --
            typical calm volatility, typical stress duration -- change on a scale
            of years, while which regime is active right now changes weekly. The
            filter tracks the second continuously between refits; only the first
            needs re-estimating.
        lookback: returns each refit uses. `None` means expanding.
    """

    name = "regime_hmm"

    def __init__(
        self,
        min_train: int = 756,
        retrain_every: int = 126,
        lookback: int | None = 1260,
        seed: int = 0,
    ) -> None:
        if min_train < 100:
            raise ValueError("min_train below 100 returns cannot identify two states")
        if retrain_every < 1:
            raise ValueError("retrain_every must be at least 1")
        self.min_train = int(min_train)
        self.retrain_every = int(retrain_every)
        self.lookback = lookback
        self.seed = int(seed)
        self.last_parameters = None

    def probabilities(self, data: MarketData, ticker: str | None = None) -> pd.DataFrame:
        """Causal regime probabilities, indexed like `data.prices`.

        Rows before the model has enough history are `NaN` rather than a guess.
        Filling them with 0.5, or with the eventual answer, would be lookahead
        wearing a friendly face.
        """
        if ticker is None:
            ticker = data.tickers[0]

        prices = data.prices[ticker]
        returns = np.log(prices).diff().to_numpy(dtype=float)[1:]  # r[i] = i -> i+1
        n = len(prices)

        probabilities = np.full((n, 2), np.nan)
        variances = np.full((n, 2), np.nan)
        model: GaussianHMM2State | None = None
        state: np.ndarray | None = None
        fitted_through = -1

        # Row t needs returns r[0:t]; the first row with min_train of them is t.
        for t in range(self.min_train, n):
            if model is None or (t - self.min_train) % self.retrain_every == 0:
                start = 0 if self.lookback is None else max(0, t - self.lookback)
                window = returns[start:t]
                if len(window) < 20 or np.std(window) <= 0:
                    continue
                model = GaussianHMM2State(seed=self.seed).fit(window)
                self.last_parameters = model.params
                # Re-filter the window under the new parameters to land on a
                # current state estimate consistent with them. Carrying the old
                # alpha forward would mix two different models' beliefs.
                state = model.filter(window)[-1]
                fitted_through = t
            elif state is not None:
                # One incremental forward step per new return. Cheap, and
                # identical to re-filtering from scratch under fixed parameters.
                for i in range(fitted_through, t):
                    state = self._advance(model, state, returns[i])
                fitted_through = t

            if model is None or state is None:
                continue

            probabilities[t] = state @ model.params.transition
            # Record the variances *in force at t*, not the ones the final refit
            # will eventually settle on. See `forecast_variance`.
            variances[t] = model.params.variances

        frame = pd.DataFrame(
            probabilities, index=prices.index, columns=["p_calm", "p_stressed"]
        )
        frame["variance_calm"] = variances[:, 0]
        frame["variance_stressed"] = variances[:, 1]
        return frame

    def forecast_variance(self, data: MarketData, ticker: str | None = None) -> pd.Series:
        """Expected variance of the next return, annualised. Causal.

            E[variance] = P(calm)*variance_calm + P(stressed)*variance_stressed

        Both the probabilities and the per-state variances must come from the
        model that was in force on that row. Using `last_parameters` instead --
        the parameters from the most recent refit, applied to every row -- is
        lookahead, because the final refit has seen the whole series.

        That is not hypothetical. It was written that way first, and
        `check_causality` caught it immediately: `LOOKAHEAD DETECTED at row 756,
        past decisions changed by 2.67e-02 when future data arrived`. The bug is
        invisible on inspection -- `last_parameters` reads like an accessor, and
        the resulting volatility forecast looks entirely sensible -- which is
        exactly the class of error the truncation check exists to catch.
        """
        frame = self.probabilities(data, ticker=ticker)
        annualised = frame[["variance_calm", "variance_stressed"]].to_numpy() * 252
        weights = frame[["p_calm", "p_stressed"]].to_numpy()
        return pd.Series(
            (weights * annualised).sum(axis=1), index=frame.index, name="forecast_variance"
        )

    @staticmethod
    def _advance(model: GaussianHMM2State, state: np.ndarray, observation: float) -> np.ndarray:
        """One forward-filter step: predict, weight by the new observation, renormalise."""
        params = model.params
        variance = np.maximum(params.variances, 1e-12)
        likelihood = np.exp(-0.5 * (observation - params.means) ** 2 / variance) / np.sqrt(
            2.0 * np.pi * variance
        )
        weighted = (state @ params.transition) * likelihood
        total = weighted.sum()
        return weighted / total if total > 0 else state
