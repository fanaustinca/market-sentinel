"""The AI: a walk-forward supervised learner that trades its own predictions.

This is deliberately the *canonical* retail quant design -- features in, next-day
return out, position sized by the prediction -- because that is exactly what makes
it the right first subject for the Null Test. It is what almost everyone builds,
and it is how almost everyone loses money. Testing our own version of it honestly
is worth more than testing something exotic.

How it stays causal
-------------------
Retraining walks forward. At row `t` the model may be fitted only on examples
whose outcome was already observable at `t`.

The label for row `i` is the return from `i` to `i + 1`, which is not known until
day `i + 1`. So a model used on day `t` may train on labels up to and including
`y[t - 1]`, and no further. That single off-by-one is the difference between a
system that works and one that quietly reads tomorrow's newspaper, and it is
enforced here and verified in `tests/test_no_lookahead.py`.

Why not deep learning
---------------------
Gradient-boosted trees are the right tool for this shape of problem: tabular
features, tens of thousands of rows, a signal-to-noise ratio so poor that model
capacity is far more likely to memorise noise than to find structure. They train
in milliseconds, which makes walk-forward retraining practical, and they report
which features they used, which keeps the system interpretable. A neural network
would be slower, less transparent, and -- at this data volume -- worse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sentinel.features.build import DEFAULT_WINDOWS, build_features, feature_warmup
from sentinel.sandbox.market import MarketData
from sentinel.strategies.base import Strategy


def _make_regressor(seed: int):
    """A deliberately modest gradient-boosting model.

    The hyperparameters are conservative on purpose. Given noisy financial data,
    a deeper or longer-trained model does not find more signal -- it memorises
    more noise, and does so with increasing confidence. Capacity here is a
    liability, not an asset.
    """
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        verbose=-1,
        n_jobs=1,
    )


@dataclass(frozen=True)
class SizingRule:
    """Turns a predicted return into a position size.

    The prediction is divided by recent volatility, so the same forecast produces
    a smaller position in a turbulent market than in a calm one. `aggression`
    scales the whole thing; `max_weight` caps it.

    A weak prediction therefore produces a small position automatically. That is
    the intended behaviour -- a model that says "I don't know" and sizes down is
    worth considerably more than one that is confidently wrong.
    """

    aggression: float = 1.0
    max_weight: float = 1.0
    long_only: bool = True

    def size(self, predicted_return: np.ndarray, volatility: np.ndarray) -> np.ndarray:
        safe_vol = np.where(volatility > 1e-8, volatility, np.nan)
        raw = self.aggression * predicted_return / safe_vol
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        if self.long_only:
            raw = np.maximum(raw, 0.0)
        return np.clip(raw, -self.max_weight, self.max_weight)


class WalkForwardModel(Strategy):
    """Predicts next-period returns and trades them, retraining as it goes.

    Args:
        min_train: rows required before the first prediction. Three years by
            default -- a model fitted on less has seen too few market conditions
            to have learned anything transferable.
        retrain_every: days between refits. Daily retraining is far more expensive
            and changes almost nothing, since one extra observation barely moves a
            model fitted on thousands.
        lookback: rows of history each fit uses. `None` means expanding (use
            everything). A finite window adapts faster to changing conditions at
            the cost of a smaller sample.
        sizing: how predictions become positions.
    """

    name = "ai_walkforward"

    def __init__(
        self,
        min_train: int = 756,
        retrain_every: int = 63,
        lookback: int | None = None,
        sizing: SizingRule | None = None,
        windows: tuple[int, ...] = DEFAULT_WINDOWS,
        seed: int = 0,
    ) -> None:
        if min_train < 100:
            raise ValueError("min_train below 100 rows cannot fit a useful model")
        if retrain_every < 1:
            raise ValueError("retrain_every must be at least 1")
        self.min_train = int(min_train)
        self.retrain_every = int(retrain_every)
        self.lookback = lookback
        self.sizing = sizing or SizingRule()
        self.windows = windows
        self.seed = int(seed)
        self.last_feature_importance: pd.Series | None = None

    def compute_weights(self, data: MarketData) -> pd.DataFrame:
        ticker = data.tickers[0]
        prices = data.prices

        features = build_features(data, ticker=ticker, windows=self.windows)
        feature_matrix = features.to_numpy(dtype=float)

        # y[i] is the return from i to i+1, first observable on day i+1.
        labels = prices[ticker].pct_change().shift(-1).to_numpy(dtype=float)

        volatility = features[f"volatility_{self.windows[1]}d"].to_numpy(dtype=float)
        volatility = volatility / np.sqrt(252)  # annualised -> per period

        n = len(prices)
        warmup = feature_warmup(self.windows)
        start = max(self.min_train, warmup + 2)

        weights = np.zeros(n)
        model = None

        for t in range(start, n):
            if model is None or (t - start) % self.retrain_every == 0:
                # Everything from `warmup` up to `t - 1` inclusive: the last usable
                # label is y[t-1], the return from t-1 to t, observed on day t.
                train_end = t - 1
                train_start = warmup if self.lookback is None else max(warmup, train_end - self.lookback)

                x_train = feature_matrix[train_start:train_end]
                y_train = labels[train_start:train_end]
                usable = np.isfinite(x_train).all(axis=1) & np.isfinite(y_train)

                if usable.sum() >= self.min_train // 2:
                    model = _make_regressor(self.seed)
                    model.fit(x_train[usable], y_train[usable])
                    self.last_feature_importance = pd.Series(
                        model.feature_importances_, index=features.columns
                    ).sort_values(ascending=False)

            if model is None or not np.isfinite(feature_matrix[t]).all():
                continue

            prediction = float(model.predict(feature_matrix[t : t + 1])[0])
            weights[t] = self.sizing.size(
                np.array([prediction]), np.array([volatility[t]])
            )[0]

        return pd.DataFrame({ticker: weights}, index=prices.index)


class WalkForwardClassifier(WalkForwardModel):
    """Same machinery, but predicts the probability of an up day.

    Included as a comparison because practitioners disagree about which framing
    works better, and the sandbox lets us settle it by measurement rather than
    argument. Position size comes from how far the probability sits from a
    coin flip, so a model with no view holds nothing.
    """

    name = "ai_walkforward_classifier"

    def compute_weights(self, data: MarketData) -> pd.DataFrame:
        import lightgbm as lgb

        ticker = data.tickers[0]
        prices = data.prices

        features = build_features(data, ticker=ticker, windows=self.windows)
        feature_matrix = features.to_numpy(dtype=float)
        forward_returns = prices[ticker].pct_change().shift(-1).to_numpy(dtype=float)
        labels = (forward_returns > 0).astype(float)
        labels[~np.isfinite(forward_returns)] = np.nan

        n = len(prices)
        warmup = feature_warmup(self.windows)
        start = max(self.min_train, warmup + 2)

        weights = np.zeros(n)
        model = None

        for t in range(start, n):
            if model is None or (t - start) % self.retrain_every == 0:
                train_end = t - 1
                train_start = warmup if self.lookback is None else max(warmup, train_end - self.lookback)

                x_train = feature_matrix[train_start:train_end]
                y_train = labels[train_start:train_end]
                usable = np.isfinite(x_train).all(axis=1) & np.isfinite(y_train)

                if usable.sum() >= self.min_train // 2 and len(np.unique(y_train[usable])) > 1:
                    model = lgb.LGBMClassifier(
                        n_estimators=200,
                        learning_rate=0.03,
                        num_leaves=15,
                        min_child_samples=40,
                        subsample=0.8,
                        subsample_freq=1,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        random_state=self.seed,
                        verbose=-1,
                        n_jobs=1,
                    )
                    model.fit(x_train[usable], y_train[usable])

            if model is None or not np.isfinite(feature_matrix[t]).all():
                continue

            probability_up = float(model.predict_proba(feature_matrix[t : t + 1])[0, 1])
            edge = 2.0 * (probability_up - 0.5)
            weights[t] = np.clip(
                edge * self.sizing.aggression, 0.0 if self.sizing.long_only else -1.0, self.sizing.max_weight
            )

        return pd.DataFrame({ticker: weights}, index=prices.index)
