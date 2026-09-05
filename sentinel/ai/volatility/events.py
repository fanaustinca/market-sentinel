"""A volatility forecast that knows an earnings release just landed.

Everything else in this project that tried to predict *direction* from events
failed, and the oracle experiment says why: by the time a filing is public and
tradable, the repricing is over. But the same experiment measured something that
did not fail. Days on which an earnings release becomes actionable move about
2.8 times as much as an ordinary day for the same stock, and that is a statement
about magnitude, which needs no view on which way.

The timing here is unusually clean, and it is worth being precise about why,
because "uses news" normally means "probably peeks".

Roughly nine in ten earnings releases in this universe are filed at or after
16:00 New York time, once the session has closed. So at the close of day *t* the
filing is already public and already timestamped, and the question being asked --
how much will this stock move between *t* and *t+1*? -- is about a session that
has not started. Nothing is being anticipated. The forecaster is told a fact that
was on the SEC's public wire before it spoke, and asked about a day that had not
happened. That is the ordinary causal setting, not a favourable one.

The multiplier is estimated walk-forward from earnings that had already occurred,
never from the sample being scored, so a stock whose reactions grew larger over
time cannot lend its later behaviour to its earlier forecasts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.ai.volatility.forecasters import VolatilityForecaster

#: Minimum past earnings reactions required before the multiplier is trusted.
#: Below this the estimate is noise, and the forecaster falls back to 1.0 -- that
#: is, to exactly the base model, which is the right default for "I don't know".
MIN_HISTORY = 8

#: Median of |z| for a standard normal. A correctly-calibrated forecast produces
#: |return| / forecast with this median, so it is the divisor that converts a
#: median of such ratios into a multiplier on standard deviation.
MEDIAN_ABS_NORMAL = 0.6744897501960817

#: Cap on the learned multiplier. A single 30% reaction in a thin sample can push
#: an unclamped estimate to absurd values, and a forecast four times too high is
#: penalised by QLIKE almost as heavily as one four times too low.
MAX_MULTIPLIER = 4.0


class EarningsAwareVolatility(VolatilityForecaster):
    """Any base forecaster, scaled up on days a release is already public.

    Args:
        base: the forecaster to correct. Its output is used unchanged on all
            ordinary days, so this can only help or hurt on the ~1% of days that
            carry an event -- which is a deliberately narrow claim.
        events: 0/1 flags from `features.events.announcement_flag`, marking rows
            whose forecast window contains a release. Either a `Series` for a
            single asset, or a `DataFrame` whose columns are tickers -- the
            multi-asset form is selected by the name of the price series it is
            asked about, which is how `VolatilityTarget` passes one asset at a
            time. A ticker absent from the frame gets the base forecast unchanged
            rather than an error, since an index fund has no earnings date and
            that is a legitimate answer rather than a missing one.
    """

    def __init__(self, base: VolatilityForecaster, events: pd.Series | pd.DataFrame) -> None:
        self.base = base
        self.events = events
        self.name = f"{base.name}+earnings"

    def _flags_for(self, prices: pd.Series) -> pd.Series:
        if isinstance(self.events, pd.DataFrame):
            ticker = prices.name
            if ticker not in self.events.columns:
                return pd.Series(0.0, index=prices.index)
            return self.events[ticker]
        return self.events

    def forecast(self, prices: pd.Series) -> pd.Series:
        base = self.base.forecast(prices)
        flags = self._flags_for(prices).reindex(prices.index).fillna(0.0).to_numpy() > 0

        returns = np.log(prices).diff()
        # Row t forecasts the return from t to t+1, so the outcome that tests row
        # t is returns[t+1]. Shifting here keeps the multiplier's training target
        # on the same footing as the score the forecaster will be graded by.
        realised = returns.shift(-1).to_numpy(dtype=float)
        predicted = base.to_numpy(dtype=float) / np.sqrt(252.0)

        # Ratio of what actually happened to what the base model expected, on
        # event days only. A value of 2 means the base model was half as large as
        # it should have been.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.abs(realised) / predicted
        usable = flags & np.isfinite(ratio) & (predicted > 0)

        multiplier = np.ones(len(prices))
        seen: list[float] = []
        for t in range(len(prices)):
            if flags[t] and len(seen) >= MIN_HISTORY:
                # Median, not mean: reaction ratios are right-skewed and a single
                # earnings shock would otherwise dominate the estimate for years.
                multiplier[t] = min(float(np.median(seen)), MAX_MULTIPLIER)
            # Append only after using, so day t's own outcome never informs day t.
            if usable[t]:
                seen.append(float(ratio[t]))

        # The ratio is built from a single |return|, which is a very noisy view of
        # a standard deviation, and the *median* of |z| for a correctly-calibrated
        # normal forecast is 0.6745 rather than 1. Dividing by that constant is
        # what turns a median of ratios back into a volatility multiplier. Using
        # the mean correction (0.798) here instead -- which is the constant that
        # belongs with a mean estimator -- biases every multiplier down by 18%.
        multiplier = np.where(multiplier != 1.0, multiplier / MEDIAN_ABS_NORMAL, multiplier)

        out = pd.Series(base.to_numpy() * multiplier, index=prices.index)
        out.name = self.name
        return out
