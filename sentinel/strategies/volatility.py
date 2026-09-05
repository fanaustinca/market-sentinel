"""Volatility targeting: hold a constant amount of *risk*, not of capital.

This strategy exists because of a measurement, recorded in
`experiments/simulator_gap.py`, that contradicted the assumption every earlier
regime strategy was built on.

The assumption was that a high-volatility market is one to sit out. Inside the
sandbox that was true by construction -- `RegimeSwitchingGenerator` welds high
volatility to negative drift, so fleeing volatility meant fleeing losses, and
every rung-1 experiment rewarded it.

On real SPY from 1993 it is false, and not marginally so. The state the
classifier calls stressed has an annualised forward return of **+17%** against
the calm state's +9%. Sorting by trailing realised volatility instead of by the
model gives the same answer -- the highest-volatility quintile has the highest
next-day return -- so it is a property of the market rather than an artefact of
the classifier. Equity volatility is compensated. Selling into it means selling
precisely the periods you are being paid to hold.

But the two states are *not* equally attractive per unit of risk taken. On real
SPY the calm state runs at roughly 13% volatility for 9% return, and the stressed
state at 28% for 17%. The Sharpe ratios are 0.67 and 0.62 -- close enough to call
equal. So the market is not offering a better deal in either state; it is offering
the same deal at two different sizes.

Which points at the correct response. Do not choose *whether* to be exposed.
Choose *how much*, so the risk taken is the same in both states:

    weight = target_volatility / forecast_volatility

Exposure falls when volatility rises, and never reaches zero, so the compensated
periods are still held -- just smaller. This is `plan.md` section 4's "uncertainty
shrinks positions" applied to the quantity that actually varies.

What it gives up
----------------
It is not a crash defence. Volatility targeting reduces exposure *after*
volatility rises, which means it is in the market for the first leg down every
time. It cuts the depth and length of a drawdown; it does not avoid one. A
strategy that claimed to would be claiming to predict the crash, which is the
thing nobody can do and the thing this project has already measured itself unable
to do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.features.build import TRADING_DAYS_PER_YEAR
from sentinel.sandbox.market import MarketData
from sentinel.strategies.base import Strategy


class VolatilityTarget(Strategy):
    """Size positions so forecast risk is constant, capped at full investment.

    Args:
        target_volatility: annualised portfolio volatility to aim for. 12% is a
            deliberately modest budget -- below a broad equity index's long-run
            16-20%, so the strategy is usually holding less than everything and
            has room to size *up* in calm periods without ever borrowing.
        max_weight: hard cap. 1.0 means no leverage, ever -- a permanent policy
            in this project, not a parameter to revisit. Without it, volatility
            targeting quietly becomes a leveraged strategy in calm markets, which
            is how the approach usually blows up.
        band: no-trade band, to keep turnover and therefore cost down.
        floor_volatility: the forecast is floored here before dividing, so an
            unusually quiet stretch cannot produce an enormous position. Dividing
            by a small estimated number is the standard way this family of
            strategies fails.
        forecaster: how volatility is predicted. Defaults to `EWMAVolatility()`
            -- see the class docstring for why that replaced a 21-day rolling
            standard deviation. Pass any other `VolatilityForecaster` to compare;
            everything else about the strategy is held identical, so a difference
            in result is attributable to the forecast and nothing else.
    """

    name = "volatility_target"

    def __init__(
        self,
        target_volatility: float = 0.12,
        max_weight: float = 1.0,
        band: float = 0.10,
        floor_volatility: float = 0.04,
        forecaster=None,
    ) -> None:
        if target_volatility <= 0:
            raise ValueError("target_volatility must be positive")
        if not 0.0 < max_weight <= 1.0:
            raise ValueError("max_weight must be in (0, 1]; this project does not use leverage")
        if not 0.0 <= band < 1.0:
            raise ValueError("band must be in [0, 1)")
        if floor_volatility <= 0:
            raise ValueError("floor_volatility must be positive")

        self.target_volatility = float(target_volatility)
        self.max_weight = float(max_weight)
        self.band = float(band)
        self.floor_volatility = float(floor_volatility)

        if forecaster is None:
            from sentinel.ai.volatility import EWMAVolatility

            forecaster = EWMAVolatility()
        else:
            self.name = f"voltarget_{forecaster.name}"
        self.forecaster = forecaster

    def forecast_volatility(self, data: MarketData, ticker: str) -> np.ndarray:
        """Annualised volatility for the next period, from the forecaster.

        Causality is the forecaster's responsibility and every one of them is
        checked for it; `check_causality` on the strategy verifies the whole
        chain regardless, which is what caught the lookahead in the regime-based
        variant.
        """
        return self.forecaster.forecast(data.prices[ticker]).to_numpy(dtype=float)

    def compute_weights(self, data: MarketData) -> pd.DataFrame:
        ticker = data.tickers[0]
        volatility = self.forecast_volatility(data, ticker)

        safe = np.maximum(volatility, self.floor_volatility)
        desired = np.clip(self.target_volatility / safe, 0.0, self.max_weight)
        # Warmup rows have no forecast. Cash, because nothing is known yet.
        desired = np.where(np.isfinite(volatility), desired, 0.0)

        weights = np.empty(len(desired))
        held = 0.0
        for t, target in enumerate(desired):
            if abs(target - held) > self.band:
                held = float(target)
            weights[t] = held

        return pd.DataFrame({ticker: weights}, index=data.prices.index)


class RegimeVolatilityTarget(VolatilityTarget):
    """Volatility targeting using the regime model's forecast instead of realised vol.

    The classifier estimates a volatility for each state and a probability of
    being in each, so the expected variance one step ahead is available in closed
    form:

        E[variance] = P(calm)·variance_calm + P(stressed)·variance_stressed

    That is a genuine *forecast* rather than an extrapolation of the recent past,
    and it should react faster at a regime boundary -- the model can shift its
    probability mass in a day, while a 21-day realised window needs weeks to
    reflect a change.

    Whether the extra machinery earns its place is a question for measurement,
    not argument, and the plain `VolatilityTarget` is the control that decides it.
    On the project's standing principle, if the two match, the simpler one wins.
    """

    name = "regime_volatility_target"

    def __init__(self, classifier=None, **kwargs) -> None:
        super().__init__(**kwargs)
        if classifier is None:
            from sentinel.ai.regime.classifier import WalkForwardRegimeClassifier

            classifier = WalkForwardRegimeClassifier()
        self.classifier = classifier

    def forecast_volatility(self, data: MarketData, ticker: str) -> np.ndarray:
        # `forecast_variance` uses the model parameters in force on each row.
        # The first version of this method read `classifier.last_parameters` --
        # the final refit, applied to every row -- and `check_causality` reported
        # LOOKAHEAD DETECTED at row 756 within seconds of it being written. Rows
        # before the classifier has fitted anything come back NaN, which the
        # caller turns into cash: the right answer for "nothing is known yet".
        return np.sqrt(self.classifier.forecast_variance(data, ticker=ticker).to_numpy())
