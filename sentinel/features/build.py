"""Feature construction, under one absolute rule.

**Every feature at time t uses only information available at time t.**

That sentence is the whole file. Violating it is called lookahead bias, and it is
the most expensive bug in quantitative finance because it does not announce
itself -- it makes results *better*. A model that can see one day into the future
produces a beautiful equity curve, passes every sanity check a beginner would
think to run, and loses money immediately in live trading.

Two habits enforce the rule here:

1.  Every rolling window is backward-looking and right-aligned. pandas defaults to
    this, but `center=True` would silently break causality, and never appears.
2.  Nothing is standardised using statistics of the whole series. Scaling a
    feature by its full-sample mean and standard deviation leaks the future into
    every row -- one of the most common ways lookahead sneaks into otherwise
    careful code, because it feels like preprocessing rather than modelling.

The rule is not merely documented, it is tested: `tests/test_no_lookahead.py`
recomputes these features on truncated data and asserts nothing earlier changed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.sandbox.market import MarketData

TRADING_DAYS_PER_YEAR = 252

#: Windows in trading days: roughly one week, month, quarter, half-year, year.
DEFAULT_WINDOWS = (5, 21, 63, 126, 252)


def build_features(
    data: MarketData,
    ticker: str | None = None,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Build the feature set for one asset.

    Args:
        data: the market. Only prices are available -- by construction there is no
            route to the ground truth from here.
        ticker: which column to use. Defaults to the first.
        windows: lookback lengths in trading days.

    Returns:
        A frame indexed like `data.prices`, with early rows containing NaN where
        a window has not yet filled. Those NaNs are deliberate and must not be
        back-filled: doing so would import future values into the past.
    """
    if ticker is None:
        ticker = data.tickers[0]

    prices = data.prices[ticker]
    log_prices = np.log(prices)
    returns = log_prices.diff()

    features = pd.DataFrame(index=prices.index)

    for window in windows:
        # Trailing return: momentum over the window, ending today.
        features[f"return_{window}d"] = log_prices.diff(window)

        # Realised volatility, annualised.
        features[f"volatility_{window}d"] = returns.rolling(window).std() * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )

        # Where today's price sits relative to its own trailing average, in units
        # of trailing volatility. Scale-free, so it means the same thing on any
        # asset at any price level.
        moving_average = log_prices.rolling(window).mean()
        rolling_sd = log_prices.rolling(window).std()
        features[f"distance_ma_{window}d"] = (log_prices - moving_average) / rolling_sd.replace(0, np.nan)

    # Drawdown from the running maximum *so far* -- an expanding maximum, which
    # only ever looks backward. A full-sample maximum would be a textbook leak.
    running_peak = prices.cummax()
    features["drawdown"] = prices / running_peak - 1.0

    # Ratio of short to long volatility: is risk rising or falling right now?
    # This is the main input for regime detection, since volatility shifts are
    # what actually distinguish a calm market from a stressed one.
    features["volatility_ratio"] = (
        features[f"volatility_{windows[0]}d"] / features[f"volatility_{windows[-1]}d"]
    )

    # Yesterday's return, the rawest possible momentum signal.
    features["return_1d"] = returns

    # Persistence of magnitude, which is what volatility clustering looks like
    # from inside the data.
    features["abs_return_5d_mean"] = returns.abs().rolling(5).mean()

    return features


def feature_warmup(windows: tuple[int, ...] = DEFAULT_WINDOWS) -> int:
    """Rows at the start that will contain NaN, and must be discarded.

    Callers need this to know where usable data begins. Silently dropping NaN
    rows instead would misalign features against labels, which is another route
    to accidental lookahead.
    """
    return max(windows)
