"""Proof that the GBM generator produces genuine random walks.

Everything in this project is built on trusting this generator. If the "empty"
market is not actually empty, every Null Test result downstream is void -- so the
burden of proof here is high, and these tests carry it.

Two ideas run through the file.

**Test many paths, not one.** Any single random path will contain accidental
patterns; that is what randomness looks like up close. Judging the generator by
one path would be the same error the whole project is designed to avoid, so these
tests either pool hundreds of thousands of observations or measure behaviour
across a thousand independent paths.

**Fix the seed.** A statistical test that draws fresh data on every run will fail
roughly one time in twenty for no reason, and a test suite that cries wolf gets
ignored. Every test here is deterministic.

Independent paths are produced as columns of one multi-asset call with the default
identity correlation, which is both faster and exactly equivalent to generating
them one at a time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.sandbox.generators.gbm import GBMGenerator
from sentinel.sandbox.market import MarketData
from sentinel.stats.randomwalk import ljung_box, normality, variance_ratio

MU = 0.08
SIGMA = 0.16
TRADING_DAYS = 252


@pytest.fixture(scope="module")
def wide_paths() -> np.ndarray:
    """500 independent 10-year paths, as log prices. Shape (2520, 500)."""
    scenario = GBMGenerator(mu=MU, sigma=SIGMA).generate(n_steps=2520, n_assets=500, seed=0)
    return np.log(scenario.data.prices.to_numpy())


# --------------------------------------------------------------------------
# Does it produce the parameters we asked for?
# --------------------------------------------------------------------------

def test_volatility_matches_specification(wide_paths: np.ndarray) -> None:
    """Realised volatility should equal the sigma we requested."""
    daily_returns = np.diff(wide_paths, axis=0)
    realised = daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    assert realised == pytest.approx(SIGMA, rel=0.01)


def test_drift_includes_volatility_drag(wide_paths: np.ndarray) -> None:
    """The -sigma^2/2 correction must be present.

    This is the test that catches the single easiest mistake in the generator.
    `mu` is the drift of the price, but we simulate log prices, and volatility
    drags on compounded growth -- so the log-space drift is lower by exactly
    sigma^2/2. Drop that term and the generator hands out roughly 1.3% a year of
    free return that was never requested, which would flow silently into every
    result built on top of it.

    With 1.26 million observations the standard error is small enough to tell the
    two apart decisively, so this test would fail loudly if the term went missing.
    """
    daily_returns = np.diff(wide_paths, axis=0)
    observed_daily_drift = daily_returns.mean()

    correct = (MU - 0.5 * SIGMA**2) / TRADING_DAYS
    uncorrected = MU / TRADING_DAYS

    standard_error = daily_returns.std(ddof=1) / np.sqrt(daily_returns.size)

    assert abs(observed_daily_drift - correct) < 4 * standard_error
    # And it is nowhere near the value we would see if the drag were missing.
    assert abs(observed_daily_drift - uncorrected) > 4 * standard_error


def test_correlation_is_reproduced() -> None:
    """Assets should come out with the correlation structure we specified."""
    target = np.array([[1.0, 0.6, 0.2], [0.6, 1.0, 0.4], [0.2, 0.4, 1.0]])
    scenario = GBMGenerator(correlation=target).generate(n_steps=25_000, n_assets=3, seed=7)
    observed = np.corrcoef(scenario.data.log_returns().to_numpy(), rowvar=False)
    assert observed == pytest.approx(target, abs=0.02)


def test_terminal_distribution_is_lognormal(wide_paths: np.ndarray) -> None:
    """Total log return over the full path should be normal with variance sigma^2 * T."""
    total_log_return = wide_paths[-1] - wide_paths[0]
    years = (wide_paths.shape[0] - 1) / TRADING_DAYS

    expected_mean = (MU - 0.5 * SIGMA**2) * years
    expected_sd = SIGMA * np.sqrt(years)

    assert total_log_return.mean() == pytest.approx(
        expected_mean, abs=4 * expected_sd / np.sqrt(len(total_log_return))
    )
    assert total_log_return.std(ddof=1) == pytest.approx(expected_sd, rel=0.10)


# --------------------------------------------------------------------------
# Is it actually a random walk?
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "test_name,run_test",
    [
        ("ljung_box", lambda lp: ljung_box(np.diff(lp), lags=10)),
        ("variance_ratio_q2", lambda lp: variance_ratio(lp, q=2)),
        ("variance_ratio_q5", lambda lp: variance_ratio(lp, q=5)),
        ("jarque_bera", lambda lp: normality(np.diff(lp))),
    ],
)
def test_rejection_rate_matches_alpha(test_name: str, run_test) -> None:
    """Across many null markets, each test should reject exactly 5% of the time.

    This is the right way to validate the generator, and the reasoning is worth
    following closely.

    Running one path and asking "did the test say random walk?" is meaningless: at
    a 5% significance level, one path in twenty gets flagged no matter how perfect
    the generator is. Seed 42 does exactly this -- it produces a market our tests
    call "trending", purely by chance.

    So we invert the question. Instead of demanding no rejections, we demand that
    rejections arrive at precisely the rate the mathematics predicts. Too many and
    the generator is producing structure it should not. Too *few* and something is
    equally wrong -- the paths would be suspiciously well behaved, which random
    data never is.

    The same logic is what makes the Null Test work in Phase 1. We will not ask
    "did the AI lose money on noise?"; we will ask "did the AI's results fall
    within the distribution that pure chance produces?"
    """
    alpha = 0.05
    n_paths = 1000

    scenario = GBMGenerator(mu=MU, sigma=SIGMA).generate(n_steps=1260, n_assets=n_paths, seed=1234)
    log_prices = np.log(scenario.data.prices.to_numpy())

    rejections = sum(run_test(log_prices[:, i]).p_value < alpha for i in range(n_paths))
    rate = rejections / n_paths

    # Rejections are binomial, so three standard errors is a ~99.7% band.
    standard_error = np.sqrt(alpha * (1 - alpha) / n_paths)
    assert abs(rate - alpha) < 3 * standard_error, (
        f"{test_name} rejected {rate:.2%} of null markets, expected {alpha:.0%} "
        f"+/- {3 * standard_error:.2%}"
    )


def test_autocorrelation_is_negligible(wide_paths: np.ndarray) -> None:
    """Pooled across all paths, return autocorrelation should be indistinguishable
    from zero at every short lag. Any reliable nonzero value here would be a
    tradeable edge -- in a market built to contain none."""
    daily_returns = np.diff(wide_paths, axis=0)
    n = daily_returns.shape[0]
    # Standard error of a sample autocorrelation is roughly 1/sqrt(n) per path;
    # averaging over the paths shrinks it by another sqrt(n_paths).
    tolerance = 4 / np.sqrt(n * daily_returns.shape[1])

    for lag in range(1, 6):
        per_path = [
            np.corrcoef(daily_returns[lag:, i], daily_returns[:-lag, i])[0, 1]
            for i in range(daily_returns.shape[1])
        ]
        assert abs(np.mean(per_path)) < tolerance, f"lag {lag} shows autocorrelation"


# --------------------------------------------------------------------------
# Mechanical correctness
# --------------------------------------------------------------------------

def test_same_seed_gives_same_market() -> None:
    a = GBMGenerator().generate(n_steps=500, n_assets=3, seed=99)
    b = GBMGenerator().generate(n_steps=500, n_assets=3, seed=99)
    pd.testing.assert_frame_equal(a.data.prices, b.data.prices)


def test_different_seeds_give_different_markets() -> None:
    a = GBMGenerator().generate(n_steps=500, seed=1)
    b = GBMGenerator().generate(n_steps=500, seed=2)
    assert not np.allclose(a.data.prices.to_numpy(), b.data.prices.to_numpy())


def test_path_starts_at_initial_price() -> None:
    """Guards an off-by-one that would shift every series by one day.

    A misalignment like that is not a cosmetic problem: it is precisely how
    lookahead bias gets in, because a series shifted relative to its dates makes
    tomorrow's information appear to be available today.
    """
    scenario = GBMGenerator(initial_price=100.0).generate(n_steps=100, n_assets=2, seed=3)
    assert (scenario.data.prices.iloc[0] == 100.0).all()


def test_shape_and_index() -> None:
    scenario = GBMGenerator().generate(n_steps=250, n_assets=4, seed=5)
    prices = scenario.data.prices
    assert prices.shape == (250, 4)
    assert list(prices.columns) == ["SYN0", "SYN1", "SYN2", "SYN3"]
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert prices.index.is_monotonic_increasing
    assert len(scenario.data.log_returns()) == 249


def test_zero_volatility_is_deterministic() -> None:
    """With sigma=0 the price compounds smoothly at mu. A useful edge case, and
    a direct check that drift is wired to the right parameter."""
    scenario = GBMGenerator(mu=0.10, sigma=0.0).generate(n_steps=253, seed=1)
    prices = scenario.data.prices.to_numpy().ravel()
    assert prices[-1] / prices[0] == pytest.approx(np.exp(0.10), rel=1e-9)


def test_custom_tickers() -> None:
    scenario = GBMGenerator().generate(n_steps=50, n_assets=2, seed=1, tickers=["SPY", "AGG"])
    assert scenario.data.tickers == ["SPY", "AGG"]


# --------------------------------------------------------------------------
# The answer key
# --------------------------------------------------------------------------

def test_ground_truth_declares_no_signal() -> None:
    """The field the entire Null Test asserts against."""
    scenario = GBMGenerator().generate(n_steps=100, seed=1)
    assert scenario.truth.model == "gbm"
    assert scenario.truth.has_exploitable_signal is False
    assert scenario.truth.regimes is None


def test_ground_truth_records_parameters() -> None:
    scenario = GBMGenerator(mu=0.05, sigma=0.20).generate(n_steps=100, seed=17)
    assert scenario.truth.params["mu"] == [0.05]
    assert scenario.truth.params["sigma"] == [0.20]
    assert scenario.truth.params["seed"] == 17


def test_market_data_carries_no_hint_of_its_origin() -> None:
    """The AI receives `MarketData` and must not be able to tell where it came from.

    `Scenario` holds the answer key alongside the prices, but the two are separate
    objects, so passing prices to a model cannot accidentally pass the answer too.
    """
    scenario = GBMGenerator().generate(n_steps=100, seed=1)
    assert not hasattr(scenario.data, "truth")
    assert set(vars(scenario.data)) == {"prices", "name"}


# --------------------------------------------------------------------------
# Bad input should fail loudly and early
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"n_steps": 1}, "at least 2 steps"),
        ({"n_assets": 0}, "at least one asset"),
        ({"n_steps": 50, "n_assets": 2, "tickers": ["ONLY_ONE"]}, "tickers"),
    ],
)
def test_invalid_arguments_rejected(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GBMGenerator().generate(seed=1, **kwargs)


def test_negative_volatility_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        GBMGenerator(sigma=-0.1).generate(n_steps=50, seed=1)


def test_impossible_correlation_rejected() -> None:
    """A correlation matrix typed in by hand is very easily impossible.

    Here A and B are strongly positive, B and C strongly positive, but A and C
    strongly negative -- which no set of real assets can satisfy. Catching this
    with an explanation beats an opaque failure inside numpy's Cholesky routine.
    """
    impossible = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    with pytest.raises(ValueError, match="not positive definite"):
        GBMGenerator(correlation=impossible).generate(n_steps=50, n_assets=3, seed=1)


def test_asymmetric_correlation_rejected() -> None:
    bad = np.array([[1.0, 0.5], [0.2, 1.0]])
    with pytest.raises(ValueError, match="symmetric"):
        GBMGenerator(correlation=bad).generate(n_steps=50, n_assets=2, seed=1)


def test_wrong_size_correlation_rejected() -> None:
    with pytest.raises(ValueError, match="must be 3x3"):
        GBMGenerator(correlation=np.eye(2)).generate(n_steps=50, n_assets=3, seed=1)


# --------------------------------------------------------------------------
# The MarketData contract
# --------------------------------------------------------------------------

def test_market_data_requires_dates() -> None:
    with pytest.raises(TypeError, match="indexed by dates"):
        MarketData(prices=pd.DataFrame({"A": [1.0, 2.0]}))


def test_market_data_rejects_unsorted_dates() -> None:
    index = pd.to_datetime(["2020-01-02", "2020-01-01"])
    with pytest.raises(ValueError, match="sorted"):
        MarketData(prices=pd.DataFrame({"A": [1.0, 2.0]}, index=index))


def test_market_data_rejects_non_positive_prices() -> None:
    index = pd.to_datetime(["2020-01-01", "2020-01-02"])
    with pytest.raises(ValueError, match="positive"):
        MarketData(prices=pd.DataFrame({"A": [1.0, 0.0]}, index=index))
