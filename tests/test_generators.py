"""Statistical validation for every generator beyond GBM.

`test_gbm.py` proves the null market is genuinely empty. This file proves the
others contain exactly the structure they advertise -- no more and no less.

That matters as much as the null case. If the AR(1) generator claims an
autocorrelation of 0.05 but actually delivers 0.15, the Recovery Test would report
the AI as far more sensitive than it is, and we would carry that false confidence
into real markets. A measuring instrument that lies optimistically is worse than
none at all.

The pattern throughout is: derive what the parameter implies analytically, then
check the simulation reproduces it.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from sentinel.sandbox.generators import (
    AR1Generator,
    GBMGenerator,
    HestonGenerator,
    JumpDiffusionGenerator,
    OUGenerator,
    RegimeSwitchingGenerator,
)
from sentinel.stats.randomwalk import ljung_box, variance_ratio

TRADING_DAYS = 252

ALL_GENERATORS = [
    GBMGenerator(),
    AR1Generator(phi=0.1),
    OUGenerator(half_life=0.25),
    RegimeSwitchingGenerator(),
    JumpDiffusionGenerator(),
    HestonGenerator(),
]


def annualised_vol(log_returns: np.ndarray) -> float:
    return float(log_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def lag1(x: np.ndarray) -> float:
    return float(np.corrcoef(x[1:], x[:-1])[0, 1])


# --------------------------------------------------------------------------
# The contract every generator must honour
# --------------------------------------------------------------------------

@pytest.mark.parametrize("generator", ALL_GENERATORS, ids=lambda g: g.model_name)
class TestSharedContract:
    """Applied uniformly, so a new generator cannot skip the basics."""

    def test_shape_and_start(self, generator) -> None:
        scenario = generator.generate(n_steps=500, n_assets=3, seed=11)
        assert scenario.data.prices.shape == (500, 3)
        assert (scenario.data.prices.iloc[0] == 100.0).all()
        assert len(scenario.data.log_returns()) == 499

    def test_deterministic_under_seed(self, generator) -> None:
        a = generator.generate(n_steps=300, n_assets=2, seed=7)
        b = generator.generate(n_steps=300, n_assets=2, seed=7)
        assert np.array_equal(a.data.prices.to_numpy(), b.data.prices.to_numpy())

    def test_seeds_differ(self, generator) -> None:
        a = generator.generate(n_steps=300, seed=1)
        b = generator.generate(n_steps=300, seed=2)
        assert not np.allclose(a.data.prices.to_numpy(), b.data.prices.to_numpy())

    def test_prices_stay_positive(self, generator) -> None:
        """Log-space simulation should make this structural, but a jump large
        enough to drive a price to zero would break every downstream log return."""
        scenario = generator.generate(n_steps=3000, n_assets=5, seed=13)
        assert (scenario.data.prices.to_numpy() > 0).all()

    def test_ground_truth_is_declared(self, generator) -> None:
        scenario = generator.generate(n_steps=100, seed=1)
        assert scenario.truth.model == generator.model_name
        assert isinstance(scenario.truth.has_exploitable_signal, bool)
        assert scenario.truth.params["seed"] == 1

    def test_answer_key_is_not_reachable_from_market_data(self, generator) -> None:
        """The AI receives `MarketData`. It must carry no route to the truth."""
        scenario = generator.generate(n_steps=100, seed=1)
        assert set(vars(scenario.data)) == {"prices", "name"}


# --------------------------------------------------------------------------
# AR(1) -- the Recovery Test's dial
# --------------------------------------------------------------------------

class TestAR1:
    @pytest.mark.parametrize("phi", [-0.2, -0.05, 0.02, 0.05, 0.1, 0.3])
    def test_recovers_the_requested_autocorrelation(self, phi: float) -> None:
        """phi must be exactly the lag-1 autocorrelation, including small values.

        The small end is what matters. Real daily equity autocorrelation is around
        0.01-0.05, so if the generator were inaccurate down there the Recovery
        Test would be measuring the generator's error rather than the AI's
        sensitivity.
        """
        scenario = AR1Generator(phi=phi).generate(n_steps=200_000, seed=5)
        observed = lag1(scenario.data.log_returns().to_numpy().ravel())
        standard_error = 1 / np.sqrt(200_000)
        assert abs(observed - phi) < 5 * standard_error

    @pytest.mark.parametrize("phi", [0.0, 0.1, 0.3, 0.5])
    def test_volatility_does_not_change_with_phi(self, phi: float) -> None:
        """The whole point of the sqrt(1 - phi^2) compensation.

        Without it, turning up the signal would also turn up volatility, and any
        change in the AI's results could be blamed on either. The dial has to move
        exactly one thing.
        """
        scenario = AR1Generator(sigma=0.16, phi=phi).generate(n_steps=200_000, seed=6)
        assert annualised_vol(scenario.data.log_returns().to_numpy().ravel()) == pytest.approx(0.16, rel=0.02)

    def test_phi_zero_is_a_random_walk(self, ) -> None:
        """At phi = 0 this generator must be indistinguishable from GBM, and must
        honestly declare that it holds no signal."""
        generator = AR1Generator(phi=0.0)
        assert generator.has_exploitable_signal is False

        scenario = generator.generate(n_steps=5000, seed=8)
        returns = scenario.data.log_returns().to_numpy().ravel()
        assert not ljung_box(returns, lags=10).rejects_random_walk()

    def test_declares_signal_when_phi_is_nonzero(self) -> None:
        assert AR1Generator(phi=0.01).has_exploitable_signal is True

    def test_momentum_is_detectable_at_realistic_length(self) -> None:
        """A strong signal must be visible in ten years of daily data -- otherwise
        the Recovery Test could never find a threshold at all."""
        scenario = AR1Generator(phi=0.15).generate(n_steps=2520, seed=9)
        result = variance_ratio(np.log(scenario.data.prices.to_numpy().ravel()), q=2)
        assert result.rejects_random_walk()
        assert result.statistic > 1  # above 1 means trending

    @pytest.mark.parametrize("phi", [1.0, -1.0, 1.5])
    def test_explosive_phi_rejected(self, phi: float) -> None:
        with pytest.raises(ValueError, match="strictly between"):
            AR1Generator(phi=phi)


# --------------------------------------------------------------------------
# Ornstein-Uhlenbeck -- mean reversion
# --------------------------------------------------------------------------

class TestOU:
    def test_half_life_maps_to_theta(self) -> None:
        generator = OUGenerator(half_life=0.5)
        assert generator.theta == pytest.approx(np.log(2) / 0.5)
        assert generator.half_life_years == pytest.approx(0.5)

    def test_stationary_spread_matches_theory(self) -> None:
        """Log price should settle into a band of width sigma / sqrt(2 * theta).

        The first half-life is discarded as burn-in: the path is anchored at the
        mean level, so it starts with zero spread and has to widen out.
        """
        generator = OUGenerator(theta=4.0, sigma=0.20)
        scenario = generator.generate(n_steps=200_000, seed=12)
        log_prices = np.log(scenario.data.prices.to_numpy().ravel())

        burn_in = int(generator.half_life_years * TRADING_DAYS) * 2
        observed = log_prices[burn_in:].std(ddof=1)
        expected = 0.20 / np.sqrt(2 * 4.0)
        assert observed == pytest.approx(expected, rel=0.15)

    def test_mean_reversion_shows_up_in_the_variance_ratio(self) -> None:
        """VR below 1 is the signature of moves that partly cancel.

        Note this is where the reversion is visible -- *not* in return
        autocorrelation, which stays near zero. The signal lives in the price
        level, not in recent returns, and a model looking only at returns would
        miss it entirely. That is exactly the blind spot this generator exists to
        expose.
        """
        scenario = OUGenerator(half_life=0.15).generate(n_steps=20_000, seed=14)
        result = variance_ratio(np.log(scenario.data.prices.to_numpy().ravel()), q=20)
        assert result.rejects_random_walk()
        assert result.statistic < 1

    def test_rejects_non_reverting_parameters(self) -> None:
        with pytest.raises(ValueError, match="theta must be positive"):
            OUGenerator(theta=0.0)
        with pytest.raises(ValueError, match="half_life must be positive"):
            OUGenerator(half_life=-1.0)


# --------------------------------------------------------------------------
# Regime switching -- the market the AI must learn to read
# --------------------------------------------------------------------------

class TestRegimeSwitching:
    def test_transition_matrix_and_stationary_distribution(self) -> None:
        generator = RegimeSwitchingGenerator(persistence=(0.99, 0.97))
        assert generator.transition_matrix.sum(axis=1) == pytest.approx([1.0, 1.0])
        # Calm runs average 100 days, stressed 33, so calm holds ~75% of the time.
        assert generator.stationary_distribution == pytest.approx([0.75, 0.25], abs=0.01)
        assert generator.expected_durations == pytest.approx((100.0, 33.333), rel=0.01)

    def test_time_spent_in_each_state_matches_theory(self) -> None:
        scenario = RegimeSwitchingGenerator(persistence=(0.99, 0.97)).generate(
            n_steps=200_000, seed=15
        )
        stressed_share = scenario.truth.regimes.mean()
        assert stressed_share == pytest.approx(0.25, abs=0.02)

    def test_run_lengths_match_expected_durations(self) -> None:
        """How long the market stays put is what makes regimes learnable at all."""
        scenario = RegimeSwitchingGenerator(persistence=(0.99, 0.97)).generate(
            n_steps=200_000, seed=16
        )
        states = scenario.truth.regimes
        boundaries = np.flatnonzero(np.diff(states)) + 1
        runs = np.split(states, boundaries)

        calm = [len(r) for r in runs if r[0] == 0][1:-1]  # drop truncated ends
        stressed = [len(r) for r in runs if r[0] == 1][1:-1]
        assert np.mean(calm) == pytest.approx(100.0, rel=0.15)
        assert np.mean(stressed) == pytest.approx(33.3, rel=0.15)

    def test_regimes_align_with_returns(self) -> None:
        """One label per return, not per price. An off-by-one here would hand the
        classifier tomorrow's label for today's data -- lookahead bias smuggled in
        through the answer key itself."""
        scenario = RegimeSwitchingGenerator().generate(n_steps=1000, seed=17)
        assert scenario.truth.regimes.shape == (999,)
        assert len(scenario.data.log_returns()) == 999

    def test_stressed_days_really_are_more_volatile(self) -> None:
        """The labels must correspond to genuinely different behaviour, or the
        classification problem the AI is being set is not solvable in principle."""
        scenario = RegimeSwitchingGenerator(sigma=(0.12, 0.32)).generate(
            n_steps=100_000, seed=18
        )
        returns = scenario.data.log_returns().to_numpy().ravel()
        states = scenario.truth.regimes

        assert annualised_vol(returns[states == 0]) == pytest.approx(0.12, rel=0.05)
        assert annualised_vol(returns[states == 1]) == pytest.approx(0.32, rel=0.05)
        # And the stressed state loses money on average, which is what makes
        # detecting it worth anything.
        assert returns[states == 1].mean() < 0 < returns[states == 0].mean()

    def test_all_assets_share_the_market_regime(self) -> None:
        scenario = RegimeSwitchingGenerator().generate(n_steps=500, n_assets=4, seed=19)
        assert scenario.truth.regimes.ndim == 1

    def test_rejects_impossible_persistence(self) -> None:
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            RegimeSwitchingGenerator(persistence=(1.0, 0.97))


# --------------------------------------------------------------------------
# Jump diffusion -- the fat-tailed null
# --------------------------------------------------------------------------

class TestJumpDiffusion:
    def test_declares_no_exploitable_signal(self) -> None:
        """Poisson arrivals are memoryless, so nothing about the past predicts the
        next jump. This makes it a second null market -- a harder one than GBM,
        because it has the fat tails GBM lacks."""
        assert JumpDiffusionGenerator().has_exploitable_signal is False

    def test_produces_fat_tails(self) -> None:
        """The property that separates this from GBM.

        Excess kurtosis near zero means normal tails; large positive means
        extreme days far more often than a normal distribution allows. Real
        markets look like the latter, and a model validated only on GBM has never
        seen a day that should have been impossible.
        """
        jumpy = JumpDiffusionGenerator().generate(n_steps=100_000, seed=20)
        plain = GBMGenerator().generate(n_steps=100_000, seed=20)

        jumpy_kurtosis = stats.kurtosis(jumpy.data.log_returns().to_numpy().ravel())
        plain_kurtosis = stats.kurtosis(plain.data.log_returns().to_numpy().ravel())

        assert jumpy_kurtosis > 5.0
        assert abs(plain_kurtosis) < 0.5
        assert jumpy_kurtosis > 10 * abs(plain_kurtosis)

    def test_has_no_autocorrelation_despite_the_tails(self) -> None:
        """Fat tails must not come with predictability, or it is not a null market."""
        scenario = JumpDiffusionGenerator().generate(n_steps=50_000, seed=21)
        returns = scenario.data.log_returns().to_numpy().ravel()
        assert abs(lag1(returns)) < 5 / np.sqrt(len(returns))

    def test_jump_count_matches_intensity(self) -> None:
        years = 50_000 / TRADING_DAYS
        scenario = JumpDiffusionGenerator(jump_intensity=3.0).generate(n_steps=50_001, seed=22)
        observed = scenario.truth.params["realised_jump_count"]
        expected = 3.0 * years
        assert observed == pytest.approx(expected, rel=0.15)

    def test_compensator_holds_drift_fixed(self) -> None:
        """Changing the jump settings must not silently change expected return.

        Without the compensator, a market with more crashes would also have lower
        drift, and any comparison against a jump-free market would be confounded
        by two changes at once.
        """
        years = 400_000 / TRADING_DAYS
        calm = JumpDiffusionGenerator(mu=0.08, jump_intensity=0.0).generate(n_steps=400_001, seed=23)
        crashy = JumpDiffusionGenerator(mu=0.08, jump_intensity=6.0).generate(n_steps=400_001, seed=23)

        for scenario in (calm, crashy):
            mean_simple_return = scenario.data.simple_returns().to_numpy().mean() * TRADING_DAYS
            assert mean_simple_return == pytest.approx(0.08, abs=0.04)

    def test_rejects_negative_intensity(self) -> None:
        with pytest.raises(ValueError, match="jump_intensity cannot be negative"):
            JumpDiffusionGenerator(jump_intensity=-1.0)


# --------------------------------------------------------------------------
# Heston -- forecastable risk, unforecastable direction
# --------------------------------------------------------------------------

class TestHeston:
    def test_separates_direction_from_volatility(self) -> None:
        generator = HestonGenerator()
        assert generator.has_exploitable_signal is False
        assert generator.has_predictable_volatility is True

    def test_volatility_clusters(self) -> None:
        """The defining property: calm follows calm, storms follow storms.

        Measured as autocorrelation of *absolute* returns. Raw returns show no
        autocorrelation -- direction is unpredictable -- but their magnitude is
        strongly persistent, which is precisely the split this generator exists to
        make visible.
        """
        heston = HestonGenerator().generate(n_steps=100_000, seed=24)
        plain = GBMGenerator().generate(n_steps=100_000, seed=24)

        heston_returns = heston.data.log_returns().to_numpy().ravel()
        plain_returns = plain.data.log_returns().to_numpy().ravel()

        assert lag1(np.abs(heston_returns)) > 0.10
        assert abs(lag1(np.abs(plain_returns))) < 0.02

        # Direction remains unpredictable, which is the other half of the claim.
        assert abs(lag1(heston_returns)) < 5 / np.sqrt(len(heston_returns))

    def test_long_run_volatility_matches_specification(self) -> None:
        scenario = HestonGenerator(long_run_variance=0.04).generate(n_steps=200_000, seed=25)
        observed = annualised_vol(scenario.data.log_returns().to_numpy().ravel())
        assert observed == pytest.approx(0.20, rel=0.10)

    def test_leverage_effect(self) -> None:
        """Negative rho should make volatility spike as prices fall -- the reason
        crashes are violent and rallies are calm."""
        scenario = HestonGenerator(rho=-0.7).generate(n_steps=100_000, seed=26)
        returns = scenario.data.log_returns().to_numpy().ravel()
        # Realised volatility proxied by absolute return, compared with the return
        # that preceded it.
        assert np.corrcoef(returns[:-1], np.abs(returns[1:]))[0, 1] < -0.02

    def test_feller_condition_is_reported(self) -> None:
        assert HestonGenerator(kappa=3.0, long_run_variance=0.0256, vol_of_vol=0.3).satisfies_feller
        assert not HestonGenerator(kappa=0.5, long_run_variance=0.01, vol_of_vol=0.5).satisfies_feller

    def test_variance_never_goes_negative_in_output(self) -> None:
        """Full truncation permits a negative variance *state* but never a
        negative variance in use. A negative slipping through would produce NaN
        prices, silently, deep in a long simulation."""
        scenario = HestonGenerator(kappa=0.5, long_run_variance=0.01, vol_of_vol=0.9).generate(
            n_steps=50_000, n_assets=3, seed=27
        )
        assert np.isfinite(scenario.data.prices.to_numpy()).all()

    @pytest.mark.parametrize(
        "kwargs,message",
        [
            ({"kappa": 0.0}, "kappa must be positive"),
            ({"long_run_variance": 0.0}, "long_run_variance must be positive"),
            ({"rho": 1.5}, "rho must be between"),
        ],
    )
    def test_invalid_parameters_rejected(self, kwargs: dict, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            HestonGenerator(**kwargs)
