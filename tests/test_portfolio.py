"""Tests for covariance forecasting and the portfolios built on it.

The covariance matrix is the one input this project trusts, so the tests here are
about it staying trustworthy: causal, symmetric, positive semi-definite, and
never emitted from too few observations. The portfolio tests check that each
strategy is actually solving the problem it claims to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.evaluation.causality import check_causality
from sentinel.risk.covariance import (
    MIN_OBSERVATIONS,
    EWMACovariance,
    constant_correlation_target,
    shrink,
)
from sentinel.sandbox.market import MarketData
from sentinel.strategies.portfolio import MinimumVariance, RiskParity


@pytest.fixture
def panel() -> pd.DataFrame:
    """Three assets: two correlated and volatile, one quiet and independent."""
    rng = np.random.default_rng(7)
    n = 900
    a = rng.normal(0.0, 0.020, n)
    b = 0.8 * a + rng.normal(0.0, 0.012, n)
    c = rng.normal(0.0, 0.004, n)
    returns = np.column_stack([a, b, c])
    index = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame(100.0 * np.exp(np.cumsum(returns, axis=0)),
                        index=index, columns=["A", "B", "C"])


class TestShrinkage:
    def test_target_keeps_own_volatilities(self, panel):
        sample = np.cov(np.log(panel).diff().dropna().to_numpy(), rowvar=False)
        target = constant_correlation_target(sample)
        np.testing.assert_allclose(np.diag(target), np.diag(sample), rtol=1e-10)

    def test_target_equalises_correlations(self, panel):
        sample = np.cov(np.log(panel).diff().dropna().to_numpy(), rowvar=False)
        target = constant_correlation_target(sample)
        deviations = np.sqrt(np.diag(target))
        correlation = target / np.outer(deviations, deviations)
        off = correlation[~np.eye(3, dtype=bool)]
        assert np.allclose(off, off[0])

    def test_intensity_endpoints(self, panel):
        sample = np.cov(np.log(panel).diff().dropna().to_numpy(), rowvar=False)
        np.testing.assert_allclose(shrink(sample, 0.0), sample)
        np.testing.assert_allclose(shrink(sample, 1.0), constant_correlation_target(sample))

    def test_intensity_outside_zero_to_one_is_refused(self, panel):
        sample = np.cov(np.log(panel).diff().dropna().to_numpy(), rowvar=False)
        with pytest.raises(ValueError, match="intensity"):
            shrink(sample, 1.5)


class TestEWMACovariance:
    def test_nothing_is_emitted_before_the_minimum(self, panel):
        matrices = EWMACovariance().estimate(panel)
        assert all(m is None for m in matrices[:MIN_OBSERVATIONS])
        assert matrices[-1] is not None

    def test_matrices_are_symmetric_and_psd(self, panel):
        """A minimum-variance optimiser handed a non-PSD matrix answers confidently."""
        for matrix in EWMACovariance().estimate(panel)[-50:]:
            assert np.allclose(matrix, matrix.T)
            assert np.all(np.linalg.eigvalsh(matrix) >= -1e-12)

    def test_it_recovers_the_relative_volatilities(self, panel):
        matrix = EWMACovariance(shrinkage=0.0).estimate(panel)[-1]
        deviations = np.sqrt(np.diag(matrix)) * np.sqrt(252)
        assert deviations[0] > 3 * deviations[2]

    def test_truncating_the_future_does_not_change_the_past(self, panel):
        full = EWMACovariance().estimate(panel)
        for cut in (400, 650, 850):
            short = EWMACovariance().estimate(panel.iloc[:cut])
            assert len(short) == cut
            np.testing.assert_allclose(full[cut - 1], short[-1], rtol=1e-12)

    def test_bad_lambda_is_refused(self):
        with pytest.raises(ValueError, match="lambda_"):
            EWMACovariance(lambda_=1.0)


class TestPortfolios:
    def test_risk_parity_holds_more_of_the_quiet_asset(self, panel):
        weights = RiskParity().compute_weights(MarketData(prices=panel)).iloc[-1]
        assert weights["C"] > weights["A"]

    def test_minimum_variance_concentrates_in_the_quiet_asset(self, panel):
        """With one asset far quieter than the rest, that is the whole answer."""
        weights = MinimumVariance().compute_weights(MarketData(prices=panel)).iloc[-1]
        assert weights["C"] > weights["A"] + weights["B"]

    def test_minimum_variance_forecasts_lower_risk_than_risk_parity(self, panel):
        """Otherwise it is not solving the problem its name claims."""
        matrix = EWMACovariance().estimate(panel)[-1]
        mv = MinimumVariance().allocate(matrix)
        rp = RiskParity().allocate(matrix)
        assert float(mv @ matrix @ mv) <= float(rp @ matrix @ rp) + 1e-12

    @pytest.mark.parametrize("factory", [RiskParity, MinimumVariance])
    def test_never_leveraged_and_never_short(self, panel, factory):
        weights = factory().compute_weights(MarketData(prices=panel))
        assert (weights.to_numpy() >= -1e-12).all()
        assert weights.abs().sum(axis=1).max() <= 1.0 + 1e-9

    @pytest.mark.parametrize("factory", [RiskParity, MinimumVariance])
    def test_weights_are_causal(self, panel, factory):
        assert check_causality(factory(), MarketData(prices=panel)).is_causal

    @pytest.mark.parametrize("factory", [RiskParity, MinimumVariance])
    def test_rebalancing_holds_weights_between_dates(self, panel, factory):
        """Turnover is a cost; the schedule must actually be respected."""
        weights = factory(rebalance_days=21).compute_weights(MarketData(prices=panel))
        live = weights.loc[(weights.abs().sum(axis=1) > 0)]
        changed = (live.diff().abs().sum(axis=1) > 1e-12).sum()
        assert changed <= len(live) / 15

    def test_leverage_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="leverage"):
            RiskParity(max_gross=1.5)
