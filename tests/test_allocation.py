"""Tests for multi-asset rotation.

The strategy's whole premise -- hold something defensive instead of cash when
equities are stressed -- is also its main risk, because "defensive" is a claim
about the future dressed as a fact about the past. Treasuries rose through 2000,
2008 and 2020 and fell alongside equities through 2022.

These tests cannot check whether the premise holds; nothing can, in advance. What
they check is that the mechanism does what it says, that it stays causal, and that
the sleeve is a parameter the caller chose rather than something the code decided
on the strength of history.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.engine.backtest import UNLIMITED, run_backtest
from sentinel.evaluation.causality import check_causality
from sentinel.sandbox.generators.gbm import GBMGenerator
from sentinel.sandbox.generators.regime import RegimeSwitchingGenerator
from sentinel.strategies.allocation import RegimeRotation


@pytest.fixture(scope="module")
def panel():
    """A three-asset market. The regime is shared, as it is in real markets."""
    scenario = RegimeSwitchingGenerator().generate(n_steps=2000, n_assets=3, seed=21)
    return scenario.data


class TestMechanics:
    def test_is_causal(self, panel) -> None:
        strategy = RegimeRotation(risk_assets=["SYN0"], defensive_assets=["SYN1"])
        assert check_causality(strategy, panel).is_causal, "rotation must not see the future"

    def test_momentum_selection_is_causal(self, panel) -> None:
        strategy = RegimeRotation(
            risk_assets=["SYN0", "SYN1"], defensive_assets=["SYN2"], select_by_momentum=True
        )
        report = check_causality(strategy, panel)
        assert report.is_causal, str(report)

    def test_holds_cash_through_the_warmup(self, panel) -> None:
        weights = RegimeRotation(risk_assets=["SYN0"], defensive_assets=["SYN1"]).compute_weights(
            panel
        )
        assert (weights.iloc[:756] == 0.0).all().all()

    def test_blends_rather_than_switching(self, panel) -> None:
        """At 70% calm the portfolio is 70% risk, not all-in on one sleeve.

        This is where "uncertainty shrinks positions" comes from: an unsure model
        lands near half and half, which is the allocation that regrets least
        whichever state turns out to hold.
        """
        weights = RegimeRotation(
            risk_assets=["SYN0"], defensive_assets=["SYN1"], band=0.0
        ).compute_weights(panel)
        after_warmup = weights.iloc[800:]
        mixed = ((after_warmup["SYN0"] > 0.05) & (after_warmup["SYN1"] > 0.05)).sum()
        assert mixed > 100, "a pure switch would never hold both sleeves at once"

    def test_weights_never_exceed_full_investment(self, panel) -> None:
        """The blend is a partition of one, so leverage cannot appear by accident."""
        weights = RegimeRotation(
            risk_assets=["SYN0", "SYN1"], defensive_assets=["SYN2"]
        ).compute_weights(panel)
        assert weights.to_numpy().sum(axis=1).max() <= 1.0 + 1e-9
        assert (weights.to_numpy() >= -1e-12).all()

    def test_an_empty_defensive_sleeve_means_cash(self, panel) -> None:
        """The control the rotation must be compared against.

        Without it, "rotation beats buy-and-hold" cannot be separated into "the
        timing worked" and "the defensive asset happened to go up".
        """
        weights = RegimeRotation(risk_assets=["SYN0"], defensive_assets=[]).compute_weights(panel)
        assert (weights["SYN1"] == 0.0).all()
        assert (weights["SYN2"] == 0.0).all()
        assert weights["SYN0"].iloc[800:].max() > 0.0

    def test_the_no_trade_band_cuts_turnover(self, panel) -> None:
        wide = run_backtest(
            panel, RegimeRotation(risk_assets=["SYN0"], defensive_assets=["SYN1"], band=0.30),
            limits=UNLIMITED,
        )
        tight = run_backtest(
            panel, RegimeRotation(risk_assets=["SYN0"], defensive_assets=["SYN1"], band=0.0),
            limits=UNLIMITED,
        )
        assert wide.annual_turnover < tight.annual_turnover

    def test_momentum_selection_picks_the_leader(self) -> None:
        """Among risk assets, hold the one that has actually been going up."""
        scenario = GBMGenerator(mu=np.array([0.20, -0.10]), sigma=0.14).generate(
            n_steps=1600, n_assets=2, seed=5
        )
        weights = RegimeRotation(
            risk_assets=["SYN0", "SYN1"], select_by_momentum=True, lookback=126, band=0.0
        ).compute_weights(scenario.data)
        held = weights.iloc[900:]
        assert held["SYN0"].mean() > held["SYN1"].mean()


class TestGuards:
    def test_rejects_tickers_the_market_does_not_have(self, panel) -> None:
        with pytest.raises(ValueError, match=r"\['NOPE'\]"):
            RegimeRotation(risk_assets=["SYN0"], defensive_assets=["NOPE"]).compute_weights(panel)

    def test_rejects_a_regime_ticker_outside_the_market(self, panel) -> None:
        strategy = RegimeRotation(risk_assets=["SYN0"], regime_ticker="MISSING")
        with pytest.raises(ValueError, match="regime ticker MISSING"):
            strategy.compute_weights(panel)

    def test_needs_at_least_one_risk_asset(self) -> None:
        with pytest.raises(ValueError, match="at least one risk asset"):
            RegimeRotation(risk_assets=[])

    def test_rejects_an_impossible_band(self) -> None:
        with pytest.raises(ValueError, match="band must be"):
            RegimeRotation(risk_assets=["SYN0"], band=1.0)

    def test_reads_the_regime_from_the_named_asset_only(self, panel) -> None:
        """Averaging the state across the panel would blur the equity signal with
        the behaviour of the very assets the strategy rotates into."""
        from_first = RegimeRotation(
            risk_assets=["SYN0"], defensive_assets=["SYN1"], regime_ticker="SYN0", band=0.0
        ).compute_weights(panel)
        from_third = RegimeRotation(
            risk_assets=["SYN0"], defensive_assets=["SYN1"], regime_ticker="SYN2", band=0.0
        ).compute_weights(panel)
        assert not np.allclose(from_first.to_numpy(), from_third.to_numpy())


def test_each_configuration_gets_its_own_name() -> None:
    """Reports key results by strategy name, so a shared name loses rows.

    Comparing rotations that differ only in their defensive sleeve is the whole
    point of the ladder -- into cash, into treasuries, into treasuries and gold.
    With one class-level name they collapse to a single entry and the last one
    quietly wins. Nothing crashes; the table just reports one strategy three
    times under three sets of numbers.
    """
    into_cash = RegimeRotation(risk_assets=["SPY"], defensive_assets=[])
    into_bonds = RegimeRotation(risk_assets=["SPY"], defensive_assets=["IEF"])
    with_momentum = RegimeRotation(
        risk_assets=["SPY", "IWM"], defensive_assets=["IEF"], select_by_momentum=True
    )

    names = {into_cash.name, into_bonds.name, with_momentum.name}
    assert len(names) == 3, names
    assert into_cash.name == "rotate_SPY_to_cash"
    assert into_bonds.name == "rotate_SPY_to_IEF"
    assert with_momentum.name.endswith("_mom")
