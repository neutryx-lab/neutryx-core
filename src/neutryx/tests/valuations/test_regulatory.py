"""Tests for regulatory capital calculations (FRTB, SA-CCR, Basel III)."""

import math

import pytest

from neutryx.valuations.regulatory import (
    AssetClass,
    BaselCapitalCalculator,
    BaselCapitalInputs,
    BaselExposure,
    FRTBSensitivity,
    FRTBStandardizedApproach,
    RegulatoryCapitalEngine,
    SACCRCalculator,
    SACCRTrade,
)
from neutryx.valuations.regulatory.frtb import (
    DEFAULT_CROSS_BUCKET_CORRELATION,
    FRTBChargeBreakdown,
)
from neutryx.valuations.regulatory.saccr import DEFAULT_SUPERVISORY_FACTORS
from neutryx.valuations.simm.calculator import SIMMCalculator
from neutryx.valuations.simm.risk_weights import RiskClass, get_risk_weights
from neutryx.valuations.simm.sensitivities import (
    RiskFactorSensitivity,
    RiskFactorType,
    SensitivityType,
)


class TestFRTBStandardizedApproach:
    """Tests for FRTB standardized approach implementation."""

    def test_single_bucket_delta_charge(self):
        """Single sensitivity should produce weighted exposure as capital."""
        calc = FRTBStandardizedApproach()

        sensitivities = [
            FRTBSensitivity(
                risk_factor_type=RiskFactorType.IR,
                sensitivity_type=SensitivityType.DELTA,
                bucket="USD",
                risk_factor="USD-5Y",
                amount=1.0,
                tenor="5Y",
            ),
            FRTBSensitivity(
                risk_factor_type=RiskFactorType.EQUITY,
                sensitivity_type=SensitivityType.DELTA,
                bucket="1",
                risk_factor="AAPL",
                amount=2.0,
            ),
        ]

        result = calc.calculate(sensitivities)

        expected_ir = get_risk_weights(RiskClass.INTEREST_RATE, bucket="USD", tenor="5Y")
        expected_eq = get_risk_weights(RiskClass.EQUITY, bucket="1") * 2.0

        ir_charge: FRTBChargeBreakdown = result.charges_by_risk_class[RiskClass.INTEREST_RATE]
        eq_charge: FRTBChargeBreakdown = result.charges_by_risk_class[RiskClass.EQUITY]

        assert pytest.approx(ir_charge.delta, rel=1e-6) == expected_ir
        assert pytest.approx(eq_charge.delta, rel=1e-6) == expected_eq
        assert result.delta_charge == pytest.approx(expected_ir + expected_eq)
        assert result.total_capital == pytest.approx(expected_ir + expected_eq)

    def test_cross_bucket_correlation_effect(self):
        """Cross-bucket sensitivities should reflect supervisory correlation."""
        calc = FRTBStandardizedApproach()

        usd = FRTBSensitivity(
            risk_factor_type=RiskFactorType.IR,
            sensitivity_type=SensitivityType.DELTA,
            bucket="USD",
            risk_factor="USD-5Y",
            amount=1.0,
            tenor="5Y",
        )
        eur = FRTBSensitivity(
            risk_factor_type=RiskFactorType.IR,
            sensitivity_type=SensitivityType.DELTA,
            bucket="EUR",
            risk_factor="EUR-5Y",
            amount=1.0,
            tenor="5Y",
        )

        result = calc.calculate([usd, eur])

        w_usd = get_risk_weights(RiskClass.INTEREST_RATE, bucket="USD", tenor="5Y")
        w_eur = get_risk_weights(RiskClass.INTEREST_RATE, bucket="EUR", tenor="5Y")
        rho = DEFAULT_CROSS_BUCKET_CORRELATION[RiskClass.INTEREST_RATE]
        expected = math.sqrt(w_usd**2 + w_eur**2 + 2.0 * rho * w_usd * w_eur)

        ir_charge = result.charges_by_risk_class[RiskClass.INTEREST_RATE]
        assert pytest.approx(ir_charge.delta, rel=1e-6) == expected


class TestSACCRCalculator:
    """Tests for SA-CCR exposure calculations."""

    def test_sa_ccr_component_calculation(self):
        """Verify replacement cost, addon, multiplier, and EAD composition."""
        calc = SACCRCalculator(alpha=1.4, multiplier_floor=0.05)
        trades = [
            SACCRTrade(
                asset_class=AssetClass.INTEREST_RATE,
                notional=100.0,
                direction=1,
                supervisory_duration=1.0,
                hedging_set="USD",
            ),
            SACCRTrade(
                asset_class=AssetClass.INTEREST_RATE,
                notional=40.0,
                direction=-1,
                supervisory_duration=1.0,
                hedging_set="USD",
            ),
        ]

        result = calc.calculate(trades, mark_to_market=10.0, collateral=5.0)

        expected_addon = (100.0 - 40.0) * DEFAULT_SUPERVISORY_FACTORS[AssetClass.INTEREST_RATE]

        assert result.replacement_cost == pytest.approx(5.0)
        assert result.addon_by_asset_class[AssetClass.INTEREST_RATE] == pytest.approx(expected_addon)
        assert result.addon == pytest.approx(expected_addon)
        assert result.multiplier == pytest.approx(1.0)
        assert result.potential_future_exposure == pytest.approx(expected_addon)
        assert result.ead == pytest.approx(calc.alpha * (5.0 + expected_addon))

    def test_sa_ccr_capital_requirement_helper(self):
        """Capital requirement helper should reflect EAD * RW * capital ratio."""
        calc = SACCRCalculator(alpha=1.4)
        trades = [
            SACCRTrade(
                asset_class=AssetClass.EQUITY,
                notional=20.0,
                supervisory_duration=1.0,
                hedging_set="DM",
            )
        ]
        result = calc.calculate(trades, mark_to_market=2.0, collateral=0.0)

        charge = result.capital_requirement(risk_weight=0.5, capital_ratio=0.08)
        assert charge == pytest.approx(result.ead * 0.5 * 0.08)


class TestBaselCapitalCalculator:
    """Tests for Basel III capital ratio assessment."""

    def test_capital_ratios_and_requirements(self):
        exposures = [
            BaselExposure(amount=100_000_000.0, risk_weight=0.5),
            BaselExposure(amount=50_000_000.0, risk_weight=1.0),
        ]

        rwa = BaselCapitalCalculator.calculate_rwa(exposures)
        assert rwa == pytest.approx(100_000_000.0)

        capital_inputs = BaselCapitalInputs(
            cet1=6_000_000.0,
            additional_tier1=2_000_000.0,
            tier2=1_000_000.0,
            leverage_exposure=150_000_000.0,
        )

        calculator = BaselCapitalCalculator(capital_conservation_buffer=0.025)
        result = calculator.assess_capital(capital_inputs, rwa)

        assert result.cet1_ratio == pytest.approx(0.06)
        assert result.tier1_ratio == pytest.approx(0.08)
        assert result.total_capital_ratio == pytest.approx(0.09)
        assert result.leverage_ratio == pytest.approx(8_000_000.0 / 150_000_000.0)

        # Buffer-adjusted minimum ratios (4.5% + 2.5% = 7%, etc.)
        assert result.required_cet1 == pytest.approx(7_000_000.0)
        assert result.required_tier1 == pytest.approx(8_500_000.0)
        assert result.required_total_capital == pytest.approx(10_500_000.0)

        assert not result.meets_cet1_requirement
        assert not result.meets_tier1_requirement
        assert not result.meets_total_requirement
        assert result.meets_leverage_requirement


class TestRegulatoryCapitalEngine:
    """Integration tests for the regulatory capital engine."""

    def test_engine_combines_regulatory_measures(self):
        frtb_sensitivities = [
            FRTBSensitivity(
                risk_factor_type=RiskFactorType.IR,
                sensitivity_type=SensitivityType.DELTA,
                bucket="USD",
                risk_factor="USD-5Y",
                amount=1.0,
                tenor="5Y",
            )
        ]

        saccr_trades = [
            SACCRTrade(
                asset_class=AssetClass.INTEREST_RATE,
                notional=50.0,
                supervisory_duration=1.0,
                hedging_set="USD",
            )
        ]

        simm_sensitivities = [
            RiskFactorSensitivity(
                RiskFactorType.IR,
                SensitivityType.DELTA,
                "USD",
                "USD-LIBOR-5Y",
                10000.0,
                "5Y",
            )
        ]

        basel_exposures = [BaselExposure(amount=80_000_000.0, risk_weight=0.5)]
        basel_inputs = BaselCapitalInputs(
            cet1=5_000_000.0,
            additional_tier1=1_500_000.0,
            tier2=1_000_000.0,
            leverage_exposure=120_000_000.0,
        )

        engine = RegulatoryCapitalEngine(simm_calculator=SIMMCalculator())
        summary = engine.run(
            frtb_sensitivities=frtb_sensitivities,
            saccr_trades=saccr_trades,
            saccr_mark_to_market=4.0,
            saccr_collateral=1.0,
            saccr_risk_weight=0.5,
            simm_sensitivities=simm_sensitivities,
            basel_exposures=basel_exposures,
            basel_capital_inputs=basel_inputs,
        )

        standalone_frtb = FRTBStandardizedApproach().calculate(frtb_sensitivities)
        assert summary.frtb is not None
        assert summary.frtb.total_capital == pytest.approx(standalone_frtb.total_capital)

        standalone_saccr = SACCRCalculator().calculate(
            saccr_trades,
            mark_to_market=4.0,
            collateral=1.0,
        )
        assert summary.saccr is not None
        assert summary.saccr.ead == pytest.approx(standalone_saccr.ead)

        expected_saccr_charge = standalone_saccr.capital_requirement(0.5)
        assert summary.saccr_capital_requirement == pytest.approx(expected_saccr_charge)

        standalone_basel = BaselCapitalCalculator().assess_capital(
            basel_inputs,
            BaselCapitalCalculator.calculate_rwa(basel_exposures),
        )
        assert summary.basel is not None
        assert summary.total_capital_requirement == pytest.approx(
            standalone_basel.required_total_capital
        )

        assert summary.simm is not None
        assert summary.initial_margin == pytest.approx(summary.simm.total_im)
