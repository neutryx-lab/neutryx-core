"""Tests for FRTB Default Risk Charge (DRC) and Residual Risk Add-On (RRAO).

Tests cover:
- DRC calculation for non-securitized exposures
- DRC calculation for securitized products
- Issuer-level aggregation and correlation
- Sector-based correlation effects
- RRAO calculation for exotic underlyings
- RRAO calculation for exotic payoffs
- Complexity and liquidity multipliers
- Hedge effectiveness recognition
"""
import pytest
import jax.numpy as jnp

from neutryx.valuations.regulatory.frtb_drc import (
    CreditRating,
    DefaultExposure,
    DRCResult,
    FRTBDefaultRiskCharge,
    SecuritizedExposure,
    Sector,
    SecuritizationType,
    Seniority,
    calculate_lgd_from_recovery,
    map_external_rating_to_cqs,
)
from neutryx.valuations.regulatory.frtb_rrao import (
    ExoticUnderlying,
    FRTBResidualRiskAddOn,
    LiquidityClass,
    PayoffComplexity,
    RRAOExposure,
    RRAOResult,
    calculate_basis_risk_rrao,
    classify_payoff_complexity,
    estimate_hedge_effectiveness,
    tenor_adjustment,
)


# ==============================================================================
# DRC Tests
# ==============================================================================


class TestDefaultExposure:
    """Test DefaultExposure data structure."""

    def test_exposure_creation(self):
        """Test creating a default exposure."""
        exposure = DefaultExposure(
            issuer_id="CORP_A",
            instrument_type="bond",
            notional=1_000_000.0,
            credit_rating=CreditRating.BBB,
            seniority=Seniority.SENIOR_UNSECURED,
            sector=Sector.INDUSTRIAL,
            maturity_years=5.0,
            long_short="long",
        )

        assert exposure.issuer_id == "CORP_A"
        assert exposure.notional == 1_000_000.0
        assert exposure.is_long is True
        assert exposure.is_short is False

    def test_short_position(self):
        """Test short position identification."""
        exposure = DefaultExposure(
            issuer_id="CORP_B",
            instrument_type="CDS",
            notional=500_000.0,
            credit_rating=CreditRating.A,
            seniority=Seniority.SENIOR_UNSECURED,
            sector=Sector.FINANCIAL,
            maturity_years=3.0,
            long_short="short",
        )

        assert exposure.is_short is True
        assert exposure.is_long is False


class TestFRTBDefaultRiskCharge:
    """Test FRTB DRC calculator."""

    def test_single_issuer_drc(self):
        """Test DRC for single issuer."""
        calculator = FRTBDefaultRiskCharge()

        exposure = DefaultExposure(
            issuer_id="ISSUER_1",
            instrument_type="bond",
            notional=10_000_000.0,
            credit_rating=CreditRating.BBB,  # RW = 0.01
            seniority=Seniority.SENIOR_UNSECURED,  # LGD = 0.40
            sector=Sector.INDUSTRIAL,
            maturity_years=5.0,
        )

        result = calculator.calculate(non_securitized=[exposure])

        # Expected JTD = 10M × 0.40 × 0.01 = 40,000
        expected_jtd = 10_000_000.0 * 0.40 * 0.01
        assert abs(result.non_securitized_drc - expected_jtd) < 1000.0
        assert result.securitized_drc == 0.0
        assert result.total_drc == result.non_securitized_drc

    def test_multiple_issuers_with_correlation(self):
        """Test DRC for multiple issuers with correlation."""
        calculator = FRTBDefaultRiskCharge()

        exposures = [
            DefaultExposure(
                issuer_id="ISSUER_1",
                instrument_type="bond",
                notional=5_000_000.0,
                credit_rating=CreditRating.A,
                seniority=Seniority.SENIOR_UNSECURED,
                sector=Sector.INDUSTRIAL,
                maturity_years=5.0,
            ),
            DefaultExposure(
                issuer_id="ISSUER_2",
                instrument_type="bond",
                notional=5_000_000.0,
                credit_rating=CreditRating.A,
                seniority=Seniority.SENIOR_UNSECURED,
                sector=Sector.INDUSTRIAL,  # Same sector
                maturity_years=5.0,
            ),
        ]

        result = calculator.calculate(non_securitized=exposures)

        # With correlation, DRC < sum of individual JTDs
        individual_jtd = 5_000_000.0 * 0.40 * 0.005  # 10,000 each
        sum_jtds = 2 * individual_jtd

        assert result.non_securitized_drc > individual_jtd  # More than one
        assert result.non_securitized_drc < sum_jtds  # But less than sum due to correlation

    def test_long_short_netting(self):
        """Test netting between long and short positions."""
        calculator = FRTBDefaultRiskCharge()

        exposures = [
            DefaultExposure(
                issuer_id="ISSUER_1",
                instrument_type="bond",
                notional=10_000_000.0,
                credit_rating=CreditRating.BBB,
                seniority=Seniority.SENIOR_UNSECURED,
                sector=Sector.FINANCIAL,
                maturity_years=5.0,
                long_short="long",
            ),
            DefaultExposure(
                issuer_id="ISSUER_1",
                instrument_type="CDS",
                notional=5_000_000.0,
                credit_rating=CreditRating.BBB,
                seniority=Seniority.SENIOR_UNSECURED,
                sector=Sector.FINANCIAL,
                maturity_years=5.0,
                long_short="short",  # Hedge
            ),
        ]

        result = calculator.calculate(non_securitized=exposures)

        # Net exposure should be 5M long (10M - 5M), not 10M
        # Net JTD = 5M × 0.40 × 0.01 = 20,000
        assert result.net_long_jtd > 0
        # When positions net to long, there should be no net short JTD
        assert result.net_short_jtd == 0.0
        # DRC should recognize netting
        assert result.non_securitized_drc < 10_000_000.0 * 0.40 * 0.01

    def test_securitized_drc(self):
        """Test DRC for securitized products."""
        calculator = FRTBDefaultRiskCharge()

        sec_exposure = SecuritizedExposure(
            instrument_id="RMBS_1",
            securitization_type=SecuritizationType.RMBS,
            notional=5_000_000.0,
            tranche_attachment=0.05,
            tranche_detachment=0.10,
            credit_rating=CreditRating.AA,
            underlying_pool_rating=CreditRating.BBB,
            long_short="long",
        )

        result = calculator.calculate(non_securitized=[], securitized=[sec_exposure])

        assert result.securitized_drc > 0.0
        assert result.non_securitized_drc == 0.0
        assert result.total_drc == result.securitized_drc

    def test_drc_by_sector_breakdown(self):
        """Test DRC breakdown by sector."""
        calculator = FRTBDefaultRiskCharge()

        exposures = [
            DefaultExposure(
                issuer_id="FIN_1",
                instrument_type="bond",
                notional=5_000_000.0,
                credit_rating=CreditRating.A,
                seniority=Seniority.SENIOR_UNSECURED,
                sector=Sector.FINANCIAL,
                maturity_years=3.0,
            ),
            DefaultExposure(
                issuer_id="IND_1",
                instrument_type="bond",
                notional=5_000_000.0,
                credit_rating=CreditRating.A,
                seniority=Seniority.SENIOR_UNSECURED,
                sector=Sector.INDUSTRIAL,
                maturity_years=3.0,
            ),
        ]

        result = calculator.calculate(non_securitized=exposures)

        # Should have breakdown for both sectors
        assert Sector.FINANCIAL in result.drc_by_sector
        assert Sector.INDUSTRIAL in result.drc_by_sector
        assert result.drc_by_sector[Sector.FINANCIAL] > 0
        assert result.drc_by_sector[Sector.INDUSTRIAL] > 0


class TestDRCUtilities:
    """Test DRC utility functions."""

    def test_map_external_rating(self):
        """Test rating string mapping."""
        assert map_external_rating_to_cqs("AAA") == CreditRating.AAA
        assert map_external_rating_to_cqs("Aaa") == CreditRating.AAA
        assert map_external_rating_to_cqs("AA+") == CreditRating.AA
        assert map_external_rating_to_cqs("BBB-") == CreditRating.BBB
        assert map_external_rating_to_cqs("B2") == CreditRating.B
        assert map_external_rating_to_cqs("Unknown") == CreditRating.UNRATED

    def test_calculate_lgd_from_recovery(self):
        """Test LGD calculation."""
        assert calculate_lgd_from_recovery(0.40) == 0.60
        assert calculate_lgd_from_recovery(0.60) == 0.40
        assert calculate_lgd_from_recovery(0.0) == 1.0


# ==============================================================================
# RRAO Tests
# ==============================================================================


class TestRRAOExposure:
    """Test RRAOExposure data structure."""

    def test_exposure_creation(self):
        """Test creating RRAO exposure."""
        exposure = RRAOExposure(
            instrument_id="WEATHER_1",
            instrument_type="weather_derivative",
            underlying_type=ExoticUnderlying.WEATHER,
            payoff_complexity=PayoffComplexity.EXOTIC_MEDIUM,
            liquidity_class=LiquidityClass.VERY_ILLIQUID,
            notional=1_000_000.0,
            tenor_years=2.0,
            is_hedged=False,
        )

        assert exposure.underlying_type == ExoticUnderlying.WEATHER
        assert exposure.notional == 1_000_000.0
        assert exposure.is_hedged is False

    def test_hedged_exposure(self):
        """Test hedged exposure."""
        exposure = RRAOExposure(
            instrument_id="LONGEVITY_1",
            instrument_type="longevity_swap",
            underlying_type=ExoticUnderlying.LONGEVITY,
            payoff_complexity=PayoffComplexity.EXOTIC_HIGH,
            liquidity_class=LiquidityClass.VERY_ILLIQUID,
            notional=10_000_000.0,
            tenor_years=10.0,
            is_hedged=True,
            hedge_effectiveness=0.70,
        )

        assert exposure.is_hedged is True
        assert exposure.hedge_effectiveness == 0.70


class TestFRTBResidualRiskAddOn:
    """Test FRTB RRAO calculator."""

    def test_single_exposure_rrao(self):
        """Test RRAO for single exposure."""
        calculator = FRTBResidualRiskAddOn()

        exposure = RRAOExposure(
            instrument_id="WEATHER_1",
            instrument_type="weather_derivative",
            underlying_type=ExoticUnderlying.WEATHER,  # Base RF = 15%
            payoff_complexity=PayoffComplexity.EXOTIC_MEDIUM,  # Mult = 1.5
            liquidity_class=LiquidityClass.VERY_ILLIQUID,  # Mult = 2.0
            notional=1_000_000.0,
            tenor_years=1.0,  # Tenor adj = 1.0
        )

        result = calculator.calculate([exposure])

        # Expected RRAO ≈ 1M × 0.15 × 1.5 × 2.0 × 1.0 × 0.8 (netting) = 360,000
        assert result.total_rrao > 0.0
        assert result.gross_notional == 1_000_000.0

    def test_multiple_exposures_aggregation(self):
        """Test RRAO aggregation across multiple exposures."""
        calculator = FRTBResidualRiskAddOn()

        exposures = [
            RRAOExposure(
                instrument_id="LONGEVITY_1",
                instrument_type="longevity_swap",
                underlying_type=ExoticUnderlying.LONGEVITY,
                payoff_complexity=PayoffComplexity.EXOTIC_HIGH,
                liquidity_class=LiquidityClass.VERY_ILLIQUID,
                notional=5_000_000.0,
                tenor_years=5.0,
            ),
            RRAOExposure(
                instrument_id="WEATHER_1",
                instrument_type="weather_derivative",
                underlying_type=ExoticUnderlying.WEATHER,
                payoff_complexity=PayoffComplexity.EXOTIC_MEDIUM,
                liquidity_class=LiquidityClass.VERY_ILLIQUID,
                notional=3_000_000.0,
                tenor_years=2.0,
            ),
        ]

        result = calculator.calculate(exposures)

        assert result.total_rrao > 0.0
        assert result.gross_notional == 8_000_000.0
        assert len(result.rrao_by_underlying) == 2

    def test_hedge_effectiveness_reduces_rrao(self):
        """Test that hedging reduces RRAO."""
        calculator = FRTBResidualRiskAddOn()

        # Unhedged
        exposure_unhedged = RRAOExposure(
            instrument_id="CAT_1",
            instrument_type="catastrophe_bond",
            underlying_type=ExoticUnderlying.NATURAL_CATASTROPHE,
            payoff_complexity=PayoffComplexity.EXOTIC_LOW,
            liquidity_class=LiquidityClass.ILLIQUID,
            notional=10_000_000.0,
            tenor_years=3.0,
            is_hedged=False,
        )

        result_unhedged = calculator.calculate([exposure_unhedged])

        # Hedged (70% effectiveness)
        exposure_hedged = RRAOExposure(
            instrument_id="CAT_1",
            instrument_type="catastrophe_bond",
            underlying_type=ExoticUnderlying.NATURAL_CATASTROPHE,
            payoff_complexity=PayoffComplexity.EXOTIC_LOW,
            liquidity_class=LiquidityClass.ILLIQUID,
            notional=10_000_000.0,
            tenor_years=3.0,
            is_hedged=True,
            hedge_effectiveness=0.70,
        )

        result_hedged = calculator.calculate([exposure_hedged])

        # Hedged RRAO should be lower
        assert result_hedged.total_rrao < result_unhedged.total_rrao
        # Should be approximately 30% of unhedged (1 - 0.7 effectiveness)
        ratio = result_hedged.total_rrao / result_unhedged.total_rrao
        assert 0.20 < ratio < 0.40  # Allow for netting factor

    def test_complexity_multiplier_effect(self):
        """Test that complexity increases RRAO."""
        calculator = FRTBResidualRiskAddOn()

        # Low complexity
        exposure_low = RRAOExposure(
            instrument_id="DIV_1",
            instrument_type="dividend_swap",
            underlying_type=ExoticUnderlying.DIVIDEND,
            payoff_complexity=PayoffComplexity.EXOTIC_LOW,
            liquidity_class=LiquidityClass.MODERATELY_LIQUID,
            notional=5_000_000.0,
            tenor_years=1.0,
        )

        result_low = calculator.calculate([exposure_low])

        # High complexity (same underlying, higher complexity)
        exposure_high = RRAOExposure(
            instrument_id="DIV_2",
            instrument_type="dividend_derivative",
            underlying_type=ExoticUnderlying.DIVIDEND,
            payoff_complexity=PayoffComplexity.EXOTIC_VERY_HIGH,
            liquidity_class=LiquidityClass.MODERATELY_LIQUID,
            notional=5_000_000.0,
            tenor_years=1.0,
        )

        result_high = calculator.calculate([exposure_high])

        # Higher complexity should yield higher RRAO
        assert result_high.total_rrao > result_low.total_rrao

    def test_liquidity_multiplier_effect(self):
        """Test that illiquidity increases RRAO."""
        calculator = FRTBResidualRiskAddOn()

        # Liquid
        exposure_liquid = RRAOExposure(
            instrument_id="VOL_1",
            instrument_type="volatility_swap",
            underlying_type=ExoticUnderlying.VOLATILITY,
            payoff_complexity=PayoffComplexity.EXOTIC_LOW,
            liquidity_class=LiquidityClass.LIQUID,
            notional=2_000_000.0,
            tenor_years=1.0,
        )

        result_liquid = calculator.calculate([exposure_liquid])

        # Very illiquid
        exposure_illiquid = RRAOExposure(
            instrument_id="VOL_2",
            instrument_type="volatility_derivative",
            underlying_type=ExoticUnderlying.VOLATILITY,
            payoff_complexity=PayoffComplexity.EXOTIC_LOW,
            liquidity_class=LiquidityClass.VERY_ILLIQUID,
            notional=2_000_000.0,
            tenor_years=1.0,
        )

        result_illiquid = calculator.calculate([exposure_illiquid])

        # Illiquid should yield higher RRAO
        assert result_illiquid.total_rrao > result_liquid.total_rrao


class TestRRAOUtilities:
    """Test RRAO utility functions."""

    def test_classify_payoff_complexity(self):
        """Test payoff complexity classification."""
        # Vanilla
        assert classify_payoff_complexity() == PayoffComplexity.VANILLA_OPTION

        # Exotic low (barrier)
        assert classify_payoff_complexity(has_barrier=True) == PayoffComplexity.EXOTIC_LOW

        # Exotic medium (path-dependent)
        assert (
            classify_payoff_complexity(is_path_dependent=True) == PayoffComplexity.EXOTIC_MEDIUM
        )

        # Exotic high (multi-asset)
        assert classify_payoff_complexity(is_multi_asset=True) == PayoffComplexity.EXOTIC_HIGH

        # Exotic very high (correlation)
        assert (
            classify_payoff_complexity(is_correlation_dependent=True)
            == PayoffComplexity.EXOTIC_VERY_HIGH
        )

    def test_estimate_hedge_effectiveness(self):
        """Test hedge effectiveness estimation."""
        # Perfect correlation, full hedge
        effectiveness = estimate_hedge_effectiveness(hedge_pnl_correlation=1.0, hedge_ratio=1.0)
        assert abs(effectiveness - 1.0) < 0.01

        # 50% correlation, full hedge
        effectiveness = estimate_hedge_effectiveness(hedge_pnl_correlation=0.5, hedge_ratio=1.0)
        assert abs(effectiveness - 0.5) < 0.01

        # Perfect correlation, 50% hedge ratio
        effectiveness = estimate_hedge_effectiveness(hedge_pnl_correlation=1.0, hedge_ratio=0.5)
        assert abs(effectiveness - 0.5) < 0.01

        # No correlation
        effectiveness = estimate_hedge_effectiveness(hedge_pnl_correlation=0.0, hedge_ratio=1.0)
        assert abs(effectiveness - 0.0) < 0.01

    def test_tenor_adjustment(self):
        """Test tenor adjustment factor."""
        # 1 year = 1.0
        assert abs(tenor_adjustment(1.0) - 1.0) < 0.01

        # 4 years = 2.0 (square root)
        assert abs(tenor_adjustment(4.0) - 2.0) < 0.01

        # 9 years = 3.0
        assert abs(tenor_adjustment(9.0) - 3.0) < 0.01

    def test_calculate_basis_risk_rrao(self):
        """Test basis risk RRAO calculation."""
        rrao = calculate_basis_risk_rrao(
            notional=10_000_000.0,
            basis_volatility=0.05,  # 5% vol
            tenor_years=1.0,
        )

        # RRAO ≈ 10M × 0.05 × 1.0 × 2.33 = 1,165,000
        assert rrao > 1_000_000.0
        assert rrao < 1_500_000.0


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestDRCRRAOIntegration:
    """Test DRC and RRAO integration."""

    def test_combined_capital_charge(self):
        """Test combined DRC and RRAO charge."""
        # DRC component
        drc_calculator = FRTBDefaultRiskCharge()
        drc_exposures = [
            DefaultExposure(
                issuer_id="CORP_1",
                instrument_type="bond",
                notional=10_000_000.0,
                credit_rating=CreditRating.BBB,
                seniority=Seniority.SENIOR_UNSECURED,
                sector=Sector.FINANCIAL,
                maturity_years=5.0,
            )
        ]
        drc_result = drc_calculator.calculate(non_securitized=drc_exposures)

        # RRAO component
        rrao_calculator = FRTBResidualRiskAddOn()
        rrao_exposures = [
            RRAOExposure(
                instrument_id="EXOTIC_1",
                instrument_type="longevity_swap",
                underlying_type=ExoticUnderlying.LONGEVITY,
                payoff_complexity=PayoffComplexity.EXOTIC_HIGH,
                liquidity_class=LiquidityClass.VERY_ILLIQUID,
                notional=5_000_000.0,
                tenor_years=10.0,
            )
        ]
        rrao_result = rrao_calculator.calculate(rrao_exposures)

        # Total market risk capital = FRTB + DRC + RRAO
        total_additional_capital = drc_result.total_drc + rrao_result.total_rrao

        assert total_additional_capital > 0.0
        assert drc_result.total_drc > 0.0
        assert rrao_result.total_rrao > 0.0

    def test_portfolio_with_all_components(self):
        """Test realistic portfolio with multiple risk types."""
        # Non-securitized credit (DRC)
        drc_calc = FRTBDefaultRiskCharge()
        credit_exposures = [
            DefaultExposure(
                issuer_id=f"ISSUER_{i}",
                instrument_type="bond",
                notional=5_000_000.0,
                credit_rating=CreditRating.A if i % 2 == 0 else CreditRating.BBB,
                seniority=Seniority.SENIOR_UNSECURED,
                sector=Sector.INDUSTRIAL if i % 2 == 0 else Sector.FINANCIAL,
                maturity_years=5.0,
            )
            for i in range(5)
        ]

        # Securitized (DRC)
        sec_exposures = [
            SecuritizedExposure(
                instrument_id="RMBS_1",
                securitization_type=SecuritizationType.RMBS,
                notional=10_000_000.0,
                tranche_attachment=0.05,
                tranche_detachment=0.15,
                credit_rating=CreditRating.AA,
                underlying_pool_rating=CreditRating.BBB,
            )
        ]

        drc_result = drc_calc.calculate(
            non_securitized=credit_exposures, securitized=sec_exposures
        )

        # Exotic underlyings (RRAO)
        rrao_calc = FRTBResidualRiskAddOn()
        exotic_exposures = [
            RRAOExposure(
                instrument_id="LONGEVITY_1",
                instrument_type="longevity_swap",
                underlying_type=ExoticUnderlying.LONGEVITY,
                payoff_complexity=PayoffComplexity.EXOTIC_HIGH,
                liquidity_class=LiquidityClass.VERY_ILLIQUID,
                notional=15_000_000.0,
                tenor_years=20.0,
            ),
            RRAOExposure(
                instrument_id="WEATHER_1",
                instrument_type="weather_derivative",
                underlying_type=ExoticUnderlying.WEATHER,
                payoff_complexity=PayoffComplexity.EXOTIC_MEDIUM,
                liquidity_class=LiquidityClass.ILLIQUID,
                notional=8_000_000.0,
                tenor_years=3.0,
            ),
        ]

        rrao_result = rrao_calc.calculate(exotic_exposures)

        # Verify all components calculated
        assert drc_result.non_securitized_drc > 0.0
        assert drc_result.securitized_drc > 0.0
        assert drc_result.total_drc > 0.0
        assert rrao_result.total_rrao > 0.0

        # Total capital add-on
        total_capital_addon = drc_result.total_drc + rrao_result.total_rrao
        assert total_capital_addon > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
