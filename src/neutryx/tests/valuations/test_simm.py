"""Tests for ISDA SIMM implementation."""
import pytest

from neutryx.valuations.simm import (
    RiskClass,
    RiskFactorSensitivity,
    RiskFactorType,
    SensitivityType,
    SIMMCalculator,
    calculate_simm,
    get_correlations,
    get_risk_weights,
)
from neutryx.valuations.simm.sensitivities import (
    aggregate_sensitivities_by_tenor,
    bucket_sensitivities,
    get_credit_bucket,
    get_equity_bucket,
    get_fx_bucket,
    get_ir_bucket,
)


class TestRiskWeights:
    """Tests for SIMM risk weights."""

    def test_ir_risk_weights(self):
        """Test interest rate risk weights."""
        # USD 5Y risk weight
        rw = get_risk_weights(RiskClass.INTEREST_RATE, bucket="USD", tenor="5Y")
        assert rw == 52.0

        # EUR 10Y risk weight
        rw_eur = get_risk_weights(RiskClass.INTEREST_RATE, bucket="EUR", tenor="10Y")
        assert rw_eur == 50.0

        # GBP 2Y risk weight
        rw_gbp = get_risk_weights(RiskClass.INTEREST_RATE, bucket="GBP", tenor="2Y")
        assert rw_gbp == 58.0

    def test_fx_risk_weights(self):
        """Test FX risk weights."""
        rw = get_risk_weights(RiskClass.FX)
        assert rw == 10.0  # Simplified single weight

    def test_equity_risk_weights(self):
        """Test equity risk weights."""
        # Bucket 1 (Emerging Markets - Consumer)
        rw1 = get_risk_weights(RiskClass.EQUITY, bucket="1")
        assert rw1 == 26.0

        # Bucket 10 (Developed Markets Indices)
        rw10 = get_risk_weights(RiskClass.EQUITY, bucket="10")
        assert rw10 == 16.0

    def test_credit_risk_weights(self):
        """Test credit risk weights."""
        # Bucket 1 (Sovereigns IG)
        rw1 = get_risk_weights(RiskClass.CREDIT_QUALIFYING, bucket="1")
        assert rw1 == 85.0

        # Bucket 7 (TMT IG)
        rw7 = get_risk_weights(RiskClass.CREDIT_QUALIFYING, bucket="7")
        assert rw7 == 161.0

    def test_missing_tenor_fallback(self):
        """Test fallback for missing tenor."""
        # Request unknown tenor, should use default
        rw = get_risk_weights(RiskClass.INTEREST_RATE, bucket="USD", tenor="100Y")
        assert rw == 100.0  # Default fallback

    def test_missing_currency_fallback(self):
        """Test fallback for missing currency."""
        # Unknown currency should fall back to USD
        rw = get_risk_weights(RiskClass.INTEREST_RATE, bucket="XXX", tenor="5Y")
        assert rw == 52.0  # USD 5Y weight


class TestCorrelations:
    """Tests for SIMM correlation parameters."""

    def test_within_bucket_correlations(self):
        """Test within-bucket correlations."""
        # IR within bucket (same currency)
        rho_ir = get_correlations(RiskClass.INTEREST_RATE, within_bucket=True)
        assert rho_ir == 0.99  # Very high correlation

        # Equity within bucket
        rho_eq = get_correlations(RiskClass.EQUITY, within_bucket=True)
        assert rho_eq == 0.15  # Lower correlation

    def test_cross_bucket_correlations(self):
        """Test cross-bucket correlations."""
        # IR cross bucket (different currencies)
        rho_ir = get_correlations(RiskClass.INTEREST_RATE, within_bucket=False)
        assert rho_ir == 0.27

        # FX cross bucket
        rho_fx = get_correlations(RiskClass.FX, within_bucket=False)
        assert rho_fx == 0.60


class TestSensitivities:
    """Tests for risk factor sensitivities."""

    def test_risk_factor_sensitivity_creation(self):
        """Test creating risk factor sensitivity."""
        sens = RiskFactorSensitivity(
            risk_factor_type=RiskFactorType.IR,
            sensitivity_type=SensitivityType.DELTA,
            bucket="USD",
            risk_factor="USD-LIBOR-3M",
            sensitivity=10000.0,
            tenor="5Y",
        )

        assert sens.risk_factor_type == RiskFactorType.IR
        assert sens.sensitivity_type == SensitivityType.DELTA
        assert sens.bucket == "USD"
        assert sens.sensitivity == 10000.0
        assert sens.tenor == "5Y"

    def test_bucket_sensitivities(self):
        """Test bucketing of sensitivities."""
        sensitivities = [
            RiskFactorSensitivity(
                RiskFactorType.IR, SensitivityType.DELTA,
                "USD", "USD-LIBOR-3M", 10000.0, "3M"
            ),
            RiskFactorSensitivity(
                RiskFactorType.IR, SensitivityType.DELTA,
                "USD", "USD-LIBOR-6M", 15000.0, "6M"
            ),
            RiskFactorSensitivity(
                RiskFactorType.IR, SensitivityType.DELTA,
                "EUR", "EUR-EURIBOR-3M", 8000.0, "3M"
            ),
            RiskFactorSensitivity(
                RiskFactorType.FX, SensitivityType.DELTA,
                "1", "EURUSD", 5000.0
            ),
        ]

        bucketed = bucket_sensitivities(sensitivities)

        # Should have two keys: (IR, Delta) and (FX, Delta)
        assert len(bucketed) == 2
        assert (RiskFactorType.IR, SensitivityType.DELTA) in bucketed
        assert (RiskFactorType.FX, SensitivityType.DELTA) in bucketed

        # IR should have two buckets (USD, EUR)
        ir_bucketed = bucketed[(RiskFactorType.IR, SensitivityType.DELTA)]
        assert len(ir_bucketed.bucket_sensitivities) == 2
        assert "USD" in ir_bucketed.bucket_sensitivities
        assert "EUR" in ir_bucketed.bucket_sensitivities

        # USD bucket should have 2 sensitivities
        assert len(ir_bucketed.bucket_sensitivities["USD"]) == 2

    def test_bucketing_helpers(self):
        """Test bucket assignment helpers."""
        # IR bucket
        assert get_ir_bucket("USD") == "USD"
        assert get_ir_bucket("EUR") == "EUR"

        # FX bucket (single bucket)
        assert get_fx_bucket("EURUSD") == "1"

        # Equity bucket
        eq_bucket = get_equity_bucket("EmergingMarkets", "Consumer")
        assert eq_bucket == "1"

        eq_bucket2 = get_equity_bucket("Developed", "Energy")
        assert eq_bucket2 == "7"

        # Credit bucket
        cr_bucket = get_credit_bucket("AAA", "Sovereign")
        assert cr_bucket == "1"

        cr_bucket2 = get_credit_bucket("BB", "Industrial")
        assert cr_bucket2 == "2"

    def test_aggregate_sensitivities_by_tenor(self):
        """Test aggregating IR sensitivities by tenor."""
        sensitivities = [
            RiskFactorSensitivity(
                RiskFactorType.IR, SensitivityType.DELTA,
                "USD", "USD-LIBOR-5Y", 10000.0, "5Y"
            ),
            RiskFactorSensitivity(
                RiskFactorType.IR, SensitivityType.DELTA,
                "USD", "USD-OIS-5Y", 5000.0, "5Y"
            ),
            RiskFactorSensitivity(
                RiskFactorType.IR, SensitivityType.DELTA,
                "USD", "USD-LIBOR-10Y", 8000.0, "10Y"
            ),
            RiskFactorSensitivity(
                RiskFactorType.IR, SensitivityType.DELTA,
                "EUR", "EUR-EURIBOR-5Y", 3000.0, "5Y"
            ),
        ]

        # Aggregate USD sensitivities
        usd_by_tenor = aggregate_sensitivities_by_tenor(sensitivities, "USD")

        assert "5Y" in usd_by_tenor
        assert "10Y" in usd_by_tenor
        assert usd_by_tenor["5Y"] == 15000.0  # 10000 + 5000
        assert usd_by_tenor["10Y"] == 8000.0


class TestSIMMCalculator:
    """Tests for SIMM calculator."""

    def test_simm_calculator_creation(self):
        """Test creating SIMM calculator."""
        calc = SIMMCalculator(product_class_multiplier=1.0)
        assert calc.product_class_multiplier == 1.0

    def test_simple_ir_simm_calculation(self):
        """Test SIMM with simple IR sensitivities."""
        sensitivities = [
            RiskFactorSensitivity(
                RiskFactorType.IR, SensitivityType.DELTA,
                "USD", "USD-LIBOR-5Y", 100000.0, "5Y"
            ),
            RiskFactorSensitivity(
                RiskFactorType.IR, SensitivityType.DELTA,
                "USD", "USD-LIBOR-10Y", 150000.0, "10Y"
            ),
        ]

        result = calculate_simm(sensitivities)

        # Should have positive IM
        assert result.total_im > 0

        # Should have delta component
        assert result.delta_im > 0

        # Vega should be 0 (no vega sensitivities)
        assert result.vega_im == 0.0

        # Should have IR in risk class breakdown
        assert RiskClass.INTEREST_RATE in result.im_by_risk_class

    def test_multi_currency_ir_simm(self):
        """Test SIMM with multiple currencies."""
        sensitivities = [
            RiskFactorSensitivity(
                RiskFactorType.IR, SensitivityType.DELTA,
                "USD", "USD-LIBOR-5Y", 100000.0, "5Y"
            ),
            RiskFactorSensitivity(
                RiskFactorType.IR, SensitivityType.DELTA,
                "EUR", "EUR-EURIBOR-5Y", 80000.0, "5Y"
            ),
            RiskFactorSensitivity(
                RiskFactorType.IR, SensitivityType.DELTA,
                "GBP", "GBP-LIBOR-5Y", 60000.0, "5Y"
            ),
        ]

        result = calculate_simm(sensitivities)

        # Cross-currency IM should benefit from diversification
        # compared to sum of individual currency IMs
        assert result.total_im > 0
        assert result.delta_im > 0

    def test_fx_simm_calculation(self):
        """Test SIMM with FX sensitivities."""
        sensitivities = [
            RiskFactorSensitivity(
                RiskFactorType.FX, SensitivityType.DELTA,
                "1", "EURUSD", 50000.0
            ),
            RiskFactorSensitivity(
                RiskFactorType.FX, SensitivityType.DELTA,
                "1", "GBPUSD", 30000.0
            ),
        ]

        result = calculate_simm(sensitivities)

        assert result.total_im > 0
        assert RiskClass.FX in result.im_by_risk_class

    def test_equity_simm_calculation(self):
        """Test SIMM with equity sensitivities."""
        sensitivities = [
            RiskFactorSensitivity(
                RiskFactorType.EQUITY, SensitivityType.DELTA,
                "1", "AAPL", 25000.0
            ),
            RiskFactorSensitivity(
                RiskFactorType.EQUITY, SensitivityType.DELTA,
                "1", "MSFT", 30000.0
            ),
            RiskFactorSensitivity(
                RiskFactorType.EQUITY, SensitivityType.DELTA,
                "2", "XOM", 20000.0
            ),
        ]

        result = calculate_simm(sensitivities)

        assert result.total_im > 0
        assert RiskClass.EQUITY in result.im_by_risk_class

    def test_vega_simm_calculation(self):
        """Test SIMM with vega sensitivities."""
        sensitivities = [
            RiskFactorSensitivity(
                RiskFactorType.IR, SensitivityType.VEGA,
                "USD", "USD-SWAPTION-5Y", 5000.0, "5Y"
            ),
            RiskFactorSensitivity(
                RiskFactorType.EQUITY, SensitivityType.VEGA,
                "1", "SPX-VOL", 8000.0
            ),
        ]

        result = calculate_simm(sensitivities)

        # Should have vega component
        assert result.vega_im > 0

        # Delta should be 0 (no delta sensitivities)
        assert result.delta_im == 0.0

    def test_delta_and_vega_combined(self):
        """Test SIMM with both delta and vega."""
        sensitivities = [
            RiskFactorSensitivity(
                RiskFactorType.IR, SensitivityType.DELTA,
                "USD", "USD-LIBOR-5Y", 100000.0, "5Y"
            ),
            RiskFactorSensitivity(
                RiskFactorType.IR, SensitivityType.VEGA,
                "USD", "USD-SWAPTION-5Y", 10000.0, "5Y"
            ),
        ]

        result = calculate_simm(sensitivities)

        # Should have both components
        assert result.delta_im > 0
        assert result.vega_im > 0

        # Total should be sum (simplified aggregation)
        assert result.total_im == result.delta_im + result.vega_im

    def test_product_class_multiplier(self):
        """Test product class multiplier effect."""
        sensitivities = [
            RiskFactorSensitivity(
                RiskFactorType.IR, SensitivityType.DELTA,
                "USD", "USD-LIBOR-5Y", 100000.0, "5Y"
            ),
        ]

        # Calculate with multiplier = 1.0
        result_1 = calculate_simm(sensitivities, product_class_multiplier=1.0)

        # Calculate with multiplier = 1.5
        result_15 = calculate_simm(sensitivities, product_class_multiplier=1.5)

        # Total IM should be 1.5x
        assert abs(result_15.total_im - result_1.total_im * 1.5) < 0.01

    def test_offsetting_sensitivities(self):
        """Test that offsetting sensitivities reduce IM."""
        # Long and short in same risk factor
        sensitivities_long = [
            RiskFactorSensitivity(
                RiskFactorType.IR, SensitivityType.DELTA,
                "USD", "USD-LIBOR-5Y", 100000.0, "5Y"
            ),
        ]

        sensitivities_offset = [
            RiskFactorSensitivity(
                RiskFactorType.IR, SensitivityType.DELTA,
                "USD", "USD-LIBOR-5Y", 100000.0, "5Y"
            ),
            RiskFactorSensitivity(
                RiskFactorType.IR, SensitivityType.DELTA,
                "USD", "USD-LIBOR-5Y", -80000.0, "5Y"
            ),
        ]

        result_long = calculate_simm(sensitivities_long)
        result_offset = calculate_simm(sensitivities_offset)

        # Offsetting position should have lower IM
        assert result_offset.total_im < result_long.total_im

    def test_simm_result_repr(self):
        """Test SIMMResult string representation."""
        sensitivities = [
            RiskFactorSensitivity(
                RiskFactorType.IR, SensitivityType.DELTA,
                "USD", "USD-LIBOR-5Y", 100000.0, "5Y"
            ),
        ]

        result = calculate_simm(sensitivities)
        repr_str = repr(result)

        # Should contain key information
        assert "SIMMResult" in repr_str
        assert "total_im" in repr_str
        assert "delta" in repr_str
        assert "vega" in repr_str


class TestSIMMEdgeCases:
    """Tests for SIMM edge cases."""

    def test_empty_sensitivities(self):
        """Test SIMM with no sensitivities."""
        result = calculate_simm([])

        # Should return zero IM
        assert result.total_im == 0.0
        assert result.delta_im == 0.0
        assert result.vega_im == 0.0

    def test_zero_sensitivity(self):
        """Test SIMM with zero sensitivity value."""
        sensitivities = [
            RiskFactorSensitivity(
                RiskFactorType.IR, SensitivityType.DELTA,
                "USD", "USD-LIBOR-5Y", 0.0, "5Y"
            ),
        ]

        result = calculate_simm(sensitivities)

        # Should return zero IM
        assert result.total_im == 0.0

    def test_small_sensitivity(self):
        """Test SIMM with very small sensitivity."""
        sensitivities = [
            RiskFactorSensitivity(
                RiskFactorType.IR, SensitivityType.DELTA,
                "USD", "USD-LIBOR-5Y", 1.0, "5Y"  # Very small
            ),
        ]

        result = calculate_simm(sensitivities)

        # Should return small but positive IM
        assert result.total_im > 0
        assert result.total_im < 100  # Should be quite small


def test_risk_factor_sensitivity_repr():
    """Test RiskFactorSensitivity string representation."""
    sens = RiskFactorSensitivity(
        RiskFactorType.IR,
        SensitivityType.DELTA,
        "USD",
        "USD-LIBOR-5Y",
        10000.0,
        "5Y",
    )

    repr_str = repr(sens)

    # Should contain key information
    assert "RiskFactorSensitivity" in repr_str
    assert "InterestRate" in repr_str
    assert "Delta" in repr_str
    assert "USD" in repr_str
    assert "10000.00" in repr_str
    assert "5Y" in repr_str
