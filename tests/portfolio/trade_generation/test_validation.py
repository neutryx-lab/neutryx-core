"""Tests for Validation and Warning System"""

import pytest
from neutryx.portfolio.trade_generation.validation import (
    ValidationWarning,
    ValidationSeverity,
    ValidationResult,
    ConventionValidator,
    validate_trade,
)
from neutryx.market.convention_profiles import (
    ProductTypeConvention,
    get_convention_profile,
)
from neutryx.core.dates.schedule import Frequency
from neutryx.core.dates.day_count import ACT_360, ACT_365, THIRTY_360
from neutryx.core.dates.business_day import MODIFIED_FOLLOWING


class TestValidationWarning:
    """Test ValidationWarning class"""

    def test_warning_creation(self):
        """Test creating a validation warning"""
        warning = ValidationWarning(
            severity=ValidationSeverity.WARNING,
            field="fixed_leg.frequency",
            message="Non-standard payment frequency",
            expected=Frequency.SEMI_ANNUAL,
            actual=Frequency.QUARTERLY,
        )

        assert warning.severity == ValidationSeverity.WARNING
        assert warning.field == "fixed_leg.frequency"
        assert warning.expected == Frequency.SEMI_ANNUAL
        assert warning.actual == Frequency.QUARTERLY

    def test_warning_to_dict(self):
        """Test converting warning to dictionary"""
        warning = ValidationWarning(
            severity=ValidationSeverity.ERROR,
            field="day_count",
            message="Invalid day count",
            recommendation="Use ACT/360",
        )

        warning_dict = warning.to_dict()
        assert warning_dict["severity"] == "error"
        assert warning_dict["field"] == "day_count"
        assert warning_dict["recommendation"] == "Use ACT/360"

    def test_warning_str(self):
        """Test string representation of warning"""
        warning = ValidationWarning(
            severity=ValidationSeverity.WARNING,
            field="test_field",
            message="Test message",
        )

        warning_str = str(warning)
        assert "WARNING" in warning_str
        assert "test_field" in warning_str
        assert "Test message" in warning_str


class TestValidationResult:
    """Test ValidationResult class"""

    def test_result_with_no_warnings(self):
        """Test validation result with no warnings"""
        result = ValidationResult()

        assert result.is_valid
        assert not result.has_warnings()
        assert not result.has_errors()
        assert len(result.warnings) == 0

    def test_result_with_warnings(self):
        """Test validation result with warnings"""
        result = ValidationResult()

        warning1 = ValidationWarning(
            severity=ValidationSeverity.WARNING,
            field="field1",
            message="Warning 1",
        )
        warning2 = ValidationWarning(
            severity=ValidationSeverity.INFO,
            field="field2",
            message="Info 1",
        )

        result.add_warning(warning1)
        result.add_warning(warning2)

        assert result.has_warnings()
        assert len(result.warnings) == 2

    def test_get_warnings_by_severity(self):
        """Test filtering warnings by severity"""
        result = ValidationResult()

        result.add_warning(ValidationWarning(ValidationSeverity.WARNING, "f1", "w1"))
        result.add_warning(ValidationWarning(ValidationSeverity.INFO, "f2", "i1"))
        result.add_warning(ValidationWarning(ValidationSeverity.ERROR, "f3", "e1"))

        warnings = result.get_warnings_by_severity(ValidationSeverity.WARNING)
        assert len(warnings) == 1
        assert warnings[0].field == "f1"

        errors = result.get_warnings_by_severity(ValidationSeverity.ERROR)
        assert len(errors) == 1
        assert errors[0].field == "f3"

    def test_result_to_dict(self):
        """Test converting result to dictionary"""
        result = ValidationResult()
        result.add_warning(ValidationWarning(ValidationSeverity.INFO, "f1", "m1"))

        result_dict = result.to_dict()
        assert result_dict["is_valid"]
        assert result_dict["has_warnings"]
        assert len(result_dict["warnings"]) == 1


class TestConventionValidator:
    """Test ConventionValidator class"""

    def test_validator_creation(self):
        """Test creating a validator"""
        validator = ConventionValidator()
        assert validator.strict_mode == False

        strict_validator = ConventionValidator(strict_mode=True)
        assert strict_validator.strict_mode == True

    def test_validate_standard_usd_irs(self):
        """Test validating standard USD IRS with no deviations"""
        validator = ConventionValidator()

        result = validator.validate_trade_parameters(
            currency="USD",
            product_type=ProductTypeConvention.INTEREST_RATE_SWAP,
            fixed_leg_params={
                "frequency": Frequency.SEMI_ANNUAL,
                "day_count": THIRTY_360,
            },
            floating_leg_params={
                "frequency": Frequency.QUARTERLY,
                "day_count": ACT_360,
            },
        )

        # Should have no warnings for standard conventions
        assert not result.has_warnings()

    def test_validate_non_standard_frequency(self):
        """Test validating IRS with non-standard frequency"""
        validator = ConventionValidator()

        result = validator.validate_trade_parameters(
            currency="USD",
            product_type=ProductTypeConvention.INTEREST_RATE_SWAP,
            fixed_leg_params={
                "frequency": Frequency.QUARTERLY,  # Non-standard for USD fixed leg
                "day_count": THIRTY_360,
            },
            floating_leg_params={
                "frequency": Frequency.QUARTERLY,
                "day_count": ACT_360,
            },
        )

        # Should have warning for non-standard frequency
        assert result.has_warnings()
        warnings = result.get_warnings_by_severity(ValidationSeverity.WARNING)
        assert len(warnings) >= 1
        assert any("frequency" in w.field for w in warnings)

    def test_validate_non_standard_day_count(self):
        """Test validating IRS with non-standard day count"""
        validator = ConventionValidator()

        result = validator.validate_trade_parameters(
            currency="USD",
            product_type=ProductTypeConvention.INTEREST_RATE_SWAP,
            fixed_leg_params={
                "frequency": Frequency.SEMI_ANNUAL,
                "day_count": ACT_360,  # Non-standard for USD fixed leg
            },
            floating_leg_params={
                "frequency": Frequency.QUARTERLY,
                "day_count": ACT_360,
            },
        )

        # Should have warning for non-standard day count
        assert result.has_warnings()
        warnings = result.get_warnings_by_severity(ValidationSeverity.WARNING)
        assert any("day_count" in w.field for w in warnings)

    def test_validate_strict_mode(self):
        """Test validator in strict mode"""
        validator = ConventionValidator(strict_mode=True)

        result = validator.validate_trade_parameters(
            currency="USD",
            product_type=ProductTypeConvention.INTEREST_RATE_SWAP,
            fixed_leg_params={
                "frequency": Frequency.QUARTERLY,  # Non-standard
                "day_count": THIRTY_360,
            },
        )

        # In strict mode, should generate ERROR-level warnings
        assert result.has_errors()
        errors = result.get_warnings_by_severity(ValidationSeverity.ERROR)
        assert len(errors) >= 1

    def test_validate_unknown_currency(self):
        """Test validating with unknown currency"""
        validator = ConventionValidator()

        result = validator.validate_trade_parameters(
            currency="ZZZ",
            product_type=ProductTypeConvention.INTEREST_RATE_SWAP,
            fixed_leg_params={"frequency": Frequency.SEMI_ANNUAL},
        )

        # Should have INFO warning about missing profile
        assert result.has_warnings()
        info_warnings = result.get_warnings_by_severity(ValidationSeverity.INFO)
        assert len(info_warnings) >= 1

    def test_validate_other_params(self):
        """Test validating other parameters like spot_lag"""
        validator = ConventionValidator()

        result = validator.validate_trade_parameters(
            currency="USD",
            product_type=ProductTypeConvention.INTEREST_RATE_SWAP,
            other_params={
                "spot_lag": 5,  # Non-standard for USD (should be 2)
                "calendars": ["USD"],
            },
        )

        # Should have warning for non-standard spot lag
        assert result.has_warnings()
        warnings = result.warnings
        assert any("spot_lag" in w.field for w in warnings)

    def test_validate_eur_ois(self):
        """Test validating EUR OIS"""
        validator = ConventionValidator()

        result = validator.validate_trade_parameters(
            currency="EUR",
            product_type=ProductTypeConvention.OVERNIGHT_INDEX_SWAP,
            fixed_leg_params={
                "frequency": Frequency.ANNUAL,
                "day_count": ACT_360,
            },
            floating_leg_params={
                "frequency": Frequency.ANNUAL,
                "day_count": ACT_360,
            },
        )

        # Standard EUR OIS should have no warnings
        assert not result.has_warnings()


class TestConvenienceFunction:
    """Test convenience validation function"""

    def test_validate_trade_function(self):
        """Test validate_trade convenience function"""
        result = validate_trade(
            "USD",
            ProductTypeConvention.INTEREST_RATE_SWAP,
            fixed_leg_params={
                "frequency": Frequency.QUARTERLY,  # Non-standard
                "day_count": DayCountConvention.THIRTY_360,
            },
        )

        assert result.has_warnings()


class TestMultiCurrencyValidation:
    """Test validation across multiple currencies"""

    @pytest.mark.parametrize("currency,fixed_freq", [
        ("USD", Frequency.SEMI_ANNUAL),
        ("EUR", Frequency.ANNUAL),
        ("GBP", Frequency.SEMI_ANNUAL),
        ("JPY", Frequency.SEMI_ANNUAL),
    ])
    def test_standard_conventions_no_warnings(self, currency, fixed_freq):
        """Test that standard conventions produce no warnings"""
        validator = ConventionValidator()
        profile = get_convention_profile(currency, ProductTypeConvention.INTEREST_RATE_SWAP)

        result = validator.validate_trade_parameters(
            currency=currency,
            product_type=ProductTypeConvention.INTEREST_RATE_SWAP,
            fixed_leg_params={
                "frequency": profile.fixed_leg.frequency,
                "day_count": profile.fixed_leg.day_count,
            },
            floating_leg_params={
                "frequency": profile.floating_leg.frequency,
                "day_count": profile.floating_leg.day_count,
            },
        )

        assert not result.has_warnings()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
