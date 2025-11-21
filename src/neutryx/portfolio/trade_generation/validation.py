"""
Validation and Warning System for Convention-Based Trade Generation

This module provides validation capabilities to detect deviations from
market conventions. It generates warnings (not errors) when trades use
non-standard conventions, but still allows the trade to be created.

Key Features:
- Multi-level severity (Info, Warning, Error)
- Detailed violation descriptions
- Convention comparison (expected vs actual)
- Configurable validation rules
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Any, Dict

from neutryx.core.dates.schedule import Frequency
from neutryx.core.dates.day_count import DayCountConvention
from neutryx.core.dates.business_day import BusinessDayConvention
from neutryx.market.convention_profiles import (
    ConventionProfile,
    LegConvention,
    ProductTypeConvention,
    get_convention_profile,
)


class ValidationSeverity(Enum):
    """Severity levels for validation warnings"""
    INFO = "info"           # Informational message
    WARNING = "warning"     # Convention deviation but acceptable
    ERROR = "error"         # Significant issue (but still allowed)


@dataclass
class ValidationWarning:
    """
    Represents a validation warning for convention deviation

    Attributes:
        severity: Warning severity level
        field: Field name that deviates from convention
        message: Human-readable description
        expected: Expected value according to convention
        actual: Actual value provided
        recommendation: Suggested action (optional)
    """
    severity: ValidationSeverity
    field: str
    message: str
    expected: Optional[Any] = None
    actual: Optional[Any] = None
    recommendation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            "severity": self.severity.value,
            "field": self.field,
            "message": self.message,
        }
        if self.expected is not None:
            result["expected"] = str(self.expected)
        if self.actual is not None:
            result["actual"] = str(self.actual)
        if self.recommendation:
            result["recommendation"] = self.recommendation
        return result

    def __str__(self) -> str:
        """String representation"""
        parts = [f"[{self.severity.value.upper()}] {self.field}: {self.message}"]
        if self.expected is not None and self.actual is not None:
            parts.append(f"  Expected: {self.expected}, Actual: {self.actual}")
        if self.recommendation:
            parts.append(f"  Recommendation: {self.recommendation}")
        return "\n".join(parts)


@dataclass
class ValidationResult:
    """
    Result of convention validation

    Attributes:
        warnings: List of validation warnings
        is_valid: Whether the trade passes validation (always True for warning-only mode)
        convention_profile: The convention profile used for validation
    """
    warnings: List[ValidationWarning] = field(default_factory=list)
    is_valid: bool = True
    convention_profile: Optional[ConventionProfile] = None

    def has_warnings(self) -> bool:
        """Check if any warnings were generated"""
        return len(self.warnings) > 0

    def has_errors(self) -> bool:
        """Check if any ERROR-level warnings were generated"""
        return any(w.severity == ValidationSeverity.ERROR for w in self.warnings)

    def get_warnings_by_severity(self, severity: ValidationSeverity) -> List[ValidationWarning]:
        """Get warnings filtered by severity"""
        return [w for w in self.warnings if w.severity == severity]

    def add_warning(self, warning: ValidationWarning):
        """Add a warning to the result"""
        self.warnings.append(warning)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "is_valid": self.is_valid,
            "warnings": [w.to_dict() for w in self.warnings],
            "has_warnings": self.has_warnings(),
            "has_errors": self.has_errors(),
        }

    def __str__(self) -> str:
        """String representation"""
        if not self.has_warnings():
            return "Validation passed with no warnings"

        lines = [f"Validation completed with {len(self.warnings)} warning(s):"]
        for warning in self.warnings:
            lines.append(str(warning))
        return "\n".join(lines)


class ConventionValidator:
    """
    Validator for checking convention compliance

    This validator compares actual trade parameters against market conventions
    and generates warnings for deviations. It operates in warning-only mode,
    meaning trades are never rejected, only flagged.
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator

        Args:
            strict_mode: If True, generate ERROR-level warnings for serious deviations
                        If False (default), all warnings are WARNING-level
        """
        self.strict_mode = strict_mode

    def validate_trade_parameters(
        self,
        currency: str,
        product_type: ProductTypeConvention,
        fixed_leg_params: Optional[Dict[str, Any]] = None,
        floating_leg_params: Optional[Dict[str, Any]] = None,
        other_params: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate trade parameters against market conventions

        Args:
            currency: Currency code
            product_type: Product type
            fixed_leg_params: Fixed leg parameters (if applicable)
            floating_leg_params: Floating leg parameters (if applicable)
            other_params: Other trade parameters (spot_lag, calendars, etc.)

        Returns:
            ValidationResult with warnings
        """
        result = ValidationResult()

        # Get convention profile
        profile = get_convention_profile(currency, product_type)
        if profile is None:
            result.add_warning(ValidationWarning(
                severity=ValidationSeverity.INFO,
                field="convention_profile",
                message=f"No standard convention profile found for {currency} {product_type.value}",
                recommendation="Trade will be created with provided parameters",
            ))
            return result

        result.convention_profile = profile

        # Validate fixed leg
        if fixed_leg_params and profile.fixed_leg:
            self._validate_leg(
                result,
                "fixed_leg",
                profile.fixed_leg,
                fixed_leg_params,
            )

        # Validate floating leg
        if floating_leg_params and profile.floating_leg:
            self._validate_leg(
                result,
                "floating_leg",
                profile.floating_leg,
                floating_leg_params,
            )

        # Validate other parameters
        if other_params:
            self._validate_other_params(result, profile, other_params)

        return result

    def _validate_leg(
        self,
        result: ValidationResult,
        leg_name: str,
        convention_leg: LegConvention,
        actual_params: Dict[str, Any],
    ):
        """Validate a single leg against conventions"""
        # Check frequency
        if "frequency" in actual_params:
            actual_freq = actual_params["frequency"]
            if actual_freq != convention_leg.frequency:
                severity = ValidationSeverity.ERROR if self.strict_mode else ValidationSeverity.WARNING
                result.add_warning(ValidationWarning(
                    severity=severity,
                    field=f"{leg_name}.frequency",
                    message=f"Non-standard payment frequency for {leg_name}",
                    expected=convention_leg.frequency,
                    actual=actual_freq,
                    recommendation=f"Standard market convention is {convention_leg.frequency}",
                ))

        # Check day count
        if "day_count" in actual_params:
            actual_dc = actual_params["day_count"]
            # Compare by type and string representation (DayCountConvention instances)
            if type(actual_dc) != type(convention_leg.day_count):
                severity = ValidationSeverity.ERROR if self.strict_mode else ValidationSeverity.WARNING
                result.add_warning(ValidationWarning(
                    severity=severity,
                    field=f"{leg_name}.day_count",
                    message=f"Non-standard day count convention for {leg_name}",
                    expected=convention_leg.day_count,
                    actual=actual_dc,
                    recommendation=f"Standard market convention is {convention_leg.day_count}",
                ))

        # Check business day convention
        if "business_day_convention" in actual_params:
            actual_bdc = actual_params["business_day_convention"]
            # Compare by type (BusinessDayConvention instances)
            if type(actual_bdc) != type(convention_leg.business_day_convention):
                severity = ValidationSeverity.INFO  # Less critical
                result.add_warning(ValidationWarning(
                    severity=severity,
                    field=f"{leg_name}.business_day_convention",
                    message=f"Non-standard business day convention for {leg_name}",
                    expected=convention_leg.business_day_convention,
                    actual=actual_bdc,
                ))

        # Check rate index for floating legs
        if "rate_index" in actual_params and convention_leg.rate_index:
            actual_index = actual_params["rate_index"]
            expected_index_name = convention_leg.rate_index.name
            actual_index_name = actual_index.name if hasattr(actual_index, "name") else str(actual_index)

            if actual_index_name != expected_index_name:
                severity = ValidationSeverity.WARNING
                result.add_warning(ValidationWarning(
                    severity=severity,
                    field=f"{leg_name}.rate_index",
                    message=f"Non-standard rate index for {leg_name}",
                    expected=expected_index_name,
                    actual=actual_index_name,
                    recommendation=f"Standard market index is {expected_index_name}",
                ))

        # Check payment lag
        if "payment_lag" in actual_params:
            actual_lag = actual_params["payment_lag"]
            if actual_lag != convention_leg.payment_lag and convention_leg.payment_lag != 0:
                severity = ValidationSeverity.INFO
                result.add_warning(ValidationWarning(
                    severity=severity,
                    field=f"{leg_name}.payment_lag",
                    message=f"Non-standard payment lag for {leg_name}",
                    expected=convention_leg.payment_lag,
                    actual=actual_lag,
                ))

    def _validate_other_params(
        self,
        result: ValidationResult,
        profile: ConventionProfile,
        actual_params: Dict[str, Any],
    ):
        """Validate other trade parameters"""
        # Check spot lag
        if "spot_lag" in actual_params:
            actual_lag = actual_params["spot_lag"]
            if actual_lag != profile.spot_lag:
                severity = ValidationSeverity.WARNING
                result.add_warning(ValidationWarning(
                    severity=severity,
                    field="spot_lag",
                    message="Non-standard spot lag",
                    expected=profile.spot_lag,
                    actual=actual_lag,
                    recommendation=f"Standard spot lag for {profile.currency} is {profile.spot_lag} days",
                ))

        # Check calendars
        if "calendars" in actual_params:
            actual_calendars = actual_params["calendars"]
            if set(actual_calendars) != set(profile.calendars):
                severity = ValidationSeverity.INFO
                result.add_warning(ValidationWarning(
                    severity=severity,
                    field="calendars",
                    message="Non-standard holiday calendars",
                    expected=profile.calendars,
                    actual=actual_calendars,
                ))

        # Check end of month rule
        if "end_of_month" in actual_params:
            actual_eom = actual_params["end_of_month"]
            if actual_eom != profile.end_of_month:
                severity = ValidationSeverity.INFO
                result.add_warning(ValidationWarning(
                    severity=severity,
                    field="end_of_month",
                    message="Non-standard end-of-month rule",
                    expected=profile.end_of_month,
                    actual=actual_eom,
                ))

    def validate_conventions_match_profile(
        self,
        profile: ConventionProfile,
        actual_conventions: Dict[str, Any],
    ) -> ValidationResult:
        """
        Validate that actual conventions match a specific profile

        Args:
            profile: Convention profile to validate against
            actual_conventions: Actual conventions used

        Returns:
            ValidationResult with warnings
        """
        result = ValidationResult(convention_profile=profile)

        # Extract leg parameters
        fixed_leg_params = actual_conventions.get("fixed_leg")
        floating_leg_params = actual_conventions.get("floating_leg")
        other_params = {
            k: v for k, v in actual_conventions.items()
            if k not in ["fixed_leg", "floating_leg"]
        }

        # Validate legs
        if fixed_leg_params and profile.fixed_leg:
            self._validate_leg(result, "fixed_leg", profile.fixed_leg, fixed_leg_params)

        if floating_leg_params and profile.floating_leg:
            self._validate_leg(result, "floating_leg", profile.floating_leg, floating_leg_params)

        # Validate other parameters
        if other_params:
            self._validate_other_params(result, profile, other_params)

        return result


# Convenience function for quick validation
def validate_trade(
    currency: str,
    product_type: ProductTypeConvention,
    **kwargs
) -> ValidationResult:
    """
    Convenience function for quick trade validation

    Args:
        currency: Currency code
        product_type: Product type
        **kwargs: Trade parameters to validate

    Returns:
        ValidationResult

    Example:
        >>> result = validate_trade(
        ...     "USD",
        ...     ProductTypeConvention.INTEREST_RATE_SWAP,
        ...     fixed_leg_params={"frequency": Frequency.QUARTERLY},
        ... )
        >>> print(result)
    """
    validator = ConventionValidator()
    return validator.validate_trade_parameters(currency, product_type, **kwargs)
