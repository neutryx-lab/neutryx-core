"""
Trade Generation Module

Convention-based trade generation system that creates trades conforming
to market standards while allowing individual overrides.

Modules:
    factory: Core trade factory for generating trades from conventions
    validation: Validation and warning system for convention compliance
    generators: Product-specific trade generators
"""

from neutryx.portfolio.trade_generation.validation import (
    ValidationWarning,
    ValidationSeverity,
    ConventionValidator,
)

__all__ = [
    "ValidationWarning",
    "ValidationSeverity",
    "ConventionValidator",
]
