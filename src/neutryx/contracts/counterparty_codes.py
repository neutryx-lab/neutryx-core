"""Counterparty code generation and management system.

Provides systematic generation and validation of counterparty codes with:
- Pattern-based code generation
- LEI (Legal Entity Identifier) integration
- Code uniqueness validation
- Hierarchical code structures
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Pattern, Set

from pydantic import BaseModel, Field


class CounterpartyType(Enum):
    """Counterparty classification types."""

    BANK = "bank"
    CORPORATE = "corporate"
    HEDGE_FUND = "hedge_fund"
    INSURANCE = "insurance"
    PENSION_FUND = "pension_fund"
    GOVERNMENT = "government"
    CENTRAL_BANK = "central_bank"
    MULTILATERAL = "multilateral"
    OTHER = "other"


class CounterpartyCodeConfig(BaseModel):
    """Configuration for counterparty code generation."""

    prefix: str = Field(default="CP", description="Code prefix")
    separator: str = Field(default="-", description="Separator between components")
    sequence_width: int = Field(default=4, ge=1, description="Width of sequence number")
    use_lei_prefix: bool = Field(default=False, description="Include LEI prefix in code")
    use_type_code: bool = Field(default=False, description="Include counterparty type code")
    validate_uniqueness: bool = Field(default=True, description="Track code uniqueness")


@dataclass
class CounterpartyCodeMapping:
    """Mapping of counterparty code to details."""

    code: str
    counterparty_id: str
    lei: Optional[str] = None
    name: Optional[str] = None
    counterparty_type: Optional[CounterpartyType] = None
    metadata: Dict[str, str] = field(default_factory=dict)


class CounterpartyCodeGenerator:
    """Generates and validates counterparty codes.

    Example:
        >>> generator = CounterpartyCodeGenerator()
        >>> code = generator.generate("CPTY-001")  # CP-0001
        >>> code = generator.generate("CPTY-002", lei="549300ABCDEF123456")  # CP-0002
        >>> generator.is_valid("CP-0001")  # True
    """

    def __init__(self, config: Optional[CounterpartyCodeConfig] = None):
        """Initialize counterparty code generator.

        Args:
            config: Code generation configuration
        """
        self.config = config or CounterpartyCodeConfig()
        self._counter = 0
        self._generated_codes: Set[str] = set()
        self._code_mappings: Dict[str, CounterpartyCodeMapping] = {}

    def generate(
        self,
        counterparty_id: str,
        lei: Optional[str] = None,
        counterparty_type: Optional[CounterpartyType] = None,
        name: Optional[str] = None,
    ) -> str:
        """Generate a counterparty code.

        Args:
            counterparty_id: Internal counterparty ID
            lei: Legal Entity Identifier (optional)
            counterparty_type: Counterparty type (optional)
            name: Counterparty name (optional)

        Returns:
            Generated counterparty code

        Raises:
            ValueError: If code generation fails or code already exists
        """
        # Build code components
        components = [self.config.prefix]

        # Add LEI prefix if configured and available
        if self.config.use_lei_prefix and lei:
            lei_prefix = lei[:6]  # First 6 chars of LEI
            components.append(lei_prefix)

        # Add type code if configured
        if self.config.use_type_code and counterparty_type:
            type_code = self._get_type_code(counterparty_type)
            components.append(type_code)

        # Add sequence number
        self._counter += 1
        seq = str(self._counter).zfill(self.config.sequence_width)
        components.append(seq)

        # Build final code
        code = self.config.separator.join(components)

        # Validate uniqueness
        if self.config.validate_uniqueness and code in self._generated_codes:
            raise ValueError(f"Counterparty code already exists: {code}")

        # Store mapping
        mapping = CounterpartyCodeMapping(
            code=code,
            counterparty_id=counterparty_id,
            lei=lei,
            name=name,
            counterparty_type=counterparty_type,
        )
        self._code_mappings[code] = mapping
        self._generated_codes.add(code)

        return code

    def _get_type_code(self, counterparty_type: CounterpartyType) -> str:
        """Get short code for counterparty type."""
        type_codes = {
            CounterpartyType.BANK: "BNK",
            CounterpartyType.CORPORATE: "CRP",
            CounterpartyType.HEDGE_FUND: "HF",
            CounterpartyType.INSURANCE: "INS",
            CounterpartyType.PENSION_FUND: "PF",
            CounterpartyType.GOVERNMENT: "GOV",
            CounterpartyType.CENTRAL_BANK: "CB",
            CounterpartyType.MULTILATERAL: "MLT",
            CounterpartyType.OTHER: "OTH",
        }
        return type_codes.get(counterparty_type, "OTH")

    def generate_from_lei(self, lei: str, counterparty_id: str, name: Optional[str] = None) -> str:
        """Generate code primarily from LEI.

        Args:
            lei: Legal Entity Identifier
            counterparty_id: Internal counterparty ID
            name: Counterparty name

        Returns:
            Generated code based on LEI
        """
        if not self._is_valid_lei(lei):
            raise ValueError(f"Invalid LEI format: {lei}")

        return self.generate(counterparty_id, lei=lei, name=name)

    def _is_valid_lei(self, lei: str) -> bool:
        """Validate LEI format (20 alphanumeric characters)."""
        if not lei or len(lei) != 20:
            return False
        return lei.isalnum()

    def is_valid(self, code: str) -> bool:
        """Check if a code is valid.

        Args:
            code: Code to validate

        Returns:
            True if code is valid
        """
        if not code:
            return False

        # Check if code was generated
        if self.config.validate_uniqueness and code in self._generated_codes:
            return True

        # Validate format
        pattern = self._get_validation_pattern()
        return bool(pattern.match(code))

    def _get_validation_pattern(self) -> Pattern:
        """Get regex pattern for code validation."""
        sep = re.escape(self.config.separator)
        prefix = re.escape(self.config.prefix)

        # Basic pattern: PREFIX-NNNN
        if not self.config.use_lei_prefix and not self.config.use_type_code:
            return re.compile(rf"^{prefix}{sep}\d{{{self.config.sequence_width}}}$")

        # With LEI: PREFIX-LEIPFX-NNNN
        if self.config.use_lei_prefix and not self.config.use_type_code:
            return re.compile(rf"^{prefix}{sep}[A-Z0-9]{{6}}{sep}\d{{{self.config.sequence_width}}}$")

        # With type: PREFIX-TYP-NNNN
        if not self.config.use_lei_prefix and self.config.use_type_code:
            return re.compile(rf"^{prefix}{sep}[A-Z]{{2,3}}{sep}\d{{{self.config.sequence_width}}}$")

        # With both: PREFIX-LEIPFX-TYP-NNNN
        return re.compile(rf"^{prefix}{sep}[A-Z0-9]{{6}}{sep}[A-Z]{{2,3}}{sep}\d{{{self.config.sequence_width}}}$")

    def get_mapping(self, code: str) -> Optional[CounterpartyCodeMapping]:
        """Get counterparty details for a code.

        Args:
            code: Counterparty code

        Returns:
            Code mapping or None if not found
        """
        return self._code_mappings.get(code)

    def lookup_by_counterparty_id(self, counterparty_id: str) -> Optional[str]:
        """Find code by counterparty ID.

        Args:
            counterparty_id: Internal counterparty ID

        Returns:
            Counterparty code or None
        """
        for code, mapping in self._code_mappings.items():
            if mapping.counterparty_id == counterparty_id:
                return code
        return None

    def lookup_by_lei(self, lei: str) -> list[str]:
        """Find all codes for a given LEI.

        Args:
            lei: Legal Entity Identifier

        Returns:
            List of counterparty codes
        """
        return [code for code, mapping in self._code_mappings.items() if mapping.lei == lei]

    def register_code(
        self,
        code: str,
        counterparty_id: str,
        lei: Optional[str] = None,
        name: Optional[str] = None,
        counterparty_type: Optional[CounterpartyType] = None,
    ) -> None:
        """Register an existing code.

        Useful for importing codes from external systems.

        Args:
            code: Counterparty code
            counterparty_id: Internal counterparty ID
            lei: Legal Entity Identifier
            name: Counterparty name
            counterparty_type: Counterparty type

        Raises:
            ValueError: If code already registered
        """
        if self.config.validate_uniqueness and code in self._generated_codes:
            raise ValueError(f"Code already registered: {code}")

        mapping = CounterpartyCodeMapping(
            code=code,
            counterparty_id=counterparty_id,
            lei=lei,
            name=name,
            counterparty_type=counterparty_type,
        )
        self._code_mappings[code] = mapping
        self._generated_codes.add(code)

    def get_all_codes(self) -> list[str]:
        """Get all generated codes.

        Returns:
            List of all counterparty codes
        """
        return list(self._generated_codes)

    def get_counter(self) -> int:
        """Get current counter value.

        Returns:
            Current sequence counter
        """
        return self._counter


def create_simple_counterparty_code_generator() -> CounterpartyCodeGenerator:
    """Create a generator with simple sequential codes (CP-0001, CP-0002, ...).

    Returns:
        Configured CounterpartyCodeGenerator
    """
    config = CounterpartyCodeConfig(
        prefix="CP",
        separator="-",
        sequence_width=4,
        use_lei_prefix=False,
        use_type_code=False,
    )
    return CounterpartyCodeGenerator(config)


def create_lei_based_counterparty_code_generator() -> CounterpartyCodeGenerator:
    """Create a generator with LEI-based codes (CP-549300-0001, ...).

    Returns:
        Configured CounterpartyCodeGenerator
    """
    config = CounterpartyCodeConfig(
        prefix="CP",
        separator="-",
        sequence_width=4,
        use_lei_prefix=True,
        use_type_code=False,
    )
    return CounterpartyCodeGenerator(config)


def create_typed_counterparty_code_generator() -> CounterpartyCodeGenerator:
    """Create a generator with type-based codes (CP-BNK-0001, CP-CRP-0002, ...).

    Returns:
        Configured CounterpartyCodeGenerator
    """
    config = CounterpartyCodeConfig(
        prefix="CP",
        separator="-",
        sequence_width=4,
        use_lei_prefix=False,
        use_type_code=True,
    )
    return CounterpartyCodeGenerator(config)


__all__ = [
    "CounterpartyType",
    "CounterpartyCodeConfig",
    "CounterpartyCodeMapping",
    "CounterpartyCodeGenerator",
    "create_simple_counterparty_code_generator",
    "create_lei_based_counterparty_code_generator",
    "create_typed_counterparty_code_generator",
]
