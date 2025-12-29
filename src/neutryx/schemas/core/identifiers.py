"""Core identifier and reference types.

This module provides fundamental types used across all domains:
- Party: Trading counterparty identification
- Money: Monetary amount with currency
- Identifier validators: LEI, ISIN, CUSIP, UTI
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Optional

from pydantic import Field, field_validator

from neutryx.schemas.core.base import ImmutableSchema
from neutryx.schemas.core.enums import CurrencyCode, EntityType


class Party(ImmutableSchema):
    """Trading party identification.

    Consolidates party definitions from clearing and FpML domains.
    This is the canonical party representation for all Neutryx modules.

    Attributes:
        party_id: Unique internal identifier
        name: Legal name of the party
        short_name: Short/display name
        lei: Legal Entity Identifier (ISO 17442)
        bic: Bank Identifier Code (SWIFT)
        member_id: CCP member identifier
        entity_type: Classification of the entity
    """

    __schema_version__ = "1.0.0"
    __schema_domain__ = "core"

    party_id: str = Field(
        ...,
        min_length=1,
        description="Unique party identifier",
    )
    name: str = Field(
        ...,
        min_length=1,
        description="Party legal name",
    )
    short_name: Optional[str] = Field(
        default=None,
        description="Short display name",
    )
    lei: Optional[str] = Field(
        default=None,
        min_length=20,
        max_length=20,
        description="Legal Entity Identifier (ISO 17442)",
    )
    bic: Optional[str] = Field(
        default=None,
        min_length=8,
        max_length=11,
        description="Bank Identifier Code (SWIFT)",
    )
    member_id: Optional[str] = Field(
        default=None,
        description="CCP member ID",
    )
    entity_type: Optional[EntityType] = Field(
        default=None,
        description="Entity classification",
    )

    @field_validator("lei")
    @classmethod
    def validate_lei(cls, v: Optional[str]) -> Optional[str]:
        """Validate LEI format (20 alphanumeric characters)."""
        if v is None:
            return v
        if not re.match(r"^[A-Z0-9]{20}$", v):
            raise ValueError("LEI must be exactly 20 alphanumeric characters")
        return v

    @field_validator("bic")
    @classmethod
    def validate_bic(cls, v: Optional[str]) -> Optional[str]:
        """Validate BIC/SWIFT format (8 or 11 characters)."""
        if v is None:
            return v
        if not re.match(r"^[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?$", v):
            raise ValueError("Invalid BIC/SWIFT code format")
        return v


class Money(ImmutableSchema):
    """Monetary amount with currency.

    Immutable value object representing a monetary amount in a specific currency.

    Attributes:
        currency: ISO 4217 currency code
        amount: Monetary amount as Decimal for precision
    """

    __schema_version__ = "1.0.0"
    __schema_domain__ = "core"

    currency: CurrencyCode = Field(
        ...,
        description="Currency code (ISO 4217)",
    )
    amount: Decimal = Field(
        ...,
        description="Monetary amount",
    )

    def __str__(self) -> str:
        """Return formatted string like 'USD 1,000.00'."""
        return f"{self.currency.value} {self.amount:,.2f}"

    def __add__(self, other: Money) -> Money:
        """Add two Money objects (must be same currency)."""
        if self.currency != other.currency:
            raise ValueError(
                f"Cannot add {self.currency.value} and {other.currency.value}"
            )
        return Money(currency=self.currency, amount=self.amount + other.amount)

    def __sub__(self, other: Money) -> Money:
        """Subtract two Money objects (must be same currency)."""
        if self.currency != other.currency:
            raise ValueError(
                f"Cannot subtract {other.currency.value} from {self.currency.value}"
            )
        return Money(currency=self.currency, amount=self.amount - other.amount)

    def __mul__(self, factor: float | Decimal) -> Money:
        """Multiply amount by a factor."""
        return Money(
            currency=self.currency, amount=self.amount * Decimal(str(factor))
        )

    def __neg__(self) -> Money:
        """Negate the amount."""
        return Money(currency=self.currency, amount=-self.amount)


class PartyReference(ImmutableSchema):
    """Reference to a Party by ID.

    Used in documents where parties are defined separately and referenced.

    Attributes:
        href: Reference to party_id
    """

    __schema_version__ = "1.0.0"
    __schema_domain__ = "core"

    href: str = Field(
        ...,
        description="Reference to party_id",
    )


def validate_isin(value: str) -> bool:
    """Validate ISIN (International Securities Identification Number).

    ISIN format: 2-letter country code + 9-char national ID + 1 check digit

    Args:
        value: The ISIN to validate

    Returns:
        True if valid, False otherwise
    """
    if not re.match(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$", value):
        return False

    # Luhn checksum validation
    chars = value[:-1]
    digits = ""
    for char in chars:
        if char.isdigit():
            digits += char
        else:
            digits += str(ord(char) - 55)  # A=10, B=11, etc.

    total = 0
    for i, digit in enumerate(reversed(digits)):
        d = int(digit)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d

    check_digit = (10 - (total % 10)) % 10
    return check_digit == int(value[-1])


def validate_cusip(value: str) -> bool:
    """Validate CUSIP (Committee on Uniform Securities Identification Procedures).

    CUSIP format: 9 characters (6 issuer + 2 issue + 1 check digit)

    Args:
        value: The CUSIP to validate

    Returns:
        True if valid, False otherwise
    """
    if not re.match(r"^[A-Z0-9]{9}$", value):
        return False

    # Modulus 10 checksum validation
    total = 0
    for i, char in enumerate(value[:-1]):
        if char.isdigit():
            v = int(char)
        else:
            v = ord(char) - 55  # A=10, B=11, etc.

        if i % 2 == 1:
            v *= 2

        total += v // 10 + v % 10

    check_digit = (10 - (total % 10)) % 10
    return check_digit == int(value[-1])


def validate_uti(value: str) -> bool:
    """Validate UTI (Unique Transaction Identifier).

    UTI format: 52 alphanumeric characters (LEI + unique suffix)

    Args:
        value: The UTI to validate

    Returns:
        True if valid, False otherwise
    """
    return bool(re.match(r"^[A-Z0-9]{1,52}$", value))
