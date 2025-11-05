"""SWIFT MT (Message Type) format implementation.

MT messages are the traditional SWIFT plain-text format used for
financial messaging. This module implements key MT message types
for securities settlement.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator

from .base import SwiftMessage, SwiftMessageType, SwiftParseError, SwiftValidationError


class MTMessage(SwiftMessage):
    """Base class for MT (plain text) SWIFT messages."""

    def _format_field(self, tag: str, value: str) -> str:
        """Format a single MT field.

        Args:
            tag: Field tag (e.g., "20", "23")
            value: Field value

        Returns:
            Formatted field line
        """
        return f":{tag}:{value}"

    def _format_optional_field(self, tag: str, value: Optional[str]) -> str:
        """Format an optional field.

        Args:
            tag: Field tag
            value: Field value (or None)

        Returns:
            Formatted field line or empty string
        """
        if value is None:
            return ""
        return self._format_field(tag, value)

    @staticmethod
    def _parse_field(line: str) -> tuple[str, str]:
        """Parse an MT field line.

        Args:
            line: Field line (e.g., ":20:REF123")

        Returns:
            Tuple of (tag, value)

        Raises:
            SwiftParseError: If line format is invalid
        """
        match = re.match(r":(\w+):(.*)", line)
        if not match:
            raise SwiftParseError(f"Invalid MT field format: {line}")
        return match.group(1), match.group(2)


class MT540(MTMessage):
    """MT540 - Receive Free (securities settlement instruction).

    Used to instruct the receipt of financial instruments free of payment.
    """

    message_type: str = Field(default=SwiftMessageType.MT540.value, frozen=True)

    # Mandatory fields
    sender_reference: str = Field(..., description="Sender's reference (:20:)")
    trade_date: date = Field(..., description="Trade date (:30:)")
    settlement_date: date = Field(..., description="Settlement date (:98A:)")

    # Securities details
    isin: str = Field(..., description="ISIN code (:35B:)")
    quantity: Decimal = Field(..., description="Quantity of securities")
    security_description: Optional[str] = Field(None, description="Security description")

    # Party details
    account_owner: str = Field(..., description="Account owner party (:95P:)")
    safekeeping_account: str = Field(..., description="Safekeeping account (:97A:)")
    place_of_settlement: str = Field(..., description="Place of settlement BIC (:53A:)")

    # Optional fields
    delivery_agent: Optional[str] = Field(None, description="Delivering agent BIC (:53B:)")
    narrative: Optional[str] = Field(None, description="Additional narrative (:70E:)")

    @field_validator("isin")
    @classmethod
    def validate_isin(cls, v: str) -> str:
        """Validate ISIN format."""
        if not re.match(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$", v):
            raise ValueError(f"Invalid ISIN format: {v}")
        return v

    def to_swift(self) -> str:
        """Convert to MT540 SWIFT format."""
        lines = [
            "{1:F01" + self.sender_bic + "0000000000}",
            "{2:O5401200" + datetime.now().strftime("%y%m%d%H%M") + self.receiver_bic + "0000000000}",
            "{4:",
            self._format_field("20", self.sender_reference),
            self._format_field("23", "RFRE"),  # Receive Free
            self._format_field("30", self.trade_date.strftime("%Y%m%d")),
            self._format_field("98A", ":SETT//" + self.settlement_date.strftime("%Y%m%d")),
            self._format_field("35B", "ISIN " + self.isin),
        ]

        if self.security_description:
            lines.append(self.security_description)

        lines.extend([
            self._format_field("36B", f":SETT//{float(self.quantity):.2f}"),
            self._format_field("95P", ":ACCW//" + self.account_owner),
            self._format_field("97A", ":SAFE//" + self.safekeeping_account),
            self._format_field("53A", "//" + self.place_of_settlement),
        ])

        if self.delivery_agent:
            lines.append(self._format_field("53B", "//" + self.delivery_agent))

        if self.narrative:
            lines.append(self._format_field("70E", ":SPRO//" + self.narrative))

        lines.append("-}")

        return "\n".join(lines)

    @classmethod
    def from_swift(cls, swift_text: str) -> MT540:
        """Parse MT540 from SWIFT text."""
        # This is a simplified parser - production implementation would be more robust
        fields: Dict[str, Any] = {}

        for line in swift_text.split("\n"):
            if not line.startswith(":"):
                continue

            tag, value = cls._parse_field(line)

            if tag == "20":
                fields["sender_reference"] = value
            elif tag == "30":
                fields["trade_date"] = datetime.strptime(value, "%Y%m%d").date()
            elif tag == "98A" and "SETT" in line:
                date_str = value.split("//")[1]
                fields["settlement_date"] = datetime.strptime(date_str, "%Y%m%d").date()
            elif tag == "35B":
                isin_match = re.search(r"ISIN ([A-Z0-9]{12})", value)
                if isin_match:
                    fields["isin"] = isin_match.group(1)
            elif tag == "36B":
                qty_match = re.search(r"([0-9.]+)", value)
                if qty_match:
                    fields["quantity"] = Decimal(qty_match.group(1))

        # Add required fields with dummy values for parsing
        fields.setdefault("sender_bic", "UNKNOWN")
        fields.setdefault("receiver_bic", "UNKNOWN")
        fields.setdefault("message_ref", "PARSE")
        fields.setdefault("account_owner", "UNKNOWN")
        fields.setdefault("safekeeping_account", "UNKNOWN")
        fields.setdefault("place_of_settlement", "UNKNOWN")

        return cls(**fields)

    def validate(self) -> bool:
        """Validate MT540 message."""
        if self.settlement_date < self.trade_date:
            raise SwiftValidationError("Settlement date cannot be before trade date")

        if self.quantity <= 0:
            raise SwiftValidationError("Quantity must be positive")

        return True


class MT542(MTMessage):
    """MT542 - Deliver Free (securities settlement instruction).

    Used to instruct the delivery of financial instruments free of payment.
    """

    message_type: str = Field(default=SwiftMessageType.MT542.value, frozen=True)

    # Mandatory fields
    sender_reference: str = Field(..., description="Sender's reference (:20:)")
    trade_date: date = Field(..., description="Trade date (:30:)")
    settlement_date: date = Field(..., description="Settlement date (:98A:)")

    # Securities details
    isin: str = Field(..., description="ISIN code (:35B:)")
    quantity: Decimal = Field(..., description="Quantity of securities")
    security_description: Optional[str] = Field(None, description="Security description")

    # Party details
    account_owner: str = Field(..., description="Account owner party (:95P:)")
    safekeeping_account: str = Field(..., description="Safekeeping account (:97A:)")
    place_of_settlement: str = Field(..., description="Place of settlement BIC (:53A:)")
    receiving_agent: str = Field(..., description="Receiving agent BIC (:57A:)")

    # Optional fields
    narrative: Optional[str] = Field(None, description="Additional narrative (:70E:)")

    @field_validator("isin")
    @classmethod
    def validate_isin(cls, v: str) -> str:
        """Validate ISIN format."""
        if not re.match(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$", v):
            raise ValueError(f"Invalid ISIN format: {v}")
        return v

    def to_swift(self) -> str:
        """Convert to MT542 SWIFT format."""
        lines = [
            "{1:F01" + self.sender_bic + "0000000000}",
            "{2:O5421200" + datetime.now().strftime("%y%m%d%H%M") + self.receiver_bic + "0000000000}",
            "{4:",
            self._format_field("20", self.sender_reference),
            self._format_field("23", "DFRE"),  # Deliver Free
            self._format_field("30", self.trade_date.strftime("%Y%m%d")),
            self._format_field("98A", ":SETT//" + self.settlement_date.strftime("%Y%m%d")),
            self._format_field("35B", "ISIN " + self.isin),
        ]

        if self.security_description:
            lines.append(self.security_description)

        lines.extend([
            self._format_field("36B", f":SETT//{float(self.quantity):.2f}"),
            self._format_field("95P", ":ACCW//" + self.account_owner),
            self._format_field("97A", ":SAFE//" + self.safekeeping_account),
            self._format_field("53A", "//" + self.place_of_settlement),
            self._format_field("57A", "//" + self.receiving_agent),
        ])

        if self.narrative:
            lines.append(self._format_field("70E", ":SPRO//" + self.narrative))

        lines.append("-}")

        return "\n".join(lines)

    @classmethod
    def from_swift(cls, swift_text: str) -> MT542:
        """Parse MT542 from SWIFT text."""
        # Simplified parser
        fields: Dict[str, Any] = {}

        for line in swift_text.split("\n"):
            if not line.startswith(":"):
                continue

            tag, value = cls._parse_field(line)

            if tag == "20":
                fields["sender_reference"] = value
            elif tag == "30":
                fields["trade_date"] = datetime.strptime(value, "%Y%m%d").date()
            elif tag == "98A" and "SETT" in line:
                date_str = value.split("//")[1]
                fields["settlement_date"] = datetime.strptime(date_str, "%Y%m%d").date()
            elif tag == "35B":
                isin_match = re.search(r"ISIN ([A-Z0-9]{12})", value)
                if isin_match:
                    fields["isin"] = isin_match.group(1)
            elif tag == "36B":
                qty_match = re.search(r"([0-9.]+)", value)
                if qty_match:
                    fields["quantity"] = Decimal(qty_match.group(1))

        # Add required fields
        fields.setdefault("sender_bic", "UNKNOWN")
        fields.setdefault("receiver_bic", "UNKNOWN")
        fields.setdefault("message_ref", "PARSE")
        fields.setdefault("account_owner", "UNKNOWN")
        fields.setdefault("safekeeping_account", "UNKNOWN")
        fields.setdefault("place_of_settlement", "UNKNOWN")
        fields.setdefault("receiving_agent", "UNKNOWN")

        return cls(**fields)

    def validate(self) -> bool:
        """Validate MT542 message."""
        if self.settlement_date < self.trade_date:
            raise SwiftValidationError("Settlement date cannot be before trade date")

        if self.quantity <= 0:
            raise SwiftValidationError("Quantity must be positive")

        return True


class MT543(MTMessage):
    """MT543 - Deliver Against Payment (DVP instruction).

    Used to instruct the delivery of financial instruments against payment.
    """

    message_type: str = Field(default=SwiftMessageType.MT543.value, frozen=True)

    # Mandatory fields
    sender_reference: str = Field(..., description="Sender's reference")
    trade_date: date = Field(..., description="Trade date")
    settlement_date: date = Field(..., description="Settlement date")

    # Securities details
    isin: str = Field(..., description="ISIN code")
    quantity: Decimal = Field(..., description="Quantity")

    # Settlement amount
    settlement_amount: Decimal = Field(..., description="Settlement amount")
    settlement_currency: str = Field(..., description="Settlement currency")

    # Party details
    account_owner: str = Field(..., description="Account owner")
    safekeeping_account: str = Field(..., description="Safekeeping account")
    place_of_settlement: str = Field(..., description="Place of settlement BIC")

    @field_validator("isin")
    @classmethod
    def validate_isin(cls, v: str) -> str:
        """Validate ISIN format."""
        if not re.match(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$", v):
            raise ValueError(f"Invalid ISIN format: {v}")
        return v

    def to_swift(self) -> str:
        """Convert to MT543 format."""
        lines = [
            "{1:F01" + self.sender_bic + "0000000000}",
            "{2:O5431200" + datetime.now().strftime("%y%m%d%H%M") + self.receiver_bic + "0000000000}",
            "{4:",
            self._format_field("20", self.sender_reference),
            self._format_field("23", "DVPA"),  # Deliver vs Payment
            self._format_field("30", self.trade_date.strftime("%Y%m%d")),
            self._format_field("98A", ":SETT//" + self.settlement_date.strftime("%Y%m%d")),
            self._format_field("35B", "ISIN " + self.isin),
            self._format_field("36B", f":SETT//{float(self.quantity):.2f}"),
            self._format_field("19A", f":SETT//{self.settlement_currency}{float(self.settlement_amount):.2f}"),
            self._format_field("95P", ":ACCW//" + self.account_owner),
            self._format_field("97A", ":SAFE//" + self.safekeeping_account),
            self._format_field("53A", "//" + self.place_of_settlement),
            "-}",
        ]
        return "\n".join(lines)

    @classmethod
    def from_swift(cls, swift_text: str) -> MT543:
        """Parse MT543 from text."""
        raise NotImplementedError("MT543 parsing not yet implemented")

    def validate(self) -> bool:
        """Validate MT543."""
        if self.settlement_date < self.trade_date:
            raise SwiftValidationError("Settlement date cannot be before trade date")
        if self.quantity <= 0 or self.settlement_amount <= 0:
            raise SwiftValidationError("Quantity and amount must be positive")
        return True


class MT544(MTMessage):
    """MT544 - Receive Free Confirmation.

    Confirmation message for receipt of financial instruments free of payment.
    """

    message_type: str = Field(default=SwiftMessageType.MT544.value, frozen=True)

    sender_reference: str = Field(..., description="Sender's reference")
    related_reference: str = Field(..., description="Reference to original instruction")
    settlement_date: date = Field(..., description="Settlement date")
    isin: str = Field(..., description="ISIN code")
    quantity: Decimal = Field(..., description="Settled quantity")
    status: str = Field(..., description="Settlement status")

    def to_swift(self) -> str:
        """Convert to MT544 format."""
        lines = [
            "{1:F01" + self.sender_bic + "0000000000}",
            "{2:O5441200" + datetime.now().strftime("%y%m%d%H%M") + self.receiver_bic + "0000000000}",
            "{4:",
            self._format_field("20", self.sender_reference),
            self._format_field("21", self.related_reference),
            self._format_field("98A", ":SETT//" + self.settlement_date.strftime("%Y%m%d")),
            self._format_field("35B", "ISIN " + self.isin),
            self._format_field("36B", f":SETT//{float(self.quantity):.2f}"),
            self._format_field("25D", ":MTCH//" + self.status),
            "-}",
        ]
        return "\n".join(lines)

    @classmethod
    def from_swift(cls, swift_text: str) -> MT544:
        """Parse MT544 from text."""
        raise NotImplementedError("MT544 parsing not yet implemented")

    def validate(self) -> bool:
        """Validate MT544."""
        if self.quantity <= 0:
            raise SwiftValidationError("Quantity must be positive")
        return True


__all__ = [
    "MTMessage",
    "MT540",
    "MT542",
    "MT543",
    "MT544",
]
