"""Euroclear message formats and data models."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator
import re

from ..swift.base import SwiftValidationError


class SettlementType(str, Enum):
    """Settlement type for securities transactions."""
    DVP = "DVP"  # Delivery versus Payment
    RVP = "RVP"  # Receive versus Payment
    FOP = "FOP"  # Free of Payment
    DFP = "DFP"  # Delivery Free of Payment
    RFP = "RFP"  # Receive Free of Payment


class SettlementStatus(str, Enum):
    """Euroclear settlement status."""
    PENDING = "pending"
    MATCHED = "matched"
    AFFIRMED = "affirmed"
    SETTLED = "settled"
    FAILING = "failing"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class SettlementMethod(str, Enum):
    """Settlement method."""
    EUROCLEAR = "EUROCLEAR"
    AGAINST_PAYMENT = "AGAINST_PAYMENT"
    FREE_DELIVERY = "FREE_DELIVERY"


class EuroclearSettlementInstruction(BaseModel):
    """Euroclear settlement instruction for securities transaction.

    Represents a settlement instruction submitted to Euroclear for
    securities delivery or receipt with or without payment.
    """

    # Instruction identification
    instruction_id: str = Field(..., description="Unique instruction ID")
    sender_reference: str = Field(..., description="Sender's reference")
    linked_reference: Optional[str] = Field(None, description="Linked instruction reference")

    # Settlement details
    settlement_type: SettlementType = Field(..., description="Type of settlement")
    settlement_date: date = Field(..., description="Intended settlement date")
    trade_date: date = Field(..., description="Trade execution date")

    # Securities details
    isin: str = Field(..., description="ISIN code")
    security_name: Optional[str] = Field(None, description="Security description")
    quantity: Decimal = Field(..., description="Quantity of securities", gt=0)

    # Party details
    delivering_party: str = Field(..., description="Delivering party account")
    receiving_party: str = Field(..., description="Receiving party account")
    participant_bic: str = Field(..., description="Euroclear participant BIC")

    # Payment details (if DVP/RVP)
    settlement_amount: Optional[Decimal] = Field(None, description="Settlement amount")
    settlement_currency: Optional[str] = Field(None, description="Settlement currency")
    payment_bank_bic: Optional[str] = Field(None, description="Payment bank BIC")

    # Optional fields
    place_of_settlement: str = Field(default="EUROCLEAR", description="Place of settlement")
    safekeeping_account: Optional[str] = Field(None, description="Safekeeping account")
    partial_settlement_allowed: bool = Field(default=False, description="Allow partial settlement")

    # Status tracking
    status: SettlementStatus = Field(default=SettlementStatus.PENDING)
    submission_timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("isin")
    @classmethod
    def validate_isin(cls, v: str) -> str:
        """Validate ISIN format."""
        if not re.match(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$", v):
            raise ValueError(f"Invalid ISIN format: {v}")
        return v

    @field_validator("settlement_currency")
    @classmethod
    def validate_currency(cls, v: Optional[str]) -> Optional[str]:
        """Validate currency code."""
        if v and len(v) != 3:
            raise ValueError(f"Invalid currency code: {v}")
        return v.upper() if v else None

    def validate_dvp_fields(self) -> bool:
        """Validate that DVP/RVP instructions have payment details."""
        if self.settlement_type in (SettlementType.DVP, SettlementType.RVP):
            if not self.settlement_amount or not self.settlement_currency:
                raise ValueError(f"{self.settlement_type} requires settlement amount and currency")
        return True

    def to_mt540(self) -> str:
        """Convert to SWIFT MT540 format for Euroclear.

        Returns:
            MT540 formatted message
        """
        from ..swift.mt import MT540, MT543, MT544

        # For receive free instructions
        if self.settlement_type in (SettlementType.RFP, SettlementType.FOP):
            mt540 = MT540(
                sender_bic=self.participant_bic,
                receiver_bic="MGTCBEBEECL",  # Euroclear Belgium
                message_ref=self.instruction_id,
                sender_reference=self.sender_reference,
                trade_date=self.trade_date,
                settlement_date=self.settlement_date,
                isin=self.isin,
                quantity=self.quantity,
                security_description=self.security_name,
                account_owner=self.receiving_party,
                safekeeping_account=self.safekeeping_account or "DEFAULT",
                place_of_settlement=self.place_of_settlement,
                delivery_agent=self.delivering_party,
            )
            return mt540.to_swift()

        if self.settlement_type == SettlementType.DVP:
            self.validate_dvp_fields()

            if self.settlement_amount is None or self.settlement_currency is None:
                raise SwiftValidationError("DVP instruction missing settlement amount or currency")

            mt543 = MT543(
                sender_bic=self.participant_bic,
                receiver_bic="MGTCBEBEECL",
                message_ref=self.instruction_id,
                sender_reference=self.sender_reference,
                trade_date=self.trade_date,
                settlement_date=self.settlement_date,
                isin=self.isin,
                quantity=self.quantity,
                settlement_amount=self.settlement_amount,
                settlement_currency=self.settlement_currency,
                account_owner=self.delivering_party,
                safekeeping_account=self.safekeeping_account or "DEFAULT",
                place_of_settlement=self.place_of_settlement,
            )
            return mt543.to_swift()

        if self.settlement_type == SettlementType.RVP:
            self.validate_dvp_fields()

            if self.settlement_amount is None or self.settlement_currency is None:
                raise SwiftValidationError("RVP instruction missing settlement amount or currency")

            mt544 = MT544(
                sender_bic=self.participant_bic,
                receiver_bic="MGTCBEBEECL",
                message_ref=self.instruction_id,
                sender_reference=self.sender_reference,
                related_reference=self.linked_reference or self.instruction_id,
                settlement_date=self.settlement_date,
                isin=self.isin,
                quantity=self.quantity,
                status=self.status.value.upper(),
                settlement_amount=self.settlement_amount,
                settlement_currency=self.settlement_currency,
            )
            return mt544.to_swift()

        raise NotImplementedError(f"MT message generation for {self.settlement_type} not yet implemented")

    def to_euroclear_message(self) -> str:
        """Convert to Euroclear proprietary message format.

        Returns:
            Euroclear-formatted message string
        """
        lines = [
            f"MSG_TYPE=SETTLEMENT_INSTRUCTION",
            f"INSTRUCTION_ID={self.instruction_id}",
            f"SENDER_REF={self.sender_reference}",
            f"SETTLEMENT_TYPE={self.settlement_type.value}",
            f"SETTLEMENT_DATE={self.settlement_date.strftime('%Y%m%d')}",
            f"TRADE_DATE={self.trade_date.strftime('%Y%m%d')}",
            f"ISIN={self.isin}",
            f"QUANTITY={float(self.quantity):.2f}",
            f"DELIVERING_PARTY={self.delivering_party}",
            f"RECEIVING_PARTY={self.receiving_party}",
            f"PARTICIPANT_BIC={self.participant_bic}",
            f"PLACE_OF_SETTLEMENT={self.place_of_settlement}",
            f"PARTIAL_ALLOWED={self.partial_settlement_allowed}",
            f"STATUS={self.status.value}",
        ]

        if self.settlement_amount and self.settlement_currency:
            lines.append(f"SETTLEMENT_AMOUNT={float(self.settlement_amount):.2f}")
            lines.append(f"SETTLEMENT_CURRENCY={self.settlement_currency}")

        if self.payment_bank_bic:
            lines.append(f"PAYMENT_BANK={self.payment_bank_bic}")

        return "\n".join(lines)


class EuroclearConfirmation(BaseModel):
    """Euroclear settlement confirmation message."""

    confirmation_id: str = Field(..., description="Confirmation ID")
    instruction_id: str = Field(..., description="Related instruction ID")
    sender_reference: str = Field(..., description="Sender's reference")

    # Settlement details
    settlement_date: date = Field(..., description="Actual settlement date")
    actual_settlement_time: Optional[datetime] = Field(None, description="Actual settlement timestamp")

    # Securities details
    isin: str = Field(..., description="ISIN code")
    quantity_settled: Decimal = Field(..., description="Quantity actually settled")
    quantity_instructed: Decimal = Field(..., description="Quantity originally instructed")

    # Payment details (if applicable)
    settlement_amount: Optional[Decimal] = Field(None, description="Settlement amount")
    settlement_currency: Optional[str] = Field(None, description="Settlement currency")

    # Status
    status: SettlementStatus = Field(..., description="Final settlement status")
    confirmation_timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Optional details
    euroclear_reference: Optional[str] = Field(None, description="Euroclear system reference")
    partial_settlement: bool = Field(default=False, description="Was this a partial settlement")
    rejection_reason: Optional[str] = Field(None, description="Rejection reason if applicable")
    rejection_code: Optional[str] = Field(None, description="Rejection code")

    def is_successful(self) -> bool:
        """Check if settlement was successful."""
        return self.status == SettlementStatus.SETTLED

    def is_partial(self) -> bool:
        """Check if settlement was partial."""
        return self.quantity_settled < self.quantity_instructed

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "confirmation_id": self.confirmation_id,
            "instruction_id": self.instruction_id,
            "sender_reference": self.sender_reference,
            "settlement_date": self.settlement_date.isoformat(),
            "actual_settlement_time": self.actual_settlement_time.isoformat() if self.actual_settlement_time else None,
            "isin": self.isin,
            "quantity_settled": float(self.quantity_settled),
            "quantity_instructed": float(self.quantity_instructed),
            "settlement_amount": float(self.settlement_amount) if self.settlement_amount else None,
            "settlement_currency": self.settlement_currency,
            "status": self.status.value,
            "is_successful": self.is_successful(),
            "is_partial": self.is_partial(),
            "rejection_reason": self.rejection_reason,
            "rejection_code": self.rejection_code,
        }


class EuroclearStatus(BaseModel):
    """Euroclear settlement status query response."""

    instruction_id: str = Field(..., description="Instruction ID")
    sender_reference: str = Field(..., description="Sender's reference")
    status: SettlementStatus = Field(..., description="Current status")
    last_updated: datetime = Field(..., description="Last status update time")

    # Settlement progress
    matched: bool = Field(default=False, description="Instruction matched")
    affirmed: bool = Field(default=False, description="Instruction affirmed")
    securities_delivered: bool = Field(default=False, description="Securities delivered")
    payment_received: bool = Field(default=False, description="Payment received")

    # Optional details
    status_message: Optional[str] = Field(None, description="Status message")
    failing_reason: Optional[str] = Field(None, description="Reason if failing")
    expected_settlement_date: Optional[date] = Field(None, description="Expected settlement date")

    def is_final(self) -> bool:
        """Check if status is final."""
        return self.status in (
            SettlementStatus.SETTLED,
            SettlementStatus.CANCELLED,
            SettlementStatus.REJECTED,
        )

    def is_failing(self) -> bool:
        """Check if settlement is currently failing."""
        return self.status == SettlementStatus.FAILING


__all__ = [
    "EuroclearSettlementInstruction",
    "EuroclearConfirmation",
    "EuroclearStatus",
    "SettlementType",
    "SettlementStatus",
    "SettlementMethod",
]
