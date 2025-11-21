"""Settlement instruction generation and processing.

This module handles settlement workflows:
- Settlement instruction generation (SSI)
- Payment calculations and netting
- Standing Settlement Instructions (SSI) management
- Integration with CLS, Euroclear, Clearstream
- Settlement status tracking and exceptions
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict


class SettlementMethod(Enum):
    """Settlement method."""

    DVP = "dvp"  # Delivery versus Payment
    FOP = "fop"  # Free of Payment
    RVP = "rvp"  # Receive versus Payment
    CASH = "cash"  # Cash settlement only


class SettlementStatus(Enum):
    """Settlement instruction status."""

    PENDING = "pending"  # Awaiting execution
    MATCHED = "matched"  # Matched with counterparty
    AFFIRMED = "affirmed"  # Affirmed by both parties
    SETTLED = "settled"  # Successfully settled
    FAILED = "failed"  # Settlement failed
    CANCELLED = "cancelled"  # Instruction cancelled
    PENDING_CANCEL = "pending_cancel"  # Cancellation pending


class FailureReason(Enum):
    """Settlement failure reason."""

    INSUFFICIENT_FUNDS = "insufficient_funds"
    INSUFFICIENT_SECURITIES = "insufficient_securities"
    INCORRECT_DETAILS = "incorrect_details"
    SYSTEM_ERROR = "system_error"
    COUNTERPARTY_FAIL = "counterparty_fail"
    CUT_OFF_MISSED = "cut_off_missed"
    OTHER = "other"


class Currency(str):
    """ISO 4217 currency code."""

    pass


class SettlementAccount(BaseModel):
    """Settlement account details."""

    account_id: str = Field(..., description="Account identifier")
    account_name: str = Field(..., description="Account name")
    account_number: str = Field(..., description="Account number")
    currency: str = Field(..., description="Currency code (ISO 4217)")
    bank_name: str = Field(..., description="Bank name")
    swift_code: str = Field(..., description="SWIFT/BIC code")
    iban: Optional[str] = Field(None, description="IBAN (if applicable)")
    custodian: Optional[str] = Field(None, description="Custodian name")
    account_type: str = Field(default="cash", description="Account type (cash, securities)")


class StandingSettlementInstruction(BaseModel):
    """Standing Settlement Instructions (SSI) for a counterparty."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ssi_id: str = Field(default_factory=lambda: f"SSI-{uuid4().hex[:12].upper()}")
    counterparty_id: str = Field(..., description="Counterparty identifier")
    currency: str = Field(..., description="Currency code")
    product_type: str = Field(..., description="Product type")
    settlement_method: SettlementMethod = SettlementMethod.DVP
    beneficiary_account: SettlementAccount = Field(..., description="Beneficiary account")
    intermediary_account: Optional[SettlementAccount] = Field(
        None, description="Intermediary bank account"
    )
    valid_from: date = Field(default_factory=date.today)
    valid_until: Optional[date] = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @property
    def is_valid(self) -> bool:
        """Check if SSI is currently valid."""
        today = date.today()
        if not self.is_active:
            return False
        if today < self.valid_from:
            return False
        if self.valid_until and today > self.valid_until:
            return False
        return True


class SettlementInstruction(BaseModel):
    """Settlement instruction for a trade."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    instruction_id: str = Field(default_factory=lambda: f"SI-{uuid4().hex[:12].upper()}")
    trade_id: str = Field(..., description="Associated trade ID")
    settlement_date: date = Field(..., description="Settlement date")
    settlement_method: SettlementMethod = SettlementMethod.DVP
    currency: str = Field(..., description="Settlement currency")
    amount: Decimal = Field(..., description="Settlement amount")
    payer_id: str = Field(..., description="Payer party ID")
    receiver_id: str = Field(..., description="Receiver party ID")
    payer_account: SettlementAccount = Field(..., description="Payer account details")
    receiver_account: SettlementAccount = Field(..., description="Receiver account details")
    security_details: Optional[Dict[str, Any]] = Field(
        None, description="Security details for DVP/RVP"
    )
    status: SettlementStatus = SettlementStatus.PENDING
    generated_at: datetime = Field(default_factory=datetime.now)
    matched_at: Optional[datetime] = None
    settled_at: Optional[datetime] = None
    failure_reason: Optional[FailureReason] = None
    failure_details: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    reference: Optional[str] = Field(None, description="External reference")
    metadata: Dict[str, str] = Field(default_factory=dict)

    @property
    def is_settled(self) -> bool:
        """Check if instruction is settled."""
        return self.status == SettlementStatus.SETTLED

    @property
    def can_retry(self) -> bool:
        """Check if instruction can be retried."""
        return self.status == SettlementStatus.FAILED and self.retry_count < self.max_retries


@dataclass
class PaymentCalculation:
    """Payment calculation result."""

    trade_id: str
    settlement_date: date
    currency: str
    gross_amount: Decimal
    fees: Decimal = Decimal("0")
    taxes: Decimal = Decimal("0")
    net_amount: Decimal = field(init=False)
    breakdown: Dict[str, Decimal] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate net amount."""
        self.net_amount = self.gross_amount - self.fees - self.taxes


@dataclass
class NettingResult:
    """Result of payment netting."""

    settlement_date: date
    currency: str
    gross_payments: List[PaymentCalculation]
    net_amount: Decimal
    payer_id: str
    receiver_id: str
    netted_trade_ids: List[str]


class SettlementManager:
    """Manages settlement instructions and workflows.

    Example:
        >>> manager = SettlementManager()
        >>>
        >>> # Register SSI
        >>> ssi = StandingSettlementInstruction(
        ...     counterparty_id="COUNTERPARTY-001",
        ...     currency="USD",
        ...     product_type="interest_rate_swap",
        ...     beneficiary_account=SettlementAccount(
        ...         account_id="ACC-001",
        ...         account_name="ABC Bank",
        ...         account_number="123456789",
        ...         currency="USD",
        ...         bank_name="ABC Bank",
        ...         swift_code="ABCBUS33"
        ...     )
        ... )
        >>> manager.register_ssi(ssi)
        >>>
        >>> # Generate settlement instruction
        >>> payment = PaymentCalculation(
        ...     trade_id="TRD-001",
        ...     settlement_date=date.today() + timedelta(days=2),
        ...     currency="USD",
        ...     gross_amount=Decimal("1000000.00"),
        ...     fees=Decimal("500.00")
        ... )
        >>> instruction = manager.generate_instruction(
        ...     payment=payment,
        ...     payer_id="PARTY-001",
        ...     receiver_id="COUNTERPARTY-001"
        ... )
        >>>
        >>> # Process settlement
        >>> manager.process_settlement(instruction.instruction_id)
    """

    def __init__(self):
        """Initialize settlement manager."""
        self._ssis: Dict[str, List[StandingSettlementInstruction]] = {}  # counterparty_id -> SSIs
        self._instructions: Dict[str, SettlementInstruction] = {}  # instruction_id -> instruction
        self._payments: Dict[str, List[PaymentCalculation]] = {}  # settlement_date -> payments

    def register_ssi(self, ssi: StandingSettlementInstruction) -> None:
        """Register standing settlement instructions.

        Args:
            ssi: SSI to register
        """
        if ssi.counterparty_id not in self._ssis:
            self._ssis[ssi.counterparty_id] = []
        self._ssis[ssi.counterparty_id].append(ssi)

    def get_ssi(
        self,
        counterparty_id: str,
        currency: str,
        product_type: str,
        settlement_date: date,
    ) -> Optional[StandingSettlementInstruction]:
        """Get applicable SSI for counterparty and currency.

        Args:
            counterparty_id: Counterparty identifier
            currency: Currency code
            product_type: Product type
            settlement_date: Settlement date

        Returns:
            Matching SSI or None
        """
        ssis = self._ssis.get(counterparty_id, [])

        # Find matching SSI
        for ssi in ssis:
            if (
                ssi.currency == currency
                and ssi.product_type == product_type
                and ssi.is_valid
                and ssi.valid_from <= settlement_date
                and (ssi.valid_until is None or ssi.valid_until >= settlement_date)
            ):
                return ssi

        return None

    def generate_instruction(
        self,
        payment: PaymentCalculation,
        payer_id: str,
        receiver_id: str,
        payer_account: Optional[SettlementAccount] = None,
        receiver_account: Optional[SettlementAccount] = None,
        settlement_method: SettlementMethod = SettlementMethod.CASH,
        security_details: Optional[Dict[str, Any]] = None,
    ) -> SettlementInstruction:
        """Generate settlement instruction.

        Args:
            payment: Payment calculation
            payer_id: Payer party ID
            receiver_id: Receiver party ID
            payer_account: Payer account (if not using SSI)
            receiver_account: Receiver account (if not using SSI)
            settlement_method: Settlement method
            security_details: Security details for DVP/RVP

        Returns:
            Generated settlement instruction

        Raises:
            ValueError: If accounts not provided and SSI not found
        """
        # Try to get SSI if accounts not provided
        if receiver_account is None:
            ssi = self.get_ssi(
                receiver_id, payment.currency, "default", payment.settlement_date
            )
            if ssi:
                receiver_account = ssi.beneficiary_account
            else:
                raise ValueError(f"No SSI found for {receiver_id} in {payment.currency}")

        if payer_account is None:
            ssi = self.get_ssi(payer_id, payment.currency, "default", payment.settlement_date)
            if ssi:
                payer_account = ssi.beneficiary_account
            else:
                raise ValueError(f"No SSI found for {payer_id} in {payment.currency}")

        instruction = SettlementInstruction(
            trade_id=payment.trade_id,
            settlement_date=payment.settlement_date,
            settlement_method=settlement_method,
            currency=payment.currency,
            amount=payment.net_amount,
            payer_id=payer_id,
            receiver_id=receiver_id,
            payer_account=payer_account,
            receiver_account=receiver_account,
            security_details=security_details,
        )

        self._instructions[instruction.instruction_id] = instruction

        # Track payment
        date_key = str(payment.settlement_date)
        if date_key not in self._payments:
            self._payments[date_key] = []
        self._payments[date_key].append(payment)

        return instruction

    def net_payments(
        self,
        settlement_date: date,
        currency: str,
        party_id: str,
        counterparty_id: str,
    ) -> Optional[NettingResult]:
        """Net payments between two parties for a settlement date.

        Args:
            settlement_date: Settlement date
            currency: Currency for netting
            party_id: First party ID
            counterparty_id: Second party ID

        Returns:
            Netting result or None if no payments to net
        """
        date_key = str(settlement_date)
        payments = self._payments.get(date_key, [])

        # Filter payments for this currency and parties
        relevant_payments = [
            p
            for p in payments
            if p.currency == currency
            and (
                (p.breakdown.get("payer") == party_id and p.breakdown.get("receiver") == counterparty_id)
                or (p.breakdown.get("payer") == counterparty_id and p.breakdown.get("receiver") == party_id)
            )
        ]

        if not relevant_payments:
            return None

        # Calculate net amount
        party_pays = sum(
            p.net_amount
            for p in relevant_payments
            if p.breakdown.get("payer") == party_id
        )
        party_receives = sum(
            p.net_amount
            for p in relevant_payments
            if p.breakdown.get("receiver") == party_id
        )

        net_amount = party_receives - party_pays

        # Determine net payer/receiver
        if net_amount > 0:
            payer_id = counterparty_id
            receiver_id = party_id
            net_amount = abs(net_amount)
        else:
            payer_id = party_id
            receiver_id = counterparty_id
            net_amount = abs(net_amount)

        return NettingResult(
            settlement_date=settlement_date,
            currency=currency,
            gross_payments=relevant_payments,
            net_amount=Decimal(str(net_amount)),
            payer_id=payer_id,
            receiver_id=receiver_id,
            netted_trade_ids=[p.trade_id for p in relevant_payments],
        )

    def process_settlement(self, instruction_id: str) -> bool:
        """Process settlement instruction.

        Args:
            instruction_id: Instruction ID

        Returns:
            True if settled successfully, False otherwise

        Raises:
            ValueError: If instruction not found
        """
        if instruction_id not in self._instructions:
            raise ValueError(f"Instruction {instruction_id} not found")

        instruction = self._instructions[instruction_id]

        # Simulate settlement processing
        # In production, this would integrate with payment systems
        if instruction.status == SettlementStatus.PENDING:
            # Mark as matched (simplified)
            instruction.status = SettlementStatus.MATCHED
            instruction.matched_at = datetime.now()

            # Mark as settled (simplified)
            instruction.status = SettlementStatus.SETTLED
            instruction.settled_at = datetime.now()
            return True

        return False

    def fail_settlement(
        self,
        instruction_id: str,
        reason: FailureReason,
        details: Optional[str] = None,
    ) -> None:
        """Mark settlement as failed.

        Args:
            instruction_id: Instruction ID
            reason: Failure reason
            details: Additional details

        Raises:
            ValueError: If instruction not found
        """
        if instruction_id not in self._instructions:
            raise ValueError(f"Instruction {instruction_id} not found")

        instruction = self._instructions[instruction_id]
        instruction.status = SettlementStatus.FAILED
        instruction.failure_reason = reason
        instruction.failure_details = details
        instruction.retry_count += 1

    def retry_settlement(self, instruction_id: str) -> bool:
        """Retry failed settlement.

        Args:
            instruction_id: Instruction ID

        Returns:
            True if retry was attempted, False otherwise

        Raises:
            ValueError: If instruction not found or cannot be retried
        """
        if instruction_id not in self._instructions:
            raise ValueError(f"Instruction {instruction_id} not found")

        instruction = self._instructions[instruction_id]

        if not instruction.can_retry:
            raise ValueError(f"Instruction {instruction_id} cannot be retried")

        instruction.status = SettlementStatus.PENDING
        return self.process_settlement(instruction_id)

    def cancel_instruction(self, instruction_id: str, reason: Optional[str] = None) -> None:
        """Cancel settlement instruction.

        Args:
            instruction_id: Instruction ID
            reason: Cancellation reason

        Raises:
            ValueError: If instruction not found or already settled
        """
        if instruction_id not in self._instructions:
            raise ValueError(f"Instruction {instruction_id} not found")

        instruction = self._instructions[instruction_id]

        if instruction.status == SettlementStatus.SETTLED:
            raise ValueError(f"Cannot cancel settled instruction {instruction_id}")

        instruction.status = SettlementStatus.CANCELLED

    def get_instruction(self, instruction_id: str) -> Optional[SettlementInstruction]:
        """Get settlement instruction.

        Args:
            instruction_id: Instruction ID

        Returns:
            Instruction or None
        """
        return self._instructions.get(instruction_id)

    def get_instructions_by_date(self, settlement_date: date) -> List[SettlementInstruction]:
        """Get all instructions for a settlement date.

        Args:
            settlement_date: Settlement date

        Returns:
            List of instructions
        """
        return [
            inst
            for inst in self._instructions.values()
            if inst.settlement_date == settlement_date
        ]

    def get_failed_instructions(self) -> List[SettlementInstruction]:
        """Get all failed settlement instructions.

        Returns:
            List of failed instructions
        """
        return [
            inst
            for inst in self._instructions.values()
            if inst.status == SettlementStatus.FAILED
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get settlement statistics.

        Returns:
            Dictionary with statistics
        """
        total = len(self._instructions)
        settled = sum(1 for i in self._instructions.values() if i.status == SettlementStatus.SETTLED)
        failed = sum(1 for i in self._instructions.values() if i.status == SettlementStatus.FAILED)
        pending = sum(1 for i in self._instructions.values() if i.status == SettlementStatus.PENDING)

        return {
            "total_instructions": total,
            "settled": settled,
            "failed": failed,
            "pending": pending,
            "settlement_rate": settled / total if total else 0.0,
            "failure_rate": failed / total if total else 0.0,
        }


__all__ = [
    "SettlementMethod",
    "SettlementStatus",
    "FailureReason",
    "SettlementAccount",
    "StandingSettlementInstruction",
    "SettlementInstruction",
    "PaymentCalculation",
    "NettingResult",
    "SettlementManager",
]
