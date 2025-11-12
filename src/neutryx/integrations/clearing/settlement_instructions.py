"""Settlement instruction generation utilities.

This module handles settlement instruction (SI) generation for cleared trades:
1. SI generation from confirmed trades
2. DVP (Delivery vs Payment) instruction matching
3. Settlement status tracking
4. SWIFT MT5xx message generation
5. ISO 20022 SEMT/SESE message generation
6. Settlement exception handling (fails, partial settlements)

Supports multiple settlement types:
- DVP (Delivery vs Payment)
- RVP (Receive vs Payment)
- FOP (Free of Payment)
- DAP (Delivery Against Payment)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from .base import Party


class SettlementType(str, Enum):
    """Settlement type."""
    DVP = "delivery_vs_payment"  # Delivery vs Payment (simultaneous)
    RVP = "receive_vs_payment"   # Receive vs Payment
    FOP = "free_of_payment"      # Free of Payment (no cash leg)
    DAP = "delivery_against_payment"  # Delivery against payment
    PFOD = "payment_free_of_delivery"  # Payment free of delivery


class SettlementStatus(str, Enum):
    """Settlement status."""
    PENDING = "pending"
    INSTRUCTED = "instructed"
    MATCHED = "matched"
    AFFIRMED = "affirmed"
    SETTLED = "settled"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PENDING_CANCELLATION = "pending_cancellation"


class SettlementMethod(str, Enum):
    """Settlement method."""
    CCP = "ccp"              # CCP settlement
    BILATERAL = "bilateral"   # Bilateral settlement
    CSD = "csd"              # Central Securities Depository
    ICSD = "icsd"            # International CSD
    TARGET2 = "target2"      # TARGET2 (ECB)
    FEDWIRE = "fedwire"      # Fedwire (US)
    SWIFT = "swift"          # SWIFT messaging


class SwiftSettlementMessageType(str, Enum):
    """SWIFT settlement messaging types supported by the generator."""

    MT540 = "MT540"  # Receive free
    MT541 = "MT541"  # Receive against payment
    MT542 = "MT542"  # Deliver free
    MT543 = "MT543"  # Deliver against payment


class FailReason(str, Enum):
    """Settlement fail reasons."""
    INSUFFICIENT_SECURITIES = "insufficient_securities"
    INSUFFICIENT_CASH = "insufficient_cash"
    ACCOUNT_BLOCKED = "account_blocked"
    CUTOFF_TIME_MISSED = "cutoff_time_missed"
    HOLIDAY = "holiday"
    COUNTERPARTY_FAIL = "counterparty_fail"
    SYSTEM_ERROR = "system_error"
    CANCELLED = "cancelled"
    PENDING_DOCUMENTATION = "pending_documentation"
    CORPORATE_ACTION = "corporate_action"


class CashFlow(BaseModel):
    """Cash flow for settlement."""

    amount: Decimal = Field(..., description="Cash amount")
    currency: str = Field(..., description="Currency code")
    direction: str = Field(..., description="pay or receive")

    payment_date: date = Field(..., description="Payment date")
    value_date: date = Field(..., description="Value date")

    payer_account: str = Field(..., description="Payer account")
    receiver_account: str = Field(..., description="Receiver account")

    # Bank details
    payer_bank: Optional[str] = Field(None, description="Payer bank BIC")
    receiver_bank: Optional[str] = Field(None, description="Receiver bank BIC")
    intermediary_bank: Optional[str] = Field(None, description="Intermediary bank")

    # References
    payment_reference: Optional[str] = Field(None, description="Payment reference")
    remittance_info: Optional[str] = Field(None, description="Remittance information")

    metadata: Dict[str, Any] = Field(default_factory=dict)


class SecuritiesMovement(BaseModel):
    """Securities movement for settlement."""

    security_id: str = Field(..., description="Security identifier (ISIN, CUSIP, etc.)")
    security_id_type: str = Field(default="ISIN", description="ID type")
    quantity: Decimal = Field(..., description="Quantity")
    direction: str = Field(..., description="deliver or receive")

    settlement_date: date = Field(..., description="Settlement date")

    # Accounts
    deliverer_account: str = Field(..., description="Delivering account")
    receiver_account: str = Field(..., description="Receiving account")

    # Custodian
    deliverer_custodian: Optional[str] = Field(None, description="Delivering custodian")
    receiver_custodian: Optional[str] = Field(None, description="Receiving custodian")

    # Place of settlement
    place_of_settlement: Optional[str] = Field(None, description="Settlement place (CSD)")

    metadata: Dict[str, Any] = Field(default_factory=dict)


class SettlementInstruction(BaseModel):
    """Settlement instruction."""

    instruction_id: str = Field(
        default_factory=lambda: f"SI-{uuid4().hex[:12].upper()}"
    )
    status: SettlementStatus = Field(default=SettlementStatus.PENDING)

    # Trade reference
    trade_id: str = Field(..., description="Related trade ID")
    confirmation_id: Optional[str] = Field(None, description="Related confirmation ID")

    # Settlement type
    settlement_type: SettlementType = Field(..., description="Settlement type")
    settlement_method: SettlementMethod = Field(..., description="Settlement method")
    settlement_date: date = Field(..., description="Settlement date")

    # Counterparties
    deliverer: Party = Field(..., description="Delivering party")
    receiver: Party = Field(..., description="Receiving party")

    # Cash leg
    cash_flows: List[CashFlow] = Field(default_factory=list, description="Cash flows")

    # Securities leg
    securities_movements: List[SecuritiesMovement] = Field(
        default_factory=list,
        description="Securities movements"
    )

    # Settlement details
    trade_date: date = Field(..., description="Trade date")
    intended_settlement_date: date = Field(..., description="Intended settlement date")
    actual_settlement_date: Optional[date] = Field(None, description="Actual settlement date")

    # Priority
    priority: str = Field(default="normal", description="Priority: high/normal/low")

    # Partial settlement
    partial_settlement_allowed: bool = Field(default=False)
    partial_amount_settled: Optional[Decimal] = Field(None)

    # Lifecycle
    created_time: datetime = Field(default_factory=datetime.utcnow)
    instructed_time: Optional[datetime] = Field(None)
    matched_time: Optional[datetime] = Field(None)
    settled_time: Optional[datetime] = Field(None)

    # Failure tracking
    fail_reason: Optional[FailReason] = Field(None)
    fail_details: Optional[str] = Field(None)
    retry_count: int = Field(default=0)
    last_retry_time: Optional[datetime] = Field(None)

    # External references
    ccp_instruction_id: Optional[str] = Field(None)
    csd_instruction_id: Optional[str] = Field(None)
    swift_reference: Optional[str] = Field(None)

    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('settlement_date', 'intended_settlement_date')
    @classmethod
    def validate_dates(cls, v, info):
        if 'trade_date' in info.data:
            if v < info.data['trade_date']:
                raise ValueError(f"{info.field_name} cannot be before trade_date")
        return v


class SettlementSchedule(BaseModel):
    """Settlement schedule for multi-period settlements."""

    schedule_id: str = Field(
        default_factory=lambda: f"SCH-{uuid4().hex[:12].upper()}"
    )
    trade_id: str = Field(..., description="Related trade ID")

    settlement_dates: List[date] = Field(..., description="Settlement dates")
    settlement_amounts: List[Decimal] = Field(..., description="Amounts per date")

    frequency: str = Field(..., description="Settlement frequency")
    day_count: str = Field(default="ACT/360", description="Day count convention")

    next_settlement_date: Optional[date] = Field(None)
    last_settlement_date: Optional[date] = Field(None)

    metadata: Dict[str, Any] = Field(default_factory=dict)


class SwiftSettlementMessage(BaseModel):
    """Structured representation of a SWIFT settlement instruction."""

    message_type: SwiftSettlementMessageType = Field(...)
    instruction_id: str = Field(...)
    sender: str = Field(...)
    receiver: str = Field(...)
    trade_date: date = Field(...)
    settlement_date: date = Field(...)
    settlement_type: SettlementType = Field(...)
    securities: List[Dict[str, Any]] = Field(default_factory=list)
    cash_flows: List[Dict[str, Any]] = Field(default_factory=list)
    narrative: Optional[str] = Field(None)
    references: Dict[str, str] = Field(default_factory=dict)


class CLSInstruction(BaseModel):
    """Representation of a CLS pay-in instruction."""

    instruction_id: str = Field(...)
    trade_id: str = Field(...)
    currency_pair: str = Field(...)
    value_date: date = Field(...)
    gross_amount: Decimal = Field(...)
    direction: str = Field(..., description="pay or receive")
    session: str = Field(default="DEFAULT")
    funding_cutoff: datetime = Field(...)
    status: SettlementStatus = Field(default=SettlementStatus.PENDING)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EuroclearInstruction(BaseModel):
    """Instruction payload for Euroclear connectivity."""

    instruction_id: str = Field(...)
    trade_id: str = Field(...)
    place_of_settlement: str = Field(...)
    participant_account: str = Field(...)
    counterparty_account: str = Field(...)
    settlement_date: date = Field(...)
    settlement_type: SettlementType = Field(...)
    securities: List[Dict[str, Any]] = Field(default_factory=list)
    cash_flows: List[Dict[str, Any]] = Field(default_factory=list)
    priority: str = Field(...)
    references: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class SettlementStatistics:
    """Settlement statistics."""
    total_instructions: int = 0
    settled_instructions: int = 0
    pending_instructions: int = 0
    failed_instructions: int = 0
    partial_settlements: int = 0

    total_value: Decimal = Decimal("0")
    settled_value: Decimal = Decimal("0")
    pending_value: Decimal = Decimal("0")

    avg_settlement_time_hours: float = 0.0
    settlement_rate: float = 100.0
    fail_rate: float = 0.0

    def update_rates(self):
        """Update settlement and fail rates."""
        if self.total_instructions > 0:
            self.settlement_rate = (self.settled_instructions / self.total_instructions) * 100
            self.fail_rate = (self.failed_instructions / self.total_instructions) * 100


class SettlementInstructionGenerator:
    """Generate settlement instructions from confirmed trades."""

    def __init__(self):
        self.instructions: Dict[str, SettlementInstruction] = {}
        self.statistics = SettlementStatistics()

    def generate_instruction(
        self,
        trade_id: str,
        trade_date: date,
        settlement_date: date,
        deliverer: Party,
        receiver: Party,
        settlement_type: SettlementType = SettlementType.DVP,
        settlement_method: SettlementMethod = SettlementMethod.CCP,
        **kwargs
    ) -> SettlementInstruction:
        """Generate settlement instruction."""
        instruction = SettlementInstruction(
            trade_id=trade_id,
            trade_date=trade_date,
            settlement_date=settlement_date,
            intended_settlement_date=settlement_date,
            deliverer=deliverer,
            receiver=receiver,
            settlement_type=settlement_type,
            settlement_method=settlement_method,
            **kwargs
        )

        self.instructions[instruction.instruction_id] = instruction
        self.statistics.total_instructions += 1
        self.statistics.pending_instructions += 1

        return instruction

    def add_cash_flow(
        self,
        instruction_id: str,
        cash_flow: CashFlow
    ) -> SettlementInstruction:
        """Add cash flow to instruction."""
        instruction = self.instructions.get(instruction_id)
        if not instruction:
            raise ValueError(f"Instruction {instruction_id} not found")

        instruction.cash_flows.append(cash_flow)
        self.statistics.total_value += cash_flow.amount
        self.statistics.pending_value += cash_flow.amount

        return instruction

    def add_securities_movement(
        self,
        instruction_id: str,
        movement: SecuritiesMovement
    ) -> SettlementInstruction:
        """Add securities movement to instruction."""
        instruction = self.instructions.get(instruction_id)
        if not instruction:
            raise ValueError(f"Instruction {instruction_id} not found")

        instruction.securities_movements.append(movement)

        return instruction

    def match_instruction(
        self,
        instruction_id: str,
        counterparty_instruction_id: Optional[str] = None
    ) -> SettlementInstruction:
        """Mark instruction as matched."""
        instruction = self.instructions.get(instruction_id)
        if not instruction:
            raise ValueError(f"Instruction {instruction_id} not found")

        if instruction.status != SettlementStatus.INSTRUCTED:
            raise ValueError(f"Instruction must be in INSTRUCTED status")

        instruction.status = SettlementStatus.MATCHED
        instruction.matched_time = datetime.utcnow()

        if counterparty_instruction_id:
            instruction.metadata['counterparty_instruction_id'] = counterparty_instruction_id

        return instruction

    def settle_instruction(
        self,
        instruction_id: str,
        actual_settlement_date: Optional[date] = None,
        partial_amount: Optional[Decimal] = None
    ) -> SettlementInstruction:
        """Mark instruction as settled."""
        instruction = self.instructions.get(instruction_id)
        if not instruction:
            raise ValueError(f"Instruction {instruction_id} not found")

        instruction.actual_settlement_date = actual_settlement_date or date.today()
        instruction.settled_time = datetime.utcnow()

        if partial_amount is not None and partial_amount < sum(cf.amount for cf in instruction.cash_flows):
            instruction.status = SettlementStatus.PARTIAL
            instruction.partial_amount_settled = partial_amount
            self.statistics.partial_settlements += 1
        else:
            instruction.status = SettlementStatus.SETTLED
            self.statistics.settled_instructions += 1
            self.statistics.pending_instructions -= 1

            # Update statistics
            total_amount = sum(cf.amount for cf in instruction.cash_flows)
            self.statistics.settled_value += total_amount
            self.statistics.pending_value -= total_amount

            # Calculate settlement time
            if instruction.instructed_time:
                settlement_time = (instruction.settled_time - instruction.instructed_time).total_seconds() / 3600
                # Update running average
                total = self.statistics.settled_instructions
                current_avg = self.statistics.avg_settlement_time_hours
                self.statistics.avg_settlement_time_hours = (
                    (current_avg * (total - 1) + settlement_time) / total
                )

        self.statistics.update_rates()

        return instruction

    def fail_instruction(
        self,
        instruction_id: str,
        fail_reason: FailReason,
        fail_details: Optional[str] = None
    ) -> SettlementInstruction:
        """Mark instruction as failed."""
        instruction = self.instructions.get(instruction_id)
        if not instruction:
            raise ValueError(f"Instruction {instruction_id} not found")

        previous_status = instruction.status
        instruction.status = SettlementStatus.FAILED
        instruction.fail_reason = fail_reason
        instruction.fail_details = fail_details

        self.statistics.failed_instructions += 1
        if previous_status == SettlementStatus.PENDING:
            self.statistics.pending_instructions -= 1

        self.statistics.update_rates()

        return instruction

    def retry_instruction(
        self,
        instruction_id: str,
        new_settlement_date: Optional[date] = None
    ) -> SettlementInstruction:
        """Retry failed instruction."""
        instruction = self.instructions.get(instruction_id)
        if not instruction:
            raise ValueError(f"Instruction {instruction_id} not found")

        if instruction.status != SettlementStatus.FAILED:
            raise ValueError("Can only retry failed instructions")

        instruction.status = SettlementStatus.PENDING
        instruction.retry_count += 1
        instruction.last_retry_time = datetime.utcnow()

        if new_settlement_date:
            instruction.settlement_date = new_settlement_date

        # Update statistics
        self.statistics.failed_instructions -= 1
        self.statistics.pending_instructions += 1
        self.statistics.update_rates()

        return instruction

    def cancel_instruction(
        self,
        instruction_id: str
    ) -> SettlementInstruction:
        """Cancel instruction."""
        instruction = self.instructions.get(instruction_id)
        if not instruction:
            raise ValueError(f"Instruction {instruction_id} not found")

        if instruction.status in [SettlementStatus.SETTLED, SettlementStatus.CANCELLED]:
            raise ValueError(f"Cannot cancel instruction in status {instruction.status}")

        previous_status = instruction.status
        instruction.status = SettlementStatus.CANCELLED

        if previous_status == SettlementStatus.PENDING:
            self.statistics.pending_instructions -= 1

        return instruction

    def generate_swift_message(self, instruction_id: str) -> SwiftSettlementMessage:
        """Generate a SWIFT MT54x representation for an instruction."""

        instruction = self.instructions.get(instruction_id)
        if not instruction:
            raise ValueError(f"Instruction {instruction_id} not found")

        message_type = self._resolve_swift_message_type(instruction)

        securities = [
            {
                "isin": movement.security_id,
                "quantity": str(movement.quantity),
                "direction": movement.direction,
            }
            for movement in instruction.securities_movements
        ]

        cash_flows = [
            {
                "amount": str(flow.amount),
                "currency": flow.currency,
                "direction": flow.direction,
            }
            for flow in instruction.cash_flows
        ]

        references = {
            "trade_reference": instruction.trade_id,
            "instruction_reference": instruction.instruction_id,
        }
        if instruction.confirmation_id:
            references["confirmation_id"] = instruction.confirmation_id

        return SwiftSettlementMessage(
            message_type=message_type,
            instruction_id=instruction.instruction_id,
            sender=instruction.deliverer.bic or instruction.deliverer.party_id,
            receiver=instruction.receiver.bic or instruction.receiver.party_id,
            trade_date=instruction.trade_date,
            settlement_date=instruction.settlement_date,
            settlement_type=instruction.settlement_type,
            securities=securities,
            cash_flows=cash_flows,
            narrative=f"Settlement of trade {instruction.trade_id}",
            references=references,
        )

    def generate_swift_mt540(self, instruction_id: str) -> Dict[str, str]:
        """Convenience wrapper returning MT540 message as dict."""

        message = self.generate_swift_message(instruction_id)
        if message.message_type != SwiftSettlementMessageType.MT540:
            raise ValueError("Instruction does not map to MT540 (receive free)")

        return message.model_dump()

    def generate_cls_instruction(
        self,
        instruction_id: str,
        currency_pair: str,
        session: str = "DEFAULT",
    ) -> CLSInstruction:
        """Generate a CLS pay-in instruction for FX settlements."""

        instruction = self.instructions.get(instruction_id)
        if not instruction:
            raise ValueError(f"Instruction {instruction_id} not found")

        if len(currency_pair) != 7 or currency_pair[3] != "/":
            raise ValueError("currency_pair must be formatted as 'XXX/YYY'")

        if not instruction.cash_flows:
            raise ValueError("CLS instructions require cash flows")

        gross_amount = sum(abs(flow.amount) for flow in instruction.cash_flows)
        direction = instruction.cash_flows[0].direction

        funding_cutoff = datetime.combine(
            instruction.settlement_date,
            datetime.min.time(),
        ) - timedelta(hours=2)

        return CLSInstruction(
            instruction_id=instruction.instruction_id,
            trade_id=instruction.trade_id,
            currency_pair=currency_pair,
            value_date=instruction.settlement_date,
            gross_amount=gross_amount,
            direction=direction,
            session=session,
            funding_cutoff=funding_cutoff,
            status=instruction.status,
            metadata={
                "deliverer": instruction.deliverer.party_id,
                "receiver": instruction.receiver.party_id,
            },
        )

    def generate_euroclear_instruction(
        self,
        instruction_id: str,
        safekeeping_account: Optional[str] = None,
    ) -> EuroclearInstruction:
        """Generate Euroclear-compatible payload for settlement instruction."""

        instruction = self.instructions.get(instruction_id)
        if not instruction:
            raise ValueError(f"Instruction {instruction_id} not found")

        if not instruction.securities_movements:
            raise ValueError("Euroclear instruction requires securities movements")

        place_of_settlement = next(
            (
                movement.place_of_settlement
                for movement in instruction.securities_movements
                if movement.place_of_settlement
            ),
            "EUROCLEAR",
        )

        participant_account = safekeeping_account or instruction.deliverer.member_id
        if not participant_account:
            participant_account = instruction.deliverer.party_id

        counterparty_account = instruction.receiver.member_id or instruction.receiver.party_id

        securities = [
            {
                "security_id": movement.security_id,
                "quantity": str(movement.quantity),
                "direction": movement.direction,
            }
            for movement in instruction.securities_movements
        ]

        cash_flows = [
            {
                "amount": str(flow.amount),
                "currency": flow.currency,
                "direction": flow.direction,
            }
            for flow in instruction.cash_flows
        ]

        references = {
            "trade_id": instruction.trade_id,
            "instruction_id": instruction.instruction_id,
        }
        if instruction.confirmation_id:
            references["confirmation_id"] = instruction.confirmation_id

        return EuroclearInstruction(
            instruction_id=instruction.instruction_id,
            trade_id=instruction.trade_id,
            place_of_settlement=place_of_settlement,
            participant_account=participant_account,
            counterparty_account=counterparty_account,
            settlement_date=instruction.settlement_date,
            settlement_type=instruction.settlement_type,
            securities=securities,
            cash_flows=cash_flows,
            priority=instruction.priority,
            references=references,
            metadata={
                "method": instruction.settlement_method.value,
                "partial_allowed": instruction.partial_settlement_allowed,
            },
        )

    @staticmethod
    def _resolve_swift_message_type(
        instruction: SettlementInstruction,
    ) -> SwiftSettlementMessageType:
        """Infer SWIFT message type from settlement instruction."""

        if instruction.settlement_type == SettlementType.FOP:
            return SwiftSettlementMessageType.MT540

        if instruction.settlement_type == SettlementType.RVP:
            return SwiftSettlementMessageType.MT541

        if instruction.settlement_type == SettlementType.DVP:
            direction = None
            if instruction.securities_movements:
                direction = instruction.securities_movements[0].direction

            if direction == "receive":
                return SwiftSettlementMessageType.MT541

            return SwiftSettlementMessageType.MT543

        if instruction.settlement_type == SettlementType.DAP:
            return SwiftSettlementMessageType.MT543

        return SwiftSettlementMessageType.MT542

    def get_pending_instructions(
        self,
        settlement_date: Optional[date] = None
    ) -> List[SettlementInstruction]:
        """Get pending instructions."""
        instructions = [
            instr for instr in self.instructions.values()
            if instr.status in [SettlementStatus.PENDING, SettlementStatus.INSTRUCTED, SettlementStatus.MATCHED]
        ]

        if settlement_date:
            instructions = [
                instr for instr in instructions
                if instr.settlement_date == settlement_date
            ]

        return instructions

    def get_failed_instructions(self) -> List[SettlementInstruction]:
        """Get failed instructions."""
        return [
            instr for instr in self.instructions.values()
            if instr.status == SettlementStatus.FAILED
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get settlement statistics."""
        return {
            "total_instructions": self.statistics.total_instructions,
            "settled_instructions": self.statistics.settled_instructions,
            "pending_instructions": self.statistics.pending_instructions,
            "failed_instructions": self.statistics.failed_instructions,
            "partial_settlements": self.statistics.partial_settlements,
            "total_value": float(self.statistics.total_value),
            "settled_value": float(self.statistics.settled_value),
            "pending_value": float(self.statistics.pending_value),
            "avg_settlement_time_hours": self.statistics.avg_settlement_time_hours,
            "settlement_rate": self.statistics.settlement_rate,
            "fail_rate": self.statistics.fail_rate,
        }


__all__ = [
    "SettlementInstruction",
    "SettlementType",
    "SettlementStatus",
    "SettlementMethod",
    "SwiftSettlementMessageType",
    "FailReason",
    "CashFlow",
    "SecuritiesMovement",
    "SettlementSchedule",
    "SettlementStatistics",
    "SettlementInstructionGenerator",
    "SwiftSettlementMessage",
    "CLSInstruction",
    "EuroclearInstruction",
]
