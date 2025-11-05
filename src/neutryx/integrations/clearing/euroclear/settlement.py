"""Euroclear settlement service and orchestration."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Dict, List, Optional

from .connector import EuroclearConnector
from .messages import (
    EuroclearSettlementInstruction,
    EuroclearConfirmation,
    EuroclearStatus,
    SettlementStatus,
    SettlementType,
)


class EuroclearSettlementService:
    """High-level service for Euroclear settlement operations.

    Provides business logic and orchestration for securities settlement
    through Euroclear, including instruction generation, submission,
    and lifecycle management.
    """

    def __init__(self, connector: EuroclearConnector):
        """Initialize Euroclear settlement service.

        Args:
            connector: Euroclear connector instance
        """
        self.connector = connector
        self._pending_instructions: Dict[str, EuroclearSettlementInstruction] = {}

    async def settle_securities_trade(
        self,
        sender_reference: str,
        settlement_type: SettlementType,
        settlement_date: date,
        trade_date: date,
        isin: str,
        quantity: Decimal,
        delivering_party: str,
        receiving_party: str,
        participant_bic: str,
        settlement_amount: Optional[Decimal] = None,
        settlement_currency: Optional[str] = None,
        security_name: Optional[str] = None,
        partial_settlement_allowed: bool = False,
    ) -> EuroclearConfirmation:
        """Submit securities trade for Euroclear settlement.

        Args:
            sender_reference: Sender's trade reference
            settlement_type: Type of settlement (DVP, RVP, FOP, etc.)
            settlement_date: Intended settlement date
            trade_date: Trade execution date
            isin: ISIN code of security
            quantity: Quantity of securities
            delivering_party: Delivering party account
            receiving_party: Receiving party account
            participant_bic: Euroclear participant BIC
            settlement_amount: Settlement amount (required for DVP/RVP)
            settlement_currency: Settlement currency (required for DVP/RVP)
            security_name: Optional security description
            partial_settlement_allowed: Allow partial settlement

        Returns:
            Settlement confirmation

        Raises:
            ValueError: If validation fails
        """
        # Generate instruction ID
        instruction_id = f"ECSI_{uuid.uuid4().hex[:12].upper()}"

        # Create settlement instruction
        instruction = EuroclearSettlementInstruction(
            instruction_id=instruction_id,
            sender_reference=sender_reference,
            settlement_type=settlement_type,
            settlement_date=settlement_date,
            trade_date=trade_date,
            isin=isin,
            security_name=security_name,
            quantity=quantity,
            delivering_party=delivering_party,
            receiving_party=receiving_party,
            participant_bic=participant_bic,
            settlement_amount=settlement_amount,
            settlement_currency=settlement_currency,
            partial_settlement_allowed=partial_settlement_allowed,
        )

        # Validate DVP fields
        instruction.validate_dvp_fields()

        # Store instruction
        self._pending_instructions[instruction_id] = instruction

        # Submit to Euroclear
        confirmation = await self.connector.submit_settlement_instruction(instruction)

        return confirmation

    async def get_instruction_status(
        self,
        instruction_id: str
    ) -> EuroclearStatus:
        """Get status of a settlement instruction.

        Args:
            instruction_id: Settlement instruction ID

        Returns:
            Current settlement status
        """
        return await self.connector.get_settlement_status(instruction_id)

    async def cancel_instruction(
        self,
        instruction_id: str
    ) -> bool:
        """Cancel a pending settlement instruction.

        Args:
            instruction_id: Instruction ID to cancel

        Returns:
            True if cancellation successful
        """
        result = await self.connector.cancel_settlement_instruction(instruction_id)

        if result and instruction_id in self._pending_instructions:
            self._pending_instructions[instruction_id].status = SettlementStatus.CANCELLED

        return result

    async def amend_instruction(
        self,
        instruction_id: str,
        new_settlement_date: Optional[date] = None,
        new_quantity: Optional[Decimal] = None,
    ) -> EuroclearConfirmation:
        """Amend a pending settlement instruction.

        Args:
            instruction_id: Instruction ID to amend
            new_settlement_date: New settlement date
            new_quantity: New quantity

        Returns:
            Amendment confirmation
        """
        confirmation = await self.connector.amend_settlement_instruction(
            instruction_id=instruction_id,
            new_settlement_date=new_settlement_date,
            new_quantity=new_quantity,
        )

        # Update local cache
        if instruction_id in self._pending_instructions:
            if new_settlement_date:
                self._pending_instructions[instruction_id].settlement_date = new_settlement_date
            if new_quantity:
                self._pending_instructions[instruction_id].quantity = new_quantity

        return confirmation

    async def get_pending_instructions(
        self,
        settlement_date: Optional[date] = None,
        isin: Optional[str] = None,
    ) -> List[EuroclearSettlementInstruction]:
        """Get all pending settlement instructions.

        Args:
            settlement_date: Optional filter by settlement date
            isin: Optional filter by ISIN

        Returns:
            List of pending instructions
        """
        instructions = [
            inst for inst in self._pending_instructions.values()
            if inst.status in (
                SettlementStatus.PENDING,
                SettlementStatus.MATCHED,
                SettlementStatus.AFFIRMED,
            )
        ]

        if settlement_date:
            instructions = [
                inst for inst in instructions
                if inst.settlement_date == settlement_date
            ]

        if isin:
            instructions = [
                inst for inst in instructions
                if inst.isin == isin
            ]

        return instructions

    async def get_holdings(
        self,
        account_id: str,
        as_of_date: Optional[date] = None
    ) -> Dict:
        """Get securities holdings for an account.

        Args:
            account_id: Euroclear account ID
            as_of_date: As-of date

        Returns:
            Holdings information
        """
        return await self.connector.get_holdings(account_id, as_of_date)

    async def reconcile_settlements(
        self,
        settlement_date: date
    ) -> Dict:
        """Reconcile settlements for a given settlement date.

        Args:
            settlement_date: Settlement date to reconcile

        Returns:
            Reconciliation report
        """
        instructions = await self.get_pending_instructions(settlement_date)

        # Get statuses for all instructions
        statuses = []
        for inst in instructions:
            try:
                status = await self.get_instruction_status(inst.instruction_id)
                statuses.append(status)
            except Exception as e:
                statuses.append({
                    "instruction_id": inst.instruction_id,
                    "error": str(e)
                })

        # Calculate statistics
        total_instructions = len(instructions)
        settled = sum(1 for s in statuses if isinstance(s, EuroclearStatus) and s.status == SettlementStatus.SETTLED)
        pending = sum(1 for s in statuses if isinstance(s, EuroclearStatus) and s.status == SettlementStatus.PENDING)
        failing = sum(1 for s in statuses if isinstance(s, EuroclearStatus) and s.status == SettlementStatus.FAILING)

        # Calculate by ISIN
        by_isin: Dict[str, Dict] = {}
        for inst in instructions:
            if inst.isin not in by_isin:
                by_isin[inst.isin] = {
                    "isin": inst.isin,
                    "total_quantity": Decimal(0),
                    "settled_quantity": Decimal(0),
                    "pending_quantity": Decimal(0),
                }
            by_isin[inst.isin]["total_quantity"] += inst.quantity

        return {
            "settlement_date": settlement_date.isoformat(),
            "total_instructions": total_instructions,
            "settled": settled,
            "pending": pending,
            "failing": failing,
            "settlement_rate": (settled / total_instructions * 100) if total_instructions > 0 else 0,
            "by_isin": list(by_isin.values()),
            "statuses": [
                {
                    "instruction_id": s.instruction_id if isinstance(s, EuroclearStatus) else s.get("instruction_id"),
                    "status": s.status.value if isinstance(s, EuroclearStatus) else "error",
                    "matched": s.matched if isinstance(s, EuroclearStatus) else False,
                    "affirmed": s.affirmed if isinstance(s, EuroclearStatus) else False,
                    "securities_delivered": s.securities_delivered if isinstance(s, EuroclearStatus) else False,
                }
                for s in statuses
            ],
        }


__all__ = ["EuroclearSettlementService"]
