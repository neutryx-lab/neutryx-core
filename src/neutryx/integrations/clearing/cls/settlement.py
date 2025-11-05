"""CLS settlement service and orchestration."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Dict, List, Optional

from .connector import CLSConnector
from .messages import (
    CLSCurrency,
    CLSSettlementInstruction,
    CLSConfirmation,
    CLSStatus,
    CLSSettlementStatus,
)


class CLSSettlementService:
    """High-level service for CLS settlement operations.

    Provides business logic and orchestration for FX settlement
    through CLS, including instruction generation, submission,
    and lifecycle management.
    """

    def __init__(self, connector: CLSConnector):
        """Initialize CLS settlement service.

        Args:
            connector: CLS connector instance
        """
        self.connector = connector
        self._pending_instructions: Dict[str, CLSSettlementInstruction] = {}

    async def settle_fx_trade(
        self,
        trade_id: str,
        buy_currency: str,
        buy_amount: Decimal,
        sell_currency: str,
        sell_amount: Decimal,
        value_date: date,
        submitter_bic: str,
        counterparty_bic: str,
        settlement_member: str,
        execution_timestamp: Optional[datetime] = None,
    ) -> CLSConfirmation:
        """Submit FX trade for CLS settlement.

        Args:
            trade_id: Unique trade identifier
            buy_currency: Currency being bought (ISO 4217)
            buy_amount: Amount being bought
            sell_currency: Currency being sold (ISO 4217)
            sell_amount: Amount being sold
            value_date: Settlement value date
            submitter_bic: Submitting party BIC code
            counterparty_bic: Counterparty BIC code
            settlement_member: CLS settlement member code
            execution_timestamp: Optional trade execution timestamp

        Returns:
            Settlement confirmation

        Raises:
            ValueError: If currencies are not CLS-eligible or invalid
        """
        # Validate currencies
        try:
            buy_ccy = CLSCurrency(buy_currency)
            sell_ccy = CLSCurrency(sell_currency)
        except ValueError as e:
            raise ValueError(f"Invalid or non-CLS-eligible currency: {e}")

        # Calculate FX rate
        fx_rate = sell_amount / buy_amount

        # Generate settlement session ID (typically based on value date)
        settlement_session_id = f"CLS_{value_date.strftime('%Y%m%d')}"

        # Create settlement instruction
        instruction = CLSSettlementInstruction(
            instruction_id=f"CLSI_{uuid.uuid4().hex[:12].upper()}",
            trade_id=trade_id,
            settlement_session_id=settlement_session_id,
            buy_currency=buy_ccy,
            buy_amount=buy_amount,
            sell_currency=sell_ccy,
            sell_amount=sell_amount,
            fx_rate=fx_rate,
            value_date=value_date,
            settlement_date=value_date,  # CLS settles on value date
            submitter_bic=submitter_bic,
            counterparty_bic=counterparty_bic,
            settlement_member=settlement_member,
            execution_timestamp=execution_timestamp,
        )

        # Validate currencies are different
        instruction.validate_currencies()

        # Store instruction
        self._pending_instructions[instruction.instruction_id] = instruction

        # Submit to CLS
        confirmation = await self.connector.submit_settlement_instruction(instruction)

        return confirmation

    async def get_instruction_status(
        self,
        instruction_id: str
    ) -> CLSStatus:
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
            self._pending_instructions[instruction_id].status = CLSSettlementStatus.CANCELLED

        return result

    async def get_pending_instructions(
        self,
        value_date: Optional[date] = None
    ) -> List[CLSSettlementInstruction]:
        """Get all pending settlement instructions.

        Args:
            value_date: Optional filter by value date

        Returns:
            List of pending instructions
        """
        instructions = [
            inst for inst in self._pending_instructions.values()
            if inst.status in (CLSSettlementStatus.PENDING, CLSSettlementStatus.MATCHED)
        ]

        if value_date:
            instructions = [
                inst for inst in instructions
                if inst.value_date == value_date
            ]

        return instructions

    async def get_session_liquidity(
        self,
        value_date: date
    ) -> Dict:
        """Get liquidity information for a settlement session.

        Args:
            value_date: Settlement value date

        Returns:
            Dictionary with liquidity details
        """
        settlement_session_id = f"CLS_{value_date.strftime('%Y%m%d')}"
        return await self.connector.get_session_liquidity(settlement_session_id)

    def calculate_settlement_exposure(
        self,
        instructions: List[CLSSettlementInstruction]
    ) -> Dict[str, Decimal]:
        """Calculate net settlement exposure by currency.

        Args:
            instructions: List of settlement instructions

        Returns:
            Dictionary mapping currency to net exposure
        """
        exposures: Dict[str, Decimal] = {}

        for inst in instructions:
            # Add buy amount
            buy_ccy = inst.buy_currency.value
            exposures[buy_ccy] = exposures.get(buy_ccy, Decimal(0)) + inst.buy_amount

            # Subtract sell amount
            sell_ccy = inst.sell_currency.value
            exposures[sell_ccy] = exposures.get(sell_ccy, Decimal(0)) - inst.sell_amount

        return exposures

    async def reconcile_settlements(
        self,
        value_date: date
    ) -> Dict:
        """Reconcile settlements for a given value date.

        Args:
            value_date: Value date to reconcile

        Returns:
            Reconciliation report
        """
        instructions = await self.get_pending_instructions(value_date)

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
        settled = sum(1 for s in statuses if isinstance(s, CLSStatus) and s.status == CLSSettlementStatus.SETTLED)
        pending = sum(1 for s in statuses if isinstance(s, CLSStatus) and s.status == CLSSettlementStatus.PENDING)
        failed = sum(1 for s in statuses if isinstance(s, CLSStatus) and s.status == CLSSettlementStatus.FAILED)

        return {
            "value_date": value_date.isoformat(),
            "total_instructions": total_instructions,
            "settled": settled,
            "pending": pending,
            "failed": failed,
            "settlement_rate": (settled / total_instructions * 100) if total_instructions > 0 else 0,
            "statuses": [
                {
                    "instruction_id": s.instruction_id if isinstance(s, CLSStatus) else s.get("instruction_id"),
                    "status": s.status.value if isinstance(s, CLSStatus) else "error",
                    "pay_in_complete": s.pay_in_complete if isinstance(s, CLSStatus) else False,
                    "pay_out_complete": s.pay_out_complete if isinstance(s, CLSStatus) else False,
                }
                for s in statuses
            ],
        }


__all__ = ["CLSSettlementService"]
