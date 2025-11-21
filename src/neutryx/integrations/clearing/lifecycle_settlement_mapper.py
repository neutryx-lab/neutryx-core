"""Trade lifecycle to settlement mapping and synchronization.

This module automatically handles settlement instruction updates when trade
lifecycle events occur:
1. Amendments → Update existing settlement instructions
2. Novations → Generate new instructions for new counterparty
3. Terminations → Generate close-out settlement instructions
4. Track lifecycle event to settlement instruction relationships
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from neutryx.portfolio.lifecycle import (
    LifecycleEvent,
    LifecycleEventType,
    TradeAmendment,
    TradeNovation,
    TradeTermination,
)
from neutryx.portfolio.contracts.trade import Trade

from .base import Party
from .settlement_instructions import (
    CashFlow,
    SettlementInstruction,
    SettlementInstructionGenerator,
    SettlementMethod,
    SettlementStatus,
    SettlementType,
)


class SettlementAction(str, Enum):
    """Action to take on settlement instructions."""
    UPDATE = "update"  # Update existing instruction
    CANCEL = "cancel"  # Cancel existing instruction
    GENERATE_NEW = "generate_new"  # Generate new instruction
    GENERATE_CLOSEOUT = "generate_closeout"  # Generate close-out settlement
    NO_ACTION = "no_action"  # No settlement impact


class LifecycleSettlementImpact(BaseModel):
    """Impact of lifecycle event on settlement."""

    event_id: str = Field(..., description="Lifecycle event ID")
    event_type: LifecycleEventType = Field(..., description="Event type")
    trade_id: str = Field(..., description="Trade ID")
    event_date: date = Field(..., description="Event date")

    # Settlement action required
    settlement_action: SettlementAction = Field(..., description="Action required")

    # Affected settlement instructions
    affected_instructions: List[str] = Field(
        default_factory=list,
        description="IDs of affected settlement instructions"
    )

    # New settlement instructions to generate
    new_instructions_required: int = Field(
        default=0,
        description="Number of new instructions to generate"
    )

    # Settlement changes
    settlement_changes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Changes to settlement amounts/dates"
    )

    # Close-out details (for terminations)
    closeout_amount: Optional[Decimal] = Field(None, description="Close-out payment amount")
    closeout_payer: Optional[str] = Field(None, description="Party paying close-out")
    closeout_date: Optional[date] = Field(None, description="Close-out payment date")

    # Processing status
    processed: bool = Field(default=False, description="Whether settlement update completed")
    processed_time: Optional[datetime] = Field(None, description="Processing timestamp")
    error: Optional[str] = Field(None, description="Error message if processing failed")

    metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class LifecycleSettlementConfig:
    """Configuration for lifecycle settlement mapping."""

    # Automatic processing
    auto_process_amendments: bool = True
    auto_process_novations: bool = True
    auto_process_terminations: bool = True

    # Settlement date adjustments
    novation_settlement_days: int = 2  # T+2 for novation settlements
    termination_settlement_days: int = 2  # T+2 for termination settlements

    # Thresholds for requiring manual review
    manual_review_threshold: Optional[Decimal] = None  # Amount threshold

    # Notification callbacks
    notification_callbacks: List[Callable] = None

    def __post_init__(self):
        if self.notification_callbacks is None:
            self.notification_callbacks = []


class LifecycleSettlementMapper:
    """Maps trade lifecycle events to settlement instruction updates.

    This service automatically:
    1. Analyzes lifecycle events to determine settlement impact
    2. Updates existing settlement instructions when trades are amended
    3. Generates new settlement instructions for novations
    4. Creates close-out settlements for terminations
    5. Tracks the relationship between lifecycle events and settlements
    """

    def __init__(
        self,
        settlement_generator: Optional[SettlementInstructionGenerator] = None,
        config: Optional[LifecycleSettlementConfig] = None,
    ):
        """Initialize lifecycle settlement mapper.

        Args:
            settlement_generator: Settlement instruction generator
            config: Mapper configuration
        """
        self.generator = settlement_generator or SettlementInstructionGenerator()
        self.config = config or LifecycleSettlementConfig()

        # Track lifecycle event to settlement mapping
        self._event_impacts: Dict[str, LifecycleSettlementImpact] = {}

        # Track trade to settlement instruction mapping
        self._trade_settlements: Dict[str, List[str]] = {}  # trade_id -> [instruction_ids]

    async def process_lifecycle_event(
        self,
        event: LifecycleEvent,
        trade: Trade,
        parties: Optional[Dict[str, Party]] = None,
    ) -> LifecycleSettlementImpact:
        """Process lifecycle event and update settlements.

        Args:
            event: Lifecycle event
            trade: Trade after lifecycle event applied
            parties: Party information (buyer, seller, new_counterparty)

        Returns:
            Impact assessment with actions taken
        """
        # Analyze impact
        impact = self._analyze_settlement_impact(event, trade)

        # Process based on event type
        try:
            if event.event_type == LifecycleEventType.AMENDMENT:
                if self.config.auto_process_amendments:
                    await self._process_amendment(event, trade, impact)

            elif event.event_type == LifecycleEventType.NOVATION:
                if self.config.auto_process_novations:
                    await self._process_novation(event, trade, impact, parties)

            elif event.event_type == LifecycleEventType.TERMINATION:
                if self.config.auto_process_terminations:
                    await self._process_termination(event, trade, impact, parties)

            elif event.event_type == LifecycleEventType.PARTIAL_TERMINATION:
                if self.config.auto_process_terminations:
                    await self._process_partial_termination(event, trade, impact, parties)

            impact.processed = True
            impact.processed_time = datetime.utcnow()

            # Notify callbacks
            await self._notify_impact(impact)

        except Exception as e:
            impact.error = str(e)
            impact.processed = False

        # Store impact
        self._event_impacts[event.event_id] = impact

        return impact

    def _analyze_settlement_impact(
        self,
        event: LifecycleEvent,
        trade: Trade,
    ) -> LifecycleSettlementImpact:
        """Analyze settlement impact of lifecycle event.

        Args:
            event: Lifecycle event
            trade: Trade after event

        Returns:
            Settlement impact assessment
        """
        impact = LifecycleSettlementImpact(
            event_id=event.event_id,
            event_type=event.event_type,
            trade_id=event.trade_id,
            event_date=event.event_date,
            settlement_action=SettlementAction.NO_ACTION,
        )

        # Get existing settlement instructions for this trade
        existing_instructions = self._trade_settlements.get(trade.id, [])
        impact.affected_instructions = existing_instructions.copy()

        # Determine action based on event type
        if event.event_type == LifecycleEventType.AMENDMENT:
            impact.settlement_action = self._determine_amendment_action(event, trade)

        elif event.event_type == LifecycleEventType.NOVATION:
            impact.settlement_action = SettlementAction.GENERATE_NEW
            impact.new_instructions_required = 1

        elif event.event_type == LifecycleEventType.TERMINATION:
            impact.settlement_action = SettlementAction.GENERATE_CLOSEOUT
            impact.new_instructions_required = 1
            # Estimate close-out amount (would be calculated properly in practice)
            impact.closeout_amount = Decimal(str(trade.notional or 0)) * Decimal("0.02")  # 2% estimate
            impact.closeout_date = event.effective_date + timedelta(
                days=self.config.termination_settlement_days
            )

        elif event.event_type == LifecycleEventType.PARTIAL_TERMINATION:
            impact.settlement_action = SettlementAction.UPDATE
            if existing_instructions:
                impact.affected_instructions = existing_instructions

        return impact

    def _determine_amendment_action(
        self,
        event: LifecycleEvent,
        trade: Trade,
    ) -> SettlementAction:
        """Determine settlement action for amendment.

        Args:
            event: Amendment event
            trade: Amended trade

        Returns:
            Settlement action
        """
        # Check which fields were amended
        changed_fields = set(event.changes.keys())

        # Fields that affect settlement amounts
        settlement_affecting_fields = {
            "notional", "fixed_rate", "spread", "maturity_date",
            "currency", "settlement_type"
        }

        if changed_fields & settlement_affecting_fields:
            return SettlementAction.UPDATE

        # Fields that don't affect settlement
        return SettlementAction.NO_ACTION

    async def _process_amendment(
        self,
        event: LifecycleEvent,
        trade: Trade,
        impact: LifecycleSettlementImpact,
    ):
        """Process amendment settlement updates.

        Args:
            event: Amendment event
            trade: Amended trade
            impact: Impact assessment
        """
        if impact.settlement_action != SettlementAction.UPDATE:
            return

        # Get affected instructions
        for instruction_id in impact.affected_instructions:
            instruction = self.generator.instructions.get(instruction_id)
            if not instruction:
                continue

            # Only update pending/instructed settlements
            if instruction.status not in [
                SettlementStatus.PENDING,
                SettlementStatus.INSTRUCTED
            ]:
                continue

            # Update settlement amounts based on amended trade
            changes_made = []

            # Update notional in cash flows
            if "notional" in event.changes and instruction.cash_flows:
                new_notional = Decimal(str(event.changes["notional"]))
                for cash_flow in instruction.cash_flows:
                    old_amount = cash_flow.amount
                    cash_flow.amount = new_notional
                    changes_made.append(f"Cash flow updated: {old_amount} → {new_notional}")

            # Update maturity date
            if "maturity_date" in event.changes:
                new_maturity = event.changes["maturity_date"]
                if hasattr(new_maturity, 'date'):
                    new_maturity = new_maturity.date()
                instruction.settlement_date = new_maturity
                instruction.intended_settlement_date = new_maturity
                changes_made.append(f"Settlement date updated to {new_maturity}")

            # Track changes in impact
            impact.settlement_changes[instruction_id] = changes_made

    async def _process_novation(
        self,
        event: LifecycleEvent,
        trade: Trade,
        impact: LifecycleSettlementImpact,
        parties: Optional[Dict[str, Party]],
    ):
        """Process novation settlement generation.

        Args:
            event: Novation event
            trade: Novated trade
            impact: Impact assessment
            parties: Party information
        """
        if not parties or "new_counterparty" not in parties:
            impact.error = "New counterparty party information required for novation"
            return

        # Cancel existing settlement instructions
        for instruction_id in impact.affected_instructions:
            instruction = self.generator.instructions.get(instruction_id)
            if instruction and instruction.status in [
                SettlementStatus.PENDING,
                SettlementStatus.INSTRUCTED
            ]:
                instruction.status = SettlementStatus.CANCELLED
                instruction.metadata["cancelled_reason"] = f"Trade novated: {event.event_id}"

        # Generate new settlement instruction with new counterparty
        settlement_date = event.effective_date + timedelta(
            days=self.config.novation_settlement_days
        )

        buyer_party = parties.get("buyer", parties.get("new_counterparty"))
        seller_party = parties.get("seller")

        if not buyer_party or not seller_party:
            impact.error = "Both buyer and seller party information required"
            return

        new_instruction = self.generator.generate_instruction(
            trade_id=trade.id,
            trade_date=trade.trade_date.date() if hasattr(trade.trade_date, 'date') else trade.trade_date,
            settlement_date=settlement_date.date() if hasattr(settlement_date, 'date') else settlement_date,
            deliverer=seller_party,
            receiver=buyer_party,
            settlement_type=SettlementType.DVP,
            settlement_method=SettlementMethod.CCP,
        )

        # Add cash flow for novation settlement
        if trade.notional:
            cash_flow = CashFlow(
                amount=Decimal(str(trade.notional)),
                currency=trade.currency or "USD",
                direction="pay",
                payment_date=settlement_date.date() if hasattr(settlement_date, 'date') else settlement_date,
                value_date=settlement_date.date() if hasattr(settlement_date, 'date') else settlement_date,
                payer_account=buyer_party.member_id or "ACCT_BUYER",
                receiver_account=seller_party.member_id or "ACCT_SELLER",
                payment_reference=f"Novation settlement for trade {trade.id}",
            )
            self.generator.add_cash_flow(new_instruction.instruction_id, cash_flow)

        # Track new instruction
        if trade.id not in self._trade_settlements:
            self._trade_settlements[trade.id] = []
        self._trade_settlements[trade.id].append(new_instruction.instruction_id)

        impact.settlement_changes["new_instruction"] = new_instruction.instruction_id

    async def _process_termination(
        self,
        event: LifecycleEvent,
        trade: Trade,
        impact: LifecycleSettlementImpact,
        parties: Optional[Dict[str, Party]],
    ):
        """Process termination close-out settlement.

        Args:
            event: Termination event
            trade: Terminated trade
            impact: Impact assessment
            parties: Party information
        """
        # Cancel existing settlement instructions
        for instruction_id in impact.affected_instructions:
            instruction = self.generator.instructions.get(instruction_id)
            if instruction and instruction.status in [
                SettlementStatus.PENDING,
                SettlementStatus.INSTRUCTED
            ]:
                instruction.status = SettlementStatus.CANCELLED
                instruction.metadata["cancelled_reason"] = f"Trade terminated: {event.event_id}"

        # Generate close-out settlement
        if not parties or "buyer" not in parties or "seller" not in parties:
            impact.error = "Party information required for termination settlement"
            return

        buyer_party = parties["buyer"]
        seller_party = parties["seller"]

        closeout_date = impact.closeout_date or (
            event.effective_date + timedelta(days=self.config.termination_settlement_days)
        )

        closeout_instruction = self.generator.generate_instruction(
            trade_id=trade.id,
            trade_date=event.event_date,
            settlement_date=closeout_date.date() if hasattr(closeout_date, 'date') else closeout_date,
            deliverer=seller_party,
            receiver=buyer_party,
            settlement_type=SettlementType.DVP,
            settlement_method=SettlementMethod.CCP,
        )

        # Add close-out cash flow
        if impact.closeout_amount:
            # Determine payer based on MTM (simplified)
            payer = buyer_party if impact.closeout_amount > 0 else seller_party
            receiver = seller_party if impact.closeout_amount > 0 else buyer_party

            cash_flow = CashFlow(
                amount=abs(impact.closeout_amount),
                currency=trade.currency or "USD",
                direction="pay",
                payment_date=closeout_date.date() if hasattr(closeout_date, 'date') else closeout_date,
                value_date=closeout_date.date() if hasattr(closeout_date, 'date') else closeout_date,
                payer_account=payer.member_id or "ACCT_PAYER",
                receiver_account=receiver.member_id or "ACCT_RECEIVER",
                payment_reference=f"Close-out settlement for trade {trade.id}",
            )
            self.generator.add_cash_flow(closeout_instruction.instruction_id, cash_flow)

        # Track close-out instruction
        if trade.id not in self._trade_settlements:
            self._trade_settlements[trade.id] = []
        self._trade_settlements[trade.id].append(closeout_instruction.instruction_id)

        impact.settlement_changes["closeout_instruction"] = closeout_instruction.instruction_id
        impact.closeout_payer = payer.party_id

    async def _process_partial_termination(
        self,
        event: LifecycleEvent,
        trade: Trade,
        impact: LifecycleSettlementImpact,
        parties: Optional[Dict[str, Party]],
    ):
        """Process partial termination settlement updates.

        Args:
            event: Partial termination event
            trade: Partially terminated trade
            impact: Impact assessment
            parties: Party information
        """
        # Update existing settlement instructions to reflect reduced notional
        partial_amount = event.changes.get("partial_termination_amount", 0)
        remaining_notional = Decimal(str(trade.notional or 0))

        for instruction_id in impact.affected_instructions:
            instruction = self.generator.instructions.get(instruction_id)
            if not instruction:
                continue

            if instruction.status in [SettlementStatus.PENDING, SettlementStatus.INSTRUCTED]:
                # Update cash flows to reflect remaining notional
                for cash_flow in instruction.cash_flows:
                    cash_flow.amount = remaining_notional
                    cash_flow.metadata["partial_termination"] = {
                        "event_id": event.event_id,
                        "terminated_amount": float(partial_amount),
                    }

        # Generate close-out settlement for terminated portion
        if parties and partial_amount > 0:
            await self._generate_partial_closeout(
                event, trade, Decimal(str(partial_amount)), parties, impact
            )

    async def _generate_partial_closeout(
        self,
        event: LifecycleEvent,
        trade: Trade,
        partial_amount: Decimal,
        parties: Dict[str, Party],
        impact: LifecycleSettlementImpact,
    ):
        """Generate close-out settlement for partial termination.

        Args:
            event: Lifecycle event
            trade: Trade
            partial_amount: Terminated amount
            parties: Party information
            impact: Impact assessment
        """
        buyer_party = parties.get("buyer")
        seller_party = parties.get("seller")

        if not buyer_party or not seller_party:
            return

        closeout_date = event.effective_date + timedelta(
            days=self.config.termination_settlement_days
        )

        partial_instruction = self.generator.generate_instruction(
            trade_id=trade.id,
            trade_date=event.event_date,
            settlement_date=closeout_date.date() if hasattr(closeout_date, 'date') else closeout_date,
            deliverer=seller_party,
            receiver=buyer_party,
            settlement_type=SettlementType.DVP,
            settlement_method=SettlementMethod.CCP,
        )

        # Add partial close-out cash flow
        cash_flow = CashFlow(
            amount=partial_amount,
            currency=trade.currency or "USD",
            direction="pay",
            payment_date=closeout_date.date() if hasattr(closeout_date, 'date') else closeout_date,
            value_date=closeout_date.date() if hasattr(closeout_date, 'date') else closeout_date,
            payer_account=buyer_party.member_id or "ACCT_BUYER",
            receiver_account=seller_party.member_id or "ACCT_SELLER",
            payment_reference=f"Partial close-out settlement for trade {trade.id}",
        )
        self.generator.add_cash_flow(partial_instruction.instruction_id, cash_flow)

        # Track instruction
        if trade.id not in self._trade_settlements:
            self._trade_settlements[trade.id] = []
        self._trade_settlements[trade.id].append(partial_instruction.instruction_id)

        impact.settlement_changes["partial_closeout_instruction"] = partial_instruction.instruction_id

    async def _notify_impact(self, impact: LifecycleSettlementImpact):
        """Send notifications for settlement impact.

        Args:
            impact: Settlement impact
        """
        for callback in self.config.notification_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(impact)
                else:
                    callback(impact)
            except Exception:
                pass  # Don't fail on notification errors

    def register_settlement_instruction(
        self,
        trade_id: str,
        instruction_id: str,
    ):
        """Register a settlement instruction for a trade.

        Args:
            trade_id: Trade identifier
            instruction_id: Settlement instruction identifier
        """
        if trade_id not in self._trade_settlements:
            self._trade_settlements[trade_id] = []
        if instruction_id not in self._trade_settlements[trade_id]:
            self._trade_settlements[trade_id].append(instruction_id)

    def get_trade_settlement_instructions(self, trade_id: str) -> List[str]:
        """Get all settlement instructions for a trade.

        Args:
            trade_id: Trade identifier

        Returns:
            List of settlement instruction IDs
        """
        return self._trade_settlements.get(trade_id, [])

    def get_event_impact(self, event_id: str) -> Optional[LifecycleSettlementImpact]:
        """Get settlement impact for a lifecycle event.

        Args:
            event_id: Event identifier

        Returns:
            Settlement impact or None
        """
        return self._event_impacts.get(event_id)

    def get_impacts_by_trade(self, trade_id: str) -> List[LifecycleSettlementImpact]:
        """Get all settlement impacts for a trade.

        Args:
            trade_id: Trade identifier

        Returns:
            List of settlement impacts
        """
        return [
            impact for impact in self._event_impacts.values()
            if impact.trade_id == trade_id
        ]

    def get_unprocessed_impacts(self) -> List[LifecycleSettlementImpact]:
        """Get all unprocessed settlement impacts.

        Returns:
            List of unprocessed impacts
        """
        return [
            impact for impact in self._event_impacts.values()
            if not impact.processed
        ]


__all__ = [
    "LifecycleSettlementMapper",
    "LifecycleSettlementConfig",
    "LifecycleSettlementImpact",
    "SettlementAction",
]
