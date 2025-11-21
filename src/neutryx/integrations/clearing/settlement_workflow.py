"""Automatic settlement workflow orchestration.

This module provides end-to-end automation for trade settlement:
1. Automatically generate settlement instructions from CCP confirmations
2. Route instructions to appropriate settlement systems (CLS, Euroclear, etc.)
3. Track settlement status and handle exceptions
4. Manage settlement lifecycle from instruction to settlement
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from .base import Party, Trade, TradeSubmissionResponse, TradeStatus
from .settlement_instructions import (
    CashFlow,
    CLSInstruction,
    EuroclearInstruction,
    FailReason,
    SecuritiesMovement,
    SettlementInstruction,
    SettlementInstructionGenerator,
    SettlementMethod,
    SettlementStatus,
    SettlementType,
    SwiftSettlementMessage,
)


class SettlementRoutingStrategy(str, Enum):
    """Strategy for routing settlement instructions."""
    AUTO = "auto"  # Automatically determine based on trade characteristics
    CLS = "cls"  # Force CLS settlement
    EUROCLEAR = "euroclear"  # Force Euroclear settlement
    CCP = "ccp"  # CCP-managed settlement
    BILATERAL = "bilateral"  # Bilateral settlement


class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    INITIATED = "initiated"
    INSTRUCTION_GENERATED = "instruction_generated"
    INSTRUCTION_SENT = "instruction_sent"
    AWAITING_MATCH = "awaiting_match"
    MATCHED = "matched"
    SETTLING = "settling"
    SETTLED = "settled"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SettlementWorkflowConfig:
    """Configuration for settlement workflow."""

    # Default settlement cycles (T+X)
    default_settlement_cycle: int = 2  # T+2 for most markets

    # Product-specific settlement cycles
    settlement_cycles: Dict[str, int] = None  # {"IRS": 2, "CDS": 1}

    # Automatic retry settings
    auto_retry_enabled: bool = True
    max_retry_attempts: int = 3
    retry_delay_minutes: int = 30

    # Routing preferences
    default_routing_strategy: SettlementRoutingStrategy = SettlementRoutingStrategy.AUTO
    cls_enabled: bool = True
    euroclear_enabled: bool = True

    # CLS routing rules
    cls_eligible_currencies: set = None  # {"USD", "EUR", "GBP", "JPY"}

    # Euroclear routing rules
    euroclear_eligible_securities: set = None  # Securities clearable via Euroclear

    # Notification settings
    notification_enabled: bool = True
    notification_callbacks: List[Callable] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.settlement_cycles is None:
            self.settlement_cycles = {
                "IRS": 2,
                "CDS": 1,
                "FX_FORWARD": 2,
                "FX_SWAP": 2,
                "EQUITY_OPTION": 1,
                "REPO": 0,
            }

        if self.cls_eligible_currencies is None:
            # CLS settles 18 major currencies
            self.cls_eligible_currencies = {
                "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD",
                "SEK", "NOK", "DKK", "SGD", "HKD", "KRW", "ZAR", "ILS", "MXN", "CZK"
            }

        if self.notification_callbacks is None:
            self.notification_callbacks = []


class SettlementWorkflowEvent(BaseModel):
    """Event in settlement workflow."""

    event_id: str = Field(..., description="Event identifier")
    workflow_id: str = Field(..., description="Related workflow ID")
    event_type: str = Field(..., description="Event type")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: WorkflowStatus = Field(..., description="Workflow status after event")
    details: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = Field(None, description="Error message if applicable")


class SettlementWorkflow(BaseModel):
    """Settlement workflow instance."""

    workflow_id: str = Field(..., description="Workflow identifier")
    trade_id: str = Field(..., description="Related trade ID")
    ccp_submission_id: Optional[str] = Field(None, description="CCP submission ID")

    # Status tracking
    status: WorkflowStatus = Field(default=WorkflowStatus.INITIATED)
    created_time: datetime = Field(default_factory=datetime.utcnow)
    updated_time: datetime = Field(default_factory=datetime.utcnow)

    # Settlement instruction
    instruction_id: Optional[str] = Field(None, description="Generated instruction ID")
    settlement_date: Optional[date] = Field(None, description="Calculated settlement date")
    settlement_method: Optional[SettlementMethod] = Field(None, description="Settlement method")

    # Event log
    events: List[SettlementWorkflowEvent] = Field(default_factory=list)

    # Retry tracking
    retry_count: int = Field(default=0)
    last_retry_time: Optional[datetime] = Field(None)

    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_event(
        self,
        event_type: str,
        status: WorkflowStatus,
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ):
        """Add event to workflow."""
        event = SettlementWorkflowEvent(
            event_id=f"{self.workflow_id}-E{len(self.events) + 1:03d}",
            workflow_id=self.workflow_id,
            event_type=event_type,
            status=status,
            details=details or {},
            error=error,
        )
        self.events.append(event)
        self.status = status
        self.updated_time = datetime.utcnow()


class AutomaticSettlementService:
    """Automatic settlement instruction generation and routing service.

    This service automates the entire settlement workflow:
    1. Monitors CCP confirmations
    2. Generates settlement instructions automatically
    3. Routes to appropriate settlement systems
    4. Tracks settlement status
    5. Handles retries and exceptions
    """

    def __init__(
        self,
        config: Optional[SettlementWorkflowConfig] = None,
        settlement_generator: Optional[SettlementInstructionGenerator] = None,
    ):
        """Initialize automatic settlement service.

        Args:
            config: Workflow configuration
            settlement_generator: Settlement instruction generator
        """
        self.config = config or SettlementWorkflowConfig()
        self.generator = settlement_generator or SettlementInstructionGenerator()

        # Workflow tracking
        self.workflows: Dict[str, SettlementWorkflow] = {}

        # Settlement system connectors (to be injected)
        self.cls_connector: Optional[Any] = None
        self.euroclear_connector: Optional[Any] = None

    def set_cls_connector(self, connector: Any):
        """Set CLS settlement connector."""
        self.cls_connector = connector

    def set_euroclear_connector(self, connector: Any):
        """Set Euroclear settlement connector."""
        self.euroclear_connector = connector

    async def process_ccp_confirmation(
        self,
        trade: Trade,
        ccp_response: TradeSubmissionResponse,
        buyer_party: Party,
        seller_party: Party,
    ) -> SettlementWorkflow:
        """Process CCP confirmation and initiate settlement workflow.

        Args:
            trade: Confirmed trade
            ccp_response: CCP submission response
            buyer_party: Buyer party details
            seller_party: Seller party details

        Returns:
            Settlement workflow instance
        """
        # Create workflow
        workflow_id = f"SW-{trade.trade_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        workflow = SettlementWorkflow(
            workflow_id=workflow_id,
            trade_id=trade.trade_id,
            ccp_submission_id=ccp_response.submission_id,
        )

        workflow.add_event(
            event_type="WORKFLOW_INITIATED",
            status=WorkflowStatus.INITIATED,
            details={"ccp_trade_id": ccp_response.ccp_trade_id},
        )

        self.workflows[workflow_id] = workflow

        # Only proceed if trade is accepted
        if ccp_response.status != TradeStatus.ACCEPTED:
            workflow.add_event(
                event_type="CCP_REJECTION",
                status=WorkflowStatus.FAILED,
                error=f"Trade rejected by CCP: {ccp_response.rejection_reason}",
            )
            return workflow

        try:
            # Step 1: Calculate settlement date
            settlement_date = self._calculate_settlement_date(trade)
            workflow.settlement_date = settlement_date

            # Step 2: Determine settlement routing
            settlement_method, routing_strategy = self._determine_settlement_routing(trade)
            workflow.settlement_method = settlement_method

            workflow.add_event(
                event_type="ROUTING_DETERMINED",
                status=WorkflowStatus.INITIATED,
                details={
                    "settlement_date": settlement_date.isoformat(),
                    "settlement_method": settlement_method.value,
                    "routing_strategy": routing_strategy.value,
                },
            )

            # Step 3: Generate settlement instruction
            instruction = await self._generate_settlement_instruction(
                trade=trade,
                settlement_date=settlement_date,
                settlement_method=settlement_method,
                buyer_party=buyer_party,
                seller_party=seller_party,
            )

            workflow.instruction_id = instruction.instruction_id

            workflow.add_event(
                event_type="INSTRUCTION_GENERATED",
                status=WorkflowStatus.INSTRUCTION_GENERATED,
                details={
                    "instruction_id": instruction.instruction_id,
                    "settlement_type": instruction.settlement_type.value,
                },
            )

            # Step 4: Send instruction to settlement system
            await self._send_settlement_instruction(
                instruction=instruction,
                settlement_method=settlement_method,
                workflow=workflow,
            )

            workflow.add_event(
                event_type="INSTRUCTION_SENT",
                status=WorkflowStatus.INSTRUCTION_SENT,
            )

            # Step 5: Notify callbacks
            await self._notify_workflow_update(workflow)

        except Exception as e:
            workflow.add_event(
                event_type="WORKFLOW_ERROR",
                status=WorkflowStatus.FAILED,
                error=str(e),
            )

        return workflow

    def _calculate_settlement_date(self, trade: Trade) -> date:
        """Calculate settlement date based on product type and market conventions.

        Args:
            trade: Trade to calculate settlement date for

        Returns:
            Settlement date
        """
        # Get settlement cycle for product
        product_type = trade.product_type.value.upper().replace(" ", "_")
        settlement_cycle = self.config.settlement_cycles.get(
            product_type,
            self.config.default_settlement_cycle
        )

        # Calculate T+X settlement date
        settlement_date = trade.trade_date + timedelta(days=settlement_cycle)

        # Adjust for weekends (simplified - should use holiday calendars)
        while settlement_date.weekday() >= 5:  # Saturday or Sunday
            settlement_date += timedelta(days=1)

        return settlement_date.date() if hasattr(settlement_date, 'date') else settlement_date

    def _determine_settlement_routing(
        self,
        trade: Trade,
    ) -> tuple[SettlementMethod, SettlementRoutingStrategy]:
        """Determine settlement method and routing strategy.

        Args:
            trade: Trade to route

        Returns:
            Tuple of (settlement_method, routing_strategy)
        """
        strategy = self.config.default_routing_strategy

        # CLS routing for eligible FX trades
        if strategy == SettlementRoutingStrategy.AUTO:
            if trade.product_type.value in ["fx_forward", "fx_swap"]:
                if trade.economics.currency in self.config.cls_eligible_currencies:
                    if self.config.cls_enabled and self.cls_connector:
                        return SettlementMethod.CCP, SettlementRoutingStrategy.CLS

            # Euroclear for securities settlements
            if trade.product_type.value in ["equity_option", "repo"]:
                if self.config.euroclear_enabled and self.euroclear_connector:
                    return SettlementMethod.ICSD, SettlementRoutingStrategy.EUROCLEAR

        # Default to CCP settlement
        return SettlementMethod.CCP, SettlementRoutingStrategy.CCP

    async def _generate_settlement_instruction(
        self,
        trade: Trade,
        settlement_date: date,
        settlement_method: SettlementMethod,
        buyer_party: Party,
        seller_party: Party,
    ) -> SettlementInstruction:
        """Generate settlement instruction from trade.

        Args:
            trade: Trade to settle
            settlement_date: Settlement date
            settlement_method: Settlement method
            buyer_party: Buyer party
            seller_party: Seller party

        Returns:
            Settlement instruction
        """
        # Determine settlement type (DVP for most products)
        settlement_type = SettlementType.DVP

        # Generate base instruction
        instruction = self.generator.generate_instruction(
            trade_id=trade.trade_id,
            trade_date=trade.trade_date.date() if hasattr(trade.trade_date, 'date') else trade.trade_date,
            settlement_date=settlement_date,
            deliverer=seller_party,
            receiver=buyer_party,
            settlement_type=settlement_type,
            settlement_method=settlement_method,
        )

        # Add cash flows based on trade economics
        cash_flow = CashFlow(
            amount=trade.economics.notional,
            currency=trade.economics.currency,
            direction="pay",
            payment_date=settlement_date,
            value_date=settlement_date,
            payer_account=buyer_party.member_id or "ACCT_BUYER",
            receiver_account=seller_party.member_id or "ACCT_SELLER",
            payer_bank=buyer_party.bic,
            receiver_bank=seller_party.bic,
            payment_reference=f"Settlement for trade {trade.trade_id}",
        )

        self.generator.add_cash_flow(instruction.instruction_id, cash_flow)

        return instruction

    async def _send_settlement_instruction(
        self,
        instruction: SettlementInstruction,
        settlement_method: SettlementMethod,
        workflow: SettlementWorkflow,
    ):
        """Send settlement instruction to appropriate settlement system.

        Args:
            instruction: Settlement instruction
            settlement_method: Settlement method
            workflow: Settlement workflow
        """
        try:
            if settlement_method == SettlementMethod.CCP:
                # CCP-managed settlement
                workflow.metadata["settlement_system"] = "CCP"
                # Update instruction status
                instruction.status = SettlementStatus.INSTRUCTED
                instruction.instructed_time = datetime.utcnow()

            elif settlement_method == SettlementMethod.ICSD and self.euroclear_connector:
                # Send to Euroclear
                euroclear_instr = self._convert_to_euroclear_instruction(instruction)
                # await self.euroclear_connector.submit_instruction(euroclear_instr)
                workflow.metadata["settlement_system"] = "Euroclear"
                workflow.metadata["euroclear_instruction_id"] = euroclear_instr.instruction_id
                instruction.status = SettlementStatus.INSTRUCTED
                instruction.instructed_time = datetime.utcnow()

            else:
                # Default: mark as instructed
                instruction.status = SettlementStatus.INSTRUCTED
                instruction.instructed_time = datetime.utcnow()
                workflow.metadata["settlement_system"] = "Unknown"

        except Exception as e:
            workflow.add_event(
                event_type="INSTRUCTION_SEND_FAILED",
                status=WorkflowStatus.FAILED,
                error=str(e),
            )
            raise

    def _convert_to_euroclear_instruction(
        self,
        instruction: SettlementInstruction
    ) -> EuroclearInstruction:
        """Convert settlement instruction to Euroclear format.

        Args:
            instruction: Settlement instruction

        Returns:
            Euroclear instruction
        """
        return EuroclearInstruction(
            instruction_id=instruction.instruction_id,
            trade_id=instruction.trade_id,
            place_of_settlement="EUROCLEAR",
            participant_account=instruction.deliverer.member_id or "UNKNOWN",
            counterparty_account=instruction.receiver.member_id or "UNKNOWN",
            settlement_date=instruction.settlement_date,
            settlement_type=instruction.settlement_type,
            securities=[],  # Would be populated from securities_movements
            cash_flows=[cf.model_dump() for cf in instruction.cash_flows],
            priority=instruction.priority,
            references={
                "trade_id": instruction.trade_id,
                "instruction_id": instruction.instruction_id,
            },
        )

    async def _notify_workflow_update(self, workflow: SettlementWorkflow):
        """Send notifications for workflow updates.

        Args:
            workflow: Updated workflow
        """
        if not self.config.notification_enabled:
            return

        for callback in self.config.notification_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(workflow)
                else:
                    callback(workflow)
            except Exception as e:
                # Log but don't fail on notification errors
                pass

    async def retry_failed_workflow(self, workflow_id: str) -> SettlementWorkflow:
        """Retry a failed settlement workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Updated workflow

        Raises:
            ValueError: If workflow not found or not in failed state
        """
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")

        if workflow.status != WorkflowStatus.FAILED:
            raise ValueError(f"Workflow {workflow_id} is not in failed state")

        if workflow.retry_count >= self.config.max_retry_attempts:
            raise ValueError(
                f"Workflow {workflow_id} has exceeded max retry attempts "
                f"({self.config.max_retry_attempts})"
            )

        workflow.retry_count += 1
        workflow.last_retry_time = datetime.utcnow()

        workflow.add_event(
            event_type="RETRY_ATTEMPT",
            status=WorkflowStatus.INITIATED,
            details={"retry_count": workflow.retry_count},
        )

        # Implement retry logic here
        # This would re-attempt instruction generation and sending

        return workflow

    def get_workflow(self, workflow_id: str) -> Optional[SettlementWorkflow]:
        """Get workflow by ID."""
        return self.workflows.get(workflow_id)

    def get_workflows_by_trade(self, trade_id: str) -> List[SettlementWorkflow]:
        """Get all workflows for a trade."""
        return [w for w in self.workflows.values() if w.trade_id == trade_id]

    def get_workflows_by_status(self, status: WorkflowStatus) -> List[SettlementWorkflow]:
        """Get workflows by status."""
        return [w for w in self.workflows.values() if w.status == status]

    async def monitor_settlement_status(self, workflow_id: str) -> SettlementWorkflow:
        """Monitor and update settlement status from settlement systems.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Updated workflow
        """
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")

        # Query settlement status from appropriate system
        # This would poll CLS, Euroclear, etc. for status updates

        return workflow

    def get_settlement_statistics(self) -> Dict[str, Any]:
        """Get settlement workflow statistics.

        Returns:
            Dictionary with workflow statistics
        """
        total_workflows = len(self.workflows)
        status_counts = {}

        for status in WorkflowStatus:
            count = len([w for w in self.workflows.values() if w.status == status])
            status_counts[status.value] = count

        return {
            "total_workflows": total_workflows,
            "status_breakdown": status_counts,
            "settlement_rate": (
                status_counts.get(WorkflowStatus.SETTLED.value, 0) / total_workflows * 100
                if total_workflows > 0 else 0
            ),
            "failure_rate": (
                status_counts.get(WorkflowStatus.FAILED.value, 0) / total_workflows * 100
                if total_workflows > 0 else 0
            ),
        }


__all__ = [
    "AutomaticSettlementService",
    "SettlementWorkflow",
    "SettlementWorkflowConfig",
    "SettlementWorkflowEvent",
    "SettlementRoutingStrategy",
    "WorkflowStatus",
]
