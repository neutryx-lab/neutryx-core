"""RFQ to CCP workflow integration.

This module integrates the RFQ auction system with CCP clearing infrastructure:
1. Execute RFQ auction
2. Route winning trades to optimal CCP
3. Generate settlement instructions
4. Track margins and reconcile positions
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .base import Party, ProductType, Trade, TradeEconomics, TradeSubmissionResponse
from .rfq import (
    RFQ,
    RFQManager,
    RFQStatus,
    Quote,
    AuctionResult,
    AuctionType,
    OrderSide,
)
from .ccp_router import CCPRouter, RoutingDecision, RoutingStrategy
from .settlement_workflow import AutomaticSettlementService, SettlementWorkflow
from .margin_aggregator import MarginAggregationService
from .reconciliation import CCPReconciliationEngine


class RFQExecutionStatus(str, Enum):
    """Status of RFQ execution workflow."""
    AUCTION_PENDING = "auction_pending"
    AUCTION_EXECUTED = "auction_executed"
    CCP_ROUTING = "ccp_routing"
    CCP_SUBMITTED = "ccp_submitted"
    SETTLEMENT_GENERATED = "settlement_generated"
    COMPLETED = "completed"
    FAILED = "failed"


class TradeAllocation(BaseModel):
    """Trade allocation from auction result."""

    allocation_id: str = Field(..., description="Allocation identifier")
    rfq_id: str = Field(..., description="Source RFQ ID")
    quote_id: str = Field(..., description="Source quote ID")

    # Trade details
    trade_id: Optional[str] = Field(None, description="Generated trade ID")
    counterparty: Party = Field(..., description="Counterparty (quoter)")
    quantity: Decimal = Field(..., description="Allocated quantity")
    price: Decimal = Field(..., description="Execution price")

    # CCP workflow
    selected_ccp: Optional[str] = Field(None, description="Selected CCP")
    ccp_trade_id: Optional[str] = Field(None, description="CCP trade ID")
    ccp_status: Optional[str] = Field(None, description="CCP submission status")

    # Settlement
    settlement_instruction_id: Optional[str] = Field(None, description="Settlement instruction ID")
    settlement_date: Optional[date] = Field(None, description="Settlement date")

    metadata: Dict[str, Any] = Field(default_factory=dict)


class RFQExecutionResult(BaseModel):
    """Complete RFQ execution result with CCP workflow."""

    execution_id: str = Field(..., description="Execution identifier")
    rfq_id: str = Field(..., description="RFQ identifier")
    status: RFQExecutionStatus = Field(..., description="Execution status")

    # Auction results
    auction_result: Optional[AuctionResult] = Field(None, description="Auction result")

    # Trade allocations
    allocations: List[TradeAllocation] = Field(default_factory=list, description="Trade allocations")

    # CCP workflow results
    routing_decisions: Dict[str, RoutingDecision] = Field(
        default_factory=dict,
        description="CCP routing decisions by allocation ID"
    )
    ccp_submissions: Dict[str, TradeSubmissionResponse] = Field(
        default_factory=dict,
        description="CCP submission responses by allocation ID"
    )
    settlement_workflows: Dict[str, SettlementWorkflow] = Field(
        default_factory=dict,
        description="Settlement workflows by allocation ID"
    )

    # Summary
    total_allocations: int = Field(default=0, description="Number of allocations")
    successful_submissions: int = Field(default=0, description="Successful CCP submissions")
    failed_submissions: int = Field(default=0, description="Failed CCP submissions")

    # Timing
    execution_start: datetime = Field(default_factory=datetime.utcnow)
    execution_end: Optional[datetime] = Field(None)
    total_duration_ms: Optional[float] = Field(None)

    # Errors
    errors: List[str] = Field(default_factory=list, description="Execution errors")

    metadata: Dict[str, Any] = Field(default_factory=dict)

    def calculate_summary(self):
        """Calculate summary statistics."""
        self.total_allocations = len(self.allocations)
        self.successful_submissions = sum(
            1 for alloc in self.allocations
            if alloc.ccp_status == "ACCEPTED"
        )
        self.failed_submissions = self.total_allocations - self.successful_submissions


@dataclass
class RFQCCPIntegrationConfig:
    """Configuration for RFQ-CCP integration."""

    # CCP routing
    default_routing_strategy: RoutingStrategy = RoutingStrategy.LOWEST_MARGIN
    auto_route_to_ccp: bool = True

    # Settlement
    auto_generate_settlement: bool = True
    settlement_cycle_days: int = 2

    # Error handling
    continue_on_ccp_rejection: bool = True
    max_ccp_retries: int = 2

    # Reconciliation
    auto_reconcile_after_execution: bool = False

    # Notifications
    notification_enabled: bool = True


class RFQCCPIntegrationService:
    """Service integrating RFQ auctions with CCP clearing workflow.

    This service provides end-to-end automation:
    1. Execute RFQ auction
    2. Create trades from allocations
    3. Route trades to optimal CCPs
    4. Submit to CCPs
    5. Generate settlement instructions
    6. Track in reconciliation
    """

    def __init__(
        self,
        rfq_manager: RFQManager,
        ccp_router: CCPRouter,
        settlement_service: AutomaticSettlementService,
        margin_service: Optional[MarginAggregationService] = None,
        recon_engine: Optional[CCPReconciliationEngine] = None,
        config: Optional[RFQCCPIntegrationConfig] = None,
    ):
        """Initialize RFQ-CCP integration service.

        Args:
            rfq_manager: RFQ manager instance
            ccp_router: CCP router instance
            settlement_service: Settlement workflow service
            margin_service: Optional margin aggregation service
            recon_engine: Optional reconciliation engine
            config: Integration configuration
        """
        self.rfq_manager = rfq_manager
        self.ccp_router = ccp_router
        self.settlement_service = settlement_service
        self.margin_service = margin_service
        self.recon_engine = recon_engine
        self.config = config or RFQCCPIntegrationConfig()

        # Execution tracking
        self.executions: Dict[str, RFQExecutionResult] = {}

    async def execute_rfq_workflow(
        self,
        rfq_id: str,
        requester_party: Party,
    ) -> RFQExecutionResult:
        """Execute complete RFQ-to-CCP workflow.

        Args:
            rfq_id: RFQ identifier
            requester_party: Party requesting the RFQ

        Returns:
            Complete execution result
        """
        execution_id = f"EXEC-{rfq_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        result = RFQExecutionResult(
            execution_id=execution_id,
            rfq_id=rfq_id,
            status=RFQExecutionStatus.AUCTION_PENDING,
        )

        try:
            # Step 1: Execute auction
            auction_result = self.rfq_manager.execute_auction(rfq_id)
            result.auction_result = auction_result
            result.status = RFQExecutionStatus.AUCTION_EXECUTED

            # Step 2: Create trade allocations
            rfq = self.rfq_manager.get_rfq(rfq_id)
            if not rfq:
                raise ValueError(f"RFQ {rfq_id} not found")

            quotes = self.rfq_manager.get_quotes(rfq_id)
            allocations = self._create_allocations(rfq, auction_result, quotes, requester_party)
            result.allocations = allocations

            # Step 3: Route and submit to CCPs
            if self.config.auto_route_to_ccp:
                result.status = RFQExecutionStatus.CCP_ROUTING
                await self._route_and_submit_allocations(rfq, allocations, result)

            # Step 4: Generate settlement instructions
            if self.config.auto_generate_settlement:
                result.status = RFQExecutionStatus.SETTLEMENT_GENERATED
                await self._generate_settlements(rfq, allocations, result)

            # Step 5: Reconcile (if enabled)
            if self.config.auto_reconcile_after_execution and self.recon_engine:
                await self._reconcile_executions(result)

            # Finalize
            result.status = RFQExecutionStatus.COMPLETED
            result.execution_end = datetime.utcnow()
            result.total_duration_ms = (
                result.execution_end - result.execution_start
            ).total_seconds() * 1000
            result.calculate_summary()

        except Exception as e:
            result.status = RFQExecutionStatus.FAILED
            result.errors.append(str(e))
            result.execution_end = datetime.utcnow()

        # Store result
        self.executions[execution_id] = result

        return result

    def _create_allocations(
        self,
        rfq: RFQ,
        auction_result: AuctionResult,
        quotes: List[Quote],
        requester_party: Party,
    ) -> List[TradeAllocation]:
        """Create trade allocations from auction result.

        Args:
            rfq: RFQ object
            auction_result: Auction result
            quotes: List of quotes
            requester_party: Party requesting the RFQ

        Returns:
            List of trade allocations
        """
        allocations = []
        quote_map = {q.quote_id: q for q in quotes}

        for alloc_data in auction_result.allocations:
            quote_id = alloc_data.get("quote_id")
            quote = quote_map.get(quote_id)

            if not quote:
                continue

            allocation = TradeAllocation(
                allocation_id=f"ALLOC-{rfq.rfq_id}-{len(allocations) + 1:03d}",
                rfq_id=rfq.rfq_id,
                quote_id=quote_id,
                trade_id=f"TRD-{rfq.rfq_id}-{len(allocations) + 1:03d}",
                counterparty=quote.quoter,
                quantity=Decimal(str(alloc_data.get("quantity", 0))),
                price=Decimal(str(alloc_data.get("price", 0))),
            )

            allocations.append(allocation)

        return allocations

    async def _route_and_submit_allocations(
        self,
        rfq: RFQ,
        allocations: List[TradeAllocation],
        result: RFQExecutionResult,
    ):
        """Route allocations to CCPs and submit.

        Args:
            rfq: RFQ object
            allocations: Trade allocations
            result: Execution result to update
        """
        for allocation in allocations:
            try:
                # Create trade object
                trade = self._create_trade_from_allocation(rfq, allocation)

                # Route to optimal CCP
                routing_decision, ccp_response = await self.ccp_router.route_and_submit(
                    trade,
                    strategy=self.config.default_routing_strategy,
                )

                # Update allocation
                allocation.selected_ccp = routing_decision.selected_ccp
                allocation.ccp_trade_id = ccp_response.ccp_trade_id
                allocation.ccp_status = ccp_response.status.value

                # Store results
                result.routing_decisions[allocation.allocation_id] = routing_decision
                result.ccp_submissions[allocation.allocation_id] = ccp_response

            except Exception as e:
                allocation.ccp_status = "FAILED"
                result.errors.append(
                    f"Failed to route allocation {allocation.allocation_id}: {str(e)}"
                )

                if not self.config.continue_on_ccp_rejection:
                    raise

        result.status = RFQExecutionStatus.CCP_SUBMITTED

    async def _generate_settlements(
        self,
        rfq: RFQ,
        allocations: List[TradeAllocation],
        result: RFQExecutionResult,
    ):
        """Generate settlement instructions for allocations.

        Args:
            rfq: RFQ object
            allocations: Trade allocations
            result: Execution result to update
        """
        for allocation in allocations:
            if allocation.ccp_status != "ACCEPTED":
                continue

            try:
                # Get trade and CCP response
                trade = self._create_trade_from_allocation(rfq, allocation)
                ccp_response = result.ccp_submissions.get(allocation.allocation_id)

                if not ccp_response:
                    continue

                # Determine parties (RFQ requester and quote provider)
                if rfq.side == OrderSide.BUY:
                    buyer_party = rfq.requester
                    seller_party = allocation.counterparty
                else:
                    buyer_party = allocation.counterparty
                    seller_party = rfq.requester

                # Generate settlement instruction
                workflow = await self.settlement_service.process_ccp_confirmation(
                    trade=trade,
                    ccp_response=ccp_response,
                    buyer_party=buyer_party,
                    seller_party=seller_party,
                )

                # Update allocation
                allocation.settlement_instruction_id = workflow.instruction_id
                allocation.settlement_date = workflow.settlement_date

                # Store workflow
                result.settlement_workflows[allocation.allocation_id] = workflow

            except Exception as e:
                result.errors.append(
                    f"Failed to generate settlement for allocation {allocation.allocation_id}: {str(e)}"
                )

    async def _reconcile_executions(self, result: RFQExecutionResult):
        """Reconcile executed trades with CCPs.

        Args:
            result: Execution result
        """
        if not self.recon_engine:
            return

        # Build internal positions from allocations
        internal_positions = []
        for allocation in result.allocations:
            if allocation.ccp_status == "ACCEPTED":
                internal_positions.append({
                    "trade_id": allocation.trade_id,
                    "notional": float(allocation.quantity),
                    "price": float(allocation.price),
                    "status": "ACTIVE",
                })

        # Reconcile with each CCP
        ccp_names = {alloc.selected_ccp for alloc in result.allocations if alloc.selected_ccp}

        for ccp_name in ccp_names:
            try:
                ccp_positions = [
                    pos for pos in internal_positions
                    if any(
                        alloc.selected_ccp == ccp_name and alloc.trade_id == pos["trade_id"]
                        for alloc in result.allocations
                    )
                ]

                recon_result = await self.recon_engine.reconcile_positions(
                    ccp_name=ccp_name,
                    internal_positions=ccp_positions,
                )

                result.metadata[f"reconciliation_{ccp_name}"] = {
                    "total_breaks": recon_result.total_breaks,
                    "reconciliation_passed": recon_result.reconciliation_passed,
                }

            except Exception as e:
                result.errors.append(f"Reconciliation failed for {ccp_name}: {str(e)}")

    def _create_trade_from_allocation(
        self,
        rfq: RFQ,
        allocation: TradeAllocation,
    ) -> Trade:
        """Create Trade object from allocation.

        Args:
            rfq: RFQ object
            allocation: Trade allocation

        Returns:
            Trade object
        """
        spec = rfq.specification

        # Determine buyer/seller based on RFQ side
        if rfq.side == OrderSide.BUY:
            buyer = rfq.requester
            seller = allocation.counterparty
        else:
            buyer = allocation.counterparty
            seller = rfq.requester

        economics = TradeEconomics(
            notional=allocation.quantity,
            currency=spec.currency,
            fixed_rate=allocation.price if spec.fixed_rate else None,
            strike=spec.strike,
        )

        trade = Trade(
            trade_id=allocation.trade_id or f"TRD-{allocation.allocation_id}",
            product_type=spec.product_type,
            trade_date=datetime.utcnow(),
            effective_date=spec.effective_date,
            maturity_date=spec.maturity_date,
            buyer=buyer,
            seller=seller,
            economics=economics,
            uti=f"UTI-{allocation.allocation_id}",
            metadata={
                "rfq_id": rfq.rfq_id,
                "allocation_id": allocation.allocation_id,
                "quote_id": allocation.quote_id,
                "auction_type": rfq.auction_type.value,
            },
        )

        return trade

    async def get_margin_impact(
        self,
        execution_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get margin impact of RFQ execution.

        Args:
            execution_id: Execution identifier

        Returns:
            Margin impact summary
        """
        if not self.margin_service:
            return None

        result = self.executions.get(execution_id)
        if not result:
            return None

        # Refresh margins after execution
        report = await self.margin_service.generate_aggregated_report()

        # Calculate margin by CCP
        margin_by_ccp = {}
        for ccp_name, req in report.ccp_requirements.items():
            margin_by_ccp[ccp_name] = {
                "initial_margin": float(req.initial_margin),
                "variation_margin": float(req.variation_margin),
                "total_margin": float(req.total_margin_required),
            }

        return {
            "execution_id": execution_id,
            "total_portfolio_margin": float(report.total_margin_required),
            "margin_by_ccp": margin_by_ccp,
            "margin_deficit": float(report.total_margin_deficit),
        }

    def get_execution_result(self, execution_id: str) -> Optional[RFQExecutionResult]:
        """Get execution result by ID."""
        return self.executions.get(execution_id)

    def get_executions_by_rfq(self, rfq_id: str) -> List[RFQExecutionResult]:
        """Get all executions for an RFQ."""
        return [
            result for result in self.executions.values()
            if result.rfq_id == rfq_id
        ]

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics.

        Returns:
            Dictionary with statistics
        """
        total_executions = len(self.executions)
        completed = sum(
            1 for r in self.executions.values()
            if r.status == RFQExecutionStatus.COMPLETED
        )
        failed = sum(
            1 for r in self.executions.values()
            if r.status == RFQExecutionStatus.FAILED
        )

        total_allocations = sum(r.total_allocations for r in self.executions.values())
        successful_submissions = sum(
            r.successful_submissions for r in self.executions.values()
        )

        return {
            "total_executions": total_executions,
            "completed_executions": completed,
            "failed_executions": failed,
            "success_rate": (completed / total_executions * 100) if total_executions > 0 else 0,
            "total_allocations": total_allocations,
            "successful_ccp_submissions": successful_submissions,
            "ccp_submission_rate": (
                successful_submissions / total_allocations * 100
                if total_allocations > 0 else 0
            ),
        }


__all__ = [
    "RFQCCPIntegrationService",
    "RFQCCPIntegrationConfig",
    "RFQExecutionResult",
    "RFQExecutionStatus",
    "TradeAllocation",
]
