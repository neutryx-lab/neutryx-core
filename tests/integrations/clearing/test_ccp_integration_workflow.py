"""Integration tests for end-to-end CCP and settlement workflows.

This test suite validates the complete workflow:
1. CCP routing and trade submission
2. Automatic settlement instruction generation
3. Margin aggregation across CCPs
4. Lifecycle event to settlement mapping
5. Position reconciliation
"""

import pytest
from datetime import datetime, date, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from neutryx.integrations.clearing.base import (
    CCPConfig,
    CCPConnector,
    Trade,
    TradeEconomics,
    Party,
    ProductType,
    TradeStatus,
    TradeSubmissionResponse,
    PositionReport,
)
from neutryx.integrations.clearing.ccp_router import (
    CCPRouter,
    RoutingStrategy,
    MarginQuote,
)
from neutryx.integrations.clearing.settlement_workflow import (
    AutomaticSettlementService,
    SettlementWorkflowConfig,
    WorkflowStatus,
)
from neutryx.integrations.clearing.margin_aggregator import (
    MarginAggregationService,
    MarginAggregatorConfig,
    CCPMarginRequirement,
)
from neutryx.integrations.clearing.lifecycle_settlement_mapper import (
    LifecycleSettlementMapper,
    LifecycleSettlementConfig,
    SettlementAction,
)
from neutryx.integrations.clearing.reconciliation import (
    CCPReconciliationEngine,
    ReconciliationConfig,
    BreakType,
    BreakSeverity,
)
from neutryx.portfolio.lifecycle import (
    LifecycleEvent,
    LifecycleEventType,
    TradeAmendment,
)


class MockCCPConnector(CCPConnector):
    """Mock CCP connector for testing."""

    async def connect(self) -> bool:
        self._connected = True
        self._session_id = "MOCK_SESSION_123"
        return True

    async def disconnect(self) -> bool:
        self._connected = False
        return True

    async def submit_trade(self, trade: Trade) -> TradeSubmissionResponse:
        return TradeSubmissionResponse(
            submission_id=f"SUB-{trade.trade_id}",
            trade_id=trade.trade_id,
            status=TradeStatus.ACCEPTED,
            ccp_trade_id=f"CCP-{trade.trade_id}",
        )

    async def get_trade_status(self, trade_id: str) -> TradeStatus:
        return TradeStatus.ACCEPTED

    async def cancel_trade(self, trade_id: str) -> bool:
        return True

    async def get_margin_requirements(self, member_id: str = None) -> dict:
        return {
            "initial_margin": 1500000.0,
            "variation_margin": 50000.0,
            "additional_margin": 100000.0,
            "posted_margin": 1650000.0,
        }

    async def get_position_report(self, as_of_date: datetime = None) -> PositionReport:
        return PositionReport(
            report_id="POS-001",
            member_id=self.config.member_id,
            as_of_date=as_of_date or datetime.utcnow(),
            positions=[],
            total_exposure=Decimal("50000000"),
            initial_margin=Decimal("1500000"),
            variation_margin=Decimal("50000"),
        )

    async def healthcheck(self) -> bool:
        return self._connected


@pytest.fixture
def sample_parties():
    """Create sample trading parties."""
    return {
        "buyer": Party(
            party_id="BUYER_001",
            name="Buyer Bank",
            lei="5493001234567890BUYER",
            bic="BUYERB2L",
            member_id="MEMBER_BUYER",
        ),
        "seller": Party(
            party_id="SELLER_001",
            name="Seller Bank",
            lei="5493001234567890SELL",
            bic="SELLRB2L",
            member_id="MEMBER_SELLER",
        ),
    }


@pytest.fixture
def sample_trade(sample_parties):
    """Create sample trade."""
    return Trade(
        trade_id="IRS_TEST_001",
        product_type=ProductType.IRS,
        trade_date=datetime(2025, 1, 15),
        effective_date=datetime(2025, 1, 17),
        maturity_date=datetime(2030, 1, 17),
        buyer=sample_parties["buyer"],
        seller=sample_parties["seller"],
        economics=TradeEconomics(
            notional=Decimal("10000000"),
            currency="USD",
            fixed_rate=Decimal("0.045"),
        ),
    )


@pytest.fixture
def mock_connectors():
    """Create mock CCP connectors."""
    connectors = {}
    for ccp_name in ["LCH SwapClear", "CME Clearing", "ICE Clear"]:
        config = CCPConfig(
            ccp_name=ccp_name,
            member_id=f"MEMBER_{ccp_name.replace(' ', '_')}",
            api_endpoint=f"https://api.{ccp_name.lower().replace(' ', '')}.com",
        )
        connector = MockCCPConnector(config)
        connectors[ccp_name] = connector

    return connectors


@pytest.fixture
async def connected_connectors(mock_connectors):
    """Connect all mock connectors."""
    for connector in mock_connectors.values():
        await connector.connect()
    return mock_connectors


@pytest.mark.asyncio
class TestCCPRoutingWorkflow:
    """Test CCP routing workflow."""

    async def test_route_trade_lowest_margin(self, connected_connectors, sample_trade):
        """Test routing trade with lowest margin strategy."""
        router = CCPRouter(
            connectors=connected_connectors,
            default_strategy=RoutingStrategy.LOWEST_MARGIN,
        )

        decision = await router.route_trade(sample_trade)

        assert decision.selected_ccp in connected_connectors.keys()
        assert decision.routing_strategy == RoutingStrategy.LOWEST_MARGIN
        assert len(decision.eligible_ccps) > 0
        assert decision.selection_score > 0

    async def test_route_and_submit_trade(self, connected_connectors, sample_trade):
        """Test routing and submitting trade to CCP."""
        router = CCPRouter(
            connectors=connected_connectors,
            default_strategy=RoutingStrategy.LOWEST_MARGIN,
        )

        decision, response = await router.route_and_submit(sample_trade)

        assert decision.selected_ccp is not None
        assert response.status == TradeStatus.ACCEPTED
        assert response.ccp_trade_id is not None

    async def test_route_manual_ccp_selection(self, connected_connectors, sample_trade):
        """Test manual CCP selection."""
        router = CCPRouter(
            connectors=connected_connectors,
            default_strategy=RoutingStrategy.MANUAL,
        )

        decision = await router.route_trade(
            sample_trade,
            strategy=RoutingStrategy.MANUAL,
            preferred_ccp="CME Clearing",
        )

        assert decision.selected_ccp == "CME Clearing"
        assert decision.routing_strategy == RoutingStrategy.MANUAL

    async def test_routing_history_tracking(self, connected_connectors, sample_trade):
        """Test routing history is tracked."""
        router = CCPRouter(connectors=connected_connectors)

        # Route multiple trades
        await router.route_trade(sample_trade)

        history = router.get_routing_history()
        assert len(history) > 0
        assert history[0].trade_id == sample_trade.trade_id


@pytest.mark.asyncio
class TestAutomaticSettlementWorkflow:
    """Test automatic settlement workflow."""

    async def test_process_ccp_confirmation(
        self, connected_connectors, sample_trade, sample_parties
    ):
        """Test processing CCP confirmation and generating settlement."""
        config = SettlementWorkflowConfig(default_settlement_cycle=2)
        service = AutomaticSettlementService(config=config)

        # Create CCP response
        ccp_response = TradeSubmissionResponse(
            submission_id="SUB-001",
            trade_id=sample_trade.trade_id,
            status=TradeStatus.ACCEPTED,
            ccp_trade_id="CCP-001",
        )

        # Process confirmation
        workflow = await service.process_ccp_confirmation(
            trade=sample_trade,
            ccp_response=ccp_response,
            buyer_party=sample_parties["buyer"],
            seller_party=sample_parties["seller"],
        )

        assert workflow.status in [
            WorkflowStatus.INSTRUCTION_GENERATED,
            WorkflowStatus.INSTRUCTION_SENT,
        ]
        assert workflow.instruction_id is not None
        assert workflow.settlement_date is not None

    async def test_settlement_date_calculation(
        self, connected_connectors, sample_trade, sample_parties
    ):
        """Test settlement date is calculated correctly."""
        config = SettlementWorkflowConfig(default_settlement_cycle=2)
        service = AutomaticSettlementService(config=config)

        ccp_response = TradeSubmissionResponse(
            submission_id="SUB-001",
            trade_id=sample_trade.trade_id,
            status=TradeStatus.ACCEPTED,
            ccp_trade_id="CCP-001",
        )

        workflow = await service.process_ccp_confirmation(
            trade=sample_trade,
            ccp_response=ccp_response,
            buyer_party=sample_parties["buyer"],
            seller_party=sample_parties["seller"],
        )

        # Settlement date should be T+2
        expected_settlement = sample_trade.trade_date.date() + timedelta(days=2)
        # Adjust for weekends
        while expected_settlement.weekday() >= 5:
            expected_settlement += timedelta(days=1)

        assert workflow.settlement_date == expected_settlement

    async def test_rejected_trade_workflow(
        self, connected_connectors, sample_trade, sample_parties
    ):
        """Test workflow handles rejected trades."""
        service = AutomaticSettlementService()

        ccp_response = TradeSubmissionResponse(
            submission_id="SUB-001",
            trade_id=sample_trade.trade_id,
            status=TradeStatus.REJECTED,
            rejection_reason="Invalid counterparty",
        )

        workflow = await service.process_ccp_confirmation(
            trade=sample_trade,
            ccp_response=ccp_response,
            buyer_party=sample_parties["buyer"],
            seller_party=sample_parties["seller"],
        )

        assert workflow.status == WorkflowStatus.FAILED


@pytest.mark.asyncio
class TestMarginAggregation:
    """Test margin aggregation across CCPs."""

    async def test_refresh_margin_requirements(self, connected_connectors):
        """Test refreshing margin requirements from all CCPs."""
        service = MarginAggregationService(connectors=connected_connectors)

        margins = await service.refresh_margin_requirements()

        assert len(margins) > 0
        for ccp_name, margin in margins.items():
            assert margin.ccp_name == ccp_name
            assert margin.total_margin_required >= 0

    async def test_aggregated_report_generation(self, connected_connectors):
        """Test generating aggregated margin report."""
        service = MarginAggregationService(connectors=connected_connectors)

        report = await service.generate_aggregated_report()

        assert report.total_margin_required >= 0
        assert len(report.ccp_requirements) > 0
        assert report.total_initial_margin >= 0
        assert report.total_variation_margin >= 0

    async def test_margin_deficit_detection(self, connected_connectors):
        """Test detecting margin deficits."""
        service = MarginAggregationService(connectors=connected_connectors)

        await service.refresh_margin_requirements()
        deficit = service.get_total_margin_deficit()

        assert isinstance(deficit, Decimal)

    async def test_margin_alerts(self, connected_connectors):
        """Test margin alert generation."""
        config = MarginAggregatorConfig(margin_deficit_threshold=Decimal("100"))
        service = MarginAggregationService(
            connectors=connected_connectors,
            config=config,
        )

        await service.refresh_margin_requirements()
        alerts = service.get_margin_alerts()

        assert isinstance(alerts, list)


@pytest.mark.asyncio
class TestLifecycleSettlementMapping:
    """Test lifecycle event to settlement mapping."""

    async def test_process_amendment_event(self, sample_trade, sample_parties):
        """Test processing trade amendment."""
        mapper = LifecycleSettlementMapper()

        # Create amendment event
        event = LifecycleEvent(
            event_id="AMN-001",
            trade_id=sample_trade.id,
            event_type=LifecycleEventType.AMENDMENT,
            event_date=date.today(),
            effective_date=date.today(),
            description="Notional increase",
            changes={"notional": 15000000},
            previous_values={"notional": 10000000},
        )

        # Process event
        impact = await mapper.process_lifecycle_event(
            event=event,
            trade=sample_trade,
            parties=sample_parties,
        )

        assert impact.event_type == LifecycleEventType.AMENDMENT
        assert impact.settlement_action in [
            SettlementAction.UPDATE,
            SettlementAction.NO_ACTION,
        ]

    async def test_process_novation_event(self, sample_trade, sample_parties):
        """Test processing trade novation."""
        mapper = LifecycleSettlementMapper()

        # Add a new counterparty
        new_counterparty = Party(
            party_id="NEW_CP_001",
            name="New Counterparty",
            lei="5493001234567890NEWC",
            bic="NEWCPB2L",
            member_id="MEMBER_NEWCP",
        )
        sample_parties["new_counterparty"] = new_counterparty

        # Create novation event
        event = LifecycleEvent(
            event_id="NOV-001",
            trade_id=sample_trade.id,
            event_type=LifecycleEventType.NOVATION,
            event_date=date.today(),
            effective_date=date.today(),
            description="Novate to new counterparty",
            changes={"counterparty_id": "NEW_CP_001"},
            previous_values={"counterparty_id": "SELLER_001"},
        )

        # Process event
        impact = await mapper.process_lifecycle_event(
            event=event,
            trade=sample_trade,
            parties=sample_parties,
        )

        assert impact.event_type == LifecycleEventType.NOVATION
        assert impact.settlement_action == SettlementAction.GENERATE_NEW
        assert impact.new_instructions_required == 1

    async def test_process_termination_event(self, sample_trade, sample_parties):
        """Test processing trade termination."""
        mapper = LifecycleSettlementMapper()

        # Create termination event
        event = LifecycleEvent(
            event_id="TERM-001",
            trade_id=sample_trade.id,
            event_type=LifecycleEventType.TERMINATION,
            event_date=date.today(),
            effective_date=date.today(),
            description="Early termination",
            changes={"status": "TERMINATED"},
            previous_values={"status": "ACTIVE"},
        )

        # Process event
        impact = await mapper.process_lifecycle_event(
            event=event,
            trade=sample_trade,
            parties=sample_parties,
        )

        assert impact.event_type == LifecycleEventType.TERMINATION
        assert impact.settlement_action == SettlementAction.GENERATE_CLOSEOUT
        assert impact.closeout_amount is not None


@pytest.mark.asyncio
class TestCCPReconciliation:
    """Test CCP reconciliation."""

    async def test_position_reconciliation_match(self, connected_connectors):
        """Test position reconciliation with matching positions."""
        engine = CCPReconciliationEngine(connectors=connected_connectors)

        internal_positions = [
            {
                "trade_id": "IRS_001",
                "notional": 10000000,
                "fixed_rate": 0.045,
                "status": "ACTIVE",
            }
        ]

        result = await engine.reconcile_positions(
            ccp_name="LCH SwapClear",
            internal_positions=internal_positions,
        )

        assert result.recon_type.value == "position"
        assert result.total_internal_records == 1

    async def test_position_reconciliation_quantity_break(self, connected_connectors):
        """Test detecting quantity mismatches."""
        # Mock connector to return different notional
        connector = connected_connectors["LCH SwapClear"]
        original_get_position = connector.get_position_report

        async def mock_position_report(as_of_date=None):
            report = await original_get_position(as_of_date)
            report.positions = [
                {
                    "trade_id": "IRS_001",
                    "notional": 12000000,  # Different from internal
                    "fixed_rate": 0.045,
                    "status": "ACTIVE",
                }
            ]
            return report

        connector.get_position_report = mock_position_report

        engine = CCPReconciliationEngine(connectors=connected_connectors)

        internal_positions = [
            {
                "trade_id": "IRS_001",
                "notional": 10000000,
                "fixed_rate": 0.045,
                "status": "ACTIVE",
            }
        ]

        result = await engine.reconcile_positions(
            ccp_name="LCH SwapClear",
            internal_positions=internal_positions,
        )

        # Should detect quantity mismatch
        assert result.total_breaks > 0
        assert any(
            b.break_type == BreakType.QUANTITY_MISMATCH for b in result.breaks
        )

    async def test_break_resolution(self, connected_connectors):
        """Test resolving reconciliation breaks."""
        engine = CCPReconciliationEngine(connectors=connected_connectors)

        internal_positions = [
            {
                "trade_id": "IRS_001",
                "notional": 10000000,
                "fixed_rate": 0.045,
                "status": "ACTIVE",
            }
        ]

        result = await engine.reconcile_positions(
            ccp_name="LCH SwapClear",
            internal_positions=internal_positions,
        )

        # Get open breaks
        open_breaks = engine.get_open_breaks()
        if open_breaks:
            break_id = open_breaks[0].break_id

            # Resolve the break
            engine.resolve_break(
                break_id=break_id,
                resolution_notes="Corrected internal position",
                resolved_by="TestUser",
            )

            # Check break is resolved
            resolved_breaks = [
                b for b in result.breaks if b.break_id == break_id
            ]
            assert len(resolved_breaks) > 0


@pytest.mark.asyncio
class TestEndToEndWorkflow:
    """Test complete end-to-end workflow."""

    async def test_complete_trade_workflow(
        self, connected_connectors, sample_trade, sample_parties
    ):
        """Test complete workflow from routing to settlement."""
        # Step 1: Route trade to optimal CCP
        router = CCPRouter(
            connectors=connected_connectors,
            default_strategy=RoutingStrategy.LOWEST_MARGIN,
        )

        routing_decision, submission_response = await router.route_and_submit(
            sample_trade
        )

        assert routing_decision.selected_ccp is not None
        assert submission_response.status == TradeStatus.ACCEPTED

        # Step 2: Generate settlement instruction
        settlement_service = AutomaticSettlementService()

        workflow = await settlement_service.process_ccp_confirmation(
            trade=sample_trade,
            ccp_response=submission_response,
            buyer_party=sample_parties["buyer"],
            seller_party=sample_parties["seller"],
        )

        assert workflow.instruction_id is not None

        # Step 3: Update margin aggregation
        margin_service = MarginAggregationService(connectors=connected_connectors)

        report = await margin_service.generate_aggregated_report()

        assert report.total_margin_required > 0

        # Step 4: Verify in reconciliation
        recon_engine = CCPReconciliationEngine(connectors=connected_connectors)

        internal_positions = [
            {
                "trade_id": sample_trade.trade_id,
                "notional": float(sample_trade.economics.notional),
                "fixed_rate": float(sample_trade.economics.fixed_rate),
                "status": "ACTIVE",
            }
        ]

        recon_result = await recon_engine.reconcile_positions(
            ccp_name=routing_decision.selected_ccp,
            internal_positions=internal_positions,
        )

        # Reconciliation should pass (assuming CCP returns same data)
        assert recon_result is not None

    async def test_workflow_with_lifecycle_event(
        self, connected_connectors, sample_trade, sample_parties
    ):
        """Test workflow including lifecycle event."""
        # Step 1: Initial trade submission
        router = CCPRouter(connectors=connected_connectors)
        _, submission_response = await router.route_and_submit(sample_trade)

        # Step 2: Generate initial settlement
        settlement_service = AutomaticSettlementService()
        initial_workflow = await settlement_service.process_ccp_confirmation(
            trade=sample_trade,
            ccp_response=submission_response,
            buyer_party=sample_parties["buyer"],
            seller_party=sample_parties["seller"],
        )

        # Step 3: Process amendment
        mapper = LifecycleSettlementMapper(
            settlement_generator=settlement_service.generator
        )

        amendment_event = LifecycleEvent(
            event_id="AMN-001",
            trade_id=sample_trade.trade_id,
            event_type=LifecycleEventType.AMENDMENT,
            event_date=date.today(),
            effective_date=date.today(),
            description="Increase notional",
            changes={"notional": 15000000},
            previous_values={"notional": 10000000},
        )

        impact = await mapper.process_lifecycle_event(
            event=amendment_event,
            trade=sample_trade,
            parties=sample_parties,
        )

        assert impact.processed is True

        # Step 4: Refresh margins after amendment
        margin_service = MarginAggregationService(connectors=connected_connectors)
        updated_report = await margin_service.generate_aggregated_report()

        assert updated_report.total_margin_required > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
