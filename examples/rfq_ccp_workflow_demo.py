"""RFQ-to-CCP Complete Workflow Demonstration.

This example demonstrates the full RFQ workflow integrated with CCP clearing:
1. Create and submit RFQ for interest rate swap
2. Collect quotes from multiple market makers
3. Execute auction (multiple types demonstrated)
4. Route winning trades to optimal CCPs
5. Generate settlement instructions
6. Track margin impact
7. Reconcile positions
"""

import asyncio
from datetime import datetime, timedelta, date
from decimal import Decimal

from neutryx.integrations.clearing import (
    # RFQ components
    RFQManager,
    RFQ,
    RFQSpecification,
    Quote,
    Party,
    ProductType,
    OrderSide,
    AuctionType,
    TimeInForce,

    # CCP components
    CCPRouter,
    RoutingStrategy,
    CCPConfig,

    # Settlement
    AutomaticSettlementService,

    # Margin aggregation
    MarginAggregationService,

    # Reconciliation
    CCPReconciliationEngine,

    # RFQ-CCP Integration
    RFQCCPIntegrationService,
    RFQCCPIntegrationConfig,
)


# Mock CCP connector for demonstration
class MockCCPConnector:
    """Simple mock CCP connector for demonstration."""

    def __init__(self, config):
        self.config = config
        self._connected = False

    async def connect(self):
        self._connected = True
        return True

    async def submit_trade(self, trade):
        from neutryx.integrations.clearing.base import TradeSubmissionResponse, TradeStatus
        return TradeSubmissionResponse(
            submission_id=f"SUB-{trade.trade_id}",
            trade_id=trade.trade_id,
            status=TradeStatus.ACCEPTED,
            ccp_trade_id=f"CCP-{trade.trade_id}",
        )

    async def get_margin_requirements(self, member_id=None):
        return {
            "initial_margin": 1500000.0,
            "variation_margin": 50000.0,
            "posted_margin": 1600000.0,
        }

    async def get_position_report(self, as_of_date=None):
        from neutryx.integrations.clearing.base import PositionReport
        return PositionReport(
            report_id="POS-001",
            member_id=self.config.member_id,
            as_of_date=as_of_date or datetime.utcnow(),
            positions=[],
        )

    async def get_trade_status(self, trade_id):
        from neutryx.integrations.clearing.base import TradeStatus
        return TradeStatus.ACCEPTED

    async def cancel_trade(self, trade_id):
        return True

    async def healthcheck(self):
        return self._connected

    @property
    def is_connected(self):
        return self._connected


async def main():
    """Run RFQ-CCP workflow demonstration."""

    print("=" * 80)
    print("RFQ-to-CCP Complete Workflow Demonstration")
    print("=" * 80)
    print()

    # ========================================================================
    # Setup: Initialize services
    # ========================================================================

    print("1. Initializing services...")

    # Create RFQ manager
    rfq_manager = RFQManager()

    # Create mock CCP connectors
    ccp_configs = {
        "LCH SwapClear": CCPConfig(
            ccp_name="LCH SwapClear",
            member_id="MEMBER_LCH",
            api_endpoint="https://api.lch.com"
        ),
        "CME Clearing": CCPConfig(
            ccp_name="CME Clearing",
            member_id="MEMBER_CME",
            api_endpoint="https://api.cmegroup.com"
        ),
    }

    ccp_connectors = {}
    for ccp_name, config in ccp_configs.items():
        connector = MockCCPConnector(config)
        await connector.connect()
        ccp_connectors[ccp_name] = connector
        print(f"   ✓ Connected to {ccp_name}")

    # Create CCP router
    ccp_router = CCPRouter(
        connectors=ccp_connectors,
        default_strategy=RoutingStrategy.LOWEST_MARGIN,
    )

    # Create settlement service
    settlement_service = AutomaticSettlementService()

    # Create margin service
    margin_service = MarginAggregationService(connectors=ccp_connectors)

    # Create reconciliation engine
    recon_engine = CCPReconciliationEngine(connectors=ccp_connectors)

    # Create RFQ-CCP integration service
    integration_config = RFQCCPIntegrationConfig(
        default_routing_strategy=RoutingStrategy.LOWEST_MARGIN,
        auto_route_to_ccp=True,
        auto_generate_settlement=True,
    )

    integration_service = RFQCCPIntegrationService(
        rfq_manager=rfq_manager,
        ccp_router=ccp_router,
        settlement_service=settlement_service,
        margin_service=margin_service,
        recon_engine=recon_engine,
        config=integration_config,
    )

    print("   ✓ All services initialized\n")

    # ========================================================================
    # Step 1: Create RFQ
    # ========================================================================

    print("2. Creating RFQ for 5Y USD Interest Rate Swap...")

    requester = Party(
        party_id="BANK_A",
        name="Bank A Trading",
        lei="5493001234567890BANKA",
        bic="BANKA2L",
        member_id="MEMBER_BANKA",
    )

    specification = RFQSpecification(
        product_type=ProductType.IRS,
        notional=Decimal("50000000"),  # $50M
        currency="USD",
        effective_date=datetime(2025, 2, 1),
        maturity_date=datetime(2030, 2, 1),
        tenor="5Y",
        payment_frequency="6M",
        day_count="ACT/360",
    )

    quote_deadline = datetime.utcnow() + timedelta(minutes=30)

    rfq = rfq_manager.create_rfq(
        requester=requester,
        specification=specification,
        side=OrderSide.BUY,  # Buyer wants to receive fixed
        quote_deadline=quote_deadline,
        auction_type=AuctionType.SINGLE_PRICE,  # Uniform price auction
    )

    print(f"   ✓ RFQ Created: {rfq.rfq_id}")
    print(f"     Product: {specification.tenor} {specification.currency} IRS")
    print(f"     Notional: ${float(specification.notional):,.0f}")
    print(f"     Side: {rfq.side.value.upper()}")
    print(f"     Auction Type: {rfq.auction_type.value}")
    print(f"     Quote Deadline: {quote_deadline.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Submit RFQ to market
    rfq_manager.submit_rfq(rfq.rfq_id)
    print(f"   ✓ RFQ submitted to market (Status: {rfq.status.value})\n")

    # ========================================================================
    # Step 2: Collect quotes from market makers
    # ========================================================================

    print("3. Collecting quotes from market makers...")

    market_makers = [
        Party(
            party_id=f"MM_{i}",
            name=f"Market Maker {i}",
            lei=f"5493001234567890MM00{i}",
            bic=f"MM{i}BANK",
            member_id=f"MEMBER_MM{i}",
        )
        for i in range(1, 6)
    ]

    # Simulate quotes from 5 market makers with varying prices
    base_rate = Decimal("0.0425")
    for i, mm in enumerate(market_makers):
        price_adjustment = Decimal(str(i * 0.0005))  # Spread prices

        quote = Quote(
            rfq_id=rfq.rfq_id,
            quoter=mm,
            quoter_member_id=mm.member_id,
            side=OrderSide.SELL,  # Offering to sell (pay fixed)
            quantity=Decimal("50000000"),  # Full notional
            price=base_rate + price_adjustment,  # Fixed rate
            time_in_force=TimeInForce.GTC,
        )

        rfq_manager.submit_quote(rfq.rfq_id, quote)
        print(f"   ✓ Quote from {mm.name}: Fixed rate {float(quote.price):.4%}")

    print(f"\n   Total quotes received: {rfq.quotes_received}\n")

    # ========================================================================
    # Step 3: Execute RFQ auction
    # ========================================================================

    print("4. Executing RFQ auction...")

    auction_result = rfq_manager.execute_auction(rfq.rfq_id)

    print(f"   ✓ Auction executed successfully")
    print(f"     Auction Type: {auction_result.auction_type.value}")
    print(f"     Clearing Price: {float(auction_result.clearing_price):.4%}")
    print(f"     Total Filled: ${float(auction_result.total_quantity_filled):,.0f}")
    print(f"     Participants: {auction_result.num_participants}")
    print(f"     Competition Score: {auction_result.competition_score:.1f}/100")
    print(f"     Execution Time: {auction_result.execution_duration_ms:.2f}ms\n")

    print("   Winning allocations:")
    for i, alloc in enumerate(auction_result.allocations, 1):
        print(f"     {i}. Quoter: {alloc['quoter_id']}, "
              f"Quantity: ${alloc['quantity']:,.0f}, "
              f"Price: {alloc['price']:.4%}")
    print()

    # ========================================================================
    # Step 4: Execute complete CCP workflow
    # ========================================================================

    print("5. Executing RFQ-to-CCP workflow...")

    execution_result = await integration_service.execute_rfq_workflow(
        rfq_id=rfq.rfq_id,
        requester_party=requester,
    )

    print(f"   ✓ Workflow completed: {execution_result.status.value}")
    print(f"     Execution ID: {execution_result.execution_id}")
    print(f"     Total Duration: {execution_result.total_duration_ms:.2f}ms\n")

    # ========================================================================
    # Step 5: Review trade allocations and CCP routing
    # ========================================================================

    print("6. Reviewing trade allocations and CCP routing...")

    for i, allocation in enumerate(execution_result.allocations, 1):
        print(f"\n   Allocation {i}:")
        print(f"     Trade ID: {allocation.trade_id}")
        print(f"     Counterparty: {allocation.counterparty.name}")
        print(f"     Quantity: ${float(allocation.quantity):,.0f}")
        print(f"     Price: {float(allocation.price):.4%}")
        print(f"     Selected CCP: {allocation.selected_ccp}")
        print(f"     CCP Trade ID: {allocation.ccp_trade_id}")
        print(f"     CCP Status: {allocation.ccp_status}")
        print(f"     Settlement Instruction: {allocation.settlement_instruction_id}")
        print(f"     Settlement Date: {allocation.settlement_date}")

    print(f"\n   Summary:")
    print(f"     Total Allocations: {execution_result.total_allocations}")
    print(f"     Successful CCP Submissions: {execution_result.successful_submissions}")
    print(f"     Failed Submissions: {execution_result.failed_submissions}\n")

    # ========================================================================
    # Step 6: Check margin impact
    # ========================================================================

    print("7. Analyzing margin impact...")

    margin_impact = await integration_service.get_margin_impact(
        execution_result.execution_id
    )

    if margin_impact:
        print(f"   ✓ Margin impact calculated")
        print(f"     Total Portfolio Margin: ${margin_impact['total_portfolio_margin']:,.2f}")
        print(f"     Margin Deficit: ${margin_impact['margin_deficit']:,.2f}")
        print(f"\n     Margin by CCP:")
        for ccp_name, margins in margin_impact['margin_by_ccp'].items():
            print(f"       {ccp_name}:")
            print(f"         Initial Margin: ${margins['initial_margin']:,.2f}")
            print(f"         Variation Margin: ${margins['variation_margin']:,.2f}")
            print(f"         Total: ${margins['total_margin']:,.2f}")
    print()

    # ========================================================================
    # Step 7: Execution statistics
    # ========================================================================

    print("8. Execution statistics...")

    stats = integration_service.get_execution_statistics()
    print(f"   Total Executions: {stats['total_executions']}")
    print(f"   Completed: {stats['completed_executions']}")
    print(f"   Success Rate: {stats['success_rate']:.1f}%")
    print(f"   Total Allocations: {stats['total_allocations']}")
    print(f"   CCP Submission Rate: {stats['ccp_submission_rate']:.1f}%\n")

    # ========================================================================
    # Demonstration of other auction types
    # ========================================================================

    print("=" * 80)
    print("9. Demonstrating other auction types...")
    print("=" * 80)
    print()

    auction_types = [
        (AuctionType.MULTI_PRICE, "Multi-Price (Discriminatory)"),
        (AuctionType.VICKREY, "Vickrey (Second-Price)"),
        (AuctionType.DUTCH, "Dutch (Descending Price)"),
    ]

    for auction_type, description in auction_types:
        print(f"\n{description} Auction:")

        # Create new RFQ
        rfq2 = rfq_manager.create_rfq(
            requester=requester,
            specification=specification,
            side=OrderSide.BUY,
            quote_deadline=datetime.utcnow() + timedelta(minutes=30),
            auction_type=auction_type,
        )

        rfq_manager.submit_rfq(rfq2.rfq_id)

        # Submit quotes
        for i, mm in enumerate(market_makers):
            price_adj = Decimal(str(i * 0.0005))
            quote = Quote(
                rfq_id=rfq2.rfq_id,
                quoter=mm,
                quoter_member_id=mm.member_id,
                side=OrderSide.SELL,
                quantity=Decimal("50000000"),
                price=base_rate + price_adj,
                time_in_force=TimeInForce.GTC,
            )
            rfq_manager.submit_quote(rfq2.rfq_id, quote)

        # Execute auction
        result2 = rfq_manager.execute_auction(rfq2.rfq_id)

        print(f"   Clearing Price: {float(result2.clearing_price):.4%}")
        print(f"   Total Filled: ${float(result2.total_quantity_filled):,.0f}")
        print(f"   Competition Score: {result2.competition_score:.1f}/100")
        print(f"   Winning Quotes: {len(result2.winning_quotes)}")

    print("\n" + "=" * 80)
    print("✓ RFQ-CCP Workflow Demonstration Complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
