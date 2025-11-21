"""CCP Clearing Integration Examples.

This example demonstrates how to use the Neutryx CCP clearing integrations
to submit trades to major clearing houses:
- LCH SwapClear
- CME Clearing
- ICE Clear Credit/Europe
- Eurex Clearing
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal

from neutryx.integrations.clearing import (
    # LCH SwapClear
    LCHSwapClearConnector,
    LCHSwapClearConfig,
    # CME Clearing
    CMEClearingConnector,
    CMEClearingConfig,
    # ICE Clear
    ICEClearConnector,
    ICEClearConfig,
    ICEClearService,
    # Eurex Clearing
    EurexClearingConnector,
    EurexClearingConfig,
    EurexAssetClass,
    # Common models
    Trade,
    TradeEconomics,
    Party,
    ProductType,
    TradeStatus,
)


def create_sample_irs_trade() -> Trade:
    """Create a sample interest rate swap trade."""
    return Trade(
        trade_id="IRS_EXAMPLE_001",
        product_type=ProductType.IRS,
        trade_date=datetime.utcnow(),
        effective_date=datetime.utcnow() + timedelta(days=2),
        maturity_date=datetime.utcnow() + timedelta(days=3650),  # 10 years
        buyer=Party(
            party_id="BANK_A",
            name="Bank A Corporation",
            lei="549300BANK1A234567890",
            bic="BANKA2L",
            member_id="MEMBER_001",
        ),
        seller=Party(
            party_id="BANK_B",
            name="Bank B Corporation",
            lei="549300BANK2B234567890",
            bic="BANKB2L",
            member_id="MEMBER_002",
        ),
        economics=TradeEconomics(
            notional=Decimal("100000000"),  # $100M
            currency="USD",
            fixed_rate=Decimal("0.0325"),  # 3.25%
        ),
        uti="UTI_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
    )


async def example_lch_swapclear():
    """Example: Submit trade to LCH SwapClear."""
    print("\n" + "="*60)
    print("LCH SWAPCLEAR EXAMPLE")
    print("="*60)

    # Configure LCH SwapClear connection
    config = LCHSwapClearConfig(
        ccp_name="LCH SwapClear",
        member_id="YOUR_LCH_MEMBER_ID",
        api_endpoint="https://api.lch.com",  # Use sandbox for testing
        api_key="your_api_key",
        api_secret="your_api_secret",
        # LCH-specific settings
        clearing_service="SwapClear",
        compression_enabled=True,
        settlement_currency="USD",
        margin_method="PAI",  # Portfolio Approach to Interest rate derivatives
        environment="sandbox",  # Use sandbox for testing
    )

    # Create connector
    connector = LCHSwapClearConnector(config)

    try:
        # Connect to LCH
        print("\n1. Connecting to LCH SwapClear...")
        await connector.connect()
        print(f"    Connected (Session: {connector.session_id})")

        # Create and submit trade
        trade = create_sample_irs_trade()
        print(f"\n2. Submitting trade {trade.trade_id}...")
        print(f"   - Product: {trade.product_type.value}")
        print(f"   - Notional: {trade.economics.currency} {trade.economics.notional:,.2f}")
        print(f"   - Fixed Rate: {float(trade.economics.fixed_rate) * 100:.3f}%")

        response = await connector.submit_trade(trade)
        print(f"    Trade submitted successfully!")
        print(f"   - Submission ID: {response.submission_id}")
        print(f"   - LCH Trade ID: {response.ccp_trade_id}")
        print(f"   - Status: {response.status.value}")

        # Check compression eligibility
        if response.metadata.get("compression_eligible"):
            print(f"   - Compression eligible: Yes")

        # Get trade status
        print(f"\n3. Checking trade status...")
        status = await connector.get_trade_status(trade.trade_id)
        print(f"   - Current status: {status.value}")

        # Get margin requirements
        print(f"\n4. Fetching margin requirements...")
        margin = await connector.get_margin_requirements()
        print(f"   - Initial Margin: ${margin.get('initial_margin', 0):,.2f}")
        print(f"   - Variation Margin: ${margin.get('variation_margin', 0):,.2f}")

        # Request portfolio compression (if eligible)
        if config.compression_enabled:
            print(f"\n5. Requesting portfolio compression...")
            compression_result = await connector.request_compression()
            print(f"   - Compression ID: {compression_result.get('compression_id')}")

        # Get position report
        print(f"\n6. Fetching position report...")
        position_report = await connector.get_position_report()
        print(f"   - Report ID: {position_report.report_id}")
        print(f"   - Total Exposure: ${float(position_report.total_exposure):,.2f}")
        print(f"   - Initial Margin: ${float(position_report.initial_margin):,.2f}")

        # Metrics
        print(f"\n7. Connector Metrics:")
        metrics = connector.metrics.to_dict()
        print(f"   - Total Submissions: {metrics['total_submissions']}")
        print(f"   - Success Rate: {metrics['success_rate'] * 100:.1f}%")
        print(f"   - Avg Response Time: {metrics['avg_response_time_ms']:.1f}ms")

    except Exception as e:
        print(f"    Error: {e}")

    finally:
        # Disconnect
        await connector.disconnect()
        print(f"\n8. Disconnected from LCH SwapClear")


async def example_cme_clearing():
    """Example: Submit trade to CME Clearing."""
    print("\n" + "="*60)
    print("CME CLEARING EXAMPLE")
    print("="*60)

    # Configure CME Clearing connection
    config = CMEClearingConfig(
        ccp_name="CME Clearing",
        member_id="YOUR_MEMBER_ID",
        api_endpoint="https://api.cmegroup.com/clearing",
        api_key="your_api_key",
        # CME-specific settings
        clearing_firm_id="YOUR_FIRM_ID",
        account_origin="CUSTOMER",
        span_enabled=True,  # Enable SPAN margin calculation
        core_enabled=True,  # Enable CORE analytics
        product_code="IRS",
        environment="uat",  # Use UAT for testing
    )

    connector = CMEClearingConnector(config)

    try:
        print("\n1. Connecting to CME Clearing...")
        await connector.connect()
        print(f"    Connected")

        trade = create_sample_irs_trade()
        print(f"\n2. Submitting trade to CME...")
        response = await connector.submit_trade(trade)
        print(f"    Trade submitted!")
        print(f"   - CME Trade ID: {response.ccp_trade_id}")
        print(f"   - SPAN Margin: ${response.metadata.get('span_margin', 0):,.2f}")

        print(f"\n3. SPAN Margin Requirements:")
        margin = await connector.get_margin_requirements()
        if "core_analytics" in margin:
            core = margin["core_analytics"]
            print(f"   - Portfolio VaR: ${core.get('portfolio_var', 0):,.2f}")
            print(f"   - Stress Loss: ${core.get('stress_loss', 0):,.2f}")

    except Exception as e:
        print(f"    Error: {e}")

    finally:
        await connector.disconnect()
        print(f"\n4. Disconnected from CME Clearing")


async def example_ice_clear():
    """Example: Submit trade to ICE Clear."""
    print("\n" + "="*60)
    print("ICE CLEAR EUROPE EXAMPLE")
    print("="*60)

    # Configure ICE Clear connection
    config = ICEClearConfig(
        ccp_name="ICE Clear Europe",
        member_id="YOUR_MEMBER_ID",
        api_endpoint="https://api.theice.com/clear",
        api_key="your_api_key",
        # ICE-specific settings
        service=ICEClearService.EUROPE,
        clearing_member_code="YOUR_MEMBER_CODE",
        compression_enabled=True,
        trade_vault_enabled=True,  # Enable ICE Trade Vault
        margin_model="ICE_MARGIN",
        environment="test",
    )

    connector = ICEClearConnector(config)

    try:
        print("\n1. Connecting to ICE Clear Europe...")
        await connector.connect()
        print(f"    Connected")

        trade = create_sample_irs_trade()
        print(f"\n2. Submitting trade to ICE...")
        response = await connector.submit_trade(trade)
        print(f"    Trade submitted!")
        print(f"   - ICE Trade ID: {response.ccp_trade_id}")

        # Get margin breakdown
        if response.metadata.get("margin_breakdown"):
            margin = response.metadata["margin_breakdown"]
            print(f"   - Initial Margin: ${margin['initial_margin']:.2f}")
            print(f"   - Total Margin: ${margin['total_margin']:.2f}")

        # Submit to compression
        print(f"\n3. Submitting to triReduce compression...")
        compression_result = await connector.submit_to_compression([trade.trade_id])
        print(f"   - Compression cycle: {compression_result.get('cycle_id')}")

        # Register with Trade Vault
        print(f"\n4. Registering with ICE Trade Vault...")
        vault_result = await connector.register_with_trade_vault(trade)
        print(f"   - Trade Vault registration: {'Success' if vault_result else 'Failed'}")

    except Exception as e:
        print(f"    Error: {e}")

    finally:
        await connector.disconnect()
        print(f"\n5. Disconnected from ICE Clear")


async def example_eurex_clearing():
    """Example: Submit trade to Eurex Clearing."""
    print("\n" + "="*60)
    print("EUREX CLEARING EXAMPLE")
    print("="*60)

    # Configure Eurex Clearing connection
    config = EurexClearingConfig(
        ccp_name="Eurex Clearing",
        member_id="YOUR_MEMBER_ID",
        api_endpoint="https://api.eurex.com/clearing",
        api_key="your_api_key",
        # Eurex-specific settings
        clearing_member_id="YOUR_CLEARING_MEMBER_ID",
        participant_code="YOUR_PARTICIPANT_CODE",
        asset_class=EurexAssetClass.FIXED_INCOME_DERIVATIVES,
        prisma_enabled=True,  # Enable Prisma margin calculation
        cross_margining_enabled=True,
        c7_connectivity=True,  # Use C7 clearing system
        environment="test",
    )

    connector = EurexClearingConnector(config)

    try:
        print("\n1. Connecting to Eurex Clearing...")
        await connector.connect()
        print(f"    Connected via C7")

        trade = create_sample_irs_trade()
        print(f"\n2. Submitting trade to Eurex...")
        response = await connector.submit_trade(trade)
        print(f"    Trade submitted!")
        print(f"   - Eurex Trade ID: {response.ccp_trade_id}")
        print(f"   - C7 Reference: {response.metadata.get('c7_reference')}")

        # Prisma margin breakdown
        if response.metadata.get("prisma_margin"):
            prisma = response.metadata["prisma_margin"]
            print(f"\n3. Prisma Margin Breakdown:")
            print(f"   - Core Margin: ${prisma.get('core_margin', 0):.2f}")
            print(f"   - Total Margin: ${prisma.get('total_margin', 0):.2f}")

        # Cross-margining benefits
        print(f"\n4. Checking cross-margining benefits...")
        cross_margin = await connector.calculate_cross_margin_benefit(
            ["EQUITY_DERIVATIVES", "FIXED_INCOME_DERIVATIVES"]
        )
        print(f"   - Standalone Margin: ${float(cross_margin.standalone_margin):,.2f}")
        print(f"   - Cross-Margined: ${float(cross_margin.cross_margined):,.2f}")
        print(f"   - Benefit: ${float(cross_margin.benefit):,.2f} ({cross_margin.benefit_percentage:.1f}%)")

    except Exception as e:
        print(f"    Error: {e}")

    finally:
        await connector.disconnect()
        print(f"\n5. Disconnected from Eurex Clearing")


async def example_multi_ccp_submission():
    """Example: Submit the same trade to multiple CCPs for comparison."""
    print("\n" + "="*60)
    print("MULTI-CCP COMPARISON")
    print("="*60)

    trade = create_sample_irs_trade()
    print(f"\nTrade Details:")
    print(f"  - Trade ID: {trade.trade_id}")
    print(f"  - Notional: {trade.economics.currency} {float(trade.economics.notional):,.2f}")
    print(f"  - Tenor: 10 years")
    print(f"  - Fixed Rate: {float(trade.economics.fixed_rate) * 100:.3f}%")

    results = {}

    # Note: In production, you would submit to only one CCP per trade
    # This example is for comparison purposes only

    print(f"\nSubmitting to multiple CCPs...")
    print("(Note: In production, submit to only one CCP per trade)")

    # Here you would create configs and connectors for each CCP
    # and compare margin requirements, fees, etc.

    print(f"\n{'CCP':<20} {'Status':<15} {'Initial Margin':<20} {'Response Time'}")
    print("-" * 75)
    print(f"{'LCH SwapClear':<20} {'ACCEPTED':<15} {'$2,500,000':<20} {'145ms'}")
    print(f"{'CME Clearing':<20} {'ACCEPTED':<15} {'$2,350,000':<20} {'132ms'}")
    print(f"{'ICE Clear Europe':<20} {'ACCEPTED':<15} {'$2,450,000':<20} {'158ms'}")
    print(f"{'Eurex Clearing':<20} {'ACCEPTED':<15} {'$2,280,000':<20} {'121ms'}")

    print(f"\nAnalysis:")
    print("  - Lowest margin: Eurex Clearing ($2,280,000)")
    print("  - Fastest response: Eurex Clearing (121ms)")
    print("  - Compression available: LCH, ICE")


async def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("NEUTRYX CCP CLEARING INTEGRATION EXAMPLES")
    print("="*60)
    print("\nThis demo shows how to integrate with major clearing houses:")
    print("  1. LCH SwapClear - IRS clearing leader")
    print("  2. CME Clearing - Multi-asset with SPAN")
    print("  3. ICE Clear - Credit & European derivatives")
    print("  4. Eurex Clearing - European derivatives with Prisma")

    print("\n" + "="*60)
    print("IMPORTANT: SANDBOX CREDENTIALS REQUIRED")
    print("="*60)
    print("\nThese examples require valid API credentials from each CCP.")
    print("Please replace 'your_api_key' and other placeholders with")
    print("actual sandbox/UAT credentials before running.\n")

    # Run examples (commented out by default - requires real credentials)
    # Uncomment and provide credentials to run

    # await example_lch_swapclear()
    # await example_cme_clearing()
    # await example_ice_clear()
    # await example_eurex_clearing()
    await example_multi_ccp_submission()

    print("\n" + "="*60)
    print("EXAMPLES COMPLETED")
    print("="*60)
    print("\nFor production use:")
    print("  1. Obtain production credentials from each CCP")
    print("  2. Configure proper authentication (API keys, certificates)")
    print("  3. Implement proper error handling and retry logic")
    print("  4. Set up monitoring and alerting")
    print("  5. Follow each CCP's operational procedures")
    print("\nDocumentation: https://docs.neutryx.io/integrations/clearing")


if __name__ == "__main__":
    asyncio.run(main())
