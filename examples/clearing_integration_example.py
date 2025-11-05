"""CCP Clearing Integration Examples.

This script demonstrates how to use Neutryx's CCP clearing integrations
to submit trades to major clearing houses.

Examples include:
- LCH SwapClear: Interest rate swap clearing
- CME Clearing: Multi-asset clearing with SPAN
- ICE Clear: Credit derivatives and IRS
- Eurex Clearing: European derivatives with Prisma
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal

from neutryx.integrations.clearing import (
    # Base types
    Party,
    ProductType,
    Trade,
    TradeEconomics,
    # LCH
    LCHSwapClearConfig,
    LCHSwapClearConnector,
    # CME
    CMEClearingConfig,
    CMEClearingConnector,
    # ICE
    ICEClearConfig,
    ICEClearConnector,
    ICEClearService,
    # Eurex
    EurexClearingConfig,
    EurexClearingConnector,
    EurexAssetClass,
    # Errors
    CCPConnectionError,
    CCPTradeRejectionError,
)


def create_sample_irs_trade() -> Trade:
    """Create a sample interest rate swap trade.

    Returns:
        Sample IRS trade
    """
    buyer = Party(
        party_id="BANK001",
        name="Example Bank Corp",
        lei="549300XXXXXXXXXXXX01",
        member_id="MEMBER001",
    )

    seller = Party(
        party_id="HEDGE001",
        name="Example Hedge Fund",
        lei="549300XXXXXXXXXXXX02",
        member_id="MEMBER002",
    )

    economics = TradeEconomics(
        notional=Decimal("10000000"),  # $10M notional
        currency="USD",
        fixed_rate=Decimal("0.025"),  # 2.5% fixed rate
    )

    trade_date = datetime.utcnow()

    return Trade(
        trade_id=f"IRS-{trade_date.strftime('%Y%m%d')}-001",
        product_type=ProductType.IRS,
        trade_date=trade_date,
        effective_date=trade_date + timedelta(days=2),  # T+2 settlement
        maturity_date=trade_date + timedelta(days=365 * 5),  # 5-year swap
        buyer=buyer,
        seller=seller,
        economics=economics,
        uti=f"UTI-{trade_date.timestamp()}",
        collateral_currency="USD",
    )


async def example_lch_swapclear():
    """Example: Submit trade to LCH SwapClear.

    LCH SwapClear is the world's leading IRS clearing service.
    """
    print("\n" + "=" * 60)
    print("LCH SwapClear Example")
    print("=" * 60)

    # Configure LCH connection
    config = LCHSwapClearConfig(
        ccp_name="LCH",
        member_id="MEMBER123",
        api_endpoint="https://api.lch.com/swapclear",  # Production endpoint
        api_key="your_api_key_here",
        api_secret="your_api_secret_here",
        environment="production",
        compression_enabled=True,  # Enable portfolio compression
        margin_method="PAI",  # Portfolio approach to interest rate risk
    )

    # Create connector
    connector = LCHSwapClearConnector(config)

    try:
        # Connect to LCH
        print("\nConnecting to LCH SwapClear...")
        await connector.connect()
        print(f"Connected! Session ID: {connector.session_id}")

        # Create and submit trade
        trade = create_sample_irs_trade()
        print(f"\nSubmitting trade {trade.trade_id}...")
        print(f"  Notional: {trade.economics.currency} {trade.economics.notional:,.0f}")
        print(f"  Fixed Rate: {float(trade.economics.fixed_rate) * 100:.2f}%")
        print(f"  Maturity: {trade.maturity_date.strftime('%Y-%m-%d')}")

        response = await connector.submit_trade(trade)

        print(f"\nTrade submitted successfully!")
        print(f"  Status: {response.status}")
        print(f"  LCH Trade ID: {response.ccp_trade_id}")
        print(f"  Compression Eligible: {response.metadata.get('compression_eligible')}")

        # Get margin requirements
        print("\nFetching margin requirements...")
        margin = await connector.get_margin_requirements()
        print(f"  Initial Margin: ${margin['initial_margin']:,.2f}")
        print(f"  Variation Margin: ${margin.get('variation_margin', 0):,.2f}")

        # Get position report
        print("\nFetching position report...")
        report = await connector.get_position_report()
        print(f"  Report ID: {report.report_id}")
        print(f"  Total Positions: {len(report.positions)}")
        print(f"  Total Exposure: ${report.total_exposure:,.2f}")

        # Check health
        healthy = await connector.healthcheck()
        print(f"\nConnection Health: {'OK' if healthy else 'FAILED'}")

        # Print metrics
        metrics = connector.metrics
        print(f"\nPerformance Metrics:")
        print(f"  Total Submissions: {metrics.total_submissions}")
        print(f"  Success Rate: {metrics.success_rate():.1%}")
        print(f"  Avg Response Time: {metrics.avg_response_time_ms:.0f}ms")

    except CCPTradeRejectionError as e:
        print(f"\nTrade rejected: {e}")
        print(f"  Rejection Code: {e.rejection_code}")

    except CCPConnectionError as e:
        print(f"\nConnection error: {e}")

    finally:
        # Disconnect
        await connector.disconnect()
        print("\nDisconnected from LCH SwapClear")


async def example_cme_clearing():
    """Example: Submit trade to CME Clearing with SPAN margin calculation."""
    print("\n" + "=" * 60)
    print("CME Clearing Example")
    print("=" * 60)

    config = CMEClearingConfig(
        ccp_name="CME",
        member_id="MEMBER456",
        clearing_firm_id="FIRM123",
        api_endpoint="https://api.cmegroup.com/clearing",
        api_key="your_api_key_here",
        span_enabled=True,  # Enable SPAN margining
        core_enabled=True,  # Enable CORE analytics
        product_code="SR3",  # Product code for IRS
    )

    connector = CMEClearingConnector(config)

    try:
        print("\nConnecting to CME Clearing...")
        await connector.connect()
        print("Connected to CME Clearing")

        trade = create_sample_irs_trade()
        print(f"\nSubmitting trade {trade.trade_id}...")

        response = await connector.submit_trade(trade)
        print(f"\nTrade accepted!")
        print(f"  CME Trade ID: {response.ccp_trade_id}")
        print(f"  SPAN Margin: ${response.metadata.get('span_margin', 0):,.2f}")
        print(f"  Globex Matched: {response.metadata.get('globex_matched')}")

        # Get SPAN margin requirements
        margin = await connector.get_margin_requirements()
        print(f"\nSPAN Margin Breakdown:")
        print(f"  Total Margin: ${margin['total_margin']:,.2f}")
        print(f"  Scanning Risk: ${margin['scanning_risk']:,.2f}")
        print(f"  Inter-Commodity Spread: ${margin.get('inter_commodity_spread', 0):,.2f}")

        # Get CORE analytics if enabled
        if config.core_enabled and "core_analytics" in margin:
            core = margin["core_analytics"]
            print(f"\nCORE Analytics:")
            print(f"  Portfolio VaR: ${core.get('portfolio_var', 0):,.2f}")
            print(f"  Stress Loss: ${core.get('stress_loss', 0):,.2f}")

    except Exception as e:
        print(f"\nError: {e}")

    finally:
        await connector.disconnect()
        print("\nDisconnected from CME Clearing")


async def example_ice_clear():
    """Example: Submit CDS trade to ICE Clear Credit."""
    print("\n" + "=" * 60)
    print("ICE Clear Credit Example")
    print("=" * 60)

    config = ICEClearConfig(
        ccp_name="ICE",
        member_id="MEMBER789",
        service=ICEClearService.CREDIT,  # ICE Clear Credit
        clearing_member_code="CLS123",
        api_endpoint="https://api.theice.com/clear",
        api_key="your_api_key_here",
        compression_enabled=True,  # Enable triReduce compression
        trade_vault_enabled=True,  # Enable Trade Vault reporting
    )

    connector = ICEClearConnector(config)

    try:
        print("\nConnecting to ICE Clear Credit...")
        await connector.connect()
        print("Connected to ICE Clear Credit")

        # Create CDS trade (modified from IRS example)
        trade = create_sample_irs_trade()
        trade.product_type = ProductType.CDS

        print(f"\nSubmitting CDS trade {trade.trade_id}...")
        response = await connector.submit_trade(trade)

        print(f"\nTrade accepted!")
        print(f"  ICE Trade ID: {response.ccp_trade_id}")
        print(f"  Trade Vault Registered: {response.metadata.get('trade_vault_registered')}")

        if "margin_breakdown" in response.metadata:
            margin = response.metadata["margin_breakdown"]
            print(f"\nICE Margin Breakdown:")
            print(f"  Total Margin: ${margin['total_margin']}")
            print(f"  Jump-to-Default: ${margin.get('jump_to_default', 0)}")
            print(f"  Concentration Margin: ${margin.get('concentration_margin', 0)}")

    except Exception as e:
        print(f"\nError: {e}")

    finally:
        await connector.disconnect()
        print("\nDisconnected from ICE Clear")


async def example_eurex_clearing():
    """Example: Submit trade to Eurex Clearing with Prisma margin."""
    print("\n" + "=" * 60)
    print("Eurex Clearing Example")
    print("=" * 60)

    config = EurexClearingConfig(
        ccp_name="EUREX",
        member_id="MEMBER999",
        clearing_member_id="CLM999",
        participant_code="PART99",
        asset_class=EurexAssetClass.FIXED_INCOME_DERIVATIVES,
        api_endpoint="https://api.eurex.com/clearing",
        api_key="your_api_key_here",
        prisma_enabled=True,  # Enable Prisma margining
        cross_margining_enabled=True,  # Enable cross-margining
        c7_connectivity=True,  # Use C7 clearing system
    )

    connector = EurexClearingConnector(config)

    try:
        print("\nConnecting to Eurex Clearing...")
        await connector.connect()
        print("Connected to Eurex Clearing (C7 System)")

        trade = create_sample_irs_trade()
        print(f"\nSubmitting trade {trade.trade_id}...")

        response = await connector.submit_trade(trade)
        print(f"\nTrade accepted!")
        print(f"  Eurex Trade ID: {response.ccp_trade_id}")
        print(f"  C7 Reference: {response.metadata.get('c7_reference')}")

        if "prisma_margin" in response.metadata:
            prisma = response.metadata["prisma_margin"]
            print(f"\nPrisma Margin Breakdown:")
            print(f"  Total Margin: EUR {prisma['total_margin']}")
            print(f"  Core Margin: EUR {prisma['core_margin']}")
            print(f"  Cross-Margining Benefit: EUR {prisma.get('cross_margining_benefit', 0)}")

        # Get cross-margining benefits
        margin = await connector.get_margin_requirements()
        if "cross_margining" in margin:
            print(f"\nCross-Margining Benefits:")
            cross = margin["cross_margining"]
            print(f"  Total Benefit: EUR {cross.get('total_benefit', 0):,.2f}")

    except Exception as e:
        print(f"\nError: {e}")

    finally:
        await connector.disconnect()
        print("\nDisconnected from Eurex Clearing")


async def main():
    """Run all CCP integration examples."""
    print("\n" + "=" * 60)
    print("Neutryx CCP Clearing Integration Examples")
    print("=" * 60)
    print("\nThese examples demonstrate connectivity to major clearing houses.")
    print("Replace API keys and credentials with your actual values.")
    print("\nNote: These examples use simulated endpoints. Update to production")
    print("endpoints when deploying to production.")

    # Run examples (comment out as needed)
    await example_lch_swapclear()
    # await example_cme_clearing()
    # await example_ice_clear()
    # await example_eurex_clearing()

    print("\n" + "=" * 60)
    print("Examples Complete")
    print("=" * 60)


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())
