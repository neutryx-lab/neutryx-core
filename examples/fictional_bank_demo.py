"""Comprehensive demonstration of Fictional Bank capabilities.

This example showcases the complete fictional bank system including:
1. Bank initialization and setup
2. Loading fictional portfolio with trades
3. Executing trading scenarios
4. Generating comprehensive reports
5. Real-time exposure monitoring
6. Database persistence and retrieval

Prerequisites
-------------
- PostgreSQL database running
- asyncpg installed: pip install asyncpg
- Database created: neutryx_bank

Environment Variables (Optional)
--------------------------------
- DATABASE_HOST (default: localhost)
- DATABASE_PORT (default: 5432)
- DATABASE_NAME (default: neutryx_bank)
- DATABASE_USER (default: postgres)
- DATABASE_PASSWORD (default: postgres)
"""
import asyncio
from datetime import date
import os

from neutryx.integrations.databases.base import DatabaseConfig
from neutryx.portfolio.fictional_bank import create_fictional_bank
from neutryx.portfolio.trading_scenarios import (
    DailyTradingScenario,
    CounterpartyOnboardingScenario,
    ExposureMonitoringScenario,
    run_all_scenarios,
)
from neutryx.portfolio.bank_reports import BankReportGenerator


async def demo_1_bank_initialization():
    """Demo 1: Initialize fictional bank with database."""
    print("\n" + "=" * 80)
    print("DEMO 1: Bank Initialization")
    print("=" * 80)

    # Create database configuration
    config = DatabaseConfig(
        host=os.getenv("DATABASE_HOST", "localhost"),
        port=int(os.getenv("DATABASE_PORT", "5432")),
        database=os.getenv("DATABASE_NAME", "neutryx_bank"),
        user=os.getenv("DATABASE_USER", "postgres"),
        password=os.getenv("DATABASE_PASSWORD", "postgres"),
    )

    # Create and initialize bank
    bank = await create_fictional_bank(
        database_config=config,
        load_portfolio=True,  # Load standard fictional portfolio
    )

    # Health check
    health = await bank.health_check()
    print(f"\n📊 Health Check:")
    print(f"  Status: {health['status'] if 'status' in health else 'OK'}")
    print(f"  Database Connected: {health['database_connected']}")
    print(f"  Counterparties: {health['portfolio']['counterparties']}")
    print(f"  Trades: {health['portfolio']['trades']}")
    print(f"  CSAs: {health['portfolio']['csas']}")

    return bank


async def demo_2_trading_scenarios(bank):
    """Demo 2: Execute various trading scenarios."""
    print("\n" + "=" * 80)
    print("DEMO 2: Trading Scenarios")
    print("=" * 80)

    # Scenario 1: Daily trading
    print("\n📈 Scenario 1: Daily Trading Operations")
    daily_scenario = DailyTradingScenario()
    await daily_scenario.execute(bank)

    # Scenario 2: Counterparty onboarding
    print("\n📝 Scenario 2: Counterparty Onboarding")
    onboarding_scenario = CounterpartyOnboardingScenario()
    await onboarding_scenario.execute(bank)

    # Scenario 3: Exposure monitoring
    print("\n📊 Scenario 3: Exposure Monitoring")
    exposure_scenario = ExposureMonitoringScenario()
    await exposure_scenario.execute(bank)


async def demo_3_reporting(bank):
    """Demo 3: Generate comprehensive reports."""
    print("\n" + "=" * 80)
    print("DEMO 3: Reporting and Analytics")
    print("=" * 80)

    report_generator = BankReportGenerator(bank)

    # Executive Dashboard
    print("\n📊 Generating Executive Dashboard...\n")
    dashboard = await report_generator.generate_executive_dashboard()
    report_generator.print_executive_dashboard(dashboard)

    # Daily P&L Report
    print("\n📈 Daily P&L Report:")
    pnl_report = await report_generator.generate_daily_pnl_report()
    print(f"  Total MTM: ${pnl_report['total_mtm']:,.2f}")
    print(f"  Gross Notional: ${pnl_report['gross_notional']:,.2f}")
    print(f"\n  By Desk:")
    for desk_id, desk_data in pnl_report["by_desk"].items():
        print(
            f"    {desk_data['name']:<30} MTM: ${desk_data['mtm']:>12,.2f}  "
            f"Notional: ${desk_data['notional']:>12,.2f}"
        )

    # Risk Report
    print("\n\n⚠️  Risk Report:")
    risk_report = await report_generator.generate_risk_report()
    print(f"  Total Exposure: ${risk_report['concentration_metrics']['total_exposure']:,.2f}")
    print(
        f"  Collateralized: {risk_report['credit_metrics']['collateralized_percentage']:.1f}%"
    )
    print("\n  Top 3 Exposures:")
    for i, exp in enumerate(risk_report["concentration_metrics"]["top_exposures"][:3], 1):
        print(
            f"    {i}. {exp['name']:<30} ${exp['exposure']:>12,.2f}  [{exp['rating']}]"
        )

    # Counterparty Credit Report
    print("\n\n🏦 Counterparty Credit Summary:")
    credit_report = await report_generator.generate_counterparty_credit_report()
    summary = credit_report["summary"]
    print(f"  Total Counterparties: {summary['total_counterparties']}")
    print(f"  With CSA: {summary['with_csa']}")
    print(f"  Without CSA: {summary['without_csa']}")
    print(f"  Banks: {summary['banks']}")
    print(f"  Corporates: {summary['corporates']}")
    print(f"  Funds: {summary['funds']}")

    # Export reports
    print("\n\n💾 Exporting reports to JSON...")
    report_generator.export_report_to_json(dashboard, "executive_dashboard.json")
    report_generator.export_report_to_json(pnl_report, "daily_pnl_report.json")
    report_generator.export_report_to_json(risk_report, "risk_report.json")
    report_generator.export_report_to_json(credit_report, "credit_report.json")


async def demo_4_counterparty_analysis(bank):
    """Demo 4: Detailed counterparty analysis."""
    print("\n" + "=" * 80)
    print("DEMO 4: Counterparty Analysis")
    print("=" * 80)

    # Analyze each counterparty
    for cp_id, counterparty in list(bank.portfolio.counterparties.items())[:3]:
        print(f"\n📊 {counterparty.name} ({cp_id}):")
        print(f"  Type: {counterparty.entity_type.value}")
        print(f"  LEI: {counterparty.lei}")
        print(f"  Jurisdiction: {counterparty.jurisdiction}")

        if counterparty.credit:
            print(f"  Rating: {counterparty.credit.rating.value if counterparty.credit.rating else 'NR'}")
            print(f"  LGD: {counterparty.credit.lgd:.2%}")
            if counterparty.credit.credit_spread_bps:
                print(f"  Credit Spread: {counterparty.credit.credit_spread_bps:.1f} bps")

        # Get exposure
        exposure = await bank.get_counterparty_exposure(cp_id)
        print(f"\n  Exposure:")
        print(f"    Active Trades: {exposure['trade_count']}")
        print(f"    Total MTM: ${exposure['total_mtm']:,.2f}")
        print(f"    Has CSA: {'Yes' if len(exposure['csas']) > 0 else 'No'}")

        # Show trades
        print(f"\n  Active Trades:")
        for trade in exposure["active_trades"][:3]:  # Show first 3
            print(f"    - {trade.id}: {trade.product_type.value}")
            print(f"      Notional: ${trade.notional:,.0f} {trade.currency}")
            print(f"      Maturity: {trade.maturity_date}")
            if trade.mtm:
                print(f"      MTM: ${trade.mtm:,.2f}")


async def demo_5_database_operations(bank):
    """Demo 5: Database operations and queries."""
    print("\n" + "=" * 80)
    print("DEMO 5: Database Operations")
    print("=" * 80)

    if not bank.manager:
        print("\n⚠️  No database configured. Skipping database demo.")
        return

    # Query statistics from database
    print("\n📊 Database Statistics:")

    trade_count = await bank.manager.trade_repo.count_async()
    print(f"  Total Trades in DB: {trade_count}")

    counterparties = await bank.manager.counterparty_repo.find_all_async()
    print(f"  Total Counterparties in DB: {len(counterparties)}")

    csas = await bank.manager.csa_repo.find_all_async()
    print(f"  Total CSAs in DB: {len(csas)}")

    # Query specific counterparty
    print("\n\n🔍 Querying Specific Counterparty:")
    cp = await bank.manager.counterparty_repo.find_by_id_async("CP_BANK_AAA")
    if cp:
        print(f"  Found: {cp.name}")
        print(f"  Rating: {cp.credit.rating.value if cp.credit and cp.credit.rating else 'NR'}")

        # Get trades for this counterparty
        trades = await bank.manager.trade_repo.find_by_counterparty_async("CP_BANK_AAA")
        print(f"  Trades: {len(trades)}")

        # Get CSAs
        csas = await bank.manager.csa_repo.find_by_counterparty_async("CP_BANK_AAA")
        print(f"  CSAs: {len(csas)}")

    # Query by product type
    print("\n\n📈 Trades by Product Type:")
    from neutryx.portfolio.contracts.trade import ProductType

    for product_type in [
        ProductType.INTEREST_RATE_SWAP,
        ProductType.FX_OPTION,
        ProductType.EQUITY_OPTION,
    ]:
        # Count in memory
        memory_trades = bank.portfolio.get_trades_by_product_type(product_type)
        print(f"  {product_type.value}: {len(memory_trades)} trades")


async def demo_6_real_time_monitoring(bank):
    """Demo 6: Real-time monitoring dashboard."""
    print("\n" + "=" * 80)
    print("DEMO 6: Real-Time Monitoring")
    print("=" * 80)

    print("\n📊 Current Portfolio State:\n")

    # Overall statistics
    stats = bank.portfolio.summary()
    print(f"Portfolio Statistics:")
    print(f"  Counterparties: {stats['counterparties']}")
    print(f"  Netting Sets: {stats['netting_sets']}")
    print(f"  Total Trades: {stats['trades']}")
    print(f"  Active Trades: {stats['active_trades']}")

    # By desk
    print(f"\n\nDesk Breakdown:")
    desk_ids = set(t.desk_id for t in bank.portfolio.trades.values() if t.desk_id)
    for desk_id in sorted(desk_ids):
        desk = bank.book_hierarchy.desks.get(desk_id)
        if desk:
            summary = await bank.get_desk_summary(desk_id)
            print(f"\n  {desk.name}:")
            print(f"    Trades: {summary['num_trades']}")
            print(f"    MTM: ${summary['total_mtm']:,.2f}")
            print(f"    Notional: ${summary['total_notional']:,.2f}")

    # Risk concentration
    print(f"\n\nRisk Concentration:")
    total_mtm = bank.portfolio.calculate_total_mtm()

    counterparty_mtms = []
    for cp_id in bank.portfolio.counterparties.keys():
        cp_mtm = bank.portfolio.calculate_net_mtm_by_counterparty(cp_id)
        if cp_mtm != 0:
            counterparty_mtms.append((cp_id, cp_mtm))

    # Sort by absolute MTM
    counterparty_mtms.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"  Top 5 by Absolute MTM:")
    for i, (cp_id, mtm) in enumerate(counterparty_mtms[:5], 1):
        cp = bank.portfolio.counterparties[cp_id]
        concentration = abs(mtm) / abs(total_mtm) * 100 if total_mtm != 0 else 0
        print(f"    {i}. {cp.name:<30} ${mtm:>12,.2f}  ({concentration:.1f}%)")


async def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("FICTIONAL BANK - COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)
    print("\nThis demo showcases the complete fictional bank system with:")
    print("  - Database-backed trade repository")
    print("  - Client and CSA management")
    print("  - Trade execution workflows")
    print("  - Trading scenarios and simulations")
    print("  - Comprehensive reporting and analytics")

    try:
        # Demo 1: Initialize bank
        bank = await demo_1_bank_initialization()

        # Demo 2: Trading scenarios
        await demo_2_trading_scenarios(bank)

        # Demo 3: Reporting
        await demo_3_reporting(bank)

        # Demo 4: Counterparty analysis
        await demo_4_counterparty_analysis(bank)

        # Demo 5: Database operations
        await demo_5_database_operations(bank)

        # Demo 6: Real-time monitoring
        await demo_6_real_time_monitoring(bank)

        # Final summary
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETE")
        print("=" * 80)
        print(f"\n✓ Bank: {bank.name}")
        print(f"✓ Status: {bank}")
        print(f"✓ Total Counterparties: {len(bank.portfolio.counterparties)}")
        print(f"✓ Total Trades: {len(bank.portfolio.trades)}")
        print(f"✓ Total MTM: ${bank.portfolio.calculate_total_mtm():,.2f}")
        print(f"✓ Gross Notional: ${bank.portfolio.calculate_gross_notional():,.2f}")

        print("\n📄 Reports exported to:")
        print("  - executive_dashboard.json")
        print("  - daily_pnl_report.json")
        print("  - risk_report.json")
        print("  - credit_report.json")

        # Cleanup
        await bank.shutdown()

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())
