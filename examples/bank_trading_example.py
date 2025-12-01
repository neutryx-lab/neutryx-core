"""Example usage of Bank Trade Repository and Client Repository system.

This example demonstrates:
1. Setting up database connection and repositories
2. Creating and managing counterparties (clients)
3. Creating and managing CSA agreements
4. Executing trades through the trade execution service
5. Querying trades and exposures
6. Managing trade lifecycle

Prerequisites
-------------
- PostgreSQL database running
- asyncpg installed: pip install asyncpg
- Database created and accessible

Environment Variables
--------------------
Set these before running:
- DATABASE_HOST (default: localhost)
- DATABASE_PORT (default: 5432)
- DATABASE_NAME (required)
- DATABASE_USER (required)
- DATABASE_PASSWORD (required)
"""
import asyncio
from datetime import date, timedelta
from decimal import Decimal

from neutryx.integrations.databases.base import DatabaseConfig
from neutryx.portfolio.bank_connection_manager import BankConnectionManager, RepositoryFactory
from neutryx.portfolio.trade_execution_service import TradeExecutionService
from neutryx.portfolio.contracts.counterparty import (
    Counterparty,
    CounterpartyCredit,
    CreditRating,
    EntityType,
)
from neutryx.portfolio.contracts.csa import (
    CSA,
    CollateralTerms,
    ThresholdTerms,
    EligibleCollateral,
    CollateralType,
    ValuationFrequency,
    DisputeResolution,
)
from neutryx.portfolio.contracts.trade import Trade, TradeStatus, ProductType, SettlementType


async def example_1_setup_database():
    """Example 1: Set up database connection and initialize schemas."""
    print("\n" + "="*80)
    print("Example 1: Database Setup and Initialization")
    print("="*80)

    # Create database configuration
    config = DatabaseConfig(
        host="localhost",
        port=5432,
        database="neutryx_bank",  # Change this to your database name
        user="postgres",           # Change this to your username
        password="postgres",       # Change this to your password
    )

    # Create connection manager
    manager = BankConnectionManager(
        config=config,
        trade_schema="trading",
        client_schema="clients",
    )

    try:
        # Initialize schemas
        await manager.initialize_schemas()
        print("\n✓ Database schemas initialized successfully")

        # Perform health check
        health = await manager.health_check()
        print(f"\n✓ Health Check: {health}")

    finally:
        await manager.disconnect()


async def example_2_create_counterparties():
    """Example 2: Create and manage counterparties (clients)."""
    print("\n" + "="*80)
    print("Example 2: Creating Counterparties")
    print("="*80)

    config = DatabaseConfig(
        host="localhost",
        port=5432,
        database="neutryx_bank",
        user="postgres",
        password="postgres",
    )

    manager = BankConnectionManager(config=config)

    try:
        await manager.connect()

        # Create a corporate counterparty
        corporate_cp = Counterparty(
            id="CP001",
            name="ABC Corporation",
            entity_type=EntityType.CORPORATE,
            lei="ABC123456789CORPORATE",  # 20 chars required
            jurisdiction="US",
            credit=CounterpartyCredit(
                rating=CreditRating.A,
                lgd=0.4,
                credit_spread_bps=150.0,
            ),
            is_bank=False,
        )

        await manager.counterparty_repo.save_async(corporate_cp)
        print(f"\n✓ Created counterparty: {corporate_cp}")

        # Create a bank counterparty
        bank_cp = Counterparty(
            id="CP002",
            name="Global Bank Ltd",
            entity_type=EntityType.FINANCIAL,
            lei="BANK123456789FINANCI",
            jurisdiction="GB",
            credit=CounterpartyCredit(
                rating=CreditRating.AA,
                lgd=0.3,
                credit_spread_bps=50.0,
            ),
            is_bank=True,
        )

        await manager.counterparty_repo.save_async(bank_cp)
        print(f"✓ Created bank counterparty: {bank_cp}")

        # Retrieve all counterparties
        all_counterparties = await manager.counterparty_repo.find_all_async()
        print(f"\n✓ Total counterparties in database: {len(all_counterparties)}")

        # Find banks
        banks = await manager.counterparty_repo.find_banks_async()
        print(f"✓ Total bank counterparties: {len(banks)}")

    finally:
        await manager.disconnect()


async def example_3_create_csa_agreements():
    """Example 3: Create CSA agreements between parties."""
    print("\n" + "="*80)
    print("Example 3: Creating CSA Agreements")
    print("="*80)

    config = DatabaseConfig(
        host="localhost",
        port=5432,
        database="neutryx_bank",
        user="postgres",
        password="postgres",
    )

    manager = BankConnectionManager(config=config)

    try:
        await manager.connect()

        # Create CSA between our bank and counterparty
        # Assume our bank is CP002 (Global Bank Ltd)
        csa = CSA(
            id="CSA001",
            party_a_id="CP002",  # Our bank
            party_b_id="CP001",  # Corporate counterparty
            effective_date="2025-01-01",
            threshold_terms=ThresholdTerms(
                threshold_party_a=0.0,      # Our bank: fully collateralized
                threshold_party_b=5_000_000.0,  # Client: $5M threshold
                mta_party_a=250_000.0,      # $250k minimum transfer
                mta_party_b=250_000.0,
                independent_amount_party_a=0.0,
                independent_amount_party_b=0.0,
                rounding=100_000.0,         # Round to nearest $100k
            ),
            collateral_terms=CollateralTerms(
                base_currency="USD",
                valuation_frequency=ValuationFrequency.DAILY,
                valuation_time="10:00:00 EST",
                dispute_threshold=500_000.0,  # $500k dispute threshold
                dispute_resolution=DisputeResolution.MARKET_QUOTATION,
                eligible_collateral=[
                    EligibleCollateral(
                        collateral_type=CollateralType.CASH,
                        currency="USD",
                        haircut=0.0,  # No haircut on cash
                    ),
                    EligibleCollateral(
                        collateral_type=CollateralType.GOVERNMENT_BOND,
                        currency="USD",
                        haircut=0.02,  # 2% haircut on govt bonds
                        concentration_limit=0.5,  # Max 50% of collateral
                        rating_threshold="AA-",
                        maturity_max_years=10.0,
                    ),
                ],
            ),
            initial_margin_required=False,
            variation_margin_required=True,
            rehypothecation_allowed=False,
            segregation_required=True,
        )

        await manager.csa_repo.save_async(csa)
        print(f"\n✓ Created CSA agreement: {csa}")

        # Retrieve CSA
        retrieved_csa = await manager.csa_repo.find_by_id_async("CSA001")
        print(f"✓ Retrieved CSA: {retrieved_csa}")

        # Find CSAs for a counterparty
        cp_csas = await manager.csa_repo.find_by_counterparty_async("CP001")
        print(f"✓ CSAs for CP001: {len(cp_csas)}")

    finally:
        await manager.disconnect()


async def example_4_execute_trades():
    """Example 4: Execute trades using the trade execution service."""
    print("\n" + "="*80)
    print("Example 4: Executing Trades")
    print("="*80)

    config = DatabaseConfig(
        host="localhost",
        port=5432,
        database="neutryx_bank",
        user="postgres",
        password="postgres",
    )

    manager = BankConnectionManager(config=config)

    try:
        await manager.connect()

        # Create trade execution service
        execution_service = TradeExecutionService(manager)

        # Book a new Interest Rate Swap trade
        print("\n--- Booking IRS Trade ---")
        result = await execution_service.book_trade(
            counterparty_id="CP001",
            product_type=ProductType.INTEREST_RATE_SWAP,
            trade_date=date.today(),
            notional=10_000_000.0,
            currency="USD",
            maturity_date=date.today() + timedelta(days=365*5),  # 5 years
            product_details={
                "fixed_rate": 0.045,
                "floating_index": "USD-LIBOR-3M",
                "payment_frequency": "Quarterly",
            },
            validate_counterparty=True,
            validate_csa=True,
        )

        print(f"✓ Trade booking result: {result.status.value}")
        if result.is_success():
            print(f"  Trade ID: {result.trade_id}")
        else:
            print(f"  Error: {result.error_message}")

        # Confirm the trade
        if result.is_success():
            print(f"\n--- Confirming Trade {result.trade_id} ---")
            confirm_result = await execution_service.confirm_trade(result.trade_id)
            print(f"✓ Confirmation result: {confirm_result.status.value}")

        # Book another trade (FX Option)
        print("\n--- Booking FX Option Trade ---")
        result2 = await execution_service.book_trade(
            counterparty_id="CP001",
            product_type=ProductType.FX_OPTION,
            trade_date=date.today(),
            notional=5_000_000.0,
            currency="EUR",
            maturity_date=date.today() + timedelta(days=90),
            product_details={
                "option_type": "Call",
                "strike": 1.10,
                "underlying": "EUR/USD",
            },
            validate_counterparty=True,
        )

        print(f"✓ Trade booking result: {result2.status.value}")
        if result2.is_success():
            print(f"  Trade ID: {result2.trade_id}")

            # Auto-confirm this trade
            confirm_result2 = await execution_service.confirm_trade(result2.trade_id)
            print(f"  Confirmed: {confirm_result2.is_success()}")

    finally:
        await manager.disconnect()


async def example_5_query_trades_and_exposure():
    """Example 5: Query trades and calculate exposure."""
    print("\n" + "="*80)
    print("Example 5: Querying Trades and Exposure")
    print("="*80)

    config = DatabaseConfig(
        host="localhost",
        port=5432,
        database="neutryx_bank",
        user="postgres",
        password="postgres",
    )

    manager = BankConnectionManager(config=config)

    try:
        await manager.connect()

        execution_service = TradeExecutionService(manager)

        # Get all active trades for counterparty
        print("\n--- Active Trades for CP001 ---")
        active_trades = await execution_service.get_active_trades_by_counterparty("CP001")
        print(f"✓ Found {len(active_trades)} active trades")

        for trade in active_trades:
            print(f"\n  Trade: {trade.id}")
            print(f"  Product: {trade.product_type.value}")
            print(f"  Notional: {trade.notional:,.0f} {trade.currency}")
            print(f"  Maturity: {trade.maturity_date}")
            print(f"  Status: {trade.status.value}")

        # Get counterparty exposure
        print("\n--- Counterparty Exposure Summary ---")
        exposure = await execution_service.get_counterparty_exposure("CP001")

        if exposure["counterparty"]:
            print(f"✓ Counterparty: {exposure['counterparty'].name}")
            print(f"  Active Trades: {exposure['trade_count']}")
            print(f"  Total MTM: ${exposure['total_mtm']:,.2f}")
            print(f"  CSA Agreements: {len(exposure['csas'])}")

        # Get all trades by product type
        print("\n--- All IRS Trades ---")
        all_trades = await manager.trade_repo.find_all_async()
        irs_trades = [t for t in all_trades if t.product_type == ProductType.INTEREST_RATE_SWAP]
        print(f"✓ Found {len(irs_trades)} IRS trades")

        # Get pending trades
        print("\n--- Pending Trades ---")
        pending = await execution_service.get_pending_trades()
        print(f"✓ Found {len(pending)} pending trades")

    finally:
        await manager.disconnect()


async def example_6_trade_lifecycle():
    """Example 6: Manage complete trade lifecycle."""
    print("\n" + "="*80)
    print("Example 6: Trade Lifecycle Management")
    print("="*80)

    config = DatabaseConfig(
        host="localhost",
        port=5432,
        database="neutryx_bank",
        user="postgres",
        password="postgres",
    )

    manager = BankConnectionManager(config=config)

    try:
        await manager.connect()

        execution_service = TradeExecutionService(manager)

        # 1. Book a trade (Pending)
        print("\n--- Step 1: Book Trade ---")
        result = await execution_service.book_trade(
            counterparty_id="CP001",
            product_type=ProductType.FORWARD,
            trade_date=date.today(),
            notional=1_000_000.0,
            currency="GBP",
            maturity_date=date.today() + timedelta(days=180),
        )
        print(f"✓ Trade booked: {result.trade_id}")

        trade_id = result.trade_id

        # 2. Confirm the trade (Active)
        print("\n--- Step 2: Confirm Trade ---")
        confirm_result = await execution_service.confirm_trade(trade_id)
        print(f"✓ Trade confirmed: {confirm_result.is_success()}")

        # 3. Check trade status
        print("\n--- Step 3: Check Trade Status ---")
        trade = await manager.trade_repo.find_by_id_async(trade_id)
        print(f"✓ Trade status: {trade.status.value}")

        # 4. Terminate the trade early
        print("\n--- Step 4: Early Termination ---")
        termination_date = date.today() + timedelta(days=30)
        terminate_result = await execution_service.terminate_trade(trade_id, termination_date)
        print(f"✓ Trade terminated: {terminate_result.is_success()}")

        # 5. Verify final status
        print("\n--- Step 5: Verify Final Status ---")
        trade = await manager.trade_repo.find_by_id_async(trade_id)
        print(f"✓ Final status: {trade.status.value}")
        print(f"  Termination date: {trade.maturity_date}")

    finally:
        await manager.disconnect()


async def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("Bank Trading System - Complete Example Suite")
    print("="*80)
    print("\nNOTE: Update database credentials in each example before running!")
    print("      Current defaults: host=localhost, db=neutryx_bank, user=postgres")

    try:
        # Run all examples in sequence
        await example_1_setup_database()
        await example_2_create_counterparties()
        await example_3_create_csa_agreements()
        await example_4_execute_trades()
        await example_5_query_trades_and_exposure()
        await example_6_trade_lifecycle()

        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
