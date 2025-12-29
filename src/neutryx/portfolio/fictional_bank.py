"""Fictional bank implementation with complete trading infrastructure.

This module provides a fully-featured fictional bank that combines:
- Portfolio management
- Database persistence via Bank Trading System
- Trade execution workflows
- Real-time exposure monitoring
- Reporting and analytics

The fictional bank serves as:
- A complete reference implementation
- Testing infrastructure for XVA and risk systems
- Educational demonstration of trading operations
- Simulation environment for complex scenarios
"""
from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import Dict, List, Optional

from neutryx.integrations.databases.base import DatabaseConfig
from neutryx.portfolio.bank_connection_manager import BankConnectionManager
from neutryx.portfolio.trade_execution_service import (
    TradeExecutionService,
    ExecutionResult,
)
from neutryx.portfolio.contracts.counterparty import Counterparty
from neutryx.portfolio.contracts.csa import CSA
from neutryx.portfolio.contracts.trade import Trade
from neutryx.portfolio.portfolio import Portfolio
from neutryx.portfolio.books import BookHierarchy


class FictionalBank:
    """Complete fictional bank with trading infrastructure.

    This class provides a comprehensive banking simulation that includes:
    - Multi-desk trading operations
    - Client relationship management
    - CSA agreement management
    - Trade execution and lifecycle
    - Real-time exposure monitoring
    - Reporting and analytics

    Attributes
    ----------
    name : str
        Name of the fictional bank
    lei : str
        Legal Entity Identifier for the bank
    manager : BankConnectionManager
        Database connection manager
    execution_service : TradeExecutionService
        Trade execution service
    portfolio : Portfolio
        In-memory portfolio representation
    book_hierarchy : BookHierarchy
        Organization structure
    """

    def __init__(
        self,
        name: str = "Global Investment Bank Ltd",
        lei: str = "529900T8BM49AURSDO55",
        jurisdiction: str = "GB",
        database_config: Optional[DatabaseConfig] = None,
    ):
        """Initialize fictional bank.

        Parameters
        ----------
        name : str
            Bank name
        lei : str
            Legal Entity Identifier
        jurisdiction : str
            Jurisdiction code
        database_config : DatabaseConfig, optional
            Database configuration (if None, uses in-memory only)
        """
        self.name = name
        self.lei = lei
        self.jurisdiction = jurisdiction
        self.bank_id = "OUR_INSTITUTION"

        # Database integration
        self.manager = None
        self.execution_service = None
        if database_config:
            self.manager = BankConnectionManager(config=database_config)
            self.execution_service = TradeExecutionService(self.manager)

        # In-memory structures
        self.portfolio = Portfolio(name=f"{name} Portfolio", base_currency="USD")
        self.book_hierarchy = BookHierarchy()

        # Statistics
        self._initialized = False

    async def initialize(self, create_schema: bool = True) -> None:
        """Initialize bank infrastructure.

        Parameters
        ----------
        create_schema : bool
            Whether to create database schemas
        """
        if self._initialized:
            return

        if self.manager:
            if create_schema:
                await self.manager.initialize_schemas()
            else:
                await self.manager.connect()

        self._initialized = True
        print(f"✓ {self.name} initialized")

    async def shutdown(self) -> None:
        """Shutdown bank infrastructure."""
        if self.manager:
            await self.manager.disconnect()
        self._initialized = False
        print(f"✓ {self.name} shutdown complete")

    async def load_fictional_portfolio(self) -> None:
        """Load the standard fictional portfolio into the bank.

        This loads the test portfolio structure from fictional_portfolio.py
        and persists it to the database if connected.
        """
        from tests.fixtures.fictional_portfolio import create_fictional_portfolio

        # Create fictional portfolio
        portfolio, book_hierarchy = create_fictional_portfolio()

        # Store in memory
        self.portfolio = portfolio
        self.book_hierarchy = book_hierarchy

        # Persist to database if connected
        if self.manager:
            await self._persist_portfolio()

        print(f"✓ Loaded fictional portfolio:")
        print(f"  - {len(self.portfolio.counterparties)} counterparties")
        print(f"  - {len(self.portfolio.csas)} CSA agreements")
        print(f"  - {len(self.portfolio.trades)} trades")
        print(f"  - {len(self.book_hierarchy.books)} books")

    async def _persist_portfolio(self) -> None:
        """Persist current portfolio to database."""
        await self.manager.connect()

        # Persist counterparties
        for counterparty in self.portfolio.counterparties.values():
            await self.manager.counterparty_repo.save_async(counterparty)

        # Persist CSAs
        for csa in self.portfolio.csas.values():
            await self.manager.csa_repo.save_async(csa)

        # Persist trades
        for trade in self.portfolio.trades.values():
            await self.manager.trade_repo.save_async(trade)

        print(f"✓ Portfolio persisted to database")

    async def add_counterparty(
        self, counterparty: Counterparty, persist: bool = True
    ) -> None:
        """Add a counterparty to the bank.

        Parameters
        ----------
        counterparty : Counterparty
            Counterparty to add
        persist : bool
            Whether to persist to database
        """
        self.portfolio.add_counterparty(counterparty)

        if persist and self.manager:
            await self.manager.counterparty_repo.save_async(counterparty)

        print(f"✓ Added counterparty: {counterparty.name} ({counterparty.id})")

    async def add_csa(self, csa: CSA, persist: bool = True) -> None:
        """Add a CSA agreement.

        Parameters
        ----------
        csa : CSA
            CSA agreement to add
        persist : bool
            Whether to persist to database
        """
        self.portfolio.add_csa(csa)

        if persist and self.manager:
            await self.manager.csa_repo.save_async(csa)

        print(f"✓ Added CSA: {csa.id} between {csa.party_a_id} and {csa.party_b_id}")

    async def execute_trade(
        self,
        trade: Trade,
        validate_counterparty: bool = True,
        validate_csa: bool = False,
        auto_confirm: bool = False,
    ) -> ExecutionResult:
        """Execute a trade through the execution service.

        Parameters
        ----------
        trade : Trade
            Trade to execute
        validate_counterparty : bool
            Whether to validate counterparty exists
        validate_csa : bool
            Whether to validate CSA exists
        auto_confirm : bool
            Whether to auto-confirm the trade

        Returns
        -------
        ExecutionResult
            Result of trade execution
        """
        if not self.execution_service:
            raise RuntimeError("Execution service not available (no database configured)")

        result = await self.execution_service.execute_trade(
            trade=trade,
            validate_counterparty=validate_counterparty,
            validate_csa=validate_csa,
            auto_confirm=auto_confirm,
        )

        if result.is_success():
            self.portfolio.add_trade(trade)
            print(f"✓ Executed trade: {trade.id} ({trade.product_type.value})")
        else:
            print(f"✗ Trade execution failed: {result.error_message}")

        return result

    async def book_trade(
        self,
        counterparty_id: str,
        product_type,
        trade_date: date,
        notional: float,
        currency: str,
        maturity_date: Optional[date] = None,
        book_id: Optional[str] = None,
        desk_id: Optional[str] = None,
        trader_id: Optional[str] = None,
        product_details: Optional[Dict] = None,
        validate_counterparty: bool = True,
        validate_csa: bool = False,
        auto_confirm: bool = True,
    ) -> ExecutionResult:
        """Book a new trade with automatic ID generation.

        Parameters
        ----------
        counterparty_id : str
            Counterparty ID
        product_type : ProductType
            Type of product
        trade_date : date
            Trade date
        notional : float
            Notional amount
        currency : str
            Currency code
        maturity_date : date, optional
            Maturity date
        book_id : str, optional
            Book ID
        desk_id : str, optional
            Desk ID
        trader_id : str, optional
            Trader ID
        product_details : dict, optional
            Product-specific details
        validate_counterparty : bool
            Whether to validate counterparty
        validate_csa : bool
            Whether to validate CSA
        auto_confirm : bool
            Whether to auto-confirm

        Returns
        -------
        ExecutionResult
            Result of trade booking
        """
        if not self.execution_service:
            raise RuntimeError("Execution service not available (no database configured)")

        result = await self.execution_service.book_trade(
            counterparty_id=counterparty_id,
            product_type=product_type,
            trade_date=trade_date,
            notional=notional,
            currency=currency,
            maturity_date=maturity_date,
            book_id=book_id,
            product_details=product_details,
            validate_counterparty=validate_counterparty,
            validate_csa=validate_csa,
        )

        if result.is_success():
            if auto_confirm:
                await self.execution_service.confirm_trade(result.trade_id)

            # Reload trade into portfolio
            trade = await self.manager.trade_repo.find_by_id_async(result.trade_id)
            if trade:
                self.portfolio.add_trade(trade)

        return result

    async def get_counterparty_exposure(
        self, counterparty_id: str, as_of_date: Optional[date] = None
    ) -> Dict:
        """Get exposure for a counterparty.

        Parameters
        ----------
        counterparty_id : str
            Counterparty ID
        as_of_date : date, optional
            Valuation date

        Returns
        -------
        dict
            Exposure information
        """
        if self.execution_service:
            return await self.execution_service.get_counterparty_exposure(
                counterparty_id, as_of_date
            )
        else:
            # Calculate from in-memory portfolio
            counterparty = self.portfolio.get_counterparty(counterparty_id)
            trades = self.portfolio.get_trades_by_counterparty(counterparty_id)

            as_of = as_of_date or date.today()
            active_trades = [
                t for t in trades if t.is_active() and not t.is_expired(as_of)
            ]

            total_mtm = sum(t.get_mtm(0.0) for t in active_trades)

            return {
                "counterparty": counterparty,
                "active_trades": active_trades,
                "total_mtm": total_mtm,
                "trade_count": len(active_trades),
                "csas": self.portfolio.get_csas_by_counterparty(counterparty_id),
            }

    async def get_desk_summary(self, desk_id: str) -> Dict:
        """Get summary for a desk.

        Parameters
        ----------
        desk_id : str
            Desk ID

        Returns
        -------
        dict
            Desk summary
        """
        return self.portfolio.get_desk_summary(desk_id)

    async def get_book_summary(self, book_id: str) -> Dict:
        """Get summary for a book.

        Parameters
        ----------
        book_id : str
            Book ID

        Returns
        -------
        dict
            Book summary
        """
        return self.portfolio.get_book_summary(book_id)

    async def generate_daily_report(self, as_of_date: Optional[date] = None) -> Dict:
        """Generate comprehensive daily report.

        Parameters
        ----------
        as_of_date : date, optional
            Report date

        Returns
        -------
        dict
            Daily report with all statistics
        """
        as_of = as_of_date or date.today()

        report = {
            "bank_name": self.name,
            "report_date": as_of.isoformat(),
            "portfolio_statistics": self.portfolio.summary(),
            "total_mtm": self.portfolio.calculate_total_mtm(),
            "gross_notional": self.portfolio.calculate_gross_notional(),
            "counterparties": {},
            "desks": {},
            "books": {},
        }

        # Counterparty exposures
        for cp_id in self.portfolio.counterparties.keys():
            exposure = await self.get_counterparty_exposure(cp_id, as_of)
            counterparty = exposure["counterparty"]

            report["counterparties"][cp_id] = {
                "name": counterparty.name,
                "entity_type": counterparty.entity_type.value,
                "rating": (
                    counterparty.credit.rating.value
                    if counterparty.credit and counterparty.credit.rating
                    else "NR"
                ),
                "active_trades": exposure["trade_count"],
                "total_mtm": exposure["total_mtm"],
                "has_csa": len(exposure["csas"]) > 0,
            }

        # Desk summaries
        desk_ids = set(t.desk_id for t in self.portfolio.trades.values() if t.desk_id)
        for desk_id in desk_ids:
            report["desks"][desk_id] = await self.get_desk_summary(desk_id)

        # Book summaries
        book_ids = set(t.book_id for t in self.portfolio.trades.values() if t.book_id)
        for book_id in book_ids:
            report["books"][book_id] = await self.get_book_summary(book_id)

        return report

    async def health_check(self) -> Dict:
        """Perform health check on bank infrastructure.

        Returns
        -------
        dict
            Health check results
        """
        health = {
            "bank_name": self.name,
            "initialized": self._initialized,
            "database_connected": self.manager is not None and self.manager.is_connected(),
            "portfolio": {
                "counterparties": len(self.portfolio.counterparties),
                "trades": len(self.portfolio.trades),
                "netting_sets": len(self.portfolio.netting_sets),
                "csas": len(self.portfolio.csas),
            },
        }

        if self.manager:
            db_health = await self.manager.health_check()
            health["database"] = db_health

        return health

    def __repr__(self) -> str:
        """String representation."""
        status = "initialized" if self._initialized else "not initialized"
        db_status = "with DB" if self.manager else "in-memory only"
        return (
            f"FictionalBank(name='{self.name}', "
            f"status={status}, "
            f"mode={db_status}, "
            f"trades={len(self.portfolio.trades)})"
        )


async def create_fictional_bank(
    database_config: Optional[DatabaseConfig] = None,
    load_portfolio: bool = True,
) -> FictionalBank:
    """Create and initialize a fictional bank.

    Parameters
    ----------
    database_config : DatabaseConfig, optional
        Database configuration (if None, uses in-memory only)
    load_portfolio : bool
        Whether to load the standard fictional portfolio

    Returns
    -------
    FictionalBank
        Initialized fictional bank
    """
    bank = FictionalBank(
        name="Global Investment Bank Ltd",
        lei="529900T8BM49AURSDO55",
        jurisdiction="GB",
        database_config=database_config,
    )

    await bank.initialize(create_schema=True)

    if load_portfolio:
        await bank.load_fictional_portfolio()

    return bank


__all__ = ["FictionalBank", "create_fictional_bank"]
