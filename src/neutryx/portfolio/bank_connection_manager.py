"""Bank connection manager for integrating trade and client repositories.

This module provides a centralized connection manager that coordinates:
- Trade repository connections
- Client/counterparty repository connections
- CSA agreement repository connections
- Transaction management across repositories
- Lifecycle management for database connections
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Optional, Tuple

from neutryx.integrations.databases.base import DatabaseConfig
from neutryx.portfolio.repository_postgres import PostgresTradeRepository
from neutryx.portfolio.client_repository import (
    PostgresCounterpartyRepository,
    PostgresCSARepository,
)


class BankConnectionManager:
    """Centralized manager for bank trading and client data repositories.

    This manager provides:
    - Unified connection management for all repositories
    - Transaction coordination across multiple tables
    - Graceful connection lifecycle management
    - Configuration sharing across repositories

    Attributes
    ----------
    config : DatabaseConfig
        Shared database configuration
    trade_repo : PostgresTradeRepository
        Repository for trade data
    counterparty_repo : PostgresCounterpartyRepository
        Repository for counterparty data
    csa_repo : PostgresCSARepository
        Repository for CSA agreements
    """

    def __init__(
        self,
        config: DatabaseConfig,
        trade_schema: str = "trading",
        client_schema: str = "clients",
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        """Initialize bank connection manager.

        Parameters
        ----------
        config : DatabaseConfig
            Database connection configuration
        trade_schema : str
            Schema name for trade data (default: "trading")
        client_schema : str
            Schema name for client data (default: "clients")
        loop : asyncio.AbstractEventLoop, optional
            Event loop for async operations
        """
        self.config = config
        self._loop = loop or asyncio.get_event_loop()

        # Initialize repositories with shared configuration
        self.trade_repo = PostgresTradeRepository(
            config=config,
            schema=trade_schema,
            loop=self._loop,
        )

        self.counterparty_repo = PostgresCounterpartyRepository(
            config=config,
            schema=client_schema,
            loop=self._loop,
        )

        self.csa_repo = PostgresCSARepository(
            config=config,
            schema=client_schema,
            loop=self._loop,
        )

        self._connected = False

    async def connect(self) -> None:
        """Connect all repositories to the database.

        This method establishes connections for:
        - Trade repository
        - Counterparty repository
        - CSA repository

        All repositories share the same connection pool configuration.
        """
        if self._connected:
            return

        # Connect all repositories
        await self.trade_repo.connect()
        await self.counterparty_repo.connect()
        await self.csa_repo.connect()

        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect all repositories from the database.

        This method gracefully closes all database connections.
        """
        if not self._connected:
            return

        # Disconnect all repositories
        await self.trade_repo.disconnect()
        await self.counterparty_repo.disconnect()
        await self.csa_repo.disconnect()

        self._connected = False

    def is_connected(self) -> bool:
        """Check if connection manager is connected.

        Returns
        -------
        bool
            True if all repositories are connected
        """
        return self._connected

    @asynccontextmanager
    async def connection(self):
        """Context manager for managing database connections.

        Usage
        -----
        ```python
        async with manager.connection():
            # Perform database operations
            await manager.trade_repo.save_async(trade)
            await manager.counterparty_repo.save_async(counterparty)
        ```

        Yields
        ------
        BankConnectionManager
            The connection manager instance with active connections
        """
        await self.connect()
        try:
            yield self
        finally:
            # Don't disconnect on exit to allow connection pooling
            # Call disconnect() explicitly when done with the manager
            pass

    async def initialize_schemas(self) -> None:
        """Initialize all database schemas and tables.

        This method creates all necessary schemas and tables if they don't exist.
        It's safe to call multiple times (idempotent).
        """
        await self.connect()

        # Schema creation is handled automatically by each repository
        # when they connect, so this just ensures connection is established
        print(f"✓ Trade schema initialized: {self.trade_repo.schema}")
        print(f"✓ Client schema initialized: {self.counterparty_repo.schema}")

    async def get_trade_with_counterparty(self, trade_id: str):
        """Get a trade along with its counterparty information.

        Parameters
        ----------
        trade_id : str
            Trade ID to retrieve

        Returns
        -------
        tuple[Trade, Counterparty] or None
            Trade and counterparty objects, or None if trade not found
        """
        await self.connect()

        trade = await self.trade_repo.find_by_id_async(trade_id)
        if not trade:
            return None

        counterparty = await self.counterparty_repo.find_by_id_async(trade.counterparty_id)

        return trade, counterparty

    async def get_counterparty_trades_with_csa(self, counterparty_id: str):
        """Get all trades for a counterparty along with their CSA agreement.

        Parameters
        ----------
        counterparty_id : str
            Counterparty ID

        Returns
        -------
        dict
            Dictionary containing:
            - 'counterparty': Counterparty object
            - 'trades': List of Trade objects
            - 'csas': List of CSA agreements involving this counterparty
        """
        await self.connect()

        counterparty = await self.counterparty_repo.find_by_id_async(counterparty_id)
        trades = await self.trade_repo.find_by_counterparty_async(counterparty_id)
        csas = await self.csa_repo.find_by_counterparty_async(counterparty_id)

        return {
            "counterparty": counterparty,
            "trades": trades,
            "csas": csas,
        }

    async def health_check(self) -> dict:
        """Perform health check on all repositories.

        Returns
        -------
        dict
            Health check results with connection status and counts
        """
        try:
            await self.connect()

            trade_count = await self.trade_repo.count_async()
            counterparty_count = len(await self.counterparty_repo.find_all_async())
            csa_count = len(await self.csa_repo.find_all_async())

            return {
                "status": "healthy",
                "connected": True,
                "trades": trade_count,
                "counterparties": counterparty_count,
                "csa_agreements": csa_count,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
            }

    def __repr__(self) -> str:
        """String representation."""
        status = "connected" if self._connected else "disconnected"
        return (
            f"BankConnectionManager("
            f"status={status}, "
            f"trade_schema={self.trade_repo.schema}, "
            f"client_schema={self.counterparty_repo.schema})"
        )


class RepositoryFactory:
    """Factory for creating repository instances with database backends."""

    @staticmethod
    def create_postgres_repositories(
        config: DatabaseConfig,
        trade_schema: str = "trading",
        client_schema: str = "clients",
    ) -> BankConnectionManager:
        """Create a complete set of PostgreSQL repositories.

        Parameters
        ----------
        config : DatabaseConfig
            Database connection configuration
        trade_schema : str
            Schema name for trade data
        client_schema : str
            Schema name for client data

        Returns
        -------
        BankConnectionManager
            Configured connection manager with all repositories
        """
        return BankConnectionManager(
            config=config,
            trade_schema=trade_schema,
            client_schema=client_schema,
        )

    @staticmethod
    def create_from_env() -> BankConnectionManager:
        """Create repositories from environment variables.

        Reads configuration from:
        - DATABASE_HOST (default: localhost)
        - DATABASE_PORT (default: 5432)
        - DATABASE_NAME (required)
        - DATABASE_USER (required)
        - DATABASE_PASSWORD (required)
        - TRADE_SCHEMA (default: trading)
        - CLIENT_SCHEMA (default: clients)

        Returns
        -------
        BankConnectionManager
            Configured connection manager
        """
        import os

        config = DatabaseConfig(
            host=os.getenv("DATABASE_HOST", "localhost"),
            port=int(os.getenv("DATABASE_PORT", "5432")),
            database=os.getenv("DATABASE_NAME"),
            user=os.getenv("DATABASE_USER"),
            password=os.getenv("DATABASE_PASSWORD"),
        )

        return BankConnectionManager(
            config=config,
            trade_schema=os.getenv("TRADE_SCHEMA", "trading"),
            client_schema=os.getenv("CLIENT_SCHEMA", "clients"),
        )


__all__ = ["BankConnectionManager", "RepositoryFactory"]
