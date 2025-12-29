"""PostgreSQL-based repository implementations for trade and entity persistence.

Provides PostgreSQL implementations for:
- Trade storage and retrieval
- Book hierarchy persistence
- Counterparty management
- CSA agreement storage
- Query capabilities with advanced filtering
"""
from __future__ import annotations

import asyncio
import json
from datetime import date
from typing import Any, Dict, List, Optional

from neutryx.portfolio.contracts.counterparty import Counterparty
from neutryx.portfolio.contracts.trade import ProductType, Trade, TradeStatus
from neutryx.portfolio.contracts.csa import CSA
from neutryx.portfolio.books import Book, Desk, LegalEntity, BusinessUnit, Trader
from neutryx.portfolio.repository import (
    TradeRepository,
    BookRepository,
    CounterpartyRepository,
)
from neutryx.integrations.databases.base import DatabaseConfig


class PostgresTradeRepository(TradeRepository):
    """PostgreSQL-based trade repository for production use.

    Provides persistent storage of trades with:
    - Full ACID compliance
    - Advanced querying capabilities
    - Indexing for performance
    - Support for complex product details via JSONB

    Attributes
    ----------
    config : DatabaseConfig
        Database connection configuration
    schema : str
        PostgreSQL schema name (default: "trading")
    """

    def __init__(
        self,
        config: DatabaseConfig,
        schema: str = "trading",
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        """Initialize PostgreSQL trade repository.

        Parameters
        ----------
        config : DatabaseConfig
            Database connection configuration
        schema : str
            PostgreSQL schema name
        loop : asyncio.AbstractEventLoop, optional
            Event loop for async operations
        """
        self.config = config
        self.schema = schema
        self._pool = None
        self._loop = loop
        self._connected = False

    async def connect(self) -> bool:
        """Establish database connection and ensure schema exists."""
        if self._connected:
            return True

        try:
            import asyncpg
        except ImportError as exc:
            raise RuntimeError(
                "asyncpg is required for PostgreSQL repository. "
                "Install with `pip install asyncpg`."
            ) from exc

        dsn = self._build_dsn()
        self._pool = await asyncpg.create_pool(
            dsn=dsn,
            timeout=self.config.connect_timeout,
            **self.config.options,
        )

        await self._ensure_schema()
        self._connected = True
        return True

    async def disconnect(self) -> None:
        """Close database connection."""
        if self._pool:
            await self._pool.close()
            self._pool = None
        self._connected = False

    async def _ensure_schema(self) -> None:
        """Create schema and tables if they don't exist."""
        async with self._pool.acquire() as conn:
            # Create schema
            await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")

            # Create trades table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.trades (
                    id TEXT PRIMARY KEY,
                    trade_number TEXT,
                    external_id TEXT,
                    usi TEXT,
                    counterparty_id TEXT NOT NULL,
                    netting_set_id TEXT,
                    book_id TEXT,
                    desk_id TEXT,
                    trader_id TEXT,
                    product_type TEXT NOT NULL,
                    trade_date DATE NOT NULL,
                    effective_date DATE,
                    maturity_date DATE,
                    status TEXT NOT NULL,
                    notional NUMERIC,
                    currency TEXT,
                    settlement_type TEXT,
                    product_details JSONB,
                    convention_profile_id TEXT,
                    generated_from_convention BOOLEAN DEFAULT FALSE,
                    mtm NUMERIC,
                    last_valuation_date DATE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)

            # Create indexes
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_trades_counterparty
                ON {self.schema}.trades (counterparty_id)
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_trades_book
                ON {self.schema}.trades (book_id)
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_trades_status
                ON {self.schema}.trades (status)
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_trades_product_type
                ON {self.schema}.trades (product_type)
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_trades_trade_date
                ON {self.schema}.trades (trade_date DESC)
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_trades_maturity_date
                ON {self.schema}.trades (maturity_date)
            """)

    def save(self, trade: Trade) -> None:
        """Save a trade (synchronous wrapper)."""
        loop = self._loop or asyncio.get_event_loop()
        loop.run_until_complete(self.save_async(trade))

    async def save_async(self, trade: Trade) -> None:
        """Save a trade asynchronously."""
        await self.connect()

        async with self._pool.acquire() as conn:
            import asyncpg

            await conn.execute(
                f"""
                INSERT INTO {self.schema}.trades (
                    id, trade_number, external_id, usi, counterparty_id,
                    netting_set_id, book_id, desk_id, trader_id, product_type,
                    trade_date, effective_date, maturity_date, status, notional,
                    currency, settlement_type, product_details, convention_profile_id,
                    generated_from_convention, mtm, last_valuation_date
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                    $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22
                )
                ON CONFLICT (id) DO UPDATE SET
                    trade_number = EXCLUDED.trade_number,
                    external_id = EXCLUDED.external_id,
                    usi = EXCLUDED.usi,
                    counterparty_id = EXCLUDED.counterparty_id,
                    netting_set_id = EXCLUDED.netting_set_id,
                    book_id = EXCLUDED.book_id,
                    desk_id = EXCLUDED.desk_id,
                    trader_id = EXCLUDED.trader_id,
                    product_type = EXCLUDED.product_type,
                    trade_date = EXCLUDED.trade_date,
                    effective_date = EXCLUDED.effective_date,
                    maturity_date = EXCLUDED.maturity_date,
                    status = EXCLUDED.status,
                    notional = EXCLUDED.notional,
                    currency = EXCLUDED.currency,
                    settlement_type = EXCLUDED.settlement_type,
                    product_details = EXCLUDED.product_details,
                    convention_profile_id = EXCLUDED.convention_profile_id,
                    generated_from_convention = EXCLUDED.generated_from_convention,
                    mtm = EXCLUDED.mtm,
                    last_valuation_date = EXCLUDED.last_valuation_date,
                    updated_at = NOW()
                """,
                trade.id,
                trade.trade_number,
                trade.external_id,
                trade.usi,
                trade.counterparty_id,
                trade.netting_set_id,
                trade.book_id,
                trade.desk_id,
                trade.trader_id,
                trade.product_type.value,
                trade.trade_date,
                trade.effective_date,
                trade.maturity_date,
                trade.status.value,
                trade.notional,
                trade.currency,
                trade.settlement_type.value if trade.settlement_type else None,
                asyncpg.Json(trade.product_details) if trade.product_details else None,
                trade.convention_profile_id,
                trade.generated_from_convention,
                trade.mtm,
                trade.last_valuation_date,
            )

    def find_by_id(self, trade_id: str) -> Optional[Trade]:
        """Find a trade by ID (synchronous wrapper)."""
        loop = self._loop or asyncio.get_event_loop()
        return loop.run_until_complete(self.find_by_id_async(trade_id))

    async def find_by_id_async(self, trade_id: str) -> Optional[Trade]:
        """Find a trade by ID asynchronously."""
        await self.connect()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM {self.schema}.trades WHERE id = $1",
                trade_id,
            )

            if row:
                return self._row_to_trade(row)
            return None

    def find_all(self) -> List[Trade]:
        """Find all trades (synchronous wrapper)."""
        loop = self._loop or asyncio.get_event_loop()
        return loop.run_until_complete(self.find_all_async())

    async def find_all_async(self) -> List[Trade]:
        """Find all trades asynchronously."""
        await self.connect()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM {self.schema}.trades ORDER BY trade_date DESC"
            )

            return [self._row_to_trade(row) for row in rows]

    def find_by_counterparty(self, counterparty_id: str) -> List[Trade]:
        """Find trades by counterparty (synchronous wrapper)."""
        loop = self._loop or asyncio.get_event_loop()
        return loop.run_until_complete(self.find_by_counterparty_async(counterparty_id))

    async def find_by_counterparty_async(self, counterparty_id: str) -> List[Trade]:
        """Find trades by counterparty asynchronously."""
        await self.connect()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM {self.schema}.trades WHERE counterparty_id = $1",
                counterparty_id,
            )

            return [self._row_to_trade(row) for row in rows]

    def find_by_book(self, book_id: str) -> List[Trade]:
        """Find trades by book (synchronous wrapper)."""
        loop = self._loop or asyncio.get_event_loop()
        return loop.run_until_complete(self.find_by_book_async(book_id))

    async def find_by_book_async(self, book_id: str) -> List[Trade]:
        """Find trades by book asynchronously."""
        await self.connect()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM {self.schema}.trades WHERE book_id = $1",
                book_id,
            )

            return [self._row_to_trade(row) for row in rows]

    def find_by_status(self, status: TradeStatus) -> List[Trade]:
        """Find trades by status (synchronous wrapper)."""
        loop = self._loop or asyncio.get_event_loop()
        return loop.run_until_complete(self.find_by_status_async(status))

    async def find_by_status_async(self, status: TradeStatus) -> List[Trade]:
        """Find trades by status asynchronously."""
        await self.connect()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM {self.schema}.trades WHERE status = $1",
                status.value,
            )

            return [self._row_to_trade(row) for row in rows]

    def delete(self, trade_id: str) -> bool:
        """Delete a trade (synchronous wrapper)."""
        loop = self._loop or asyncio.get_event_loop()
        return loop.run_until_complete(self.delete_async(trade_id))

    async def delete_async(self, trade_id: str) -> bool:
        """Delete a trade asynchronously."""
        await self.connect()

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM {self.schema}.trades WHERE id = $1",
                trade_id,
            )

            return result != "DELETE 0"

    def count(self) -> int:
        """Count total trades (synchronous wrapper)."""
        loop = self._loop or asyncio.get_event_loop()
        return loop.run_until_complete(self.count_async())

    async def count_async(self) -> int:
        """Count total trades asynchronously."""
        await self.connect()

        async with self._pool.acquire() as conn:
            count = await conn.fetchval(f"SELECT COUNT(*) FROM {self.schema}.trades")
            return count

    def _build_dsn(self) -> str:
        """Build PostgreSQL DSN connection string."""
        if self.config.dsn:
            return self.config.dsn

        if not self.config.database:
            raise ValueError("Database name is required for PostgreSQL repository")

        host = self.config.host or "localhost"
        port = self.config.port or 5432
        user = self.config.user or ""
        password = self.config.password or ""

        auth = ""
        if user:
            from urllib.parse import quote_plus

            auth = quote_plus(user)
            if password:
                auth += f":{quote_plus(password)}"
            auth += "@"

        return f"postgresql://{auth}{host}:{port}/{self.config.database}"

    def _row_to_trade(self, row: Any) -> Trade:
        """Convert database row to Trade object."""
        return Trade(
            id=row["id"],
            trade_number=row["trade_number"],
            external_id=row["external_id"],
            usi=row["usi"],
            counterparty_id=row["counterparty_id"],
            netting_set_id=row["netting_set_id"],
            book_id=row["book_id"],
            desk_id=row["desk_id"],
            trader_id=row["trader_id"],
            product_type=ProductType(row["product_type"]),
            trade_date=row["trade_date"],
            effective_date=row["effective_date"],
            maturity_date=row["maturity_date"],
            status=TradeStatus(row["status"]),
            notional=float(row["notional"]) if row["notional"] else None,
            currency=row["currency"],
            settlement_type=row["settlement_type"],
            product_details=row["product_details"],
            convention_profile_id=row["convention_profile_id"],
            generated_from_convention=row["generated_from_convention"],
            mtm=float(row["mtm"]) if row["mtm"] else None,
            last_valuation_date=row["last_valuation_date"],
        )


__all__ = ["PostgresTradeRepository"]
