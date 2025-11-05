"""PostgreSQL database connector."""

from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, Optional, Sequence

from .base import DatabaseConfig, DatabaseConnector, DatabaseConnectorError
from neutryx.market.data_models import MarketDataPoint

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class PostgresConnector(DatabaseConnector):
    """Connector that persists market data into PostgreSQL."""

    def __init__(
        self,
        config: DatabaseConfig,
        *,
        schema: str = "public",
        table: str = "market_data",
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        super().__init__(config)
        self.schema = schema
        self.table = table
        self._pool = None
        self._loop = loop

    async def connect(self) -> bool:
        if self._connected:
            return True

        try:
            import asyncpg
        except ImportError as exc:  # pragma: no cover - depends on runtime env
            raise DatabaseConnectorError(
                "asyncpg is required for PostgreSQL connector. "
                "Install with `pip install asyncpg`."
            ) from exc

        dsn = self._build_dsn()

        self._pool = await asyncpg.create_pool(
            dsn=dsn,
            timeout=self.config.connect_timeout,
            **self.config.options,
        )

        await self._ensure_table()
        self._connected = True
        return True

    async def disconnect(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None
        self._connected = False

    async def write_market_data(
        self,
        points: Sequence[MarketDataPoint],
        *,
        asset_class: Optional[Any] = None,
        instrument_id: Optional[str] = None,
    ) -> None:
        if not points:
            return

        await self.connect()
        records = self._prepare_records(
            points,
            asset_class=asset_class,
            instrument_id=instrument_id,
        )

        import asyncpg  # Local import to keep optional dependency isolated

        data = [
            (
                record["timestamp"],
                record["source"],
                record["quality"],
                record["asset_class"],
                record["instrument_id"],
                asyncpg.Json(record["payload"]),
                asyncpg.Json(record["metadata"]),
            )
            for record in records
        ]

        async with self._pool.acquire() as connection:
            # Table name is validated in _ensure_table() via _validate_identifier()
            await connection.executemany(
                f"""
                INSERT INTO {self._qualified_table()} (
                    ts, source, quality, asset_class, instrument_id, payload, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,  # nosec B608
                data,
            )

    async def _ensure_table(self) -> None:
        self._validate_identifier(self.schema)
        self._validate_identifier(self.table)

        async with self._pool.acquire() as connection:
            await connection.execute(
                f"CREATE SCHEMA IF NOT EXISTS {self.schema}"
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._qualified_table()} (
                    id BIGSERIAL PRIMARY KEY,
                    ts TIMESTAMPTZ NOT NULL,
                    source TEXT NOT NULL,
                    quality TEXT NOT NULL,
                    asset_class TEXT,
                    instrument_id TEXT,
                    payload JSONB NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await connection.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table}_instrument ON {self._qualified_table()} (instrument_id)"
            )
            await connection.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table}_ts ON {self._qualified_table()} (ts DESC)"
            )

    def _build_dsn(self) -> str:
        if self.config.dsn:
            return self.config.dsn

        if not self.config.database:
            raise DatabaseConnectorError("Database name is required for PostgreSQL connector")

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

    def _qualified_table(self) -> str:
        return f"{self.schema}.{self.table}"

    @staticmethod
    def _validate_identifier(name: str) -> None:
        if not _NAME_RE.match(name):
            raise DatabaseConnectorError(f"Invalid PostgreSQL identifier: {name}")


__all__ = ["PostgresConnector"]
