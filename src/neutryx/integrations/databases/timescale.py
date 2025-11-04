"""TimescaleDB connector built on top of the PostgreSQL connector."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from .base import DatabaseConfig
from .postgres import PostgresConnector
from neutryx.market.data_models import MarketDataPoint


class TimescaleConnector(PostgresConnector):
    """Connector for TimescaleDB hypertables."""

    def __init__(
        self,
        config: DatabaseConfig,
        *,
        schema: str = "public",
        table: str = "market_data",
        chunk_interval: Optional[str] = None,
    ):
        super().__init__(config, schema=schema, table=table)
        self._chunk_interval = chunk_interval

    async def _ensure_table(self) -> None:
        await super()._ensure_table()

        async with self._pool.acquire() as connection:
            await connection.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
            await connection.execute(
                "SELECT create_hypertable($1, 'ts', if_not_exists => TRUE)",
                self._qualified_table(),
            )
            if self._chunk_interval:
                await connection.execute(
                    "SELECT set_chunk_time_interval($1, $2::interval)",
                    self._qualified_table(),
                    self._chunk_interval,
                )

    async def write_market_data(
        self,
        points: Sequence[MarketDataPoint],
        *,
        asset_class: Optional[Any] = None,
        instrument_id: Optional[str] = None,
    ) -> None:
        await super().write_market_data(
            points,
            asset_class=asset_class,
            instrument_id=instrument_id,
        )


__all__ = ["TimescaleConnector"]
