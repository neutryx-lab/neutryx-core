"""TimescaleDB storage backend for market data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .postgresql import PostgreSQLStorage, PostgreSQLConfig
from .base import StorageType

logger = logging.getLogger(__name__)


@dataclass
class TimescaleDBConfig(PostgreSQLConfig):
    """
    TimescaleDB-specific configuration.

    Attributes:
        chunk_time_interval: Time interval for chunk partitioning
        compression_enabled: Enable automatic compression
        compression_after: Compress chunks older than this duration
        retention_policy_days: Auto-delete data older than this (0 = no retention)
    """
    chunk_time_interval: str = "1 day"
    compression_enabled: bool = True
    compression_after: str = "7 days"
    retention_policy_days: int = 90

    def __post_init__(self):
        self.storage_type = StorageType.TIMESCALEDB
        super().__post_init__()


class TimescaleDBStorage(PostgreSQLStorage):
    """
    TimescaleDB storage backend.

    Extends PostgreSQL with TimescaleDB-specific features optimized for
    time-series data including automatic partitioning, compression, and
    continuous aggregates.

    Features:
    - Automatic hypertable partitioning by time
    - Native compression for old data
    - Continuous aggregates for OHLCV
    - Automatic retention policies
    - Optimized time-series queries

    Example:
        >>> config = TimescaleDBConfig(
        ...     host="localhost",
        ...     port=5432,
        ...     database="market_data",
        ...     username="trader",
        ...     password="secret",
        ...     chunk_time_interval="1 day",
        ...     compression_enabled=True
        ... )
        >>> storage = TimescaleDBStorage(config)
        >>> await storage.connect()
    """

    def __init__(self, config: TimescaleDBConfig):
        """Initialize TimescaleDB storage."""
        super().__init__(config)
        self.config: TimescaleDBConfig = config

    async def create_indexes(self) -> bool:
        """Create TimescaleDB hypertables and indexes."""
        if not self.is_connected():
            return False

        try:
            async with self._pool.acquire() as conn:
                # Enable TimescaleDB extension
                await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")

                # Common table types
                table_types = ["equity", "fx", "bond", "commodity", "rates"]

                for data_type in table_types:
                    table_name = f"{self.config.schema}.{self.config.table_prefix}{data_type}"

                    # Create table if not exists
                    await conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS {table_name} (
                            symbol VARCHAR(50) NOT NULL,
                            timestamp TIMESTAMPTZ NOT NULL,
                            price DOUBLE PRECISION,
                            bid DOUBLE PRECISION,
                            ask DOUBLE PRECISION,
                            volume DOUBLE PRECISION,
                            open DOUBLE PRECISION,
                            high DOUBLE PRECISION,
                            low DOUBLE PRECISION,
                            close DOUBLE PRECISION,
                            vwap DOUBLE PRECISION,
                            data JSONB
                        )
                    """)

                    # Convert to hypertable (TimescaleDB-specific)
                    try:
                        await conn.execute(f"""
                            SELECT create_hypertable(
                                '{table_name}',
                                'timestamp',
                                chunk_time_interval => INTERVAL '{self.config.chunk_time_interval}',
                                if_not_exists => TRUE
                            )
                        """)
                    except Exception as e:
                        # Table might already be a hypertable
                        logger.debug(f"Hypertable creation info: {e}")

                    # Create indexes
                    await conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_{data_type}_symbol_timestamp
                        ON {table_name} (symbol, timestamp DESC)
                    """)

                    # Enable compression if configured
                    if self.config.compression_enabled:
                        await conn.execute(f"""
                            ALTER TABLE {table_name} SET (
                                timescaledb.compress,
                                timescaledb.compress_segmentby = 'symbol'
                            )
                        """)

                        # Add compression policy
                        await conn.execute(f"""
                            SELECT add_compression_policy(
                                '{table_name}',
                                INTERVAL '{self.config.compression_after}',
                                if_not_exists => TRUE
                            )
                        """)

                    # Add retention policy if configured
                    if self.config.retention_policy_days > 0:
                        await conn.execute(f"""
                            SELECT add_retention_policy(
                                '{table_name}',
                                INTERVAL '{self.config.retention_policy_days} days',
                                if_not_exists => TRUE
                            )
                        """)

                    # Create continuous aggregate for OHLCV
                    cagg_name = f"{self.config.table_prefix}{data_type}_1min"
                    try:
                        await conn.execute(f"""
                            CREATE MATERIALIZED VIEW IF NOT EXISTS {self.config.schema}.{cagg_name}
                            WITH (timescaledb.continuous) AS
                            SELECT
                                time_bucket('1 minute', timestamp) AS bucket,
                                symbol,
                                FIRST(price, timestamp) AS open,
                                MAX(price) AS high,
                                MIN(price) AS low,
                                LAST(price, timestamp) AS close,
                                SUM(volume) AS volume,
                                COUNT(*) AS tick_count
                            FROM {table_name}
                            GROUP BY bucket, symbol
                            WITH NO DATA
                        """)

                        # Add refresh policy
                        await conn.execute(f"""
                            SELECT add_continuous_aggregate_policy(
                                '{self.config.schema}.{cagg_name}',
                                start_offset => INTERVAL '1 hour',
                                end_offset => INTERVAL '1 minute',
                                schedule_interval => INTERVAL '1 minute',
                                if_not_exists => TRUE
                            )
                        """)
                    except Exception as e:
                        logger.debug(f"Continuous aggregate info: {e}")

            logger.info("Created TimescaleDB hypertables and policies")
            return True

        except Exception as e:
            logger.error(f"Error creating TimescaleDB structures: {e}")
            return False

    async def query_aggregated(
        self,
        data_type: str,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1min",
    ) -> List[Dict[str, Any]]:
        """Query aggregated market data using continuous aggregates."""
        if not self.is_connected():
            return []

        # Use continuous aggregate for 1min interval
        if interval == "1min":
            try:
                cagg_name = f"{self.config.schema}.{self.config.table_prefix}{data_type}_1min"

                query = f"""
                    SELECT
                        bucket AS timestamp,
                        symbol,
                        open,
                        high,
                        low,
                        close,
                        volume,
                        tick_count
                    FROM {cagg_name}
                    WHERE symbol = $1
                      AND bucket >= $2
                      AND bucket <= $3
                    ORDER BY bucket ASC
                """

                async with self._pool.acquire() as conn:
                    rows = await conn.fetch(query, symbol, start_time, end_time)
                    return [dict(row) for row in rows]

            except Exception as e:
                logger.error(f"Error querying continuous aggregate: {e}")
                return []
        else:
            # Fall back to parent implementation for other intervals
            return await super().query_aggregated(
                data_type, symbol, start_time, end_time, interval
            )

    async def get_statistics(self) -> Dict[str, Any]:
        """Get TimescaleDB-specific statistics."""
        stats = await super().get_statistics()

        if not self.is_connected():
            return stats

        try:
            async with self._pool.acquire() as conn:
                # Get hypertable info
                hypertables_query = """
                    SELECT
                        hypertable_schema,
                        hypertable_name,
                        num_chunks,
                        compression_enabled,
                        compressed_total_bytes,
                        uncompressed_total_bytes
                    FROM timescaledb_information.hypertables
                    WHERE hypertable_schema = $1
                """
                hypertables = await conn.fetch(hypertables_query, self.config.schema)

                stats["timescaledb"] = {
                    "hypertables": [],
                    "total_chunks": 0,
                    "compression_ratio": 0.0,
                }

                total_compressed = 0
                total_uncompressed = 0

                for ht in hypertables:
                    ht_info = {
                        "name": ht["hypertable_name"],
                        "num_chunks": ht["num_chunks"],
                        "compression_enabled": ht["compression_enabled"],
                        "compressed_bytes": ht["compressed_total_bytes"] or 0,
                        "uncompressed_bytes": ht["uncompressed_total_bytes"] or 0,
                    }
                    stats["timescaledb"]["hypertables"].append(ht_info)
                    stats["timescaledb"]["total_chunks"] += ht["num_chunks"]
                    total_compressed += ht_info["compressed_bytes"]
                    total_uncompressed += ht_info["uncompressed_bytes"]

                if total_uncompressed > 0:
                    stats["timescaledb"]["compression_ratio"] = (
                        1 - (total_compressed / total_uncompressed)
                    )

        except Exception as e:
            logger.error(f"Error getting TimescaleDB statistics: {e}")

        return stats

    async def vacuum_and_analyze(self) -> bool:
        """Run VACUUM and ANALYZE on all tables."""
        if not self.is_connected():
            return False

        try:
            table_types = ["equity", "fx", "bond", "commodity", "rates"]

            async with self._pool.acquire() as conn:
                for data_type in table_types:
                    table_name = f"{self.config.schema}.{self.config.table_prefix}{data_type}"
                    await conn.execute(f"VACUUM ANALYZE {table_name}")

            logger.info("Completed VACUUM ANALYZE")
            return True

        except Exception as e:
            logger.error(f"Error running VACUUM ANALYZE: {e}")
            return False
