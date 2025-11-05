"""PostgreSQL storage backend for market data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import BaseStorage, StorageConfig, StorageType

logger = logging.getLogger(__name__)


@dataclass
class PostgreSQLConfig(StorageConfig):
    """
    PostgreSQL-specific configuration.

    Attributes:
        schema: Database schema name
        table_prefix: Prefix for table names
        use_connection_pooling: Enable connection pooling
        statement_timeout: Query timeout in milliseconds
    """
    schema: str = "market_data"
    table_prefix: str = "md_"
    use_connection_pooling: bool = True
    statement_timeout: int = 30000

    def __post_init__(self):
        self.storage_type = StorageType.POSTGRESQL
        if self.port == 5432:  # Default already set
            pass


class PostgreSQLStorage(BaseStorage):
    """
    PostgreSQL storage backend.

    Provides high-performance storage for market data using PostgreSQL
    with optimized indexes and partitioning support.

    Features:
    - Time-series partitioning for efficient queries
    - Composite indexes on symbol + timestamp
    - Bulk insert optimization
    - Automatic data retention policies

    Example:
        >>> config = PostgreSQLConfig(
        ...     host="localhost",
        ...     port=5432,
        ...     database="market_data",
        ...     username="trader",
        ...     password="secret"
        ... )
        >>> storage = PostgreSQLStorage(config)
        >>> await storage.connect()
        >>> await storage.store_quote(
        ...     "equity",
        ...     "AAPL",
        ...     datetime.utcnow(),
        ...     {"price": 150.25, "volume": 1000000}
        ... )
    """

    def __init__(self, config: PostgreSQLConfig):
        """Initialize PostgreSQL storage."""
        super().__init__(config)
        self.config: PostgreSQLConfig = config
        self._pool = None

    async def connect(self) -> bool:
        """Connect to PostgreSQL database."""
        try:
            # Import asyncpg for async PostgreSQL
            import asyncpg
        except ImportError:
            logger.error("asyncpg not installed. Install with: pip install asyncpg")
            return False

        try:
            # Create connection pool
            self._pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                min_size=1,
                max_size=self.config.connection_pool_size,
                timeout=self.config.connection_timeout,
                command_timeout=self.config.statement_timeout / 1000,
                ssl=self.config.ssl_enabled,
            )

            # Create schema if not exists
            async with self._pool.acquire() as conn:
                await conn.execute(
                    f"CREATE SCHEMA IF NOT EXISTS {self.config.schema}"
                )

            self._connected = True
            logger.info(f"Connected to PostgreSQL: {self.get_connection_string()}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            return False

    async def disconnect(self) -> bool:
        """Disconnect from PostgreSQL database."""
        try:
            if self._pool:
                await self._pool.close()
                self._pool = None
            self._connected = False
            logger.info("Disconnected from PostgreSQL")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from PostgreSQL: {e}")
            return False

    def is_connected(self) -> bool:
        """Check if connected to PostgreSQL."""
        return self._connected and self._pool is not None

    async def store_quote(
        self,
        data_type: str,
        symbol: str,
        timestamp: datetime,
        data: Dict[str, Any],
    ) -> bool:
        """Store market data quote."""
        if not self.is_connected():
            logger.error("Not connected to PostgreSQL")
            return False

        try:
            table_name = f"{self.config.schema}.{self.config.table_prefix}{data_type}"

            # Build insert query dynamically
            columns = ["symbol", "timestamp"] + list(data.keys())
            values_placeholders = [f"${i+1}" for i in range(len(columns))]
            values = [symbol, timestamp] + list(data.values())

            query = f"""
                INSERT INTO {table_name} ({', '.join(columns)})
                VALUES ({', '.join(values_placeholders)})
                ON CONFLICT (symbol, timestamp) DO UPDATE
                SET {', '.join([f"{col} = EXCLUDED.{col}" for col in data.keys()])}
            """

            async with self._pool.acquire() as conn:
                await conn.execute(query, *values)

            return True

        except Exception as e:
            logger.error(f"Error storing quote: {e}")
            return False

    async def store_batch(
        self,
        data_type: str,
        records: List[Dict[str, Any]],
    ) -> int:
        """Store batch of market data records."""
        if not self.is_connected() or not records:
            return 0

        try:
            table_name = f"{self.config.schema}.{self.config.table_prefix}{data_type}"

            # Extract columns from first record
            first_record = records[0]
            columns = list(first_record.keys())

            # Prepare batch insert
            values = []
            for record in records:
                values.append([record.get(col) for col in columns])

            query = f"""
                INSERT INTO {table_name} ({', '.join(columns)})
                VALUES ({', '.join([f'${i+1}' for i in range(len(columns))])})
                ON CONFLICT (symbol, timestamp) DO NOTHING
            """

            async with self._pool.acquire() as conn:
                await conn.executemany(query, values)

            return len(records)

        except Exception as e:
            logger.error(f"Error storing batch: {e}")
            return 0

    async def query_latest(
        self,
        data_type: str,
        symbol: str,
    ) -> Optional[Dict[str, Any]]:
        """Query latest market data for symbol."""
        if not self.is_connected():
            return None

        try:
            table_name = f"{self.config.schema}.{self.config.table_prefix}{data_type}"

            query = f"""
                SELECT *
                FROM {table_name}
                WHERE symbol = $1
                ORDER BY timestamp DESC
                LIMIT 1
            """

            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(query, symbol)
                if row:
                    return dict(row)
                return None

        except Exception as e:
            logger.error(f"Error querying latest data: {e}")
            return None

    async def query_time_range(
        self,
        data_type: str,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Query market data for time range."""
        if not self.is_connected():
            return []

        try:
            table_name = f"{self.config.schema}.{self.config.table_prefix}{data_type}"

            query = f"""
                SELECT *
                FROM {table_name}
                WHERE symbol = $1
                  AND timestamp >= $2
                  AND timestamp <= $3
                ORDER BY timestamp ASC
            """

            if limit:
                query += f" LIMIT {limit}"

            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query, symbol, start_time, end_time)
                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Error querying time range: {e}")
            return []

    async def query_aggregated(
        self,
        data_type: str,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1min",
    ) -> List[Dict[str, Any]]:
        """Query aggregated market data (OHLCV)."""
        if not self.is_connected():
            return []

        try:
            table_name = f"{self.config.schema}.{self.config.table_prefix}{data_type}"

            # Convert interval string to PostgreSQL interval
            pg_interval = self._convert_interval(interval)

            query = f"""
                SELECT
                    time_bucket($1::interval, timestamp) AS bucket,
                    symbol,
                    FIRST(price, timestamp) AS open,
                    MAX(price) AS high,
                    MIN(price) AS low,
                    LAST(price, timestamp) AS close,
                    SUM(volume) AS volume
                FROM {table_name}
                WHERE symbol = $2
                  AND timestamp >= $3
                  AND timestamp <= $4
                GROUP BY bucket, symbol
                ORDER BY bucket ASC
            """

            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query, pg_interval, symbol, start_time, end_time)
                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Error querying aggregated data: {e}")
            return []

    async def delete_old_data(
        self,
        data_type: str,
        before_timestamp: datetime,
    ) -> int:
        """Delete old data before specified timestamp."""
        if not self.is_connected():
            return 0

        try:
            table_name = f"{self.config.schema}.{self.config.table_prefix}{data_type}"

            query = f"""
                DELETE FROM {table_name}
                WHERE timestamp < $1
            """

            async with self._pool.acquire() as conn:
                result = await conn.execute(query, before_timestamp)
                # Extract count from result string like "DELETE 123"
                count = int(result.split()[-1]) if result else 0
                return count

        except Exception as e:
            logger.error(f"Error deleting old data: {e}")
            return 0

    async def create_indexes(self) -> bool:
        """Create database indexes for optimized queries."""
        if not self.is_connected():
            return False

        try:
            # Common table types
            table_types = ["equity", "fx", "bond", "commodity", "rates"]

            async with self._pool.acquire() as conn:
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
                            data JSONB,
                            PRIMARY KEY (symbol, timestamp)
                        )
                    """
                    )

                    # Create indexes
                    await conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_{data_type}_timestamp
                        ON {table_name} (timestamp DESC)
                    """)

                    await conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_{data_type}_symbol_timestamp
                        ON {table_name} (symbol, timestamp DESC)
                    """)

            logger.info("Created PostgreSQL indexes")
            return True

        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
            return False

    async def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        if not self.is_connected():
            return {}

        try:
            async with self._pool.acquire() as conn:
                # Get table sizes
                tables_query = f"""
                    SELECT
                        schemaname,
                        tablename,
                        pg_total_relation_size(schemaname||'.'||tablename) AS size_bytes,
                        n_live_tup AS row_count
                    FROM pg_stat_user_tables
                    WHERE schemaname = $1
                """
                tables = await conn.fetch(tables_query, self.config.schema)

                stats = {
                    "database": self.config.database,
                    "schema": self.config.schema,
                    "tables": [],
                    "total_size_bytes": 0,
                    "total_rows": 0,
                }

                for table in tables:
                    table_info = {
                        "name": table["tablename"],
                        "size_bytes": table["size_bytes"],
                        "row_count": table["row_count"],
                    }
                    stats["tables"].append(table_info)
                    stats["total_size_bytes"] += table["size_bytes"]
                    stats["total_rows"] += table["row_count"]

                return stats

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}

    @staticmethod
    def _convert_interval(interval: str) -> str:
        """Convert interval string to PostgreSQL interval."""
        # Map common intervals
        interval_map = {
            "1s": "1 second",
            "1min": "1 minute",
            "5min": "5 minutes",
            "15min": "15 minutes",
            "1hour": "1 hour",
            "1day": "1 day",
        }
        return interval_map.get(interval, interval)
