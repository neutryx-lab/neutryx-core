"""Base storage interface for market data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum


class StorageType(Enum):
    """Storage backend type."""
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"
    TIMESCALEDB = "timescaledb"
    REDIS = "redis"


@dataclass
class StorageConfig:
    """
    Base configuration for storage backends.

    Attributes:
        storage_type: Type of storage backend
        host: Database host
        port: Database port
        database: Database name
        username: Database username
        password: Database password
        ssl_enabled: Enable SSL/TLS
        connection_pool_size: Maximum connection pool size
        connection_timeout: Connection timeout in seconds
        extra_params: Additional vendor-specific parameters
    """
    storage_type: StorageType
    host: str = "localhost"
    port: int = 5432
    database: str = "market_data"
    username: str = ""
    password: str = ""
    ssl_enabled: bool = False
    connection_pool_size: int = 10
    connection_timeout: int = 30
    extra_params: Dict[str, Any] = field(default_factory=dict)


class BaseStorage(ABC):
    """
    Abstract base class for market data storage.

    Provides common interface for storing and retrieving market data
    from various database backends optimized for time-series data.
    """

    def __init__(self, config: StorageConfig):
        """
        Initialize storage backend.

        Args:
            config: Storage configuration
        """
        self.config = config
        self._connection = None
        self._connected = False

    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to storage backend.

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from storage backend.

        Returns:
            True if disconnection successful, False otherwise
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if connected to storage backend.

        Returns:
            True if connected, False otherwise
        """
        pass

    @abstractmethod
    async def store_quote(
        self,
        data_type: str,
        symbol: str,
        timestamp: datetime,
        data: Dict[str, Any],
    ) -> bool:
        """
        Store market data quote.

        Args:
            data_type: Type of data (equity, fx, bond, etc.)
            symbol: Instrument symbol
            timestamp: Quote timestamp
            data: Quote data dictionary

        Returns:
            True if stored successfully, False otherwise
        """
        pass

    @abstractmethod
    async def store_batch(
        self,
        data_type: str,
        records: List[Dict[str, Any]],
    ) -> int:
        """
        Store batch of market data records.

        Args:
            data_type: Type of data
            records: List of records to store

        Returns:
            Number of records stored successfully
        """
        pass

    @abstractmethod
    async def query_latest(
        self,
        data_type: str,
        symbol: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Query latest market data for symbol.

        Args:
            data_type: Type of data
            symbol: Instrument symbol

        Returns:
            Latest data record or None if not found
        """
        pass

    @abstractmethod
    async def query_time_range(
        self,
        data_type: str,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query market data for time range.

        Args:
            data_type: Type of data
            symbol: Instrument symbol
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum number of records to return

        Returns:
            List of data records
        """
        pass

    @abstractmethod
    async def query_aggregated(
        self,
        data_type: str,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1min",
    ) -> List[Dict[str, Any]]:
        """
        Query aggregated market data (OHLCV).

        Args:
            data_type: Type of data
            symbol: Instrument symbol
            start_time: Start of time range
            end_time: End of time range
            interval: Aggregation interval (1min, 5min, 1hour, etc.)

        Returns:
            List of aggregated records
        """
        pass

    @abstractmethod
    async def delete_old_data(
        self,
        data_type: str,
        before_timestamp: datetime,
    ) -> int:
        """
        Delete old data before specified timestamp.

        Args:
            data_type: Type of data
            before_timestamp: Delete data before this timestamp

        Returns:
            Number of records deleted
        """
        pass

    @abstractmethod
    async def create_indexes(self) -> bool:
        """
        Create database indexes for optimized queries.

        Returns:
            True if indexes created successfully, False otherwise
        """
        pass

    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary with storage statistics
        """
        pass

    def get_connection_string(self) -> str:
        """
        Get database connection string.

        Returns:
            Connection string (with password masked)
        """
        password = "***" if self.config.password else ""
        return (
            f"{self.config.storage_type.value}://"
            f"{self.config.username}:{password}@"
            f"{self.config.host}:{self.config.port}/{self.config.database}"
        )
