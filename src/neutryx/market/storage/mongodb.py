"""MongoDB storage backend for market data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import BaseStorage, StorageConfig, StorageType

logger = logging.getLogger(__name__)


@dataclass
class MongoDBConfig(StorageConfig):
    """
    MongoDB-specific configuration.

    Attributes:
        auth_database: Authentication database
        replica_set: Replica set name
        collection_prefix: Prefix for collection names
        use_time_series: Use MongoDB time-series collections (MongoDB 5.0+)
    """
    auth_database: str = "admin"
    replica_set: Optional[str] = None
    collection_prefix: str = "md_"
    use_time_series: bool = True

    def __post_init__(self):
        self.storage_type = StorageType.MONGODB
        if self.port == 5432:
            self.port = 27017  # Default MongoDB port


class MongoDBStorage(BaseStorage):
    """
    MongoDB storage backend.

    Provides flexible document storage for market data with MongoDB's
    powerful querying and aggregation capabilities.

    Features:
    - Time-series collections for efficient storage
    - Flexible schema for heterogeneous data
    - Compound indexes for fast queries
    - Aggregation pipeline support

    Example:
        >>> config = MongoDBConfig(
        ...     host="localhost",
        ...     port=27017,
        ...     database="market_data",
        ...     username="trader",
        ...     password="secret"
        ... )
        >>> storage = MongoDBStorage(config)
        >>> await storage.connect()
    """

    def __init__(self, config: MongoDBConfig):
        """Initialize MongoDB storage."""
        super().__init__(config)
        self.config: MongoDBConfig = config
        self._client = None
        self._db = None

    async def connect(self) -> bool:
        """Connect to MongoDB."""
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
        except ImportError:
            logger.error("motor not installed. Install with: pip install motor")
            return False

        try:
            # Build connection string
            auth = f"{self.config.username}:{self.config.password}@" if self.config.username else ""
            replica_set_param = f"?replicaSet={self.config.replica_set}" if self.config.replica_set else ""

            connection_string = (
                f"mongodb://{auth}{self.config.host}:{self.config.port}/"
                f"{self.config.auth_database}{replica_set_param}"
            )

            self._client = AsyncIOMotorClient(
                connection_string,
                maxPoolSize=self.config.connection_pool_size,
                serverSelectionTimeoutMS=self.config.connection_timeout * 1000,
                tls=self.config.ssl_enabled,
            )

            # Test connection
            await self._client.admin.command("ping")

            self._db = self._client[self.config.database]
            self._connected = True
            logger.info(f"Connected to MongoDB: {self.get_connection_string()}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False

    async def disconnect(self) -> bool:
        """Disconnect from MongoDB."""
        try:
            if self._client:
                self._client.close()
                self._client = None
                self._db = None
            self._connected = False
            logger.info("Disconnected from MongoDB")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from MongoDB: {e}")
            return False

    def is_connected(self) -> bool:
        """Check if connected to MongoDB."""
        return self._connected and self._db is not None

    async def store_quote(
        self,
        data_type: str,
        symbol: str,
        timestamp: datetime,
        data: Dict[str, Any],
    ) -> bool:
        """Store market data quote."""
        if not self.is_connected():
            return False

        try:
            collection_name = f"{self.config.collection_prefix}{data_type}"
            collection = self._db[collection_name]

            document = {
                "symbol": symbol,
                "timestamp": timestamp,
                **data,
            }

            # Upsert document
            await collection.update_one(
                {"symbol": symbol, "timestamp": timestamp},
                {"$set": document},
                upsert=True
            )

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
            collection_name = f"{self.config.collection_prefix}{data_type}"
            collection = self._db[collection_name]

            # Use ordered=False for better performance
            result = await collection.insert_many(records, ordered=False)
            return len(result.inserted_ids)

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
            collection_name = f"{self.config.collection_prefix}{data_type}"
            collection = self._db[collection_name]

            document = await collection.find_one(
                {"symbol": symbol},
                sort=[("timestamp", -1)]
            )

            if document:
                document.pop("_id", None)
                return document
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
            collection_name = f"{self.config.collection_prefix}{data_type}"
            collection = self._db[collection_name]

            cursor = collection.find({
                "symbol": symbol,
                "timestamp": {"$gte": start_time, "$lte": end_time}
            }).sort("timestamp", 1)

            if limit:
                cursor = cursor.limit(limit)

            documents = await cursor.to_list(length=limit or 10000)

            # Remove _id field
            for doc in documents:
                doc.pop("_id", None)

            return documents

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
            collection_name = f"{self.config.collection_prefix}{data_type}"
            collection = self._db[collection_name]

            # Convert interval to milliseconds
            interval_ms = self._interval_to_ms(interval)

            pipeline = [
                {
                    "$match": {
                        "symbol": symbol,
                        "timestamp": {"$gte": start_time, "$lte": end_time}
                    }
                },
                {
                    "$group": {
                        "_id": {
                            "bucket": {
                                "$toDate": {
                                    "$subtract": [
                                        {"$toLong": "$timestamp"},
                                        {"$mod": [{"$toLong": "$timestamp"}, interval_ms]}
                                    ]
                                }
                            },
                            "symbol": "$symbol"
                        },
                        "open": {"$first": "$price"},
                        "high": {"$max": "$price"},
                        "low": {"$min": "$price"},
                        "close": {"$last": "$price"},
                        "volume": {"$sum": "$volume"}
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "bucket": "$_id.bucket",
                        "symbol": "$_id.symbol",
                        "open": 1,
                        "high": 1,
                        "low": 1,
                        "close": 1,
                        "volume": 1
                    }
                },
                {"$sort": {"bucket": 1}}
            ]

            documents = await collection.aggregate(pipeline).to_list(length=None)
            return documents

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
            collection_name = f"{self.config.collection_prefix}{data_type}"
            collection = self._db[collection_name]

            result = await collection.delete_many({
                "timestamp": {"$lt": before_timestamp}
            })

            return result.deleted_count

        except Exception as e:
            logger.error(f"Error deleting old data: {e}")
            return 0

    async def create_indexes(self) -> bool:
        """Create database indexes for optimized queries."""
        if not self.is_connected():
            return False

        try:
            table_types = ["equity", "fx", "bond", "commodity", "rates"]

            for data_type in table_types:
                collection_name = f"{self.config.collection_prefix}{data_type}"
                collection = self._db[collection_name]

                # Create compound index for symbol + timestamp
                await collection.create_index([
                    ("symbol", 1),
                    ("timestamp", -1)
                ], background=True)

                # Create timestamp index
                await collection.create_index([("timestamp", -1)], background=True)

            logger.info("Created MongoDB indexes")
            return True

        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
            return False

    async def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        if not self.is_connected():
            return {}

        try:
            db_stats = await self._db.command("dbStats")

            stats = {
                "database": self.config.database,
                "collections": db_stats.get("collections", 0),
                "total_size_bytes": db_stats.get("dataSize", 0),
                "index_size_bytes": db_stats.get("indexSize", 0),
                "storage_size_bytes": db_stats.get("storageSize", 0),
            }

            return stats

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}

    @staticmethod
    def _interval_to_ms(interval: str) -> int:
        """Convert interval string to milliseconds."""
        interval_map = {
            "1s": 1000,
            "1min": 60000,
            "5min": 300000,
            "15min": 900000,
            "1hour": 3600000,
            "1day": 86400000,
        }
        return interval_map.get(interval, 60000)
