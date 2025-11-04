"""MongoDB connector using Motor (async driver)."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from .base import DatabaseConfig, DatabaseConnector, DatabaseConnectorError
from neutryx.market.data_models import MarketDataPoint


class MongoConnector(DatabaseConnector):
    """Connector that stores market data into MongoDB collections."""

    def __init__(
        self,
        config: DatabaseConfig,
        *,
        collection: str = "market_data",
    ):
        super().__init__(config)
        self.collection_name = collection
        self._client = None
        self._collection = None

    async def connect(self) -> bool:
        if self._connected:
            return True

        try:
            from motor.motor_asyncio import AsyncIOMotorClient
        except ImportError as exc:  # pragma: no cover - depends on runtime env
            raise DatabaseConnectorError(
                "motor is required for MongoDB connector. "
                "Install with `pip install motor`."
            ) from exc

        uri = self._build_uri()
        self._client = AsyncIOMotorClient(
            uri,
            serverSelectionTimeoutMS=int(self.config.connect_timeout * 1000),
            **self.config.options,
        )

        if not self.config.database:
            raise DatabaseConnectorError("Database name is required for MongoDB connector")

        self._collection = self._client[self.config.database][self.collection_name]

        try:
            await self._collection.database.command("ping")
        except Exception as exc:
            raise DatabaseConnectorError(f"Unable to connect to MongoDB: {exc}") from exc

        await self._ensure_indexes()

        self._connected = True
        return True

    async def disconnect(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
            self._collection = None
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

        documents = [
            {
                "timestamp": record["timestamp"],
                "source": record["source"],
                "quality": record["quality"],
                "asset_class": record["asset_class"],
                "instrument_id": record["instrument_id"],
                "payload": record["payload"],
                "metadata": record["metadata"],
            }
            for record in records
        ]

        await self._collection.insert_many(documents)

    async def _ensure_indexes(self) -> None:
        if not self._collection:
            return

        await self._collection.create_index("instrument_id")
        await self._collection.create_index("timestamp")

    def _build_uri(self) -> str:
        if self.config.dsn:
            return self.config.dsn

        host = self.config.host or "localhost"
        port = self.config.port or 27017

        if self.config.user:
            from urllib.parse import quote_plus

            auth = quote_plus(self.config.user)
            password = quote_plus(self.config.password or "")
            return f"mongodb://{auth}:{password}@{host}:{port}"

        return f"mongodb://{host}:{port}"


__all__ = ["MongoConnector"]
