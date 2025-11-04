"""Base classes for database connectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence
import json

from neutryx.market.data_models import MarketDataPoint, DataQuality


class DatabaseConnectorError(RuntimeError):
    """Raised when database operations fail."""


@dataclass
class DatabaseConfig:
    """Generic database configuration."""

    dsn: Optional[str] = None
    host: str = "localhost"
    port: Optional[int] = None
    database: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    connect_timeout: float = 5.0
    options: Dict[str, Any] = field(default_factory=dict)


class DatabaseConnector(ABC):
    """Abstract base class for database connectors."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._connected = False

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the database."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the database."""

    def is_connected(self) -> bool:
        """Return connection status."""
        return self._connected

    async def write_market_data(
        self,
        points: Sequence[MarketDataPoint],
        *,
        asset_class: Optional[Any] = None,
        instrument_id: Optional[str] = None,
    ) -> None:
        """Persist a batch of market data points."""
        raise NotImplementedError

    def _serialize_point(self, point: MarketDataPoint) -> Dict[str, Any]:
        """Convert a market data point into JSON-serializable payload."""
        payload = asdict(point)

        for key, value in list(payload.items()):
            if isinstance(value, datetime):
                payload[key] = value.isoformat()
            elif isinstance(value, date):
                payload[key] = value.isoformat()

        return payload

    def _prepare_records(
        self,
        points: Sequence[MarketDataPoint],
        *,
        asset_class: Optional[Any] = None,
        instrument_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Prepare generic records for persistence."""
        records: List[Dict[str, Any]] = []

        for point in points:
            payload = self._serialize_point(point)
            quality = (
                point.quality.value
                if isinstance(point.quality, DataQuality)
                else str(point.quality)
            )

            records.append(
                {
                    "timestamp": point.timestamp,
                    "source": point.source,
                    "quality": quality,
                    "asset_class": str(asset_class) if asset_class else None,
                    "instrument_id": instrument_id,
                    "payload": payload,
                    "metadata": payload.get("metadata", {}),
                }
            )

        return records

    @staticmethod
    def _dump_json(value: Any) -> str:
        """Serialize value to JSON with sane defaults."""
        def _default(obj: Any):
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)!r} not JSON serializable")

        return json.dumps(value, default=_default)


__all__ = ["DatabaseConfig", "DatabaseConnector", "DatabaseConnectorError"]
