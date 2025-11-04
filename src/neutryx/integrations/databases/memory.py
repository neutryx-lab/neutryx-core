"""In-memory connector for testing and development."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from .base import DatabaseConfig, DatabaseConnector
from neutryx.market.data_models import MarketDataPoint


class InMemoryConnector(DatabaseConnector):
    """Simple in-memory connector that stores records in a list."""

    def __init__(self):
        super().__init__(DatabaseConfig())
        self._storage: List[dict[str, Any]] = []

    async def connect(self) -> bool:
        self._connected = True
        return True

    async def disconnect(self) -> None:
        self._connected = False

    async def write_market_data(
        self,
        points: Sequence[MarketDataPoint],
        *,
        asset_class: Optional[Any] = None,
        instrument_id: Optional[str] = None,
    ) -> None:
        records = self._prepare_records(
            points,
            asset_class=asset_class,
            instrument_id=instrument_id,
        )
        self._storage.extend(records)

    def get_records(self) -> List[dict[str, Any]]:
        """Return stored records."""
        return list(self._storage)

    def clear(self) -> None:
        """Remove all stored records."""
        self._storage.clear()


__all__ = ["InMemoryConnector"]
