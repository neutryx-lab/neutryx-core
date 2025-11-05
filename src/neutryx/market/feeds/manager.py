"""Real-time market data feed manager."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from ..adapters.base import BaseMarketDataAdapter
from ..storage.base import BaseStorage
from ..validation.pipeline import ValidationPipeline

logger = logging.getLogger(__name__)


class FeedStatus(Enum):
    """Feed status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class FeedConfig:
    """
    Feed manager configuration.

    Attributes:
        enable_validation: Enable data validation
        enable_storage: Enable automatic storage
        buffer_size: Size of data buffer
        flush_interval_seconds: How often to flush buffer to storage
        max_errors: Maximum consecutive errors before pausing
        enable_failover: Enable automatic failover between adapters
    """
    enable_validation: bool = True
    enable_storage: bool = True
    buffer_size: int = 1000
    flush_interval_seconds: int = 60
    max_errors: int = 10
    enable_failover: bool = True


class FeedManager:
    """
    Real-time market data feed manager.

    Orchestrates market data adapters, validation, and storage with
    automatic failover and error handling.

    Features:
    - Multi-adapter support with failover
    - Real-time data validation
    - Automatic storage persistence
    - Subscription management
    - Quality monitoring

    Example:
        >>> from neutryx.market.feeds import FeedManager, FeedConfig
        >>> from neutryx.market.adapters import BloombergAdapter
        >>> from neutryx.market.storage import TimescaleDBStorage
        >>> from neutryx.market.validation import ValidationPipeline
        >>>
        >>> # Initialize components
        >>> adapter = BloombergAdapter(config)
        >>> storage = TimescaleDBStorage(storage_config)
        >>> pipeline = ValidationPipeline()
        >>>
        >>> # Create feed manager
        >>> manager = FeedManager(
        ...     adapters=[adapter],
        ...     storage=storage,
        ...     validation_pipeline=pipeline,
        ...     config=FeedConfig()
        ... )
        >>>
        >>> # Start feed
        >>> await manager.start()
        >>> await manager.subscribe("equity", ["AAPL", "MSFT"])
    """

    def __init__(
        self,
        adapters: List[BaseMarketDataAdapter],
        storage: Optional[BaseStorage] = None,
        validation_pipeline: Optional[ValidationPipeline] = None,
        config: Optional[FeedConfig] = None,
    ):
        """
        Initialize feed manager.

        Args:
            adapters: List of market data adapters (ordered by priority)
            storage: Storage backend for persisting data
            validation_pipeline: Validation pipeline for quality checks
            config: Feed configuration
        """
        self.adapters = adapters
        self.storage = storage
        self.validation_pipeline = validation_pipeline
        self.config = config or FeedConfig()

        self.status = FeedStatus.STOPPED
        self._current_adapter_idx = 0
        self._buffer: List[Dict[str, Any]] = []
        self._subscriptions: Dict[str, List[str]] = {}
        self._callbacks: List[Callable] = []
        self._error_count = 0
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> bool:
        """
        Start feed manager.

        Returns:
            True if started successfully, False otherwise
        """
        if self.status == FeedStatus.RUNNING:
            logger.warning("Feed manager already running")
            return True

        try:
            self.status = FeedStatus.STARTING

            # Connect to primary adapter
            if not await self._connect_adapter(0):
                logger.error("Failed to connect to any adapter")
                self.status = FeedStatus.ERROR
                return False

            # Connect to storage if enabled
            if self.config.enable_storage and self.storage:
                if not await self.storage.connect():
                    logger.error("Failed to connect to storage")
                    self.status = FeedStatus.ERROR
                    return False

            self.status = FeedStatus.RUNNING
            logger.info("Feed manager started")

            # Start background tasks
            self._task = asyncio.create_task(self._run())

            return True

        except Exception as e:
            logger.error(f"Error starting feed manager: {e}")
            self.status = FeedStatus.ERROR
            return False

    async def stop(self) -> bool:
        """
        Stop feed manager.

        Returns:
            True if stopped successfully, False otherwise
        """
        if self.status == FeedStatus.STOPPED:
            return True

        try:
            self.status = FeedStatus.STOPPED

            # Cancel background task
            if self._task:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass

            # Flush buffer
            await self._flush_buffer()

            # Disconnect adapters
            for adapter in self.adapters:
                if adapter.is_connected():
                    await asyncio.to_thread(adapter.disconnect)

            # Disconnect storage
            if self.storage and self.storage.is_connected():
                await self.storage.disconnect()

            logger.info("Feed manager stopped")
            return True

        except Exception as e:
            logger.error(f"Error stopping feed manager: {e}")
            return False

    async def subscribe(self, data_type: str, symbols: List[str]):
        """
        Subscribe to market data.

        Args:
            data_type: Type of data (equity, fx, etc.)
            symbols: List of symbols to subscribe to
        """
        if data_type not in self._subscriptions:
            self._subscriptions[data_type] = []

        self._subscriptions[data_type].extend(symbols)
        logger.info(f"Subscribed to {len(symbols)} {data_type} symbols")

    async def unsubscribe(self, data_type: str, symbols: List[str]):
        """
        Unsubscribe from market data.

        Args:
            data_type: Type of data
            symbols: List of symbols to unsubscribe from
        """
        if data_type in self._subscriptions:
            for symbol in symbols:
                if symbol in self._subscriptions[data_type]:
                    self._subscriptions[data_type].remove(symbol)
            logger.info(f"Unsubscribed from {len(symbols)} {data_type} symbols")

    def add_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Add callback for data updates.

        Args:
            callback: Callback function receiving data updates
        """
        self._callbacks.append(callback)

    async def _run(self):
        """Main event loop."""
        while self.status == FeedStatus.RUNNING:
            try:
                # Flush buffer periodically
                await asyncio.sleep(self.config.flush_interval_seconds)
                await self._flush_buffer()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in feed loop: {e}")
                self._handle_error()

    async def _connect_adapter(self, adapter_idx: int) -> bool:
        """Connect to adapter at given index."""
        if adapter_idx >= len(self.adapters):
            return False

        adapter = self.adapters[adapter_idx]
        success = await asyncio.to_thread(adapter.connect)

        if success:
            self._current_adapter_idx = adapter_idx
            self._error_count = 0
            logger.info(f"Connected to adapter: {adapter.config.adapter_name}")
            return True

        return False

    async def _failover(self) -> bool:
        """Failover to next adapter."""
        if not self.config.enable_failover:
            return False

        next_idx = (self._current_adapter_idx + 1) % len(self.adapters)
        logger.warning(f"Attempting failover to adapter {next_idx}")

        return await self._connect_adapter(next_idx)

    def _handle_error(self):
        """Handle error and trigger failover if needed."""
        self._error_count += 1

        if self._error_count >= self.config.max_errors:
            logger.error(f"Max errors ({self.config.max_errors}) reached")
            asyncio.create_task(self._failover())

    async def _flush_buffer(self):
        """Flush data buffer to storage."""
        if not self._buffer or not self.config.enable_storage or not self.storage:
            return

        try:
            # Group by data type
            by_type: Dict[str, List[Dict[str, Any]]] = {}
            for data in self._buffer:
                data_type = data.get("data_type", "unknown")
                if data_type not in by_type:
                    by_type[data_type] = []
                by_type[data_type].append(data)

            # Store each type
            for data_type, records in by_type.items():
                count = await self.storage.store_batch(data_type, records)
                logger.debug(f"Stored {count} {data_type} records")

            self._buffer.clear()

        except Exception as e:
            logger.error(f"Error flushing buffer: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get feed statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "status": self.status.value,
            "current_adapter": self.adapters[self._current_adapter_idx].config.adapter_name,
            "error_count": self._error_count,
            "buffer_size": len(self._buffer),
            "subscriptions": {
                data_type: len(symbols)
                for data_type, symbols in self._subscriptions.items()
            },
        }

        if self.validation_pipeline:
            stats["quality_score"] = self.validation_pipeline.get_quality_score()

        return stats
