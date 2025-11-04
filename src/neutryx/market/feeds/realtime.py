"""Real-time market data feed implementation."""

from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence

from neutryx.data.validation import DataValidator, ValidationResult
from neutryx.market.adapters.base import BaseMarketDataAdapter
from neutryx.market.data_models import AssetClass, MarketDataPoint

from .base import CallbackFn, FeedError, FeedState, FeedSubscription, MarketDataFeed

if TYPE_CHECKING:
    from neutryx.integrations.databases.base import DatabaseConnector


logger = logging.getLogger(__name__)


@dataclass
class FeedMetrics:
    """Metrics collected by the real-time feed."""

    delivered: int = 0
    dropped: int = 0
    errors: int = 0
    last_error: Optional[str] = None


class PollingMarketDataFeed(MarketDataFeed):
    """
    Real-time feed that polls adapter methods at a fixed cadence.

    The feed coordinates adapter access, applies validation rules, and
    optionally persists market data points using the configured database
    connector.
    """

    def __init__(
        self,
        adapter: BaseMarketDataAdapter,
        *,
        validator: Optional[DataValidator] = None,
        storage: Optional["DatabaseConnector"] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        super().__init__()
        self._adapter = adapter
        self._validator = validator
        self._storage = storage
        self._loop = loop
        self._tasks: Dict[str, asyncio.Task] = {}
        self._metrics = FeedMetrics()

    @property
    def metrics(self) -> FeedMetrics:
        """Return feed metrics."""
        return self._metrics

    def subscribe_instrument(
        self,
        instrument_id: str,
        *,
        asset_class: AssetClass,
        adapter_method: str,
        polling_interval: float = 1.0,
        params: Optional[Dict[str, object]] = None,
        callback: Optional[CallbackFn] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> str:
        """Convenience wrapper to register an instrument subscription."""
        subscription_id = str(uuid.uuid4())
        subscription = FeedSubscription(
            subscription_id=subscription_id,
            asset_class=asset_class,
            instrument_id=instrument_id,
            adapter_method=adapter_method,
            polling_interval=polling_interval,
            params=dict(params or {}),
            callback=callback,
            metadata=dict(metadata or {}),
        )

        super().subscribe(subscription)

        if self.state == FeedState.RUNNING:
            self._spawn_task(subscription.subscription_id, subscription)

        return subscription_id

    def start(self) -> None:
        """Start polling all registered subscriptions."""
        if self.state in (FeedState.RUNNING, FeedState.STARTING):
            return

        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.get_event_loop_policy().get_event_loop()

        self._state = FeedState.STARTING

        if not self._adapter.is_connected():
            connected = self._adapter.connect()
            if not connected:
                self._state = FeedState.STOPPED
                raise FeedError("Failed to connect to market data adapter")

        self._state = FeedState.RUNNING
        for subscription_id, subscription in list(self._subscriptions.items()):
            self._spawn_task(subscription_id, subscription)

    async def stop(self) -> None:
        """Stop polling and cancel all tasks."""
        if self.state == FeedState.STOPPED:
            return

        self._state = FeedState.STOPPING

        for task in list(self._tasks.values()):
            task.cancel()

        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()

        if self._adapter.is_connected():
            self._adapter.disconnect()

        self._state = FeedState.STOPPED

    def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe and cancel associated polling task."""
        if subscription_id in self._tasks:
            self._tasks[subscription_id].cancel()
            del self._tasks[subscription_id]

        super().unsubscribe(subscription_id)

    def _spawn_task(self, subscription_id: str, subscription: FeedSubscription) -> None:
        """Spawn polling task for a subscription."""
        if self._loop is None:
            raise FeedError("Event loop not initialized; call start() first")

        if subscription_id in self._tasks:
            return

        task = self._loop.create_task(self._run_subscription(subscription_id))
        self._tasks[subscription_id] = task

    async def _run_subscription(self, subscription_id: str) -> None:
        """Polling loop for a subscription."""
        subscription = self.get_subscription(subscription_id)
        if subscription is None:
            return

        adapter_method = getattr(self._adapter, subscription.adapter_method, None)
        if adapter_method is None:
            self._metrics.errors += 1
            self._metrics.last_error = (
                f"Adapter missing method '{subscription.adapter_method}'"
            )
            logger.error(self._metrics.last_error)
            return

        interval = max(subscription.polling_interval, 0.05)

        while self.state == FeedState.RUNNING and subscription.is_active:
            try:
                result = await asyncio.to_thread(
                    adapter_method,
                    **subscription.params,
                )
            except asyncio.CancelledError:  # pragma: no cover - handled by asyncio
                raise
            except Exception as exc:  # pragma: no cover - defensive against adapter failures
                self._metrics.errors += 1
                self._metrics.last_error = str(exc)
                logger.exception(
                    "Polling error for %s (%s): %s",
                    subscription.instrument_id,
                    subscription.adapter_method,
                    exc,
                )
                await asyncio.sleep(interval)
                continue

            data_points = self._normalize_result(result)

            if not data_points:
                self._metrics.dropped += 1
                await asyncio.sleep(interval)
                continue

            for point in data_points:
                subscription.last_update = datetime.utcnow()

                validation_result: Optional[ValidationResult] = None
                if self._validator:
                    validation_result = self._validator.validate(point)

                if self._storage:
                    await self._persist(point, subscription)

                await self._dispatch(subscription, point, validation_result)
                self._metrics.delivered += 1

            await asyncio.sleep(interval)

    async def _dispatch(
        self,
        subscription: FeedSubscription,
        data_point: MarketDataPoint,
        validation_result: Optional[ValidationResult],
    ) -> None:
        """Invoke the subscription callback with the data point."""
        callback = subscription.callback
        if not callback:
            return

        try:
            result = callback(data_point)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # pragma: no cover - defensive
            self._metrics.errors += 1
            self._metrics.last_error = str(exc)
            logger.exception(
                "Callback error for %s (%s): %s",
                subscription.instrument_id,
                subscription.adapter_method,
                exc,
            )

    async def _persist(
        self,
        data_point: MarketDataPoint,
        subscription: FeedSubscription,
    ) -> None:
        """Persist data point using configured storage connector."""
        if not self._storage:
            return

        try:
            await self._storage.write_market_data(
                [data_point],
                asset_class=subscription.asset_class,
                instrument_id=subscription.instrument_id,
            )
        except Exception as exc:  # pragma: no cover - database failures should not crash feed
            self._metrics.errors += 1
            self._metrics.last_error = str(exc)
            logger.exception("Storage error for %s: %s", subscription.instrument_id, exc)

    @staticmethod
    def _normalize_result(result: object) -> List[MarketDataPoint]:
        """Normalize adapter responses into a list of data points."""
        if result is None:
            return []

        if isinstance(result, MarketDataPoint):
            return [result]

        if isinstance(result, Iterable):
            points = [
                value for value in result if isinstance(value, MarketDataPoint)
            ]
            return points

        logger.debug("Dropping unsupported adapter result type: %s", type(result))
        return []
