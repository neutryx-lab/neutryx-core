"""Vendor-specific feed implementations and validation-aware callbacks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from neutryx.market.adapters.base import BaseMarketDataAdapter
from neutryx.market.data_models import AssetClass, MarketDataPoint

from neutryx.data.validation import ValidationResult

from .base import CallbackFn, FeedState
from .realtime import PollingMarketDataFeed


@dataclass(frozen=True)
class VendorSubscription:
    """Declarative configuration for vendor-specific subscriptions."""

    name: str
    instrument_id: str
    asset_class: AssetClass
    adapter_method: str
    polling_interval: float = 1.0
    params: Dict[str, object] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)
    callback: Optional[CallbackFn] = None


class ValidationCallback:
    """Callback helper that stores validation results for downstream processing."""

    def __init__(
        self,
        *,
        store_results: bool = True,
        forward_to: Optional[CallbackFn] = None,
    ) -> None:
        self._store_results = store_results
        self._forward_to = forward_to
        self._records: List[tuple[MarketDataPoint, ValidationResult]] = []

    def __call__(
        self,
        data_point: MarketDataPoint,
        validation_result: Optional[ValidationResult] = None,
    ) -> None:
        if validation_result is None:
            validation_result = ValidationResult(
                is_valid=True,
                quality=data_point.quality,
                issues=[],
                metadata=data_point.metadata,
            )

        if self._store_results:
            self._records.append((data_point, validation_result))

        if self._forward_to is not None:
            self._forward_to(data_point, validation_result)

    @property
    def results(self) -> Sequence[tuple[MarketDataPoint, ValidationResult]]:
        """Return collected validation results."""

        return tuple(self._records)

    def clear(self) -> None:
        """Clear stored validation records."""

        self._records.clear()


class VendorPollingFeed(PollingMarketDataFeed):
    """Base class for vendor-specific polling feeds with default subscriptions."""

    def __init__(
        self,
        adapter: BaseMarketDataAdapter,
        *,
        default_subscriptions: Optional[Iterable[VendorSubscription]] = None,
        **kwargs,
    ) -> None:
        super().__init__(adapter, **kwargs)
        self._default_subscriptions: Dict[str, VendorSubscription] = {
            subscription.name: subscription
            for subscription in (default_subscriptions or [])
        }
        self._default_subscription_ids: Dict[str, str] = {}

    def add_default_subscription(self, subscription: VendorSubscription) -> None:
        """Register a default subscription to be activated on start."""

        self._default_subscriptions[subscription.name] = subscription

        if self.state == FeedState.RUNNING:
            self._default_subscription_ids[subscription.name] = self._register_subscription(
                subscription
            )

    def remove_default_subscription(self, name: str) -> None:
        """Remove a default subscription and unsubscribe if active."""

        self._default_subscriptions.pop(name, None)

        subscription_id = self._default_subscription_ids.pop(name, None)
        if subscription_id:
            super().unsubscribe(subscription_id)

    def _register_subscription(self, subscription: VendorSubscription) -> str:
        callback = subscription.callback
        return self.subscribe_instrument(
            subscription.instrument_id,
            asset_class=subscription.asset_class,
            adapter_method=subscription.adapter_method,
            polling_interval=subscription.polling_interval,
            params=subscription.params,
            callback=callback,
            metadata=subscription.metadata,
        )

    def start(self) -> None:
        """Connect adapter and activate default subscriptions."""

        for name, subscription in self._default_subscriptions.items():
            if name not in self._default_subscription_ids:
                self._default_subscription_ids[name] = self._register_subscription(subscription)

        super().start()

    async def stop(self) -> None:
        """Stop feed and remove default subscriptions."""

        await super().stop()

        for name, subscription_id in list(self._default_subscription_ids.items()):
            super().unsubscribe(subscription_id)
            self._default_subscription_ids.pop(name, None)

    @property
    def default_subscription_ids(self) -> Dict[str, str]:
        """Return mapping of default subscription names to ids."""

        return dict(self._default_subscription_ids)


class BloombergMarketDataFeed(VendorPollingFeed):
    """Polling feed configured for the Bloomberg adapter."""

    pass


class RefinitivMarketDataFeed(VendorPollingFeed):
    """Polling feed configured for the Refinitiv adapter."""

    pass


class SimulatedMarketDataFeed(VendorPollingFeed):
    """Polling feed configured for the simulated adapter."""

    pass


__all__ = [
    "BloombergMarketDataFeed",
    "RefinitivMarketDataFeed",
    "SimulatedMarketDataFeed",
    "ValidationCallback",
    "VendorPollingFeed",
    "VendorSubscription",
]

