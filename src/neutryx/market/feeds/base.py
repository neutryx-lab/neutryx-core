"""Base classes for real-time market data feeds."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional

from neutryx.market.data_models import AssetClass, MarketDataPoint


class FeedError(RuntimeError):
    """Raised when the feed encounters an unrecoverable error."""


class FeedState(Enum):
    """Lifecycle states for a market data feed."""

    STOPPED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()


CallbackFn = Callable[[MarketDataPoint], Any]


@dataclass
class FeedSubscription:
    """Represents a feed subscription."""

    subscription_id: str
    asset_class: AssetClass
    instrument_id: str
    adapter_method: str
    polling_interval: float
    params: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[CallbackFn] = None
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_update: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MarketDataFeed:
    """Abstract base class for market data feeds."""

    def __init__(self) -> None:
        self._state = FeedState.STOPPED
        self._subscriptions: Dict[str, FeedSubscription] = {}

    @property
    def state(self) -> FeedState:
        """Current feed state."""
        return self._state

    def subscribe(self, subscription: FeedSubscription) -> str:
        """Register a subscription with the feed."""
        self._subscriptions[subscription.subscription_id] = subscription
        return subscription.subscription_id

    def unsubscribe(self, subscription_id: str) -> None:
        """Remove a subscription."""
        if subscription_id in self._subscriptions:
            self._subscriptions[subscription_id].is_active = False
            del self._subscriptions[subscription_id]

    def get_subscription(self, subscription_id: str) -> Optional[FeedSubscription]:
        """Return a subscription by id."""
        return self._subscriptions.get(subscription_id)

    def start(self) -> None:  # pragma: no cover - to be implemented by subclasses
        """Start the feed."""
        raise NotImplementedError

    async def stop(self) -> None:  # pragma: no cover - to be implemented by subclasses
        """Stop the feed."""
        raise NotImplementedError

