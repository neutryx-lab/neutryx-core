"""Data subscriber for market data feeds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class SubscriptionRequest:
    """
    Subscription request.

    Attributes:
        data_type: Type of data (equity, fx, etc.)
        symbols: List of symbols to subscribe to
        fields: Optional list of fields to subscribe to
        callback: Optional callback for data updates
    """
    data_type: str
    symbols: List[str]
    fields: Optional[List[str]] = None
    callback: Optional[Callable[[Dict[str, Any]], None]] = None


class DataSubscriber:
    """
    Data subscriber for market data.

    Manages subscriptions and callbacks for real-time market data.

    Example:
        >>> subscriber = DataSubscriber()
        >>> def on_data(data):
        ...     print(f"Received: {data}")
        >>>
        >>> subscriber.subscribe(
        ...     SubscriptionRequest(
        ...         data_type="equity",
        ...         symbols=["AAPL", "MSFT"],
        ...         callback=on_data
        ...     )
        ... )
    """

    def __init__(self):
        """Initialize subscriber."""
        self._subscriptions: Dict[str, SubscriptionRequest] = {}
        self._global_callbacks: List[Callable] = []

    def subscribe(self, request: SubscriptionRequest) -> str:
        """
        Subscribe to data.

        Args:
            request: Subscription request

        Returns:
            Subscription ID
        """
        subscription_id = f"{request.data_type}_{len(self._subscriptions)}"
        self._subscriptions[subscription_id] = request
        logger.info(
            f"Subscribed to {len(request.symbols)} {request.data_type} symbols: {subscription_id}"
        )
        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from data.

        Args:
            subscription_id: Subscription ID

        Returns:
            True if unsubscribed, False if not found
        """
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            logger.info(f"Unsubscribed: {subscription_id}")
            return True
        return False

    def add_global_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Add global callback for all data updates.

        Args:
            callback: Callback function
        """
        self._global_callbacks.append(callback)

    def notify(self, data: Dict[str, Any]):
        """
        Notify subscribers of data update.

        Args:
            data: Data update
        """
        # Call global callbacks
        for callback in self._global_callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in global callback: {e}")

        # Call subscription-specific callbacks
        data_type = data.get("data_type")
        symbol = data.get("symbol")

        for subscription in self._subscriptions.values():
            if subscription.data_type == data_type and symbol in subscription.symbols:
                if subscription.callback:
                    try:
                        subscription.callback(data)
                    except Exception as e:
                        logger.error(f"Error in subscription callback: {e}")

    def get_subscriptions(self) -> List[SubscriptionRequest]:
        """
        Get all subscriptions.

        Returns:
            List of subscription requests
        """
        return list(self._subscriptions.values())
