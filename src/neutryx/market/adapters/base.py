"""
Base market data adapter with common functionality.

Provides abstract interface and shared utilities for all data adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Callable
import logging
from enum import Enum

from ..data_models import (
    MarketDataPoint,
    EquityQuote,
    BondQuote,
    FXQuote,
    InterestRateQuote,
    CommodityQuote,
    CreditSpreadQuote,
    VolatilityQuote,
    YieldCurve,
    DataRequest,
    DataResponse,
    DataQuality,
)


logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Adapter connection state."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class AdapterConfig:
    """
    Configuration for market data adapters.

    Attributes:
        adapter_name: Adapter identifier
        timeout_ms: Request timeout in milliseconds
        retry_attempts: Number of retry attempts
        retry_delay_ms: Delay between retries in milliseconds
        cache_enabled: Enable caching
        cache_ttl_seconds: Cache time-to-live
        rate_limit_per_second: Rate limit (requests per second)
        connection_params: Vendor-specific connection parameters
    """
    adapter_name: str
    timeout_ms: int = 5000
    retry_attempts: int = 3
    retry_delay_ms: int = 1000
    cache_enabled: bool = True
    cache_ttl_seconds: int = 60
    rate_limit_per_second: int = 100
    connection_params: Dict[str, Any] = field(default_factory=dict)


class BaseMarketDataAdapter(ABC):
    """
    Abstract base class for all market data adapters.

    Provides common functionality for connection management, error handling,
    retry logic, and data normalization.

    Attributes:
        config: Adapter configuration
        connection_state: Current connection state
        statistics: Adapter statistics (requests, errors, latency)
    """

    def __init__(self, config: AdapterConfig):
        """
        Initialize adapter.

        Args:
            config: Adapter configuration
        """
        self.config = config
        self.connection_state = ConnectionState.DISCONNECTED
        self.statistics = {
            "requests": 0,
            "successes": 0,
            "errors": 0,
            "avg_latency_ms": 0.0,
            "last_request_time": None,
        }
        self._cache: Dict[str, tuple[Any, datetime]] = {}
        self._rate_limiter: Optional[Any] = None
        self._connection_callbacks: List[Callable] = []

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to data source.

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from data source.

        Returns:
            True if disconnection successful, False otherwise
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if adapter is connected.

        Returns:
            True if connected, False otherwise
        """
        pass

    @abstractmethod
    def get_equity_quote(
        self, ticker: str, exchange: Optional[str] = None
    ) -> Optional[EquityQuote]:
        """
        Get equity quote.

        Args:
            ticker: Stock ticker
            exchange: Exchange code (optional)

        Returns:
            EquityQuote or None if not available
        """
        pass

    @abstractmethod
    def get_bond_quote(
        self, identifier: str, id_type: str = "isin"
    ) -> Optional[BondQuote]:
        """
        Get bond quote.

        Args:
            identifier: Bond identifier (ISIN, CUSIP, etc.)
            id_type: Identifier type

        Returns:
            BondQuote or None if not available
        """
        pass

    @abstractmethod
    def get_fx_quote(
        self, base_currency: str, quote_currency: str
    ) -> Optional[FXQuote]:
        """
        Get FX quote.

        Args:
            base_currency: Base currency code
            quote_currency: Quote currency code

        Returns:
            FXQuote or None if not available
        """
        pass

    @abstractmethod
    def get_interest_rate(
        self, rate_type: str, currency: str, tenor: str
    ) -> Optional[InterestRateQuote]:
        """
        Get interest rate quote.

        Args:
            rate_type: Rate type (LIBOR, SOFR, etc.)
            currency: Currency code
            tenor: Tenor (e.g., "3M", "6M")

        Returns:
            InterestRateQuote or None if not available
        """
        pass

    @abstractmethod
    def get_commodity_quote(
        self, commodity_code: str, exchange: Optional[str] = None
    ) -> Optional[CommodityQuote]:
        """
        Get commodity quote.

        Args:
            commodity_code: Commodity code
            exchange: Exchange code (optional)

        Returns:
            CommodityQuote or None if not available
        """
        pass

    @abstractmethod
    def get_credit_spread(
        self, issuer: str, tenor: str, currency: str = "USD"
    ) -> Optional[CreditSpreadQuote]:
        """
        Get credit spread quote.

        Args:
            issuer: Issuer identifier
            tenor: Tenor
            currency: Currency code

        Returns:
            CreditSpreadQuote or None if not available
        """
        pass

    @abstractmethod
    def get_volatility_quote(
        self, underlying: str, strike: float, expiry: date, option_type: str = "call"
    ) -> Optional[VolatilityQuote]:
        """
        Get volatility quote.

        Args:
            underlying: Underlying asset identifier
            strike: Strike price
            expiry: Expiry date
            option_type: "call" or "put"

        Returns:
            VolatilityQuote or None if not available
        """
        pass

    @abstractmethod
    def get_yield_curve(
        self, curve_name: str, currency: str = "USD"
    ) -> Optional[YieldCurve]:
        """
        Get yield curve.

        Args:
            curve_name: Curve identifier
            currency: Currency code

        Returns:
            YieldCurve or None if not available
        """
        pass

    def execute_request(self, request: DataRequest) -> DataResponse:
        """
        Execute a market data request.

        Handles retry logic, error handling, and statistics tracking.

        Args:
            request: Data request specification

        Returns:
            DataResponse with results or error information
        """
        start_time = datetime.utcnow()
        self.statistics["requests"] += 1
        self.statistics["last_request_time"] = start_time

        for attempt in range(self.config.retry_attempts):
            try:
                # Check rate limit
                if self._rate_limiter:
                    self._rate_limiter.wait_if_needed()

                # Execute request
                data = self._execute_request_internal(request)

                # Calculate latency
                latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

                # Update statistics
                self.statistics["successes"] += 1
                self._update_avg_latency(latency_ms)

                return DataResponse(
                    request=request,
                    data=data,
                    success=True,
                    latency_ms=latency_ms,
                )

            except Exception as e:
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.config.retry_attempts}): {e}"
                )
                if attempt < self.config.retry_attempts - 1:
                    import time
                    time.sleep(self.config.retry_delay_ms / 1000)
                else:
                    self.statistics["errors"] += 1
                    latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                    return DataResponse(
                        request=request,
                        data=[],
                        success=False,
                        error_message=str(e),
                        latency_ms=latency_ms,
                    )

    @abstractmethod
    def _execute_request_internal(self, request: DataRequest) -> List[MarketDataPoint]:
        """
        Internal method to execute request (to be implemented by subclasses).

        Args:
            request: Data request

        Returns:
            List of market data points
        """
        pass

    def _update_avg_latency(self, latency_ms: float):
        """Update average latency statistic."""
        current_avg = self.statistics["avg_latency_ms"]
        count = self.statistics["successes"]
        self.statistics["avg_latency_ms"] = (
            current_avg * (count - 1) + latency_ms
        ) / count

    def _check_cache(self, cache_key: str) -> Optional[Any]:
        """
        Check cache for data.

        Args:
            cache_key: Cache key

        Returns:
            Cached data or None if not found or expired
        """
        if not self.config.cache_enabled:
            return None

        if cache_key in self._cache:
            data, timestamp = self._cache[cache_key]
            age_seconds = (datetime.utcnow() - timestamp).total_seconds()
            if age_seconds < self.config.cache_ttl_seconds:
                return data
            else:
                del self._cache[cache_key]

        return None

    def _store_cache(self, cache_key: str, data: Any):
        """
        Store data in cache.

        Args:
            cache_key: Cache key
            data: Data to cache
        """
        if self.config.cache_enabled:
            self._cache[cache_key] = (data, datetime.utcnow())

    def _clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get adapter statistics.

        Returns:
            Dictionary with statistics
        """
        return self.statistics.copy()

    def reset_statistics(self):
        """Reset adapter statistics."""
        self.statistics = {
            "requests": 0,
            "successes": 0,
            "errors": 0,
            "avg_latency_ms": 0.0,
            "last_request_time": None,
        }

    def add_connection_callback(self, callback: Callable[[ConnectionState], None]):
        """
        Add callback for connection state changes.

        Args:
            callback: Callback function taking ConnectionState parameter
        """
        self._connection_callbacks.append(callback)

    def _notify_connection_state_change(self, new_state: ConnectionState):
        """Notify callbacks of connection state change."""
        old_state = self.connection_state
        self.connection_state = new_state
        logger.info(f"Connection state: {old_state} -> {new_state}")

        for callback in self._connection_callbacks:
            try:
                callback(new_state)
            except Exception as e:
                logger.error(f"Error in connection callback: {e}")
