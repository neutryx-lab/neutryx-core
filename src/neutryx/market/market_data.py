"""
Market data source integration for real-time and historical data.

Supports Bloomberg, Refinitiv, and simulated data feeds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import jax.numpy as jnp

from neutryx.data.validation import (
    DataValidator,
    RangeRule,
    RequiredFieldRule,
    Severity,
    StalenessRule,
)
from neutryx.market.adapters import SimulatedAdapter, SimulatedConfig
from neutryx.market.adapters.bloomberg import (
    BloombergAdapter as _BloombergAdapter,
    BloombergConfig,
)
from neutryx.market.adapters.refinitiv import (
    RefinitivAdapter as _RefinitivAdapter,
    RefinitivConfig,
)
from neutryx.market.feeds import PollingMarketDataFeed

from .fx import FXVolatilityQuote

if TYPE_CHECKING:
    from neutryx.integrations.databases.base import DatabaseConnector


class MarketDataSource(ABC):
    """
    Abstract base class for market data sources.

    Implementations can connect to Bloomberg, Refinitiv, or other data providers.
    """

    @abstractmethod
    def get_fx_vol_quote(
        self,
        from_ccy: str,
        to_ccy: str,
        expiry: float,
        as_of_date: Optional[date] = None,
    ) -> FXVolatilityQuote:
        """
        Fetch FX volatility market quote.

        Args:
            from_ccy: Source currency (e.g., "EUR")
            to_ccy: Target currency (e.g., "USD")
            expiry: Time to expiry in years
            as_of_date: Valuation date (None = today)

        Returns:
            FXVolatilityQuote with ATM/BF/RR data
        """
        pass

    @abstractmethod
    def get_fx_vol_surface(
        self,
        from_ccy: str,
        to_ccy: str,
        tenors: List[float],
        as_of_date: Optional[date] = None,
    ) -> List[FXVolatilityQuote]:
        """
        Fetch FX volatility surface (multiple tenors).

        Args:
            from_ccy: Source currency
            to_ccy: Target currency
            tenors: List of expiries in years (e.g., [0.25, 0.5, 1.0, 2.0])
            as_of_date: Valuation date (None = today)

        Returns:
            List of FXVolatilityQuote for each tenor
        """
        pass

    @abstractmethod
    def get_fx_spot(
        self,
        from_ccy: str,
        to_ccy: str,
        as_of_date: Optional[date] = None,
    ) -> float:
        """
        Fetch FX spot rate.

        Args:
            from_ccy: Source currency
            to_ccy: Target currency
            as_of_date: Valuation date (None = today)

        Returns:
            Spot rate (units of to_ccy per unit of from_ccy)
        """
        pass


@dataclass
class SimulatedMarketData(MarketDataSource):
    """
    Simulated market data source for testing and examples.

    Generates realistic-looking market data using parametric models.
    Useful for backtesting, stress testing, and demonstrations.

    Attributes:
        base_vol: Base volatility level (default 0.10 = 10%)
        vol_term_structure: Volatility term structure steepness
        rr_skew: Risk reversal skew strength
        bf_convexity: Butterfly convexity strength
        seed: Random seed for reproducibility

    Example:
        >>> data = SimulatedMarketData(base_vol=0.12, rr_skew=0.02)
        >>> quote = data.get_fx_vol_quote("EUR", "USD", 1.0)
        >>> quote.atm_vol
        0.12
    """

    base_vol: float = 0.10
    vol_term_structure: float = 0.05  # Vol increases with tenor
    rr_skew: float = 0.015             # Positive RR (calls > puts)
    bf_convexity: float = 0.005        # Positive BF (wings > ATM)
    seed: Optional[int] = None

    def __post_init__(self):
        """Initialize random state."""
        if self.seed is not None:
            import numpy as np
            np.random.seed(self.seed)

    def get_fx_vol_quote(
        self,
        from_ccy: str,
        to_ccy: str,
        expiry: float,
        as_of_date: Optional[date] = None,
    ) -> FXVolatilityQuote:
        """
        Generate simulated FX volatility quote.

        ATM vol increases with tenor (term structure).
        RR and BF scaled by sqrt(tenor) to reflect typical market behavior.
        """
        import numpy as np

        # ATM vol with term structure
        atm_vol = self.base_vol + self.vol_term_structure * jnp.sqrt(expiry)
        atm_vol = float(atm_vol)

        # RR scales with sqrt(tenor)
        rr_25d = self.rr_skew * jnp.sqrt(expiry)
        rr_25d = float(rr_25d)

        # BF scales with sqrt(tenor)
        bf_25d = self.bf_convexity * jnp.sqrt(expiry)
        bf_25d = float(bf_25d)

        # Add small noise if desired
        if self.seed is None:
            noise = np.random.normal(0, 0.002)
            atm_vol += noise

        # Simulated forward (around 1.10 for EUR/USD)
        forward = 1.10 + np.random.normal(0, 0.02) if self.seed is None else 1.10

        # Typical rates
        domestic_rate = 0.025  # USD
        foreign_rate = 0.015   # EUR

        return FXVolatilityQuote(
            expiry=expiry,
            atm_vol=atm_vol,
            rr_25d=rr_25d,
            bf_25d=bf_25d,
            forward=forward,
            domestic_rate=domestic_rate,
            foreign_rate=foreign_rate,
        )

    def get_fx_vol_surface(
        self,
        from_ccy: str,
        to_ccy: str,
        tenors: List[float],
        as_of_date: Optional[date] = None,
    ) -> List[FXVolatilityQuote]:
        """Generate simulated FX volatility surface."""
        return [
            self.get_fx_vol_quote(from_ccy, to_ccy, tenor, as_of_date)
            for tenor in tenors
        ]

    def get_fx_spot(
        self,
        from_ccy: str,
        to_ccy: str,
        as_of_date: Optional[date] = None,
    ) -> float:
        """Generate simulated FX spot rate."""
        # Typical EUR/USD spot around 1.10
        if from_ccy == "EUR" and to_ccy == "USD":
            return 1.10
        elif from_ccy == "GBP" and to_ccy == "USD":
            return 1.30
        elif from_ccy == "USD" and to_ccy == "JPY":
            return 110.0
        else:
            return 1.0


@dataclass
class BloombergDataAdapter(MarketDataSource):
    """
    Bloomberg-backed market data source for FX volatility and spot data.

    This adapter wraps :class:`neutryx.market.adapters.bloomberg.BloombergAdapter`
    and exposes the simplified :class:`MarketDataSource` interface.

    The implementation requires the Bloomberg `blpapi` package at runtime and
    a valid Bloomberg Terminal or Server API session. When the dependency is
    missing or the connection fails, a :class:`RuntimeError` is raised with
    a descriptive message.
    """

    host: str = "localhost"
    port: int = 8194
    timeout: int = 5000
    application_name: str = "Neutryx"
    identity: Optional[str] = None
    use_enterprise: bool = False

    def __post_init__(self) -> None:
        self._config = BloombergConfig(
            adapter_name="bloomberg",
            host=self.host,
            port=self.port,
            timeout_ms=self.timeout,
            application_name=self.application_name,
            identity=self.identity,
            use_enterprise=self.use_enterprise,
        )
        self._adapter = _BloombergAdapter(self._config)
        self._connected = False

    def _ensure_connection(self) -> None:
        if not self._connected:
            if not self._adapter.connect():
                raise RuntimeError(
                    "Unable to connect to Bloomberg API. "
                    "Ensure 'blpapi' is installed and the session is authorized."
                )
            self._connected = True

    def get_fx_vol_quote(
        self,
        from_ccy: str,
        to_ccy: str,
        expiry: float,
        as_of_date: Optional[date] = None,
    ) -> FXVolatilityQuote:
        self._ensure_connection()
        spot_quote = self._adapter.get_fx_quote(from_ccy, to_ccy)
        if spot_quote is None:
            raise RuntimeError(f"No FX quote returned for {from_ccy}/{to_ccy}")

        tenor_days = max(1, int(expiry * 365))
        expiry_date = (as_of_date or date.today()) + timedelta(days=tenor_days)
        underlying = f"{from_ccy}{to_ccy} Curncy"

        vol_quote = self._adapter.get_volatility_quote(
            underlying=underlying,
            strike=spot_quote.spot,
            expiry=expiry_date,
            option_type="call",
        )

        if vol_quote is None:
            raise RuntimeError(
                f"No Bloomberg volatility data for {underlying} at {expiry_date}"
            )

        return FXVolatilityQuote(
            expiry=expiry,
            atm_vol=vol_quote.volatility,
            rr_25d=0.0,
            bf_25d=0.0,
            forward=spot_quote.spot,
            domestic_rate=0.0,
            foreign_rate=0.0,
        )

    def get_fx_vol_surface(
        self,
        from_ccy: str,
        to_ccy: str,
        tenors: List[float],
        as_of_date: Optional[date] = None,
    ) -> List[FXVolatilityQuote]:
        return [
            self.get_fx_vol_quote(from_ccy, to_ccy, tenor, as_of_date)
            for tenor in tenors
        ]

    def get_fx_spot(
        self,
        from_ccy: str,
        to_ccy: str,
        as_of_date: Optional[date] = None,
    ) -> float:
        self._ensure_connection()
        quote = self._adapter.get_fx_quote(from_ccy, to_ccy)
        if quote is None:
            raise RuntimeError(f"No FX spot data for {from_ccy}/{to_ccy}")
        return quote.spot


@dataclass
class RefinitivDataAdapter(MarketDataSource):
    """
    Refinitiv (formerly Thomson Reuters) data adapter.

    To use this adapter, you need:
    1. Refinitiv Workspace or Eikon license
    2. Python Eikon/Refinitiv Data Library
    3. App key for API access

    Example:
        >>> # Requires Refinitiv credentials
        >>> adapter = RefinitivDataAdapter(app_key="your_app_key")
        >>> quote = adapter.get_fx_vol_quote("EUR", "USD", 1.0)

    Note:
        This is a template. Actual implementation requires Refinitiv API credentials.
    """

    app_key: str = ""
    username: str = ""
    password: str = ""
    use_desktop: bool = True
    timeout: int = 5000

    def __post_init__(self) -> None:
        self._config = RefinitivConfig(
            adapter_name="refinitiv",
            timeout_ms=self.timeout,
            app_key=self.app_key,
            username=self.username,
            password=self.password,
            use_desktop=self.use_desktop,
        )
        self._adapter = _RefinitivAdapter(self._config)
        self._connected = False

    def _ensure_connection(self) -> None:
        if not self._connected:
            if not self._adapter.connect():
                raise RuntimeError(
                    "Unable to connect to Refinitiv. Install the required SDK "
                    "and provide valid credentials."
                )
            self._connected = True

    def get_fx_vol_quote(
        self,
        from_ccy: str,
        to_ccy: str,
        expiry: float,
        as_of_date: Optional[date] = None,
    ) -> FXVolatilityQuote:
        self._ensure_connection()
        fx_quote = self._adapter.get_fx_quote(from_ccy, to_ccy)
        if fx_quote is None:
            raise RuntimeError(f"No Refinitiv FX quote for {from_ccy}/{to_ccy}")

        tenor_days = max(1, int(expiry * 365))
        expiry_date = (as_of_date or date.today()) + timedelta(days=tenor_days)
        underlying = f"{from_ccy}{to_ccy}=X"

        vol_quote = self._adapter.get_volatility_quote(
            underlying=underlying,
            strike=fx_quote.spot,
            expiry=expiry_date,
            option_type="call",
        )

        if vol_quote is None:
            raise RuntimeError(
                f"No Refinitiv volatility data for {underlying} at {expiry_date}"
            )

        return FXVolatilityQuote(
            expiry=expiry,
            atm_vol=vol_quote.volatility,
            rr_25d=0.0,
            bf_25d=0.0,
            forward=fx_quote.spot,
            domestic_rate=0.0,
            foreign_rate=0.0,
        )

    def get_fx_vol_surface(
        self,
        from_ccy: str,
        to_ccy: str,
        tenors: List[float],
        as_of_date: Optional[date] = None,
    ) -> List[FXVolatilityQuote]:
        return [
            self.get_fx_vol_quote(from_ccy, to_ccy, tenor, as_of_date)
            for tenor in tenors
        ]

    def get_fx_spot(
        self,
        from_ccy: str,
        to_ccy: str,
        as_of_date: Optional[date] = None,
    ) -> float:
        self._ensure_connection()
        quote = self._adapter.get_fx_quote(from_ccy, to_ccy)
        if quote is None:
            raise RuntimeError(f"No Refinitiv FX quote for {from_ccy}/{to_ccy}")
        return quote.spot


# Convenience function for getting market data
def get_market_data_source(source_type: str = "simulated", **kwargs) -> MarketDataSource:
    """
    Factory function to create market data source.

    Args:
        source_type: Type of data source ("simulated", "bloomberg", "refinitiv")
        **kwargs: Arguments passed to data source constructor

    Returns:
        MarketDataSource instance

    Example:
        >>> # Simulated data
        >>> source = get_market_data_source("simulated", base_vol=0.12)
        >>> # Bloomberg (requires credentials)
        >>> source = get_market_data_source("bloomberg", host="localhost", port=8194)
        >>> # Refinitiv (requires credentials)
        >>> source = get_market_data_source("refinitiv", app_key="your_key")
    """
    if source_type == "simulated":
        return SimulatedMarketData(**kwargs)
    elif source_type == "bloomberg":
        return BloombergDataAdapter(**kwargs)
    elif source_type == "refinitiv":
        return RefinitivDataAdapter(**kwargs)
    else:
        raise ValueError(
            f"Unknown source type: {source_type}. "
            f"Choose from: simulated, bloomberg, refinitiv"
        )


def create_market_data_feed(
    source_type: str,
    *,
    validator: Optional[DataValidator] = None,
    storage: Optional["DatabaseConnector"] = None,
    adapter_kwargs: Optional[Dict[str, Any]] = None,
) -> PollingMarketDataFeed:
    """
    Create a :class:`PollingMarketDataFeed` for the specified source type.

    Args:
        source_type: ``"simulated"``, ``"bloomberg"``, or ``"refinitiv"``
        validator: Optional :class:`DataValidator` instance
        storage: Optional database connector for persistence
        adapter_kwargs: Additional keyword arguments passed to the adapter config
    """
    adapter_kwargs = adapter_kwargs or {}

    if source_type == "simulated":
        config = SimulatedConfig(adapter_name="simulated", **adapter_kwargs)
        adapter = SimulatedAdapter(config)
    elif source_type == "bloomberg":
        config = BloombergConfig(adapter_name="bloomberg", **adapter_kwargs)
        adapter = _BloombergAdapter(config)
    elif source_type == "refinitiv":
        config = RefinitivConfig(adapter_name="refinitiv", **adapter_kwargs)
        adapter = _RefinitivAdapter(config)
    else:
        raise ValueError(
            f"Unknown source type: {source_type}. "
            "Supported values: simulated, bloomberg, refinitiv."
        )

    return PollingMarketDataFeed(adapter, validator=validator, storage=storage)


def create_default_validator(max_age_seconds: int = 5) -> DataValidator:
    """
    Build a default validator suitable for FX and rates data.

    The validator enforces presence of timestamps, non-negative spot/price,
    and flags stale observations.
    """
    return DataValidator(
        rules=[
            RequiredFieldRule(["timestamp"], severity=Severity.ERROR),
            RangeRule("spot", minimum=0.0, severity=Severity.ERROR),
            RangeRule("price", minimum=0.0, severity=Severity.WARNING),
            StalenessRule(
                timedelta(seconds=max_age_seconds),
                severity=Severity.WARNING,
            ),
        ]
    )


def get_market_data_feed(
    source_type: str,
    *,
    validator: Optional[DataValidator] = None,
    storage: Optional["DatabaseConnector"] = None,
    adapter_kwargs: Optional[Dict[str, Any]] = None,
) -> PollingMarketDataFeed:
    """Alias for ``create_market_data_feed`` to match legacy naming."""
    return create_market_data_feed(
        source_type,
        validator=validator,
        storage=storage,
        adapter_kwargs=adapter_kwargs,
    )
