"""
Market data source integration for real-time and historical data.

Supports Bloomberg, Refinitiv, and simulated data feeds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, List, Optional

import jax.numpy as jnp

from .fx import FXVolatilityQuote


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
    Bloomberg data adapter (placeholder for actual Bloomberg API integration).

    To use this adapter, you need:
    1. Bloomberg Terminal or Bloomberg Server API license
    2. Python Bloomberg API (blpapi package)
    3. Active Bloomberg session

    Example:
        >>> # Requires Bloomberg Terminal/API
        >>> adapter = BloombergDataAdapter(host="localhost", port=8194)
        >>> quote = adapter.get_fx_vol_quote("EUR", "USD", 1.0)

    Note:
        This is a template. Actual implementation requires Bloomberg API credentials.
    """

    host: str = "localhost"
    port: int = 8194
    timeout: int = 5000

    def __post_init__(self):
        """Initialize Bloomberg connection."""
        # In production, this would establish Bloomberg API connection
        # import blpapi
        # self.session = blpapi.Session(...)
        # self.session.start()
        pass

    def get_fx_vol_quote(
        self,
        from_ccy: str,
        to_ccy: str,
        expiry: float,
        as_of_date: Optional[date] = None,
    ) -> FXVolatilityQuote:
        """
        Fetch FX vol quote from Bloomberg.

        Bloomberg tickers for FX vol:
        - ATM: <ccy_pair> <tenor> ATM Vol
        - RR: <ccy_pair> <tenor> 25D RR
        - BF: <ccy_pair> <tenor> 25D BF

        Example tickers:
        - EURUSD Curncy 1Y ATM Vol
        - EURUSD Curncy 1Y 25D RR
        - EURUSD Curncy 1Y 25D BF
        """
        # Placeholder: In production, this would call Bloomberg API
        # pair = f"{from_ccy}{to_ccy}"
        # tenor_str = self._format_tenor(expiry)
        # atm_ticker = f"{pair} Curncy {tenor_str} ATM Vol"
        # rr_ticker = f"{pair} Curncy {tenor_str} 25D RR"
        # bf_ticker = f"{pair} Curncy {tenor_str} 25D BF"
        #
        # atm_vol = self._fetch_bloomberg_field(atm_ticker, "PX_LAST")
        # rr_25d = self._fetch_bloomberg_field(rr_ticker, "PX_LAST")
        # bf_25d = self._fetch_bloomberg_field(bf_ticker, "PX_LAST")
        #
        # return FXVolatilityQuote(...)

        raise NotImplementedError(
            "Bloomberg adapter requires blpapi package and Bloomberg Terminal/API access. "
            "Use SimulatedMarketData for testing."
        )

    def get_fx_vol_surface(
        self,
        from_ccy: str,
        to_ccy: str,
        tenors: List[float],
        as_of_date: Optional[date] = None,
    ) -> List[FXVolatilityQuote]:
        """Fetch FX vol surface from Bloomberg."""
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
        """Fetch FX spot from Bloomberg."""
        # pair = f"{from_ccy}{to_ccy}"
        # ticker = f"{pair} Curncy"
        # return self._fetch_bloomberg_field(ticker, "PX_LAST")
        raise NotImplementedError("Bloomberg adapter not implemented")

    @staticmethod
    def _format_tenor(expiry: float) -> str:
        """Convert expiry in years to Bloomberg tenor format (e.g., 1.0 -> "1Y")."""
        if expiry < 1.0:
            months = int(expiry * 12)
            return f"{months}M"
        else:
            years = int(expiry)
            return f"{years}Y"


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
    timeout: int = 5000

    def __post_init__(self):
        """Initialize Refinitiv connection."""
        # In production, this would establish Refinitiv API connection
        # import refinitiv.dataplatform as rdp
        # rdp.open_platform_session(self.app_key, ...)
        pass

    def get_fx_vol_quote(
        self,
        from_ccy: str,
        to_ccy: str,
        expiry: float,
        as_of_date: Optional[date] = None,
    ) -> FXVolatilityQuote:
        """
        Fetch FX vol quote from Refinitiv.

        Refinitiv RICs for FX vol:
        - ATM: <ccy_pair>=IMP<tenor>
        - RR: <ccy_pair>=RR<tenor>
        - BF: <ccy_pair>=BF<tenor>

        Example RICs:
        - EUR=IMP1Y (1Y ATM implied vol)
        - EUR=RR1Y25 (1Y 25-delta risk reversal)
        - EUR=BF1Y25 (1Y 25-delta butterfly)
        """
        # Placeholder: In production, this would call Refinitiv API
        raise NotImplementedError(
            "Refinitiv adapter requires refinitiv.dataplatform package and API credentials. "
            "Use SimulatedMarketData for testing."
        )

    def get_fx_vol_surface(
        self,
        from_ccy: str,
        to_ccy: str,
        tenors: List[float],
        as_of_date: Optional[date] = None,
    ) -> List[FXVolatilityQuote]:
        """Fetch FX vol surface from Refinitiv."""
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
        """Fetch FX spot from Refinitiv."""
        raise NotImplementedError("Refinitiv adapter not implemented")


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
