"""
ICE Data Services market data adapter.

Provides connectivity to ICE Data Services for:
- Real-time and delayed market data
- Fixed income (bonds, credit)
- Derivatives (options, futures)
- Indices and reference data
- Corporate actions

ICE provides comprehensive market data across multiple asset classes
through their consolidated feed and API offerings.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Any
import logging

from .base import BaseMarketDataAdapter, AdapterConfig, ConnectionState
from ..data_models import (
    EquityQuote,
    BondQuote,
    FXQuote,
    InterestRateQuote,
    CommodityQuote,
    CreditSpreadQuote,
    VolatilityQuote,
    YieldCurve,
    DataRequest,
    MarketDataPoint,
    DataQuality,
)
from ..storage.security_master import CorporateActionEvent, CorporateActionType

logger = logging.getLogger(__name__)


@dataclass
class ICEDataServicesConfig(AdapterConfig):
    """
    Configuration for ICE Data Services adapter.

    Attributes:
    -----------
    adapter_name : str
        Adapter identifier
    api_endpoint : str
        ICE API endpoint URL
    api_key : str
        API authentication key
    api_secret : str
        API secret
    feed_type : str
        Feed type ("realtime", "delayed", "consolidated")
    data_format : str
        Data format ("json", "xml", "fix")
    enable_ssl : bool
        Enable SSL/TLS
    connection_timeout : int
        Connection timeout in seconds
    """

    api_endpoint: str = "https://api.ice.com/v1"
    api_key: str = ""
    api_secret: str = ""
    feed_type: str = "realtime"
    data_format: str = "json"
    enable_ssl: bool = True
    connection_timeout: int = 30


class ICEDataServicesAdapter(BaseMarketDataAdapter):
    """
    Market data adapter for ICE Data Services.

    Supports:
    ---------
    - Real-time market data feeds
    - Historical data requests
    - Fixed income pricing (bonds, credit)
    - Derivatives (options, futures, swaps)
    - Indices and benchmarks
    - Corporate actions and reference data
    - Order book depth (Level 2 data)

    Data Coverage:
    --------------
    - NYSE, NASDAQ, Euronext, TSX
    - ICE Futures (Energy, Agriculturals, Financials)
    - Fixed Income (US Treasuries, Corporate Bonds, Munis)
    - Credit Default Swaps
    - FX Spot and Forwards
    """

    def __init__(self, config: ICEDataServicesConfig):
        """
        Initialize ICE Data Services adapter.

        Parameters:
        -----------
        config : ICEDataServicesConfig
            Adapter configuration
        """
        super().__init__(config)
        self.ice_config = config
        self._session: Optional[Any] = None
        self._websocket: Optional[Any] = None
        self._subscriptions: Dict[str, List[str]] = {}

        logger.info(f"Initialized ICE Data Services adapter: {config.adapter_name}")

    def connect(self) -> bool:
        """
        Establish connection to ICE Data Services.

        Returns:
        --------
        bool
            True if connection successful, False otherwise
        """
        try:
            self._notify_connection_state_change(ConnectionState.CONNECTING)

            # In production, this would establish actual connection
            # For now, simulate connection
            logger.info(
                f"Connecting to ICE Data Services at {self.ice_config.api_endpoint}"
            )

            # Authenticate with API key/secret
            if not self.ice_config.api_key:
                raise ValueError("API key required for ICE Data Services")

            # Establish session (simulated)
            self._session = self._create_session()

            self._notify_connection_state_change(ConnectionState.CONNECTED)
            logger.info("Successfully connected to ICE Data Services")

            return True

        except Exception as e:
            logger.error(f"Failed to connect to ICE Data Services: {e}")
            self._notify_connection_state_change(ConnectionState.ERROR)
            return False

    def disconnect(self) -> bool:
        """
        Disconnect from ICE Data Services.

        Returns:
        --------
        bool
            True if disconnection successful, False otherwise
        """
        try:
            if self._websocket:
                # Close websocket connection
                self._websocket = None

            if self._session:
                # Close session
                self._session = None

            self._notify_connection_state_change(ConnectionState.DISCONNECTED)
            logger.info("Disconnected from ICE Data Services")

            return True

        except Exception as e:
            logger.error(f"Error disconnecting from ICE Data Services: {e}")
            return False

    def is_connected(self) -> bool:
        """Check if adapter is connected."""
        return self.connection_state == ConnectionState.CONNECTED

    def get_equity_quote(
        self, ticker: str, exchange: Optional[str] = None
    ) -> Optional[EquityQuote]:
        """
        Get equity quote from ICE Data Services.

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        exchange : str, optional
            Exchange code (NYSE, NASDAQ, etc.)

        Returns:
        --------
        EquityQuote or None
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to ICE Data Services")

        cache_key = f"equity:{ticker}:{exchange}"
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        try:
            # In production, make actual API call
            # For now, return simulated data
            quote = self._fetch_equity_quote_from_ice(ticker, exchange)

            self._store_cache(cache_key, quote)
            return quote

        except Exception as e:
            logger.error(f"Error fetching equity quote for {ticker}: {e}")
            return None

    def get_bond_quote(
        self, identifier: str, id_type: str = "isin"
    ) -> Optional[BondQuote]:
        """
        Get bond quote from ICE Data Services.

        Parameters:
        -----------
        identifier : str
            Bond identifier (ISIN, CUSIP, etc.)
        id_type : str
            Identifier type

        Returns:
        --------
        BondQuote or None
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to ICE Data Services")

        cache_key = f"bond:{id_type}:{identifier}"
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        try:
            quote = self._fetch_bond_quote_from_ice(identifier, id_type)

            self._store_cache(cache_key, quote)
            return quote

        except Exception as e:
            logger.error(f"Error fetching bond quote for {identifier}: {e}")
            return None

    def get_fx_quote(
        self, base_currency: str, quote_currency: str
    ) -> Optional[FXQuote]:
        """Get FX quote from ICE Data Services."""
        if not self.is_connected():
            raise RuntimeError("Not connected to ICE Data Services")

        cache_key = f"fx:{base_currency}{quote_currency}"
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        try:
            quote = self._fetch_fx_quote_from_ice(base_currency, quote_currency)

            self._store_cache(cache_key, quote)
            return quote

        except Exception as e:
            logger.error(
                f"Error fetching FX quote for {base_currency}/{quote_currency}: {e}"
            )
            return None

    def get_interest_rate(
        self, rate_type: str, currency: str, tenor: str
    ) -> Optional[InterestRateQuote]:
        """Get interest rate quote from ICE Data Services."""
        if not self.is_connected():
            raise RuntimeError("Not connected to ICE Data Services")

        cache_key = f"rate:{rate_type}:{currency}:{tenor}"
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        try:
            quote = self._fetch_interest_rate_from_ice(rate_type, currency, tenor)

            self._store_cache(cache_key, quote)
            return quote

        except Exception as e:
            logger.error(
                f"Error fetching rate for {rate_type} {currency} {tenor}: {e}"
            )
            return None

    def get_commodity_quote(
        self, commodity_code: str, exchange: Optional[str] = None
    ) -> Optional[CommodityQuote]:
        """Get commodity quote from ICE Data Services."""
        if not self.is_connected():
            raise RuntimeError("Not connected to ICE Data Services")

        cache_key = f"commodity:{commodity_code}:{exchange}"
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        try:
            quote = self._fetch_commodity_quote_from_ice(commodity_code, exchange)

            self._store_cache(cache_key, quote)
            return quote

        except Exception as e:
            logger.error(f"Error fetching commodity quote for {commodity_code}: {e}")
            return None

    def get_credit_spread(
        self, issuer: str, tenor: str, currency: str = "USD"
    ) -> Optional[CreditSpreadQuote]:
        """Get credit spread quote from ICE Data Services."""
        if not self.is_connected():
            raise RuntimeError("Not connected to ICE Data Services")

        cache_key = f"credit:{issuer}:{tenor}:{currency}"
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        try:
            quote = self._fetch_credit_spread_from_ice(issuer, tenor, currency)

            self._store_cache(cache_key, quote)
            return quote

        except Exception as e:
            logger.error(f"Error fetching credit spread for {issuer}: {e}")
            return None

    def get_volatility_quote(
        self, underlying: str, strike: float, expiry: date, option_type: str = "call"
    ) -> Optional[VolatilityQuote]:
        """Get volatility quote from ICE Data Services."""
        if not self.is_connected():
            raise RuntimeError("Not connected to ICE Data Services")

        cache_key = f"vol:{underlying}:{strike}:{expiry}:{option_type}"
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        try:
            quote = self._fetch_volatility_from_ice(
                underlying, strike, expiry, option_type
            )

            self._store_cache(cache_key, quote)
            return quote

        except Exception as e:
            logger.error(f"Error fetching volatility for {underlying}: {e}")
            return None

    def get_yield_curve(
        self, curve_name: str, currency: str = "USD"
    ) -> Optional[YieldCurve]:
        """Get yield curve from ICE Data Services."""
        if not self.is_connected():
            raise RuntimeError("Not connected to ICE Data Services")

        cache_key = f"yield_curve:{curve_name}:{currency}"
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        try:
            curve = self._fetch_yield_curve_from_ice(curve_name, currency)

            self._store_cache(cache_key, curve)
            return curve

        except Exception as e:
            logger.error(f"Error fetching yield curve {curve_name}: {e}")
            return None

    def get_corporate_actions(
        self, identifier: str, start_date: date, end_date: date
    ) -> List[CorporateActionEvent]:
        """Get corporate actions from ICE Data Services."""
        if not self.is_connected():
            raise RuntimeError("Not connected to ICE Data Services")

        try:
            actions = self._fetch_corporate_actions_from_ice(
                identifier, start_date, end_date
            )
            return actions

        except Exception as e:
            logger.error(f"Error fetching corporate actions for {identifier}: {e}")
            return []

    def _execute_request_internal(self, request: DataRequest) -> List[MarketDataPoint]:
        """Execute internal data request."""
        # Implementation would make actual API calls
        return []

    # ------------------------------------------------------------------
    # Private helper methods (simulated API calls)
    # ------------------------------------------------------------------

    def _create_session(self) -> Any:
        """Create authenticated session."""
        # In production, create actual HTTP session with authentication
        return {"authenticated": True, "api_key": self.ice_config.api_key}

    def _fetch_equity_quote_from_ice(
        self, ticker: str, exchange: Optional[str]
    ) -> EquityQuote:
        """Fetch equity quote from ICE API (simulated)."""
        return EquityQuote(
            ticker=ticker,
            exchange=exchange or "NYSE",
            price=150.25,
            bid=150.20,
            ask=150.30,
            volume=1000000,
            timestamp=datetime.utcnow(),
            quality=DataQuality.REAL_TIME,
        )

    def _fetch_bond_quote_from_ice(
        self, identifier: str, id_type: str
    ) -> BondQuote:
        """Fetch bond quote from ICE API (simulated)."""
        return BondQuote(
            identifier=identifier,
            id_type=id_type,
            price=98.75,
            yield_value=3.25,
            accrued_interest=0.5,
            timestamp=datetime.utcnow(),
            quality=DataQuality.REAL_TIME,
        )

    def _fetch_fx_quote_from_ice(
        self, base_currency: str, quote_currency: str
    ) -> FXQuote:
        """Fetch FX quote from ICE API (simulated)."""
        return FXQuote(
            base_currency=base_currency,
            quote_currency=quote_currency,
            rate=1.0850,
            bid=1.0848,
            ask=1.0852,
            timestamp=datetime.utcnow(),
            quality=DataQuality.REAL_TIME,
        )

    def _fetch_interest_rate_from_ice(
        self, rate_type: str, currency: str, tenor: str
    ) -> InterestRateQuote:
        """Fetch interest rate from ICE API (simulated)."""
        return InterestRateQuote(
            rate_type=rate_type,
            currency=currency,
            tenor=tenor,
            rate=0.0525,
            timestamp=datetime.utcnow(),
            quality=DataQuality.REAL_TIME,
        )

    def _fetch_commodity_quote_from_ice(
        self, commodity_code: str, exchange: Optional[str]
    ) -> CommodityQuote:
        """Fetch commodity quote from ICE API (simulated)."""
        return CommodityQuote(
            commodity_code=commodity_code,
            exchange=exchange or "ICE",
            price=75.50,
            timestamp=datetime.utcnow(),
            quality=DataQuality.REAL_TIME,
        )

    def _fetch_credit_spread_from_ice(
        self, issuer: str, tenor: str, currency: str
    ) -> CreditSpreadQuote:
        """Fetch credit spread from ICE API (simulated)."""
        return CreditSpreadQuote(
            issuer=issuer,
            tenor=tenor,
            currency=currency,
            spread=150.0,
            timestamp=datetime.utcnow(),
            quality=DataQuality.REAL_TIME,
        )

    def _fetch_volatility_from_ice(
        self, underlying: str, strike: float, expiry: date, option_type: str
    ) -> VolatilityQuote:
        """Fetch volatility from ICE API (simulated)."""
        return VolatilityQuote(
            underlying=underlying,
            strike=strike,
            expiry=expiry,
            option_type=option_type,
            implied_vol=0.25,
            timestamp=datetime.utcnow(),
            quality=DataQuality.REAL_TIME,
        )

    def _fetch_yield_curve_from_ice(
        self, curve_name: str, currency: str
    ) -> YieldCurve:
        """Fetch yield curve from ICE API (simulated)."""
        return YieldCurve(
            curve_name=curve_name,
            currency=currency,
            tenors=["1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y"],
            rates=[0.05, 0.051, 0.052, 0.053, 0.054, 0.055, 0.056],
            timestamp=datetime.utcnow(),
        )

    def _fetch_corporate_actions_from_ice(
        self, identifier: str, start_date: date, end_date: date
    ) -> List[CorporateActionEvent]:
        """Fetch corporate actions from ICE API (simulated)."""
        return [
            CorporateActionEvent(
                action_type=CorporateActionType.DIVIDEND,
                effective_date=date.today(),
                description="Quarterly dividend",
                details={"amount": 0.50, "currency": "USD"},
            )
        ]
