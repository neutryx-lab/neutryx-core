"""
CME Market Data adapter.

Provides connectivity to CME Group Market Data services for:
- Real-time and delayed futures and options data
- CME Globex platform data
- Interest rate derivatives (Eurodollar, Treasury futures)
- Equity indices (E-mini S&P, NASDAQ)
- FX futures and options
- Commodities (metals, energy, agriculture)
- Market depth and order book data

CME Group is the world's leading derivatives marketplace.
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
from ..storage.security_master import CorporateActionEvent

logger = logging.getLogger(__name__)


@dataclass
class CMEMarketDataConfig(AdapterConfig):
    """
    Configuration for CME Market Data adapter.

    Attributes:
    -----------
    adapter_name : str
        Adapter identifier
    api_endpoint : str
        CME API endpoint URL
    username : str
        CME credentials username
    password : str
        CME credentials password
    app_key : str
        Application key
    feed_type : str
        Feed type ("realtime", "delayed", "eod")
    protocol : str
        Protocol ("fix", "json", "websocket")
    enable_ssl : bool
        Enable SSL/TLS
    connection_timeout : int
        Connection timeout in seconds
    market_data_channel : str
        Market data channel (Globex, CME, CBOT, NYMEX, COMEX)
    """

    api_endpoint: str = "https://mdp3.cmegroup.com/v1"
    username: str = ""
    password: str = ""
    app_key: str = ""
    feed_type: str = "realtime"
    protocol: str = "fix"
    enable_ssl: bool = True
    connection_timeout: int = 30
    market_data_channel: str = "GLOBEX"


class CMEMarketDataAdapter(BaseMarketDataAdapter):
    """
    Market data adapter for CME Group Market Data Platform (MDP 3.0).

    Supports:
    ---------
    - Real-time market data from CME Globex
    - Interest rate derivatives (Eurodollar, Treasury futures & options)
    - Equity index futures (E-mini S&P 500, NASDAQ 100, Dow, Russell)
    - FX futures and options (Euro, Yen, Pound, etc.)
    - Commodity futures (metals, energy, agriculture)
    - Market depth (10-level order book)
    - Trade statistics and settlement prices
    - Open interest and volume data

    Markets Covered:
    ----------------
    - CME: Interest rates, equity indices, FX
    - CBOT: Grains, oilseeds, Treasuries
    - NYMEX: Energy (crude oil, natural gas)
    - COMEX: Precious metals (gold, silver, copper)
    """

    def __init__(self, config: CMEMarketDataConfig):
        """
        Initialize CME Market Data adapter.

        Parameters:
        -----------
        config : CMEMarketDataConfig
            Adapter configuration
        """
        super().__init__(config)
        self.cme_config = config
        self._session: Optional[Any] = None
        self._fix_session: Optional[Any] = None
        self._subscriptions: Dict[str, List[str]] = {}

        logger.info(f"Initialized CME Market Data adapter: {config.adapter_name}")

    def connect(self) -> bool:
        """
        Establish connection to CME Market Data Platform.

        Returns:
        --------
        bool
            True if connection successful, False otherwise
        """
        try:
            self._notify_connection_state_change(ConnectionState.CONNECTING)

            logger.info(
                f"Connecting to CME Market Data at {self.cme_config.api_endpoint}"
            )

            # Authenticate
            if not self.cme_config.username or not self.cme_config.password:
                raise ValueError("Username and password required for CME Market Data")

            # Establish session
            self._session = self._create_session()

            # Connect to appropriate protocol
            if self.cme_config.protocol == "fix":
                self._fix_session = self._create_fix_session()

            self._notify_connection_state_change(ConnectionState.CONNECTED)
            logger.info(f"Successfully connected to CME Market Data ({self.cme_config.market_data_channel})")

            return True

        except Exception as e:
            logger.error(f"Failed to connect to CME Market Data: {e}")
            self._notify_connection_state_change(ConnectionState.ERROR)
            return False

    def disconnect(self) -> bool:
        """
        Disconnect from CME Market Data Platform.

        Returns:
        --------
        bool
            True if disconnection successful, False otherwise
        """
        try:
            if self._fix_session:
                # Logout from FIX session
                self._fix_session = None

            if self._session:
                # Close session
                self._session = None

            self._notify_connection_state_change(ConnectionState.DISCONNECTED)
            logger.info("Disconnected from CME Market Data")

            return True

        except Exception as e:
            logger.error(f"Error disconnecting from CME Market Data: {e}")
            return False

    def is_connected(self) -> bool:
        """Check if adapter is connected."""
        return self.connection_state == ConnectionState.CONNECTED

    def get_equity_quote(
        self, ticker: str, exchange: Optional[str] = None
    ) -> Optional[EquityQuote]:
        """
        Get equity index futures quote from CME.

        Parameters:
        -----------
        ticker : str
            Futures contract symbol (e.g., "ESZ3" for E-mini S&P Dec 2023)
        exchange : str, optional
            Exchange code (CME, GLOBEX)

        Returns:
        --------
        EquityQuote or None
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to CME Market Data")

        cache_key = f"equity:{ticker}:{exchange}"
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        try:
            quote = self._fetch_equity_quote_from_cme(ticker, exchange)

            self._store_cache(cache_key, quote)
            return quote

        except Exception as e:
            logger.error(f"Error fetching equity futures quote for {ticker}: {e}")
            return None

    def get_bond_quote(
        self, identifier: str, id_type: str = "isin"
    ) -> Optional[BondQuote]:
        """
        Get Treasury futures quote from CME.

        Parameters:
        -----------
        identifier : str
            Futures contract symbol (e.g., "ZNU3" for 10-Year Note Sep 2023)
        id_type : str
            Identifier type

        Returns:
        --------
        BondQuote or None
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to CME Market Data")

        cache_key = f"bond:{id_type}:{identifier}"
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        try:
            quote = self._fetch_bond_quote_from_cme(identifier, id_type)

            self._store_cache(cache_key, quote)
            return quote

        except Exception as e:
            logger.error(f"Error fetching Treasury futures quote for {identifier}: {e}")
            return None

    def get_fx_quote(
        self, base_currency: str, quote_currency: str
    ) -> Optional[FXQuote]:
        """Get FX futures quote from CME."""
        if not self.is_connected():
            raise RuntimeError("Not connected to CME Market Data")

        cache_key = f"fx:{base_currency}{quote_currency}"
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        try:
            quote = self._fetch_fx_quote_from_cme(base_currency, quote_currency)

            self._store_cache(cache_key, quote)
            return quote

        except Exception as e:
            logger.error(
                f"Error fetching FX futures quote for {base_currency}/{quote_currency}: {e}"
            )
            return None

    def get_interest_rate(
        self, rate_type: str, currency: str, tenor: str
    ) -> Optional[InterestRateQuote]:
        """Get interest rate futures quote from CME."""
        if not self.is_connected():
            raise RuntimeError("Not connected to CME Market Data")

        cache_key = f"rate:{rate_type}:{currency}:{tenor}"
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        try:
            quote = self._fetch_interest_rate_from_cme(rate_type, currency, tenor)

            self._store_cache(cache_key, quote)
            return quote

        except Exception as e:
            logger.error(
                f"Error fetching IR futures for {rate_type} {currency} {tenor}: {e}"
            )
            return None

    def get_commodity_quote(
        self, commodity_code: str, exchange: Optional[str] = None
    ) -> Optional[CommodityQuote]:
        """Get commodity futures quote from CME."""
        if not self.is_connected():
            raise RuntimeError("Not connected to CME Market Data")

        cache_key = f"commodity:{commodity_code}:{exchange}"
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        try:
            quote = self._fetch_commodity_quote_from_cme(commodity_code, exchange)

            self._store_cache(cache_key, quote)
            return quote

        except Exception as e:
            logger.error(f"Error fetching commodity futures quote for {commodity_code}: {e}")
            return None

    def get_credit_spread(
        self, issuer: str, tenor: str, currency: str = "USD"
    ) -> Optional[CreditSpreadQuote]:
        """Get credit spread quote (not directly available from CME)."""
        logger.warning("Credit spreads not directly available from CME Market Data")
        return None

    def get_volatility_quote(
        self, underlying: str, strike: float, expiry: date, option_type: str = "call"
    ) -> Optional[VolatilityQuote]:
        """Get options volatility from CME."""
        if not self.is_connected():
            raise RuntimeError("Not connected to CME Market Data")

        cache_key = f"vol:{underlying}:{strike}:{expiry}:{option_type}"
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        try:
            quote = self._fetch_volatility_from_cme(
                underlying, strike, expiry, option_type
            )

            self._store_cache(cache_key, quote)
            return quote

        except Exception as e:
            logger.error(f"Error fetching options volatility for {underlying}: {e}")
            return None

    def get_yield_curve(
        self, curve_name: str, currency: str = "USD"
    ) -> Optional[YieldCurve]:
        """Get yield curve constructed from CME Treasury futures."""
        if not self.is_connected():
            raise RuntimeError("Not connected to CME Market Data")

        cache_key = f"yield_curve:{curve_name}:{currency}"
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        try:
            curve = self._fetch_yield_curve_from_cme(curve_name, currency)

            self._store_cache(cache_key, curve)
            return curve

        except Exception as e:
            logger.error(f"Error fetching yield curve {curve_name}: {e}")
            return None

    def get_corporate_actions(
        self, identifier: str, start_date: date, end_date: date
    ) -> List[CorporateActionEvent]:
        """Corporate actions not applicable for CME futures/options."""
        return []

    def _execute_request_internal(self, request: DataRequest) -> List[MarketDataPoint]:
        """Execute internal data request."""
        # Implementation would make actual API calls or FIX messages
        return []

    def get_market_depth(
        self, symbol: str, levels: int = 10
    ) -> Optional[Dict[str, Any]]:
        """
        Get market depth (order book) for a symbol.

        Parameters:
        -----------
        symbol : str
            Contract symbol
        levels : int
            Number of price levels (max 10)

        Returns:
        --------
        Dict with bids and asks or None
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to CME Market Data")

        try:
            depth = self._fetch_market_depth_from_cme(symbol, levels)
            return depth

        except Exception as e:
            logger.error(f"Error fetching market depth for {symbol}: {e}")
            return None

    def get_settlement_price(
        self, symbol: str, settlement_date: date
    ) -> Optional[float]:
        """
        Get official settlement price for a contract.

        Parameters:
        -----------
        symbol : str
            Contract symbol
        settlement_date : date
            Settlement date

        Returns:
        --------
        Settlement price or None
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to CME Market Data")

        try:
            price = self._fetch_settlement_price_from_cme(symbol, settlement_date)
            return price

        except Exception as e:
            logger.error(f"Error fetching settlement price for {symbol}: {e}")
            return None

    # ------------------------------------------------------------------
    # Private helper methods (simulated API calls)
    # ------------------------------------------------------------------

    def _create_session(self) -> Any:
        """Create authenticated session."""
        # In production, create actual HTTP session or FIX connection
        return {
            "authenticated": True,
            "username": self.cme_config.username,
            "channel": self.cme_config.market_data_channel,
        }

    def _create_fix_session(self) -> Any:
        """Create FIX protocol session."""
        # In production, establish FIX session
        return {"fix_connected": True, "protocol_version": "FIX.4.4"}

    def _fetch_equity_quote_from_cme(
        self, ticker: str, exchange: Optional[str]
    ) -> EquityQuote:
        """Fetch equity futures quote from CME (simulated)."""
        return EquityQuote(
            ticker=ticker,
            exchange=exchange or "GLOBEX",
            price=4550.25,
            bid=4550.00,
            ask=4550.50,
            volume=500000,
            timestamp=datetime.utcnow(),
            quality=DataQuality.REAL_TIME,
        )

    def _fetch_bond_quote_from_cme(
        self, identifier: str, id_type: str
    ) -> BondQuote:
        """Fetch Treasury futures quote from CME (simulated)."""
        return BondQuote(
            identifier=identifier,
            id_type=id_type,
            price=112.50,
            yield_value=4.15,
            accrued_interest=0.0,
            timestamp=datetime.utcnow(),
            quality=DataQuality.REAL_TIME,
        )

    def _fetch_fx_quote_from_cme(
        self, base_currency: str, quote_currency: str
    ) -> FXQuote:
        """Fetch FX futures quote from CME (simulated)."""
        return FXQuote(
            base_currency=base_currency,
            quote_currency=quote_currency,
            rate=1.0850,
            bid=1.0848,
            ask=1.0852,
            timestamp=datetime.utcnow(),
            quality=DataQuality.REAL_TIME,
        )

    def _fetch_interest_rate_from_cme(
        self, rate_type: str, currency: str, tenor: str
    ) -> InterestRateQuote:
        """Fetch IR futures quote from CME (simulated)."""
        # Convert futures price to rate (e.g., Eurodollar: rate = 100 - price)
        return InterestRateQuote(
            rate_type=rate_type,
            currency=currency,
            tenor=tenor,
            rate=0.0525,
            timestamp=datetime.utcnow(),
            quality=DataQuality.REAL_TIME,
        )

    def _fetch_commodity_quote_from_cme(
        self, commodity_code: str, exchange: Optional[str]
    ) -> CommodityQuote:
        """Fetch commodity futures quote from CME (simulated)."""
        return CommodityQuote(
            commodity_code=commodity_code,
            exchange=exchange or "NYMEX",
            price=75.50,
            timestamp=datetime.utcnow(),
            quality=DataQuality.REAL_TIME,
        )

    def _fetch_volatility_from_cme(
        self, underlying: str, strike: float, expiry: date, option_type: str
    ) -> VolatilityQuote:
        """Fetch options volatility from CME (simulated)."""
        return VolatilityQuote(
            underlying=underlying,
            strike=strike,
            expiry=expiry,
            option_type=option_type,
            implied_vol=0.18,
            timestamp=datetime.utcnow(),
            quality=DataQuality.REAL_TIME,
        )

    def _fetch_yield_curve_from_cme(
        self, curve_name: str, currency: str
    ) -> YieldCurve:
        """Fetch yield curve from CME Treasury futures (simulated)."""
        return YieldCurve(
            curve_name=curve_name,
            currency=currency,
            tenors=["2Y", "5Y", "10Y", "30Y"],
            rates=[0.045, 0.048, 0.051, 0.054],
            timestamp=datetime.utcnow(),
        )

    def _fetch_market_depth_from_cme(
        self, symbol: str, levels: int
    ) -> Dict[str, Any]:
        """Fetch market depth from CME (simulated)."""
        return {
            "symbol": symbol,
            "bids": [
                {"price": 4550.00, "quantity": 100},
                {"price": 4549.75, "quantity": 150},
                {"price": 4549.50, "quantity": 200},
            ],
            "asks": [
                {"price": 4550.25, "quantity": 120},
                {"price": 4550.50, "quantity": 180},
                {"price": 4550.75, "quantity": 150},
            ],
            "timestamp": datetime.utcnow(),
        }

    def _fetch_settlement_price_from_cme(
        self, symbol: str, settlement_date: date
    ) -> float:
        """Fetch settlement price from CME (simulated)."""
        return 4550.00
