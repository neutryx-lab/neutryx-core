"""
Bloomberg Terminal/API data adapter.

Provides comprehensive Bloomberg integration for equities, bonds, FX,
commodities, interest rates, and credit data.

Requires: blpapi (Bloomberg Python API)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, List, Optional, Any
import logging

from .base import BaseMarketDataAdapter, AdapterConfig, ConnectionState
from .corporate_actions import (
    BloombergCorporateActionParser,
    normalize_events,
)
from ..data_models import (
    EquityQuote,
    BondQuote,
    FXQuote,
    InterestRateQuote,
    CommodityQuote,
    CreditSpreadQuote,
    VolatilityQuote,
    YieldCurve,
    DataQuality,
    DataRequest,
    MarketDataPoint,
)
from ..storage.security_master import CorporateActionEvent


logger = logging.getLogger(__name__)


@dataclass
class BloombergConfig(AdapterConfig):
    """
    Bloomberg-specific configuration.

    Attributes:
        host: Bloomberg API host
        port: Bloomberg API port
        application_name: Application name for Bloomberg
        identity: Bloomberg identity (for authorization)
        use_enterprise: Use Bloomberg Enterprise service
    """
    host: str = "localhost"
    port: int = 8194
    application_name: str = "Neutryx"
    identity: Optional[str] = None
    use_enterprise: bool = False


class BloombergAdapter(BaseMarketDataAdapter):
    """
    Bloomberg Terminal/API adapter.

    Connects to Bloomberg Terminal or Bloomberg Server API (SAPI/BPIPE)
    for real-time and historical market data.

    Bloomberg Field Reference:
    - Last Price: PX_LAST
    - Bid: PX_BID
    - Ask: PX_ASK
    - Volume: PX_VOLUME
    - Yield: YLD_YTM_MID
    - Credit Spread: Z_SPRD_MID

    Example:
        >>> config = BloombergConfig(
        ...     adapter_name="bloomberg",
        ...     host="localhost",
        ...     port=8194
        ... )
        >>> adapter = BloombergAdapter(config)
        >>> adapter.connect()
        >>> quote = adapter.get_equity_quote("AAPL US Equity")
    """

    def __init__(self, config: BloombergConfig):
        """
        Initialize Bloomberg adapter.

        Args:
            config: Bloomberg configuration
        """
        super().__init__(config)
        self.config: BloombergConfig = config
        self._session = None
        self._service = None

    def connect(self) -> bool:
        """
        Connect to Bloomberg API.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Import Bloomberg API
            try:
                import blpapi
            except ImportError:
                logger.error(
                    "Bloomberg API (blpapi) not installed. "
                    "Install with: pip install blpapi"
                )
                return False

            self._notify_connection_state_change(ConnectionState.CONNECTING)

            # Create session options
            session_options = blpapi.SessionOptions()
            session_options.setServerHost(self.config.host)
            session_options.setServerPort(self.config.port)

            if self.config.application_name:
                session_options.setApplicationIdentityKey(
                    blpapi.Name("applicationName"), self.config.application_name
                )

            # Create and start session
            self._session = blpapi.Session(session_options)

            if not self._session.start():
                logger.error("Failed to start Bloomberg session")
                self._notify_connection_state_change(ConnectionState.ERROR)
                return False

            # Open service
            service_name = (
                "//blp/mktdata" if not self.config.use_enterprise else "//blp/mktvwap"
            )

            if not self._session.openService(service_name):
                logger.error(f"Failed to open Bloomberg service: {service_name}")
                self._notify_connection_state_change(ConnectionState.ERROR)
                return False

            self._service = self._session.getService(service_name)
            self._notify_connection_state_change(ConnectionState.CONNECTED)

            logger.info("Successfully connected to Bloomberg API")
            return True

        except Exception as e:
            logger.error(f"Error connecting to Bloomberg: {e}")
            self._notify_connection_state_change(ConnectionState.ERROR)
            return False

    def disconnect(self) -> bool:
        """
        Disconnect from Bloomberg API.

        Returns:
            True if disconnection successful, False otherwise
        """
        try:
            if self._session:
                self._session.stop()
                self._session = None
                self._service = None

            self._notify_connection_state_change(ConnectionState.DISCONNECTED)
            logger.info("Disconnected from Bloomberg API")
            return True

        except Exception as e:
            logger.error(f"Error disconnecting from Bloomberg: {e}")
            return False

    def is_connected(self) -> bool:
        """
        Check if connected to Bloomberg.

        Returns:
            True if connected, False otherwise
        """
        return self.connection_state == ConnectionState.CONNECTED and self._session is not None

    def get_equity_quote(
        self, ticker: str, exchange: Optional[str] = None
    ) -> Optional[EquityQuote]:
        """
        Get equity quote from Bloomberg.

        Args:
            ticker: Bloomberg ticker (e.g., "AAPL US Equity")
            exchange: Exchange code (optional, included in ticker)

        Returns:
            EquityQuote or None if not available
        """
        cache_key = f"equity:{ticker}"
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        try:
            # Bloomberg ticker format: <ticker> <country> <security_type>
            # Example: AAPL US Equity
            fields = [
                "PX_LAST",
                "PX_BID",
                "PX_ASK",
                "PX_VOLUME",
                "PX_OPEN",
                "PX_HIGH",
                "PX_LOW",
                "PX_PREVIOUS_CLOSE",
                "CRNCY",
            ]

            data = self._request_reference_data(ticker, fields)

            if not data:
                return None

            quote = EquityQuote(
                timestamp=datetime.utcnow(),
                source=self.config.adapter_name,
                quality=DataQuality.REALTIME,
                ticker=ticker,
                exchange=exchange or "",
                price=data.get("PX_LAST", 0.0),
                bid=data.get("PX_BID", 0.0),
                ask=data.get("PX_ASK", 0.0),
                volume=data.get("PX_VOLUME", 0.0),
                open_price=data.get("PX_OPEN", 0.0),
                high=data.get("PX_HIGH", 0.0),
                low=data.get("PX_LOW", 0.0),
                close=data.get("PX_PREVIOUS_CLOSE", 0.0),
                currency=data.get("CRNCY", "USD"),
            )

            self._store_cache(cache_key, quote)
            return quote

        except Exception as e:
            logger.error(f"Error getting equity quote for {ticker}: {e}")
            return None

    def get_bond_quote(
        self, identifier: str, id_type: str = "isin"
    ) -> Optional[BondQuote]:
        """
        Get bond quote from Bloomberg.

        Args:
            identifier: Bond identifier (ISIN or Bloomberg ticker)
            id_type: Identifier type ("isin", "cusip", or "ticker")

        Returns:
            BondQuote or None if not available
        """
        cache_key = f"bond:{identifier}"
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        try:
            # Bloomberg bond ticker format: <identifier> <security_type>
            # Example: US912828ZG19 Govt or ISIN
            if id_type == "ticker":
                ticker = identifier
            else:
                ticker = f"{identifier} Corp" if id_type == "cusip" else identifier

            fields = [
                "PX_LAST",
                "YLD_YTM_MID",
                "INT_ACC",
                "Z_SPRD_MID",
                "DUR_ADJ_MID",
                "CONVEXITY_MID",
                "MATURITY",
                "CPN",
                "CRNCY",
            ]

            data = self._request_reference_data(ticker, fields)

            if not data:
                return None

            maturity_date = None
            if "MATURITY" in data:
                maturity_date = self._parse_bloomberg_date(data["MATURITY"])

            quote = BondQuote(
                timestamp=datetime.utcnow(),
                source=self.config.adapter_name,
                quality=DataQuality.REALTIME,
                isin=identifier if id_type == "isin" else "",
                cusip=identifier if id_type == "cusip" else "",
                price=data.get("PX_LAST", 100.0),
                yield_to_maturity=data.get("YLD_YTM_MID", 0.0) / 100.0,  # Convert to decimal
                accrued_interest=data.get("INT_ACC", 0.0),
                spread=data.get("Z_SPRD_MID", 0.0),
                duration=data.get("DUR_ADJ_MID", 0.0),
                convexity=data.get("CONVEXITY_MID", 0.0),
                maturity_date=maturity_date,
                coupon_rate=data.get("CPN", 0.0) / 100.0,  # Convert to decimal
                currency=data.get("CRNCY", "USD"),
            )

            self._store_cache(cache_key, quote)
            return quote

        except Exception as e:
            logger.error(f"Error getting bond quote for {identifier}: {e}")
            return None

    def get_fx_quote(
        self, base_currency: str, quote_currency: str
    ) -> Optional[FXQuote]:
        """
        Get FX quote from Bloomberg.

        Args:
            base_currency: Base currency (e.g., "EUR")
            quote_currency: Quote currency (e.g., "USD")

        Returns:
            FXQuote or None if not available
        """
        cache_key = f"fx:{base_currency}{quote_currency}"
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        try:
            # Bloomberg FX ticker format: <base><quote> Curncy
            # Example: EURUSD Curncy
            ticker = f"{base_currency}{quote_currency} Curncy"

            fields = ["PX_LAST", "PX_BID", "PX_ASK"]

            data = self._request_reference_data(ticker, fields)

            if not data:
                return None

            quote = FXQuote(
                timestamp=datetime.utcnow(),
                source=self.config.adapter_name,
                quality=DataQuality.REALTIME,
                currency_pair=f"{base_currency}/{quote_currency}",
                base_currency=base_currency,
                quote_currency=quote_currency,
                spot=data.get("PX_LAST", 0.0),
                bid=data.get("PX_BID", 0.0),
                ask=data.get("PX_ASK", 0.0),
            )

            self._store_cache(cache_key, quote)
            return quote

        except Exception as e:
            logger.error(f"Error getting FX quote for {base_currency}{quote_currency}: {e}")
            return None

    def get_interest_rate(
        self, rate_type: str, currency: str, tenor: str
    ) -> Optional[InterestRateQuote]:
        """
        Get interest rate from Bloomberg.

        Args:
            rate_type: Rate type (LIBOR, SOFR, EURIBOR, etc.)
            currency: Currency code
            tenor: Tenor (e.g., "3M", "6M", "1Y")

        Returns:
            InterestRateQuote or None if not available
        """
        cache_key = f"rate:{rate_type}:{currency}:{tenor}"
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        try:
            # Bloomberg rate ticker format varies by rate type
            # Examples:
            # - LIBOR: US0003M Index (USD 3M LIBOR)
            # - SOFR: SOFRRATE Index
            # - EURIBOR: EUR003M Index (EUR 3M EURIBOR)

            ticker = self._format_rate_ticker(rate_type, currency, tenor)
            fields = ["PX_LAST"]

            data = self._request_reference_data(ticker, fields)

            if not data:
                return None

            quote = InterestRateQuote(
                timestamp=datetime.utcnow(),
                source=self.config.adapter_name,
                quality=DataQuality.REALTIME,
                rate_type=rate_type,
                currency=currency,
                tenor=tenor,
                rate=data.get("PX_LAST", 0.0) / 100.0,  # Convert to decimal
                curve_name=f"{rate_type}_{currency}",
            )

            self._store_cache(cache_key, quote)
            return quote

        except Exception as e:
            logger.error(f"Error getting interest rate for {rate_type} {currency} {tenor}: {e}")
            return None

    def get_commodity_quote(
        self, commodity_code: str, exchange: Optional[str] = None
    ) -> Optional[CommodityQuote]:
        """
        Get commodity quote from Bloomberg.

        Args:
            commodity_code: Bloomberg commodity code (e.g., "CL1 Comdty" for crude oil)
            exchange: Exchange code (optional)

        Returns:
            CommodityQuote or None if not available
        """
        cache_key = f"commodity:{commodity_code}"
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        try:
            # Bloomberg commodity ticker format: <code> Comdty
            # Examples:
            # - CL1 Comdty (WTI Crude Oil front month)
            # - GC1 Comdty (Gold front month)
            # - SI1 Comdty (Silver front month)

            ticker = commodity_code if " " in commodity_code else f"{commodity_code} Comdty"

            fields = ["PX_LAST", "CRNCY", "FUT_CUR_GEN_TICKER"]

            data = self._request_reference_data(ticker, fields)

            if not data:
                return None

            quote = CommodityQuote(
                timestamp=datetime.utcnow(),
                source=self.config.adapter_name,
                quality=DataQuality.REALTIME,
                commodity_code=commodity_code,
                commodity_name=commodity_code,
                price=data.get("PX_LAST", 0.0),
                unit="",
                contract_month=data.get("FUT_CUR_GEN_TICKER", ""),
                exchange=exchange or "",
                currency=data.get("CRNCY", "USD"),
            )

            self._store_cache(cache_key, quote)
            return quote

        except Exception as e:
            logger.error(f"Error getting commodity quote for {commodity_code}: {e}")
            return None

    def get_credit_spread(
        self, issuer: str, tenor: str, currency: str = "USD"
    ) -> Optional[CreditSpreadQuote]:
        """
        Get credit spread from Bloomberg.

        Args:
            issuer: Issuer identifier or Bloomberg ticker
            tenor: Tenor
            currency: Currency code

        Returns:
            CreditSpreadQuote or None if not available
        """
        cache_key = f"credit:{issuer}:{tenor}"
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        try:
            # Bloomberg credit ticker format varies
            # Example: T 4.5 05/15/38 Corp (Treasury bond)
            # CDS: <issuer> <currency> <tenor> CDS (e.g., "MSFT US 5Y CDS")

            ticker = issuer if " " in issuer else f"{issuer} Corp"

            fields = ["Z_SPRD_MID", "RTG_MOODY", "INDUSTRY_SECTOR", "COUNTRY_ISO"]

            data = self._request_reference_data(ticker, fields)

            if not data:
                return None

            quote = CreditSpreadQuote(
                timestamp=datetime.utcnow(),
                source=self.config.adapter_name,
                quality=DataQuality.REALTIME,
                issuer=issuer,
                ticker=ticker,
                spread=data.get("Z_SPRD_MID", 0.0),
                rating=data.get("RTG_MOODY", ""),
                sector=data.get("INDUSTRY_SECTOR", ""),
                region=data.get("COUNTRY_ISO", ""),
                currency=currency,
                tenor=tenor,
            )

            self._store_cache(cache_key, quote)
            return quote

        except Exception as e:
            logger.error(f"Error getting credit spread for {issuer}: {e}")
            return None

    def get_volatility_quote(
        self, underlying: str, strike: float, expiry: date, option_type: str = "call"
    ) -> Optional[VolatilityQuote]:
        """
        Get volatility quote from Bloomberg.

        Args:
            underlying: Underlying asset ticker
            strike: Strike price
            expiry: Expiry date
            option_type: "call" or "put"

        Returns:
            VolatilityQuote or None if not available
        """
        try:
            # Bloomberg option ticker format: <underlying> <expiry> <type><strike> <security_type>
            # Example: SPX 12/18/24 C5000 Index

            expiry_str = expiry.strftime("%m/%d/%y")
            type_code = "C" if option_type.lower() == "call" else "P"
            ticker = f"{underlying} {expiry_str} {type_code}{int(strike)} Index"

            fields = ["IVOL_MID"]

            data = self._request_reference_data(ticker, fields)

            if not data:
                return None

            quote = VolatilityQuote(
                timestamp=datetime.utcnow(),
                source=self.config.adapter_name,
                quality=DataQuality.REALTIME,
                underlying=underlying,
                strike=strike,
                expiry=expiry,
                volatility=data.get("IVOL_MID", 0.0) / 100.0,  # Convert to decimal
                option_type=option_type,
            )

            return quote

        except Exception as e:
            logger.error(f"Error getting volatility quote for {underlying}: {e}")
            return None

    def get_yield_curve(
        self, curve_name: str, currency: str = "USD"
    ) -> Optional[YieldCurve]:
        """
        Get yield curve from Bloomberg.

        Args:
            curve_name: Curve identifier (e.g., "US Treasury")
            currency: Currency code

        Returns:
            YieldCurve or None if not available
        """
        try:
            # Bloomberg curve tickers vary by curve type
            # Example Treasury tickers: GT2, GT5, GT10, GT30 (2Y, 5Y, 10Y, 30Y)

            tenors_map = {
                "GT2 Govt": 2.0,
                "GT5 Govt": 5.0,
                "GT10 Govt": 10.0,
                "GT30 Govt": 30.0,
            }

            tenors = []
            rates = []

            for ticker, tenor in tenors_map.items():
                fields = ["PX_LAST"]
                data = self._request_reference_data(ticker, fields)

                if data and "PX_LAST" in data:
                    tenors.append(tenor)
                    rates.append(data["PX_LAST"] / 100.0)  # Convert to decimal

            if not tenors:
                return None

            curve = YieldCurve(
                timestamp=datetime.utcnow(),
                source=self.config.adapter_name,
                quality=DataQuality.REALTIME,
                curve_name=curve_name,
                currency=currency,
                tenors=tenors,
                rates=rates,
                curve_type="zero",
                interpolation_method="linear",
            )

            return curve

        except Exception as e:
            logger.error(f"Error getting yield curve {curve_name}: {e}")
            return None

    def get_corporate_actions(
        self, identifier: str, start_date: date, end_date: date
    ) -> List[CorporateActionEvent]:
        """Retrieve corporate action events using Bloomberg payload parser."""

        raw_events = self._retrieve_corporate_actions(identifier, start_date, end_date)
        parser = BloombergCorporateActionParser()
        return normalize_events(raw_events, parser)

    def _retrieve_corporate_actions(
        self, identifier: str, start_date: date, end_date: date
    ) -> List[Dict[str, Any]]:
        """Placeholder for Bloomberg corporate action retrieval logic."""

        self._increment_stat("requests")
        return []

    def _execute_request_internal(self, request: DataRequest) -> List[MarketDataPoint]:
        """
        Execute internal request.

        Args:
            request: Data request

        Returns:
            List of market data points
        """
        # Implementation depends on request type
        # This is a placeholder for the generic request handler
        return []

    def _request_reference_data(
        self, security: str, fields: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Request reference data from Bloomberg.

        Args:
            security: Security identifier
            fields: List of fields to request

        Returns:
            Dictionary of field values or None if request fails
        """
        if not self.is_connected():
            logger.error("Not connected to Bloomberg")
            return None

        try:
            import blpapi

            # Create request
            request = self._service.createRequest("ReferenceDataRequest")
            request.append("securities", security)

            for field in fields:
                request.append("fields", field)

            # Send request
            self._session.sendRequest(request)

            # Process response
            while True:
                event = self._session.nextEvent(self.config.timeout_ms)

                if event.eventType() == blpapi.Event.RESPONSE or event.eventType() == blpapi.Event.PARTIAL_RESPONSE:
                    for msg in event:
                        if msg.hasElement("securityData"):
                            security_data = msg.getElement("securityData")

                            if security_data.hasElement("fieldData"):
                                field_data = security_data.getElement("fieldData")

                                result = {}
                                for field in fields:
                                    if field_data.hasElement(field):
                                        result[field] = field_data.getElementValue(field)

                                return result

                if event.eventType() == blpapi.Event.RESPONSE:
                    break

            return None

        except Exception as e:
            logger.error(f"Error requesting reference data: {e}")
            return None

    @staticmethod
    def _format_rate_ticker(rate_type: str, currency: str, tenor: str) -> str:
        """
        Format rate ticker for Bloomberg.

        Args:
            rate_type: Rate type
            currency: Currency
            tenor: Tenor

        Returns:
            Bloomberg ticker
        """
        rate_type_upper = rate_type.upper()

        if rate_type_upper == "LIBOR":
            # LIBOR format: <currency><tenor> Index
            # Example: US0003M Index (USD 3M LIBOR)
            tenor_map = {"1M": "0001M", "3M": "0003M", "6M": "0006M", "12M": "0012M"}
            tenor_code = tenor_map.get(tenor, tenor)
            return f"{currency}{tenor_code} Index"

        elif rate_type_upper == "SOFR":
            return "SOFRRATE Index"

        elif rate_type_upper == "EURIBOR":
            # EURIBOR format: EUR<tenor> Index
            # Example: EUR003M Index (EUR 3M EURIBOR)
            tenor_map = {"1M": "001M", "3M": "003M", "6M": "006M", "12M": "012M"}
            tenor_code = tenor_map.get(tenor, tenor)
            return f"EUR{tenor_code} Index"

        else:
            return f"{rate_type}_{currency}_{tenor}"

    @staticmethod
    def _parse_bloomberg_date(date_value: Any) -> Optional[date]:
        """
        Parse Bloomberg date value.

        Args:
            date_value: Bloomberg date value

        Returns:
            Python date object or None
        """
        try:
            if isinstance(date_value, date):
                return date_value
            elif isinstance(date_value, datetime):
                return date_value.date()
            elif isinstance(date_value, str):
                return datetime.strptime(date_value, "%Y-%m-%d").date()
            else:
                return None
        except Exception:
            return None
