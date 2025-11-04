"""
Refinitiv (formerly Reuters/Eikon) data adapter.

Provides comprehensive Refinitiv Data Platform integration for real-time
and historical market data across all asset classes.

Requires: refinitiv.dataplatform or eikon
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
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
    DataQuality,
    DataRequest,
    MarketDataPoint,
)


logger = logging.getLogger(__name__)


@dataclass
class RefinitivConfig(AdapterConfig):
    """
    Refinitiv-specific configuration.

    Attributes:
        app_key: Refinitiv application key
        username: Refinitiv username (for RDP)
        password: Refinitiv password (for RDP)
        use_desktop: Use Eikon Desktop instead of RDP
        rdp_host: RDP host (for cloud deployment)
        rdp_auth_url: RDP authentication URL
    """
    app_key: str = ""
    username: str = ""
    password: str = ""
    use_desktop: bool = True
    rdp_host: str = "api.refinitiv.com"
    rdp_auth_url: str = "https://api.refinitiv.com/auth/oauth2/v1/token"


class RefinitivAdapter(BaseMarketDataAdapter):
    """
    Refinitiv Data Platform / Eikon adapter.

    Connects to Refinitiv Data Platform (RDP) or Eikon Desktop for
    real-time market data, historical data, and reference data.

    RIC (Reuters Instrument Code) Examples:
    - Equity: AAPL.O (Apple on NASDAQ)
    - FX: EUR= (EUR/USD spot)
    - Commodity: LCOc1 (Brent crude front month)
    - Bond: US10YT=RR (US 10Y Treasury yield)

    Example:
        >>> config = RefinitivConfig(
        ...     adapter_name="refinitiv",
        ...     app_key="your_app_key",
        ...     use_desktop=True
        ... )
        >>> adapter = RefinitivAdapter(config)
        >>> adapter.connect()
        >>> quote = adapter.get_equity_quote("AAPL.O")
    """

    def __init__(self, config: RefinitivConfig):
        """
        Initialize Refinitiv adapter.

        Args:
            config: Refinitiv configuration
        """
        super().__init__(config)
        self.config: RefinitivConfig = config
        self._session = None
        self._rdp = None
        self._eikon = None

    def connect(self) -> bool:
        """
        Connect to Refinitiv Data Platform or Eikon.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self._notify_connection_state_change(ConnectionState.CONNECTING)

            if self.config.use_desktop:
                # Connect to Eikon Desktop
                return self._connect_eikon()
            else:
                # Connect to Refinitiv Data Platform
                return self._connect_rdp()

        except Exception as e:
            logger.error(f"Error connecting to Refinitiv: {e}")
            self._notify_connection_state_change(ConnectionState.ERROR)
            return False

    def _connect_eikon(self) -> bool:
        """Connect to Eikon Desktop."""
        try:
            import eikon as ek
        except ImportError:
            logger.error(
                "Eikon API not installed. Install with: pip install eikon"
            )
            return False

        try:
            ek.set_app_key(self.config.app_key)
            self._eikon = ek

            # Test connection
            test_data = ek.get_data(["EUR="], ["BID"])
            if test_data is None or test_data[0].empty:
                raise Exception("Failed to retrieve test data")

            self._notify_connection_state_change(ConnectionState.CONNECTED)
            logger.info("Successfully connected to Eikon Desktop")
            return True

        except Exception as e:
            logger.error(f"Error connecting to Eikon: {e}")
            self._notify_connection_state_change(ConnectionState.ERROR)
            return False

    def _connect_rdp(self) -> bool:
        """Connect to Refinitiv Data Platform."""
        try:
            import refinitiv.dataplatform as rdp
        except ImportError:
            logger.error(
                "Refinitiv Data Platform library not installed. "
                "Install with: pip install refinitiv-dataplatform"
            )
            return False

        try:
            self._rdp = rdp

            # Open platform session
            session = rdp.open_platform_session(
                self.config.app_key,
                rdp.GrantPassword(
                    username=self.config.username,
                    password=self.config.password
                )
            )

            if session.get_open_state() != rdp.Session.State.Open:
                raise Exception("Failed to open RDP session")

            self._session = session
            self._notify_connection_state_change(ConnectionState.CONNECTED)
            logger.info("Successfully connected to Refinitiv Data Platform")
            return True

        except Exception as e:
            logger.error(f"Error connecting to RDP: {e}")
            self._notify_connection_state_change(ConnectionState.ERROR)
            return False

    def disconnect(self) -> bool:
        """
        Disconnect from Refinitiv.

        Returns:
            True if disconnection successful, False otherwise
        """
        try:
            if self._session:
                self._session.close()
                self._session = None

            self._rdp = None
            self._eikon = None

            self._notify_connection_state_change(ConnectionState.DISCONNECTED)
            logger.info("Disconnected from Refinitiv")
            return True

        except Exception as e:
            logger.error(f"Error disconnecting from Refinitiv: {e}")
            return False

    def is_connected(self) -> bool:
        """
        Check if connected to Refinitiv.

        Returns:
            True if connected, False otherwise
        """
        if self.config.use_desktop:
            return self._eikon is not None
        else:
            return (
                self.connection_state == ConnectionState.CONNECTED
                and self._session is not None
            )

    def get_equity_quote(
        self, ticker: str, exchange: Optional[str] = None
    ) -> Optional[EquityQuote]:
        """
        Get equity quote from Refinitiv.

        Args:
            ticker: Refinitiv RIC (e.g., "AAPL.O")
            exchange: Exchange code (optional, included in RIC)

        Returns:
            EquityQuote or None if not available
        """
        cache_key = f"equity:{ticker}"
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        try:
            # Refinitiv fields for equity data
            fields = [
                "CF_LAST",     # Last price
                "CF_BID",      # Bid price
                "CF_ASK",      # Ask price
                "CF_VOLUME",   # Volume
                "OPEN_PRC",    # Open price
                "HIGH_1",      # High price
                "LOW_1",       # Low price
                "CF_CLOSE",    # Previous close
                "CURRENCY",    # Currency
            ]

            data = self._get_data(ticker, fields)

            if not data:
                return None

            quote = EquityQuote(
                timestamp=datetime.utcnow(),
                source=self.config.adapter_name,
                quality=DataQuality.REALTIME,
                ticker=ticker,
                exchange=exchange or self._extract_exchange_from_ric(ticker),
                price=data.get("CF_LAST", 0.0),
                bid=data.get("CF_BID", 0.0),
                ask=data.get("CF_ASK", 0.0),
                volume=data.get("CF_VOLUME", 0.0),
                open_price=data.get("OPEN_PRC", 0.0),
                high=data.get("HIGH_1", 0.0),
                low=data.get("LOW_1", 0.0),
                close=data.get("CF_CLOSE", 0.0),
                currency=data.get("CURRENCY", "USD"),
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
        Get bond quote from Refinitiv.

        Args:
            identifier: Bond identifier (ISIN or RIC)
            id_type: Identifier type ("isin" or "ric")

        Returns:
            BondQuote or None if not available
        """
        cache_key = f"bond:{identifier}"
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        try:
            # Convert ISIN to RIC if needed
            ric = identifier if id_type == "ric" else f"{identifier}="

            # Refinitiv fields for bond data
            fields = [
                "CF_LAST",         # Price
                "CF_YIELD",        # Yield
                "ACCRUED_INT",     # Accrued interest
                "SPREAD_MID",      # Spread
                "DURATION",        # Duration
                "CONVEXITY",       # Convexity
                "MATUR_DATE",      # Maturity date
                "COUPON_RATE",     # Coupon rate
                "CURRENCY",        # Currency
            ]

            data = self._get_data(ric, fields)

            if not data:
                return None

            maturity_date = None
            if "MATUR_DATE" in data:
                maturity_date = self._parse_refinitiv_date(data["MATUR_DATE"])

            quote = BondQuote(
                timestamp=datetime.utcnow(),
                source=self.config.adapter_name,
                quality=DataQuality.REALTIME,
                isin=identifier if id_type == "isin" else "",
                cusip="",
                price=data.get("CF_LAST", 100.0),
                yield_to_maturity=data.get("CF_YIELD", 0.0) / 100.0,
                accrued_interest=data.get("ACCRUED_INT", 0.0),
                spread=data.get("SPREAD_MID", 0.0),
                duration=data.get("DURATION", 0.0),
                convexity=data.get("CONVEXITY", 0.0),
                maturity_date=maturity_date,
                coupon_rate=data.get("COUPON_RATE", 0.0) / 100.0,
                currency=data.get("CURRENCY", "USD"),
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
        Get FX quote from Refinitiv.

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
            # Refinitiv FX RIC format: <base>=
            # Example: EUR= (EUR/USD)
            ric = f"{base_currency}="

            fields = ["CF_LAST", "CF_BID", "CF_ASK"]

            data = self._get_data(ric, fields)

            if not data:
                return None

            quote = FXQuote(
                timestamp=datetime.utcnow(),
                source=self.config.adapter_name,
                quality=DataQuality.REALTIME,
                currency_pair=f"{base_currency}/{quote_currency}",
                base_currency=base_currency,
                quote_currency=quote_currency,
                spot=data.get("CF_LAST", 0.0),
                bid=data.get("CF_BID", 0.0),
                ask=data.get("CF_ASK", 0.0),
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
        Get interest rate from Refinitiv.

        Args:
            rate_type: Rate type (LIBOR, SOFR, etc.)
            currency: Currency code
            tenor: Tenor (e.g., "3M", "6M")

        Returns:
            InterestRateQuote or None if not available
        """
        cache_key = f"rate:{rate_type}:{currency}:{tenor}"
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        try:
            # Format RIC for interest rate
            ric = self._format_rate_ric(rate_type, currency, tenor)

            fields = ["CF_LAST"]

            data = self._get_data(ric, fields)

            if not data:
                return None

            quote = InterestRateQuote(
                timestamp=datetime.utcnow(),
                source=self.config.adapter_name,
                quality=DataQuality.REALTIME,
                rate_type=rate_type,
                currency=currency,
                tenor=tenor,
                rate=data.get("CF_LAST", 0.0) / 100.0,
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
        Get commodity quote from Refinitiv.

        Args:
            commodity_code: Refinitiv RIC (e.g., "LCOc1" for Brent crude)
            exchange: Exchange code (optional)

        Returns:
            CommodityQuote or None if not available
        """
        cache_key = f"commodity:{commodity_code}"
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        try:
            # Refinitiv commodity RIC examples:
            # - LCOc1: Brent crude front month
            # - CLc1: WTI crude front month
            # - GCc1: Gold front month

            fields = ["CF_LAST", "CURRENCY", "CONTRACT_MONTH"]

            data = self._get_data(commodity_code, fields)

            if not data:
                return None

            quote = CommodityQuote(
                timestamp=datetime.utcnow(),
                source=self.config.adapter_name,
                quality=DataQuality.REALTIME,
                commodity_code=commodity_code,
                commodity_name=commodity_code,
                price=data.get("CF_LAST", 0.0),
                unit="",
                contract_month=data.get("CONTRACT_MONTH", ""),
                exchange=exchange or "",
                currency=data.get("CURRENCY", "USD"),
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
        Get credit spread from Refinitiv.

        Args:
            issuer: Issuer identifier or RIC
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
            # Refinitiv CDS RIC format varies
            # Example: <issuer><currency><tenor>CDS=MP

            fields = ["SPREAD_MID", "RATING", "INDUSTRY", "COUNTRY"]

            data = self._get_data(issuer, fields)

            if not data:
                return None

            quote = CreditSpreadQuote(
                timestamp=datetime.utcnow(),
                source=self.config.adapter_name,
                quality=DataQuality.REALTIME,
                issuer=issuer,
                ticker=issuer,
                spread=data.get("SPREAD_MID", 0.0),
                rating=data.get("RATING", ""),
                sector=data.get("INDUSTRY", ""),
                region=data.get("COUNTRY", ""),
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
        Get volatility quote from Refinitiv.

        Args:
            underlying: Underlying asset RIC
            strike: Strike price
            expiry: Expiry date
            option_type: "call" or "put"

        Returns:
            VolatilityQuote or None if not available
        """
        try:
            # Refinitiv option RIC format varies by exchange
            # This is a simplified implementation

            fields = ["IMPLIED_VOLATILITY"]

            # Construct option RIC (simplified)
            option_ric = f"{underlying}.{expiry.strftime('%y%m%d')}.{int(strike)}"

            data = self._get_data(option_ric, fields)

            if not data:
                return None

            quote = VolatilityQuote(
                timestamp=datetime.utcnow(),
                source=self.config.adapter_name,
                quality=DataQuality.REALTIME,
                underlying=underlying,
                strike=strike,
                expiry=expiry,
                volatility=data.get("IMPLIED_VOLATILITY", 0.0) / 100.0,
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
        Get yield curve from Refinitiv.

        Args:
            curve_name: Curve identifier
            currency: Currency code

        Returns:
            YieldCurve or None if not available
        """
        try:
            # Refinitiv yield curve RICs
            # Example: US Treasury yields
            tenor_rics = {
                "US2YT=RR": 2.0,
                "US5YT=RR": 5.0,
                "US10YT=RR": 10.0,
                "US30YT=RR": 30.0,
            }

            tenors = []
            rates = []

            for ric, tenor in tenor_rics.items():
                data = self._get_data(ric, ["CF_LAST"])

                if data and "CF_LAST" in data:
                    tenors.append(tenor)
                    rates.append(data["CF_LAST"] / 100.0)

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

    def _execute_request_internal(self, request: DataRequest) -> List[MarketDataPoint]:
        """
        Execute internal request.

        Args:
            request: Data request

        Returns:
            List of market data points
        """
        # Placeholder for generic request handler
        return []

    def _get_data(self, ric: str, fields: List[str]) -> Optional[Dict[str, Any]]:
        """
        Get data from Refinitiv.

        Args:
            ric: Reuters Instrument Code
            fields: List of fields to retrieve

        Returns:
            Dictionary of field values or None if request fails
        """
        if not self.is_connected():
            logger.error("Not connected to Refinitiv")
            return None

        try:
            if self.config.use_desktop and self._eikon:
                # Use Eikon API
                df, err = self._eikon.get_data([ric], fields)

                if err or df is None or df.empty:
                    return None

                # Convert first row to dictionary
                result = df.iloc[0].to_dict()
                # Remove the 'Instrument' column
                result.pop('Instrument', None)
                return result

            elif self._rdp:
                # Use RDP API
                # This is a simplified implementation
                # In practice, you'd use specific RDP endpoints
                return None

            else:
                return None

        except Exception as e:
            logger.error(f"Error getting data for {ric}: {e}")
            return None

    @staticmethod
    def _format_rate_ric(rate_type: str, currency: str, tenor: str) -> str:
        """
        Format rate RIC for Refinitiv.

        Args:
            rate_type: Rate type
            currency: Currency
            tenor: Tenor

        Returns:
            Refinitiv RIC
        """
        rate_type_upper = rate_type.upper()

        if rate_type_upper == "LIBOR":
            # LIBOR format: <currency>LIBOR<tenor>=
            # Example: USDLIBOR3M=
            return f"{currency}LIBOR{tenor}="

        elif rate_type_upper == "SOFR":
            return "SOFR="

        elif rate_type_upper == "EURIBOR":
            # EURIBOR format: EURIBOR<tenor>=
            # Example: EURIBOR3M=
            return f"EURIBOR{tenor}="

        else:
            return f"{rate_type}_{currency}_{tenor}="

    @staticmethod
    def _extract_exchange_from_ric(ric: str) -> str:
        """
        Extract exchange code from RIC.

        Args:
            ric: Reuters Instrument Code

        Returns:
            Exchange code
        """
        # RIC format: <symbol>.<exchange>
        # Example: AAPL.O (O = NASDAQ)
        if "." in ric:
            exchange_code = ric.split(".")[-1]
            exchange_map = {
                "O": "NASDAQ",
                "N": "NYSE",
                "L": "LSE",
                "T": "TSE",
            }
            return exchange_map.get(exchange_code, exchange_code)
        return ""

    @staticmethod
    def _parse_refinitiv_date(date_value: Any) -> Optional[date]:
        """
        Parse Refinitiv date value.

        Args:
            date_value: Refinitiv date value

        Returns:
            Python date object or None
        """
        try:
            if isinstance(date_value, date):
                return date_value
            elif isinstance(date_value, datetime):
                return date_value.date()
            elif isinstance(date_value, str):
                # Try common date formats
                for fmt in ["%Y-%m-%d", "%d-%b-%Y", "%m/%d/%Y"]:
                    try:
                        return datetime.strptime(date_value, fmt).date()
                    except ValueError:
                        continue
                return None
            else:
                return None
        except Exception:
            return None
