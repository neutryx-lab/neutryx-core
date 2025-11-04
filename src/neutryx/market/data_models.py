"""
Market Data Models - Comprehensive data structures for all asset classes.

Supports equities, fixed income, FX, commodities, credit, and derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Dict, List, Optional, Union, Any
import jax.numpy as jnp


class AssetClass(Enum):
    """Asset class enumeration."""
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    FX = "fx"
    COMMODITY = "commodity"
    CREDIT = "credit"
    RATES = "rates"
    VOLATILITY = "volatility"


class QuoteType(Enum):
    """Market quote type."""
    PRICE = "price"
    YIELD = "yield"
    SPREAD = "spread"
    RATE = "rate"
    VOLATILITY = "volatility"
    IMPLIED_VOL = "implied_vol"


class DataQuality(Enum):
    """Data quality indicators."""
    REALTIME = "realtime"
    DELAYED = "delayed"
    END_OF_DAY = "end_of_day"
    INDICATIVE = "indicative"
    STALE = "stale"
    MISSING = "missing"


@dataclass
class MarketDataPoint:
    """
    Base class for all market data points.

    Attributes:
        timestamp: Data timestamp (UTC)
        source: Data source identifier
        quality: Data quality indicator
        metadata: Additional metadata
    """
    timestamp: datetime
    source: str
    quality: DataQuality = DataQuality.REALTIME
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EquityQuote(MarketDataPoint):
    """
    Equity market quote.

    Attributes:
        ticker: Stock ticker symbol (e.g., "AAPL")
        exchange: Exchange code (e.g., "NASDAQ", "NYSE")
        price: Last traded price
        bid: Bid price
        ask: Ask price
        volume: Trading volume
        open_price: Opening price
        high: Day high
        low: Day low
        close: Previous close
        currency: Quote currency
    """
    ticker: str = ""
    exchange: str = ""
    price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    volume: float = 0.0
    open_price: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    currency: str = "USD"


@dataclass
class BondQuote(MarketDataPoint):
    """
    Bond market quote.

    Attributes:
        isin: International Securities ID Number
        cusip: CUSIP identifier
        price: Clean price (percentage of par)
        yield_to_maturity: Yield to maturity
        accrued_interest: Accrued interest
        spread: Spread over benchmark (bps)
        duration: Modified duration
        convexity: Convexity
        maturity_date: Maturity date
        coupon_rate: Coupon rate
        currency: Quote currency
    """
    isin: str = ""
    cusip: str = ""
    price: float = 100.0
    yield_to_maturity: float = 0.0
    accrued_interest: float = 0.0
    spread: float = 0.0
    duration: float = 0.0
    convexity: float = 0.0
    maturity_date: Optional[date] = None
    coupon_rate: float = 0.0
    currency: str = "USD"


@dataclass
class FXQuote(MarketDataPoint):
    """
    Foreign exchange quote.

    Attributes:
        currency_pair: Currency pair (e.g., "EUR/USD")
        base_currency: Base currency
        quote_currency: Quote currency
        spot: Spot rate
        bid: Bid rate
        ask: Ask rate
        forward_points: Forward points (for forward contracts)
        tenor: Tenor for forward quotes
    """
    currency_pair: str = ""
    base_currency: str = ""
    quote_currency: str = ""
    spot: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    forward_points: Optional[float] = None
    tenor: Optional[str] = None


@dataclass
class InterestRateQuote(MarketDataPoint):
    """
    Interest rate quote.

    Attributes:
        rate_type: Rate type (LIBOR, SOFR, EURIBOR, etc.)
        currency: Currency
        tenor: Tenor (e.g., "3M", "6M", "1Y")
        rate: Interest rate (decimal)
        curve_name: Curve name (for curve construction)
    """
    rate_type: str = ""
    currency: str = "USD"
    tenor: str = ""
    rate: float = 0.0
    curve_name: str = ""


@dataclass
class CommodityQuote(MarketDataPoint):
    """
    Commodity market quote.

    Attributes:
        commodity_code: Commodity code (e.g., "CL" for crude oil)
        commodity_name: Commodity name
        price: Price
        unit: Unit of measure (barrels, ounces, etc.)
        contract_month: Futures contract month
        exchange: Exchange code
        currency: Quote currency
    """
    commodity_code: str = ""
    commodity_name: str = ""
    price: float = 0.0
    unit: str = ""
    contract_month: Optional[str] = None
    exchange: str = ""
    currency: str = "USD"


@dataclass
class CreditSpreadQuote(MarketDataPoint):
    """
    Credit spread quote.

    Attributes:
        issuer: Issuer name or ID
        ticker: Credit ticker
        spread: Credit spread (bps)
        rating: Credit rating
        sector: Industry sector
        region: Geographic region
        currency: Quote currency
        tenor: Tenor
    """
    issuer: str = ""
    ticker: str = ""
    spread: float = 0.0
    rating: str = ""
    sector: str = ""
    region: str = ""
    currency: str = "USD"
    tenor: str = ""


@dataclass
class VolatilityQuote(MarketDataPoint):
    """
    Volatility quote (options implied volatility).

    Attributes:
        underlying: Underlying asset identifier
        strike: Strike price
        expiry: Expiry date or tenor
        volatility: Implied volatility (decimal)
        option_type: "call" or "put"
        delta: Option delta (if available)
        vega: Option vega (if available)
    """
    underlying: str = ""
    strike: float = 0.0
    expiry: Union[date, str] = ""
    volatility: float = 0.0
    option_type: str = "call"
    delta: Optional[float] = None
    vega: Optional[float] = None


@dataclass
class VolatilitySurface(MarketDataPoint):
    """
    Complete volatility surface.

    Attributes:
        underlying: Underlying asset identifier
        strikes: Array of strike prices
        expiries: Array of expiry dates/tenors
        volatilities: 2D array of implied volatilities
        quote_type: Surface type (implied vol, local vol, etc.)
    """
    underlying: str = ""
    strikes: List[float] = field(default_factory=list)
    expiries: List[Union[date, str]] = field(default_factory=list)
    volatilities: List[List[float]] = field(default_factory=list)
    quote_type: str = "implied_vol"


@dataclass
class YieldCurve(MarketDataPoint):
    """
    Yield curve data.

    Attributes:
        curve_name: Curve identifier
        currency: Currency
        tenors: List of tenors (in years)
        rates: List of rates (decimal)
        curve_type: Curve type (zero, forward, discount)
        interpolation_method: Interpolation method used
    """
    curve_name: str = ""
    currency: str = "USD"
    tenors: List[float] = field(default_factory=list)
    rates: List[float] = field(default_factory=list)
    curve_type: str = "zero"
    interpolation_method: str = "linear"


@dataclass
class MarketDataSnapshot:
    """
    Complete market data snapshot at a point in time.

    Useful for storing and retrieving complete market states.

    Attributes:
        snapshot_time: Snapshot timestamp
        equities: Dictionary of equity quotes
        bonds: Dictionary of bond quotes
        fx_rates: Dictionary of FX quotes
        interest_rates: Dictionary of interest rate quotes
        commodities: Dictionary of commodity quotes
        credit_spreads: Dictionary of credit spread quotes
        volatilities: Dictionary of volatility quotes
        curves: Dictionary of yield curves
    """
    snapshot_time: datetime
    equities: Dict[str, EquityQuote] = field(default_factory=dict)
    bonds: Dict[str, BondQuote] = field(default_factory=dict)
    fx_rates: Dict[str, FXQuote] = field(default_factory=dict)
    interest_rates: Dict[str, InterestRateQuote] = field(default_factory=dict)
    commodities: Dict[str, CommodityQuote] = field(default_factory=dict)
    credit_spreads: Dict[str, CreditSpreadQuote] = field(default_factory=dict)
    volatilities: Dict[str, VolatilityQuote] = field(default_factory=dict)
    curves: Dict[str, YieldCurve] = field(default_factory=dict)


@dataclass
class DataRequest:
    """
    Market data request specification.

    Attributes:
        asset_class: Asset class
        identifiers: Asset identifiers (tickers, ISINs, etc.)
        fields: Requested fields
        start_time: Start time (for historical data)
        end_time: End time (for historical data)
        frequency: Data frequency (tick, 1min, 1day, etc.)
        source_preference: Preferred data sources (in order)
    """
    asset_class: AssetClass
    identifiers: List[str]
    fields: List[str]
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    frequency: str = "realtime"
    source_preference: List[str] = field(default_factory=list)


@dataclass
class DataResponse:
    """
    Market data response.

    Attributes:
        request: Original request
        data: Response data (list of MarketDataPoint or subclasses)
        success: Success indicator
        error_message: Error message (if any)
        latency_ms: Response latency in milliseconds
    """
    request: DataRequest
    data: List[MarketDataPoint]
    success: bool = True
    error_message: str = ""
    latency_ms: float = 0.0
