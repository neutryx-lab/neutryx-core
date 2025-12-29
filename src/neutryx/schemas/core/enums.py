"""Consolidated enum definitions for all Neutryx domains.

This module provides the single source of truth for all enumeration types
used across the Neutryx platform. Enums are organized by domain category.

Migration Note:
    Previous enum locations are deprecated:
    - neutryx.integrations.clearing.base.TradeStatus -> use this module
    - neutryx.portfolio.contracts.trade.TradeStatus -> use this module
    - neutryx.integrations.clearing.base.ProductType -> use this module
    - neutryx.market.conventions.DayCountConvention -> use this module
"""

from enum import Enum


# =============================================================================
# Trade Lifecycle Enums
# =============================================================================


class TradeStatus(str, Enum):
    """Universal trade lifecycle status.

    Consolidates statuses from clearing and portfolio domains into a single,
    comprehensive lifecycle definition.

    Lifecycle flow:
        PENDING -> SUBMITTED -> ACCEPTED -> ACTIVE -> CLEARED -> SETTLED
                      |                                  |
                      v                                  v
                   REJECTED                           MATURED
                                                         |
                                                         v
                                                 TERMINATED / NOVATED
    """

    # Pre-trade states
    PENDING = "pending"  # Awaiting submission or confirmation
    DRAFT = "draft"  # Draft state, not yet submitted

    # Submission states
    SUBMITTED = "submitted"  # Submitted to counterparty or CCP
    ACCEPTED = "accepted"  # Accepted by counterparty or CCP
    REJECTED = "rejected"  # Rejected by counterparty or CCP

    # Active states
    ACTIVE = "active"  # Live and accruing exposure
    CLEARED = "cleared"  # Cleared through CCP

    # Settlement states
    SETTLING = "settling"  # Settlement in progress
    SETTLED = "settled"  # Fully settled

    # Termination states
    CANCELLED = "cancelled"  # Cancelled before execution
    TERMINATED = "terminated"  # Early termination
    MATURED = "matured"  # Reached maturity date
    NOVATED = "novated"  # Transferred to another party
    FAILED = "failed"  # Failed to complete lifecycle


class ProductType(str, Enum):
    """Product classification for derivatives and securities.

    Consolidates product types from clearing, portfolio, and FpML domains.
    Values use snake_case for consistency.

    Note: Common abbreviations are available as class attributes:
        - ProductType.IRS -> INTEREST_RATE_SWAP
        - ProductType.CDS -> CREDIT_DEFAULT_SWAP
        - ProductType.OIS -> OVERNIGHT_INDEX_SWAP
        - ProductType.CCS -> CROSS_CURRENCY_SWAP
    """

    # Interest Rate
    INTEREST_RATE_SWAP = "interest_rate_swap"
    OVERNIGHT_INDEX_SWAP = "overnight_index_swap"
    BASIS_SWAP = "basis_swap"
    CROSS_CURRENCY_SWAP = "cross_currency_swap"
    FRA = "forward_rate_agreement"
    SWAPTION = "swaption"
    CAP_FLOOR = "cap_floor"
    CAP = "cap"
    FLOOR = "floor"
    COLLAR = "collar"

    # Credit
    CREDIT_DEFAULT_SWAP = "credit_default_swap"
    CDS_INDEX = "cds_index"
    CDS_TRANCHE = "cds_tranche"
    CREDIT_LINKED_NOTE = "credit_linked_note"

    # FX
    FX_SPOT = "fx_spot"
    FX_FORWARD = "fx_forward"
    FX_SWAP = "fx_swap"
    FX_OPTION = "fx_option"
    FX_NDF = "fx_ndf"  # Non-deliverable forward

    # Equity
    EQUITY_OPTION = "equity_option"
    EQUITY_FORWARD = "equity_forward"
    EQUITY_SWAP = "equity_swap"
    VARIANCE_SWAP = "variance_swap"
    VOLATILITY_SWAP = "volatility_swap"
    TOTAL_RETURN_SWAP = "total_return_swap"

    # Commodity
    COMMODITY_FUTURE = "commodity_future"
    COMMODITY_OPTION = "commodity_option"
    COMMODITY_SWAP = "commodity_swap"
    COMMODITY_FORWARD = "commodity_forward"

    # Fixed Income
    BOND = "bond"
    REPO = "repo"
    REVERSE_REPO = "reverse_repo"

    # Generic
    FORWARD = "forward"
    FUTURE = "future"
    OPTION = "option"
    SWAP = "swap"
    OTHER = "other"


# Backward compatibility aliases for ProductType
# These allow code using ProductType.IRS to continue working
ProductType.IRS = ProductType.INTEREST_RATE_SWAP
ProductType.CDS = ProductType.CREDIT_DEFAULT_SWAP
ProductType.OIS = ProductType.OVERNIGHT_INDEX_SWAP
ProductType.CCS = ProductType.CROSS_CURRENCY_SWAP


class SettlementType(str, Enum):
    """Settlement method for derivatives.

    Defines how a trade is settled at maturity or exercise.
    """

    CASH = "cash"  # Cash settlement
    PHYSICAL = "physical"  # Physical delivery
    DVP = "dvp"  # Delivery vs Payment
    FOP = "fop"  # Free of Payment
    RVP = "rvp"  # Receive vs Payment
    ELECTION = "election"  # Counterparty election at expiry


class OptionType(str, Enum):
    """Option exercise style."""

    EUROPEAN = "european"  # Exercise only at expiry
    AMERICAN = "american"  # Exercise any time until expiry
    BERMUDAN = "bermudan"  # Exercise on specific dates


class PutCall(str, Enum):
    """Option direction."""

    PUT = "put"
    CALL = "call"


# =============================================================================
# Message and Communication Enums
# =============================================================================


class MessageType(str, Enum):
    """CCP and communication message types."""

    TRADE_SUBMISSION = "trade_submission"
    TRADE_CONFIRMATION = "trade_confirmation"
    TRADE_REJECTION = "trade_rejection"
    MARGIN_CALL = "margin_call"
    SETTLEMENT = "settlement"
    POSITION_REPORT = "position_report"
    RISK_REPORT = "risk_report"
    STATUS_UPDATE = "status_update"
    HEARTBEAT = "heartbeat"


# =============================================================================
# Market Data Enums
# =============================================================================


class AssetClass(str, Enum):
    """Asset class enumeration."""

    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    FX = "fx"
    COMMODITY = "commodity"
    CREDIT = "credit"
    RATES = "rates"
    VOLATILITY = "volatility"
    CRYPTO = "crypto"


class DataQuality(str, Enum):
    """Market data quality indicators."""

    REALTIME = "realtime"  # Live streaming data
    DELAYED = "delayed"  # Delayed feed (e.g., 15-min delay)
    END_OF_DAY = "end_of_day"  # EOD snapshot
    INDICATIVE = "indicative"  # Indicative/derived quote
    STALE = "stale"  # Exceeded staleness threshold
    MISSING = "missing"  # No data available


class QuoteType(str, Enum):
    """Type of market quote."""

    BID = "bid"
    ASK = "ask"
    MID = "mid"
    LAST = "last"
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    SETTLEMENT = "settlement"


# =============================================================================
# Date and Calendar Conventions
# =============================================================================


class DayCountConvention(str, Enum):
    """Standard day count conventions for accrual calculations.

    References:
        - ISDA definitions
        - Bloomberg convention codes
    """

    # Actual day count conventions
    ACT_360 = "ACT/360"  # Money market
    ACT_365 = "ACT/365"  # Fixed
    ACT_365L = "ACT/365L"  # Leap year aware
    ACT_ACT = "ACT/ACT"  # ISDA
    ACT_ACT_ISDA = "ACT/ACT_ISDA"
    ACT_ACT_ICMA = "ACT/ACT_ICMA"  # Bond

    # 30/360 conventions
    THIRTY_360 = "30/360"  # Bond basis (US)
    THIRTY_360_US = "30/360_US"
    THIRTY_E_360 = "30E/360"  # European
    THIRTY_E_360_ISDA = "30E/360_ISDA"

    # Business day based
    BUS_252 = "BUS/252"  # Brazil


class BusinessDayConvention(str, Enum):
    """Date adjustment conventions for business days."""

    FOLLOWING = "following"  # Next business day
    MODIFIED_FOLLOWING = "modified_following"  # Following unless crosses month
    PRECEDING = "preceding"  # Previous business day
    MODIFIED_PRECEDING = "modified_preceding"  # Preceding unless crosses month
    UNADJUSTED = "unadjusted"  # No adjustment
    END_OF_MONTH = "end_of_month"  # End of month adjustment
    NONE = "none"  # Explicitly no convention


# =============================================================================
# Currency Codes (ISO 4217)
# =============================================================================


class CurrencyCode(str, Enum):
    """ISO 4217 currency codes.

    Includes major and commonly traded currencies.
    """

    # G10 Currencies
    USD = "USD"  # US Dollar
    EUR = "EUR"  # Euro
    GBP = "GBP"  # British Pound
    JPY = "JPY"  # Japanese Yen
    CHF = "CHF"  # Swiss Franc
    AUD = "AUD"  # Australian Dollar
    CAD = "CAD"  # Canadian Dollar
    NZD = "NZD"  # New Zealand Dollar
    SEK = "SEK"  # Swedish Krona
    NOK = "NOK"  # Norwegian Krone

    # Asia Pacific
    HKD = "HKD"  # Hong Kong Dollar
    SGD = "SGD"  # Singapore Dollar
    CNY = "CNY"  # Chinese Yuan (onshore)
    CNH = "CNH"  # Chinese Yuan (offshore)
    KRW = "KRW"  # South Korean Won
    INR = "INR"  # Indian Rupee
    TWD = "TWD"  # Taiwan Dollar

    # Emerging Markets
    BRL = "BRL"  # Brazilian Real
    MXN = "MXN"  # Mexican Peso
    ZAR = "ZAR"  # South African Rand
    TRY = "TRY"  # Turkish Lira
    RUB = "RUB"  # Russian Ruble
    PLN = "PLN"  # Polish Zloty


# =============================================================================
# Credit Enums
# =============================================================================


class CreditRating(str, Enum):
    """S&P-style credit ratings."""

    AAA = "AAA"
    AA_PLUS = "AA+"
    AA = "AA"
    AA_MINUS = "AA-"
    A_PLUS = "A+"
    A = "A"
    A_MINUS = "A-"
    BBB_PLUS = "BBB+"
    BBB = "BBB"
    BBB_MINUS = "BBB-"
    BB_PLUS = "BB+"
    BB = "BB"
    BB_MINUS = "BB-"
    B_PLUS = "B+"
    B = "B"
    B_MINUS = "B-"
    CCC_PLUS = "CCC+"
    CCC = "CCC"
    CCC_MINUS = "CCC-"
    CC = "CC"
    C = "C"
    D = "D"  # Default
    NR = "NR"  # Not Rated


class EntityType(str, Enum):
    """Legal entity classification for counterparties."""

    CORPORATE = "corporate"
    FINANCIAL = "financial"
    SOVEREIGN = "sovereign"
    MUNICIPAL = "municipal"
    FUND = "fund"
    SPV = "spv"  # Special Purpose Vehicle
    CCP = "ccp"  # Central Counterparty
    OTHER = "other"


# =============================================================================
# Regulatory Enums
# =============================================================================


class RegulatoryRegime(str, Enum):
    """Regulatory reporting regimes."""

    EMIR = "emir"  # European Market Infrastructure Regulation
    DODD_FRANK = "dodd_frank"  # US Dodd-Frank
    MIFID2 = "mifid2"  # Markets in Financial Instruments Directive
    SFTR = "sftr"  # Securities Financing Transactions Regulation
    BASEL3 = "basel3"  # Basel III banking regulation
    FRTB = "frtb"  # Fundamental Review of the Trading Book


class ReportType(str, Enum):
    """Regulatory report types."""

    TRADE = "trade"
    VALUATION = "valuation"
    COLLATERAL = "collateral"
    POSITION = "position"
    MARGIN = "margin"
