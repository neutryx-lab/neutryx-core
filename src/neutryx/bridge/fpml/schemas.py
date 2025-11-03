"""FpML schema definitions using Pydantic models.

This module defines data models for FpML (Financial products Markup Language)
documents, focusing on equity and FX derivatives.

Supports FpML 5.x standard.
"""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class PutCallEnum(str, Enum):
    """Put or call option type."""

    PUT = "Put"
    CALL = "Call"


class OptionTypeEnum(str, Enum):
    """European or American exercise style."""

    EUROPEAN = "European"
    AMERICAN = "American"


class CurrencyCode(str, Enum):
    """ISO 4217 currency codes (subset)."""

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CHF = "CHF"
    AUD = "AUD"
    CAD = "CAD"
    HKD = "HKD"


class BusinessDayConventionEnum(str, Enum):
    """Business day adjustment conventions."""

    FOLLOWING = "FOLLOWING"
    MODFOLLOWING = "MODFOLLOWING"
    PRECEDING = "PRECEDING"
    MODPRECEDING = "MODPRECEDING"
    NONE = "NONE"


# Core building blocks


class Party(BaseModel):
    """Party identification."""

    id: str = Field(..., description="Party identifier")
    name: Optional[str] = Field(None, description="Party name")


class PartyReference(BaseModel):
    """Reference to a party."""

    href: str = Field(..., description="Reference to party id")


class Money(BaseModel):
    """Monetary amount with currency."""

    currency: CurrencyCode = Field(..., description="Currency code")
    amount: Decimal = Field(..., description="Monetary amount")


class PositiveDecimal(BaseModel):
    """Positive decimal value (e.g., prices, rates)."""

    value: Decimal = Field(..., gt=0, description="Positive decimal value")


class DateAdjustments(BaseModel):
    """Business day adjustments for dates."""

    businessDayConvention: BusinessDayConventionEnum
    businessCenters: Optional[list[str]] = None


class AdjustableDate(BaseModel):
    """Date with optional business day adjustments."""

    unadjustedDate: date
    dateAdjustments: Optional[DateAdjustments] = None


class AdjustableOrRelativeDate(BaseModel):
    """Date that can be adjusted or relative."""

    adjustableDate: Optional[AdjustableDate] = None
    unadjustedDate: Optional[date] = None


# Equity components


class EquityUnderlyer(BaseModel):
    """Equity underlying asset."""

    instrumentId: str = Field(..., description="Instrument identifier (ISIN, RIC, etc.)")
    description: Optional[str] = Field(None, description="Instrument description")
    exchangeId: Optional[str] = Field(None, description="Exchange code")
    currency: Optional[CurrencyCode] = Field(None, description="Trading currency")


class EquityExercise(BaseModel):
    """Equity option exercise terms."""

    optionType: OptionTypeEnum = Field(..., description="European or American")
    expirationDate: AdjustableDate = Field(..., description="Expiration date")
    equityExpirationTimeType: Optional[str] = Field(None, description="Close, Open, etc.")
    automaticExercise: bool = Field(True, description="Automatic exercise if in-the-money")


class EquityStrike(BaseModel):
    """Equity option strike."""

    strikePrice: Decimal = Field(..., description="Strike price")
    currency: Optional[CurrencyCode] = Field(None, description="Strike currency")


class EquityPremium(BaseModel):
    """Option premium."""

    payerPartyReference: PartyReference
    receiverPartyReference: PartyReference
    paymentAmount: Money
    paymentDate: AdjustableDate
    pricePerOption: Optional[Money] = None


class EquityOption(BaseModel):
    """FpML Equity Option representation."""

    productType: Literal["EquityOption"] = "EquityOption"
    buyerPartyReference: PartyReference
    sellerPartyReference: PartyReference
    optionType: PutCallEnum
    underlyer: EquityUnderlyer
    notional: Optional[Money] = None
    equityExercise: EquityExercise
    strike: EquityStrike
    numberOfOptions: Decimal = Field(..., gt=0, description="Number of options")
    optionEntitlement: Decimal = Field(
        default=Decimal("1.0"), description="Multiplier per option"
    )
    settlementType: Optional[str] = Field(None, description="Cash or Physical")
    premium: Optional[EquityPremium] = None


# FX components


class QuotedCurrencyPair(BaseModel):
    """Currency pair quotation."""

    currency1: CurrencyCode
    currency2: CurrencyCode
    quoteBasis: Literal["Currency1PerCurrency2", "Currency2PerCurrency1"] = (
        "Currency2PerCurrency1"
    )


class FxRate(BaseModel):
    """FX exchange rate."""

    quotedCurrencyPair: QuotedCurrencyPair
    rate: Decimal = Field(..., gt=0, description="Exchange rate")


class FxSpotRate(BaseModel):
    """FX spot rate at trade inception."""

    quotedCurrencyPair: QuotedCurrencyPair
    spotRate: Decimal = Field(..., gt=0, description="Spot rate")


class FxExercise(BaseModel):
    """FX option exercise terms."""

    optionType: OptionTypeEnum = Field(..., description="European or American")
    expiryDate: date
    expiryTime: Optional[str] = None
    cutName: Optional[str] = Field(None, description="e.g., 'NewYork', 'Tokyo'")
    latestValueDate: Optional[date] = None


class FxStrike(BaseModel):
    """FX option strike."""

    rate: Decimal = Field(..., gt=0, description="Strike rate")
    strikeQuoteBasis: Optional[str] = None


class FxOption(BaseModel):
    """FpML FX Option representation."""

    productType: Literal["FxOption"] = "FxOption"
    buyerPartyReference: PartyReference
    sellerPartyReference: PartyReference
    putCurrencyAmount: Money
    callCurrencyAmount: Money
    strike: FxStrike
    spotRate: Optional[FxSpotRate] = None
    fxExercise: FxExercise
    premium: Optional[EquityPremium] = None


# Interest Rate Swap components


class FloatingRateIndex(str, Enum):
    """Standard floating rate indices."""

    USD_LIBOR = "USD-LIBOR"
    EUR_EURIBOR = "EUR-EURIBOR"
    GBP_LIBOR = "GBP-LIBOR"
    JPY_LIBOR = "JPY-LIBOR"
    USD_SOFR = "USD-SOFR"
    EUR_ESTR = "EUR-ESTR"


class DayCountFraction(str, Enum):
    """Day count conventions."""

    ACT_360 = "ACT/360"
    ACT_365 = "ACT/365.FIXED"
    ACT_ACT = "ACT/ACT.ISDA"
    THIRTY_360 = "30/360"
    THIRTY_E_360 = "30E/360"


class PaymentFrequency(BaseModel):
    """Payment frequency specification."""

    periodMultiplier: int = Field(..., gt=0)
    period: Literal["D", "W", "M", "Y"] = "M"  # Day, Week, Month, Year


class CalculationPeriodDates(BaseModel):
    """Schedule for calculation periods."""

    effectiveDate: AdjustableDate
    terminationDate: AdjustableDate
    calculationPeriodFrequency: PaymentFrequency
    calculationPeriodDatesAdjustments: Optional[DateAdjustments] = None


class PaymentDates(BaseModel):
    """Schedule for payment dates."""

    calculationPeriodDatesReference: Optional[str] = None
    paymentFrequency: PaymentFrequency
    payRelativeTo: Literal["CalculationPeriodStartDate", "CalculationPeriodEndDate"] = (
        "CalculationPeriodEndDate"
    )
    paymentDatesAdjustments: Optional[DateAdjustments] = None


class FixedRateSchedule(BaseModel):
    """Fixed rate schedule."""

    initialValue: Decimal = Field(..., description="Fixed rate (e.g., 0.05 for 5%)")


class FloatingRateCalculation(BaseModel):
    """Floating rate calculation parameters."""

    floatingRateIndex: FloatingRateIndex
    indexTenor: Optional[PaymentFrequency] = None
    spreadSchedule: Optional[Decimal] = Field(
        None, description="Spread over index (bp as decimal)"
    )
    dayCountFraction: DayCountFraction = DayCountFraction.ACT_360


class Calculation(BaseModel):
    """Calculation details for swap leg."""

    notionalSchedule: Money
    fixedRateSchedule: Optional[FixedRateSchedule] = None
    floatingRateCalculation: Optional[FloatingRateCalculation] = None
    dayCountFraction: DayCountFraction


class SwapStream(BaseModel):
    """Single leg of an interest rate swap."""

    payerPartyReference: PartyReference
    receiverPartyReference: PartyReference
    calculationPeriodDates: CalculationPeriodDates
    paymentDates: PaymentDates
    calculationPeriodAmount: Calculation


class InterestRateSwap(BaseModel):
    """FpML Interest Rate Swap representation."""

    productType: Literal["InterestRateSwap"] = "InterestRateSwap"
    swapStream: list[SwapStream] = Field(..., min_length=2, max_length=2)


# Trade container


class TradeHeader(BaseModel):
    """Trade header with identifiers."""

    partyTradeIdentifier: Optional[str] = None
    tradeDate: date


class Trade(BaseModel):
    """FpML Trade container."""

    tradeHeader: TradeHeader
    equityOption: Optional[EquityOption] = None
    fxOption: Optional[FxOption] = None
    swap: Optional[InterestRateSwap] = None


class FpMLDocument(BaseModel):
    """Root FpML document."""

    party: list[Party]
    trade: list[Trade] = Field(..., min_length=1)

    @property
    def primary_trade(self) -> Trade:
        """Return the first trade (most common case)."""
        return self.trade[0]


__all__ = [
    "FpMLDocument",
    "Trade",
    "TradeHeader",
    "EquityOption",
    "FxOption",
    "InterestRateSwap",
    "Party",
    "PartyReference",
    "Money",
    "CurrencyCode",
    "PutCallEnum",
    "OptionTypeEnum",
    "EquityUnderlyer",
    "EquityExercise",
    "EquityStrike",
    "FxExercise",
    "FxStrike",
    "SwapStream",
]
