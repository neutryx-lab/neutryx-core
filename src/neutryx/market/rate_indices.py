"""
Interest rate indices and conventions for major currencies.

This module defines standard interest rate indices (RFRs, LIBORs) with their
market conventions including day count, payment frequencies, and fixing rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class DayCountConvention(Enum):
    """Day count conventions for interest rate calculations."""
    ACT_360 = "ACT/360"  # Actual days / 360 (money market)
    ACT_365 = "ACT/365"  # Actual days / 365 (bond market)
    THIRTY_360 = "30/360"  # 30 days per month, 360 days per year
    ACT_ACT = "ACT/ACT"  # Actual days / actual days in year


class BusinessDayConvention(Enum):
    """Business day adjustment conventions."""
    FOLLOWING = "following"  # Next business day
    MODIFIED_FOLLOWING = "modified_following"  # Next business day unless different month
    PRECEDING = "preceding"  # Previous business day
    UNADJUSTED = "unadjusted"  # No adjustment


class CompoundingMethod(Enum):
    """Compounding methods for overnight rates."""
    COMPOUND = "compound"  # Daily compounding
    AVERAGING = "averaging"  # Simple averaging
    FLAT = "flat"  # Flat rate


@dataclass
class RateIndex:
    """
    Definition of an interest rate index.

    Attributes:
        name: Index name (e.g., "SOFR", "ESTR", "SONIA")
        currency: Currency code
        tenor: Tenor (e.g., "ON" for overnight, "3M" for 3 months)
        day_count: Day count convention
        payment_lag: Payment lag in days
        fixing_lag: Fixing lag in days (e.g., 2 for T+2 fixing)
        calendar: Business day calendar
        is_rfr: Whether this is a risk-free rate (OIS-style)
        compounding: Compounding method for overnight rates
    """

    name: str
    currency: str
    tenor: str
    day_count: DayCountConvention
    payment_lag: int = 0
    fixing_lag: int = 0
    calendar: str = "USD"
    is_rfr: bool = False
    compounding: CompoundingMethod = CompoundingMethod.COMPOUND


# =============================================================================
# USD Rates
# =============================================================================

SOFR = RateIndex(
    name="SOFR",
    currency="USD",
    tenor="ON",
    day_count=DayCountConvention.ACT_360,
    payment_lag=0,
    fixing_lag=0,
    calendar="USD",
    is_rfr=True,
    compounding=CompoundingMethod.COMPOUND,
)

USD_LIBOR_3M = RateIndex(
    name="USD-LIBOR-3M",
    currency="USD",
    tenor="3M",
    day_count=DayCountConvention.ACT_360,
    payment_lag=0,
    fixing_lag=2,
    calendar="USD",
    is_rfr=False,
)

USD_LIBOR_6M = RateIndex(
    name="USD-LIBOR-6M",
    currency="USD",
    tenor="6M",
    day_count=DayCountConvention.ACT_360,
    payment_lag=0,
    fixing_lag=2,
    calendar="USD",
    is_rfr=False,
)


# =============================================================================
# EUR Rates
# =============================================================================

EONIA = RateIndex(
    name="EONIA",
    currency="EUR",
    tenor="ON",
    day_count=DayCountConvention.ACT_360,
    payment_lag=0,
    fixing_lag=0,
    calendar="TARGET",
    is_rfr=True,
    compounding=CompoundingMethod.COMPOUND,
)

ESTR = RateIndex(
    name="ESTR",
    currency="EUR",
    tenor="ON",
    day_count=DayCountConvention.ACT_360,
    payment_lag=0,
    fixing_lag=0,
    calendar="TARGET",
    is_rfr=True,
    compounding=CompoundingMethod.COMPOUND,
)

EURIBOR_3M = RateIndex(
    name="EURIBOR-3M",
    currency="EUR",
    tenor="3M",
    day_count=DayCountConvention.ACT_360,
    payment_lag=0,
    fixing_lag=2,
    calendar="TARGET",
    is_rfr=False,
)

EURIBOR_6M = RateIndex(
    name="EURIBOR-6M",
    currency="EUR",
    tenor="6M",
    day_count=DayCountConvention.ACT_360,
    payment_lag=0,
    fixing_lag=2,
    calendar="TARGET",
    is_rfr=False,
)


# =============================================================================
# GBP Rates
# =============================================================================

SONIA = RateIndex(
    name="SONIA",
    currency="GBP",
    tenor="ON",
    day_count=DayCountConvention.ACT_365,
    payment_lag=0,
    fixing_lag=0,
    calendar="GBP",
    is_rfr=True,
    compounding=CompoundingMethod.COMPOUND,
)

GBP_LIBOR_3M = RateIndex(
    name="GBP-LIBOR-3M",
    currency="GBP",
    tenor="3M",
    day_count=DayCountConvention.ACT_365,
    payment_lag=0,
    fixing_lag=0,
    calendar="GBP",
    is_rfr=False,
)

GBP_LIBOR_6M = RateIndex(
    name="GBP-LIBOR-6M",
    currency="GBP",
    tenor="6M",
    day_count=DayCountConvention.ACT_365,
    payment_lag=0,
    fixing_lag=0,
    calendar="GBP",
    is_rfr=False,
)


# =============================================================================
# JPY Rates
# =============================================================================

TONAR = RateIndex(
    name="TONAR",
    currency="JPY",
    tenor="ON",
    day_count=DayCountConvention.ACT_365,
    payment_lag=0,
    fixing_lag=0,
    calendar="JPY",
    is_rfr=True,
    compounding=CompoundingMethod.COMPOUND,
)

JPY_LIBOR_3M = RateIndex(
    name="JPY-LIBOR-3M",
    currency="JPY",
    tenor="3M",
    day_count=DayCountConvention.ACT_360,
    payment_lag=0,
    fixing_lag=2,
    calendar="JPY",
    is_rfr=False,
)

JPY_LIBOR_6M = RateIndex(
    name="JPY-LIBOR-6M",
    currency="JPY",
    tenor="6M",
    day_count=DayCountConvention.ACT_360,
    payment_lag=0,
    fixing_lag=2,
    calendar="JPY",
    is_rfr=False,
)


# =============================================================================
# CHF Rates
# =============================================================================

SARON = RateIndex(
    name="SARON",
    currency="CHF",
    tenor="ON",
    day_count=DayCountConvention.ACT_360,
    payment_lag=0,
    fixing_lag=0,
    calendar="CHF",
    is_rfr=True,
    compounding=CompoundingMethod.COMPOUND,
)


# =============================================================================
# Registry
# =============================================================================

RATE_INDEX_REGISTRY: Dict[str, RateIndex] = {
    "SOFR": SOFR,
    "USD-LIBOR-3M": USD_LIBOR_3M,
    "USD-LIBOR-6M": USD_LIBOR_6M,
    "EONIA": EONIA,
    "ESTR": ESTR,
    "EURIBOR-3M": EURIBOR_3M,
    "EURIBOR-6M": EURIBOR_6M,
    "SONIA": SONIA,
    "GBP-LIBOR-3M": GBP_LIBOR_3M,
    "GBP-LIBOR-6M": GBP_LIBOR_6M,
    "TONAR": TONAR,
    "JPY-LIBOR-3M": JPY_LIBOR_3M,
    "JPY-LIBOR-6M": JPY_LIBOR_6M,
    "SARON": SARON,
}


def get_rate_index(name: str) -> Optional[RateIndex]:
    """Get rate index by name."""
    return RATE_INDEX_REGISTRY.get(name)


def get_rfr_index(currency: str) -> Optional[RateIndex]:
    """Get the risk-free rate index for a currency."""
    rfr_map = {
        "USD": SOFR,
        "EUR": ESTR,
        "GBP": SONIA,
        "JPY": TONAR,
        "CHF": SARON,
    }
    return rfr_map.get(currency)


def get_ibor_index(currency: str, tenor: str) -> Optional[RateIndex]:
    """Get the IBOR index for a currency and tenor."""
    key = f"{currency}-LIBOR-{tenor}" if currency != "EUR" else f"EURIBOR-{tenor}"
    return RATE_INDEX_REGISTRY.get(key)


@dataclass
class SwapConvention:
    """
    Market conventions for interest rate swaps.

    Attributes:
        currency: Currency code
        fixed_leg_frequency: Fixed leg payment frequency per year
        float_leg_frequency: Floating leg payment frequency per year
        fixed_leg_day_count: Day count for fixed leg
        float_leg_day_count: Day count for floating leg
        business_day_convention: Business day adjustment
        calendar: Business day calendar
        spot_lag: Settlement lag in days (e.g., T+2)
    """

    currency: str
    fixed_leg_frequency: int  # Payments per year
    float_leg_frequency: int
    fixed_leg_day_count: DayCountConvention
    float_leg_day_count: DayCountConvention
    business_day_convention: BusinessDayConvention
    calendar: str
    spot_lag: int = 2


# Standard swap conventions
USD_SWAP_CONVENTION = SwapConvention(
    currency="USD",
    fixed_leg_frequency=2,  # Semi-annual
    float_leg_frequency=4,  # Quarterly (3M SOFR)
    fixed_leg_day_count=DayCountConvention.THIRTY_360,
    float_leg_day_count=DayCountConvention.ACT_360,
    business_day_convention=BusinessDayConvention.MODIFIED_FOLLOWING,
    calendar="USD",
    spot_lag=2,
)

EUR_SWAP_CONVENTION = SwapConvention(
    currency="EUR",
    fixed_leg_frequency=1,  # Annual
    float_leg_frequency=2,  # Semi-annual (6M EURIBOR)
    fixed_leg_day_count=DayCountConvention.THIRTY_360,
    float_leg_day_count=DayCountConvention.ACT_360,
    business_day_convention=BusinessDayConvention.MODIFIED_FOLLOWING,
    calendar="TARGET",
    spot_lag=2,
)

GBP_SWAP_CONVENTION = SwapConvention(
    currency="GBP",
    fixed_leg_frequency=2,  # Semi-annual
    float_leg_frequency=2,  # Semi-annual (6M SONIA)
    fixed_leg_day_count=DayCountConvention.ACT_365,
    float_leg_day_count=DayCountConvention.ACT_365,
    business_day_convention=BusinessDayConvention.MODIFIED_FOLLOWING,
    calendar="GBP",
    spot_lag=0,
)

JPY_SWAP_CONVENTION = SwapConvention(
    currency="JPY",
    fixed_leg_frequency=2,  # Semi-annual
    float_leg_frequency=2,  # Semi-annual (6M TONAR)
    fixed_leg_day_count=DayCountConvention.ACT_365,
    float_leg_day_count=DayCountConvention.ACT_365,
    business_day_convention=BusinessDayConvention.MODIFIED_FOLLOWING,
    calendar="JPY",
    spot_lag=2,
)

SWAP_CONVENTIONS: Dict[str, SwapConvention] = {
    "USD": USD_SWAP_CONVENTION,
    "EUR": EUR_SWAP_CONVENTION,
    "GBP": GBP_SWAP_CONVENTION,
    "JPY": JPY_SWAP_CONVENTION,
}


def get_swap_convention(currency: str) -> Optional[SwapConvention]:
    """Get swap convention for a currency."""
    return SWAP_CONVENTIONS.get(currency)
