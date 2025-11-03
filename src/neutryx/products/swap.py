"""Interest Rate Swap valuation and cash flow generation.

Implements vanilla fixed-floating interest rate swaps with:
- Cash flow schedule generation
- Present value calculation
- Fixed and floating leg valuation
- DV01 and duration metrics
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from enum import Enum
from typing import Sequence

import jax.numpy as jnp
from jax import jit


class DayCountConvention(Enum):
    """Day count conventions for interest calculations."""

    ACT_360 = "ACT/360"
    ACT_365 = "ACT/365"
    ACT_ACT = "ACT/ACT"
    THIRTY_360 = "30/360"


class PaymentFrequency(Enum):
    """Payment frequency enum."""

    MONTHLY = "M"
    QUARTERLY = "Q"
    SEMIANNUAL = "S"
    ANNUAL = "A"


@dataclass
class SwapLeg:
    """Single leg of an interest rate swap."""

    notional: float
    rate: float | None  # Fixed rate (if fixed leg)
    start_date: date
    end_date: date
    payment_frequency: PaymentFrequency
    day_count: DayCountConvention
    is_fixed: bool
    spread: float = 0.0  # Spread for floating leg


@dataclass
class CashFlow:
    """Individual cash flow."""

    payment_date: date
    amount: float
    discount_factor: float
    present_value: float


def generate_schedule(
    start_date: date, end_date: date, frequency: PaymentFrequency
) -> list[date]:
    """Generate payment schedule dates.

    Args:
        start_date: Start date
        end_date: End date
        frequency: Payment frequency

    Returns:
        List of payment dates
    """
    dates = []
    current = start_date

    # Map frequency to months
    freq_map = {
        PaymentFrequency.MONTHLY: 1,
        PaymentFrequency.QUARTERLY: 3,
        PaymentFrequency.SEMIANNUAL: 6,
        PaymentFrequency.ANNUAL: 12,
    }

    months_delta = freq_map[frequency]

    while current < end_date:
        # Add months
        year = current.year
        month = current.month + months_delta
        day = current.day

        # Handle month overflow
        while month > 12:
            month -= 12
            year += 1

        # Handle day overflow (e.g., Jan 31 -> Feb 28)
        try:
            next_date = date(year, month, day)
        except ValueError:
            # Day doesn't exist in target month, use last day of month
            if month == 12:
                next_date = date(year, month, 31)
            else:
                next_date = date(year, month + 1, 1) - timedelta(days=1)

        if next_date > end_date:
            dates.append(end_date)
            break

        dates.append(next_date)
        current = next_date

    if dates[-1] != end_date:
        dates.append(end_date)

    return dates


def calculate_day_count_fraction(
    start: date, end: date, convention: DayCountConvention
) -> float:
    """Calculate day count fraction between two dates.

    Args:
        start: Start date
        end: End date
        convention: Day count convention

    Returns:
        Day count fraction
    """
    days = (end - start).days

    if convention == DayCountConvention.ACT_360:
        return days / 360.0
    elif convention == DayCountConvention.ACT_365:
        return days / 365.0
    elif convention == DayCountConvention.ACT_ACT:
        # Simplified: actual days / actual days in year
        year_days = 366 if (start.year % 4 == 0 and start.year % 100 != 0) or (start.year % 400 == 0) else 365
        return days / year_days
    elif convention == DayCountConvention.THIRTY_360:
        d1 = min(start.day, 30)
        d2 = min(end.day, 30) if d1 >= 30 else end.day
        return ((end.year - start.year) * 360 + (end.month - start.month) * 30 + (d2 - d1)) / 360.0
    else:
        raise ValueError(f"Unknown day count convention: {convention}")


def generate_cash_flows(
    leg: SwapLeg,
    discount_factors: Sequence[float],
    floating_rates: Sequence[float] | None = None,
) -> list[CashFlow]:
    """Generate cash flows for a swap leg.

    Args:
        leg: Swap leg specification
        discount_factors: Discount factors for each payment date
        floating_rates: Forward rates for floating leg (if applicable)

    Returns:
        List of cash flows with present values
    """
    schedule = generate_schedule(leg.start_date, leg.end_date, leg.payment_frequency)
    cash_flows = []

    prev_date = leg.start_date
    for i, payment_date in enumerate(schedule):
        # Calculate accrual period
        dcf = calculate_day_count_fraction(prev_date, payment_date, leg.day_count)

        # Calculate payment amount
        if leg.is_fixed:
            rate = leg.rate
        else:
            # Use forward rate for floating leg
            rate = floating_rates[i] if floating_rates else 0.0
            rate += leg.spread

        amount = leg.notional * rate * dcf

        # Apply discount factor
        df = discount_factors[i] if i < len(discount_factors) else discount_factors[-1]
        pv = amount * df

        cash_flows.append(
            CashFlow(payment_date=payment_date, amount=amount, discount_factor=df, present_value=pv)
        )

        prev_date = payment_date

    return cash_flows


@jit
def discount_factor(rate: float, time: float) -> float:
    """Calculate discount factor.

    Args:
        rate: Interest rate (annualized)
        time: Time to maturity in years

    Returns:
        Discount factor
    """
    return jnp.exp(-rate * time)


@jit
def present_value_swap_leg(
    notional: float,
    fixed_rate: float,
    year_fractions: jnp.ndarray,
    discount_factors: jnp.ndarray,
) -> float:
    """Calculate present value of fixed swap leg.

    Args:
        notional: Notional amount
        fixed_rate: Fixed rate
        year_fractions: Year fractions for each period
        discount_factors: Discount factors for each payment

    Returns:
        Present value of the leg
    """
    cash_flows = notional * fixed_rate * year_fractions
    return jnp.sum(cash_flows * discount_factors)


@jit
def swap_value(
    notional: float,
    fixed_rate: float,
    floating_rate: float,
    year_fractions: jnp.ndarray,
    discount_factors: jnp.ndarray,
    pay_fixed: bool = True,
) -> float:
    """Calculate swap value.

    Args:
        notional: Notional amount
        fixed_rate: Fixed rate on fixed leg
        floating_rate: Floating rate (or forward rate)
        year_fractions: Year fractions for each period
        discount_factors: Discount factors
        pay_fixed: True if paying fixed, receiving floating

    Returns:
        Swap value (positive = in the money)
    """
    # Fixed leg PV
    fixed_pv = present_value_swap_leg(notional, fixed_rate, year_fractions, discount_factors)

    # Floating leg PV (simplified: assuming flat forward rate)
    floating_pv = present_value_swap_leg(notional, floating_rate, year_fractions, discount_factors)

    # Swap value depends on direction - use jnp.where for JAX compatibility
    # If pay_fixed: floating_pv - fixed_pv, else: fixed_pv - floating_pv
    return jnp.where(pay_fixed, floating_pv - fixed_pv, fixed_pv - floating_pv)


def price_vanilla_swap(
    notional: float,
    fixed_rate: float,
    floating_rate: float,
    maturity: float,
    payment_frequency: int = 2,  # Semiannual
    discount_rate: float = 0.05,
    pay_fixed: bool = True,
) -> float:
    """Price a vanilla interest rate swap.

    Args:
        notional: Notional amount
        fixed_rate: Fixed rate (e.g., 0.05 for 5%)
        floating_rate: Current floating rate
        maturity: Time to maturity in years
        payment_frequency: Payments per year
        discount_rate: Discount rate for PV calculation
        pay_fixed: True if paying fixed leg

    Returns:
        Swap value

    Example:
        >>> # 5-year swap, $10M notional, 5% fixed, 4.5% floating
        >>> value = price_vanilla_swap(
        ...     notional=10_000_000,
        ...     fixed_rate=0.05,
        ...     floating_rate=0.045,
        ...     maturity=5.0,
        ...     payment_frequency=2,  # Semiannual
        ...     pay_fixed=True
        ... )
    """
    # Generate payment dates
    n_payments = int(maturity * payment_frequency)
    year_fractions = jnp.full(n_payments, 1.0 / payment_frequency, dtype=jnp.float32)

    # Generate discount factors
    payment_times = jnp.arange(1, n_payments + 1, dtype=jnp.float32) / payment_frequency
    discount_factors = jnp.exp(-discount_rate * payment_times)

    return float(
        swap_value(
            notional,
            fixed_rate,
            floating_rate,
            year_fractions,
            discount_factors,
            pay_fixed,
        )
    )


def swap_dv01(
    notional: float,
    maturity: float,
    payment_frequency: int = 2,
    discount_rate: float = 0.05,
) -> float:
    """Calculate DV01 (dollar value of 1 basis point) for a swap.

    Args:
        notional: Notional amount
        maturity: Time to maturity in years
        payment_frequency: Payments per year
        discount_rate: Discount rate

    Returns:
        DV01 (change in value for 1bp rate change)
    """
    fixed_rate = 0.05  # Reference rate
    bp = 0.0001  # 1 basis point

    pv_base = price_vanilla_swap(
        notional, fixed_rate, fixed_rate, maturity, payment_frequency, discount_rate
    )

    pv_up = price_vanilla_swap(
        notional, fixed_rate + bp, fixed_rate, maturity, payment_frequency, discount_rate
    )

    return (pv_up - pv_base) / 1.0  # Per 1bp


__all__ = [
    "SwapLeg",
    "CashFlow",
    "DayCountConvention",
    "PaymentFrequency",
    "generate_schedule",
    "calculate_day_count_fraction",
    "generate_cash_flows",
    "discount_factor",
    "present_value_swap_leg",
    "swap_value",
    "price_vanilla_swap",
    "swap_dv01",
]
