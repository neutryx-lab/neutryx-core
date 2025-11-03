"""Bond pricing and analytics.

Implements various bond types with pricing, duration, and convexity calculations:
- Zero-coupon bonds
- Fixed-rate coupon bonds
- Floating-rate notes (FRNs)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import jax.numpy as jnp
from jax import jit


class DayCountConvention(Enum):
    """Day count conventions for interest calculations."""

    ACT_360 = "ACT/360"
    ACT_365 = "ACT/365"
    ACT_ACT = "ACT/ACT"
    THIRTY_360 = "30/360"


class CompoundingFrequency(Enum):
    """Compounding frequency."""

    ANNUAL = 1
    SEMIANNUAL = 2
    QUARTERLY = 4
    MONTHLY = 12
    CONTINUOUS = -1


@dataclass
class BondSpecs:
    """Bond specification parameters."""

    face_value: float = 100.0
    coupon_rate: float = 0.0  # Annual coupon rate
    maturity: float = 1.0  # Time to maturity in years
    frequency: CompoundingFrequency = CompoundingFrequency.SEMIANNUAL


@jit
def zero_coupon_bond_price(
    face_value: float, yield_rate: float, maturity: float, frequency: int = 2
) -> float:
    """Price a zero-coupon bond.

    Parameters
    ----------
    face_value : float
        Face value (par value) of the bond
    yield_rate : float
        Yield to maturity (annual rate)
    maturity : float
        Time to maturity in years
    frequency : int
        Compounding frequency per year (default: 2 for semiannual)
        Use frequency=0 for continuous compounding

    Returns
    -------
    float
        Present value of the zero-coupon bond

    Notes
    -----
    For discrete compounding:
        P = F / (1 + y/m)^(m*T)

    where F is face value, y is yield, m is frequency, T is maturity.

    For continuous compounding (frequency = 0):
        P = F * exp(-y * T)

    Examples
    --------
    >>> # $100 face value, 5% yield, 2 years, semiannual compounding
    >>> zero_coupon_bond_price(100.0, 0.05, 2.0, frequency=2)
    90.703...
    """
    # Use jnp.where for JAX compatibility (continuous vs discrete)
    continuous_price = face_value * jnp.exp(-yield_rate * maturity)
    discrete_price = face_value / jnp.power(1.0 + yield_rate / jnp.maximum(frequency, 1), frequency * maturity)

    return jnp.where(frequency == 0, continuous_price, discrete_price)


@jit
def coupon_bond_price(
    face_value: float,
    coupon_rate: float,
    yield_rate: float,
    maturity: float,
    frequency: int = 2,
) -> float:
    """Price a fixed-rate coupon bond.

    Parameters
    ----------
    face_value : float
        Face value (par value) of the bond
    coupon_rate : float
        Annual coupon rate (e.g., 0.05 for 5%)
    yield_rate : float
        Yield to maturity (annual rate)
    maturity : float
        Time to maturity in years
    frequency : int
        Coupon payment frequency per year (default: 2 for semiannual)

    Returns
    -------
    float
        Present value of the coupon bond

    Notes
    -----
    Bond price = PV(coupons) + PV(face value)

    The bond price is calculated as:
        P = C * [1 - (1+y/m)^(-m*T)] / (y/m) + F / (1+y/m)^(m*T)

    where:
    - C = coupon payment per period = (coupon_rate * face_value) / m
    - y = yield to maturity
    - m = payment frequency
    - T = maturity in years
    - F = face value

    Examples
    --------
    >>> # $100 face, 5% coupon, 4% yield, 3 years, semiannual
    >>> coupon_bond_price(100.0, 0.05, 0.04, 3.0, frequency=2)
    102.775...
    """
    # Calculate coupon payment per period
    coupon_payment = (coupon_rate * face_value) / frequency

    # Number of periods
    n_periods = maturity * frequency

    # Discount rate per period
    discount_rate = yield_rate / frequency

    # Present value of coupon payments (annuity)
    # PV = C * [1 - (1+r)^(-n)] / r
    pv_coupons = coupon_payment * (1.0 - jnp.power(1.0 + discount_rate, -n_periods)) / discount_rate

    # Present value of face value
    pv_face = face_value / jnp.power(1.0 + discount_rate, n_periods)

    return pv_coupons + pv_face


def bond_yield_to_maturity(
    price: float,
    face_value: float,
    coupon_rate: float,
    maturity: float,
    frequency: int = 2,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> float:
    """Calculate yield to maturity using Newton-Raphson method.

    Parameters
    ----------
    price : float
        Current market price of the bond
    face_value : float
        Face value of the bond
    coupon_rate : float
        Annual coupon rate
    maturity : float
        Time to maturity in years
    frequency : int
        Payment frequency per year
    max_iterations : int
        Maximum iterations for Newton-Raphson
    tolerance : float
        Convergence tolerance

    Returns
    -------
    float
        Yield to maturity (annual rate)

    Notes
    -----
    Uses Newton-Raphson iteration to solve for yield y such that:
        price = bond_price(face_value, coupon_rate, y, maturity, frequency)

    Examples
    --------
    >>> # Bond trading at $95, face $100, 4% coupon, 5 years
    >>> bond_yield_to_maturity(95.0, 100.0, 0.04, 5.0, frequency=2)
    0.0498...
    """
    # Use bisection method for robustness
    # Initial bounds for yield search
    y_low = -0.1  # Allow slightly negative rates
    y_high = 1.0  # 100% yield as upper bound

    # Evaluate prices at bounds
    price_low = float(coupon_bond_price(face_value, coupon_rate, y_low, maturity, frequency))
    price_high = float(coupon_bond_price(face_value, coupon_rate, y_high, maturity, frequency))

    # Check if solution exists in range
    if (price_low - price) * (price_high - price) > 0:
        # Fall back to initial guess
        return float(coupon_rate + (face_value - price) / (face_value * maturity))

    # Bisection method
    for _ in range(max_iterations):
        ytm = (y_low + y_high) / 2.0
        calculated_price = float(coupon_bond_price(face_value, coupon_rate, ytm, maturity, frequency))

        price_diff = calculated_price - price

        # Check convergence
        if abs(price_diff) < tolerance:
            break

        # Update bounds
        if price_diff > 0:  # Calculated price too high, yield too low
            y_low = ytm
        else:  # Calculated price too low, yield too high
            y_high = ytm

        # Check if interval is too small
        if abs(y_high - y_low) < tolerance / 1000:
            break

    return ytm


def macaulay_duration(
    face_value: float, coupon_rate: float, yield_rate: float, maturity: float, frequency: int = 2
) -> float:
    """Calculate Macaulay duration of a bond.

    Parameters
    ----------
    face_value : float
        Face value of the bond
    coupon_rate : float
        Annual coupon rate
    yield_rate : float
        Yield to maturity
    maturity : float
        Time to maturity in years
    frequency : int
        Payment frequency per year

    Returns
    -------
    float
        Macaulay duration in years

    Notes
    -----
    Macaulay duration is the weighted average time to receive cash flows:
        D = Σ(t * PV(CF_t)) / Price

    Examples
    --------
    >>> macaulay_duration(100.0, 0.05, 0.05, 5.0, frequency=2)
    4.376...
    """
    coupon_payment = (coupon_rate * face_value) / frequency
    discount_rate = yield_rate / frequency
    n_periods = int(maturity * frequency)

    # Calculate present value of each cash flow weighted by time
    times = jnp.arange(1, n_periods + 1, dtype=jnp.float32)
    discount_factors = jnp.power(1.0 + discount_rate, -times)

    # Coupon payments at each period
    cash_flows = jnp.full(n_periods, coupon_payment, dtype=jnp.float32)
    # Add face value to last payment
    cash_flows = cash_flows.at[-1].add(face_value)

    # Weighted present values
    weighted_pv = jnp.sum(times * cash_flows * discount_factors / frequency)

    # Bond price
    bond_price = coupon_bond_price(face_value, coupon_rate, yield_rate, maturity, frequency)

    return weighted_pv / bond_price


def modified_duration(
    face_value: float, coupon_rate: float, yield_rate: float, maturity: float, frequency: int = 2
) -> float:
    """Calculate modified duration of a bond.

    Parameters
    ----------
    face_value : float
        Face value of the bond
    coupon_rate : float
        Annual coupon rate
    yield_rate : float
        Yield to maturity
    maturity : float
        Time to maturity in years
    frequency : int
        Payment frequency per year

    Returns
    -------
    float
        Modified duration

    Notes
    -----
    Modified duration measures price sensitivity to yield changes:
        ModD = MacD / (1 + y/m)

    Price change approximation:
        ΔP/P ≈ -ModD * Δy

    Examples
    --------
    >>> modified_duration(100.0, 0.05, 0.05, 5.0, frequency=2)
    4.268...
    """
    mac_dur = macaulay_duration(face_value, coupon_rate, yield_rate, maturity, frequency)
    return mac_dur / (1.0 + yield_rate / frequency)


def convexity(
    face_value: float, coupon_rate: float, yield_rate: float, maturity: float, frequency: int = 2
) -> float:
    """Calculate convexity of a bond.

    Parameters
    ----------
    face_value : float
        Face value of the bond
    coupon_rate : float
        Annual coupon rate
    yield_rate : float
        Yield to maturity
    maturity : float
        Time to maturity in years
    frequency : int
        Payment frequency per year

    Returns
    -------
    float
        Convexity measure

    Notes
    -----
    Convexity measures the curvature of the price-yield relationship:
        Convexity = Σ[t(t+1) * PV(CF_t)] / [P * (1+y/m)²]

    Better price change approximation:
        ΔP/P ≈ -ModD * Δy + 0.5 * Convexity * (Δy)²

    Examples
    --------
    >>> convexity(100.0, 0.05, 0.05, 5.0, frequency=2)
    22.96...
    """
    coupon_payment = (coupon_rate * face_value) / frequency
    discount_rate = yield_rate / frequency
    n_periods = int(maturity * frequency)

    # Calculate present value of each cash flow weighted by t(t+1)
    times = jnp.arange(1, n_periods + 1, dtype=jnp.float32)
    discount_factors = jnp.power(1.0 + discount_rate, -times)

    # Coupon payments
    cash_flows = jnp.full(n_periods, coupon_payment, dtype=jnp.float32)
    cash_flows = cash_flows.at[-1].add(face_value)

    # Weight by t(t+1)
    weights = times * (times + 1.0)
    weighted_pv = jnp.sum(weights * cash_flows * discount_factors)

    # Bond price
    bond_price = coupon_bond_price(face_value, coupon_rate, yield_rate, maturity, frequency)

    # Convexity formula
    return weighted_pv / (bond_price * frequency * frequency * (1.0 + discount_rate) ** 2)


def floating_rate_note_price(
    face_value: float,
    reference_rate: float,
    spread: float,
    discount_rate: float,
    maturity: float,
    frequency: int = 4,
) -> float:
    """Price a floating rate note (FRN).

    Parameters
    ----------
    face_value : float
        Face value of the FRN
    reference_rate : float
        Current reference rate (e.g., LIBOR)
    spread : float
        Credit spread over reference rate
    discount_rate : float
        Discount rate for valuation
    maturity : float
        Time to maturity in years
    frequency : int
        Reset frequency per year (default: 4 for quarterly)

    Returns
    -------
    float
        Present value of the FRN

    Notes
    -----
    At each reset date, the coupon rate = reference_rate + spread.
    For a newly issued FRN at par, if spread = 0 and reference_rate = discount_rate,
    the FRN trades at par (face value).

    Examples
    --------
    >>> # $100 face, 3% reference rate, 0.5% spread, 2 years
    >>> floating_rate_note_price(100.0, 0.03, 0.005, 0.035, 2.0, frequency=4)
    99.01...
    """
    # Coupon rate for next period
    coupon_rate = reference_rate + spread
    coupon_payment = (coupon_rate * face_value) / frequency

    n_periods = maturity * frequency
    discount_per_period = discount_rate / frequency

    # Generate discount factors
    times = jnp.arange(1, n_periods + 1, dtype=jnp.float32)
    discount_factors = jnp.power(1.0 + discount_per_period, -times)

    # Present value of coupons
    pv_coupons = coupon_payment * jnp.sum(discount_factors)

    # Present value of face value
    pv_face = face_value * discount_factors[-1]

    return pv_coupons + pv_face


def bond_price_from_curve(
    face_value: float,
    coupon_rate: float,
    maturity: float,
    frequency: int,
    discount_curve: Callable[[float], float],
) -> float:
    """Price a bond using a discount curve function.

    Parameters
    ----------
    face_value : float
        Face value of the bond
    coupon_rate : float
        Annual coupon rate
    maturity : float
        Time to maturity in years
    frequency : int
        Payment frequency per year
    discount_curve : Callable[[float], float]
        Function that returns discount factor for a given time

    Returns
    -------
    float
        Bond price

    Notes
    -----
    This allows pricing bonds with a full term structure rather than
    a single flat yield rate.

    Examples
    --------
    >>> # Define a flat discount curve
    >>> def flat_curve(t):
    ...     return jnp.exp(-0.05 * t)
    >>> bond_price_from_curve(100.0, 0.06, 3.0, 2, flat_curve)
    102.78...
    """
    coupon_payment = (coupon_rate * face_value) / frequency
    n_periods = int(maturity * frequency)

    # Payment times
    payment_times = jnp.arange(1, n_periods + 1, dtype=jnp.float32) / frequency

    # Get discount factors from curve by evaluating at each time
    discount_factors = jnp.array([float(discount_curve(float(t))) for t in payment_times])

    # Present value of coupons
    pv_coupons = coupon_payment * jnp.sum(discount_factors)

    # Present value of face value
    pv_face = face_value * discount_factors[-1]

    return float(pv_coupons + pv_face)


__all__ = [
    "BondSpecs",
    "CompoundingFrequency",
    "DayCountConvention",
    "bond_price_from_curve",
    "bond_yield_to_maturity",
    "convexity",
    "coupon_bond_price",
    "floating_rate_note_price",
    "macaulay_duration",
    "modified_duration",
    "zero_coupon_bond_price",
]
