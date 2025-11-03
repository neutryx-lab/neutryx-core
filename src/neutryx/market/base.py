"""
Base protocols and interfaces for market data components.

This module defines the core abstractions used throughout the market data system:
- Curve: Abstract interface for all curve types (discount, forward, dividend, etc.)
- Surface: Abstract interface for volatility and other 2D surfaces
- Extrapolation policies for handling out-of-range queries
- Time conversion utilities
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Protocol, runtime_checkable

import jax.numpy as jnp
from jax import Array


class ExtrapolationPolicy(Enum):
    """Policy for handling extrapolation beyond curve/surface boundaries."""

    FLAT = "flat"              # Use boundary value
    LINEAR = "linear"          # Linear extrapolation from last two points
    ERROR = "error"            # Raise error on out-of-bounds
    NATURAL = "natural"        # Natural continuation (curve-specific)


@runtime_checkable
class Curve(Protocol):
    """
    Abstract protocol for all curve types.

    A Curve represents a time-dependent scalar function, such as:
    - Discount factor curves
    - Forward rate curves
    - Dividend yield curves
    - Hazard rate curves

    All concrete implementations must provide:
    - value(t): Evaluate curve at time t
    - __call__(t): Alias for value(t)

    Optional methods:
    - forward(t1, t2): Forward value between two times
    - integral(t1, t2): Integral of curve between two times
    """

    def value(self, t: float | Array) -> float | Array:
        """
        Evaluate the curve at time t.

        Args:
            t: Time(s) to evaluate (in years from reference date)

        Returns:
            Curve value(s) at time t
        """
        ...

    def __call__(self, t: float | Array) -> float | Array:
        """Convenience method: curve(t) is equivalent to curve.value(t)."""
        ...


class DiscountCurve(Curve, Protocol):
    """
    Protocol for discount factor curves.

    Extends Curve with discount-specific methods.
    """

    def df(self, t: float | Array) -> float | Array:
        """
        Compute discount factor to time t.

        Args:
            t: Time(s) in years from reference date

        Returns:
            Discount factor(s) DF(0, t)
        """
        ...

    def zero_rate(self, t: float | Array) -> float | Array:
        """
        Compute continuously-compounded zero rate to time t.

        Args:
            t: Time in years from reference date

        Returns:
            Zero rate r(t) such that DF(0,t) = exp(-r(t)*t)
        """
        ...

    def forward_rate(self, t1: float | Array, t2: float | Array) -> float | Array:
        """
        Compute continuously-compounded forward rate between t1 and t2.

        Args:
            t1: Start time in years
            t2: End time in years

        Returns:
            Forward rate f(t1,t2) such that DF(t1,t2) = exp(-f(t1,t2)*(t2-t1))
        """
        ...


@runtime_checkable
class Surface(Protocol):
    """
    Abstract protocol for 2D surfaces (e.g., volatility surfaces).

    A Surface represents a function of two variables, typically:
    - (expiry, strike) for volatility surfaces
    - (maturity, tenor) for swaption volatility cubes (2D slice)

    Implementations must provide:
    - value(x, y): Evaluate surface at point (x, y)
    - __call__(x, y): Alias for value(x, y)
    """

    def value(
        self,
        x: float | Array,
        y: float | Array
    ) -> float | Array:
        """
        Evaluate surface at point(s) (x, y).

        Args:
            x: First coordinate (e.g., expiry in years)
            y: Second coordinate (e.g., strike)

        Returns:
            Surface value(s) at (x, y)
        """
        ...

    def __call__(
        self,
        x: float | Array,
        y: float | Array
    ) -> float | Array:
        """Convenience method: surface(x, y) is equivalent to surface.value(x, y)."""
        ...


class VolatilitySurface(Surface, Protocol):
    """
    Protocol for implied volatility surfaces.

    Extends Surface with volatility-specific methods.
    """

    def implied_vol(
        self,
        expiry: float | Array,
        strike: float | Array
    ) -> float | Array:
        """
        Get implied volatility for given expiry and strike.

        Args:
            expiry: Time to expiry in years
            strike: Strike price

        Returns:
            Implied volatility (annualized, e.g., 0.20 for 20%)
        """
        ...


# Time conversion utilities

def years_from_reference(
    target_date: date | datetime,
    reference_date: date | datetime,
    day_count_basis: str = "ACT/365"
) -> float:
    """
    Compute time in years between reference date and target date.

    Args:
        target_date: Target date
        reference_date: Reference date (typically pricing date)
        day_count_basis: Day count convention (default ACT/365)

    Returns:
        Time in years from reference to target

    Note:
        For full day count convention support, see conventions.py
        This is a simplified version for basic use cases.
    """
    if isinstance(target_date, datetime):
        target_date = target_date.date()
    if isinstance(reference_date, datetime):
        reference_date = reference_date.date()

    days = (target_date - reference_date).days

    if day_count_basis == "ACT/365":
        return days / 365.0
    elif day_count_basis == "ACT/360":
        return days / 360.0
    elif day_count_basis == "ACT/ACT":
        # Simplified: actual/actual (could be more sophisticated)
        return days / 365.25
    else:
        raise ValueError(f"Unsupported day count basis: {day_count_basis}")


def date_to_time(
    target_date: date | datetime,
    reference_date: date | datetime
) -> float:
    """
    Convenience alias for years_from_reference with ACT/365.

    Args:
        target_date: Target date
        reference_date: Reference date

    Returns:
        Time in years (ACT/365)
    """
    return years_from_reference(target_date, reference_date, "ACT/365")
