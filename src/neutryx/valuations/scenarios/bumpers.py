"""
Market data bumpers for scenario analysis and risk management.

This module provides tools for applying systematic shocks to market data:
- Curve bumpers (parallel shifts, bucket shifts, twists)
- Surface bumpers (vega buckets, smile rotation, term structure shifts)
- MarketScenario class for managing complete market scenarios
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Dict, Optional, Tuple

import jax.numpy as jnp
from jax import Array

from neutryx.market.base import Curve, DiscountCurve, Surface, VolatilitySurface
from neutryx.market.curves import BootstrappedCurve, DividendYieldCurve, FlatCurve, ForwardRateCurve
from neutryx.market.environment import MarketDataEnvironment
from neutryx.market.vol import ImpliedVolSurface, SABRSurface


class BumpType(Enum):
    """Type of bump to apply to market data."""

    ABSOLUTE = "absolute"  # Add bump value
    RELATIVE = "relative"  # Multiply by (1 + bump value)
    REPLACE = "replace"    # Replace with bump value


@dataclass(frozen=True)
class CurveBump:
    """
    Specification for a curve bump (shock).

    Attributes:
        bump_type: Type of bump (absolute, relative, replace)
        bump_value: Size of bump (e.g., 0.01 for 1% absolute, 0.10 for 10% relative)
        bucket_start: Start time for bucket bump (None for parallel)
        bucket_end: End time for bucket bump (None for parallel)
        interpolation: How to interpolate bump between buckets
    """

    bump_type: BumpType = BumpType.ABSOLUTE
    bump_value: float = 0.01
    bucket_start: Optional[float] = None
    bucket_end: Optional[float] = None
    interpolation: str = "flat"  # "flat", "linear", "smooth"


class CurveBumper:
    """
    Apply bumps (shocks) to interest rate curves.

    Supports:
    - Parallel shifts (entire curve)
    - Bucket shifts (specific maturity ranges)
    - Twist shifts (steepener/flattener)
    """

    @staticmethod
    def parallel_shift(
        curve: DiscountCurve,
        shift_bps: float,
        shift_type: str = "zero_rate"
    ) -> DiscountCurve:
        """
        Apply parallel shift to curve.

        Args:
            curve: Original curve
            shift_bps: Shift in basis points (e.g., 10 for +10bps)
            shift_type: What to shift ("zero_rate", "forward_rate")

        Returns:
            Shifted curve

        Example:
            >>> from neutryx.market import FlatCurve
            >>> curve = FlatCurve(0.05)
            >>> shifted = CurveBumper.parallel_shift(curve, 10)  # +10bps
            >>> shifted.zero_rate(1.0)  # 0.051
        """
        shift_decimal = shift_bps / 10000.0

        if isinstance(curve, FlatCurve):
            # For flat curve, just shift the rate
            return FlatCurve(r=curve.r + shift_decimal)

        elif isinstance(curve, BootstrappedCurve):
            # For bootstrapped curve, shift zero rates at each node
            new_nodes = {}
            for t, df in curve._nodes.items():
                if t == 0:
                    new_nodes[t] = 1.0
                else:
                    # Convert to zero rate, shift, convert back to DF
                    zero_rate = -jnp.log(df) / t
                    new_zero = zero_rate + shift_decimal
                    new_nodes[t] = float(jnp.exp(-new_zero * t))

            # Rebuild curve with new nodes
            from neutryx.market.curves import Deposit
            # Create deposits from nodes
            instruments = [
                Deposit(maturity=t, rate=(1.0/df - 1.0)/t)
                for t, df in sorted(new_nodes.items()) if t > 0
            ]
            return BootstrappedCurve(instruments)

        else:
            # Generic curve: wrap with shift
            return _ShiftedCurve(curve, shift_decimal)

    @staticmethod
    def bucket_shift(
        curve: DiscountCurve,
        bucket_start: float,
        bucket_end: float,
        shift_bps: float
    ) -> DiscountCurve:
        """
        Apply bucket shift (shock specific maturity range).

        Args:
            curve: Original curve
            bucket_start: Bucket start time (years)
            bucket_end: Bucket end time (years)
            shift_bps: Shift in basis points

        Returns:
            Shifted curve

        Example:
            >>> curve = FlatCurve(0.05)
            >>> # Shift 1Y-5Y bucket by +10bps
            >>> shifted = CurveBumper.bucket_shift(curve, 1.0, 5.0, 10)
        """
        shift_decimal = shift_bps / 10000.0
        return _BucketShiftedCurve(
            curve,
            bucket_start,
            bucket_end,
            shift_decimal
        )

    @staticmethod
    def twist(
        curve: DiscountCurve,
        pivot_point: float,
        short_shift_bps: float,
        long_shift_bps: float
    ) -> DiscountCurve:
        """
        Apply twist (steepener/flattener).

        Args:
            curve: Original curve
            pivot_point: Pivot time (no shift at this point)
            short_shift_bps: Shift for t=0
            long_shift_bps: Shift for t=infinity

        Returns:
            Twisted curve

        Example:
            >>> curve = FlatCurve(0.05)
            >>> # Steepener: short end -10bps, long end +10bps, pivot at 5Y
            >>> twisted = CurveBumper.twist(curve, 5.0, -10, 10)
        """
        short_shift = short_shift_bps / 10000.0
        long_shift = long_shift_bps / 10000.0
        return _TwistedCurve(curve, pivot_point, short_shift, long_shift)


class SurfaceBumper:
    """
    Apply bumps (shocks) to volatility surfaces.

    Supports:
    - Parallel vol shifts
    - Vega bucket shifts (specific strike/expiry ranges)
    - Smile rotation (skew adjustments)
    - Term structure shifts
    """

    @staticmethod
    def parallel_shift(
        surface: VolatilitySurface,
        shift_vol: float
    ) -> VolatilitySurface:
        """
        Apply parallel shift to entire volatility surface.

        Args:
            surface: Original surface
            shift_vol: Volatility shift (e.g., 0.01 for +1% vol)

        Returns:
            Shifted surface

        Example:
            >>> from neutryx.market import ImpliedVolSurface
            >>> surf = ImpliedVolSurface(
            ...     expiries=[0.25, 0.5, 1.0],
            ...     strikes=[90, 100, 110],
            ...     vols=[[0.20, 0.18, 0.20]] * 3
            ... )
            >>> shifted = SurfaceBumper.parallel_shift(surf, 0.01)  # +1% vol
        """
        if isinstance(surface, ImpliedVolSurface):
            # Shift grid-based surface
            vols_array = jnp.array(surface.vols)
            shifted_vols = vols_array + shift_vol
            return ImpliedVolSurface(
                expiries=surface.expiries,
                strikes=surface.strikes,
                vols=shifted_vols.tolist()
            )
        elif isinstance(surface, SABRSurface):
            # For SABR, shift alpha parameter
            from neutryx.market.vol import SABRParameters
            new_params = []
            for p in surface.params:
                # Approximate: shift alpha proportionally to achieve vol shift
                new_alpha = p.alpha * (1.0 + shift_vol / 0.20)  # Rough approximation
                new_params.append(SABRParameters(
                    alpha=new_alpha,
                    beta=p.beta,
                    rho=p.rho,
                    nu=p.nu
                ))
            return SABRSurface(
                expiries=surface.expiries,
                forwards=surface.forwards,
                params=new_params
            )
        else:
            # Generic surface: wrap with shift
            return _ShiftedSurface(surface, shift_vol)

    @staticmethod
    def vega_bucket_shift(
        surface: VolatilitySurface,
        expiry_start: float,
        expiry_end: float,
        strike_start: float,
        strike_end: float,
        shift_vol: float
    ) -> VolatilitySurface:
        """
        Apply vega bucket shift (specific expiry/strike range).

        Args:
            surface: Original surface
            expiry_start: Bucket start expiry
            expiry_end: Bucket end expiry
            strike_start: Bucket start strike
            strike_end: Bucket end strike
            shift_vol: Volatility shift

        Returns:
            Shifted surface
        """
        return _VegaBucketSurface(
            surface,
            expiry_start,
            expiry_end,
            strike_start,
            strike_end,
            shift_vol
        )

    @staticmethod
    def smile_rotation(
        surface: VolatilitySurface,
        expiry: float,
        atm_strike: float,
        rotation_rate: float
    ) -> VolatilitySurface:
        """
        Rotate smile (adjust skew) at specific expiry.

        Args:
            surface: Original surface
            expiry: Expiry to rotate
            atm_strike: ATM strike (pivot point)
            rotation_rate: Rotation in vol per strike unit

        Returns:
            Rotated surface
        """
        return _SmileRotatedSurface(surface, expiry, atm_strike, rotation_rate)


@dataclass(frozen=True)
class MarketScenario:
    """
    Complete market scenario with shocked market data.

    Represents a specific risk scenario by storing both the base environment
    and the shocks to apply.

    Attributes:
        name: Scenario name (e.g., "rates_up_100bps", "vol_shock_plus_10pct")
        description: Human-readable description
        curve_shocks: Dict mapping (curve_type, identifier) to shock function
        surface_shocks: Dict mapping (surface_type, identifier) to shock function
        fx_shocks: Dict mapping FX pair to spot shift

    Example:
        >>> scenario = MarketScenario(
        ...     name="rates_up_50bps",
        ...     description="Parallel 50bps rate shock",
        ...     curve_shocks={
        ...         ('discount', 'USD'): lambda c: CurveBumper.parallel_shift(c, 50),
        ...         ('discount', 'EUR'): lambda c: CurveBumper.parallel_shift(c, 50),
        ...     }
        ... )
        >>> shocked_env = scenario.apply(base_env)
    """

    name: str
    description: str = ""
    curve_shocks: Dict[Tuple[str, str], Callable] = None
    surface_shocks: Dict[Tuple[str, str], Callable] = None
    fx_shocks: Dict[Tuple[str, str], float] = None

    def __post_init__(self):
        """Initialize empty dicts if None."""
        if self.curve_shocks is None:
            object.__setattr__(self, "curve_shocks", {})
        if self.surface_shocks is None:
            object.__setattr__(self, "surface_shocks", {})
        if self.fx_shocks is None:
            object.__setattr__(self, "fx_shocks", {})

    def apply(self, base_env: MarketDataEnvironment) -> MarketDataEnvironment:
        """
        Apply scenario shocks to base environment.

        Args:
            base_env: Base market data environment

        Returns:
            Shocked environment
        """
        env = base_env

        # Apply discount curve shocks
        for (curve_type, currency), shock_fn in self.curve_shocks.items():
            if curve_type == 'discount':
                if currency in env.discount_curves:
                    original_curve = env.discount_curves[currency]
                    shocked_curve = shock_fn(original_curve)
                    env = env.with_discount_curve(currency, shocked_curve)
            elif curve_type == 'dividend':
                if currency in env.dividend_curves:
                    original_curve = env.dividend_curves[currency]
                    shocked_curve = shock_fn(original_curve)
                    env = env.with_dividend_curve(currency, shocked_curve)

        # Apply volatility surface shocks
        for (surface_type, underlier), shock_fn in self.surface_shocks.items():
            if surface_type == 'vol':
                if underlier in env.vol_surfaces:
                    original_surface = env.vol_surfaces[underlier]
                    shocked_surface = shock_fn(original_surface)
                    env = env.with_vol_surface(underlier, shocked_surface)

        # Apply FX spot shocks
        for (from_ccy, to_ccy), shift in self.fx_shocks.items():
            pair = (from_ccy, to_ccy)
            if pair in env.fx_spots:
                original_spot = env.fx_spots[pair]
                shocked_spot = original_spot * (1.0 + shift)
                env = env.with_fx_spot(from_ccy, to_ccy, shocked_spot)

        # Add metadata about scenario
        env = env.with_metadata('scenario_name', self.name)
        env = env.with_metadata('scenario_description', self.description)

        return env

    def __repr__(self) -> str:
        n_curve = len(self.curve_shocks)
        n_surf = len(self.surface_shocks)
        n_fx = len(self.fx_shocks)
        return (
            f"MarketScenario(name='{self.name}', "
            f"curves={n_curve}, surfaces={n_surf}, fx={n_fx})"
        )


# Helper classes for wrapped curves/surfaces with shocks

class _ShiftedCurve:
    """Curve wrapper that applies constant shift to zero rates."""

    def __init__(self, base_curve: Curve, shift: float):
        self.base_curve = base_curve
        self.shift = shift

    def df(self, t: float | Array) -> float | Array:
        """Compute shifted discount factor."""
        base_df = self.base_curve.value(t)
        t_arr = jnp.asarray(t)
        # Convert to zero rate, shift, convert back
        zero_rate = -jnp.log(base_df) / t_arr
        shifted_zero = zero_rate + self.shift
        return jnp.exp(-shifted_zero * t_arr)

    def value(self, t: float | Array) -> float | Array:
        return self.df(t)

    def __call__(self, t: float | Array) -> float | Array:
        return self.df(t)

    def zero_rate(self, t: float | Array) -> float | Array:
        base_zero = self.base_curve.zero_rate(t) if hasattr(self.base_curve, 'zero_rate') else -jnp.log(self.base_curve.value(t)) / t
        return base_zero + self.shift

    def forward_rate(self, t1: float | Array, t2: float | Array) -> float | Array:
        # Approximate
        df1 = self.df(t1)
        df2 = self.df(t2)
        return -jnp.log(df2 / df1) / (t2 - t1)


class _BucketShiftedCurve:
    """Curve wrapper that applies shift to specific maturity bucket."""

    def __init__(self, base_curve: Curve, bucket_start: float, bucket_end: float, shift: float):
        self.base_curve = base_curve
        self.bucket_start = bucket_start
        self.bucket_end = bucket_end
        self.shift = shift

    def _bucket_weight(self, t: float | Array) -> float | Array:
        """Weight function for bucket (1 inside bucket, 0 outside)."""
        t_arr = jnp.asarray(t)
        weight = jnp.where(
            (t_arr >= self.bucket_start) & (t_arr <= self.bucket_end),
            1.0,
            0.0
        )
        return weight

    def df(self, t: float | Array) -> float | Array:
        base_df = self.base_curve.value(t)
        t_arr = jnp.asarray(t)
        weight = self._bucket_weight(t_arr)

        # Apply shift only in bucket
        zero_rate = -jnp.log(base_df) / t_arr
        shifted_zero = zero_rate + self.shift * weight
        return jnp.exp(-shifted_zero * t_arr)

    def value(self, t: float | Array) -> float | Array:
        return self.df(t)

    def __call__(self, t: float | Array) -> float | Array:
        return self.df(t)

    def zero_rate(self, t: float | Array) -> float | Array:
        t_arr = jnp.asarray(t)
        weight = self._bucket_weight(t_arr)
        base_zero = -jnp.log(self.base_curve.value(t_arr)) / t_arr
        return base_zero + self.shift * weight

    def forward_rate(self, t1: float | Array, t2: float | Array) -> float | Array:
        df1 = self.df(t1)
        df2 = self.df(t2)
        return -jnp.log(df2 / df1) / (jnp.asarray(t2) - jnp.asarray(t1))


class _TwistedCurve:
    """Curve wrapper that applies linear twist."""

    def __init__(self, base_curve: Curve, pivot: float, short_shift: float, long_shift: float):
        self.base_curve = base_curve
        self.pivot = pivot
        self.short_shift = short_shift
        self.long_shift = long_shift

    def _twist_shift(self, t: float | Array) -> float | Array:
        """Compute twist shift at time t."""
        t_arr = jnp.asarray(t)
        # Linear interpolation between short and long
        if t_arr.ndim == 0:
            if t_arr <= self.pivot:
                alpha = float(t_arr) / self.pivot
                return self.short_shift * (1 - alpha)
            else:
                return self.long_shift
        else:
            shift = jnp.where(
                t_arr <= self.pivot,
                self.short_shift * (1 - t_arr / self.pivot),
                self.long_shift
            )
            return shift

    def df(self, t: float | Array) -> float | Array:
        base_df = self.base_curve.value(t)
        t_arr = jnp.asarray(t)
        twist = self._twist_shift(t_arr)

        zero_rate = -jnp.log(base_df) / t_arr
        twisted_zero = zero_rate + twist
        return jnp.exp(-twisted_zero * t_arr)

    def value(self, t: float | Array) -> float | Array:
        return self.df(t)

    def __call__(self, t: float | Array) -> float | Array:
        return self.df(t)

    def zero_rate(self, t: float | Array) -> float | Array:
        t_arr = jnp.asarray(t)
        base_zero = -jnp.log(self.base_curve.value(t_arr)) / t_arr
        return base_zero + self._twist_shift(t_arr)

    def forward_rate(self, t1: float | Array, t2: float | Array) -> float | Array:
        df1 = self.df(t1)
        df2 = self.df(t2)
        return -jnp.log(df2 / df1) / (jnp.asarray(t2) - jnp.asarray(t1))


class _ShiftedSurface:
    """Surface wrapper that applies constant shift."""

    def __init__(self, base_surface: Surface, shift: float):
        self.base_surface = base_surface
        self.shift = shift

    def implied_vol(self, expiry: float | Array, strike: float | Array) -> float | Array:
        base_vol = self.base_surface.value(expiry, strike)
        return base_vol + self.shift

    def value(self, expiry: float | Array, strike: float | Array) -> float | Array:
        return self.implied_vol(expiry, strike)

    def __call__(self, expiry: float | Array, strike: float | Array) -> float | Array:
        return self.implied_vol(expiry, strike)


class _VegaBucketSurface:
    """Surface wrapper that applies shift to specific expiry/strike bucket."""

    def __init__(
        self,
        base_surface: Surface,
        expiry_start: float,
        expiry_end: float,
        strike_start: float,
        strike_end: float,
        shift: float
    ):
        self.base_surface = base_surface
        self.expiry_start = expiry_start
        self.expiry_end = expiry_end
        self.strike_start = strike_start
        self.strike_end = strike_end
        self.shift = shift

    def _bucket_weight(self, expiry: float | Array, strike: float | Array) -> float | Array:
        """Weight function for vega bucket."""
        exp_arr = jnp.asarray(expiry)
        strike_arr = jnp.asarray(strike)

        in_bucket = (
            (exp_arr >= self.expiry_start) & (exp_arr <= self.expiry_end) &
            (strike_arr >= self.strike_start) & (strike_arr <= self.strike_end)
        )
        return jnp.where(in_bucket, 1.0, 0.0)

    def implied_vol(self, expiry: float | Array, strike: float | Array) -> float | Array:
        base_vol = self.base_surface.value(expiry, strike)
        weight = self._bucket_weight(expiry, strike)
        return base_vol + self.shift * weight

    def value(self, expiry: float | Array, strike: float | Array) -> float | Array:
        return self.implied_vol(expiry, strike)

    def __call__(self, expiry: float | Array, strike: float | Array) -> float | Array:
        return self.implied_vol(expiry, strike)


class _SmileRotatedSurface:
    """Surface wrapper that rotates smile at specific expiry."""

    def __init__(
        self,
        base_surface: Surface,
        target_expiry: float,
        atm_strike: float,
        rotation_rate: float
    ):
        self.base_surface = base_surface
        self.target_expiry = target_expiry
        self.atm_strike = atm_strike
        self.rotation_rate = rotation_rate

    def implied_vol(self, expiry: float | Array, strike: float | Array) -> float | Array:
        base_vol = self.base_surface.value(expiry, strike)

        exp_arr = jnp.asarray(expiry)
        strike_arr = jnp.asarray(strike)

        # Apply rotation only at target expiry
        is_target = jnp.abs(exp_arr - self.target_expiry) < 0.01  # Tolerance
        moneyness = strike_arr - self.atm_strike
        rotation_shift = self.rotation_rate * moneyness

        return jnp.where(is_target, base_vol + rotation_shift, base_vol)

    def value(self, expiry: float | Array, strike: float | Array) -> float | Array:
        return self.implied_vol(expiry, strike)

    def __call__(self, expiry: float | Array, strike: float | Array) -> float | Array:
        return self.implied_vol(expiry, strike)
