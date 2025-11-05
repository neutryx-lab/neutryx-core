"""Interest rate caps, floors, and collars.

Caps, floors, and collars are portfolios of options on interest rates:
- Cap: Portfolio of caplets (call options on interest rates)
- Floor: Portfolio of floorlets (put options on interest rates)
- Collar: Long cap + short floor (or vice versa)

Each caplet/floorlet has a payoff based on the difference between
a reference rate and a strike rate, applied to a notional amount.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import jax.numpy as jnp
from jax import jit, vmap

from ..base import Product

Array = jnp.ndarray


class CapFloorType(Enum):
    """Type of cap/floor instrument."""

    CAP = "cap"
    FLOOR = "floor"
    COLLAR = "collar"


@dataclass
class InterestRateCapFloorCollar(Product):
    """Interest rate cap, floor, or collar.

    A cap is a portfolio of caplets that protects against rising rates.
    A floor is a portfolio of floorlets that protects against falling rates.
    A collar combines a long cap and short floor (or vice versa).

    Pricing uses the Black model for each caplet/floorlet.

    Attributes:
        T: Maturity in years
        notional: Notional principal amount
        strike: Strike rate (cap rate or floor rate)
        cap_floor_type: CAP, FLOOR, or COLLAR
        payment_frequency: Payments per year (e.g., 4 for quarterly)
        volatility: Black volatility for the reference rate
        forward_rates: Array of forward rates for each period
        discount_rates: Array of discount rates for PV calculation
        collar_floor_strike: Floor strike for collar (if collar type)
    """

    notional: float
    strike: float
    cap_floor_type: CapFloorType = CapFloorType.CAP
    payment_frequency: int = 4  # Quarterly
    volatility: float = 0.20  # 20% vol default
    forward_rates: Array | None = None
    discount_rates: Array | None = None
    collar_floor_strike: float | None = None

    def __post_init__(self):
        """Initialize forward and discount rates if not provided."""
        n_payments = int(self.T * self.payment_frequency)
        if self.forward_rates is None:
            # Default to flat forward curve at strike
            self.forward_rates = jnp.full(n_payments, self.strike)
        if self.discount_rates is None:
            # Default to flat discount curve
            self.discount_rates = jnp.full(n_payments, self.strike)

    @property
    def requires_path(self) -> bool:
        """Caps/floors can be priced with Black model (no path needed)."""
        return False

    def payoff_terminal(self, spot: Array) -> Array:
        """Calculate cap/floor present value using Black model.

        Args:
            spot: Not used (we use forward_rates instead)

        Returns:
            Present value of cap/floor/collar
        """
        if self.cap_floor_type == CapFloorType.CAP:
            return self._price_cap()
        elif self.cap_floor_type == CapFloorType.FLOOR:
            return self._price_floor()
        else:  # COLLAR
            return self._price_collar()

    def _price_cap(self) -> float:
        """Price interest rate cap using Black model."""
        n_payments = int(self.T * self.payment_frequency)
        period_length = 1.0 / self.payment_frequency

        total_value = 0.0

        for i in range(n_payments):
            # Time to reset (start of period i)
            time_to_reset = (i + 1) * period_length

            # Forward rate for this period
            forward_rate = self.forward_rates[i]

            # Discount factor to payment date
            discount_factor = jnp.exp(-self.discount_rates[i] * time_to_reset)

            # Black model for caplet
            caplet_value = black_caplet(
                notional=self.notional,
                strike=self.strike,
                forward_rate=forward_rate,
                volatility=self.volatility,
                time_to_reset=time_to_reset,
                period_length=period_length,
                discount_factor=discount_factor,
            )

            total_value += caplet_value

        return total_value

    def _price_floor(self) -> float:
        """Price interest rate floor using Black model."""
        n_payments = int(self.T * self.payment_frequency)
        period_length = 1.0 / self.payment_frequency

        total_value = 0.0

        for i in range(n_payments):
            time_to_reset = (i + 1) * period_length
            forward_rate = self.forward_rates[i]
            discount_factor = jnp.exp(-self.discount_rates[i] * time_to_reset)

            # Black model for floorlet
            floorlet_value = black_floorlet(
                notional=self.notional,
                strike=self.strike,
                forward_rate=forward_rate,
                volatility=self.volatility,
                time_to_reset=time_to_reset,
                period_length=period_length,
                discount_factor=discount_factor,
            )

            total_value += floorlet_value

        return total_value

    def _price_collar(self) -> float:
        """Price interest rate collar (long cap, short floor)."""
        if self.collar_floor_strike is None:
            raise ValueError("collar_floor_strike must be set for collar pricing")

        # Price cap
        cap_value = self._price_cap()

        # Price floor with collar floor strike
        n_payments = int(self.T * self.payment_frequency)
        period_length = 1.0 / self.payment_frequency

        floor_value = 0.0
        for i in range(n_payments):
            time_to_reset = (i + 1) * period_length
            forward_rate = self.forward_rates[i]
            discount_factor = jnp.exp(-self.discount_rates[i] * time_to_reset)

            floorlet_value = black_floorlet(
                notional=self.notional,
                strike=self.collar_floor_strike,
                forward_rate=forward_rate,
                volatility=self.volatility,
                time_to_reset=time_to_reset,
                period_length=period_length,
                discount_factor=discount_factor,
            )

            floor_value += floorlet_value

        # Collar = long cap - short floor
        return cap_value - floor_value

    def vega(self) -> float:
        """Calculate vega (sensitivity to 1% change in volatility).

        Returns:
            Change in value for 1% absolute change in volatility
        """
        vol_shock = 0.01  # 1% volatility change

        pv_base = self.payoff_terminal(0.0)

        # Create copy with shocked volatility
        shocked_vol = self.volatility + vol_shock
        original_vol = self.volatility

        # Temporarily modify volatility
        self.volatility = shocked_vol
        pv_up = self.payoff_terminal(0.0)
        self.volatility = original_vol

        return float(pv_up - pv_base)


@jit
def black_caplet(
    notional: float,
    strike: float,
    forward_rate: float,
    volatility: float,
    time_to_reset: float,
    period_length: float,
    discount_factor: float,
) -> float:
    """Price a single caplet using Black's model.

    A caplet pays: Notional × max(Rate - Strike, 0) × Period_Length

    Args:
        notional: Notional principal
        strike: Strike rate (cap rate)
        forward_rate: Forward rate for the period
        volatility: Black volatility
        time_to_reset: Time to rate fixing
        period_length: Length of interest period
        discount_factor: Discount factor to payment date

    Returns:
        Present value of caplet
    """
    from jax.scipy.stats import norm

    # Intrinsic value for zero volatility case
    intrinsic = notional * jnp.maximum(forward_rate - strike, 0.0) * period_length * discount_factor

    # Black model calculation
    sqrt_t = jnp.sqrt(jnp.maximum(time_to_reset, 1e-10))
    vol_safe = jnp.maximum(volatility, 1e-10)

    d1 = (jnp.log(forward_rate / strike) + 0.5 * vol_safe**2 * time_to_reset) / (
        vol_safe * sqrt_t
    )
    d2 = d1 - vol_safe * sqrt_t

    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)

    black_value = notional * period_length * discount_factor * (
        forward_rate * N_d1 - strike * N_d2
    )

    # Use intrinsic value if volatility or time is too small
    use_intrinsic = (volatility < 1e-10) | (time_to_reset < 1e-10)
    return jnp.where(use_intrinsic, intrinsic, black_value)


@jit
def black_floorlet(
    notional: float,
    strike: float,
    forward_rate: float,
    volatility: float,
    time_to_reset: float,
    period_length: float,
    discount_factor: float,
) -> float:
    """Price a single floorlet using Black's model.

    A floorlet pays: Notional × max(Strike - Rate, 0) × Period_Length

    Args:
        notional: Notional principal
        strike: Strike rate (floor rate)
        forward_rate: Forward rate for the period
        volatility: Black volatility
        time_to_reset: Time to rate fixing
        period_length: Length of interest period
        discount_factor: Discount factor to payment date

    Returns:
        Present value of floorlet
    """
    from jax.scipy.stats import norm

    # Intrinsic value for zero volatility case
    intrinsic = notional * jnp.maximum(strike - forward_rate, 0.0) * period_length * discount_factor

    # Black model calculation
    sqrt_t = jnp.sqrt(jnp.maximum(time_to_reset, 1e-10))
    vol_safe = jnp.maximum(volatility, 1e-10)

    d1 = (jnp.log(forward_rate / strike) + 0.5 * vol_safe**2 * time_to_reset) / (
        vol_safe * sqrt_t
    )
    d2 = d1 - vol_safe * sqrt_t

    N_minus_d1 = norm.cdf(-d1)
    N_minus_d2 = norm.cdf(-d2)

    black_value = notional * period_length * discount_factor * (
        strike * N_minus_d2 - forward_rate * N_minus_d1
    )

    # Use intrinsic value if volatility or time is too small
    use_intrinsic = (volatility < 1e-10) | (time_to_reset < 1e-10)
    return jnp.where(use_intrinsic, intrinsic, black_value)


# Convenience type aliases for common use cases
@dataclass
class Cap(InterestRateCapFloorCollar):
    """Interest rate cap (protection against rising rates)."""

    def __init__(self, T: float, notional: float, strike: float, **kwargs):
        super().__init__(
            T=T, notional=notional, strike=strike, cap_floor_type=CapFloorType.CAP, **kwargs
        )


@dataclass
class Floor(InterestRateCapFloorCollar):
    """Interest rate floor (protection against falling rates)."""

    def __init__(self, T: float, notional: float, strike: float, **kwargs):
        super().__init__(
            T=T, notional=notional, strike=strike, cap_floor_type=CapFloorType.FLOOR, **kwargs
        )


@dataclass
class Collar(InterestRateCapFloorCollar):
    """Interest rate collar (long cap, short floor)."""

    def __init__(
        self,
        T: float,
        notional: float,
        cap_strike: float,
        floor_strike: float,
        **kwargs,
    ):
        super().__init__(
            T=T,
            notional=notional,
            strike=cap_strike,
            cap_floor_type=CapFloorType.COLLAR,
            collar_floor_strike=floor_strike,
            **kwargs,
        )


__all__ = [
    "InterestRateCapFloorCollar",
    "Cap",
    "Floor",
    "Collar",
    "CapFloorType",
    "black_caplet",
    "black_floorlet",
]
