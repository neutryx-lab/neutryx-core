"""Power derivatives - peak/off-peak electricity products."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import jax.numpy as jnp

from ..base import Product, PathProduct

Array = jnp.ndarray


class PowerPeriod(Enum):
    """Power delivery period."""

    PEAK = "peak"  # Business hours (typically 7am-11pm weekdays)
    OFF_PEAK = "off_peak"  # Nights and weekends
    AROUND_THE_CLOCK = "atc"  # 24/7 delivery


@dataclass
class PowerForward(Product):
    """Power forward contract.

    Forward contract for electricity delivery. Price includes
    period-specific premium (peak vs off-peak).

    Attributes:
        T: Time to delivery start
        forward_price: Forward price ($/MWh)
        notional: Volume (MWh)
        period: Delivery period (peak/off-peak/ATC)
        period_weight: Fraction of hours in period (e.g., 0.5 for peak)
    """

    forward_price: float
    notional: float
    period: PowerPeriod = PowerPeriod.PEAK
    period_weight: float = 0.5  # Default: peak is 50% of day

    @property
    def requires_path(self) -> bool:
        return False

    def payoff_terminal(self, spot: Array) -> Array:
        """Calculate power forward payoff."""
        # Adjust spot for period delivery
        effective_spot = spot * self.period_weight
        effective_forward = self.forward_price * self.period_weight

        return self.notional * (effective_spot - effective_forward)


@dataclass
class PeakPowerOption(Product):
    """Peak power option.

    Option on electricity delivered during peak hours.
    Peak typically has higher prices and volatility.

    Attributes:
        T: Maturity
        strike: Strike price ($/MWh)
        is_call: True for call
        notional: Volume (MWh)
        volatility: Peak power volatility (typically 80-150%)
        risk_free_rate: Risk-free rate
        peak_hours_per_day: Hours classified as peak (default: 16)
    """

    strike: float
    is_call: bool = True
    notional: float = 1.0  # MWh
    volatility: float = 1.00  # 100% volatility typical for peak power
    risk_free_rate: float = 0.03
    peak_hours_per_day: float = 16.0

    @property
    def requires_path(self) -> bool:
        return False

    def payoff_terminal(self, spot: Array) -> Array:
        """Calculate peak power option payoff."""
        # Adjust for peak period
        peak_fraction = self.peak_hours_per_day / 24.0
        effective_spot = spot * peak_fraction

        if self.is_call:
            payoff = jnp.maximum(effective_spot - self.strike, 0.0)
        else:
            payoff = jnp.maximum(self.strike - effective_spot, 0.0)

        return self.notional * payoff

    def price(self, forward_price: float) -> float:
        """Price using Black-76 model with high volatility."""
        from .oil import black_76_option

        # Adjust forward price for peak period
        peak_fraction = self.peak_hours_per_day / 24.0
        adjusted_forward = forward_price * peak_fraction

        return float(
            black_76_option(
                F=adjusted_forward,
                K=self.strike,
                T=self.T,
                r=self.risk_free_rate,
                sigma=self.volatility,
                is_call=self.is_call,
            )
            * self.notional
        )


@dataclass
class OffPeakPowerOption(Product):
    """Off-peak power option.

    Option on electricity delivered during off-peak hours
    (nights and weekends). Lower prices and volatility than peak.

    Attributes:
        T: Maturity
        strike: Strike price
        is_call: True for call
        notional: Volume (MWh)
        volatility: Off-peak volatility (typically 50-80%)
        risk_free_rate: Risk-free rate
        offpeak_hours_per_day: Off-peak hours (default: 8)
    """

    strike: float
    is_call: bool = True
    notional: float = 1.0
    volatility: float = 0.60  # Lower volatility for off-peak
    risk_free_rate: float = 0.03
    offpeak_hours_per_day: float = 8.0

    @property
    def requires_path(self) -> bool:
        return False

    def payoff_terminal(self, spot: Array) -> Array:
        """Calculate off-peak power option payoff."""
        offpeak_fraction = self.offpeak_hours_per_day / 24.0
        effective_spot = spot * offpeak_fraction

        if self.is_call:
            payoff = jnp.maximum(effective_spot - self.strike, 0.0)
        else:
            payoff = jnp.maximum(self.strike - effective_spot, 0.0)

        return self.notional * payoff

    def price(self, forward_price: float) -> float:
        """Price with off-peak adjusted parameters."""
        from .oil import black_76_option

        offpeak_fraction = self.offpeak_hours_per_day / 24.0
        adjusted_forward = forward_price * offpeak_fraction

        return float(
            black_76_option(
                F=adjusted_forward,
                K=self.strike,
                T=self.T,
                r=self.risk_free_rate,
                sigma=self.volatility,
                is_call=self.is_call,
            )
            * self.notional
        )


@dataclass
class PowerShapingContract(PathProduct):
    """Power shaping contract (load-following).

    Contract allowing buyer to vary delivery volumes to match
    actual load profile. Common for utilities.

    Attributes:
        T: Contract maturity
        base_load: Minimum steady load (MW)
        max_load: Maximum load capacity (MW)
        shaping_cost: Cost per MW deviation from base
        fixing_times: Hourly fixing times
        load_profile: Target load profile (if known)
    """

    base_load: float
    max_load: float
    shaping_cost: float
    fixing_times: Array
    load_profile: Array | None = None

    def __post_init__(self):
        """Ensure arrays."""
        self.fixing_times = jnp.asarray(self.fixing_times)
        if self.load_profile is not None:
            self.load_profile = jnp.asarray(self.load_profile)

    @property
    def requires_path(self) -> bool:
        return True

    def payoff_path(self, path: Array) -> Array:
        """Calculate power shaping payoff.

        Args:
            path: Power prices over time

        Returns:
            Value of shaping flexibility
        """
        path = jnp.asarray(path)
        n_steps = len(path)

        dt = self.T / (n_steps - 1) if n_steps > 1 else self.T
        fixing_indices = jnp.round(self.fixing_times / dt).astype(int)
        fixing_indices = jnp.clip(fixing_indices, 0, n_steps - 1)

        total_payoff = 0.0

        for i, idx in enumerate(fixing_indices):
            price = path[idx]

            # Determine optimal load for this period
            if self.load_profile is not None and i < len(self.load_profile):
                target_load = self.load_profile[i]
            else:
                # Optimize: take more power when cheap, less when expensive
                if price < jnp.mean(path):
                    target_load = self.max_load
                else:
                    target_load = self.base_load

            # Cost of power
            power_cost = target_load * price

            # Shaping cost (deviation from base)
            deviation = abs(target_load - self.base_load)
            shaping_penalty = deviation * self.shaping_cost

            # Net cost this period
            period_cost = power_cost + shaping_penalty
            total_payoff -= period_cost

        return total_payoff


__all__ = [
    "PowerPeriod",
    "PowerForward",
    "PeakPowerOption",
    "OffPeakPowerOption",
    "PowerShapingContract",
]
