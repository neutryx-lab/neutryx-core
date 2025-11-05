"""Forward Rate Agreement (FRA) implementation.

A Forward Rate Agreement is a forward contract on an interest rate.
It allows parties to lock in an interest rate for a future period.

Key features:
- Settlement at the beginning of the interest period (not end)
- Cash-settled based on difference between fixed rate and reference rate
- No exchange of notional principal
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import jax.numpy as jnp
from jax import jit

from ..base import Product

Array = jnp.ndarray


class SettlementType(Enum):
    """FRA settlement convention."""

    ADVANCE = "advance"  # Payment at start of period (market standard)
    ARREARS = "arrears"  # Payment at end of period


@dataclass
class ForwardRateAgreement(Product):
    """Forward Rate Agreement (FRA).

    A FRA is an OTC contract to lock in an interest rate for a future period.
    Settlement occurs at the start of the interest period based on the
    difference between the agreed rate and the prevailing reference rate.

    Example notation: 3x9 FRA means:
    - T: 0.25 years (3 months to settlement)
    - period_length: 0.5 years (6 month interest period, from month 3 to 9)

    Attributes:
        T: Time to settlement (in years)
        notional: Notional principal amount
        fixed_rate: Agreed fixed rate (annualized)
        period_length: Length of interest period (in years)
        day_count_factor: Day count fraction (default: period_length)
        settlement_type: ADVANCE (standard) or ARREARS
        discount_rate: Rate for discounting settlement payment
        is_payer: True if paying fixed rate, False if receiving

    The FRA payoff at settlement is:
        Notional × (Reference_Rate - Fixed_Rate) × DCF / (1 + Reference_Rate × DCF)

    For a payer FRA (long position), profit when reference rate > fixed rate.
    """

    notional: float
    fixed_rate: float
    period_length: float = 0.25  # 3 months default
    day_count_factor: float | None = None
    settlement_type: SettlementType = SettlementType.ADVANCE
    discount_rate: float = 0.03
    is_payer: bool = True  # True = pay fixed (long FRA)

    def __post_init__(self):
        """Initialize day count factor if not provided."""
        if self.day_count_factor is None:
            self.day_count_factor = self.period_length

    @property
    def requires_path(self) -> bool:
        """FRA does not require full path."""
        return False

    def payoff_terminal(self, spot: Array) -> Array:
        """Calculate FRA payoff given terminal reference rate.

        Args:
            spot: Reference rate at settlement (e.g., SOFR, LIBOR)

        Returns:
            Present value of FRA settlement payment

        The settlement amount for a payer FRA is:
            N × (R - K) × τ / (1 + R × τ)

        where:
            N = notional
            R = reference rate (spot)
            K = fixed rate
            τ = day count factor

        This is then discounted to present value.
        """
        reference_rate = spot

        # Calculate settlement payment
        rate_differential = reference_rate - self.fixed_rate

        if self.settlement_type == SettlementType.ADVANCE:
            # Market standard: discount to settlement date
            # Payment = N × (R - K) × τ / (1 + R × τ)
            discount_factor = 1.0 / (1.0 + reference_rate * self.day_count_factor)
            settlement_amount = (
                self.notional * rate_differential * self.day_count_factor * discount_factor
            )
        else:
            # Arrears settlement: payment at end of period
            settlement_amount = self.notional * rate_differential * self.day_count_factor

        # Apply sign based on payer/receiver
        if not self.is_payer:
            settlement_amount = -settlement_amount

        # Discount to present value
        present_value = settlement_amount * jnp.exp(-self.discount_rate * self.T)

        return present_value

    def forward_rate(self, discount_factor_T: float, discount_factor_T2: float) -> float:
        """Calculate the implied forward rate from discount factors.

        Args:
            discount_factor_T: Discount factor to settlement date T
            discount_factor_T2: Discount factor to end of period T + period_length

        Returns:
            Implied forward rate

        Formula:
            F = (1/τ) × (DF(T) / DF(T+τ) - 1)

        where τ is the period length.
        """
        forward_rate = (1.0 / self.day_count_factor) * (
            discount_factor_T / discount_factor_T2 - 1.0
        )
        return float(forward_rate)

    def par_rate(self, forward_rate: float) -> float:
        """Calculate the par rate (break-even fixed rate).

        The par rate is the fixed rate that makes the FRA value zero.
        This equals the forward rate for the period.

        Args:
            forward_rate: Market forward rate for the period

        Returns:
            Par fixed rate (equals forward_rate)
        """
        return forward_rate

    def dv01(self, reference_rate: float) -> float:
        """Calculate DV01 (dollar value of 1 basis point).

        Measures sensitivity to 1bp change in reference rate.

        Args:
            reference_rate: Current reference rate

        Returns:
            Change in FRA value for 1bp rate change
        """
        bp = 0.0001  # 1 basis point

        pv_base = self.payoff_terminal(reference_rate)
        pv_up = self.payoff_terminal(reference_rate + bp)

        return float(pv_up - pv_base)


@jit
def fra_settlement_amount(
    notional: float,
    fixed_rate: float,
    reference_rate: float,
    day_count_factor: float,
    is_payer: bool = True,
) -> float:
    """Calculate FRA settlement amount (advance settlement).

    This is a standalone JIT-compiled function for efficient batch calculation.

    Args:
        notional: Notional principal
        fixed_rate: Agreed fixed rate
        reference_rate: Reference rate at settlement
        day_count_factor: Day count fraction
        is_payer: True for payer FRA, False for receiver

    Returns:
        Settlement amount (not discounted)

    Formula (payer FRA):
        Settlement = N × (R - K) × τ / (1 + R × τ)
    """
    rate_diff = reference_rate - fixed_rate
    discount_factor = 1.0 / (1.0 + reference_rate * day_count_factor)
    settlement = notional * rate_diff * day_count_factor * discount_factor

    # Apply direction
    return jnp.where(is_payer, settlement, -settlement)


@jit
def fra_forward_rate(
    discount_factor_start: float,
    discount_factor_end: float,
    period_length: float,
) -> float:
    """Calculate implied forward rate from discount factors.

    Args:
        discount_factor_start: Discount factor to period start
        discount_factor_end: Discount factor to period end
        period_length: Length of forward period in years

    Returns:
        Implied forward rate (annualized)

    Formula:
        F = (1/τ) × (DF_start / DF_end - 1)
    """
    return (1.0 / period_length) * (discount_factor_start / discount_factor_end - 1.0)


__all__ = [
    "ForwardRateAgreement",
    "SettlementType",
    "fra_settlement_amount",
    "fra_forward_rate",
]
