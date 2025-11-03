"""Structured products including autocallables, reverse convertibles, and equity-linked notes.

This module implements complex structured products that combine options with
note features. These products are path-dependent and require full simulation paths
for accurate pricing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp

from .base import PathProduct
from ._utils import ensure_array, extract_terminal, vanilla_payoff


@dataclass
class PhoenixAutocallable(PathProduct):
    """Phoenix (memory) autocallable note.

    An autocallable note that pays accumulated coupons (including previously
    unpaid coupons) if the underlying is above the coupon barrier at observation.
    Autocalls if above autocall barrier at observation dates.

    Parameters
    ----------
    K : float
        Strike price (typically at-the-money)
    T : float
        Maturity in years
    autocall_barrier : float
        Barrier level for autocall (e.g., 1.0 = 100% of initial)
    coupon_barrier : float
        Barrier level for coupon payment (e.g., 0.85 = 85% of initial)
    coupon_rate : float
        Coupon rate per observation period
    observation_times : jnp.ndarray
        Array of observation times in years
    put_strike : float
        Strike for downside put protection (typically 1.0)
    """

    K: float
    T: float
    autocall_barrier: float
    coupon_barrier: float
    coupon_rate: float
    observation_times: jnp.ndarray
    put_strike: float = 1.0

    def __post_init__(self):
        self.observation_times = ensure_array(self.observation_times)

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Compute payoff for a single path.

        The payoff logic:
        1. At each observation, check if autocall barrier is hit -> early redemption
        2. If not autocalled, accumulate unpaid coupons (memory feature)
        3. At maturity, if not autocalled, apply put protection
        """
        path = ensure_array(path)
        n_steps = len(path)

        # Normalize path by initial strike
        normalized_path = path / self.K

        # Determine which indices to check based on observation times
        # Assume path is evenly spaced over [0, T]
        dt = self.T / (n_steps - 1) if n_steps > 1 else self.T
        observation_indices = jnp.round(self.observation_times / dt).astype(int)
        observation_indices = jnp.clip(observation_indices, 0, n_steps - 1)

        # Track accumulated unpaid coupons
        accumulated_coupons = 0.0
        autocalled = False
        autocall_value = 0.0

        for i, obs_idx in enumerate(observation_indices):
            if not autocalled:
                level = normalized_path[obs_idx]

                # Check autocall
                if level >= self.autocall_barrier:
                    # Autocall: return principal + all accumulated coupons
                    accumulated_coupons += self.coupon_rate
                    autocall_value = 1.0 + accumulated_coupons
                    autocalled = True
                    break

                # Check coupon payment
                if level >= self.coupon_barrier:
                    # Pay all accumulated coupons
                    accumulated_coupons += self.coupon_rate
                    # In Phoenix, coupons are paid but we track for final payoff
                else:
                    # Coupon not paid, accumulate (memory feature)
                    accumulated_coupons += self.coupon_rate

        # If autocalled, return autocall value
        if autocalled:
            return autocall_value

        # If not autocalled, check maturity payoff
        final_level = normalized_path[-1]

        if final_level >= self.coupon_barrier:
            # Principal + accumulated coupons
            return 1.0 + accumulated_coupons
        else:
            # Below barrier: put protection kicks in
            # Return principal * final_level/put_strike + accumulated coupons
            downside = jnp.maximum(final_level / self.put_strike, 0.0)
            return downside + accumulated_coupons


@dataclass
class SnowballAutocallable(PathProduct):
    """Snowball autocallable note.

    Similar to Phoenix but coupon rate increases over time if not autocalled
    (snowball effect).

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Maturity in years
    autocall_barrier : float
        Barrier level for autocall
    coupon_barrier : float
        Barrier level for coupon payment
    initial_coupon_rate : float
        Initial coupon rate
    coupon_step : float
        Increment to coupon rate at each observation
    observation_times : jnp.ndarray
        Array of observation times
    put_strike : float
        Strike for downside protection
    """

    K: float
    T: float
    autocall_barrier: float
    coupon_barrier: float
    initial_coupon_rate: float
    coupon_step: float
    observation_times: jnp.ndarray
    put_strike: float = 1.0

    def __post_init__(self):
        self.observation_times = ensure_array(self.observation_times)

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        path = ensure_array(path)
        n_steps = len(path)
        normalized_path = path / self.K

        dt = self.T / (n_steps - 1) if n_steps > 1 else self.T
        observation_indices = jnp.round(self.observation_times / dt).astype(int)
        observation_indices = jnp.clip(observation_indices, 0, n_steps - 1)

        accumulated_coupons = 0.0
        current_coupon = self.initial_coupon_rate
        autocalled = False

        for i, obs_idx in enumerate(observation_indices):
            if not autocalled:
                level = normalized_path[obs_idx]

                if level >= self.autocall_barrier:
                    accumulated_coupons += current_coupon
                    return 1.0 + accumulated_coupons

                if level >= self.coupon_barrier:
                    accumulated_coupons += current_coupon

                # Snowball: increase coupon for next period
                current_coupon += self.coupon_step

        # Maturity
        final_level = normalized_path[-1]
        if final_level >= self.coupon_barrier:
            return 1.0 + accumulated_coupons
        else:
            downside = jnp.maximum(final_level / self.put_strike, 0.0)
            return downside + accumulated_coupons


@dataclass
class StepDownAutocallable(PathProduct):
    """Step-down autocallable note.

    An autocallable where the autocall barrier decreases (steps down) over time,
    making it easier to autocall as time progresses.

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Maturity in years
    autocall_barriers : jnp.ndarray
        Decreasing barrier levels for each observation
    coupon_barrier : float
        Barrier level for coupon payment
    coupon_rate : float
        Coupon rate per period
    observation_times : jnp.ndarray
        Array of observation times
    put_strike : float
        Strike for downside protection
    """

    K: float
    T: float
    autocall_barriers: jnp.ndarray
    coupon_barrier: float
    coupon_rate: float
    observation_times: jnp.ndarray
    put_strike: float = 1.0

    def __post_init__(self):
        self.observation_times = ensure_array(self.observation_times)
        self.autocall_barriers = ensure_array(self.autocall_barriers)

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        path = ensure_array(path)
        n_steps = len(path)
        normalized_path = path / self.K

        dt = self.T / (n_steps - 1) if n_steps > 1 else self.T
        observation_indices = jnp.round(self.observation_times / dt).astype(int)
        observation_indices = jnp.clip(observation_indices, 0, n_steps - 1)

        accumulated_coupons = 0.0
        autocalled = False

        for i, obs_idx in enumerate(observation_indices):
            if not autocalled:
                level = normalized_path[obs_idx]
                barrier = self.autocall_barriers[i] if i < len(self.autocall_barriers) else self.autocall_barriers[-1]

                if level >= barrier:
                    accumulated_coupons += self.coupon_rate
                    return 1.0 + accumulated_coupons

                if level >= self.coupon_barrier:
                    accumulated_coupons += self.coupon_rate

        # Maturity
        final_level = normalized_path[-1]
        if final_level >= self.coupon_barrier:
            return 1.0 + accumulated_coupons
        else:
            downside = jnp.maximum(final_level / self.put_strike, 0.0)
            return downside + accumulated_coupons


@dataclass
class ReverseConvertible(PathProduct):
    """Reverse convertible note.

    A structured note that pays high coupons but converts to the underlying
    if the barrier is breached. Investor is short a down-and-in put.

    Parameters
    ----------
    K : float
        Strike/conversion price
    T : float
        Maturity in years
    coupon_rate : float
        Total coupon rate over life
    barrier : float
        Knock-in barrier (e.g., 0.7 = 70% of initial)
    principal : float
        Principal/notional amount
    """

    K: float
    T: float
    coupon_rate: float
    barrier: float
    principal: float = 1.0

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Compute payoff.

        If barrier not hit: principal + coupon
        If barrier hit: principal * (S_T / K) + coupon (physical delivery)
        """
        path = ensure_array(path)
        normalized_path = path / self.K

        # Check if barrier was hit
        barrier_hit = normalized_path.min() <= self.barrier

        # Terminal value
        final_level = normalized_path[-1]

        # Coupon is always paid
        coupon_payment = self.principal * self.coupon_rate

        if barrier_hit:
            # Physical delivery: get stock worth S_T
            stock_value = self.principal * final_level
            return stock_value + coupon_payment
        else:
            # Principal returned
            return self.principal + coupon_payment


@dataclass
class DualCurrencyInvestment(PathProduct):
    """Dual Currency Investment (DCI).

    A deposit that pays principal and interest in one of two currencies,
    depending on FX rate at maturity. Investor is short an FX option.

    Parameters
    ----------
    K : float
        Strike FX rate
    T : float
        Maturity in years
    interest_rate : float
        Enhanced interest rate
    base_currency_amount : float
        Investment amount in base currency
    alternate_currency : str
        Alternate currency code
    option_type : str
        'call' or 'put' - which option investor is short
    """

    K: float
    T: float
    interest_rate: float
    base_currency_amount: float = 1.0
    alternate_currency: str = "USD"
    option_type: Literal["call", "put"] = "call"

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Compute payoff in base currency terms.

        For a USD/JPY DCI where investor deposits JPY:
        - If short call and FX > K: receive USD (converted at K)
        - If short put and FX < K: receive USD (converted at K)
        - Otherwise: receive JPY

        All cases include enhanced interest.
        """
        path = ensure_array(path)
        final_fx = extract_terminal(path)

        principal_with_interest = self.base_currency_amount * (1.0 + self.interest_rate * self.T)

        if self.option_type == "call":
            # Short call: if FX > K, exercised against you
            exercised = final_fx >= self.K
        else:
            # Short put: if FX < K, exercised against you
            exercised = final_fx <= self.K

        if exercised:
            # Receive alternate currency converted at strike K
            alternate_amount = principal_with_interest / self.K
            # Convert back to base at market rate
            return alternate_amount * final_fx
        else:
            # Receive base currency
            return principal_with_interest


@dataclass
class EquityLinkedNote(PathProduct):
    """Enhanced Equity-Linked Note with participation and caps.

    A principal-protected note with leveraged upside participation
    and optional cap on returns.

    Parameters
    ----------
    K : float
        Strike price (initial stock level)
    T : float
        Maturity in years
    participation_rate : float
        Participation in upside (e.g., 1.5 = 150%)
    cap : float | None
        Cap on returns (e.g., 0.3 = 30% max return), None for uncapped
    floor : float
        Floor/protection level (e.g., 0.9 = 90% principal protection)
    principal : float
        Principal amount
    """

    K: float
    T: float
    participation_rate: float
    cap: float | None = None
    floor: float = 1.0
    principal: float = 1.0

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Compute payoff.

        Return = Principal * max(floor, min(1 + participation * (S_T/K - 1), 1 + cap))
        """
        path = ensure_array(path)
        final_price = extract_terminal(path)

        # Performance relative to strike
        performance = final_price / self.K

        # Apply participation rate
        participated_return = 1.0 + self.participation_rate * (performance - 1.0)

        # Apply cap if specified
        if self.cap is not None:
            capped_return = jnp.minimum(participated_return, 1.0 + self.cap)
        else:
            capped_return = participated_return

        # Apply floor
        final_return = jnp.maximum(capped_return, self.floor)

        return self.principal * final_return


@dataclass
class BonusEnhancedNote(PathProduct):
    """Bonus Enhanced Note (European variant).

    Note that pays bonus if barrier not hit, otherwise participates 1:1.

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Maturity
    bonus_level : float
        Bonus payment if barrier not hit (e.g., 1.2 = 120%)
    barrier : float
        Barrier level (e.g., 0.8 = 80% of initial)
    principal : float
        Principal amount
    """

    K: float
    T: float
    bonus_level: float
    barrier: float
    principal: float = 1.0

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        path = ensure_array(path)
        normalized_path = path / self.K

        barrier_hit = normalized_path.min() <= self.barrier
        final_level = normalized_path[-1]

        if barrier_hit:
            # 1:1 participation
            return self.principal * final_level
        else:
            # Bonus applied
            return self.principal * jnp.maximum(final_level, self.bonus_level)


__all__ = [
    "PhoenixAutocallable",
    "SnowballAutocallable",
    "StepDownAutocallable",
    "ReverseConvertible",
    "DualCurrencyInvestment",
    "EquityLinkedNote",
    "BonusEnhancedNote",
]
