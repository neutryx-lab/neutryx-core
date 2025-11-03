"""FX exotic derivatives including TARFs, accumulators, and other complex structures.

This module implements sophisticated FX derivatives commonly traded in Asian
and global markets. These products often have path-dependent features and
knockout provisions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp

from .base import PathProduct
from ._utils import ensure_array, extract_terminal


@dataclass
class TARF(PathProduct):
    """Target Redemption Forward (TARF).

    A strip of forwards with a cumulative profit target. Once target is reached,
    the structure terminates. Common in Asian FX markets.

    Parameters
    ----------
    K : float
        Strike rate for each fixing
    T : float
        Maturity in years
    target_profit : float
        Cumulative profit target (in base currency)
    fixing_times : jnp.ndarray
        Array of fixing times
    notional : float
        Notional amount per fixing
    leverage : float
        Leverage on each fixing (e.g., 2.0 = double leverage)
    is_long : bool
        True if long forwards (profit when FX rises), False if short
    knockout_type : str
        'target' (KO when target reached) or 'exact' (partial payment at KO)
    """

    K: float
    T: float
    target_profit: float
    fixing_times: jnp.ndarray
    notional: float = 1.0
    leverage: float = 1.0
    is_long: bool = True
    knockout_type: Literal["target", "exact"] = "target"

    def __post_init__(self):
        self.fixing_times = ensure_array(self.fixing_times)

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Compute TARF payoff.

        At each fixing, accumulate profit/loss from forward contract.
        Once cumulative profit >= target, structure knocks out.
        """
        path = ensure_array(path)
        n_steps = len(path)

        dt = self.T / (n_steps - 1) if n_steps > 1 else self.T
        fixing_indices = jnp.round(self.fixing_times / dt).astype(int)
        fixing_indices = jnp.clip(fixing_indices, 0, n_steps - 1)

        cumulative_profit = 0.0
        knocked_out = False

        for idx in fixing_indices:
            if not knocked_out:
                fx_rate = path[idx]

                # Forward payoff: notional * leverage * (FX - K)
                if self.is_long:
                    period_profit = self.notional * self.leverage * (fx_rate - self.K)
                else:
                    period_profit = self.notional * self.leverage * (self.K - fx_rate)

                cumulative_profit += period_profit

                # Check knockout
                if cumulative_profit >= self.target_profit:
                    if self.knockout_type == "exact":
                        # Exact knockout: pay only up to target
                        return self.target_profit
                    else:
                        # Target knockout: pay full amount of period that hit target
                        return cumulative_profit

        return cumulative_profit


@dataclass
class Accumulator(PathProduct):
    """Accumulator (I will kill you later).

    Investor accumulates position at strike when spot is above strike,
    but accumulates at double leverage when spot is below strike.
    Often has knockout provision.

    Parameters
    ----------
    K : float
        Strike/accumulation price
    T : float
        Maturity in years
    fixing_times : jnp.ndarray
        Fixing dates
    notional_per_fixing : float
        Notional to accumulate at each fixing
    leverage_down : float
        Leverage when spot below strike (typically 2.0)
    leverage_up : float
        Leverage when spot above strike (typically 1.0)
    knockout_barrier : float | None
        Knockout barrier (e.g., 1.1 * K), None for no knockout
    is_long : bool
        True if accumulating long position
    """

    K: float
    T: float
    fixing_times: jnp.ndarray
    notional_per_fixing: float = 1.0
    leverage_down: float = 2.0
    leverage_up: float = 1.0
    knockout_barrier: float | None = None
    is_long: bool = True

    def __post_init__(self):
        self.fixing_times = ensure_array(self.fixing_times)

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Compute accumulator payoff.

        Accumulate positions at each fixing unless knocked out.
        """
        path = ensure_array(path)
        n_steps = len(path)

        dt = self.T / (n_steps - 1) if n_steps > 1 else self.T
        fixing_indices = jnp.round(self.fixing_times / dt).astype(int)
        fixing_indices = jnp.clip(fixing_indices, 0, n_steps - 1)

        total_cost = 0.0
        total_quantity = 0.0
        knocked_out = False

        for idx in fixing_indices:
            if not knocked_out:
                spot = path[idx]

                # Check knockout first
                if self.knockout_barrier is not None and spot >= self.knockout_barrier:
                    knocked_out = True
                    break

                # Determine leverage
                if self.is_long:
                    leverage = self.leverage_down if spot < self.K else self.leverage_up
                else:
                    leverage = self.leverage_up if spot < self.K else self.leverage_down

                # Accumulate
                quantity = self.notional_per_fixing * leverage
                cost = quantity * self.K

                total_quantity += quantity
                total_cost += cost

        # Final P&L: value of accumulated position minus cost
        final_spot = path[-1]
        if self.is_long:
            final_value = total_quantity * final_spot
        else:
            final_value = total_cost - total_quantity * final_spot

        return final_value - total_cost


@dataclass
class Decumulator(PathProduct):
    """Decumulator (reverse accumulator).

    Similar to accumulator but for selling/shorting. Investor sells at strike
    with enhanced leverage when spot is above strike.

    Parameters
    ----------
    K : float
        Strike/selling price
    T : float
        Maturity in years
    fixing_times : jnp.ndarray
        Fixing dates
    notional_per_fixing : float
        Notional per fixing
    leverage_up : float
        Leverage when spot above strike
    leverage_down : float
        Leverage when spot below strike
    knockout_barrier : float | None
        Knockout barrier
    """

    K: float
    T: float
    fixing_times: jnp.ndarray
    notional_per_fixing: float = 1.0
    leverage_up: float = 2.0
    leverage_down: float = 1.0
    knockout_barrier: float | None = None

    def __post_init__(self):
        self.fixing_times = ensure_array(self.fixing_times)

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Compute decumulator payoff (short positions)."""
        path = ensure_array(path)
        n_steps = len(path)

        dt = self.T / (n_steps - 1) if n_steps > 1 else self.T
        fixing_indices = jnp.round(self.fixing_times / dt).astype(int)
        fixing_indices = jnp.clip(fixing_indices, 0, n_steps - 1)

        total_sold_value = 0.0
        total_quantity = 0.0
        knocked_out = False

        for idx in fixing_indices:
            if not knocked_out:
                spot = path[idx]

                # Check knockout
                if self.knockout_barrier is not None and spot <= self.knockout_barrier:
                    knocked_out = True
                    break

                # Leverage: higher when spot above strike
                leverage = self.leverage_up if spot > self.K else self.leverage_down

                # Sell quantity
                quantity = self.notional_per_fixing * leverage
                sold_value = quantity * self.K

                total_quantity += quantity
                total_sold_value += sold_value

        # Final P&L: sold value minus cost to buy back
        final_spot = path[-1]
        buyback_cost = total_quantity * final_spot

        return total_sold_value - buyback_cost


@dataclass
class FaderOption(PathProduct):
    """Fader option (accrual forward).

    Notional adjusts based on time spent in a range. Commonly used in
    commodity and FX markets.

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Maturity
    lower_bound : float
        Lower range bound
    upper_bound : float
        Upper range bound
    max_notional : float
        Maximum notional
    is_call : bool
        True for call, False for put
    accrual_type : str
        'in_range' (accrues when in range) or 'out_range'
    """

    K: float
    T: float
    lower_bound: float
    upper_bound: float
    max_notional: float = 1.0
    is_call: bool = True
    accrual_type: Literal["in_range", "out_range"] = "in_range"

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Compute fader option payoff.

        Effective notional = max_notional * (fraction of time in/out of range)
        """
        path = ensure_array(path)

        # Calculate fraction of time in range
        in_range = (path >= self.lower_bound) & (path <= self.upper_bound)
        fraction_in_range = jnp.mean(in_range)

        if self.accrual_type == "in_range":
            effective_notional = self.max_notional * fraction_in_range
        else:
            effective_notional = self.max_notional * (1.0 - fraction_in_range)

        # Terminal option payoff
        final_spot = extract_terminal(path)
        if self.is_call:
            intrinsic = jnp.maximum(final_spot - self.K, 0.0)
        else:
            intrinsic = jnp.maximum(self.K - final_spot, 0.0)

        return effective_notional * intrinsic


@dataclass
class Napoleon(PathProduct):
    """Napoleon option.

    An option where the strike resets to a more favorable level if certain
    conditions are met during the option's life. Common in FX markets.

    Parameters
    ----------
    K_initial : float
        Initial strike
    T : float
        Maturity
    reset_barrier : float
        Barrier for strike reset
    K_reset : float
        New strike if reset triggered
    is_call : bool
        True for call, False for put
    reset_type : str
        'up' (reset if spot goes above barrier) or 'down'
    """

    K_initial: float
    T: float
    reset_barrier: float
    K_reset: float
    is_call: bool = True
    reset_type: Literal["up", "down"] = "up"

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Compute Napoleon payoff with strike reset feature."""
        path = ensure_array(path)

        # Check if reset condition met
        if self.reset_type == "up":
            reset_triggered = path.max() >= self.reset_barrier
        else:
            reset_triggered = path.min() <= self.reset_barrier

        # Determine effective strike
        K_effective = self.K_reset if reset_triggered else self.K_initial

        # Terminal payoff
        final_spot = extract_terminal(path)
        if self.is_call:
            return jnp.maximum(final_spot - K_effective, 0.0)
        else:
            return jnp.maximum(K_effective - final_spot, 0.0)


@dataclass
class RangeAccrual(PathProduct):
    """Range accrual option (corridor note).

    Pays a fixed rate for each day the underlying stays within a range.

    Parameters
    ----------
    T : float
        Maturity
    lower_bound : float
        Lower range bound
    upper_bound : float
        Upper range bound
    accrual_rate : float
        Rate per period when in range (e.g., 0.05 = 5% annually)
    principal : float
        Principal amount
    """

    T: float
    lower_bound: float
    upper_bound: float
    accrual_rate: float
    principal: float = 1.0

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Compute range accrual payoff."""
        path = ensure_array(path)

        # Calculate fraction of time in range
        in_range = (path >= self.lower_bound) & (path <= self.upper_bound)
        fraction_in_range = jnp.mean(in_range)

        # Total accrual
        total_accrual = self.principal * self.accrual_rate * self.T * fraction_in_range

        # Return principal plus accrued interest
        return self.principal + total_accrual


__all__ = [
    "TARF",
    "Accumulator",
    "Decumulator",
    "FaderOption",
    "Napoleon",
    "RangeAccrual",
]
