"""Exotic commodity derivatives: swing options, energy spreads, and weather derivatives.

This module implements specialized commodity derivatives commonly traded in
energy and agricultural markets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

from .base import PathProduct, Product
from ._utils import ensure_array, extract_terminal


@dataclass
class SwingOption(PathProduct):
    """Swing option (take-or-pay with flexible delivery).

    A swing option gives the holder the right to purchase (or sell) varying
    quantities of a commodity over multiple periods, subject to constraints
    on minimum and maximum daily/total volumes.

    Common in natural gas and electricity markets.

    Parameters
    ----------
    T : float
        Contract maturity in years
    strike : float
        Strike price per unit
    min_daily : float
        Minimum daily take (e.g., 0 = no minimum)
    max_daily : float
        Maximum daily take capacity
    min_total : float
        Minimum total volume over contract period
    max_total : float
        Maximum total volume over contract period
    fixing_times : jnp.ndarray
        Times at which exercise decisions are made
    penalty_rate : float
        Penalty for not meeting minimum commitments (default: 0.0)
    is_call : bool
        True for call (right to buy), False for put (right to sell)
    """

    T: float
    strike: float
    min_daily: float
    max_daily: float
    min_total: float
    max_total: float
    fixing_times: jnp.ndarray
    penalty_rate: float = 0.0
    is_call: bool = True

    def __post_init__(self):
        self.fixing_times = ensure_array(self.fixing_times)

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Compute swing option payoff with optimal exercise strategy.

        This uses a greedy heuristic: exercise when profitable, subject to constraints.
        For accurate pricing, dynamic programming or LSM would be needed.
        """
        path = ensure_array(path)
        n_steps = len(path)

        dt = self.T / (n_steps - 1) if n_steps > 1 else self.T
        fixing_indices = jnp.round(self.fixing_times / dt).astype(int)
        fixing_indices = jnp.clip(fixing_indices, 0, n_steps - 1)

        total_exercised = 0.0
        total_payoff = 0.0

        for idx in fixing_indices:
            spot = path[idx]

            # Determine optimal exercise quantity
            if self.is_call:
                # Exercise if spot > strike
                intrinsic_per_unit = jnp.maximum(spot - self.strike, 0.0)
            else:
                # Exercise if strike > spot
                intrinsic_per_unit = jnp.maximum(self.strike - spot, 0.0)

            # Greedy strategy: exercise max if profitable, min otherwise
            if intrinsic_per_unit > 0:
                quantity = self.max_daily
            else:
                quantity = self.min_daily

            # Check total constraint
            quantity = jnp.minimum(quantity, self.max_total - total_exercised)

            # Update
            period_payoff = quantity * intrinsic_per_unit
            total_payoff += period_payoff
            total_exercised += quantity

        # Apply penalty for not meeting minimum total
        shortfall = jnp.maximum(self.min_total - total_exercised, 0.0)
        penalty = shortfall * self.penalty_rate

        return total_payoff - penalty


@dataclass
class SparkSpread(PathProduct):
    """Spark spread option (power - gas spread).

    Models the profit margin of a gas-fired power plant:
        Spark Spread = Power Price - (Heat Rate × Gas Price) - Variable O&M

    Parameters
    ----------
    T : float
        Maturity
    strike : float
        Strike on the spread (minimum acceptable margin)
    heat_rate : float
        Heat rate (MMBtu gas per MWh electricity, typically 7-10)
    variable_om : float
        Variable operating and maintenance cost per MWh
    max_capacity : float
        Maximum generation capacity (MWh)
    is_call : bool
        True for call on spread
    """

    T: float
    strike: float
    heat_rate: float
    variable_om: float = 0.0
    max_capacity: float = 1.0
    is_call: bool = True

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Compute spark spread payoff.

        Path is assumed to be a 2D array: [power_prices, gas_prices]
        """
        path = ensure_array(path)

        # Extract power and gas prices
        # Assume path is shape (2, n_steps) where path[0] = power, path[1] = gas
        if path.ndim == 1:
            # Single price path - assume equal power/gas for demonstration
            power_price = extract_terminal(path)
            gas_price = extract_terminal(path)
        else:
            power_price = extract_terminal(path[0])
            gas_price = extract_terminal(path[1])

        # Spark spread = power - heat_rate * gas - variable_om
        spread = power_price - self.heat_rate * gas_price - self.variable_om

        # Option payoff
        if self.is_call:
            intrinsic = jnp.maximum(spread - self.strike, 0.0)
        else:
            intrinsic = jnp.maximum(self.strike - spread, 0.0)

        return self.max_capacity * intrinsic


@dataclass
class DarkSpread(PathProduct):
    """Dark spread option (power - coal spread).

    Similar to spark spread but for coal-fired generation:
        Dark Spread = Power Price - (Fuel Cost) - Emissions Cost - Variable O&M

    Parameters
    ----------
    T : float
        Maturity
    strike : float
        Strike on the spread
    heat_rate : float
        Heat rate (tons coal per MWh)
    emissions_rate : float
        CO2 emissions rate (tons CO2 per MWh)
    variable_om : float
        Variable O&M cost
    max_capacity : float
        Maximum capacity
    is_call : bool
        True for call
    """

    T: float
    strike: float
    heat_rate: float
    emissions_rate: float = 0.0
    variable_om: float = 0.0
    max_capacity: float = 1.0
    is_call: bool = True

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Compute dark spread payoff.

        Path assumed to be [power_prices, coal_prices, carbon_prices]
        """
        path = ensure_array(path)

        if path.ndim == 1:
            power_price = extract_terminal(path)
            coal_price = extract_terminal(path) * 0.5  # Dummy
            carbon_price = 0.0
        else:
            power_price = extract_terminal(path[0])
            coal_price = extract_terminal(path[1])
            carbon_price = extract_terminal(path[2]) if len(path) > 2 else 0.0

        # Dark spread
        spread = (
            power_price
            - self.heat_rate * coal_price
            - self.emissions_rate * carbon_price
            - self.variable_om
        )

        if self.is_call:
            intrinsic = jnp.maximum(spread - self.strike, 0.0)
        else:
            intrinsic = jnp.maximum(self.strike - spread, 0.0)

        return self.max_capacity * intrinsic


@dataclass
class CrackSpread(PathProduct):
    """Crack spread option (refined products - crude oil).

    Models the profit margin of oil refining:
        Crack Spread = (Gasoline Price + Heating Oil Price) - Crude Price

    Common structures: 3-2-1 crack spread
        (3 barrels crude -> 2 barrels gasoline + 1 barrel heating oil)

    Parameters
    ----------
    T : float
        Maturity
    strike : float
        Strike on the spread
    gasoline_weight : float
        Gallons gasoline per barrel crude (e.g., 2.0)
    heating_oil_weight : float
        Gallons heating oil per barrel crude (e.g., 1.0)
    crude_weight : float
        Barrels crude consumed (e.g., 3.0 for 3-2-1 spread)
    max_capacity : float
        Maximum refining capacity
    is_call : bool
        True for call
    """

    T: float
    strike: float
    gasoline_weight: float = 2.0
    heating_oil_weight: float = 1.0
    crude_weight: float = 3.0
    max_capacity: float = 1.0
    is_call: bool = True

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Compute crack spread payoff.

        Path: [gasoline_prices, heating_oil_prices, crude_prices]
        """
        path = ensure_array(path)

        if path.ndim == 1:
            # Dummy: assume uniform prices
            price = extract_terminal(path)
            gasoline_price = price
            heating_oil_price = price * 0.95
            crude_price = price * 0.9
        else:
            gasoline_price = extract_terminal(path[0])
            heating_oil_price = extract_terminal(path[1])
            crude_price = extract_terminal(path[2])

        # Crack spread (per unit of crude)
        spread = (
            self.gasoline_weight * gasoline_price
            + self.heating_oil_weight * heating_oil_price
            - self.crude_weight * crude_price
        ) / self.crude_weight

        if self.is_call:
            intrinsic = jnp.maximum(spread - self.strike, 0.0)
        else:
            intrinsic = jnp.maximum(self.strike - spread, 0.0)

        return self.max_capacity * intrinsic


@dataclass
class HeatingDegreeDays(PathProduct):
    """Heating Degree Days (HDD) weather derivative.

    Pays based on cumulative heating degree days over a period.
    HDD = sum of max(T_base - T_avg, 0) over measurement period.

    Used to hedge weather-related energy demand.

    Parameters
    ----------
    T : float
        Contract maturity
    base_temperature : float
        Base temperature (e.g., 65°F or 18°C)
    strike_hdd : float
        Strike level for HDD index
    tick_value : float
        Payment per HDD unit
    contract_type : str
        'call' (pays if HDD > strike), 'put' (pays if HDD < strike), or 'swap'
    """

    T: float
    base_temperature: float
    strike_hdd: float
    tick_value: float
    contract_type: Literal["call", "put", "swap"] = "call"

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Compute HDD payoff.

        Path is daily average temperatures.
        """
        path = ensure_array(path)

        # Calculate daily HDD
        daily_hdd = jnp.maximum(self.base_temperature - path, 0.0)

        # Cumulative HDD over period
        total_hdd = jnp.sum(daily_hdd)

        # Payoff based on contract type
        if self.contract_type == "call":
            payoff = jnp.maximum(total_hdd - self.strike_hdd, 0.0)
        elif self.contract_type == "put":
            payoff = jnp.maximum(self.strike_hdd - total_hdd, 0.0)
        else:  # swap
            payoff = total_hdd - self.strike_hdd

        return self.tick_value * payoff


@dataclass
class CoolingDegreeDays(PathProduct):
    """Cooling Degree Days (CDD) weather derivative.

    Pays based on cumulative cooling degree days.
    CDD = sum of max(T_avg - T_base, 0) over measurement period.

    Used to hedge summer cooling demand.

    Parameters
    ----------
    T : float
        Contract maturity
    base_temperature : float
        Base temperature (e.g., 65°F or 18°C)
    strike_cdd : float
        Strike level for CDD index
    tick_value : float
        Payment per CDD unit
    contract_type : str
        'call', 'put', or 'swap'
    """

    T: float
    base_temperature: float
    strike_cdd: float
    tick_value: float
    contract_type: Literal["call", "put", "swap"] = "call"

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Compute CDD payoff."""
        path = ensure_array(path)

        # Calculate daily CDD
        daily_cdd = jnp.maximum(path - self.base_temperature, 0.0)

        # Cumulative CDD
        total_cdd = jnp.sum(daily_cdd)

        # Payoff
        if self.contract_type == "call":
            payoff = jnp.maximum(total_cdd - self.strike_cdd, 0.0)
        elif self.contract_type == "put":
            payoff = jnp.maximum(self.strike_cdd - total_cdd, 0.0)
        else:  # swap
            payoff = total_cdd - self.strike_cdd

        return self.tick_value * payoff


@dataclass
class RainfallDerivative(PathProduct):
    """Rainfall derivative (precipitation index).

    Pays based on cumulative rainfall over a measurement period.
    Common in agricultural hedging.

    Parameters
    ----------
    T : float
        Contract maturity
    strike_rainfall : float
        Strike level for cumulative rainfall (mm or inches)
    tick_value : float
        Payment per unit of rainfall
    contract_type : str
        'call' (drought protection), 'put' (flood protection), or 'collar'
    cap : float | None
        Cap for collar structures
    """

    T: float
    strike_rainfall: float
    tick_value: float
    contract_type: Literal["call", "put", "collar"] = "put"
    cap: float | None = None

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Compute rainfall derivative payoff.

        Path is daily rainfall amounts.
        """
        path = ensure_array(path)

        # Cumulative rainfall
        total_rainfall = jnp.sum(path)

        # Payoff
        if self.contract_type == "call":
            payoff = jnp.maximum(total_rainfall - self.strike_rainfall, 0.0)
        elif self.contract_type == "put":
            payoff = jnp.maximum(self.strike_rainfall - total_rainfall, 0.0)
        else:  # collar
            if self.cap is not None:
                lower_payoff = jnp.maximum(self.strike_rainfall - total_rainfall, 0.0)
                upper_payoff = jnp.maximum(total_rainfall - self.cap, 0.0)
                payoff = lower_payoff - upper_payoff
            else:
                payoff = jnp.maximum(self.strike_rainfall - total_rainfall, 0.0)

        return self.tick_value * payoff


@dataclass
class CommodityBasketOption(Product):
    """Basket option on multiple commodities.

    Payoff based on weighted average or sum of commodity prices.

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Maturity
    weights : jnp.ndarray
        Weights for each commodity in the basket
    basket_type : str
        'average' (weighted average) or 'sum' (weighted sum)
    is_call : bool
        True for call, False for put
    """

    K: float
    T: float
    weights: jnp.ndarray
    basket_type: Literal["average", "sum"] = "average"
    is_call: bool = True

    def __post_init__(self):
        self.weights = ensure_array(self.weights)

    def payoff_terminal(self, spot: jnp.ndarray) -> jnp.ndarray:
        """Compute basket option payoff.

        spot: array of terminal prices for each commodity
        """
        spot = ensure_array(spot)

        # Compute basket value
        if self.basket_type == "average":
            basket_value = jnp.sum(self.weights * spot) / jnp.sum(self.weights)
        else:  # sum
            basket_value = jnp.sum(self.weights * spot)

        # Option payoff
        if self.is_call:
            return jnp.maximum(basket_value - self.K, 0.0)
        else:
            return jnp.maximum(self.K - basket_value, 0.0)


@dataclass
class InterruptibleContract(PathProduct):
    """Interruptible supply contract (real option).

    Supplier has the right to interrupt delivery when spot prices exceed a threshold.
    Buyer receives compensation for interruption.

    Parameters
    ----------
    T : float
        Contract maturity
    contract_price : float
        Contracted delivery price
    interruption_threshold : float
        Price threshold triggering interruption right
    interruption_payment : float
        Compensation to buyer for each interruption
    fixing_times : jnp.ndarray
        Delivery dates
    max_interruptions : int
        Maximum number of interruptions allowed
    """

    T: float
    contract_price: float
    interruption_threshold: float
    interruption_payment: float
    fixing_times: jnp.ndarray
    max_interruptions: int = 10

    def __post_init__(self):
        self.fixing_times = ensure_array(self.fixing_times)

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Compute interruptible contract payoff from supplier perspective."""
        path = ensure_array(path)
        n_steps = len(path)

        dt = self.T / (n_steps - 1) if n_steps > 1 else self.T
        fixing_indices = jnp.round(self.fixing_times / dt).astype(int)
        fixing_indices = jnp.clip(fixing_indices, 0, n_steps - 1)

        interruptions_used = 0
        total_payoff = 0.0

        for idx in fixing_indices:
            spot = path[idx]

            # Decide whether to interrupt
            if (
                spot > self.interruption_threshold
                and interruptions_used < self.max_interruptions
            ):
                # Interrupt: save (spot - contract_price) but pay compensation
                payoff = (spot - self.contract_price) - self.interruption_payment
                interruptions_used += 1
            else:
                # Deliver at contract price
                payoff = self.contract_price - spot

            total_payoff += payoff

        return total_payoff


__all__ = [
    "SwingOption",
    "SparkSpread",
    "DarkSpread",
    "CrackSpread",
    "HeatingDegreeDays",
    "CoolingDegreeDays",
    "RainfallDerivative",
    "CommodityBasketOption",
    "InterruptibleContract",
]
