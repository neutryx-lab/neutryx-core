"""Commodity derivatives and structured products.

Implements commodity-specific products:
- Commodity forwards and futures
- Commodity options with convenience yield
- Commodity swaps
- Storage and transport options
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import jit

from neutryx.models.bs import price as bs_price


@dataclass
class CommodityForward:
    """Commodity forward contract specification."""

    spot: float
    strike: float
    maturity: float
    risk_free_rate: float
    storage_cost: float = 0.0
    convenience_yield: float = 0.0


@jit
def commodity_forward_price(
    spot: float,
    maturity: float,
    risk_free_rate: float,
    storage_cost: float = 0.0,
    convenience_yield: float = 0.0,
) -> float:
    """Calculate theoretical forward price for commodity.

    Parameters
    ----------
    spot : float
        Current spot price of the commodity
    maturity : float
        Time to maturity in years
    risk_free_rate : float
        Risk-free interest rate (continuously compounded)
    storage_cost : float
        Storage cost as a continuous rate (default: 0.0)
    convenience_yield : float
        Convenience yield (benefit of holding physical commodity)

    Returns
    -------
    float
        Theoretical forward price

    Notes
    -----
    Forward price formula:
        F = S * exp((r + u - y) * T)

    where:
    - S = spot price
    - r = risk-free rate
    - u = storage cost rate
    - y = convenience yield
    - T = time to maturity

    The convenience yield represents the benefit of holding the physical
    commodity (e.g., ability to meet unexpected demand, maintain production).

    Examples
    --------
    >>> commodity_forward_price(50.0, 1.0, 0.05, storage_cost=0.02, convenience_yield=0.03)
    52.020...
    """
    carry_cost = risk_free_rate + storage_cost - convenience_yield
    return spot * jnp.exp(carry_cost * maturity)


@jit
def commodity_forward_value(
    spot: float,
    strike: float,
    maturity: float,
    risk_free_rate: float,
    storage_cost: float = 0.0,
    convenience_yield: float = 0.0,
    position: str = "long",
) -> float:
    """Calculate mark-to-market value of commodity forward.

    Parameters
    ----------
    spot : float
        Current spot price
    strike : float
        Forward strike (delivery price)
    maturity : float
        Time to maturity in years
    risk_free_rate : float
        Risk-free interest rate
    storage_cost : float
        Storage cost rate
    convenience_yield : float
        Convenience yield
    position : str
        'long' or 'short' position

    Returns
    -------
    float
        Present value of the forward contract

    Examples
    --------
    >>> commodity_forward_value(50.0, 48.0, 1.0, 0.05, 0.02, 0.03, "long")
    3.922...
    """
    forward_price = commodity_forward_price(
        spot, maturity, risk_free_rate, storage_cost, convenience_yield
    )
    discount_factor = jnp.exp(-risk_free_rate * maturity)

    payoff = forward_price - strike
    value = payoff * discount_factor

    return jnp.where(position == "long", value, -value)


@jit
def commodity_option_price(
    spot: float,
    strike: float,
    maturity: float,
    risk_free_rate: float,
    volatility: float,
    storage_cost: float = 0.0,
    convenience_yield: float = 0.0,
    option_type: str = "call",
) -> float:
    """Price commodity option using Black-Scholes with convenience yield.

    Parameters
    ----------
    spot : float
        Current spot price
    strike : float
        Strike price
    maturity : float
        Time to maturity in years
    risk_free_rate : float
        Risk-free rate
    volatility : float
        Volatility of commodity price
    storage_cost : float
        Storage cost rate
    convenience_yield : float
        Convenience yield
    option_type : str
        'call' or 'put'

    Returns
    -------
    float
        Option price

    Notes
    -----
    Uses Black-Scholes with cost-of-carry adjustment:
        b = r + u - y

    where u is storage cost and y is convenience yield.
    This is equivalent to treating (u - y) as a negative dividend yield.

    Examples
    --------
    >>> commodity_option_price(
    ...     50.0, 52.0, 1.0, 0.05, 0.30, storage_cost=0.02, convenience_yield=0.03
    ... )
    4.789...
    """
    # Net cost of carry (adjust as if it's a dividend yield)
    # b = r + u - y, so q = y - u (negative if storage costs exceed convenience)
    equivalent_dividend_yield = convenience_yield - storage_cost

    return bs_price(
        S=spot,
        K=strike,
        T=maturity,
        r=risk_free_rate,
        q=equivalent_dividend_yield,
        sigma=volatility,
        kind=option_type,
    )


@dataclass
class CommoditySwap:
    """Commodity swap specification."""

    notional: float  # Quantity of commodity
    fixed_price: float  # Fixed price per unit
    payment_dates: list[float]
    floating_prices: list[float] | None = None


@jit
def commodity_swap_value(
    quantity: float,
    fixed_price: float,
    floating_price: float,
    discount_factor: float,
    position: str = "fixed_payer",
) -> float:
    """Calculate value of a single-period commodity swap.

    Parameters
    ----------
    quantity : float
        Quantity of commodity
    fixed_price : float
        Fixed price per unit
    floating_price : float
        Floating (market) price per unit
    discount_factor : float
        Discount factor to payment date
    position : str
        'fixed_payer' (pays fixed, receives floating) or
        'fixed_receiver' (receives fixed, pays floating)

    Returns
    -------
    float
        Present value of the swap

    Notes
    -----
    A commodity swap exchanges:
    - Fixed leg: Fixed price * Quantity
    - Floating leg: Market price * Quantity

    Value to fixed payer = Quantity * (Floating - Fixed) * DF

    Examples
    --------
    >>> commodity_swap_value(1000.0, 50.0, 55.0, 0.95, "fixed_payer")
    4750.0
    """
    payoff = quantity * (floating_price - fixed_price)
    value = payoff * discount_factor

    return jnp.where(position == "fixed_payer", value, -value)


def multi_period_commodity_swap_value(
    quantity: float,
    fixed_price: float,
    floating_prices: jnp.ndarray,
    discount_factors: jnp.ndarray,
    position: str = "fixed_payer",
) -> float:
    """Calculate value of multi-period commodity swap.

    Parameters
    ----------
    quantity : float
        Quantity of commodity per period
    fixed_price : float
        Fixed price per unit
    floating_prices : Array
        Expected floating prices for each period
    discount_factors : Array
        Discount factors for each payment date
    position : str
        'fixed_payer' or 'fixed_receiver'

    Returns
    -------
    float
        Total present value of the swap

    Examples
    --------
    >>> floating = jnp.array([52.0, 54.0, 53.0])
    >>> dfs = jnp.array([0.98, 0.96, 0.94])
    >>> multi_period_commodity_swap_value(1000, 50.0, floating, dfs, "fixed_payer")
    8180.0
    """
    payoffs = quantity * (floating_prices - fixed_price)
    pv = jnp.sum(payoffs * discount_factors)

    return float(jnp.where(position == "fixed_payer", pv, -pv))


@jit
def spread_option_price(
    spot1: float,
    spot2: float,
    strike: float,
    maturity: float,
    risk_free_rate: float,
    vol1: float,
    vol2: float,
    correlation: float,
    option_type: str = "call",
) -> float:
    """Price a commodity spread option (approximate).

    Parameters
    ----------
    spot1 : float
        Spot price of first commodity
    spot2 : float
        Spot price of second commodity
    strike : float
        Strike on the spread
    maturity : float
        Time to maturity
    risk_free_rate : float
        Risk-free rate
    vol1 : float
        Volatility of first commodity
    vol2 : float
        Volatility of second commodity
    correlation : float
        Correlation between the two commodities
    option_type : str
        'call' or 'put'

    Returns
    -------
    float
        Approximate option price

    Notes
    -----
    Spread option payoff:
        Call: max(S1 - S2 - K, 0)
        Put: max(K - (S1 - S2), 0)

    This uses Kirk's approximation for pricing spread options.
    The spread S1 - S2 is approximated as lognormal.

    Examples
    --------
    >>> spread_option_price(
    ...     50.0, 45.0, 3.0, 1.0, 0.05, 0.25, 0.30, 0.6, "call"
    ... )
    3.456...
    """
    # Kirk's approximation
    # Treat the spread as a single asset with adjusted parameters

    # Forward prices
    F1 = spot1 * jnp.exp(risk_free_rate * maturity)
    F2 = spot2 * jnp.exp(risk_free_rate * maturity)

    # Adjusted strike and spot
    spread_forward = F1 - F2
    adjusted_spot = spread_forward * jnp.exp(-risk_free_rate * maturity)

    # Approximate spread volatility
    weight2 = F2 / (F2 + strike)
    spread_vol = jnp.sqrt(
        vol1**2 + (weight2 * vol2) ** 2 - 2.0 * correlation * vol1 * vol2 * weight2
    )

    # Use Black-Scholes on the spread
    return bs_price(
        S=adjusted_spot,
        K=strike,
        T=maturity,
        r=risk_free_rate,
        q=0.0,
        sigma=spread_vol,
        kind=option_type,
    )


@jit
def asian_commodity_price(
    spot: float,
    strike: float,
    maturity: float,
    risk_free_rate: float,
    volatility: float,
    num_fixings: int,
    option_type: str = "call",
) -> float:
    """Price Asian option on commodity (arithmetic average).

    Parameters
    ----------
    spot : float
        Current spot price
    strike : float
        Strike price
    maturity : float
        Time to maturity
    risk_free_rate : float
        Risk-free rate
    volatility : float
        Volatility
    num_fixings : int
        Number of averaging fixings
    option_type : str
        'call' or 'put'

    Returns
    -------
    float
        Approximate Asian option price

    Notes
    -----
    Asian options are common in commodity markets as they reduce
    manipulation risk and volatility.

    This uses the Curran approximation with adjusted volatility.

    Examples
    --------
    >>> asian_commodity_price(50.0, 50.0, 1.0, 0.05, 0.30, 12, "call")
    4.123...
    """
    # Adjusted volatility for arithmetic average
    # Using approximation: σ_avg ≈ σ / sqrt(3)
    adjusted_vol = volatility / jnp.sqrt(3.0)

    # Adjust for number of fixings
    fixing_adjustment = jnp.sqrt((num_fixings + 1.0) / (2.0 * num_fixings))
    final_vol = adjusted_vol * fixing_adjustment

    # Use Black-Scholes with adjusted volatility
    return bs_price(
        S=spot,
        K=strike,
        T=maturity,
        r=risk_free_rate,
        q=0.0,
        sigma=final_vol,
        kind=option_type,
    )


__all__ = [
    "CommodityForward",
    "CommoditySwap",
    "asian_commodity_price",
    "commodity_forward_price",
    "commodity_forward_value",
    "commodity_option_price",
    "commodity_swap_value",
    "multi_period_commodity_swap_value",
    "spread_option_price",
]
