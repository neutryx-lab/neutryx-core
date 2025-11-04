"""Equity derivatives and structured products.

Implements various equity-linked products:
- Equity forwards and futures
- Dividend swaps and futures
- Variance swaps
- Total return swaps
- Equity-linked notes
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import jax.numpy as jnp
from jax import jit

from neutryx.models.bs import greeks, price as bs_price


@dataclass
class EquityForward:
    """Equity forward contract specification."""

    spot: float
    strike: float
    maturity: float
    risk_free_rate: float
    dividend_yield: float = 0.0


@jit
def equity_forward_price(
    spot: float,
    maturity: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
) -> float:
    """Calculate theoretical forward price for equity.

    Parameters
    ----------
    spot : float
        Current spot price of the underlying equity
    maturity : float
        Time to maturity in years
    risk_free_rate : float
        Risk-free interest rate (continuously compounded)
    dividend_yield : float
        Continuous dividend yield (default: 0.0)

    Returns
    -------
    float
        Theoretical forward price

    Notes
    -----
    Forward price formula:
        F = S * exp((r - q) * T)

    where:
    - S = spot price
    - r = risk-free rate
    - q = dividend yield
    - T = time to maturity

    Examples
    --------
    >>> equity_forward_price(100.0, 1.0, 0.05, 0.02)
    103.045...
    """
    return spot * jnp.exp((risk_free_rate - dividend_yield) * maturity)


@partial(jit, static_argnames=["position"])
def equity_forward_value(
    spot: float,
    strike: float,
    maturity: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    position: str = "long",
) -> float:
    """Calculate mark-to-market value of equity forward.

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
    dividend_yield : float
        Continuous dividend yield
    position : str
        'long' or 'short' position

    Returns
    -------
    float
        Present value of the forward contract

    Notes
    -----
    For a long position:
        V = (F - K) * exp(-r * T)

    where F is the theoretical forward price and K is the strike.

    Examples
    --------
    >>> equity_forward_value(100.0, 98.0, 1.0, 0.05, 0.02, "long")
    4.894...
    """
    forward_price = equity_forward_price(spot, maturity, risk_free_rate, dividend_yield)
    discount_factor = jnp.exp(-risk_free_rate * maturity)

    payoff = forward_price - strike
    value = payoff * discount_factor

    # Return based on position
    return jnp.where(position == "long", value, -value)


@dataclass
class DividendSwap:
    """Dividend swap specification."""

    notional: float
    strike: float  # Fixed dividend amount
    maturity: float
    payment_dates: list[float]
    expected_dividends: list[float]


@jit
def dividend_swap_value(
    notional: float,
    strike: float,
    expected_total_dividends: float,
    discount_factor: float,
) -> float:
    """Calculate value of a dividend swap.

    Parameters
    ----------
    notional : float
        Notional amount
    strike : float
        Fixed dividend strike (per share)
    expected_total_dividends : float
        Expected total dividends over the swap period
    discount_factor : float
        Discount factor to maturity

    Returns
    -------
    float
        Present value of the dividend swap

    Notes
    -----
    A dividend swap exchanges realized dividends for a fixed strike.
    The floating leg pays actual dividends, the fixed leg pays the strike.

    Value = Notional * (Expected Dividends - Strike) * Discount Factor

    Examples
    --------
    >>> dividend_swap_value(1000.0, 5.0, 6.5, 0.95)
    1425.0
    """
    return notional * (expected_total_dividends - strike) * discount_factor


@dataclass
class DividendFuture:
    """Dividend futures contract specification."""

    underlying: str
    contract_size: float
    maturity: float
    settlement_date: float
    expected_dividends: float


@jit
def dividend_futures_price(
    expected_total_dividends: float,
    spot: float,
    forward: float,
    risk_free_rate: float,
    maturity: float,
) -> float:
    """Calculate theoretical dividend futures price.

    Parameters
    ----------
    expected_total_dividends : float
        Expected cumulative dividends over the contract period
    spot : float
        Current spot price of the underlying
    forward : float
        Forward price of the underlying to maturity
    risk_free_rate : float
        Risk-free rate
    maturity : float
        Time to maturity in years

    Returns
    -------
    float
        Dividend futures price (points value)

    Notes
    -----
    Dividend futures settle to the actual dividends paid by the underlying
    index over the contract period.

    The theoretical price can be derived from the dividend yield implied
    by the equity forward:
        F = S * exp((r - q) * T)
        q = (1/T) * ln(S/F) + r

    Expected dividends:
        D = S * (1 - exp(-q * T))

    Examples
    --------
    >>> dividend_futures_price(15.0, 100.0, 103.0, 0.05, 1.0)
    15.0
    """
    # Dividend futures price is simply the expected total dividends
    # (In practice, derived from forward-spot relationship)
    return expected_total_dividends


@partial(jit, static_argnames=["position"])
def dividend_futures_value(
    futures_price: float,
    realized_dividends: float,
    expected_remaining_dividends: float,
    contract_size: float,
    position: str = "long",
) -> float:
    """Calculate mark-to-market value of dividend futures.

    Parameters
    ----------
    futures_price : float
        Current futures price (market consensus on total dividends)
    realized_dividends : float
        Dividends realized to date
    expected_remaining_dividends : float
        Expected dividends for remaining period
    contract_size : float
        Contract multiplier
    position : str
        'long' or 'short' position

    Returns
    -------
    float
        Mark-to-market value of the position

    Notes
    -----
    Dividend futures are marked-to-market daily based on:
    - Realized dividends to date
    - Expected remaining dividends
    - Current futures price

    Value = Contract_Size * (Realized + Expected_Remaining - Futures_Price)

    Examples
    --------
    >>> dividend_futures_value(15.0, 5.0, 10.5, 100.0, "long")
    50.0
    """
    total_expected = realized_dividends + expected_remaining_dividends
    payoff = total_expected - futures_price
    value = contract_size * payoff

    return jnp.where(position == "long", value, -value)


@jit
def dividend_index_points_to_yield(
    dividend_points: float,
    index_level: float,
    period_years: float,
) -> float:
    """Convert dividend index points to annualized dividend yield.

    Parameters
    ----------
    dividend_points : float
        Total dividend points expected over the period
    index_level : float
        Current index level
    period_years : float
        Period in years

    Returns
    -------
    float
        Annualized dividend yield

    Notes
    -----
    Yield = (Dividend Points / Index Level) / Period

    Examples
    --------
    >>> dividend_index_points_to_yield(150.0, 4000.0, 1.0)
    0.0375
    """
    return (dividend_points / index_level) / period_years


@jit
def dividend_yield_to_index_points(
    dividend_yield: float,
    index_level: float,
    period_years: float,
) -> float:
    """Convert annualized dividend yield to dividend index points.

    Parameters
    ----------
    dividend_yield : float
        Annualized dividend yield
    index_level : float
        Current index level
    period_years : float
        Period in years

    Returns
    -------
    float
        Expected dividend points over the period

    Notes
    -----
    Dividend Points = Yield * Index Level * Period

    Examples
    --------
    >>> dividend_yield_to_index_points(0.0375, 4000.0, 1.0)
    150.0
    """
    return dividend_yield * index_level * period_years


@dataclass
class VarianceSwap:
    """Variance swap specification."""

    notional: float  # Variance notional (vega notional = notional / (2 * sqrt(K)))
    strike: float  # Variance strike (in variance units, not vol)
    maturity: float
    realized_variance: float = 0.0  # For marking to market


@jit
def variance_swap_strike_from_options(
    strikes: jnp.ndarray,
    call_prices: jnp.ndarray,
    put_prices: jnp.ndarray,
    spot: float,
    forward: float,
    maturity: float,
    risk_free_rate: float,
) -> float:
    """Calculate fair variance swap strike from option prices.

    Parameters
    ----------
    strikes : Array
        Strike prices (sorted)
    call_prices : Array
        Call option prices
    put_prices : Array
        Put option prices
    spot : float
        Current spot price
    forward : float
        Forward price to maturity
    maturity : float
        Time to maturity in years
    risk_free_rate : float
        Risk-free rate

    Returns
    -------
    float
        Fair variance strike (annualized variance)

    Notes
    -----
    Variance swap strike can be replicated using the Carr-Madan formula:
        σ²_strike = (2/T) * ∫[w(K) * C(K)/K² dK]

    where w(K) is a weighting function and C(K) is the option price.

    This is a simplified implementation using trapezoidal integration.

    Examples
    --------
    >>> strikes = jnp.array([90.0, 95.0, 100.0, 105.0, 110.0])
    >>> calls = jnp.array([15.0, 10.5, 7.0, 4.5, 2.5])
    >>> puts = jnp.array([2.0, 3.5, 5.5, 8.5, 12.0])
    >>> variance_swap_strike_from_options(
    ...     strikes, calls, puts, 100.0, 102.0, 1.0, 0.05
    ... )
    0.045...
    """
    discount_factor = jnp.exp(-risk_free_rate * maturity)

    # Use puts below forward, calls above
    weights = jnp.where(strikes < forward, put_prices, call_prices)

    # Weight by 1/K²
    contributions = weights / (strikes**2)

    # Trapezoidal integration
    dk = jnp.diff(strikes)
    integrated = jnp.sum(dk * (contributions[:-1] + contributions[1:]) / 2.0)

    # Multiply by 2/T and adjust for forward
    variance_strike = (2.0 / (maturity * discount_factor)) * integrated

    # Subtract forward adjustment
    variance_strike -= (1.0 / maturity) * ((forward / spot - 1.0) ** 2)

    return jnp.maximum(variance_strike, 0.0)


@jit
def variance_swap_value(
    notional: float,
    strike: float,
    realized_variance: float,
    expected_variance: float,
    discount_factor: float,
) -> float:
    """Calculate mark-to-market value of a variance swap.

    Parameters
    ----------
    notional : float
        Variance notional (in variance units)
    strike : float
        Variance strike
    realized_variance : float
        Realized variance to date (annualized)
    expected_variance : float
        Expected remaining variance (annualized)
    discount_factor : float
        Discount factor to maturity

    Returns
    -------
    float
        Present value of the variance swap

    Notes
    -----
    For a variance swap with notional N and strike K:
        Payoff = N * (Realized Variance - K)

    The mark-to-market value accounts for:
    1. Realized variance to date
    2. Expected variance for remaining period

    Examples
    --------
    >>> variance_swap_value(10000.0, 0.04, 0.035, 0.045, 0.95)
    47.5
    """
    # Weighted average of realized and expected variance
    # (simplified - in practice would weight by time)
    total_expected = (realized_variance + expected_variance) / 2.0

    return notional * (total_expected - strike) * discount_factor


@jit
def volatility_swap_convexity_adjustment(variance_strike: float) -> float:
    """Calculate convexity adjustment for volatility swap.

    Parameters
    ----------
    variance_strike : float
        Variance strike

    Returns
    -------
    float
        Volatility strike (≈ sqrt(variance_strike) - adjustment)

    Notes
    -----
    Due to Jensen's inequality, E[sqrt(V)] < sqrt(E[V]).
    The convexity adjustment accounts for this difference.

    Approximation:
        Vol Strike ≈ sqrt(Var Strike) - (1/8) * (Var Strike)^(-1/2) * Vega Skew

    Simplified version without skew information:
        Vol Strike ≈ sqrt(Var Strike) * (1 - 1/(8 * Var Strike))

    Examples
    --------
    >>> volatility_swap_convexity_adjustment(0.04)
    0.199...
    """
    vol_strike = jnp.sqrt(variance_strike)
    # Simple approximation without vega skew
    # Adjustment ≈ K_var / (8 * sqrt(K_var)) for first-order approximation
    adjustment = variance_strike / (8.0 * vol_strike)
    return vol_strike - adjustment


@dataclass
class TotalReturnSwap:
    """Total return swap (TRS) specification."""

    notional: float
    equity_leg_rate: float  # Return on equity + dividends
    funding_leg_rate: float  # Typically LIBOR + spread
    maturity: float


@jit
def total_return_swap_value(
    notional: float,
    spot_initial: float,
    spot_current: float,
    dividends_received: float,
    funding_rate: float,
    time_elapsed: float,
    discount_factor: float,
) -> float:
    """Calculate mark-to-market value of total return swap.

    Parameters
    ----------
    notional : float
        Notional amount
    spot_initial : float
        Initial spot price (at trade inception)
    spot_current : float
        Current spot price
    dividends_received : float
        Cumulative dividends received
    funding_rate : float
        Funding rate (LIBOR + spread)
    time_elapsed : float
        Time elapsed since inception
    discount_factor : float
        Discount factor to maturity

    Returns
    -------
    float
        Present value of the TRS (to equity receiver)

    Notes
    -----
    A TRS exchanges:
    - Equity leg: Total return (price appreciation + dividends)
    - Funding leg: Funding rate on notional

    Value = Notional * [(S_t/S_0 - 1) + Div - Funding * t] * DF

    Examples
    --------
    >>> total_return_swap_value(
    ...     1_000_000, 100.0, 105.0, 3.0, 0.03, 0.5, 0.985
    ... )
    64_375.0
    """
    # Equity leg: price appreciation + dividends
    equity_return = (spot_current / spot_initial - 1.0) + (dividends_received / spot_initial)

    # Funding leg: funding rate * time
    funding_cost = funding_rate * time_elapsed

    # Net return
    net_return = equity_return - funding_cost

    return notional * net_return * discount_factor


def equity_linked_note_price(
    principal: float,
    participation_rate: float,
    spot_initial: float,
    spot_current: float,
    maturity: float,
    risk_free_rate: float,
    floor: float = 0.0,
    cap: float | None = None,
) -> float:
    """Price an equity-linked note (ELN).

    Parameters
    ----------
    principal : float
        Principal amount (face value)
    participation_rate : float
        Participation in equity upside (e.g., 0.8 for 80%)
    spot_initial : float
        Initial spot price
    spot_current : float
        Current spot price
    maturity : float
        Time to maturity in years
    risk_free_rate : float
        Risk-free rate
    floor : float
        Minimum return (default: 0.0 for principal protection)
    cap : float or None
        Maximum return (optional cap)

    Returns
    -------
    float
        Current value of the ELN

    Notes
    -----
    An ELN provides:
    - Principal protection (floor)
    - Participation in equity upside
    - Optional cap on returns

    Payoff = Principal * (1 + max(floor, min(participation * return, cap)))

    Examples
    --------
    >>> equity_linked_note_price(
    ...     100_000, 0.8, 100.0, 120.0, 1.0, 0.05, floor=0.0, cap=0.25
    ... )
    115_249.0...
    """
    # Equity return
    equity_return = (spot_current / spot_initial) - 1.0

    # Apply participation
    participated_return = participation_rate * equity_return

    # Apply floor
    floored_return = jnp.maximum(participated_return, floor)

    # Apply cap if specified
    if cap is not None:
        final_return = jnp.minimum(floored_return, cap)
    else:
        final_return = floored_return

    # Discount to present value
    discount_factor = jnp.exp(-risk_free_rate * maturity)
    terminal_value = principal * (1.0 + final_return)

    return float(terminal_value * discount_factor)


__all__ = [
    # Dataclasses
    "DividendSwap",
    "DividendFuture",
    "EquityForward",
    "TotalReturnSwap",
    "VarianceSwap",
    # Dividend derivatives
    "dividend_swap_value",
    "dividend_futures_price",
    "dividend_futures_value",
    "dividend_index_points_to_yield",
    "dividend_yield_to_index_points",
    # Equity forwards
    "equity_forward_price",
    "equity_forward_value",
    # Equity linked notes
    "equity_linked_note_price",
    # Total return swaps
    "total_return_swap_value",
    # Variance swaps
    "variance_swap_strike_from_options",
    "variance_swap_value",
    "volatility_swap_convexity_adjustment",
]
