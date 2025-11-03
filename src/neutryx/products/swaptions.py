"""European Swaption pricing and analytics.

A swaption is an option to enter into an interest rate swap. It gives
the holder the right (but not obligation) to enter into a swap at a
predetermined fixed rate (strike) at a future date (option maturity).

Types:
- Payer swaption: Right to pay fixed, receive floating
- Receiver swaption: Right to receive fixed, pay floating
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import jax.numpy as jnp
from jax import jit
from jax.scipy.stats import norm


class SwaptionType(Enum):
    """Swaption type enum."""

    PAYER = "payer"  # Option to pay fixed
    RECEIVER = "receiver"  # Option to receive fixed


@dataclass
class SwaptionSpecs:
    """Swaption specification parameters."""

    strike: float  # Fixed rate of the underlying swap
    option_maturity: float  # Time to swaption expiry in years
    swap_maturity: float  # Tenor of the underlying swap in years
    notional: float = 1_000_000.0
    payment_frequency: int = 2  # Payments per year (2 = semiannual)
    swaption_type: SwaptionType = SwaptionType.PAYER


@jit
def swap_annuity(
    discount_factors: jnp.ndarray, year_fractions: jnp.ndarray
) -> float:
    """Calculate the swap annuity factor (PV of 1bp per period).

    Parameters
    ----------
    discount_factors : Array
        Discount factors for each swap payment date
    year_fractions : Array
        Year fractions for each swap period

    Returns
    -------
    float
        Annuity factor

    Notes
    -----
    The annuity factor A is:
        A = Σ DF(t_i) * Δt_i

    where DF(t_i) is the discount factor and Δt_i is the year fraction.
    """
    return jnp.sum(discount_factors * year_fractions)


@jit
def forward_swap_rate(
    option_maturity: float,
    swap_maturity: float,
    discount_factors: jnp.ndarray,
    year_fractions: jnp.ndarray,
) -> float:
    """Calculate the forward swap rate.

    Parameters
    ----------
    option_maturity : float
        Time to swaption expiry
    swap_maturity : float
        Tenor of the underlying swap
    discount_factors : Array
        Discount factors for swap payment dates
    year_fractions : Array
        Year fractions for each period

    Returns
    -------
    float
        Forward swap rate

    Notes
    -----
    The forward swap rate S is:
        S = [DF(T) - DF(T+M)] / A

    where:
    - DF(T) is discount factor to option maturity
    - DF(T+M) is discount factor to end of swap
    - A is the annuity factor
    """
    df_start = jnp.exp(-0.03 * option_maturity)  # Placeholder
    df_end = discount_factors[-1]
    annuity = swap_annuity(discount_factors, year_fractions)

    return (df_start - df_end) / annuity


@jit
def black_swaption_price(
    forward_swap_rate: float,
    strike: float,
    option_maturity: float,
    volatility: float,
    annuity: float,
    notional: float = 1_000_000.0,
    is_payer: bool = True,
) -> float:
    """Price a European swaption using Black's formula.

    Parameters
    ----------
    forward_swap_rate : float
        Forward swap rate
    strike : float
        Strike rate of the swaption
    option_maturity : float
        Time to swaption expiry in years
    volatility : float
        Volatility of the forward swap rate (log-normal)
    annuity : float
        Swap annuity factor (present value of basis point)
    notional : float
        Notional principal
    is_payer : bool
        True for payer swaption, False for receiver

    Returns
    -------
    float
        Swaption price

    Notes
    -----
    Black's formula for swaptions:
        Payer: V = A * N * [F * Φ(d1) - K * Φ(d2)]
        Receiver: V = A * N * [K * Φ(-d2) - F * Φ(-d1)]

    where:
        d1 = [ln(F/K) + 0.5*σ²*T] / (σ*√T)
        d2 = d1 - σ*√T
        F = forward swap rate
        K = strike
        σ = volatility
        T = option maturity
        A = annuity factor
        N = notional
        Φ = cumulative normal distribution

    Examples
    --------
    >>> # 1-year payer swaption on 5-year swap, 5% strike, 20% vol
    >>> black_swaption_price(0.05, 0.05, 1.0, 0.20, 4.5, 1_000_000, True)
    """
    # Handle zero volatility or maturity using jnp.where for JAX compatibility
    is_zero_vol = (volatility < 1e-10) | (option_maturity < 1e-10)

    # Intrinsic value calculation
    payer_intrinsic = jnp.maximum(forward_swap_rate - strike, 0.0)
    receiver_intrinsic = jnp.maximum(strike - forward_swap_rate, 0.0)
    intrinsic = jnp.where(is_payer, payer_intrinsic, receiver_intrinsic)
    intrinsic_value = notional * annuity * intrinsic

    # Black-Scholes formula (safe division with jnp.maximum to avoid division by zero)
    sqrt_T = jnp.sqrt(jnp.maximum(option_maturity, 1e-10))
    vol_sqrt_T = volatility * sqrt_T
    log_moneyness = jnp.log(forward_swap_rate / strike)
    d1 = (log_moneyness + 0.5 * volatility * volatility * option_maturity) / jnp.maximum(
        vol_sqrt_T, 1e-10
    )
    d2 = d1 - vol_sqrt_T

    # Payer vs Receiver
    payer_value = forward_swap_rate * norm.cdf(d1) - strike * norm.cdf(d2)
    receiver_value = strike * norm.cdf(-d2) - forward_swap_rate * norm.cdf(-d1)
    black_value = notional * annuity * jnp.where(is_payer, payer_value, receiver_value)

    # Return intrinsic if zero vol/maturity, otherwise Black value
    return jnp.where(is_zero_vol, intrinsic_value, black_value)


def european_swaption_black(
    strike: float,
    option_maturity: float,
    swap_maturity: float,
    volatility: float,
    discount_rate: float = 0.03,
    notional: float = 1_000_000.0,
    payment_frequency: int = 2,
    is_payer: bool = True,
) -> float:
    """Price a European swaption using Black's formula.

    Parameters
    ----------
    strike : float
        Fixed rate (strike) of the swaption
    option_maturity : float
        Time to swaption expiry in years
    swap_maturity : float
        Tenor of the underlying swap in years
    volatility : float
        Implied volatility of the forward swap rate
    discount_rate : float
        Risk-free discount rate (for simplified curve)
    notional : float
        Notional principal amount
    payment_frequency : int
        Number of payments per year
    is_payer : bool
        True for payer swaption, False for receiver

    Returns
    -------
    float
        Swaption price

    Examples
    --------
    >>> # Price a 1-year payer swaption on 5-year swap
    >>> european_swaption_black(
    ...     strike=0.05,
    ...     option_maturity=1.0,
    ...     swap_maturity=5.0,
    ...     volatility=0.20,
    ...     notional=1_000_000,
    ...     is_payer=True
    ... )
    """
    # Generate swap payment schedule
    n_payments = int(swap_maturity * payment_frequency)
    year_fractions = jnp.full(n_payments, 1.0 / payment_frequency, dtype=jnp.float32)

    # Generate discount factors (start from option maturity)
    payment_times = option_maturity + jnp.arange(1, n_payments + 1, dtype=jnp.float32) / payment_frequency
    discount_factors = jnp.exp(-discount_rate * payment_times)

    # Calculate annuity
    annuity = float(swap_annuity(discount_factors, year_fractions))

    # Calculate forward swap rate (simplified)
    df_start = jnp.exp(-discount_rate * option_maturity)
    df_end = jnp.exp(-discount_rate * (option_maturity + swap_maturity))
    fwd_swap_rate = float((df_start - df_end) / annuity)

    # Price using Black's formula
    return float(
        black_swaption_price(
            fwd_swap_rate,
            strike,
            option_maturity,
            volatility,
            annuity,
            notional,
            is_payer,
        )
    )


@jit
def swaption_vega(
    forward_swap_rate: float,
    strike: float,
    option_maturity: float,
    volatility: float,
    annuity: float,
    notional: float = 1_000_000.0,
) -> float:
    """Calculate swaption vega (sensitivity to volatility change).

    Parameters
    ----------
    forward_swap_rate : float
        Forward swap rate
    strike : float
        Strike rate
    option_maturity : float
        Time to expiry
    volatility : float
        Volatility of forward swap rate
    annuity : float
        Swap annuity factor
    notional : float
        Notional amount

    Returns
    -------
    float
        Vega (price change per 1% volatility change)

    Notes
    -----
    Vega = A * N * F * φ(d1) * √T

    where φ is the standard normal PDF.
    """
    sqrt_T = jnp.sqrt(option_maturity)
    log_moneyness = jnp.log(forward_swap_rate / strike)
    d1 = (log_moneyness + 0.5 * volatility * volatility * option_maturity) / (volatility * sqrt_T)

    # Standard normal PDF
    phi_d1 = jnp.exp(-0.5 * d1 * d1) / jnp.sqrt(2.0 * jnp.pi)

    return notional * annuity * forward_swap_rate * phi_d1 * sqrt_T


def swaption_delta(
    forward_swap_rate: float,
    strike: float,
    option_maturity: float,
    volatility: float,
    annuity: float,
    notional: float = 1_000_000.0,
    is_payer: bool = True,
) -> float:
    """Calculate swaption delta (sensitivity to forward swap rate).

    Parameters
    ----------
    forward_swap_rate : float
        Forward swap rate
    strike : float
        Strike rate
    option_maturity : float
        Time to expiry
    volatility : float
        Volatility
    annuity : float
        Annuity factor
    notional : float
        Notional
    is_payer : bool
        Payer or receiver

    Returns
    -------
    float
        Delta (price change per unit change in forward rate)

    Notes
    -----
    Delta_payer = A * N * Φ(d1)
    Delta_receiver = -A * N * Φ(-d1)
    """
    sqrt_T = jnp.sqrt(option_maturity)
    log_moneyness = jnp.log(forward_swap_rate / strike)
    d1 = (log_moneyness + 0.5 * volatility * volatility * option_maturity) / (volatility * sqrt_T)

    if is_payer:
        delta = float(notional * annuity * norm.cdf(d1))
    else:
        delta = float(-notional * annuity * norm.cdf(-d1))

    return delta


def implied_swaption_volatility(
    market_price: float,
    forward_swap_rate: float,
    strike: float,
    option_maturity: float,
    annuity: float,
    notional: float = 1_000_000.0,
    is_payer: bool = True,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> float:
    """Calculate implied volatility from market swaption price.

    Uses bisection method to solve for implied volatility.

    Parameters
    ----------
    market_price : float
        Observed market price of the swaption
    forward_swap_rate : float
        Forward swap rate
    strike : float
        Strike rate
    option_maturity : float
        Time to expiry
    annuity : float
        Annuity factor
    notional : float
        Notional amount
    is_payer : bool
        Payer or receiver swaption
    max_iterations : int
        Maximum iterations for bisection
    tolerance : float
        Convergence tolerance

    Returns
    -------
    float
        Implied volatility

    Examples
    --------
    >>> # Find implied vol from market price
    >>> implied_swaption_volatility(
    ...     market_price=50000,
    ...     forward_swap_rate=0.05,
    ...     strike=0.05,
    ...     option_maturity=1.0,
    ...     annuity=4.5,
    ...     notional=1_000_000
    ... )
    """
    # Bisection bounds
    vol_low = 0.001  # 0.1%
    vol_high = 3.0  # 300%

    for _ in range(max_iterations):
        vol_mid = (vol_low + vol_high) / 2.0

        calculated_price = float(
            black_swaption_price(
                forward_swap_rate,
                strike,
                option_maturity,
                vol_mid,
                annuity,
                notional,
                is_payer,
            )
        )

        price_diff = calculated_price - market_price

        if abs(price_diff) < tolerance:
            return vol_mid

        # Update bounds
        if price_diff > 0:  # Calculated price too high, vol too high
            vol_high = vol_mid
        else:
            vol_low = vol_mid

        # Check convergence on vol
        if abs(vol_high - vol_low) < tolerance / 1000:
            break

    return vol_mid


__all__ = [
    "SwaptionSpecs",
    "SwaptionType",
    "black_swaption_price",
    "european_swaption_black",
    "forward_swap_rate",
    "implied_swaption_volatility",
    "swap_annuity",
    "swaption_delta",
    "swaption_vega",
]
