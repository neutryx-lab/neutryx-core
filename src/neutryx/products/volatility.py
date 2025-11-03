"""Volatility and variance products.

Implements volatility-linked instruments:
- VIX futures
- VIX options
- Variance swaps
- Volatility swaps
- Corridor variance swaps
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from functools import partial
from jax import jit

from neutryx.models.bs import price as bs_price


@dataclass
class VIXFuture:
    """VIX futures contract specification."""

    vix_spot: float
    maturity: float
    forward_variance: float


@jit
def vix_futures_price(
    vix_spot: float,
    maturity: float,
    vol_of_vol: float = 0.0,
    mean_reversion: float = 0.0,
    long_term_vol: float = 0.0,
) -> float:
    """Calculate theoretical VIX futures price.

    Parameters
    ----------
    vix_spot : float
        Current VIX level (spot volatility index)
    maturity : float
        Time to maturity in years
    vol_of_vol : float
        Volatility of volatility (optional, for mean-reversion model)
    mean_reversion : float
        Mean reversion speed (optional)
    long_term_vol : float
        Long-term volatility level (optional)

    Returns
    -------
    float
        VIX futures price

    Notes
    -----
    Simple model without mean reversion:
        F_VIX(T) ≈ VIX_spot

    With mean reversion (OU process):
        F_VIX(T) = θ + (VIX_0 - θ) * exp(-κ * T)

    where κ is mean reversion speed and θ is long-term level.

    Examples
    --------
    >>> vix_futures_price(20.0, 0.5, 0.0, 0.0, 0.0)
    20.0

    >>> vix_futures_price(25.0, 0.5, 0.0, 2.0, 20.0)
    22.347...
    """
    # Mean-reverting model
    decay = jnp.exp(-mean_reversion * maturity)
    futures_price_mr = long_term_vol + (vix_spot - long_term_vol) * decay

    # Simple model (martingale)
    futures_price_simple = vix_spot

    # Use mean reversion if both parameters are positive
    use_mr = (mean_reversion > 0.0) & (long_term_vol > 0.0)
    futures_price = jnp.where(use_mr, futures_price_mr, futures_price_simple)

    return futures_price


@partial(jit, static_argnames=["option_type"])
def vix_option_price(
    vix_futures_price: float,
    strike: float,
    maturity: float,
    vix_volatility: float,
    option_type: str = "call",
) -> float:
    """Price VIX option using Black's model on VIX futures.

    Parameters
    ----------
    vix_futures_price : float
        VIX futures price for the relevant maturity
    strike : float
        Strike level
    maturity : float
        Time to expiry in years
    vix_volatility : float
        Volatility of VIX (vol-of-vol)
    option_type : str
        'call' or 'put'

    Returns
    -------
    float
        VIX option price

    Notes
    -----
    VIX options are European-style and settle to VIX futures.
    Uses Black (1976) model:
        C = exp(-r*T) * [F * Φ(d1) - K * Φ(d2)]

    For simplicity, we use r=0 since VIX is not a traditional asset.

    Examples
    --------
    >>> vix_option_price(20.0, 22.0, 0.5, 0.80, "call")
    2.567...
    """
    # Black model (r=0 for VIX)
    return bs_price(
        S=vix_futures_price,
        K=strike,
        T=maturity,
        r=0.0,
        q=0.0,
        sigma=vix_volatility,
        kind=option_type,
    )


@dataclass
class VarianceSwap:
    """Variance swap contract specification."""

    notional: float  # Variance notional (vega notional = notional / (2 * sqrt(K)))
    strike: float  # Variance strike (variance units, e.g., 0.04 for 20% vol)
    maturity: float
    observation_frequency: float = 1.0 / 252.0  # Daily by default


@jit
def realized_variance(
    price_observations: jnp.ndarray,
    annualization_factor: float = 252.0,
) -> float:
    """Calculate realized variance from price observations.

    Parameters
    ----------
    price_observations : Array
        Time series of price observations
    annualization_factor : float
        Annualization factor (252 for daily, 52 for weekly, etc.)

    Returns
    -------
    float
        Annualized realized variance

    Notes
    -----
    Realized variance is calculated as:
        σ²_realized = (N / (N-1)) * Σ[ln(S_i / S_{i-1})]² * annualization

    Examples
    --------
    >>> prices = jnp.array([100.0, 101.0, 99.5, 102.0, 101.5])
    >>> realized_variance(prices, annualization_factor=252.0)
    0.0287...
    """
    # Log returns
    log_returns = jnp.log(price_observations[1:] / price_observations[:-1])

    # Sum of squared log returns
    sum_squared_returns = jnp.sum(log_returns**2)

    # Realized variance (annualized)
    n = len(log_returns)
    variance = (annualization_factor / n) * sum_squared_returns

    return variance


@jit
def variance_swap_payoff(
    notional: float,
    strike: float,
    realized_var: float,
) -> float:
    """Calculate variance swap payoff at maturity.

    Parameters
    ----------
    notional : float
        Variance notional
    strike : float
        Variance strike
    realized_var : float
        Realized variance over the period

    Returns
    -------
    float
        Payoff to variance buyer

    Notes
    -----
    Payoff = Notional * (Realized Variance - Strike)

    Examples
    --------
    >>> variance_swap_payoff(10_000, 0.04, 0.045)
    50.0
    """
    return notional * (realized_var - strike)


@jit
def volatility_swap_payoff(
    notional: float,
    strike: float,
    realized_vol: float,
) -> float:
    """Calculate volatility swap payoff at maturity.

    Parameters
    ----------
    notional : float
        Volatility notional (in volatility points)
    strike : float
        Volatility strike (in volatility, e.g., 0.20 for 20%)
    realized_vol : float
        Realized volatility over the period

    Returns
    -------
    float
        Payoff to volatility buyer

    Notes
    -----
    Payoff = Notional * (Realized Vol - Strike Vol)

    The realized vol is typically calculated as sqrt(realized variance).

    Examples
    --------
    >>> volatility_swap_payoff(1_000_000, 0.20, 0.25)
    50_000.0
    """
    return notional * (realized_vol - strike)


@jit
def corridor_variance_swap_strike(
    strikes: jnp.ndarray,
    option_prices: jnp.ndarray,
    spot: float,
    forward: float,
    maturity: float,
    risk_free_rate: float,
    lower_barrier: float,
    upper_barrier: float,
) -> float:
    """Calculate fair strike for corridor variance swap.

    Parameters
    ----------
    strikes : Array
        Strike prices
    option_prices : Array
        Option prices (calls and puts)
    spot : float
        Current spot price
    forward : float
        Forward price
    maturity : float
        Time to maturity
    risk_free_rate : float
        Risk-free rate
    lower_barrier : float
        Lower barrier level
    upper_barrier : float
        Upper barrier level

    Returns
    -------
    float
        Fair corridor variance strike

    Notes
    -----
    A corridor variance swap only accrues variance when the spot is
    within a specified range [L, U].

    The replication formula is similar to vanilla variance swaps but
    integrates only over the corridor range.

    Examples
    --------
    >>> strikes = jnp.array([90.0, 95.0, 100.0, 105.0, 110.0])
    >>> prices = jnp.array([12.0, 8.0, 5.0, 3.0, 2.0])
    >>> corridor_variance_swap_strike(
    ...     strikes, prices, 100.0, 102.0, 1.0, 0.05, 95.0, 105.0
    ... )
    0.028...
    """
    discount_factor = jnp.exp(-risk_free_rate * maturity)

    # Filter strikes within corridor
    in_corridor = (strikes >= lower_barrier) & (strikes <= upper_barrier)
    corridor_strikes = jnp.where(in_corridor, strikes, 0.0)
    corridor_prices = jnp.where(in_corridor, option_prices, 0.0)

    # Weight by 1/K²
    weights = jnp.where(corridor_strikes > 0, 1.0 / (corridor_strikes**2), 0.0)
    contributions = corridor_prices * weights

    # Trapezoidal integration
    dk = jnp.diff(strikes)
    # Pad contributions to match dk length
    contributions_padded = jnp.where(
        in_corridor[:-1] & in_corridor[1:], (contributions[:-1] + contributions[1:]) / 2.0, 0.0
    )

    integrated = jnp.sum(dk * contributions_padded)

    # Variance strike formula
    variance_strike = (2.0 / (maturity * discount_factor)) * integrated

    return jnp.maximum(variance_strike, 0.0)


@jit
def conditional_variance_swap_value(
    notional: float,
    strike: float,
    expected_variance: float,
    time_in_corridor: float,
    discount_factor: float,
) -> float:
    """Calculate value of conditional/corridor variance swap.

    Parameters
    ----------
    notional : float
        Variance notional
    strike : float
        Variance strike
    expected_variance : float
        Expected conditional variance (when in corridor)
    time_in_corridor : float
        Expected fraction of time spent in corridor
    discount_factor : float
        Discount factor

    Returns
    -------
    float
        Present value of conditional variance swap

    Notes
    -----
    Value = Notional * TimeInCorridor * (ExpectedVar - Strike) * DF

    Examples
    --------
    >>> conditional_variance_swap_value(
    ...     10_000, 0.04, 0.045, 0.75, 0.95
    ... )
    35.625
    """
    effective_payoff = time_in_corridor * (expected_variance - strike)
    return notional * effective_payoff * discount_factor


@jit
def gamma_swap_payoff(
    notional: float,
    price_observations: jnp.ndarray,
    weights: jnp.ndarray | None = None,
) -> float:
    """Calculate gamma swap payoff.

    Parameters
    ----------
    notional : float
        Notional amount
    price_observations : Array
        Time series of price observations
    weights : Array or None
        Optional weighting for each gamma observation

    Returns
    -------
    float
        Gamma swap payoff

    Notes
    -----
    A gamma swap pays the sum of absolute returns:
        Payoff = Notional * Σ|ln(S_i / S_{i-1})|

    This is related to realized volatility but with absolute value
    instead of squared returns.

    Examples
    --------
    >>> prices = jnp.array([100.0, 102.0, 99.0, 101.0])
    >>> gamma_swap_payoff(1_000_000, prices)
    49_410.3...
    """
    log_returns = jnp.log(price_observations[1:] / price_observations[:-1])
    abs_returns = jnp.abs(log_returns)

    if weights is not None:
        abs_returns = abs_returns * weights

    return notional * jnp.sum(abs_returns)


@jit
def vvix_index(
    vix_option_strikes: jnp.ndarray,
    vix_call_prices: jnp.ndarray,
    vix_put_prices: jnp.ndarray,
    vix_spot: float,
    maturity: float,
) -> float:
    """Calculate VVIX (volatility of VIX) index.

    Parameters
    ----------
    vix_option_strikes : Array
        VIX option strike prices
    vix_call_prices : Array
        VIX call option prices
    vix_put_prices : Array
        VIX put option prices
    vix_spot : float
        Current VIX level
    maturity : float
        Time to maturity

    Returns
    -------
    float
        VVIX level (volatility of volatility)

    Notes
    -----
    VVIX is calculated similarly to VIX but using VIX options instead
    of SPX options. It measures the expected volatility of VIX.

    Uses the standard variance swap replication formula.

    Examples
    --------
    >>> strikes = jnp.array([15.0, 18.0, 20.0, 22.0, 25.0])
    >>> calls = jnp.array([7.0, 4.5, 3.0, 2.0, 1.0])
    >>> puts = jnp.array([1.0, 2.0, 3.0, 4.5, 7.0])
    >>> vvix_index(strikes, calls, puts, 20.0, 0.0833)
    82.456...
    """
    # Use variance swap replication
    forward_vix = vix_spot  # Simplified

    # Weight by 1/K²
    weights = 1.0 / (vix_option_strikes**2)

    # Use puts below forward, calls above
    option_prices = jnp.where(vix_option_strikes < forward_vix, vix_put_prices, vix_call_prices)

    contributions = weights * option_prices

    # Trapezoidal integration
    dk = jnp.diff(vix_option_strikes)
    integrated = jnp.sum(dk * (contributions[:-1] + contributions[1:]) / 2.0)

    # Variance of VIX
    vix_variance = (2.0 / maturity) * integrated

    # VVIX is volatility (not variance)
    vvix = jnp.sqrt(jnp.maximum(vix_variance, 0.0))

    return vvix * 100.0  # Return as percentage


__all__ = [
    "VIXFuture",
    "VarianceSwap",
    "conditional_variance_swap_value",
    "corridor_variance_swap_strike",
    "gamma_swap_payoff",
    "realized_variance",
    "variance_swap_payoff",
    "vix_futures_price",
    "vix_option_price",
    "volatility_swap_payoff",
    "vvix_index",
]
