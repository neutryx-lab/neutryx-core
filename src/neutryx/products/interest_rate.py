"""Interest Rate Derivatives: Caps, Floors, Digital caplets, Range accruals, CMS spread options.

This module implements various interest rate derivatives including:
- Interest rate caps and floors with SABR/LMM pricing
- Digital caplets and floorlets
- Range accrual notes
- CMS (Constant Maturity Swap) spread options
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
from jax import Array, jit
from jax.scipy.stats import norm


@dataclass
class CapFloorSpecs:
    """Specification for interest rate cap/floor."""
    strike: float  # Cap/floor rate
    notional: float = 1_000_000.0
    maturity: float = 5.0  # Years
    payment_frequency: int = 4  # Quarterly
    is_cap: bool = True  # True for cap, False for floor
    reference_rate: str = "LIBOR"  # "LIBOR" or "SOFR"


@jit
def black_caplet_price(
    forward_rate: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    discount_factor: float,
    year_fraction: float,
    notional: float = 1_000_000.0,
) -> float:
    """Price a single caplet using Black's formula.

    Parameters
    ----------
    forward_rate : float
        Forward LIBOR rate
    strike : float
        Strike rate
    time_to_expiry : float
        Time to caplet expiry
    volatility : float
        Black volatility
    discount_factor : float
        Discount factor to payment date
    year_fraction : float
        Accrual period (e.g., 0.25 for quarterly)
    notional : float
        Notional principal

    Returns
    -------
    float
        Caplet price
    """
    sqrt_T = jnp.sqrt(time_to_expiry)
    d1 = (jnp.log(forward_rate / strike) + 0.5 * volatility**2 * time_to_expiry) / (
        volatility * sqrt_T
    )
    d2 = d1 - volatility * sqrt_T

    caplet_value = (
        notional
        * year_fraction
        * discount_factor
        * (forward_rate * norm.cdf(d1) - strike * norm.cdf(d2))
    )

    return caplet_value


@jit
def black_floorlet_price(
    forward_rate: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    discount_factor: float,
    year_fraction: float,
    notional: float = 1_000_000.0,
) -> float:
    """Price a single floorlet using Black's formula.

    Parameters
    ----------
    forward_rate : float
        Forward LIBOR rate
    strike : float
        Strike rate
    time_to_expiry : float
        Time to floorlet expiry
    volatility : float
        Black volatility
    discount_factor : float
        Discount factor to payment date
    year_fraction : float
        Accrual period
    notional : float
        Notional principal

    Returns
    -------
    float
        Floorlet price
    """
    sqrt_T = jnp.sqrt(time_to_expiry)
    d1 = (jnp.log(forward_rate / strike) + 0.5 * volatility**2 * time_to_expiry) / (
        volatility * sqrt_T
    )
    d2 = d1 - volatility * sqrt_T

    floorlet_value = (
        notional
        * year_fraction
        * discount_factor
        * (strike * norm.cdf(-d2) - forward_rate * norm.cdf(-d1))
    )

    return floorlet_value


def price_cap(
    forward_rates: Array,
    strike: float,
    times_to_expiry: Array,
    volatilities: Array,
    discount_factors: Array,
    year_fractions: Array,
    notional: float = 1_000_000.0,
) -> float:
    """Price an interest rate cap as a portfolio of caplets.

    Parameters
    ----------
    forward_rates : Array
        Forward LIBOR rates for each period
    strike : float
        Cap strike rate
    times_to_expiry : Array
        Times to expiry for each caplet
    volatilities : Array
        Black volatilities for each caplet
    discount_factors : Array
        Discount factors for each payment
    year_fractions : Array
        Year fractions for each period
    notional : float
        Notional principal

    Returns
    -------
    float
        Cap price (sum of caplet prices)
    """
    caplet_prices = jnp.array([
        black_caplet_price(
            forward_rates[i],
            strike,
            times_to_expiry[i],
            volatilities[i],
            discount_factors[i],
            year_fractions[i],
            notional,
        )
        for i in range(len(forward_rates))
    ])

    return float(jnp.sum(caplet_prices))


def price_floor(
    forward_rates: Array,
    strike: float,
    times_to_expiry: Array,
    volatilities: Array,
    discount_factors: Array,
    year_fractions: Array,
    notional: float = 1_000_000.0,
) -> float:
    """Price an interest rate floor as a portfolio of floorlets.

    Parameters
    ----------
    forward_rates : Array
        Forward LIBOR rates for each period
    strike : float
        Floor strike rate
    times_to_expiry : Array
        Times to expiry for each floorlet
    volatilities : Array
        Black volatilities for each floorlet
    discount_factors : Array
        Discount factors for each payment
    year_fractions : Array
        Year fractions for each period
    notional : float
        Notional principal

    Returns
    -------
    float
        Floor price (sum of floorlet prices)
    """
    floorlet_prices = jnp.array([
        black_floorlet_price(
            forward_rates[i],
            strike,
            times_to_expiry[i],
            volatilities[i],
            discount_factors[i],
            year_fractions[i],
            notional,
        )
        for i in range(len(forward_rates))
    ])

    return float(jnp.sum(floorlet_prices))


# SOFR-specific pricing
@jit
def sofr_caplet_price(
    forward_rate: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    discount_factor: float,
    compounding_days: int,
    notional: float = 1_000_000.0,
) -> float:
    """Price a SOFR caplet with daily compounding.

    SOFR caplets differ from LIBOR caplets in that SOFR is compounded daily.

    Parameters
    ----------
    forward_rate : float
        Forward SOFR rate (daily compounded equivalent)
    strike : float
        Strike rate
    time_to_expiry : float
        Time to caplet expiry
    volatility : float
        Black volatility
    discount_factor : float
        Discount factor to payment date
    compounding_days : int
        Number of compounding days in the period (e.g., 90 for quarterly)
    notional : float
        Notional principal

    Returns
    -------
    float
        SOFR caplet price

    Notes
    -----
    SOFR compounds daily using:
        (1 + SOFR_daily)^n - 1

    where n is the number of days. The pricing uses Black's formula on the
    compounded rate.
    """
    # Convert simple rate to compounded rate (approximation)
    # In practice, would use exact SOFR compounding formula
    year_fraction = compounding_days / 360.0  # SOFR uses Act/360

    # Use Black's formula with adjusted year fraction
    sqrt_T = jnp.sqrt(time_to_expiry)
    d1 = (jnp.log(forward_rate / strike) + 0.5 * volatility**2 * time_to_expiry) / (
        volatility * sqrt_T
    )
    d2 = d1 - volatility * sqrt_T

    caplet_value = (
        notional
        * year_fraction
        * discount_factor
        * (forward_rate * norm.cdf(d1) - strike * norm.cdf(d2))
    )

    return caplet_value


@jit
def sofr_floorlet_price(
    forward_rate: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    discount_factor: float,
    compounding_days: int,
    notional: float = 1_000_000.0,
) -> float:
    """Price a SOFR floorlet with daily compounding.

    Parameters
    ----------
    forward_rate : float
        Forward SOFR rate (daily compounded equivalent)
    strike : float
        Strike rate
    time_to_expiry : float
        Time to floorlet expiry
    volatility : float
        Black volatility
    discount_factor : float
        Discount factor to payment date
    compounding_days : int
        Number of compounding days in the period
    notional : float
        Notional principal

    Returns
    -------
    float
        SOFR floorlet price
    """
    year_fraction = compounding_days / 360.0  # SOFR uses Act/360

    sqrt_T = jnp.sqrt(time_to_expiry)
    d1 = (jnp.log(forward_rate / strike) + 0.5 * volatility**2 * time_to_expiry) / (
        volatility * sqrt_T
    )
    d2 = d1 - volatility * sqrt_T

    floorlet_value = (
        notional
        * year_fraction
        * discount_factor
        * (strike * norm.cdf(-d2) - forward_rate * norm.cdf(-d1))
    )

    return floorlet_value


def price_sofr_cap(
    forward_rates: Array,
    strike: float,
    times_to_expiry: Array,
    volatilities: Array,
    discount_factors: Array,
    compounding_days: Array,
    notional: float = 1_000_000.0,
) -> float:
    """Price a SOFR cap as a portfolio of SOFR caplets.

    Parameters
    ----------
    forward_rates : Array
        Forward SOFR rates for each period
    strike : float
        Cap strike rate
    times_to_expiry : Array
        Times to expiry for each caplet
    volatilities : Array
        Black volatilities for each caplet
    discount_factors : Array
        Discount factors for each payment
    compounding_days : Array
        Compounding days for each period
    notional : float
        Notional principal

    Returns
    -------
    float
        SOFR cap price
    """
    caplet_prices = jnp.array([
        sofr_caplet_price(
            forward_rates[i],
            strike,
            times_to_expiry[i],
            volatilities[i],
            discount_factors[i],
            int(compounding_days[i]),
            notional,
        )
        for i in range(len(forward_rates))
    ])

    return float(jnp.sum(caplet_prices))


def price_sofr_floor(
    forward_rates: Array,
    strike: float,
    times_to_expiry: Array,
    volatilities: Array,
    discount_factors: Array,
    compounding_days: Array,
    notional: float = 1_000_000.0,
) -> float:
    """Price a SOFR floor as a portfolio of SOFR floorlets.

    Parameters
    ----------
    forward_rates : Array
        Forward SOFR rates for each period
    strike : float
        Floor strike rate
    times_to_expiry : Array
        Times to expiry for each floorlet
    volatilities : Array
        Black volatilities for each floorlet
    discount_factors : Array
        Discount factors for each payment
    compounding_days : Array
        Compounding days for each period
    notional : float
        Notional principal

    Returns
    -------
    float
        SOFR floor price
    """
    floorlet_prices = jnp.array([
        sofr_floorlet_price(
            forward_rates[i],
            strike,
            times_to_expiry[i],
            volatilities[i],
            discount_factors[i],
            int(compounding_days[i]),
            notional,
        )
        for i in range(len(forward_rates))
    ])

    return float(jnp.sum(floorlet_prices))


# SABR model for cap/floor volatility
def sabr_implied_vol_caplet(
    forward_rate: float,
    strike: float,
    time_to_expiry: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """Calculate SABR implied volatility for caplet.

    Uses Hagan's formula for SABR model.

    Parameters
    ----------
    forward_rate : float
        Forward LIBOR rate
    strike : float
        Strike rate
    time_to_expiry : float
        Time to expiry
    alpha : float
        Initial volatility level
    beta : float
        CEV exponent
    rho : float
        Correlation
    nu : float
        Volatility of volatility

    Returns
    -------
    float
        Implied volatility
    """
    from neutryx.models.sabr import hagan_implied_vol, SABRParams

    params = SABRParams(alpha=alpha, beta=beta, rho=rho, nu=nu)
    return float(hagan_implied_vol(forward_rate, strike, time_to_expiry, params))


# Digital caplets and floorlets
@jit
def digital_caplet_price(
    forward_rate: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    discount_factor: float,
    year_fraction: float,
    notional: float = 1_000_000.0,
    payout: float = 1.0,
) -> float:
    """Price a digital caplet (pays fixed amount if rate > strike).

    Parameters
    ----------
    forward_rate : float
        Forward LIBOR rate
    strike : float
        Strike rate
    time_to_expiry : float
        Time to expiry
    volatility : float
        Black volatility
    discount_factor : float
        Discount factor
    year_fraction : float
        Accrual period
    notional : float
        Notional principal
    payout : float
        Fixed payout amount (as fraction of notional)

    Returns
    -------
    float
        Digital caplet price
    """
    sqrt_T = jnp.sqrt(time_to_expiry)
    d2 = (jnp.log(forward_rate / strike) - 0.5 * volatility**2 * time_to_expiry) / (
        volatility * sqrt_T
    )

    digital_value = (
        notional * payout * year_fraction * discount_factor * norm.cdf(d2)
    )

    return digital_value


@jit
def digital_floorlet_price(
    forward_rate: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    discount_factor: float,
    year_fraction: float,
    notional: float = 1_000_000.0,
    payout: float = 1.0,
) -> float:
    """Price a digital floorlet (pays fixed amount if rate < strike).

    Parameters
    ----------
    forward_rate : float
        Forward LIBOR rate
    strike : float
        Strike rate
    time_to_expiry : float
        Time to expiry
    volatility : float
        Black volatility
    discount_factor : float
        Discount factor
    year_fraction : float
        Accrual period
    notional : float
        Notional principal
    payout : float
        Fixed payout amount (as fraction of notional)

    Returns
    -------
    float
        Digital floorlet price
    """
    sqrt_T = jnp.sqrt(time_to_expiry)
    d2 = (jnp.log(forward_rate / strike) - 0.5 * volatility**2 * time_to_expiry) / (
        volatility * sqrt_T
    )

    digital_value = (
        notional * payout * year_fraction * discount_factor * norm.cdf(-d2)
    )

    return digital_value


# Range accrual
@dataclass
class RangeAccrualSpecs:
    """Specification for range accrual note."""
    lower_barrier: float
    upper_barrier: float
    notional: float = 1_000_000.0
    coupon_rate: float = 0.05
    maturity: float = 1.0
    observation_frequency: int = 252  # Daily observations


def price_range_accrual_mc(
    rate_paths: Array,
    lower_barrier: float,
    upper_barrier: float,
    coupon_rate: float,
    discount_factor: float,
    notional: float = 1_000_000.0,
) -> float:
    """Price range accrual note using Monte Carlo.

    The note pays coupon * (days in range / total days).

    Parameters
    ----------
    rate_paths : Array
        Simulated interest rate paths [n_paths, n_steps]
    lower_barrier : float
        Lower barrier rate
    upper_barrier : float
        Upper barrier rate
    coupon_rate : float
        Coupon rate if always in range
    discount_factor : float
        Discount factor to maturity
    notional : float
        Notional principal

    Returns
    -------
    float
        Range accrual price
    """
    # Count days in range for each path
    in_range = (rate_paths >= lower_barrier) & (rate_paths <= upper_barrier)
    fraction_in_range = jnp.mean(in_range, axis=1)  # Average across time for each path

    # Payoff for each path
    payoffs = notional * coupon_rate * fraction_in_range

    # Discounted average payoff
    price = discount_factor * jnp.mean(payoffs)

    return float(price)


# CMS spread options
@dataclass
class CMSSpreadOptionSpecs:
    """Specification for CMS spread option."""
    strike: float
    notional: float = 1_000_000.0
    maturity: float = 1.0
    cms1_tenor: float = 10.0  # e.g., 10-year CMS rate
    cms2_tenor: float = 2.0  # e.g., 2-year CMS rate
    is_call: bool = True  # Call on spread or put on spread


def price_cms_spread_option(
    cms1_forward: float,
    cms2_forward: float,
    strike: float,
    time_to_expiry: float,
    spread_volatility: float,
    discount_factor: float,
    annuity: float,
    notional: float = 1_000_000.0,
    is_call: bool = True,
) -> float:
    """Price CMS spread option using Black's formula on the spread.

    The option pays max(spread - K, 0) for a call, or max(K - spread, 0) for a put,
    where spread = CMS1 - CMS2.

    Parameters
    ----------
    cms1_forward : float
        Forward CMS rate for longer tenor
    cms2_forward : float
        Forward CMS rate for shorter tenor
    strike : float
        Strike on the spread
    time_to_expiry : float
        Time to expiry
    spread_volatility : float
        Volatility of the spread
    discount_factor : float
        Discount factor
    annuity : float
        Annuity factor
    notional : float
        Notional principal
    is_call : bool
        True for call, False for put

    Returns
    -------
    float
        CMS spread option price
    """
    # Forward spread
    forward_spread = cms1_forward - cms2_forward

    # Black's formula on the spread
    sqrt_T = jnp.sqrt(time_to_expiry)
    d1 = (jnp.log(forward_spread / strike) + 0.5 * spread_volatility**2 * time_to_expiry) / (
        spread_volatility * sqrt_T
    )
    d2 = d1 - spread_volatility * sqrt_T

    if is_call:
        value = forward_spread * norm.cdf(d1) - strike * norm.cdf(d2)
    else:
        value = strike * norm.cdf(-d2) - forward_spread * norm.cdf(-d1)

    price = notional * annuity * discount_factor * value

    return float(price)


def price_cms_spread_option_mc(
    cms1_paths: Array,
    cms2_paths: Array,
    strike: float,
    discount_factor: float,
    annuity: float,
    notional: float = 1_000_000.0,
    is_call: bool = True,
) -> float:
    """Price CMS spread option using Monte Carlo.

    Parameters
    ----------
    cms1_paths : Array
        Simulated CMS1 rate paths
    cms2_paths : Array
        Simulated CMS2 rate paths
    strike : float
        Strike on the spread
    discount_factor : float
        Discount factor
    annuity : float
        Annuity factor
    notional : float
        Notional principal
    is_call : bool
        True for call, False for put

    Returns
    -------
    float
        CMS spread option price
    """
    # Terminal spreads
    spreads = cms1_paths[:, -1] - cms2_paths[:, -1]

    # Payoffs
    if is_call:
        payoffs = jnp.maximum(spreads - strike, 0.0)
    else:
        payoffs = jnp.maximum(strike - spreads, 0.0)

    # Discounted average
    price = notional * annuity * discount_factor * jnp.mean(payoffs)

    return float(price)


# LMM (LIBOR Market Model) utilities
def lmm_simulate_forward_rates(
    initial_forwards: Array,
    volatilities: Array,
    correlations: Array,
    times: Array,
    n_paths: int,
    key: Optional[Array] = None,
) -> Array:
    """Simulate forward LIBOR rates using LIBOR Market Model.

    Parameters
    ----------
    initial_forwards : Array
        Initial forward rates
    volatilities : Array
        Volatilities for each forward rate
    correlations : Array
        Correlation matrix between forward rates
    times : Array
        Time grid
    n_paths : int
        Number of Monte Carlo paths
    key : Optional[Array]
        JAX random key

    Returns
    -------
    Array
        Simulated forward rate paths [n_paths, n_times, n_forwards]
    """
    if key is None:
        import jax.random as jrand
        key = jrand.PRNGKey(0)

    n_forwards = len(initial_forwards)
    n_times = len(times)
    dt = jnp.diff(times, prepend=0.0)

    # Cholesky decomposition of correlation matrix
    L = jnp.linalg.cholesky(correlations)

    # Initialize paths
    paths = jnp.zeros((n_paths, n_times, n_forwards))
    paths = paths.at[:, 0, :].set(initial_forwards)

    # Generate correlated normal variables
    import jax.random as jrand
    for t in range(1, n_times):
        # Independent normals
        z = jrand.normal(key, (n_paths, n_forwards))
        key, _ = jrand.split(key)

        # Correlate them
        dW = jnp.dot(z, L.T) * jnp.sqrt(dt[t])

        # LMM drift adjustment
        for i in range(n_forwards):
            drift = 0.0
            for j in range(i + 1, n_forwards):
                drift += (
                    volatilities[i] * volatilities[j] * correlations[i, j]
                    * paths[:, t - 1, j] / (1 + dt[t] * paths[:, t - 1, j])
                )

            # Update forward rate
            paths = paths.at[:, t, i].set(
                paths[:, t - 1, i]
                * jnp.exp(
                    (drift - 0.5 * volatilities[i]**2) * dt[t]
                    + volatilities[i] * dW[:, i]
                )
            )

    return paths


# CMS Caplets and Floorlets
@jit
def cms_caplet_price(
    cms_forward: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    discount_factor: float,
    annuity: float,
    notional: float = 1_000_000.0,
    convexity_adjustment: float = 0.0,
) -> float:
    """Price a CMS caplet using Black's formula with convexity adjustment.

    Parameters
    ----------
    cms_forward : float
        Forward CMS rate
    strike : float
        Strike rate
    time_to_expiry : float
        Time to caplet expiry
    volatility : float
        CMS rate volatility
    discount_factor : float
        Discount factor to payment date
    annuity : float
        Annuity factor
    notional : float
        Notional principal
    convexity_adjustment : float
        Convexity adjustment for CMS rate (optional)

    Returns
    -------
    float
        CMS caplet price

    Notes
    -----
    CMS rates require convexity adjustments due to the nonlinear relationship
    between the swap rate and the discount factors.

    Adjusted forward = Forward * (1 + convexity_adjustment)
    """
    # Apply convexity adjustment
    adjusted_forward = cms_forward * (1.0 + convexity_adjustment)

    # Black's formula
    sqrt_T = jnp.sqrt(time_to_expiry)
    d1 = (jnp.log(adjusted_forward / strike) + 0.5 * volatility**2 * time_to_expiry) / (
        volatility * sqrt_T
    )
    d2 = d1 - volatility * sqrt_T

    caplet_value = (
        notional
        * annuity
        * discount_factor
        * (adjusted_forward * norm.cdf(d1) - strike * norm.cdf(d2))
    )

    return caplet_value


@jit
def cms_floorlet_price(
    cms_forward: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    discount_factor: float,
    annuity: float,
    notional: float = 1_000_000.0,
    convexity_adjustment: float = 0.0,
) -> float:
    """Price a CMS floorlet using Black's formula with convexity adjustment.

    Parameters
    ----------
    cms_forward : float
        Forward CMS rate
    strike : float
        Strike rate
    time_to_expiry : float
        Time to floorlet expiry
    volatility : float
        CMS rate volatility
    discount_factor : float
        Discount factor to payment date
    annuity : float
        Annuity factor
    notional : float
        Notional principal
    convexity_adjustment : float
        Convexity adjustment for CMS rate

    Returns
    -------
    float
        CMS floorlet price
    """
    # Apply convexity adjustment
    adjusted_forward = cms_forward * (1.0 + convexity_adjustment)

    # Black's formula
    sqrt_T = jnp.sqrt(time_to_expiry)
    d1 = (jnp.log(adjusted_forward / strike) + 0.5 * volatility**2 * time_to_expiry) / (
        volatility * sqrt_T
    )
    d2 = d1 - volatility * sqrt_T

    floorlet_value = (
        notional
        * annuity
        * discount_factor
        * (strike * norm.cdf(-d2) - adjusted_forward * norm.cdf(-d1))
    )

    return floorlet_value


def cms_convexity_adjustment(
    forward_rate: float,
    volatility: float,
    time_to_payment: float,
    swap_tenor: float,
    payment_frequency: int = 2,
) -> float:
    """Calculate convexity adjustment for CMS rate.

    Parameters
    ----------
    forward_rate : float
        Forward swap rate
    volatility : float
        Swap rate volatility
    time_to_payment : float
        Time to payment date
    swap_tenor : float
        Tenor of the CMS swap
    payment_frequency : int
        Payment frequency per year

    Returns
    -------
    float
        Convexity adjustment (additive)

    Notes
    -----
    Approximate convexity adjustment formula:
        CA ≈ 0.5 * σ² * T * D * S

    where D is the duration of the swap annuity and S is the forward rate.

    More accurate formulas would use replication methods or
    numerical integration.
    """
    # Approximate annuity duration
    n_periods = swap_tenor * payment_frequency
    dt = 1.0 / payment_frequency

    # Simplified duration calculation
    discount_sum = 0.0
    time_weighted_sum = 0.0

    for i in range(1, int(n_periods) + 1):
        t = i * dt
        df = jnp.exp(-forward_rate * t)
        discount_sum += df
        time_weighted_sum += t * df

    duration = time_weighted_sum / discount_sum if discount_sum > 0 else swap_tenor / 2

    # Convexity adjustment
    adjustment = 0.5 * volatility**2 * time_to_payment * duration * forward_rate

    return float(adjustment)


def price_cms_cap(
    cms_forwards: Array,
    strike: float,
    times_to_expiry: Array,
    volatilities: Array,
    discount_factors: Array,
    annuities: Array,
    notional: float = 1_000_000.0,
    apply_convexity_adjustment: bool = True,
    swap_tenor: float = 10.0,
) -> float:
    """Price a CMS cap as a portfolio of CMS caplets.

    Parameters
    ----------
    cms_forwards : Array
        Forward CMS rates for each period
    strike : float
        Cap strike rate
    times_to_expiry : Array
        Times to expiry for each caplet
    volatilities : Array
        Volatilities for each caplet
    discount_factors : Array
        Discount factors for each payment
    annuities : Array
        Annuity factors for each period
    notional : float
        Notional principal
    apply_convexity_adjustment : bool
        Whether to apply convexity adjustment
    swap_tenor : float
        Tenor of CMS swap

    Returns
    -------
    float
        CMS cap price
    """
    caplet_prices = []

    for i in range(len(cms_forwards)):
        if apply_convexity_adjustment:
            conv_adj = cms_convexity_adjustment(
                cms_forwards[i], volatilities[i], times_to_expiry[i], swap_tenor
            )
        else:
            conv_adj = 0.0

        caplet = cms_caplet_price(
            cms_forwards[i],
            strike,
            times_to_expiry[i],
            volatilities[i],
            discount_factors[i],
            annuities[i],
            notional,
            conv_adj,
        )
        caplet_prices.append(caplet)

    return float(jnp.sum(jnp.array(caplet_prices)))


def price_cms_floor(
    cms_forwards: Array,
    strike: float,
    times_to_expiry: Array,
    volatilities: Array,
    discount_factors: Array,
    annuities: Array,
    notional: float = 1_000_000.0,
    apply_convexity_adjustment: bool = True,
    swap_tenor: float = 10.0,
) -> float:
    """Price a CMS floor as a portfolio of CMS floorlets.

    Parameters
    ----------
    cms_forwards : Array
        Forward CMS rates for each period
    strike : float
        Floor strike rate
    times_to_expiry : Array
        Times to expiry for each floorlet
    volatilities : Array
        Volatilities for each floorlet
    discount_factors : Array
        Discount factors for each payment
    annuities : Array
        Annuity factors for each period
    notional : float
        Notional principal
    apply_convexity_adjustment : bool
        Whether to apply convexity adjustment
    swap_tenor : float
        Tenor of CMS swap

    Returns
    -------
    float
        CMS floor price
    """
    floorlet_prices = []

    for i in range(len(cms_forwards)):
        if apply_convexity_adjustment:
            conv_adj = cms_convexity_adjustment(
                cms_forwards[i], volatilities[i], times_to_expiry[i], swap_tenor
            )
        else:
            conv_adj = 0.0

        floorlet = cms_floorlet_price(
            cms_forwards[i],
            strike,
            times_to_expiry[i],
            volatilities[i],
            discount_factors[i],
            annuities[i],
            notional,
            conv_adj,
        )
        floorlet_prices.append(floorlet)

    return float(jnp.sum(jnp.array(floorlet_prices)))


__all__ = [
    # Cap/Floor
    "CapFloorSpecs",
    "black_caplet_price",
    "black_floorlet_price",
    "price_cap",
    "price_floor",
    "sabr_implied_vol_caplet",
    # SOFR Cap/Floor
    "sofr_caplet_price",
    "sofr_floorlet_price",
    "price_sofr_cap",
    "price_sofr_floor",
    # Digital caplets/floorlets
    "digital_caplet_price",
    "digital_floorlet_price",
    # Range accrual
    "RangeAccrualSpecs",
    "price_range_accrual_mc",
    # CMS spread options
    "CMSSpreadOptionSpecs",
    "price_cms_spread_option",
    "price_cms_spread_option_mc",
    # CMS caplets/floorlets
    "cms_caplet_price",
    "cms_floorlet_price",
    "cms_convexity_adjustment",
    "price_cms_cap",
    "price_cms_floor",
    # LMM utilities
    "lmm_simulate_forward_rates",
]
