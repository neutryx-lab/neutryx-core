"""Exotic equity derivatives: Cliquet, Quanto, Forward start, Enhanced Asian options.

This module implements:
- Cliquet options (ratchet options)
- Quanto options (cross-currency derivatives)
- Forward start options
- Enhanced Asian options (geometric and arithmetic with various features)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import jax.numpy as jnp
from jax import Array
from jax.scipy.stats import norm

from .base import PathProduct, Product


@dataclass
class CliquetOption(PathProduct):
    """Cliquet option (ratchet option).

    A series of forward-starting ATM options where strikes reset periodically.
    The payoff is the sum of capped/floored returns over reset periods.

    Attributes
    ----------
    T : float
        Total maturity
    reset_dates : Array
        Dates when strike resets (as fractions of T)
    global_floor : float
        Minimum total return
    global_cap : float
        Maximum total return
    local_floor : float
        Minimum return per period
    local_cap : float
        Maximum return per period
    """

    T: float
    reset_dates: Array
    global_floor: float = 0.0
    global_cap: float = float('inf')
    local_floor: float = 0.0
    local_cap: float = float('inf')

    def __post_init__(self):
        self.reset_dates = jnp.asarray(self.reset_dates)

    def payoff_path(self, path: Array) -> Array:
        """Compute cliquet option payoff.

        At each reset date, lock in the return capped/floored at local levels.
        Apply global cap/floor to the sum of all period returns.
        """
        path = jnp.asarray(path)
        n_steps = len(path)

        # Convert reset dates to indices
        reset_indices = jnp.round(self.reset_dates * (n_steps - 1)).astype(int)
        reset_indices = jnp.clip(reset_indices, 0, n_steps - 1)

        # Include start and end
        reset_indices = jnp.concatenate([jnp.array([0]), reset_indices, jnp.array([n_steps - 1])])
        reset_indices = jnp.unique(reset_indices)

        total_return = 0.0

        for i in range(len(reset_indices) - 1):
            idx_start = reset_indices[i]
            idx_end = reset_indices[i + 1]

            S_start = path[idx_start]
            S_end = path[idx_end]

            # Period return
            period_return = (S_end / S_start) - 1.0

            # Apply local cap/floor
            capped_return = jnp.minimum(period_return, self.local_cap)
            floored_return = jnp.maximum(capped_return, self.local_floor)

            total_return += floored_return

        # Apply global cap/floor
        total_return = jnp.minimum(total_return, self.global_cap)
        total_return = jnp.maximum(total_return, self.global_floor)

        return total_return


def price_cliquet_analytical(
    S0: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n_resets: int,
    local_floor: float = 0.0,
    local_cap: float = float('inf'),
    global_floor: float = 0.0,
    global_cap: float = float('inf'),
) -> float:
    """Price cliquet option using analytical approximation.

    Parameters
    ----------
    S0 : float
        Initial spot price
    T : float
        Total maturity
    r : float
        Risk-free rate
    q : float
        Dividend yield
    sigma : float
        Volatility
    n_resets : int
        Number of reset periods
    local_floor : float
        Local floor per period
    local_cap : float
        Local cap per period
    global_floor : float
        Global floor on total return
    global_cap : float
        Global cap on total return

    Returns
    -------
    float
        Cliquet option price
    """
    # Simplified pricing using Black-Scholes for each period
    # Each period is a forward-start ATM option
    dt = T / n_resets
    discount = jnp.exp(-r * T)

    # Price of one period (ATM forward start)
    # Approximate as ATM straddle
    from neutryx.models.bs import price as bs_price

    period_value = 0.0
    for i in range(n_resets):
        t_start = i * dt
        t_end = (i + 1) * dt
        time_to_expiry = t_end - t_start

        # ATM forward start option value
        # Simplified: use Black-Scholes with forward as strike
        forward = S0 * jnp.exp((r - q) * t_start)

        # Call value for one period
        call_value = bs_price(forward, forward, time_to_expiry, r, q, sigma, "call")

        # Apply local caps/floors (approximate adjustment)
        if local_cap < float('inf'):
            call_value *= 0.8  # Rough approximation for cap
        if local_floor > 0:
            call_value += local_floor * jnp.exp(-r * t_end)

        period_value += call_value

    # Apply global cap/floor (rough approximation)
    total_value = period_value * discount

    return float(total_value)


@dataclass
class QuantoOption(Product):
    """Quanto option (cross-currency option).

    An option on a foreign asset but payoff in domestic currency at a fixed FX rate.

    Attributes
    ----------
    K : float
        Strike price (in foreign currency terms)
    T : float
        Time to maturity
    is_call : bool
        True for call, False for put
    quanto_factor : float
        Fixed FX conversion rate
    """

    K: float
    T: float
    is_call: bool = True
    quanto_factor: float = 1.0

    def payoff_terminal(self, spot: Array) -> Array:
        """Terminal payoff in domestic currency."""
        if self.is_call:
            intrinsic = jnp.maximum(spot - self.K, 0.0)
        else:
            intrinsic = jnp.maximum(self.K - spot, 0.0)

        return intrinsic * self.quanto_factor


def price_quanto_option(
    S0: float,
    K: float,
    T: float,
    r_d: float,
    r_f: float,
    q: float,
    sigma_S: float,
    sigma_FX: float,
    rho: float,
    quanto_factor: float,
    is_call: bool = True,
) -> float:
    """Price quanto option using Black-Scholes with quanto adjustment.

    Parameters
    ----------
    S0 : float
        Initial spot price (in foreign currency)
    K : float
        Strike price
    T : float
        Time to maturity
    r_d : float
        Domestic risk-free rate
    r_f : float
        Foreign risk-free rate
    q : float
        Dividend yield on foreign asset
    sigma_S : float
        Volatility of foreign asset
    sigma_FX : float
        Volatility of FX rate
    rho : float
        Correlation between asset and FX rate
    quanto_factor : float
        Fixed FX conversion rate
    is_call : bool
        True for call, False for put

    Returns
    -------
    float
        Quanto option price in domestic currency
    """
    # Quanto adjustment to drift
    # Adjusted drift: r_d - q - rho * sigma_S * sigma_FX
    drift_adjustment = rho * sigma_S * sigma_FX

    # Adjusted forward
    F = S0 * jnp.exp((r_d - q - drift_adjustment) * T)

    # Black formula
    sqrt_T = jnp.sqrt(T)
    d1 = (jnp.log(F / K) + 0.5 * sigma_S**2 * T) / (sigma_S * sqrt_T)
    d2 = d1 - sigma_S * sqrt_T

    discount = jnp.exp(-r_d * T)

    if is_call:
        price = discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        price = discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

    return float(price * quanto_factor)


@dataclass
class ForwardStartOption(Product):
    """Forward start option.

    An option that starts at a future date with strike set at-the-money at that time.

    Attributes
    ----------
    T_start : float
        Time when option starts
    T_end : float
        Time when option expires
    strike_percentage : float
        Strike as percentage of spot at T_start (1.0 = ATM, 1.1 = 110% ATM)
    is_call : bool
        True for call, False for put
    """

    T_start: float
    T_end: float
    strike_percentage: float = 1.0
    is_call: bool = True

    @property
    def T(self) -> float:
        """Total time to expiry."""
        return self.T_end

    def payoff_terminal(self, spot: Array) -> Array:
        """This is path-dependent, so not directly applicable."""
        raise NotImplementedError("Forward start option requires full path")


def price_forward_start_option(
    S0: float,
    T_start: float,
    T_end: float,
    r: float,
    q: float,
    sigma: float,
    strike_percentage: float = 1.0,
    is_call: bool = True,
) -> float:
    """Price forward start option using analytical formula.

    Parameters
    ----------
    S0 : float
        Current spot price
    T_start : float
        Time when option starts
    T_end : float
        Time when option expires
    r : float
        Risk-free rate
    q : float
        Dividend yield
    sigma : float
        Volatility
    strike_percentage : float
        Strike as percentage of spot at T_start
    is_call : bool
        True for call, False for put

    Returns
    -------
    float
        Forward start option price

    Notes
    -----
    For a forward start option, the value at time 0 is:
        V_0 = S_0 * exp(-q * T_start) * BS(1, strike_pct, T_end - T_start, ...)

    This uses the homogeneity property of Black-Scholes.
    """
    # Time from start to expiry
    T_option = T_end - T_start

    # Black-Scholes for unit spot with modified strike
    from neutryx.models.bs import price as bs_price

    # Value at T_start for unit spot
    unit_value = bs_price(
        S=1.0,
        K=strike_percentage,
        T=T_option,
        r=r,
        q=q,
        sigma=sigma,
        kind="call" if is_call else "put"
    )

    # Scale by forward to T_start
    forward_to_start = S0 * jnp.exp(-q * T_start)

    # Discount back to today
    discount = jnp.exp(-r * T_start)

    price = forward_to_start * unit_value

    return float(price)


# Enhanced Asian options
@dataclass
class AsianGeometric(PathProduct):
    """Geometric average Asian option.

    Attributes
    ----------
    K : float
        Strike price
    T : float
        Maturity
    is_call : bool
        True for call, False for put
    averaging_start : float
        Time when averaging starts (fraction of T)
    """

    K: float
    T: float
    is_call: bool = True
    averaging_start: float = 0.0

    def payoff_path(self, path: Array) -> Array:
        """Compute payoff using geometric average."""
        path = jnp.asarray(path)

        # Determine which part of path to use for averaging
        n_steps = len(path)
        start_idx = int(self.averaging_start * (n_steps - 1))

        averaging_path = path[start_idx:]

        # Geometric mean
        log_mean = jnp.mean(jnp.log(averaging_path))
        geom_avg = jnp.exp(log_mean)

        # Payoff
        if self.is_call:
            return jnp.maximum(geom_avg - self.K, 0.0)
        else:
            return jnp.maximum(self.K - geom_avg, 0.0)


def price_geometric_asian_analytical(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool = True,
) -> float:
    """Price geometric Asian option using analytical formula.

    Parameters
    ----------
    S0 : float
        Initial spot
    K : float
        Strike
    T : float
        Maturity
    r : float
        Risk-free rate
    q : float
        Dividend yield
    sigma : float
        Volatility
    is_call : bool
        Call or put

    Returns
    -------
    float
        Geometric Asian option price
    """
    # Adjusted parameters for geometric Asian
    sigma_adj = sigma / jnp.sqrt(3.0)

    # Adjusted drift
    # For geometric average: mu_G = mu - sigma^2/6
    drift_adj = 0.5 * (r - q - sigma**2 / 6.0)

    # Black-Scholes with adjusted parameters
    from neutryx.models.bs import price as bs_price

    price = bs_price(
        S=S0,
        K=K,
        T=T,
        r=r - drift_adj,
        q=q,
        sigma=sigma_adj,
        kind="call" if is_call else "put"
    )

    return float(price)


@dataclass
class AsianArithmeticEnhanced(PathProduct):
    """Enhanced arithmetic average Asian option with various features.

    Attributes
    ----------
    K : float
        Strike price
    T : float
        Maturity
    is_call : bool
        True for call, False for put
    averaging_start : float
        When averaging begins (as fraction of T)
    discrete_observations : Optional[Array]
        Specific observation times (if None, uses continuous)
    weights : Optional[Array]
        Weights for each observation (if None, equal weights)
    """

    K: float
    T: float
    is_call: bool = True
    averaging_start: float = 0.0
    discrete_observations: Optional[Array] = None
    weights: Optional[Array] = None

    def __post_init__(self):
        if self.discrete_observations is not None:
            self.discrete_observations = jnp.asarray(self.discrete_observations)
        if self.weights is not None:
            self.weights = jnp.asarray(self.weights)
            # Normalize weights
            self.weights = self.weights / jnp.sum(self.weights)

    def payoff_path(self, path: Array) -> Array:
        """Compute payoff using arithmetic average with enhancements."""
        path = jnp.asarray(path)
        n_steps = len(path)

        if self.discrete_observations is not None:
            # Use discrete observations
            obs_indices = jnp.round(self.discrete_observations * (n_steps - 1)).astype(int)
            obs_indices = jnp.clip(obs_indices, 0, n_steps - 1)
            observed_prices = path[obs_indices]

            if self.weights is not None:
                # Weighted average
                avg = jnp.sum(observed_prices * self.weights)
            else:
                # Simple average
                avg = jnp.mean(observed_prices)
        else:
            # Continuous averaging
            start_idx = int(self.averaging_start * (n_steps - 1))
            averaging_path = path[start_idx:]
            avg = jnp.mean(averaging_path)

        # Payoff
        if self.is_call:
            return jnp.maximum(avg - self.K, 0.0)
        else:
            return jnp.maximum(self.K - avg, 0.0)


__all__ = [
    # Cliquet
    "CliquetOption",
    "price_cliquet_analytical",
    # Quanto
    "QuantoOption",
    "price_quanto_option",
    # Forward start
    "ForwardStartOption",
    "price_forward_start_option",
    # Enhanced Asian
    "AsianGeometric",
    "price_geometric_asian_analytical",
    "AsianArithmeticEnhanced",
]
