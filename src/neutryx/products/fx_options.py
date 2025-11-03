"""
FX (Foreign Exchange) option products.

This module provides pricing models for FX vanilla and exotic options using
the Garman-Kohlhagen model (Black-Scholes adapted for FX).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from jax import Array
from jax.scipy.stats import norm

from ._utils import compute_d1_d2_fx


@dataclass(frozen=True)
class FXVanillaOption:
    """
    FX vanilla European call or put option.

    Uses Garman-Kohlhagen model for pricing:
    - Foreign currency is treated like stock with dividend yield = r_foreign
    - Domestic currency determines discounting = r_domestic

    Attributes:
        spot: Current FX spot rate (units of domestic per unit of foreign)
        strike: Strike price
        expiry: Time to expiry in years
        domestic_rate: Domestic risk-free rate (continuously compounded)
        foreign_rate: Foreign risk-free rate (continuously compounded)
        volatility: FX volatility (annualized)
        option_type: "call" or "put"
        notional: Notional amount in foreign currency (default 1.0)
        is_call: Alternative to option_type for backwards compatibility

    Example:
        >>> # EUR/USD call option
        >>> option = FXVanillaOption(
        ...     spot=1.10,           # 1 EUR = 1.10 USD
        ...     strike=1.12,
        ...     expiry=1.0,
        ...     domestic_rate=0.05,  # USD rate
        ...     foreign_rate=0.02,   # EUR rate
        ...     volatility=0.10,
        ...     option_type="call"
        ... )
        >>> price = option.price()
    """

    spot: float
    strike: float
    expiry: float
    domestic_rate: float
    foreign_rate: float
    volatility: float
    option_type: Literal["call", "put"] = "call"
    notional: float = 1.0
    is_call: bool = None  # For backwards compatibility

    def __post_init__(self):
        """Validate inputs."""
        if self.spot <= 0:
            raise ValueError("spot must be positive")
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.expiry <= 0:
            raise ValueError("expiry must be positive")
        if self.volatility < 0:
            raise ValueError("volatility must be non-negative")

        # Handle is_call for backwards compatibility
        if self.is_call is not None:
            option_type = "call" if self.is_call else "put"
            object.__setattr__(self, "option_type", option_type)

    def price(self) -> float:
        """
        Price the FX option using Garman-Kohlhagen formula.

        Returns:
            Option price in domestic currency
        """
        return garman_kohlhagen(
            S=self.spot,
            K=self.strike,
            T=self.expiry,
            r_d=self.domestic_rate,
            r_f=self.foreign_rate,
            sigma=self.volatility,
            is_call=(self.option_type == "call")
        ) * self.notional

    def delta(self) -> float:
        """
        Compute option delta (sensitivity to spot).

        Returns:
            Delta (∂V/∂S)
        """
        return fx_delta(
            S=self.spot,
            K=self.strike,
            T=self.expiry,
            r_d=self.domestic_rate,
            r_f=self.foreign_rate,
            sigma=self.volatility,
            is_call=(self.option_type == "call")
        ) * self.notional

    def gamma(self) -> float:
        """
        Compute option gamma (convexity).

        Returns:
            Gamma (∂²V/∂S²)
        """
        return fx_gamma(
            S=self.spot,
            K=self.strike,
            T=self.expiry,
            r_d=self.domestic_rate,
            r_f=self.foreign_rate,
            sigma=self.volatility
        ) * self.notional

    def vega(self) -> float:
        """
        Compute option vega (sensitivity to volatility).

        Returns:
            Vega (∂V/∂σ) per 1% change in vol
        """
        return fx_vega(
            S=self.spot,
            K=self.strike,
            T=self.expiry,
            r_d=self.domestic_rate,
            r_f=self.foreign_rate,
            sigma=self.volatility
        ) * self.notional / 100.0  # Per 1% vol change

    def theta(self) -> float:
        """
        Compute option theta (time decay).

        Returns:
            Theta (∂V/∂t) per day
        """
        return fx_theta(
            S=self.spot,
            K=self.strike,
            T=self.expiry,
            r_d=self.domestic_rate,
            r_f=self.foreign_rate,
            sigma=self.volatility,
            is_call=(self.option_type == "call")
        ) * self.notional / 365.0  # Per day


def garman_kohlhagen(
    S: float,
    K: float,
    T: float,
    r_d: float,
    r_f: float,
    sigma: float,
    is_call: bool = True
) -> float:
    """
    Garman-Kohlhagen FX option pricing formula.

    This is the Black-Scholes formula adapted for FX, where:
    - S: Spot FX rate (domestic per foreign)
    - r_d: Domestic interest rate (replaces r in BS)
    - r_f: Foreign interest rate (replaces q in BS)

    Formula:
        Call: S*e^(-r_f*T)*N(d1) - K*e^(-r_d*T)*N(d2)
        Put:  K*e^(-r_d*T)*N(-d2) - S*e^(-r_f*T)*N(-d1)

    where:
        d1 = [ln(S/K) + (r_d - r_f + σ²/2)*T] / (σ*√T)
        d2 = d1 - σ*√T

    Args:
        S: Spot FX rate
        K: Strike
        T: Time to expiry
        r_d: Domestic rate
        r_f: Foreign rate
        sigma: Volatility
        is_call: True for call, False for put

    Returns:
        Option price

    Example:
        >>> # EUR/USD call
        >>> price = garman_kohlhagen(
        ...     S=1.10, K=1.12, T=1.0,
        ...     r_d=0.05, r_f=0.02, sigma=0.10,
        ...     is_call=True
        ... )
    """
    if T <= 0:
        # At expiry
        intrinsic = jnp.maximum(S - K, 0) if is_call else jnp.maximum(K - S, 0)
        return float(intrinsic)

    if sigma <= 0:
        # Zero vol case
        forward = S * jnp.exp((r_d - r_f) * T)
        intrinsic = jnp.maximum(forward - K, 0) if is_call else jnp.maximum(K - forward, 0)
        return float(jnp.exp(-r_d * T) * intrinsic)

    # Standard case
    d1, d2 = compute_d1_d2_fx(S, K, T, r_d, r_f, sigma)

    if is_call:
        price = S * jnp.exp(-r_f * T) * norm.cdf(d1) - K * jnp.exp(-r_d * T) * norm.cdf(d2)
    else:
        price = K * jnp.exp(-r_d * T) * norm.cdf(-d2) - S * jnp.exp(-r_f * T) * norm.cdf(-d1)

    return float(price)


def fx_delta(
    S: float,
    K: float,
    T: float,
    r_d: float,
    r_f: float,
    sigma: float,
    is_call: bool = True
) -> float:
    """
    Compute FX option delta.

    Delta measures sensitivity to spot FX rate changes.

    Args:
        S, K, T, r_d, r_f, sigma: As in garman_kohlhagen
        is_call: True for call, False for put

    Returns:
        Delta (∂V/∂S)
    """
    if T <= 0 or sigma <= 0:
        # At expiry or zero vol
        if is_call:
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1, _ = compute_d1_d2_fx(S, K, T, r_d, r_f, sigma)

    if is_call:
        delta = jnp.exp(-r_f * T) * norm.cdf(d1)
    else:
        delta = -jnp.exp(-r_f * T) * norm.cdf(-d1)

    return float(delta)


def fx_gamma(
    S: float,
    K: float,
    T: float,
    r_d: float,
    r_f: float,
    sigma: float
) -> float:
    """
    Compute FX option gamma.

    Gamma measures convexity (second derivative w.r.t. spot).

    Args:
        S, K, T, r_d, r_f, sigma: As in garman_kohlhagen

    Returns:
        Gamma (∂²V/∂S²)
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    d1, _ = compute_d1_d2_fx(S, K, T, r_d, r_f, sigma)
    sqrt_T = jnp.sqrt(T)
    gamma = jnp.exp(-r_f * T) * norm.pdf(d1) / (S * sigma * sqrt_T)

    return float(gamma)


def fx_vega(
    S: float,
    K: float,
    T: float,
    r_d: float,
    r_f: float,
    sigma: float
) -> float:
    """
    Compute FX option vega.

    Vega measures sensitivity to volatility changes.

    Args:
        S, K, T, r_d, r_f, sigma: As in garman_kohlhagen

    Returns:
        Vega (∂V/∂σ) for 1 unit change in sigma
    """
    if T <= 0:
        return 0.0

    d1, _ = compute_d1_d2_fx(S, K, T, r_d, r_f, sigma)
    sqrt_T = jnp.sqrt(T)
    vega = S * jnp.exp(-r_f * T) * norm.pdf(d1) * sqrt_T

    return float(vega)


def fx_theta(
    S: float,
    K: float,
    T: float,
    r_d: float,
    r_f: float,
    sigma: float,
    is_call: bool = True
) -> float:
    """
    Compute FX option theta.

    Theta measures time decay (per year).

    Args:
        S, K, T, r_d, r_f, sigma: As in garman_kohlhagen
        is_call: True for call, False for put

    Returns:
        Theta (∂V/∂t) per year
    """
    if T <= 0:
        return 0.0

    d1, d2 = compute_d1_d2_fx(S, K, T, r_d, r_f, sigma)
    sqrt_T = jnp.sqrt(T)
    term1 = -(S * jnp.exp(-r_f * T) * norm.pdf(d1) * sigma) / (2 * sqrt_T)

    if is_call:
        term2 = r_f * S * jnp.exp(-r_f * T) * norm.cdf(d1)
        term3 = -r_d * K * jnp.exp(-r_d * T) * norm.cdf(d2)
    else:
        term2 = -r_f * S * jnp.exp(-r_f * T) * norm.cdf(-d1)
        term3 = r_d * K * jnp.exp(-r_d * T) * norm.cdf(-d2)

    theta = term1 + term2 + term3

    return float(theta)


@dataclass(frozen=True)
class FXBarrierOption:
    """
    FX barrier option (knock-in/knock-out).

    Simplified barrier option with continuous monitoring.

    Attributes:
        spot: Current FX spot rate
        strike: Strike price
        barrier: Barrier level
        expiry: Time to expiry
        domestic_rate: Domestic risk-free rate
        foreign_rate: Foreign risk-free rate
        volatility: FX volatility
        barrier_type: "up-and-out", "up-and-in", "down-and-out", "down-and-in"
        option_type: "call" or "put"
        notional: Notional amount (default 1.0)

    Note:
        This is a simplified implementation. Production code would include:
        - Rebate payments
        - Discrete monitoring
        - Window barriers
    """

    spot: float
    strike: float
    barrier: float
    expiry: float
    domestic_rate: float
    foreign_rate: float
    volatility: float
    barrier_type: Literal["up-and-out", "up-and-in", "down-and-out", "down-and-in"]
    option_type: Literal["call", "put"] = "call"
    notional: float = 1.0

    def price(self) -> float:
        """
        Price FX barrier option.

        Uses analytical formula for European barriers with continuous monitoring.

        Returns:
            Option price in domestic currency
        """
        # Validate barrier placement
        if self.barrier_type.startswith("up") and self.barrier <= self.spot:
            raise ValueError("Up barrier must be above spot")
        if self.barrier_type.startswith("down") and self.barrier >= self.spot:
            raise ValueError("Down barrier must be below spot")

        # For simplicity, using basic barrier formula
        # Full implementation would handle all barrier types properly

        vanilla_price = garman_kohlhagen(
            S=self.spot,
            K=self.strike,
            T=self.expiry,
            r_d=self.domestic_rate,
            r_f=self.foreign_rate,
            sigma=self.volatility,
            is_call=(self.option_type == "call")
        )

        # Approximate barrier adjustment (simplified)
        # Production code would use proper barrier formulas
        if "out" in self.barrier_type:
            # Knock-out: worth less than vanilla
            # Simple approximation: reduce value if near barrier
            distance_to_barrier = abs(jnp.log(self.spot / self.barrier))
            barrier_factor = jnp.minimum(1.0, distance_to_barrier / (self.volatility * jnp.sqrt(self.expiry)))
            return float(vanilla_price * barrier_factor * self.notional)
        else:
            # Knock-in: complement of knock-out
            # IN + OUT = Vanilla
            out_type = self.barrier_type.replace("-in", "-out")
            out_option = FXBarrierOption(
                spot=self.spot,
                strike=self.strike,
                barrier=self.barrier,
                expiry=self.expiry,
                domestic_rate=self.domestic_rate,
                foreign_rate=self.foreign_rate,
                volatility=self.volatility,
                barrier_type=out_type,  # type: ignore
                option_type=self.option_type,
                notional=self.notional
            )
            out_price = out_option.price()
            return vanilla_price * self.notional - out_price
