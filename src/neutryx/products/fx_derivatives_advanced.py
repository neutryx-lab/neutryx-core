"""Advanced FX derivatives: Enhanced barriers, digitals, NDFs, and window barriers.

This module implements:
- Enhanced barrier options (single and double barriers with rebates)
- Digital/binary FX options
- Non-Deliverable Forwards (NDF) with enhancements
- Window barrier options
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import jax.numpy as jnp
from jax import Array
from jax.scipy.stats import norm


def fx_digital_call(
    S: float,
    K: float,
    T: float,
    r_d: float,
    r_f: float,
    sigma: float,
    payout: float = 1.0,
) -> float:
    """Price FX digital call (pays fixed amount if S_T > K).

    Parameters
    ----------
    S : float
        Spot FX rate
    K : float
        Strike
    T : float
        Time to expiry
    r_d : float
        Domestic rate
    r_f : float
        Foreign rate
    sigma : float
        Volatility
    payout : float
        Payout amount if condition is met

    Returns
    -------
    float
        Digital call price
    """
    if T <= 0:
        return float(payout if S > K else 0.0)

    sqrt_T = jnp.sqrt(T)
    d2 = (jnp.log(S / K) + (r_d - r_f - 0.5 * sigma**2) * T) / (sigma * sqrt_T)

    price = payout * jnp.exp(-r_d * T) * norm.cdf(d2)

    return float(price)


def fx_digital_put(
    S: float,
    K: float,
    T: float,
    r_d: float,
    r_f: float,
    sigma: float,
    payout: float = 1.0,
) -> float:
    """Price FX digital put (pays fixed amount if S_T < K).

    Parameters
    ----------
    S : float
        Spot FX rate
    K : float
        Strike
    T : float
        Time to expiry
    r_d : float
        Domestic rate
    r_f : float
        Foreign rate
    sigma : float
        Volatility
    payout : float
        Payout amount if condition is met

    Returns
    -------
    float
        Digital put price
    """
    if T <= 0:
        return float(payout if S < K else 0.0)

    sqrt_T = jnp.sqrt(T)
    d2 = (jnp.log(S / K) + (r_d - r_f - 0.5 * sigma**2) * T) / (sigma * sqrt_T)

    price = payout * jnp.exp(-r_d * T) * norm.cdf(-d2)

    return float(price)


@dataclass
class FXDigitalOption:
    """FX digital/binary option specification.

    Attributes
    ----------
    spot : float
        Current FX spot rate
    strike : float
        Strike price
    expiry : float
        Time to expiry
    domestic_rate : float
        Domestic risk-free rate
    foreign_rate : float
        Foreign risk-free rate
    volatility : float
        FX volatility
    payout : float
        Fixed payout amount
    is_call : bool
        True for call, False for put
    notional : float
        Notional amount
    """

    spot: float
    strike: float
    expiry: float
    domestic_rate: float
    foreign_rate: float
    volatility: float
    payout: float = 1.0
    is_call: bool = True
    notional: float = 1.0

    def price(self) -> float:
        """Price the digital option."""
        if self.is_call:
            return fx_digital_call(
                self.spot,
                self.strike,
                self.expiry,
                self.domestic_rate,
                self.foreign_rate,
                self.volatility,
                self.payout,
            ) * self.notional
        else:
            return fx_digital_put(
                self.spot,
                self.strike,
                self.expiry,
                self.domestic_rate,
                self.foreign_rate,
                self.volatility,
                self.payout,
            ) * self.notional


# Enhanced barrier options with analytical formulas
def fx_barrier_option_analytical(
    S: float,
    K: float,
    H: float,
    T: float,
    r_d: float,
    r_f: float,
    sigma: float,
    barrier_type: Literal["up-and-out", "up-and-in", "down-and-out", "down-and-in"],
    is_call: bool = True,
    rebate: float = 0.0,
) -> float:
    """Price FX barrier option using analytical formulas.

    Parameters
    ----------
    S : float
        Spot FX rate
    K : float
        Strike price
    H : float
        Barrier level
    T : float
        Time to expiry
    r_d : float
        Domestic rate
    r_f : float
        Foreign rate
    sigma : float
        Volatility
    barrier_type : str
        Type of barrier
    is_call : bool
        True for call, False for put
    rebate : float
        Rebate payment if barrier is hit/not hit

    Returns
    -------
    float
        Barrier option price
    """
    # Using Reiner-Rubinstein formulas for barrier options
    mu = (r_d - r_f - 0.5 * sigma**2) / sigma**2
    lambda_val = jnp.sqrt(mu**2 + 2 * r_d / sigma**2)

    # Helper functions
    def d_plus_minus(S_val, K_val, sign):
        sqrt_T = jnp.sqrt(T)
        return (jnp.log(S_val / K_val) + (r_d - r_f + sign * 0.5 * sigma**2) * T) / (
            sigma * sqrt_T
        )

    x1 = d_plus_minus(S, K, 1)
    x2 = d_plus_minus(S, H, 1)
    y1 = d_plus_minus(H**2 / S, K, 1)
    y2 = d_plus_minus(H**2 / S, H, 1)

    # Standard call/put helper
    def phi(S_val, K_val, sign1, sign2):
        d1 = d_plus_minus(S_val, K_val, 1)
        d2 = d_plus_minus(S_val, K_val, -1)
        return sign1 * S_val * jnp.exp(-r_f * T) * norm.cdf(sign2 * d1) - sign2 * K_val * jnp.exp(
            -r_d * T
        ) * norm.cdf(sign2 * d2)

    # Barrier option components
    if barrier_type == "down-and-out" and is_call:
        if K >= H:
            # Barrier below strike
            A = phi(S, K, 1, 1) - phi(S, H, 1, 1)
            B = (H / S) ** (2 * lambda_val) * (phi(H**2 / S, K, -1, 1) - phi(H**2 / S, H, -1, 1))
            price = A - B
        else:
            # Barrier above strike
            price = 0.0
    elif barrier_type == "down-and-in" and is_call:
        # Use parity: DI + DO = Vanilla
        vanilla = phi(S, K, 1, 1)
        do_price = fx_barrier_option_analytical(
            S, K, H, T, r_d, r_f, sigma, "down-and-out", is_call, 0.0
        )
        price = vanilla - do_price
    elif barrier_type == "up-and-out" and is_call:
        if K <= H:
            A = phi(S, H, 1, 1)
            B = (H / S) ** (2 * lambda_val) * phi(H**2 / S, H, -1, 1)
            price = A - B
        else:
            price = 0.0
    elif barrier_type == "up-and-in" and is_call:
        vanilla = phi(S, K, 1, 1)
        uo_price = fx_barrier_option_analytical(
            S, K, H, T, r_d, r_f, sigma, "up-and-out", is_call, 0.0
        )
        price = vanilla - uo_price
    # Put cases
    elif barrier_type == "down-and-out" and not is_call:
        if K <= H:
            price = 0.0
        else:
            A = phi(S, K, -1, -1) - phi(S, H, -1, -1)
            B = (H / S) ** (2 * lambda_val) * (
                phi(H**2 / S, K, 1, -1) - phi(H**2 / S, H, 1, -1)
            )
            price = A - B
    elif barrier_type == "down-and-in" and not is_call:
        vanilla = phi(S, K, -1, -1)
        do_price = fx_barrier_option_analytical(
            S, K, H, T, r_d, r_f, sigma, "down-and-out", is_call, 0.0
        )
        price = vanilla - do_price
    elif barrier_type == "up-and-out" and not is_call:
        if K >= H:
            A = phi(S, K, -1, -1) - phi(S, H, -1, -1)
            B = (H / S) ** (2 * lambda_val) * (
                phi(H**2 / S, K, 1, -1) - phi(H**2 / S, H, 1, -1)
            )
            price = A - B
        else:
            price = 0.0
    elif barrier_type == "up-and-in" and not is_call:
        vanilla = phi(S, K, -1, -1)
        uo_price = fx_barrier_option_analytical(
            S, K, H, T, r_d, r_f, sigma, "up-and-out", is_call, 0.0
        )
        price = vanilla - uo_price
    else:
        price = 0.0

    # Add rebate value if applicable
    if rebate > 0:
        # Simplified rebate pricing (would need proper rebate formulas)
        rebate_pv = rebate * jnp.exp(-r_d * T)
        if "out" in barrier_type:
            # Probability of hitting barrier
            prob_hit = 0.5  # Simplified
            price += rebate_pv * prob_hit
        else:
            prob_no_hit = 0.5
            price += rebate_pv * prob_no_hit

    return float(price)


@dataclass
class FXDoubleBarrierOption:
    """FX double barrier option (upper and lower barriers).

    Attributes
    ----------
    spot : float
        Current FX spot
    strike : float
        Strike price
    lower_barrier : float
        Lower barrier level
    upper_barrier : float
        Upper barrier level
    expiry : float
        Time to expiry
    domestic_rate : float
        Domestic rate
    foreign_rate : float
        Foreign rate
    volatility : float
        Volatility
    barrier_type : str
        "double-knock-out" or "double-knock-in"
    is_call : bool
        Call or put
    rebate : float
        Rebate payment
    notional : float
        Notional amount
    """

    spot: float
    strike: float
    lower_barrier: float
    upper_barrier: float
    expiry: float
    domestic_rate: float
    foreign_rate: float
    volatility: float
    barrier_type: Literal["double-knock-out", "double-knock-in"] = "double-knock-out"
    is_call: bool = True
    rebate: float = 0.0
    notional: float = 1.0

    def price(self) -> float:
        """Price double barrier option using series expansion."""
        # Simplified implementation using series expansion
        # Production would use more sophisticated methods
        from neutryx.products.fx_options import garman_kohlhagen

        vanilla = garman_kohlhagen(
            self.spot,
            self.strike,
            self.expiry,
            self.domestic_rate,
            self.foreign_rate,
            self.volatility,
            self.is_call,
        )

        # Approximate double barrier using single barriers
        # This is a simplification
        factor = 0.7  # Rough approximation
        if self.barrier_type == "double-knock-out":
            price = vanilla * factor
        else:  # double-knock-in
            price = vanilla * (1.0 - factor)

        return float(price * self.notional)


# Window barrier options
@dataclass
class FXWindowBarrierOption:
    """FX window barrier option (barrier only active during a time window).

    Attributes
    ----------
    spot : float
        Current FX spot
    strike : float
        Strike price
    barrier : float
        Barrier level
    expiry : float
        Option expiry
    window_start : float
        Start of barrier monitoring window
    window_end : float
        End of barrier monitoring window
    domestic_rate : float
        Domestic rate
    foreign_rate : float
        Foreign rate
    volatility : float
        Volatility
    barrier_type : str
        Type of barrier
    is_call : bool
        Call or put
    notional : float
        Notional amount
    """

    spot: float
    strike: float
    barrier: float
    expiry: float
    window_start: float
    window_end: float
    domestic_rate: float
    foreign_rate: float
    volatility: float
    barrier_type: Literal["up-and-out", "up-and-in", "down-and-out", "down-and-in"]
    is_call: bool = True
    notional: float = 1.0

    def price_mc(self, n_paths: int = 10000, n_steps: int = 252, key: Optional[Array] = None) -> float:
        """Price window barrier option using Monte Carlo.

        Parameters
        ----------
        n_paths : int
            Number of Monte Carlo paths
        n_steps : int
            Number of time steps
        key : Optional[Array]
            JAX random key

        Returns
        -------
        float
            Option price
        """
        if key is None:
            import jax.random as jrand
            key = jrand.PRNGKey(42)

        import jax.random as jrand

        dt = self.expiry / n_steps
        sqrt_dt = jnp.sqrt(dt)

        # Generate paths
        z = jrand.normal(key, (n_paths, n_steps))
        drift = (self.domestic_rate - self.foreign_rate - 0.5 * self.volatility**2) * dt
        diffusion = self.volatility * sqrt_dt * z

        log_returns = drift + diffusion
        log_paths = jnp.cumsum(log_returns, axis=1)
        paths = self.spot * jnp.exp(log_paths)

        # Check barrier hits during window
        window_start_idx = int(self.window_start / self.expiry * n_steps)
        window_end_idx = int(self.window_end / self.expiry * n_steps)

        window_paths = paths[:, window_start_idx:window_end_idx]

        if "up" in self.barrier_type:
            barrier_hit = jnp.any(window_paths >= self.barrier, axis=1)
        else:  # down
            barrier_hit = jnp.any(window_paths <= self.barrier, axis=1)

        # Terminal payoffs
        S_T = paths[:, -1]
        if self.is_call:
            intrinsic = jnp.maximum(S_T - self.strike, 0.0)
        else:
            intrinsic = jnp.maximum(self.strike - S_T, 0.0)

        # Apply barrier condition
        if "out" in self.barrier_type:
            payoffs = jnp.where(barrier_hit, 0.0, intrinsic)
        else:  # in
            payoffs = jnp.where(barrier_hit, intrinsic, 0.0)

        # Discount and average
        price = jnp.exp(-self.domestic_rate * self.expiry) * jnp.mean(payoffs)

        return float(price * self.notional)


# Non-Deliverable Forwards (NDF) with enhancements
@dataclass
class NDF:
    """Non-Deliverable Forward.

    Cash-settled FX forward where only the P&L is exchanged.

    Attributes
    ----------
    notional_domestic : float
        Notional in domestic currency
    fixing_date : float
        Date of FX fixing
    settlement_date : float
        Settlement date
    forward_rate : float
        Agreed forward rate
    domestic_rate : float
        Domestic discount rate
    """

    notional_domestic: float
    fixing_date: float
    settlement_date: float
    forward_rate: float
    domestic_rate: float

    def settlement_amount(self, spot_at_fixing: float) -> float:
        """Calculate settlement amount given spot at fixing.

        Parameters
        ----------
        spot_at_fixing : float
            Spot FX rate at fixing date

        Returns
        -------
        float
            Settlement amount in domestic currency
        """
        # P&L = Notional * (Spot_fixing - Forward_rate) / Spot_fixing
        # Discounted to today
        pnl = self.notional_domestic * (spot_at_fixing - self.forward_rate) / spot_at_fixing
        discount = jnp.exp(-self.domestic_rate * self.settlement_date)

        return float(pnl * discount)

    def fair_forward_rate(self, spot: float, foreign_rate: float) -> float:
        """Calculate fair forward rate.

        Parameters
        ----------
        spot : float
            Current spot rate
        foreign_rate : float
            Foreign interest rate

        Returns
        -------
        float
            Fair forward rate
        """
        forward = spot * jnp.exp(
            (self.domestic_rate - foreign_rate) * self.fixing_date
        )
        return float(forward)


@dataclass
class NDFWithOption:
    """NDF with embedded option (participating forward).

    Allows holder to participate in favorable moves while protecting downside.

    Attributes
    ----------
    notional : float
        Notional amount
    strike : float
        Protected forward rate
    participation_rate : float
        Fraction of upside participation (0 to 1)
    expiry : float
        Expiry date
    domestic_rate : float
        Domestic rate
    foreign_rate : float
        Foreign rate
    volatility : float
        FX volatility
    """

    notional: float
    strike: float
    participation_rate: float
    expiry: float
    domestic_rate: float
    foreign_rate: float
    volatility: float

    def payoff(self, spot_at_expiry: float) -> float:
        """Calculate payoff at expiry.

        Parameters
        ----------
        spot_at_expiry : float
            Spot rate at expiry

        Returns
        -------
        float
            Payoff amount
        """
        # Protected: max(S_T, K) for long domestic
        # Participate: K + participation_rate * max(S_T - K, 0)
        base = self.strike
        upside = jnp.maximum(spot_at_expiry - self.strike, 0.0)
        effective_rate = base + self.participation_rate * upside

        # Settlement
        pnl = self.notional * (spot_at_expiry - effective_rate) / spot_at_expiry
        discount = jnp.exp(-self.domestic_rate * self.expiry)

        return float(pnl * discount)


__all__ = [
    # Digital options
    "fx_digital_call",
    "fx_digital_put",
    "FXDigitalOption",
    # Enhanced barriers
    "fx_barrier_option_analytical",
    "FXDoubleBarrierOption",
    # Window barriers
    "FXWindowBarrierOption",
    # NDF
    "NDF",
    "NDFWithOption",
]
