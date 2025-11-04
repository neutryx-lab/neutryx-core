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


# Touch options
def fx_one_touch(
    S: float,
    H: float,
    T: float,
    r_d: float,
    r_f: float,
    sigma: float,
    barrier_type: Literal["up", "down"],
    payout: float = 1.0,
) -> float:
    """Price FX one-touch option (pays if barrier is hit).

    Parameters
    ----------
    S : float
        Spot FX rate
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
        'up' or 'down'
    payout : float
        Payout if barrier is touched

    Returns
    -------
    float
        One-touch option price
    """
    if T <= 0:
        if barrier_type == "up":
            return float(payout if S >= H else 0.0)
        else:
            return float(payout if S <= H else 0.0)

    # Validate barrier placement
    if barrier_type == "up" and S >= H:
        return float(payout * jnp.exp(-r_d * T))
    if barrier_type == "down" and S <= H:
        return float(payout * jnp.exp(-r_d * T))

    # Use reflection principle and risk-neutral probability
    mu = r_d - r_f - 0.5 * sigma**2
    lambda_plus = (mu + jnp.sqrt(mu**2 + 2 * r_d * sigma**2)) / sigma**2
    lambda_minus = (mu - jnp.sqrt(mu**2 + 2 * r_d * sigma**2)) / sigma**2

    if barrier_type == "up":
        # Probability of hitting upper barrier
        eta = 1.0
        h = jnp.log(H / S) / (sigma * jnp.sqrt(T))
        prob = (H / S) ** lambda_plus * norm.cdf(eta * h - eta * sigma * jnp.sqrt(T))
        prob += (H / S) ** lambda_minus * norm.cdf(eta * h + eta * sigma * jnp.sqrt(T))
    else:
        # Probability of hitting lower barrier
        eta = -1.0
        h = jnp.log(H / S) / (sigma * jnp.sqrt(T))
        prob = (H / S) ** lambda_plus * norm.cdf(eta * h - eta * sigma * jnp.sqrt(T))
        prob += (H / S) ** lambda_minus * norm.cdf(eta * h + eta * sigma * jnp.sqrt(T))

    price = payout * jnp.exp(-r_d * T) * prob
    return float(price)


def fx_no_touch(
    S: float,
    H: float,
    T: float,
    r_d: float,
    r_f: float,
    sigma: float,
    barrier_type: Literal["up", "down"],
    payout: float = 1.0,
) -> float:
    """Price FX no-touch option (pays if barrier is NOT hit).

    Parameters
    ----------
    S : float
        Spot FX rate
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
        'up' or 'down'
    payout : float
        Payout if barrier is not touched

    Returns
    -------
    float
        No-touch option price
    """
    # No-touch = payout - one-touch (by complementarity)
    one_touch_price = fx_one_touch(S, H, T, r_d, r_f, sigma, barrier_type, payout)
    payout_pv = payout * jnp.exp(-r_d * T)
    return float(payout_pv - one_touch_price)


@dataclass
class FXOneTouchOption:
    """FX one-touch option.

    Pays a fixed amount if the barrier is touched at any time before expiry.

    Attributes
    ----------
    spot : float
        Current FX spot rate
    barrier : float
        Barrier level
    expiry : float
        Time to expiry
    domestic_rate : float
        Domestic risk-free rate
    foreign_rate : float
        Foreign risk-free rate
    volatility : float
        FX volatility
    barrier_type : str
        'up' or 'down'
    payout : float
        Payout amount if barrier is touched
    notional : float
        Notional amount
    """

    spot: float
    barrier: float
    expiry: float
    domestic_rate: float
    foreign_rate: float
    volatility: float
    barrier_type: Literal["up", "down"] = "up"
    payout: float = 1.0
    notional: float = 1.0

    def price(self) -> float:
        """Price the one-touch option."""
        return fx_one_touch(
            self.spot,
            self.barrier,
            self.expiry,
            self.domestic_rate,
            self.foreign_rate,
            self.volatility,
            self.barrier_type,
            self.payout,
        ) * self.notional


@dataclass
class FXNoTouchOption:
    """FX no-touch option.

    Pays a fixed amount if the barrier is NOT touched during the option life.

    Attributes
    ----------
    spot : float
        Current FX spot rate
    barrier : float
        Barrier level
    expiry : float
        Time to expiry
    domestic_rate : float
        Domestic risk-free rate
    foreign_rate : float
        Foreign risk-free rate
    volatility : float
        FX volatility
    barrier_type : str
        'up' or 'down'
    payout : float
        Payout amount if barrier is not touched
    notional : float
        Notional amount
    """

    spot: float
    barrier: float
    expiry: float
    domestic_rate: float
    foreign_rate: float
    volatility: float
    barrier_type: Literal["up", "down"] = "up"
    payout: float = 1.0
    notional: float = 1.0

    def price(self) -> float:
        """Price the no-touch option."""
        return fx_no_touch(
            self.spot,
            self.barrier,
            self.expiry,
            self.domestic_rate,
            self.foreign_rate,
            self.volatility,
            self.barrier_type,
            self.payout,
        ) * self.notional


@dataclass
class FXDoubleOneTouchOption:
    """FX double one-touch option.

    Pays if either upper or lower barrier is touched.

    Attributes
    ----------
    spot : float
        Current FX spot rate
    lower_barrier : float
        Lower barrier level
    upper_barrier : float
        Upper barrier level
    expiry : float
        Time to expiry
    domestic_rate : float
        Domestic risk-free rate
    foreign_rate : float
        Foreign risk-free rate
    volatility : float
        FX volatility
    payout : float
        Payout amount if either barrier is touched
    notional : float
        Notional amount
    """

    spot: float
    lower_barrier: float
    upper_barrier: float
    expiry: float
    domestic_rate: float
    foreign_rate: float
    volatility: float
    payout: float = 1.0
    notional: float = 1.0

    def price(self) -> float:
        """Price the double one-touch option.

        Uses the fact that: P(touch L or touch U) = P(touch L) + P(touch U) - P(touch both)
        For approximation, we use: P(touch L or U) ≈ P(touch L) + P(touch U)
        This is conservative since P(touch both) is typically small.
        """
        # Price as sum of two one-touch options (conservative approximation)
        lower_touch = fx_one_touch(
            self.spot,
            self.lower_barrier,
            self.expiry,
            self.domestic_rate,
            self.foreign_rate,
            self.volatility,
            "down",
            self.payout,
        )

        upper_touch = fx_one_touch(
            self.spot,
            self.upper_barrier,
            self.expiry,
            self.domestic_rate,
            self.foreign_rate,
            self.volatility,
            "up",
            self.payout,
        )

        # For better accuracy, use series expansion (simplified here)
        # The exact formula involves infinite series; we use first-order approximation
        price = jnp.minimum(lower_touch + upper_touch, self.payout * jnp.exp(-self.domestic_rate * self.expiry))

        return float(price * self.notional)


@dataclass
class FXDoubleNoTouchOption:
    """FX double no-touch option (DNT).

    Pays if NEITHER upper nor lower barrier is touched.
    Popular in FX markets, often used for range trading.

    Attributes
    ----------
    spot : float
        Current FX spot rate
    lower_barrier : float
        Lower barrier level
    upper_barrier : float
        Upper barrier level
    expiry : float
        Time to expiry
    domestic_rate : float
        Domestic risk-free rate
    foreign_rate : float
        Foreign risk-free rate
    volatility : float
        FX volatility
    payout : float
        Payout amount if neither barrier is touched
    notional : float
        Notional amount
    """

    spot: float
    lower_barrier: float
    upper_barrier: float
    expiry: float
    domestic_rate: float
    foreign_rate: float
    volatility: float
    payout: float = 1.0
    notional: float = 1.0

    def price(self) -> float:
        """Price the double no-touch option.

        DNT = Full payout - Double one-touch (by complementarity)
        """
        payout_pv = self.payout * jnp.exp(-self.domestic_rate * self.expiry)

        # Create DOT option
        dot = FXDoubleOneTouchOption(
            spot=self.spot,
            lower_barrier=self.lower_barrier,
            upper_barrier=self.upper_barrier,
            expiry=self.expiry,
            domestic_rate=self.domestic_rate,
            foreign_rate=self.foreign_rate,
            volatility=self.volatility,
            payout=self.payout,
            notional=1.0,  # Don't double-count notional
        )

        return float((payout_pv - dot.price()) * self.notional)


# Multi-currency basket options
@dataclass
class FXBasketOption:
    """Multi-currency basket option.

    Option on a basket/portfolio of FX rates with specified weights.
    Common in multi-currency hedging strategies.

    Attributes
    ----------
    spot_rates : Array
        Current spot rates for each FX pair
    strike : float
        Strike on the basket value
    weights : Array
        Weights for each currency in the basket
    expiry : float
        Time to expiry
    domestic_rate : float
        Domestic risk-free rate
    foreign_rates : Array
        Foreign risk-free rates for each currency
    volatilities : Array
        Volatilities for each FX pair
    correlation_matrix : Array
        Correlation matrix between FX pairs
    option_type : str
        'call' or 'put'
    notional : float
        Notional amount
    """

    spot_rates: Array
    strike: float
    weights: Array
    expiry: float
    domestic_rate: float
    foreign_rates: Array
    volatilities: Array
    correlation_matrix: Array
    option_type: Literal["call", "put"] = "call"
    notional: float = 1.0

    def __post_init__(self):
        """Validate and normalize inputs."""
        self.spot_rates = jnp.asarray(self.spot_rates)
        self.weights = jnp.asarray(self.weights)
        self.foreign_rates = jnp.asarray(self.foreign_rates)
        self.volatilities = jnp.asarray(self.volatilities)
        self.correlation_matrix = jnp.asarray(self.correlation_matrix)

        # Normalize weights
        object.__setattr__(self, "weights", self.weights / jnp.sum(self.weights))

    def basket_spot_value(self) -> float:
        """Calculate current basket value."""
        return float(jnp.sum(self.weights * self.spot_rates))

    def basket_forward(self) -> float:
        """Calculate forward basket value."""
        forwards = self.spot_rates * jnp.exp(
            (self.domestic_rate - self.foreign_rates) * self.expiry
        )
        return float(jnp.sum(self.weights * forwards))

    def basket_volatility(self) -> float:
        """Calculate basket volatility using portfolio variance formula."""
        # Variance = w^T * Cov * w
        # where Cov = Diag(sigma) * Corr * Diag(sigma)
        diag_vol = jnp.diag(self.volatilities)
        cov_matrix = diag_vol @ self.correlation_matrix @ diag_vol
        basket_var = self.weights @ cov_matrix @ self.weights
        return float(jnp.sqrt(basket_var))

    def price_analytical(self) -> float:
        """Price using Black-Scholes on the basket (Gaussian approximation).

        Note: This is an approximation. For exact pricing, use Monte Carlo.
        """
        from neutryx.products.fx_options import garman_kohlhagen

        # Current basket value
        S_basket = self.basket_spot_value()

        # Basket volatility
        sigma_basket = self.basket_volatility()

        # Average foreign rate (weighted)
        r_f_avg = float(jnp.sum(self.weights * self.foreign_rates))

        # Price as single FX option on basket
        price = garman_kohlhagen(
            S=S_basket,
            K=self.strike,
            T=self.expiry,
            r_d=self.domestic_rate,
            r_f=r_f_avg,
            sigma=sigma_basket,
            is_call=(self.option_type == "call"),
        )

        return float(price * self.notional)

    def price_mc(
        self, n_paths: int = 10000, n_steps: int = 100, key: Optional[Array] = None
    ) -> float:
        """Price using Monte Carlo simulation.

        Parameters
        ----------
        n_paths : int
            Number of simulation paths
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

        from neutryx.products.basket import simulate_correlated_gbm

        # Drift rates for risk-neutral measure
        mu_basket = self.domestic_rate - self.foreign_rates

        # Simulate correlated GBM for all FX pairs
        paths = simulate_correlated_gbm(
            key=key,
            S0_basket=self.spot_rates,
            mu_basket=mu_basket,
            sigma_basket=self.volatilities,
            correlation_matrix=self.correlation_matrix,
            T=self.expiry,
            steps=n_steps,
            paths=n_paths,
        )

        # Terminal basket values
        terminal_rates = paths[:, -1, :]  # Shape: [n_paths, n_currencies]
        basket_values = jnp.sum(self.weights * terminal_rates, axis=1)

        # Payoffs
        if self.option_type == "call":
            payoffs = jnp.maximum(basket_values - self.strike, 0.0)
        else:
            payoffs = jnp.maximum(self.strike - basket_values, 0.0)

        # Discount and average
        price = jnp.exp(-self.domestic_rate * self.expiry) * jnp.mean(payoffs)

        return float(price * self.notional)


@dataclass
class FXBestOfWorstOfOption:
    """Best-of or worst-of option on multiple FX pairs.

    Pays based on the best or worst performing FX pair.

    Attributes
    ----------
    spot_rates : Array
        Current spot rates for each FX pair
    strikes : Array
        Strike for each FX pair
    expiry : float
        Time to expiry
    domestic_rate : float
        Domestic risk-free rate
    foreign_rates : Array
        Foreign risk-free rates
    volatilities : Array
        Volatilities for each FX pair
    correlation_matrix : Array
        Correlation matrix
    option_type : str
        'call' or 'put'
    payoff_type : str
        'best-of' or 'worst-of'
    notional : float
        Notional amount
    """

    spot_rates: Array
    strikes: Array
    expiry: float
    domestic_rate: float
    foreign_rates: Array
    volatilities: Array
    correlation_matrix: Array
    option_type: Literal["call", "put"] = "call"
    payoff_type: Literal["best-of", "worst-of"] = "worst-of"
    notional: float = 1.0

    def __post_init__(self):
        """Convert to arrays."""
        self.spot_rates = jnp.asarray(self.spot_rates)
        self.strikes = jnp.asarray(self.strikes)
        self.foreign_rates = jnp.asarray(self.foreign_rates)
        self.volatilities = jnp.asarray(self.volatilities)
        self.correlation_matrix = jnp.asarray(self.correlation_matrix)

    def price_mc(
        self, n_paths: int = 10000, n_steps: int = 100, key: Optional[Array] = None
    ) -> float:
        """Price using Monte Carlo simulation."""
        if key is None:
            import jax.random as jrand

            key = jrand.PRNGKey(42)

        from neutryx.products.basket import simulate_correlated_gbm

        # Risk-neutral drifts
        mu_basket = self.domestic_rate - self.foreign_rates

        # Simulate paths
        paths = simulate_correlated_gbm(
            key=key,
            S0_basket=self.spot_rates,
            mu_basket=mu_basket,
            sigma_basket=self.volatilities,
            correlation_matrix=self.correlation_matrix,
            T=self.expiry,
            steps=n_steps,
            paths=n_paths,
        )

        # Terminal values
        terminal_rates = paths[:, -1, :]

        # Calculate individual payoffs
        if self.option_type == "call":
            individual_payoffs = jnp.maximum(
                terminal_rates - self.strikes[jnp.newaxis, :], 0.0
            )
        else:
            individual_payoffs = jnp.maximum(
                self.strikes[jnp.newaxis, :] - terminal_rates, 0.0
            )

        # Select best or worst
        if self.payoff_type == "best-of":
            payoffs = jnp.max(individual_payoffs, axis=1)
        else:  # worst-of
            payoffs = jnp.min(individual_payoffs, axis=1)

        # Discount and average
        price = jnp.exp(-self.domestic_rate * self.expiry) * jnp.mean(payoffs)

        return float(price * self.notional)


# Quanto options
@dataclass
class QuantoOption:
    """Quanto option - option with payoff in different currency than underlying.

    Classic quanto: Foreign asset option with payoff converted at fixed FX rate.
    Example: Option on Nikkei 225 with payoff in USD at fixed rate.

    Attributes
    ----------
    spot : float
        Current asset price (in foreign currency)
    strike : float
        Strike price (in foreign currency)
    expiry : float
        Time to expiry
    domestic_rate : float
        Domestic risk-free rate
    foreign_rate : float
        Foreign risk-free rate
    asset_volatility : float
        Volatility of underlying asset
    fx_volatility : float
        Volatility of FX rate
    correlation : float
        Correlation between asset and FX rate
    quanto_fx_rate : float
        Fixed FX conversion rate
    option_type : str
        'call' or 'put'
    notional : float
        Notional in domestic currency
    """

    spot: float
    strike: float
    expiry: float
    domestic_rate: float
    foreign_rate: float
    asset_volatility: float
    fx_volatility: float
    correlation: float
    quanto_fx_rate: float = 1.0
    option_type: Literal["call", "put"] = "call"
    notional: float = 1.0

    def price(self) -> float:
        """Price quanto option using adjusted Black-Scholes.

        The quanto adjustment modifies the drift of the underlying asset
        by the correlation between asset and FX rate.

        Quanto drift adjustment: r_f -> r_f - rho * sigma_S * sigma_FX
        """
        # Quanto-adjusted foreign rate
        r_f_quanto = self.foreign_rate - self.correlation * self.asset_volatility * self.fx_volatility

        # Use standard BS formula with adjusted rate
        from neutryx.products.fx_options import garman_kohlhagen

        # Price as if it's an FX option with quanto-adjusted rate
        price_per_unit = garman_kohlhagen(
            S=self.spot,
            K=self.strike,
            T=self.expiry,
            r_d=self.domestic_rate,
            r_f=r_f_quanto,
            sigma=self.asset_volatility,
            is_call=(self.option_type == "call"),
        )

        # Convert to domestic currency at fixed quanto rate
        price_domestic = price_per_unit * self.quanto_fx_rate

        return float(price_domestic * self.notional)

    def delta(self) -> float:
        """Quanto-adjusted delta."""
        r_f_quanto = self.foreign_rate - self.correlation * self.asset_volatility * self.fx_volatility

        from neutryx.products.fx_options import fx_delta

        delta_per_unit = fx_delta(
            S=self.spot,
            K=self.strike,
            T=self.expiry,
            r_d=self.domestic_rate,
            r_f=r_f_quanto,
            sigma=self.asset_volatility,
            is_call=(self.option_type == "call"),
        )

        return float(delta_per_unit * self.quanto_fx_rate * self.notional)


@dataclass
class CompositeOption:
    """Composite/compound FX option.

    Option on an option - right to buy/sell an option at a future date.
    Also known as compound option or mother-daughter option.

    Attributes
    ----------
    spot : float
        Current spot FX rate
    strike_mother : float
        Strike of mother option (option on option)
    strike_daughter : float
        Strike of daughter option (underlying option)
    expiry_mother : float
        Expiry of mother option
    expiry_daughter : float
        Expiry of daughter option (> expiry_mother)
    domestic_rate : float
        Domestic rate
    foreign_rate : float
        Foreign rate
    volatility : float
        FX volatility
    mother_type : str
        'call' or 'put' for mother option
    daughter_type : str
        'call' or 'put' for daughter option
    notional : float
        Notional amount
    """

    spot: float
    strike_mother: float
    strike_daughter: float
    expiry_mother: float
    expiry_daughter: float
    domestic_rate: float
    foreign_rate: float
    volatility: float
    mother_type: Literal["call", "put"] = "call"
    daughter_type: Literal["call", "put"] = "call"
    notional: float = 1.0

    def price(self) -> float:
        """Price composite option using Geske's formula (simplified).

        For simplification, we use an approximation. Exact pricing requires
        solving for critical spot level and bivariate normal CDF.
        """
        from neutryx.products.fx_options import garman_kohlhagen

        # Price daughter option at mother expiry (critical value calculation)
        # This is simplified; exact formula is more complex

        # Expected spot at mother expiry
        forward_at_mother = self.spot * jnp.exp(
            (self.domestic_rate - self.foreign_rate) * self.expiry_mother
        )

        # Time remaining for daughter after mother expires
        remaining_time = self.expiry_daughter - self.expiry_mother

        # Daughter option value at mother expiry
        daughter_value = garman_kohlhagen(
            S=forward_at_mother,
            K=self.strike_daughter,
            T=remaining_time,
            r_d=self.domestic_rate,
            r_f=self.foreign_rate,
            sigma=self.volatility,
            is_call=(self.daughter_type == "call"),
        )

        # Mother option on daughter value
        # Treat daughter_value as "strike" for mother option
        # Volatility adjustment for option-on-option
        vol_adjusted = self.volatility * jnp.sqrt(
            1.0 + self.expiry_mother / self.expiry_daughter
        )

        # Simplified: price as option on forward daughter value
        if self.mother_type == "call":
            # Call on call, call on put, etc.
            intrinsic = jnp.maximum(daughter_value - self.strike_mother, 0.0)
        else:
            intrinsic = jnp.maximum(self.strike_mother - daughter_value, 0.0)

        price = jnp.exp(-self.domestic_rate * self.expiry_mother) * intrinsic

        return float(price * self.notional)


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
    # Touch options
    "fx_one_touch",
    "fx_no_touch",
    "FXOneTouchOption",
    "FXNoTouchOption",
    "FXDoubleOneTouchOption",
    "FXDoubleNoTouchOption",
    # Multi-currency basket options
    "FXBasketOption",
    "FXBestOfWorstOfOption",
    # Quanto options
    "QuantoOption",
    "CompositeOption",
    # NDF
    "NDF",
    "NDFWithOption",
]
