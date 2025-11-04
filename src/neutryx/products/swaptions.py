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

import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy.stats import norm

from .base import Product


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


@dataclass
class EuropeanSwaption(Product):
    """European swaption payoff helper.

    Parameters
    ----------
    T : float
        Time to option expiry in years.
    strike : float
        Strike (fixed) rate of the underlying swap.
    annuity : float
        Present value of one unit payment per period (swap annuity).
    notional : float, default 1_000_000.0
        Notional of the swaption.
    swaption_type : SwaptionType, default PAYER
        Select payer (pay fixed) or receiver (receive fixed) swaption.
    """

    T: float
    strike: float
    annuity: float
    notional: float = 1_000_000.0
    swaption_type: SwaptionType = SwaptionType.PAYER

    def payoff_terminal(self, forward_swap_rate: jnp.ndarray) -> jnp.ndarray:
        """Return intrinsic value for a terminal forward swap rate."""

        rate = jnp.asarray(forward_swap_rate, dtype=jnp.float32)
        intrinsic = (
            rate - self.strike
            if self.swaption_type == SwaptionType.PAYER
            else self.strike - rate
        )
        intrinsic = jnp.maximum(intrinsic, 0.0)
        return self.notional * self.annuity * intrinsic

    def price_black(
        self,
        forward_swap_rate: float,
        volatility: float,
        *,
        option_maturity: float | None = None,
    ) -> float:
        """Price the swaption via Black's formula given a forward rate."""

        maturity = float(self.T if option_maturity is None else option_maturity)
        return float(
            black_swaption_price(
                forward_swap_rate,
                self.strike,
                maturity,
                volatility,
                self.annuity,
                self.notional,
                is_payer=self.swaption_type == SwaptionType.PAYER,
            )
        )

    def delta(
        self,
        forward_swap_rate: float,
        volatility: float,
        *,
        option_maturity: float | None = None,
    ) -> float:
        """Return Black delta with respect to the forward swap rate."""

        maturity = float(self.T if option_maturity is None else option_maturity)
        return float(
            swaption_delta(
                forward_swap_rate,
                self.strike,
                maturity,
                volatility,
                self.annuity,
                self.notional,
                is_payer=self.swaption_type == SwaptionType.PAYER,
            )
        )

    def vega(
        self,
        forward_swap_rate: float,
        volatility: float,
        *,
        option_maturity: float | None = None,
    ) -> float:
        """Return Black vega for the swaption."""

        maturity = float(self.T if option_maturity is None else option_maturity)
        return float(
            swaption_vega(
                forward_swap_rate,
                self.strike,
                maturity,
                volatility,
                self.annuity,
                self.notional,
            )
        )


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
    df_option_maturity: float | None = None,
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
    df_option_maturity : float, optional
        Discount factor to the swaption expiry.

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
    discount_factors = jnp.asarray(discount_factors, dtype=jnp.float32)
    year_fractions = jnp.asarray(year_fractions, dtype=jnp.float32)

    if discount_factors.shape != year_fractions.shape:
        raise ValueError("Discount factors and year fractions must share shape.")

    if df_option_maturity is None:
        # Bootstrap DF(T) from first payment assuming flat forward over first accrual
        first_payment = option_maturity + year_fractions[0]
        implied_rate = -jnp.log(discount_factors[0]) / first_payment
        df_option_maturity = jnp.exp(-implied_rate * option_maturity)
    else:
        df_option_maturity = jnp.asarray(df_option_maturity, dtype=jnp.float32)

    df_end = discount_factors[-1]
    annuity = swap_annuity(discount_factors, year_fractions)

    return (df_option_maturity - df_end) / jnp.maximum(annuity, 1e-12)


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


def american_swaption_tree(
    strike: float,
    option_maturity: float,
    swap_maturity: float,
    initial_rate: float,
    volatility: float,
    discount_rate: float = 0.03,
    notional: float = 1_000_000.0,
    payment_frequency: int = 2,
    is_payer: bool = True,
    n_steps: int = 100,
) -> float:
    """Price an American swaption using a trinomial tree.

    American swaptions can be exercised at any time before maturity,
    requiring numerical methods like trees or LSM.

    Parameters
    ----------
    strike : float
        Fixed rate (strike) of the swaption
    option_maturity : float
        Time to swaption expiry in years
    swap_maturity : float
        Tenor of the underlying swap in years
    initial_rate : float
        Initial short rate
    volatility : float
        Interest rate volatility
    discount_rate : float
        Risk-free discount rate
    notional : float
        Notional principal amount
    payment_frequency : int
        Number of payments per year
    is_payer : bool
        True for payer swaption, False for receiver
    n_steps : int
        Number of tree steps

    Returns
    -------
    float
        American swaption price

    Notes
    -----
    Uses a trinomial tree for the short rate with backward induction.
    At each node, compares immediate exercise value with continuation value.
    """
    dt = option_maturity / n_steps

    # Trinomial tree parameters
    dr = volatility * jnp.sqrt(3.0 * dt)
    pu = 1.0 / 6.0 + (discount_rate - initial_rate) ** 2 * dt / (6.0 * dr**2)
    pm = 2.0 / 3.0
    pd = 1.0 / 6.0 - (discount_rate - initial_rate) ** 2 * dt / (6.0 * dr**2)

    # Ensure probabilities are valid
    pu = float(jnp.clip(pu, 0.0, 1.0))
    pd = float(jnp.clip(pd, 0.0, 1.0))
    pm = float(jnp.clip(1.0 - pu - pd, 0.0, 1.0))

    # Initialize rate tree
    n_nodes = 2 * n_steps + 1
    rates = jnp.zeros((n_steps + 1, n_nodes))
    values = jnp.zeros((n_steps + 1, n_nodes))

    # Build rate lattice
    for i in range(n_steps + 1):
        for j in range(-i, i + 1):
            rates = rates.at[i, j + n_steps].set(initial_rate + j * dr)

    # Calculate swap annuity and forward swap rate at each node
    def calculate_swap_value(rate, time_left):
        """Calculate swap value at a given node."""
        if time_left <= 0:
            return 0.0

        n_payments = int(swap_maturity * payment_frequency)
        year_fraction = 1.0 / payment_frequency

        # Simplified: assume flat rate for discounting
        payment_times = jnp.arange(1, n_payments + 1) * year_fraction
        discount_factors = jnp.exp(-rate * payment_times)

        annuity = float(jnp.sum(discount_factors * year_fraction))

        # Forward swap rate (simplified)
        df_start = 1.0
        df_end = jnp.exp(-rate * swap_maturity)
        fwd_swap_rate = (df_start - df_end) / annuity

        # Intrinsic value
        if is_payer:
            intrinsic = jnp.maximum(fwd_swap_rate - strike, 0.0)
        else:
            intrinsic = jnp.maximum(strike - fwd_swap_rate, 0.0)

        return notional * annuity * intrinsic

    # Terminal condition
    for j in range(-n_steps, n_steps + 1):
        rate = rates[n_steps, j + n_steps]
        values = values.at[n_steps, j + n_steps].set(calculate_swap_value(rate, 0))

    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        time_left = option_maturity - i * dt
        for j in range(-i, i + 1):
            idx = j + n_steps
            rate = rates[i, idx]

            # Continuation value (discounted expected value)
            continuation = (
                pu * values[i + 1, idx + 1]
                + pm * values[i + 1, idx]
                + pd * values[i + 1, idx - 1]
            ) * jnp.exp(-rate * dt)

            # Exercise value
            exercise = calculate_swap_value(rate, time_left)

            # American option: take max
            values = values.at[i, idx].set(jnp.maximum(continuation, exercise))

    return float(values[0, n_steps])


def american_swaption_lsm(
    strike: float,
    option_maturity: float,
    swap_maturity: float,
    rate_paths: jnp.ndarray,
    discount_factors: jnp.ndarray,
    notional: float = 1_000_000.0,
    payment_frequency: int = 2,
    is_payer: bool = True,
) -> float:
    """Price an American swaption using Longstaff-Schwartz Monte Carlo.

    Parameters
    ----------
    strike : float
        Fixed rate (strike) of the swaption
    option_maturity : float
        Time to swaption expiry in years
    swap_maturity : float
        Tenor of the underlying swap in years
    rate_paths : Array
        Simulated short rate paths [n_paths, n_steps]
    discount_factors : Array
        Discount factors for each time step [n_steps]
    notional : float
        Notional principal amount
    payment_frequency : int
        Number of payments per year
    is_payer : bool
        True for payer swaption, False for receiver

    Returns
    -------
    float
        American swaption price

    Notes
    -----
    Uses Longstaff-Schwartz regression to estimate continuation values.
    """
    n_paths, n_steps = rate_paths.shape

    # Basis functions for regression
    def basis_functions(x):
        return jnp.array([jnp.ones_like(x), x, x**2, x**3]).T

    # Calculate intrinsic values at each time step
    def calculate_intrinsic(rate):
        """Calculate intrinsic value of swaption."""
        n_payments = int(swap_maturity * payment_frequency)
        year_fraction = 1.0 / payment_frequency

        payment_times = jnp.arange(1, n_payments + 1) * year_fraction
        discount_factors_swap = jnp.exp(-rate * payment_times)
        annuity = jnp.sum(discount_factors_swap * year_fraction)

        df_start = 1.0
        df_end = jnp.exp(-rate * swap_maturity)
        fwd_swap_rate = (df_start - df_end) / annuity

        if is_payer:
            intrinsic = jnp.maximum(fwd_swap_rate - strike, 0.0)
        else:
            intrinsic = jnp.maximum(strike - fwd_swap_rate, 0.0)

        return notional * annuity * intrinsic

    # Initialize cashflows
    cashflows = jnp.zeros(n_paths)

    # Start from second-to-last time step
    for t in range(n_steps - 1, 0, -1):
        current_rates = rate_paths[:, t]
        intrinsic = jax.vmap(calculate_intrinsic)(current_rates)

        # Only consider in-the-money paths
        itm_mask = intrinsic > 0

        if jnp.sum(itm_mask) > 10:  # Need enough paths for regression
            # Regression for continuation value
            X = basis_functions(current_rates[itm_mask])
            Y = cashflows[itm_mask] * discount_factors[t]

            # Least squares
            coeffs = jnp.linalg.lstsq(X, Y)[0]
            continuation_value = jnp.zeros_like(cashflows)
            continuation_value = continuation_value.at[itm_mask].set(
                basis_functions(current_rates[itm_mask]) @ coeffs
            )

            # Exercise if intrinsic > continuation
            exercise_now = (intrinsic > continuation_value) & itm_mask
            cashflows = jnp.where(exercise_now, intrinsic, cashflows)

    # Terminal value for paths not yet exercised
    terminal_rates = rate_paths[:, -1]
    terminal_intrinsic = jax.vmap(calculate_intrinsic)(terminal_rates)
    not_exercised = cashflows == 0
    cashflows = jnp.where(not_exercised, terminal_intrinsic, cashflows)

    # Discount back to present
    discounted_cashflows = cashflows * discount_factors[0]

    return float(jnp.mean(discounted_cashflows))


__all__ = [
    "SwaptionSpecs",
    "SwaptionType",
    "EuropeanSwaption",
    "black_swaption_price",
    "european_swaption_black",
    "forward_swap_rate",
    "implied_swaption_volatility",
    "swap_annuity",
    "swaption_delta",
    "swaption_vega",
    # American swaptions
    "american_swaption_tree",
    "american_swaption_lsm",
]
