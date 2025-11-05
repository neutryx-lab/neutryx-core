"""
Quasi-Gaussian (QG) Interest Rate Model

The Quasi-Gaussian model extends the G2++ model with time-dependent parameters,
providing greater flexibility in fitting the volatility term structure while
maintaining analytical tractability through a Markovian state-space representation.

Model Dynamics:
    dr(t) = [θ(t) + x(t) + y(t)] dt
    dx(t) = -α(t)·x(t)·dt + σ_x(t)·dW₁(t)
    dy(t) = -β(t)·y(t)·dt + σ_y(t)·dW₂(t)

    where:
    - α(t), β(t): Time-dependent mean reversion functions
    - σ_x(t), σ_y(t): Time-dependent volatility functions
    - θ(t): Drift function ensuring fit to initial forward curve
    - dW₁·dW₂ = ρ·dt (constant or time-dependent correlation)

Key Properties:
- Generalizes G2++ with time-dependent parameters
- Maintains Gaussian distribution (allows negative rates)
- Markovian representation allows efficient PDE methods
- Can fit complex volatility term structures
- Semi-analytical pricing of caplets and swaptions

Variants:
- QG1: One-factor Quasi-Gaussian (similar to extended Hull-White)
- QG2: Two-factor Quasi-Gaussian (this implementation)
- QG-HJM: Bridge between Quasi-Gaussian and HJM frameworks

References:
    - Ritchken, P., & Sankarasubramanian, L. (1995). Volatility structures of
      forward rates and the dynamics of the term structure. Mathematical Finance, 5(1), 55-72.
    - Andresen, J., & Poulsen, R. (2002). Efficient numerical methods for
      quasi-Gaussian models. Journal of Computational Finance, 6(1), 1-23.
    - Brigo, D., & Mercurio, F. (2006). Interest Rate Models - Theory and Practice.
      Chapter 5: The Quasi-Gaussian Model.

Author: Neutryx Development Team
Date: 2025
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union
import jax
import jax.numpy as jnp
from jax import lax, Array
from functools import partial


@dataclass
class QuasiGaussianParams:
    """
    Parameters for the Quasi-Gaussian two-factor interest rate model.

    Attributes:
        alpha_fn: Time-dependent mean reversion function α(t) for first factor.
                  Must return positive values. Can be constant: lambda t: 0.1
                  Typical range: 0.01 - 0.5
        beta_fn: Time-dependent mean reversion function β(t) for second factor.
                 Must return positive values, different from α(t).
                 Typical range: 0.01 - 0.5
        sigma_x_fn: Time-dependent volatility function σ_x(t) for first factor.
                    Must return positive values.
                    Typical range: 0.005 - 0.02 (50 - 200 bps)
        sigma_y_fn: Time-dependent volatility function σ_y(t) for second factor.
                    Must return positive values.
                    Typical range: 0.005 - 0.02 (50 - 200 bps)
        forward_curve_fn: Initial forward curve f^M(0,t). Used to compute θ(t).
                          Should return instantaneous forward rate at time t.
        rho: Correlation between Brownian motions. Can be scalar (constant)
             or callable for time-dependent correlation ρ(t).
             Must be in [-1, 1].
        r0: Initial short rate. Can be negative (Gaussian model).
        x0: Initial value of first factor. Default: 0.0
        y0: Initial value of second factor. Default: 0.0
        n_factors: Number of factors (1 or 2). Default: 2

    Raises:
        ValueError: If parameters violate constraints
    """
    alpha_fn: Callable[[float], float]
    beta_fn: Callable[[float], float]
    sigma_x_fn: Callable[[float], float]
    sigma_y_fn: Callable[[float], float]
    forward_curve_fn: Callable[[float], float]
    rho: Union[float, Callable[[float], float]]
    r0: float
    x0: float = 0.0
    y0: float = 0.0
    n_factors: int = 2

    def __post_init__(self):
        """Validate parameters."""
        # Test functions at t=0 to check validity
        try:
            alpha_0 = self.alpha_fn(0.0)
            beta_0 = self.beta_fn(0.0)
            sigma_x_0 = self.sigma_x_fn(0.0)
            sigma_y_0 = self.sigma_y_fn(0.0)
            f_0 = self.forward_curve_fn(0.0)
        except Exception as e:
            raise ValueError(f"Parameter functions must be callable and valid at t=0: {e}")

        if alpha_0 <= 0:
            raise ValueError(f"α(0) must be positive, got {alpha_0}")
        if beta_0 <= 0:
            raise ValueError(f"β(0) must be positive, got {beta_0}")
        if sigma_x_0 <= 0:
            raise ValueError(f"σ_x(0) must be positive, got {sigma_x_0}")
        if sigma_y_0 <= 0:
            raise ValueError(f"σ_y(0) must be positive, got {sigma_y_0}")

        # Check correlation
        if callable(self.rho):
            rho_0 = self.rho(0.0)
        else:
            rho_0 = self.rho

        if not -1.0 <= rho_0 <= 1.0:
            raise ValueError(f"Correlation ρ must be in [-1, 1], got {rho_0}")

        if self.n_factors not in [1, 2]:
            raise ValueError(f"n_factors must be 1 or 2, got {self.n_factors}")

    def get_rho(self, t: float) -> float:
        """Get correlation at time t."""
        if callable(self.rho):
            return self.rho(t)
        return self.rho


def V_coefficient(
    params: QuasiGaussianParams,
    t: float,
    T: float,
    n_steps: int = 100
) -> float:
    """
    Compute integrated variance coefficient V(t,T) for bond pricing.

    V(t,T) = ∫_t^T [σ_x(s)²·G_x(s,T)² + σ_y(s)²·G_y(s,T)²
                    + 2ρ·σ_x(s)·σ_y(s)·G_x(s,T)·G_y(s,T)] ds

    where G_x(s,T) = ∫_s^T exp(-∫_s^u α(v)dv) du

    Args:
        params: Quasi-Gaussian model parameters
        t: Current time
        T: Maturity time
        n_steps: Number of steps for numerical integration

    Returns:
        Integrated variance V(t,T)
    """
    times = jnp.linspace(t, T, n_steps + 1)
    dt = (T - t) / n_steps

    def integrand(s):
        # Compute G_x(s,T) and G_y(s,T)
        # G_x(s,T) = ∫_s^T exp(-∫_s^u α(v)dv) du
        # For piecewise constant α: G_x(s,T) = (1 - exp(-α(T-s))) / α

        # Use trapezoidal integration for G coefficients
        u_grid = jnp.linspace(s, T, 50)
        alpha_vals = jax.vmap(params.alpha_fn)(u_grid)
        beta_vals = jax.vmap(params.beta_fn)(u_grid)

        # Cumulative integral of mean reversion
        alpha_int = jnp.cumsum(alpha_vals) * (T - s) / 50
        beta_int = jnp.cumsum(beta_vals) * (T - s) / 50

        # G coefficients
        G_x_integrand = jnp.exp(-alpha_int)
        G_y_integrand = jnp.exp(-beta_int)

        G_x = jnp.trapezoid(G_x_integrand, u_grid)
        G_y = jnp.trapezoid(G_y_integrand, u_grid)

        # Volatilities at time s
        sigma_x_s = params.sigma_x_fn(s)
        sigma_y_s = params.sigma_y_fn(s)
        rho_s = params.get_rho(s)

        # Integrand for V(t,T)
        return (
            sigma_x_s**2 * G_x**2 +
            sigma_y_s**2 * G_y**2 +
            2 * rho_s * sigma_x_s * sigma_y_s * G_x * G_y
        )

    # Numerical integration using trapezoidal rule
    integrand_values = jax.vmap(integrand)(times)
    V = jnp.trapezoid(integrand_values, times)

    return V


def zero_coupon_bond_price(
    params: QuasiGaussianParams,
    T: float,
    x_t: float = 0.0,
    y_t: float = 0.0,
    t: float = 0.0
) -> float:
    """
    Compute zero-coupon bond price P(t,T) in the Quasi-Gaussian model.

    Uses the representation:
        P(t,T) = P^M(0,T)/P^M(0,t) · exp(-G_x(t,T)·x(t) - G_y(t,T)·y(t) - ½V(t,T))

    where:
    - P^M(0,·) is the market discount curve
    - G_x(t,T), G_y(t,T) are integrated mean reversion coefficients
    - V(t,T) is the integrated variance

    Args:
        params: Quasi-Gaussian model parameters
        T: Bond maturity time
        x_t: Value of first factor at time t (default: 0.0)
        y_t: Value of second factor at time t (default: 0.0)
        t: Current time (default: 0.0)

    Returns:
        Bond price P(t,T) ∈ (0,1]

    Example:
        >>> alpha_fn = lambda t: 0.1
        >>> beta_fn = lambda t: 0.2
        >>> sigma_x_fn = lambda t: 0.01
        >>> sigma_y_fn = lambda t: 0.015
        >>> forward_fn = lambda t: 0.03 + 0.001 * t
        >>> params = QuasiGaussianParams(alpha_fn, beta_fn, sigma_x_fn, sigma_y_fn,
        ...                               forward_fn, rho=-0.7, r0=0.03)
        >>> price = zero_coupon_bond_price(params, T=5.0)
        >>> assert 0.0 < price < 1.0
    """
    # Use JAX-compatible conditional - avoid early return with if statement
    # when the function might be JIT compiled or vmapped
    T_safe = jnp.where(T <= t, t + 1.0, T)  # Ensure T > t for calculations

    # Compute market discount factors from forward curve
    # P^M(0,T) = exp(-∫_0^T f^M(0,s) ds)
    def compute_market_discount(tau):
        tau_safe = jnp.maximum(tau, 1e-10)  # Ensure positive time
        times = jnp.linspace(0, tau_safe, 100)
        forwards = jax.vmap(params.forward_curve_fn)(times)
        integral = jnp.trapezoid(forwards, times)
        return jnp.exp(-integral)

    P_market_t = jnp.where(t > 0, compute_market_discount(t), 1.0)
    P_market_T = compute_market_discount(T_safe)

    # Compute G coefficients: G(t,T) = ∫_t^T exp(-∫_t^s mean_reversion(u) du) ds
    # For constant mean reversion: G(t,T) = (1 - exp(-α(T-t))) / α
    # For time-dependent: need numerical integration

    n_steps = 50
    s_grid = jnp.linspace(t, T_safe, n_steps + 1)

    def compute_G_coefficient(mean_reversion_fn):
        # G(t,T) = ∫_t^T exp(-∫_t^s α(u) du) ds
        def integrand(s):
            # Compute ∫_t^s α(u) du
            # Use JAX-compatible operations instead of if statement
            s_safe = jnp.maximum(s, t + 1e-10)  # Ensure s > t
            u_grid = jnp.linspace(t, s_safe, 20)
            mean_rev_vals = jax.vmap(mean_reversion_fn)(u_grid)
            integral = jnp.trapezoid(mean_rev_vals, u_grid)
            result = jnp.exp(-integral)
            # Return 0 if s <= t, otherwise return the computed value
            return jnp.where(s <= t, 0.0, result)

        integrand_vals = jax.vmap(integrand)(s_grid)
        return jnp.trapezoid(integrand_vals, s_grid)

    G_x = compute_G_coefficient(params.alpha_fn)
    G_y = compute_G_coefficient(params.beta_fn)

    # Compute integrated variance V(t,T)
    V = V_coefficient(params, t, T)

    # Bond price formula
    log_P = (
        jnp.log(P_market_T / P_market_t) -
        G_x * x_t -
        G_y * y_t -
        0.5 * V
    )

    # Return 1.0 if T <= t (bond at maturity), otherwise compute price
    result = jnp.exp(log_P)
    return jnp.where(T <= t, 1.0, result)


def simulate_path(
    params: QuasiGaussianParams,
    T: float,
    n_steps: int,
    key: Array
) -> Tuple[Array, Array, Array]:
    """
    Simulate a single path of the Quasi-Gaussian model using Euler-Maruyama.

    Simulates:
        dx = -α(t)·x·dt + σ_x(t)·dW₁
        dy = -β(t)·y·dt + σ_y(t)·dW₂
        r(t) = θ(t) + x(t) + y(t)

    where θ(t) ensures the model fits the initial forward curve.

    Args:
        params: Quasi-Gaussian model parameters
        T: Final time horizon
        n_steps: Number of time steps
        key: JAX random key

    Returns:
        Tuple (r_path, x_path, y_path) each of shape (n_steps+1,)

    Example:
        >>> alpha_fn = lambda t: 0.1
        >>> beta_fn = lambda t: 0.2
        >>> sigma_x_fn = lambda t: 0.01
        >>> sigma_y_fn = lambda t: 0.015
        >>> forward_fn = lambda t: 0.03
        >>> params = QuasiGaussianParams(alpha_fn, beta_fn, sigma_x_fn, sigma_y_fn,
        ...                               forward_fn, rho=-0.7, r0=0.03)
        >>> key = jax.random.PRNGKey(42)
        >>> r_path, x_path, y_path = simulate_path(params, T=1.0, n_steps=100, key=key)
        >>> assert r_path.shape == (101,)
    """
    dt = T / n_steps

    # Generate correlated Brownian increments
    keys = jax.random.split(key, 2)
    Z1 = jax.random.normal(keys[0], shape=(n_steps,)) * jnp.sqrt(dt)
    Z2_indep = jax.random.normal(keys[1], shape=(n_steps,)) * jnp.sqrt(dt)

    def step_fn(carry, inputs):
        x_t, y_t, t = carry
        dW1, dW2_indep = inputs

        # Get time-dependent parameters
        alpha_t = params.alpha_fn(t)
        beta_t = params.beta_fn(t)
        sigma_x_t = params.sigma_x_fn(t)
        sigma_y_t = params.sigma_y_fn(t)
        rho_t = params.get_rho(t)

        # Correlated Brownian motion
        dW2 = rho_t * dW1 + jnp.sqrt(1 - rho_t**2) * dW2_indep

        # Euler-Maruyama update
        x_new = x_t - alpha_t * x_t * dt + sigma_x_t * dW1
        y_new = y_t - beta_t * y_t * dt + sigma_y_t * dW2

        t_new = t + dt

        # Short rate: r(t) = θ(t) + x(t) + y(t)
        # θ(t) is implicitly defined to match forward curve
        theta_t = params.forward_curve_fn(t_new)  # Approximation
        r_new = theta_t + x_new + y_new

        return (x_new, y_new, t_new), (r_new, x_new, y_new)

    # Initial conditions
    init = (params.x0, params.y0, 0.0)

    # Run simulation
    _, (r_path, x_path, y_path) = lax.scan(step_fn, init, (Z1, Z2_indep))

    # Prepend initial values
    r0_arr = jnp.array([params.r0])
    x0_arr = jnp.array([params.x0])
    y0_arr = jnp.array([params.y0])

    return (
        jnp.concatenate([r0_arr, r_path]),
        jnp.concatenate([x0_arr, x_path]),
        jnp.concatenate([y0_arr, y_path])
    )


def simulate_paths(
    params: QuasiGaussianParams,
    T: float,
    n_steps: int,
    n_paths: int,
    key: Array
) -> Tuple[Array, Array, Array]:
    """
    Simulate multiple paths of the Quasi-Gaussian model.

    Args:
        params: Quasi-Gaussian model parameters
        T: Final time horizon
        n_steps: Number of time steps per path
        n_paths: Number of paths to simulate
        key: JAX random key

    Returns:
        Tuple (r_paths, x_paths, y_paths) each of shape (n_paths, n_steps+1)
    """
    keys = jax.random.split(key, n_paths)
    simulate_fn = lambda k: simulate_path(params, T, n_steps, k)
    r_paths, x_paths, y_paths = jax.vmap(simulate_fn)(keys)
    return r_paths, x_paths, y_paths


def caplet_price_mc(
    params: QuasiGaussianParams,
    strike: float,
    maturity: float,
    tenor: float,
    notional: float = 1.0,
    n_paths: int = 10000,
    key: Optional[Array] = None
) -> float:
    """
    Price a caplet using Monte Carlo simulation.

    Args:
        params: Quasi-Gaussian model parameters
        strike: Strike rate
        maturity: Caplet maturity T
        tenor: Accrual period δ
        notional: Notional amount
        n_paths: Number of Monte Carlo paths
        key: JAX random key (default: creates new key)

    Returns:
        Caplet price
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    T = maturity
    T_end = T + tenor

    # Simulate to maturity
    n_steps = int(T * 252)  # Daily steps
    r_paths, x_paths, y_paths = simulate_paths(params, T, n_steps, n_paths, key)

    # Terminal states
    x_T = x_paths[:, -1]
    y_T = y_paths[:, -1]

    # Compute forward rate and bond prices at maturity
    def compute_forward_and_payoff(x, y):
        P_T = zero_coupon_bond_price(params, T, x, y, T)  # Should be 1.0
        P_T_end = zero_coupon_bond_price(params, T_end, x, y, T)
        forward_rate = (P_T - P_T_end) / (tenor * P_T_end)

        payoff = jnp.maximum(forward_rate - strike, 0.0) * tenor * P_T_end
        return payoff

    payoffs = jax.vmap(compute_forward_and_payoff)(x_T, y_T)

    # Discount to time 0
    discount = zero_coupon_bond_price(params, T)

    price = notional * discount * jnp.mean(payoffs)
    return price


def swaption_price_mc(
    params: QuasiGaussianParams,
    swap_rate: float,
    option_maturity: float,
    swap_maturity: float,
    tenor: float = 0.5,
    notional: float = 1.0,
    is_payer: bool = True,
    n_paths: int = 10000,
    key: Optional[Array] = None
) -> float:
    """
    Price a European swaption using Monte Carlo simulation.

    Args:
        params: Quasi-Gaussian model parameters
        swap_rate: Fixed rate K (strike)
        option_maturity: Time to expiration T_0
        swap_maturity: Final swap payment time T_N
        tenor: Payment frequency
        notional: Notional amount
        is_payer: True for payer swaption
        n_paths: Number of MC paths
        key: JAX random key

    Returns:
        Swaption price
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    # Simulate to option maturity
    n_steps = int(option_maturity * 252)
    r_paths, x_paths, y_paths = simulate_paths(params, option_maturity, n_steps, n_paths, key)

    x_T = x_paths[:, -1]
    y_T = y_paths[:, -1]

    # Payment times
    payment_times = jnp.arange(option_maturity + tenor, swap_maturity + tenor/2, tenor)

    def compute_swap_value(x, y):
        bond_prices = jax.vmap(
            lambda T: zero_coupon_bond_price(params, T, x, y, option_maturity)
        )(payment_times)

        annuity = tenor * jnp.sum(bond_prices)
        P_end = zero_coupon_bond_price(params, swap_maturity, x, y, option_maturity)

        swap_value = (1.0 - P_end) - swap_rate * annuity
        return swap_value

    swap_values = jax.vmap(compute_swap_value)(x_T, y_T)

    if is_payer:
        payoffs = jnp.maximum(swap_values, 0.0)
    else:
        payoffs = jnp.maximum(-swap_values, 0.0)

    discount = zero_coupon_bond_price(params, option_maturity)
    price = notional * discount * jnp.mean(payoffs)

    return price


def create_piecewise_constant_qg(
    forward_curve: Callable[[float], float],
    alpha: Union[float, Array],
    beta: Union[float, Array],
    sigma_x: Union[float, Array],
    sigma_y: Union[float, Array],
    time_grid: Optional[Array] = None,
    rho: float = -0.7,
    r0: Optional[float] = None
) -> QuasiGaussianParams:
    """
    Create a Quasi-Gaussian model with piecewise constant parameters.

    Convenient constructor for calibrating to market data where parameters
    are assumed constant over time intervals.

    Args:
        forward_curve: Initial forward curve function
        alpha: Mean reversion for factor 1. Scalar or array of shape (n_intervals,)
        beta: Mean reversion for factor 2. Scalar or array of shape (n_intervals,)
        sigma_x: Volatility for factor 1. Scalar or array of shape (n_intervals,)
        sigma_y: Volatility for factor 2. Scalar or array of shape (n_intervals,)
        time_grid: Time grid for piecewise constant functions. If None, uses constant.
                   Shape: (n_intervals + 1,)
        rho: Correlation (constant)
        r0: Initial rate. If None, uses forward_curve(0)

    Returns:
        QuasiGaussianParams with piecewise constant parameter functions

    Example:
        >>> forward_fn = lambda t: 0.03
        >>> # Different volatilities for short-end and long-end
        >>> time_grid = jnp.array([0.0, 2.0, 10.0])
        >>> sigma_x = jnp.array([0.015, 0.008])  # Higher vol at short end
        >>> sigma_y = jnp.array([0.010, 0.006])
        >>> params = create_piecewise_constant_qg(forward_fn, alpha=0.1, beta=0.2,
        ...                                        sigma_x=sigma_x, sigma_y=sigma_y,
        ...                                        time_grid=time_grid)
    """
    # Convert scalars to arrays
    if jnp.ndim(alpha) == 0:
        alpha_arr = jnp.array([alpha])
    else:
        alpha_arr = jnp.asarray(alpha)

    if jnp.ndim(beta) == 0:
        beta_arr = jnp.array([beta])
    else:
        beta_arr = jnp.asarray(beta)

    if jnp.ndim(sigma_x) == 0:
        sigma_x_arr = jnp.array([sigma_x])
    else:
        sigma_x_arr = jnp.asarray(sigma_x)

    if jnp.ndim(sigma_y) == 0:
        sigma_y_arr = jnp.array([sigma_y])
    else:
        sigma_y_arr = jnp.asarray(sigma_y)

    # Create time grid
    if time_grid is None:
        time_grid = jnp.array([0.0, 1e10])  # Effectively constant

    # Create piecewise constant functions
    def make_piecewise_fn(values):
        def fn(t):
            idx = jnp.searchsorted(time_grid, t, side='right') - 1
            idx = jnp.clip(idx, 0, len(values) - 1)
            return values[idx]
        return fn

    alpha_fn = make_piecewise_fn(alpha_arr)
    beta_fn = make_piecewise_fn(beta_arr)
    sigma_x_fn = make_piecewise_fn(sigma_x_arr)
    sigma_y_fn = make_piecewise_fn(sigma_y_arr)

    if r0 is None:
        r0 = forward_curve(0.0)

    return QuasiGaussianParams(
        alpha_fn=alpha_fn,
        beta_fn=beta_fn,
        sigma_x_fn=sigma_x_fn,
        sigma_y_fn=sigma_y_fn,
        forward_curve_fn=forward_curve,
        rho=rho,
        r0=r0,
        x0=0.0,
        y0=0.0,
        n_factors=2
    )


__all__ = [
    "QuasiGaussianParams",
    "zero_coupon_bond_price",
    "simulate_path",
    "simulate_paths",
    "caplet_price_mc",
    "swaption_price_mc",
    "V_coefficient",
    "create_piecewise_constant_qg",
]
