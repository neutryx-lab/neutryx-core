"""
G2++ Two-Factor Gaussian Interest Rate Model

The G2++ model (also known as Hull-White Two-Factor or Two-Additive-Factor Gaussian model)
extends the Hull-White one-factor model with an additional factor for richer term structure
dynamics and correlation structure.

Model Dynamics:
    dr(t) = [x(t) + y(t) + φ(t)] dt
    dx(t) = -a·x(t)·dt + σ_x·dW₁(t)
    dy(t) = -b·y(t)·dt + σ_y·dW₂(t)

    where dW₁·dW₂ = ρ·dt (correlated Brownian motions)

    φ(t) is a deterministic drift function that ensures the model fits the initial
    forward curve exactly. When φ(t) = 0, the model reduces to a simple two-factor model.

Key Properties:
- Gaussian distribution of rates (allows negative rates)
- Affine term structure (analytical bond pricing)
- Two factors allow richer volatility term structure
- Analytical pricing of caps, floors, and swaptions
- Calibration to market cap/swaption volatilities

References:
    - Brigo, D., & Mercurio, F. (2006). Interest Rate Models - Theory and Practice.
      Springer Finance.
    - Hull, J., & White, A. (1994). Numerical procedures for implementing term structure
      models II: Two-factor models. Journal of Derivatives, 2(2), 37-48.

Author: Neutryx Development Team
Date: 2025
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import jax
import jax.numpy as jnp
from jax import lax, Array
from functools import partial


@dataclass
class G2PPParams:
    """
    Parameters for the G2++ two-factor Gaussian interest rate model.

    Attributes:
        a: Mean reversion speed for first factor x(t). Must be positive.
           Typical range: 0.01 - 0.5
        b: Mean reversion speed for second factor y(t). Must be positive and
           typically different from a. Typical range: 0.01 - 0.5
        sigma_x: Volatility of first factor. Must be positive.
                 Typical range: 0.005 - 0.02 (50 - 200 bps)
        sigma_y: Volatility of second factor. Must be positive.
                 Typical range: 0.005 - 0.02 (50 - 200 bps)
        rho: Correlation between the two Brownian motions W₁ and W₂.
             Must be in [-1, 1]. Typical range: -0.9 to 0.9
        r0: Initial short rate. Can be negative (Gaussian model).
            Typical range: 0.0 - 0.10 (0% - 10%)
        x0: Initial value of first factor. Default: 0.0
        y0: Initial value of second factor. Default: 0.0
        phi_fn: Optional deterministic drift function φ(t) to fit the initial
                forward curve. If None, assumes φ(t) = 0.

    Raises:
        ValueError: If parameters violate constraints
    """
    a: float
    b: float
    sigma_x: float
    sigma_y: float
    rho: float
    r0: float
    x0: float = 0.0
    y0: float = 0.0
    phi_fn: Optional[Callable[[float], float]] = None

    def __post_init__(self):
        """Validate parameters."""
        if self.a <= 0:
            raise ValueError(f"Mean reversion 'a' must be positive, got {self.a}")
        if self.b <= 0:
            raise ValueError(f"Mean reversion 'b' must be positive, got {self.b}")
        if self.sigma_x <= 0:
            raise ValueError(f"Volatility 'sigma_x' must be positive, got {self.sigma_x}")
        if self.sigma_y <= 0:
            raise ValueError(f"Volatility 'sigma_y' must be positive, got {self.sigma_y}")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError(f"Correlation 'rho' must be in [-1, 1], got {self.rho}")

        # Ensure factors are distinct for identifiability
        if jnp.abs(self.a - self.b) < 1e-6:
            raise ValueError(
                f"Mean reversion speeds must be distinct for identifiability. "
                f"Got a={self.a}, b={self.b}"
            )


def B_coefficient(a: float, t: float, T: float) -> float:
    """
    Compute B coefficient for bond pricing formula.

    B(t,T) = (1 - exp(-a(T-t))) / a

    Args:
        a: Mean reversion speed
        t: Current time
        T: Maturity time

    Returns:
        B coefficient value
    """
    tau = T - t
    return (1.0 - jnp.exp(-a * tau)) / a


def zero_coupon_bond_price(
    params: G2PPParams,
    T: float,
    x_t: float = 0.0,
    y_t: float = 0.0,
    t: float = 0.0
) -> float:
    """
    Compute zero-coupon bond price P(t,T) in the G2++ model.

    Uses the affine form:
        P(t,T) = A(t,T) · exp(-B_x(t,T)·x(t) - B_y(t,T)·y(t))

    where:
        B_x(t,T) = (1 - exp(-a(T-t))) / a
        B_y(t,T) = (1 - exp(-b(T-t))) / b
        A(t,T) = exp(V(t,T) - φ̄(t,T))

    The variance term V(t,T) captures the stochastic volatility contribution.

    Args:
        params: G2++ model parameters
        T: Bond maturity time
        x_t: Value of first factor at time t (default: 0.0)
        y_t: Value of second factor at time t (default: 0.0)
        t: Current time (default: 0.0)

    Returns:
        Bond price P(t,T) ∈ (0,1]

    Example:
        >>> params = G2PPParams(a=0.1, b=0.2, sigma_x=0.01, sigma_y=0.015,
        ...                      rho=-0.7, r0=0.03)
        >>> price = zero_coupon_bond_price(params, T=5.0)
        >>> assert 0.0 < price < 1.0
    """
    tau = T - t

    # Compute B coefficients
    B_x = B_coefficient(params.a, t, T)
    B_y = B_coefficient(params.b, t, T)

    # Compute variance contribution V(t,T)
    # V = (σ_x²/(2a²))[τ - 2B_x + B_x²/(2a)]
    #   + (σ_y²/(2b²))[τ - 2B_y + B_y²/(2b)]
    #   + (ρσ_xσ_y/(ab))[τ - B_x - B_y + B_x·B_y·exp(-aτ)/a]

    V_x = (params.sigma_x**2 / (2 * params.a**2)) * (
        tau - 2 * B_x + B_x * (1 - jnp.exp(-params.a * tau)) / params.a
    )

    V_y = (params.sigma_y**2 / (2 * params.b**2)) * (
        tau - 2 * B_y + B_y * (1 - jnp.exp(-params.b * tau)) / params.b
    )

    V_xy = (params.rho * params.sigma_x * params.sigma_y / (params.a * params.b)) * (
        tau - B_x - B_y + B_x * B_y / tau if tau > 1e-10 else 0.0
    )

    V = V_x + V_y + V_xy

    # Compute φ̄(t,T) = ∫_t^T φ(s) ds
    if params.phi_fn is not None:
        # Numerical integration of φ using trapezoidal rule
        n_steps = 50
        times = jnp.linspace(t, T, n_steps + 1)
        phi_values = jax.vmap(params.phi_fn)(times)
        phi_integral = jnp.trapz(phi_values, times)
    else:
        phi_integral = 0.0

    # Compute log bond price
    log_A = V - phi_integral
    log_P = log_A - B_x * x_t - B_y * y_t

    return jnp.exp(log_P)


@partial(jax.jit, static_argnames=["n_steps", "method"])
def simulate_path(
    params: G2PPParams,
    T: float,
    n_steps: int,
    key: Array,
    method: str = "euler"
) -> Tuple[Array, Array, Array]:
    """
    Simulate a single path of the G2++ model using Euler-Maruyama discretization.

    Simulates the SDEs:
        dx = -a·x·dt + σ_x·dW₁
        dy = -b·y·dt + σ_y·dW₂
        r = x + y + φ(t)

    with correlated Brownian motions dW₁·dW₂ = ρ·dt.

    Args:
        params: G2++ model parameters
        T: Final time horizon
        n_steps: Number of time steps
        key: JAX random key for reproducibility
        method: Discretization method. Options: "euler" (default), "exact"
                "exact" uses analytical solution for the factors

    Returns:
        Tuple (r_path, x_path, y_path) where:
            - r_path: Array of shape (n_steps+1,) with short rate values
            - x_path: Array of shape (n_steps+1,) with first factor values
            - y_path: Array of shape (n_steps+1,) with second factor values

    Example:
        >>> params = G2PPParams(a=0.1, b=0.2, sigma_x=0.01, sigma_y=0.015,
        ...                      rho=-0.7, r0=0.03)
        >>> key = jax.random.PRNGKey(42)
        >>> r_path, x_path, y_path = simulate_path(params, T=1.0, n_steps=100, key=key)
        >>> assert r_path.shape == (101,)
        >>> assert jnp.isclose(r_path[0], params.r0)
    """
    dt = T / n_steps

    # Generate correlated Brownian increments
    # Use Cholesky factorization: [dW₁, dW₂]ᵀ = L · [Z₁, Z₂]ᵀ
    # where L = [[1, 0], [ρ, √(1-ρ²)]]
    keys = jax.random.split(key, 2)
    Z1 = jax.random.normal(keys[0], shape=(n_steps,)) * jnp.sqrt(dt)
    Z2 = jax.random.normal(keys[1], shape=(n_steps,)) * jnp.sqrt(dt)

    dW1 = Z1
    dW2 = params.rho * Z1 + jnp.sqrt(1 - params.rho**2) * Z2

    if method == "exact":
        # Use exact solution for Ornstein-Uhlenbeck processes
        def step_fn(carry, inputs):
            x_t, y_t, t = carry
            dW1_t, dW2_t = inputs

            # Exact solution: x(t+dt) = x(t)·exp(-a·dt) + σ_x·∫exp(-a(t+dt-s))dW₁(s)
            # The integral of the Brownian motion gives a normal with specific variance
            exp_a_dt = jnp.exp(-params.a * dt)
            exp_b_dt = jnp.exp(-params.b * dt)

            x_new = x_t * exp_a_dt + params.sigma_x * dW1_t * jnp.sqrt((1 - exp_a_dt**2) / (2 * params.a))
            y_new = y_t * exp_b_dt + params.sigma_y * dW2_t * jnp.sqrt((1 - exp_b_dt**2) / (2 * params.b))

            t_new = t + dt
            r_new = x_new + y_new + (params.phi_fn(t_new) if params.phi_fn else 0.0)

            return (x_new, y_new, t_new), (r_new, x_new, y_new)
    else:  # euler method
        def step_fn(carry, inputs):
            x_t, y_t, t = carry
            dW1_t, dW2_t = inputs

            # Euler-Maruyama scheme
            x_new = x_t - params.a * x_t * dt + params.sigma_x * dW1_t
            y_new = y_t - params.b * y_t * dt + params.sigma_y * dW2_t

            t_new = t + dt
            r_new = x_new + y_new + (params.phi_fn(t_new) if params.phi_fn else 0.0)

            return (x_new, y_new, t_new), (r_new, x_new, y_new)

    # Initial conditions
    init = (params.x0, params.y0, 0.0)

    # Run simulation
    _, (r_path, x_path, y_path) = lax.scan(step_fn, init, (dW1, dW2))

    # Prepend initial values
    r0_arr = jnp.array([params.r0])
    x0_arr = jnp.array([params.x0])
    y0_arr = jnp.array([params.y0])

    return (
        jnp.concatenate([r0_arr, r_path]),
        jnp.concatenate([x0_arr, x_path]),
        jnp.concatenate([y0_arr, y_path])
    )


@partial(jax.jit, static_argnames=["n_steps", "n_paths", "method"])
def simulate_paths(
    params: G2PPParams,
    T: float,
    n_steps: int,
    n_paths: int,
    key: Array,
    method: str = "euler"
) -> Tuple[Array, Array, Array]:
    """
    Simulate multiple paths of the G2++ model using vectorized operations.

    Args:
        params: G2++ model parameters
        T: Final time horizon
        n_steps: Number of time steps per path
        n_paths: Number of paths to simulate
        key: JAX random key
        method: Discretization method ("euler" or "exact")

    Returns:
        Tuple (r_paths, x_paths, y_paths) where each has shape (n_paths, n_steps+1)

    Example:
        >>> params = G2PPParams(a=0.1, b=0.2, sigma_x=0.01, sigma_y=0.015,
        ...                      rho=-0.7, r0=0.03)
        >>> key = jax.random.PRNGKey(42)
        >>> r_paths, x_paths, y_paths = simulate_paths(params, T=1.0, n_steps=100,
        ...                                              n_paths=1000, key=key)
        >>> assert r_paths.shape == (1000, 101)
    """
    keys = jax.random.split(key, n_paths)

    # Vectorize over paths
    simulate_fn = lambda k: simulate_path(params, T, n_steps, k, method=method)
    r_paths, x_paths, y_paths = jax.vmap(simulate_fn)(keys)

    return r_paths, x_paths, y_paths


def caplet_price(
    params: G2PPParams,
    strike: float,
    maturity: float,
    tenor: float,
    notional: float = 1.0
) -> float:
    """
    Price a caplet using the G2++ model with analytical formula.

    A caplet pays max(L(T,T+δ) - K, 0) at time T+δ, where L is the LIBOR rate
    over [T, T+δ] and K is the strike.

    In the G2++ model, this can be computed using Black's formula with
    an adjusted volatility that accounts for both factors and their correlation.

    Args:
        params: G2++ model parameters
        strike: Strike rate (annualized)
        maturity: Caplet maturity T
        tenor: Accrual period δ (e.g., 0.25 for 3M LIBOR)
        notional: Notional amount (default: 1.0)

    Returns:
        Caplet price

    Notes:
        Uses the approximation that the forward rate is log-normally distributed
        under the T+δ forward measure, with variance computed from the two-factor
        model structure.
    """
    T = maturity
    delta = tenor
    T_end = T + delta

    # Compute forward rate F(0,T,T+δ) = (P(0,T) - P(0,T+δ)) / (δ·P(0,T+δ))
    P_T = zero_coupon_bond_price(params, T)
    P_T_end = zero_coupon_bond_price(params, T_end)
    forward_rate = (P_T - P_T_end) / (delta * P_T_end)

    # Compute integrated variance for the forward rate
    # This requires computing the variance of ln(F(t,T,T+δ)) over [0,T]
    # Approximation: use variance from both factors

    a, b = params.a, params.b
    sigma_x, sigma_y, rho = params.sigma_x, params.sigma_y, params.rho

    # Compute B coefficients
    B_x_T = B_coefficient(a, 0, T)
    B_x_T_end = B_coefficient(a, 0, T_end)
    B_y_T = B_coefficient(b, 0, T)
    B_y_T_end = B_coefficient(b, 0, T_end)

    # Delta B coefficients (sensitivity to factors)
    Delta_B_x = B_x_T - B_x_T_end
    Delta_B_y = B_y_T - B_y_T_end

    # Integrated variance contribution
    V_x = (sigma_x**2 / (2 * a**3)) * (1 - jnp.exp(-2 * a * T)) * Delta_B_x**2
    V_y = (sigma_y**2 / (2 * b**3)) * (1 - jnp.exp(-2 * b * T)) * Delta_B_y**2
    V_xy = (rho * sigma_x * sigma_y / (a * b * (a + b))) * (
        1 - jnp.exp(-(a + b) * T)
    ) * Delta_B_x * Delta_B_y

    variance = V_x + V_y + 2 * V_xy
    volatility = jnp.sqrt(variance / T) if T > 0 else 0.0

    # Black's formula for caplet
    d1 = (jnp.log(forward_rate / strike) + 0.5 * variance) / jnp.sqrt(variance)
    d2 = d1 - jnp.sqrt(variance)

    from jax.scipy.stats import norm

    price = notional * delta * P_T_end * (
        forward_rate * norm.cdf(d1) - strike * norm.cdf(d2)
    )

    return price


def swaption_price(
    params: G2PPParams,
    swap_rate: float,
    option_maturity: float,
    swap_maturity: float,
    tenor: float = 0.5,
    notional: float = 1.0,
    is_payer: bool = True
) -> float:
    """
    Price a European swaption using the G2++ model.

    A payer swaption gives the right to enter a swap paying fixed rate K
    and receiving floating. A receiver swaption is the opposite.

    Args:
        params: G2++ model parameters
        swap_rate: Fixed rate K (strike)
        option_maturity: Time to expiration T_0
        swap_maturity: Final swap payment time T_N
        tenor: Payment frequency (e.g., 0.5 for semi-annual)
        notional: Notional amount
        is_payer: True for payer swaption, False for receiver

    Returns:
        Swaption price

    Notes:
        Uses Monte Carlo simulation to compute the swaption payoff, as
        analytical formulas for G2++ swaptions are complex.
    """
    # For simplicity, use a Monte Carlo approach
    # In production, Jamshidian's trick or other advanced methods could be used

    n_sim_paths = 10000
    key = jax.random.PRNGKey(0)

    # Simulate to option maturity
    n_steps = int(option_maturity * 252)  # Daily steps
    r_paths, x_paths, y_paths = simulate_paths(
        params, option_maturity, n_steps, n_sim_paths, key
    )

    # Get terminal values of factors
    x_T = x_paths[:, -1]  # Shape: (n_sim_paths,)
    y_T = y_paths[:, -1]

    # Compute swap value at each terminal state
    # Swap value = Σᵢ δᵢ·P(T₀,Tᵢ)·(L(T₀,Tᵢ₋₁,Tᵢ) - K)
    # For fixed-for-floating: V = (1 - P(T₀,T_N)) - K·Σᵢ δᵢ·P(T₀,Tᵢ)

    payment_times = jnp.arange(option_maturity + tenor, swap_maturity + tenor/2, tenor)

    def compute_swap_value(x_t, y_t):
        # Compute annuity and discount to swap end
        bond_prices = jax.vmap(
            lambda T: zero_coupon_bond_price(params, T, x_t, y_t, option_maturity)
        )(payment_times)

        annuity = tenor * jnp.sum(bond_prices)
        P_end = zero_coupon_bond_price(params, swap_maturity, x_t, y_t, option_maturity)

        # Swap value for payer
        swap_value = (1.0 - P_end) - swap_rate * annuity
        return swap_value

    swap_values = jax.vmap(compute_swap_value)(x_T, y_T)

    # Payoff
    if is_payer:
        payoffs = jnp.maximum(swap_values, 0.0)
    else:
        payoffs = jnp.maximum(-swap_values, 0.0)

    # Discount back to time 0
    discount_factor = zero_coupon_bond_price(params, option_maturity)

    price = notional * discount_factor * jnp.mean(payoffs)

    return price


def forward_rate_correlation(
    params: G2PPParams,
    T1: float,
    T2: float,
    maturity: float
) -> float:
    """
    Compute instantaneous correlation between two forward rates.

    The correlation between f(t,T₁) and f(t,T₂) at time t, where
    f(t,T) is the instantaneous forward rate for maturity T.

    Args:
        params: G2++ model parameters
        T1: First forward rate maturity
        T2: Second forward rate maturity
        maturity: Time at which to compute correlation

    Returns:
        Correlation coefficient ∈ [-1, 1]

    Notes:
        In the G2++ model:
        df(t,T) = [dB_x/dT·σ_x·dW₁ + dB_y/dT·σ_y·dW₂]

        Correlation arises from the correlation ρ between W₁ and W₂ and
        the relative contributions of each factor to different maturities.
    """
    t = maturity

    # Compute derivative of B coefficients: dB/dT = exp(-a(T-t))
    dB_x_T1 = jnp.exp(-params.a * (T1 - t))
    dB_x_T2 = jnp.exp(-params.a * (T2 - t))
    dB_y_T1 = jnp.exp(-params.b * (T1 - t))
    dB_y_T2 = jnp.exp(-params.b * (T2 - t))

    # Instantaneous volatility of forward rates
    vol_f1_x = params.sigma_x * dB_x_T1
    vol_f1_y = params.sigma_y * dB_y_T1
    vol_f2_x = params.sigma_x * dB_x_T2
    vol_f2_y = params.sigma_y * dB_y_T2

    # Covariance: E[df(t,T₁)·df(t,T₂)]
    covariance = (
        vol_f1_x * vol_f2_x +  # Contribution from W₁
        vol_f1_y * vol_f2_y +  # Contribution from W₂
        params.rho * (vol_f1_x * vol_f2_y + vol_f1_y * vol_f2_x)  # Cross terms
    )

    # Variances
    var_f1 = vol_f1_x**2 + vol_f1_y**2 + 2 * params.rho * vol_f1_x * vol_f1_y
    var_f2 = vol_f2_x**2 + vol_f2_y**2 + 2 * params.rho * vol_f2_x * vol_f2_y

    # Correlation
    correlation = covariance / jnp.sqrt(var_f1 * var_f2 + 1e-10)

    return jnp.clip(correlation, -1.0, 1.0)


# Convenience functions for common use cases

def create_fitted_g2pp(
    initial_curve: Callable[[float], float],
    a: float = 0.1,
    b: float = 0.2,
    sigma_x: float = 0.01,
    sigma_y: float = 0.015,
    rho: float = -0.7
) -> G2PPParams:
    """
    Create a G2++ model calibrated to fit an initial forward curve.

    Computes the drift function φ(t) such that the model exactly reproduces
    the given forward curve f^M(0,t) at time 0.

    Args:
        initial_curve: Function f^M(0,t) giving initial forward rates
        a, b: Mean reversion speeds
        sigma_x, sigma_y: Volatilities
        rho: Correlation

    Returns:
        G2PPParams with calibrated φ(t) function

    Example:
        >>> forward_curve = lambda t: 0.03 + 0.001 * t  # Upward sloping
        >>> params = create_fitted_g2pp(forward_curve, a=0.1, b=0.2,
        ...                              sigma_x=0.01, sigma_y=0.015, rho=-0.7)
    """
    # Compute φ(t) such that f^model(0,t) = f^market(0,t)
    # φ(t) = ∂f^M/∂t + a·f^M(0,t) + corrections from variance

    def phi_function(t):
        # Use numerical derivative
        dt = 1e-4
        df_dt = (initial_curve(t + dt) - initial_curve(t - dt)) / (2 * dt)

        # Variance correction terms (small for typical parameters)
        var_correction = (
            sigma_x**2 / (2 * a**2) * (1 - jnp.exp(-a * t))**2 +
            sigma_y**2 / (2 * b**2) * (1 - jnp.exp(-b * t))**2
        )

        return df_dt + a * initial_curve(t) + var_correction

    r0 = initial_curve(0.0)

    return G2PPParams(
        a=a, b=b,
        sigma_x=sigma_x, sigma_y=sigma_y,
        rho=rho,
        r0=r0,
        x0=0.0, y0=0.0,
        phi_fn=phi_function
    )


__all__ = [
    "G2PPParams",
    "zero_coupon_bond_price",
    "simulate_path",
    "simulate_paths",
    "caplet_price",
    "swaption_price",
    "forward_rate_correlation",
    "create_fitted_g2pp",
]
