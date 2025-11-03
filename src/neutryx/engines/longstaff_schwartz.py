"""Longstaff-Schwartz algorithm for pricing American and Bermudan options.

The Longstaff-Schwartz (LS) method uses least-squares regression to estimate
the continuation value at each exercise point, enabling Monte Carlo pricing
of early-exercise options.

References:
    Longstaff, F. A., & Schwartz, E. S. (2001). "Valuing American Options by Simulation:
    A Simple Least-Squares Approach." The Review of Financial Studies, 14(1), 113-147.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
from jax import Array


@dataclass
class LSMConfig:
    """Configuration for Longstaff-Schwartz Monte Carlo.

    Attributes:
        basis_functions: Sequence of basis functions for regression
        polynomial_degree: Degree of polynomial basis (if using polynomials)
        paths: Number of Monte Carlo paths
        exercise_dates: Array of exercise dates (in years)
        discount_rate: Risk-free discount rate
    """

    paths: int = 10000
    polynomial_degree: int = 3
    exercise_dates: Optional[Array] = None
    discount_rate: float = 0.05
    basis_type: str = "laguerre"  # "laguerre", "hermite", "power", "chebyshev"

    def __post_init__(self):
        if self.paths <= 0:
            raise ValueError("Number of paths must be positive")
        if self.polynomial_degree < 1:
            raise ValueError("Polynomial degree must be at least 1")


def power_basis(x: Array, degree: int) -> Array:
    """Generate power basis functions: [1, x, x^2, ..., x^degree].

    Args:
        x: Input values [n_samples]
        degree: Maximum polynomial degree

    Returns:
        Basis matrix [n_samples, degree+1]
    """
    powers = jnp.arange(degree + 1)
    return jnp.power(x[:, None], powers)


def laguerre_basis(x: Array, degree: int) -> Array:
    """Generate Laguerre polynomial basis functions.

    Laguerre polynomials are well-suited for positive-valued random variables
    (like stock prices) and are orthogonal with respect to the exponential
    weight function.

    Args:
        x: Input values [n_samples]
        degree: Maximum polynomial degree

    Returns:
        Basis matrix [n_samples, degree+1]

    Notes:
        L_0(x) = 1
        L_1(x) = 1 - x
        L_2(x) = 1 - 2x + x^2/2
        Recurrence: (n+1)L_{n+1}(x) = (2n+1-x)L_n(x) - nL_{n-1}(x)
    """
    n_samples = x.shape[0]
    basis = jnp.zeros((n_samples, degree + 1))

    # L_0(x) = 1
    basis = basis.at[:, 0].set(1.0)

    if degree >= 1:
        # L_1(x) = 1 - x
        basis = basis.at[:, 1].set(1.0 - x)

    # Recurrence relation
    for n in range(1, degree):
        L_n = basis[:, n]
        L_n_minus_1 = basis[:, n - 1]
        L_n_plus_1 = ((2 * n + 1 - x) * L_n - n * L_n_minus_1) / (n + 1)
        basis = basis.at[:, n + 1].set(L_n_plus_1)

    # Weight by exp(-x/2) for better numerical properties
    weighted_basis = basis * jnp.exp(-x[:, None] / 2)

    return weighted_basis


def hermite_basis(x: Array, degree: int) -> Array:
    """Generate Hermite polynomial basis functions.

    Hermite polynomials are orthogonal with respect to the Gaussian weight
    and are well-suited for normally distributed variables.

    Args:
        x: Input values [n_samples]
        degree: Maximum polynomial degree

    Returns:
        Basis matrix [n_samples, degree+1]

    Notes:
        H_0(x) = 1
        H_1(x) = 2x
        Recurrence: H_{n+1}(x) = 2xH_n(x) - 2nH_{n-1}(x)
    """
    n_samples = x.shape[0]
    basis = jnp.zeros((n_samples, degree + 1))

    # H_0(x) = 1
    basis = basis.at[:, 0].set(1.0)

    if degree >= 1:
        # H_1(x) = 2x
        basis = basis.at[:, 1].set(2.0 * x)

    # Recurrence relation
    for n in range(1, degree):
        H_n = basis[:, n]
        H_n_minus_1 = basis[:, n - 1]
        H_n_plus_1 = 2 * x * H_n - 2 * n * H_n_minus_1
        basis = basis.at[:, n + 1].set(H_n_plus_1)

    # Weight by exp(-x^2/2) for numerical stability
    weighted_basis = basis * jnp.exp(-x[:, None]**2 / 2)

    return weighted_basis


def chebyshev_basis(x: Array, degree: int) -> Array:
    """Generate Chebyshev polynomial basis functions (first kind).

    Args:
        x: Input values [n_samples], should be in [-1, 1]
        degree: Maximum polynomial degree

    Returns:
        Basis matrix [n_samples, degree+1]

    Notes:
        T_0(x) = 1
        T_1(x) = x
        Recurrence: T_{n+1}(x) = 2xT_n(x) - T_{n-1}(x)
    """
    n_samples = x.shape[0]
    basis = jnp.zeros((n_samples, degree + 1))

    # Clip x to [-1, 1] to avoid numerical issues
    x_clipped = jnp.clip(x, -1.0, 1.0)

    # T_0(x) = 1
    basis = basis.at[:, 0].set(1.0)

    if degree >= 1:
        # T_1(x) = x
        basis = basis.at[:, 1].set(x_clipped)

    # Recurrence relation
    for n in range(1, degree):
        T_n = basis[:, n]
        T_n_minus_1 = basis[:, n - 1]
        T_n_plus_1 = 2 * x_clipped * T_n - T_n_minus_1
        basis = basis.at[:, n + 1].set(T_n_plus_1)

    return basis


def get_basis_functions(basis_type: str, degree: int) -> Callable[[Array], Array]:
    """Get basis function generator by type.

    Args:
        basis_type: Type of basis ("power", "laguerre", "hermite", "chebyshev")
        degree: Polynomial degree

    Returns:
        Function that generates basis matrix from input array
    """
    if basis_type == "power":
        return lambda x: power_basis(x, degree)
    elif basis_type == "laguerre":
        return lambda x: laguerre_basis(x, degree)
    elif basis_type == "hermite":
        return lambda x: hermite_basis(x, degree)
    elif basis_type == "chebyshev":
        return lambda x: chebyshev_basis(x, degree)
    else:
        raise ValueError(f"Unknown basis type: {basis_type}")


def longstaff_schwartz_american(
    paths: Array,
    payoff_fn: Callable[[Array], Array],
    exercise_times: Array,
    discount_factors: Array,
    config: LSMConfig
) -> tuple[float, Array, Array]:
    """Price an American option using Longstaff-Schwartz algorithm.

    Args:
        paths: Simulated asset price paths [n_paths, n_steps]
        payoff_fn: Function computing immediate exercise value
        exercise_times: Times at which exercise is allowed [n_exercise]
        discount_factors: Discount factors for each exercise time [n_exercise]
        config: LSM configuration

    Returns:
        Tuple of (option_price, optimal_exercise_times, cash_flows)
        - option_price: Estimated American option value
        - optimal_exercise_times: Exercise time for each path [n_paths]
        - cash_flows: Discounted cash flows for each path [n_paths]

    Notes:
        The algorithm works backwards from expiry:
        1. At each exercise point, compute immediate exercise value
        2. For in-the-money paths, regress continuation value on basis functions
        3. Exercise if immediate value > continuation value
        4. Update cash flows and exercise times
    """
    n_paths, n_steps = paths.shape
    n_exercise = len(exercise_times)

    # Initialize cash flow array (discounted to time 0)
    cash_flows = jnp.zeros(n_paths)

    # Exercise time for each path (0 = not exercised)
    exercise_indicator = jnp.zeros(n_paths, dtype=jnp.int32)

    # Basis function generator
    basis_fn = get_basis_functions(config.basis_type, config.polynomial_degree)

    # Start from the last exercise date and work backwards
    # At expiry, exercise value is the payoff
    final_payoff = payoff_fn(paths[:, -1])
    cash_flows = final_payoff * discount_factors[-1]
    exercise_indicator = jnp.ones(n_paths, dtype=jnp.int32) * (n_exercise - 1)

    # Backward induction through exercise dates
    for t_idx in range(n_exercise - 2, -1, -1):
        # Current stock prices at this exercise date
        # Map exercise time to path index
        path_idx = jnp.searchsorted(
            jnp.linspace(0, exercise_times[-1], n_steps),
            exercise_times[t_idx]
        )
        path_idx = jnp.clip(path_idx, 0, n_steps - 1)

        S_t = paths[:, path_idx]

        # Immediate exercise value
        immediate_exercise = payoff_fn(S_t)

        # Find in-the-money paths
        itm = immediate_exercise > 0

        if jnp.sum(itm) > 0:
            # Regression for continuation value (only on ITM paths)
            S_itm = S_t[itm]
            cash_flows_itm = cash_flows[itm]

            # Discount cash flows from future to current exercise date
            discount_ratio = discount_factors[t_idx] / discount_factors[exercise_indicator[itm]]
            discounted_continuation = cash_flows_itm / discount_ratio

            # Generate basis functions
            basis_matrix = basis_fn(S_itm)

            # Least-squares regression: continuation = basis @ coefficients
            # Solve: basis.T @ basis @ coef = basis.T @ continuation
            coefficients, _, _, _ = jnp.linalg.lstsq(
                basis_matrix,
                discounted_continuation,
                rcond=None
            )

            # Estimated continuation value
            continuation_value = basis_matrix @ coefficients

            # Exercise decision: exercise if immediate value > continuation value
            exercise_now = immediate_exercise[itm] > continuation_value

            # Update cash flows and exercise times for paths that exercise now
            # Create full update array
            new_cash_flows = jnp.where(
                itm,
                jnp.where(
                    exercise_now[jnp.cumsum(itm) - 1],  # Expand exercise_now to full size
                    immediate_exercise * discount_factors[t_idx],
                    cash_flows
                ),
                cash_flows
            )

            new_exercise_indicator = jnp.where(
                itm & exercise_now[jnp.cumsum(itm) - 1],
                t_idx,
                exercise_indicator
            )

            cash_flows = new_cash_flows
            exercise_indicator = new_exercise_indicator

    # Option price is the mean of discounted cash flows
    option_price = float(jnp.mean(cash_flows))

    # Convert exercise indicators to actual times
    optimal_exercise_times = exercise_times[exercise_indicator]

    return option_price, optimal_exercise_times, cash_flows


def longstaff_schwartz_bermudan(
    key: jax.random.KeyArray,
    S0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    exercise_schedule: Array,
    option_type: str = "put",
    config: Optional[LSMConfig] = None
) -> dict:
    """Price a Bermudan option using Longstaff-Schwartz algorithm.

    A Bermudan option can be exercised at specific dates (unlike American
    which can be exercised anytime).

    Args:
        key: JAX random key
        S0: Initial stock price
        K: Strike price
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        T: Time to maturity
        exercise_schedule: Array of exercise dates (in years)
        option_type: "call" or "put"
        config: LSM configuration (uses defaults if None)

    Returns:
        Dictionary with pricing results and diagnostics

    Example:
        >>> key = jax.random.PRNGKey(42)
        >>> # Quarterly exercise over 2 years
        >>> exercise_dates = jnp.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
        >>> result = longstaff_schwartz_bermudan(
        ...     key, S0=100, K=100, r=0.05, q=0.02, sigma=0.3,
        ...     T=2.0, exercise_schedule=exercise_dates, option_type="put"
        ... )
        >>> print(f"Bermudan put value: {result['price']:.4f}")
    """
    if config is None:
        config = LSMConfig()

    # Generate stock price paths
    n_paths = config.paths
    n_steps = len(exercise_schedule)

    dt = jnp.diff(jnp.concatenate([jnp.array([0.0]), exercise_schedule]))

    # Simulate GBM paths
    dW = jax.random.normal(key, (n_paths, n_steps))
    paths = jnp.zeros((n_paths, n_steps))

    S = jnp.ones(n_paths) * S0
    for i in range(n_steps):
        S = S * jnp.exp((r - q - 0.5 * sigma**2) * dt[i] + sigma * jnp.sqrt(dt[i]) * dW[:, i])
        paths = paths.at[:, i].set(S)

    # Define payoff function
    if option_type == "put":
        payoff_fn = lambda S: jnp.maximum(K - S, 0.0)
    elif option_type == "call":
        payoff_fn = lambda S: jnp.maximum(S - K, 0.0)
    else:
        raise ValueError(f"Unknown option type: {option_type}")

    # Discount factors for each exercise date
    discount_factors = jnp.exp(-r * exercise_schedule)

    # Run Longstaff-Schwartz algorithm
    price, exercise_times, cash_flows = longstaff_schwartz_american(
        paths,
        payoff_fn,
        exercise_schedule,
        discount_factors,
        config
    )

    # Compute diagnostics
    exercised = exercise_times < T
    avg_exercise_time = float(jnp.mean(exercise_times[exercised])) if jnp.any(exercised) else T
    exercise_rate = float(jnp.mean(exercised))

    return {
        "price": price,
        "paths": paths,
        "exercise_times": exercise_times,
        "cash_flows": cash_flows,
        "avg_exercise_time": avg_exercise_time,
        "exercise_rate": exercise_rate,
        "std_error": float(jnp.std(cash_flows) / jnp.sqrt(n_paths)),
    }


def price_american_put_lsm(
    key: jax.random.KeyArray,
    S0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    n_paths: int = 10000,
    n_steps: int = 50,
    polynomial_degree: int = 3
) -> dict:
    """Convenience function for pricing American put options.

    Args:
        key: JAX random key
        S0: Initial stock price
        K: Strike price
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        T: Time to maturity
        n_paths: Number of Monte Carlo paths
        n_steps: Number of time steps
        polynomial_degree: Degree of polynomial basis

    Returns:
        Dictionary with price and diagnostics
    """
    exercise_schedule = jnp.linspace(0, T, n_steps + 1)[1:]  # Exclude t=0

    config = LSMConfig(
        paths=n_paths,
        polynomial_degree=polynomial_degree,
        discount_rate=r,
        basis_type="laguerre"
    )

    return longstaff_schwartz_bermudan(
        key, S0, K, r, q, sigma, T,
        exercise_schedule,
        option_type="put",
        config=config
    )
