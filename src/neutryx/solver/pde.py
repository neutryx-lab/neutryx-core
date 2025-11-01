"""PDE solvers for option pricing.

This module implements various finite difference methods for solving
the Black-Scholes PDE and related PDEs.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PDEGrid:
    """Grid specification for PDE solver.

    Args:
        S_min: Minimum spot price
        S_max: Maximum spot price
        T: Time to maturity
        N_S: Number of spatial grid points
        N_T: Number of time steps
    """
    S_min: float
    S_max: float
    T: float
    N_S: int
    N_T: int

    @property
    def dS(self) -> float:
        """Spatial step size."""
        return (self.S_max - self.S_min) / (self.N_S - 1)

    @property
    def dt(self) -> float:
        """Time step size."""
        return self.T / self.N_T

    @property
    def S_grid(self) -> jnp.ndarray:
        """Spatial grid points."""
        return jnp.linspace(self.S_min, self.S_max, self.N_S)


def crank_nicolson_european(
    grid: PDEGrid,
    payoff_fn: Callable[[jnp.ndarray], jnp.ndarray],
    r: float,
    q: float,
    sigma: float,
    option_type: str = "call"
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Solve Black-Scholes PDE using Crank-Nicolson method for European options.

    The Black-Scholes PDE is:
    ∂V/∂t + (r-q)S ∂V/∂S + 0.5σ²S² ∂²V/∂S² - rV = 0

    Args:
        grid: PDE grid specification
        payoff_fn: Terminal payoff function V(S, T)
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        option_type: "call" or "put"

    Returns:
        Tuple of (option_values, S_grid) where option_values[i, j] is the
        option value at spatial grid point i and time step j
    """
    N_S = grid.N_S
    N_T = grid.N_T
    dt = grid.dt
    dS = grid.dS
    S = grid.S_grid

    # Initialize solution matrix
    V = jnp.zeros((N_S, N_T + 1))

    # Set terminal condition
    V = V.at[:, -1].set(payoff_fn(S))

    # Coefficients for finite difference scheme
    # Second derivative coefficient
    alpha = 0.5 * dt * (sigma ** 2 * S ** 2) / (dS ** 2)
    # First derivative coefficient
    beta = 0.25 * dt * (r - q) * S / dS
    # Discount coefficient
    gamma = -0.5 * r * dt

    # Build tridiagonal matrices for Crank-Nicolson
    # A * V^{n+1} = B * V^n

    # Matrix A (implicit side)
    a_lower = -alpha + beta
    a_diag = 1 + 2 * alpha - gamma
    a_upper = -alpha - beta

    # Matrix B (explicit side)
    b_lower = alpha - beta
    b_diag = 1 - 2 * alpha + gamma
    b_upper = alpha + beta

    # Build full tridiagonal matrices (excluding boundaries)
    # Interior points: indices 1 to N_S-2
    N_interior = N_S - 2

    # Build sparse tridiagonal matrix A
    A_matrix = jnp.zeros((N_interior, N_interior))

    for i in range(N_interior):
        idx = i + 1  # Index in full grid
        # Diagonal
        A_matrix = A_matrix.at[i, i].set(a_diag[idx])
        # Lower diagonal
        if i > 0:
            A_matrix = A_matrix.at[i, i - 1].set(a_lower[idx])
        # Upper diagonal
        if i < N_interior - 1:
            A_matrix = A_matrix.at[i, i + 1].set(a_upper[idx])

    # Build matrix B
    B_matrix = jnp.zeros((N_interior, N_interior))

    for i in range(N_interior):
        idx = i + 1
        # Diagonal
        B_matrix = B_matrix.at[i, i].set(b_diag[idx])
        # Lower diagonal
        if i > 0:
            B_matrix = B_matrix.at[i, i - 1].set(b_lower[idx])
        # Upper diagonal
        if i < N_interior - 1:
            B_matrix = B_matrix.at[i, i + 1].set(b_upper[idx])

    # Boundary conditions
    # For a call: V(0, t) = 0, V(S_max, t) = S_max - K*exp(-r*(T-t))
    # For a put: V(0, t) = K*exp(-r*(T-t)), V(S_max, t) = 0
    # Here we use simple Dirichlet boundaries based on option type

    # Time-stepping (backward in time)
    for j in range(N_T - 1, -1, -1):
        # Current time
        t = j * dt
        time_to_maturity = grid.T - t

        # Set boundary conditions
        if option_type == "call":
            V = V.at[0, j].set(0.0)
            # Approximate call boundary at S_max
            V = V.at[-1, j].set(jnp.maximum(S[-1] - S[-1] * jnp.exp(-r * time_to_maturity), 0.0))
        else:  # put
            # Approximate put boundary at S_min
            V = V.at[0, j].set(S[0] * jnp.exp(-r * time_to_maturity))
            V = V.at[-1, j].set(0.0)

        # Extract interior values
        V_interior = V[1:-1, j + 1]

        # Right-hand side
        rhs = B_matrix @ V_interior

        # Adjust for boundaries
        boundary_correction = jnp.zeros(N_interior)
        boundary_correction = boundary_correction.at[0].add(
            (a_lower[1] - b_lower[1]) * V[0, j]
        )
        boundary_correction = boundary_correction.at[-1].add(
            (a_upper[-2] - b_upper[-2]) * V[-1, j]
        )

        rhs = rhs + boundary_correction

        # Solve linear system A * V^{n} = rhs
        V_new_interior = jnp.linalg.solve(A_matrix, rhs)

        # Update interior points
        V = V.at[1:-1, j].set(V_new_interior)

    return V, S


def crank_nicolson_american_put(
    grid: PDEGrid,
    K: float,
    r: float,
    q: float,
    sigma: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Solve American put option using Crank-Nicolson with early exercise.

    Uses operator splitting to handle the American exercise constraint.

    Args:
        grid: PDE grid specification
        K: Strike price
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility

    Returns:
        Tuple of (option_values, S_grid)
    """
    N_S = grid.N_S
    N_T = grid.N_T
    dt = grid.dt
    dS = grid.dS
    S = grid.S_grid

    # Initialize solution matrix
    V = jnp.zeros((N_S, N_T + 1))

    # Set terminal condition (put payoff)
    V = V.at[:, -1].set(jnp.maximum(K - S, 0.0))

    # Intrinsic value for American option
    intrinsic = jnp.maximum(K - S, 0.0)

    # Coefficients
    alpha = 0.5 * dt * (sigma ** 2 * S ** 2) / (dS ** 2)
    beta = 0.25 * dt * (r - q) * S / dS
    gamma = -0.5 * r * dt

    # Build matrices (same as European case)
    a_lower = -alpha + beta
    a_diag = 1 + 2 * alpha - gamma
    a_upper = -alpha - beta

    b_lower = alpha - beta
    b_diag = 1 - 2 * alpha + gamma
    b_upper = alpha + beta

    N_interior = N_S - 2

    # Build matrix A
    A_matrix = jnp.zeros((N_interior, N_interior))
    for i in range(N_interior):
        idx = i + 1
        A_matrix = A_matrix.at[i, i].set(a_diag[idx])
        if i > 0:
            A_matrix = A_matrix.at[i, i - 1].set(a_lower[idx])
        if i < N_interior - 1:
            A_matrix = A_matrix.at[i, i + 1].set(a_upper[idx])

    # Build matrix B
    B_matrix = jnp.zeros((N_interior, N_interior))
    for i in range(N_interior):
        idx = i + 1
        B_matrix = B_matrix.at[i, i].set(b_diag[idx])
        if i > 0:
            B_matrix = B_matrix.at[i, i - 1].set(b_lower[idx])
        if i < N_interior - 1:
            B_matrix = B_matrix.at[i, i + 1].set(b_upper[idx])

    # Time-stepping with early exercise constraint
    for j in range(N_T - 1, -1, -1):
        t = j * dt
        time_to_maturity = grid.T - t

        # Boundary conditions for put
        V = V.at[0, j].set(K * jnp.exp(-r * time_to_maturity))
        V = V.at[-1, j].set(0.0)

        # Solve European step
        V_interior = V[1:-1, j + 1]
        rhs = B_matrix @ V_interior

        # Boundary correction
        boundary_correction = jnp.zeros(N_interior)
        boundary_correction = boundary_correction.at[0].add(
            (a_lower[1] - b_lower[1]) * V[0, j]
        )
        boundary_correction = boundary_correction.at[-1].add(
            (a_upper[-2] - b_upper[-2]) * V[-1, j]
        )

        rhs = rhs + boundary_correction

        V_new_interior = jnp.linalg.solve(A_matrix, rhs)

        # Apply early exercise constraint (American feature)
        V_new_interior = jnp.maximum(V_new_interior, intrinsic[1:-1])

        V = V.at[1:-1, j].set(V_new_interior)

    return V, S


def price_european_option_pde(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str = "call",
    N_S: int = 200,
    N_T: int = 100
) -> float:
    """Price a European option using PDE method (Crank-Nicolson).

    Args:
        S0: Initial spot price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        option_type: "call" or "put"
        N_S: Number of spatial grid points
        N_T: Number of time steps

    Returns:
        Option price
    """
    # Set grid boundaries
    S_min = 0.0
    S_max = max(3 * K, 3 * S0)

    grid = PDEGrid(S_min=S_min, S_max=S_max, T=T, N_S=N_S, N_T=N_T)

    # Define payoff
    if option_type == "call":
        payoff_fn = lambda S: jnp.maximum(S - K, 0.0)
    else:
        payoff_fn = lambda S: jnp.maximum(K - S, 0.0)

    # Solve PDE
    V, S_grid = crank_nicolson_european(grid, payoff_fn, r, q, sigma, option_type)

    # Interpolate to get value at S0
    price = jnp.interp(S0, S_grid, V[:, 0])

    return price


def price_american_put_pde(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    N_S: int = 200,
    N_T: int = 100
) -> float:
    """Price an American put option using PDE method.

    Args:
        S0: Initial spot price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        N_S: Number of spatial grid points
        N_T: Number of time steps

    Returns:
        American put option price
    """
    # Set grid boundaries
    S_min = 0.0
    S_max = max(3 * K, 3 * S0)

    grid = PDEGrid(S_min=S_min, S_max=S_max, T=T, N_S=N_S, N_T=N_T)

    # Solve PDE with American constraint
    V, S_grid = crank_nicolson_american_put(grid, K, r, q, sigma)

    # Interpolate to get value at S0
    price = jnp.interp(S0, S_grid, V[:, 0])

    return price
