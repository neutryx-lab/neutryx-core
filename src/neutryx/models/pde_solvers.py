"""PDE solvers for option pricing using finite difference methods.

This module implements 1D and 2D PDE solvers for pricing derivatives.
"""
from __future__ import annotations

from typing import Callable, Optional

import jax.numpy as jnp
import jax
from jax import Array


def pde_1d_solve(
    S_min: float,
    S_max: float,
    T: float,
    r: float,
    sigma: float,
    payoff_fn: Callable[[Array], Array],
    N_space: int = 100,
    N_time: int = 100,
    q: float = 0.0,
    boundary_condition: str = "dirichlet",
    theta: float = 0.5,
) -> tuple[Array, Array, Array]:
    """1D PDE solver using finite difference method.

    Solves the Black-Scholes PDE:
    dV/dt + 0.5*sigma^2*S^2*d2V/dS2 + (r-q)*S*dV/dS - r*V = 0

    Parameters
    ----------
    S_min : float
        Minimum spot price for grid
    S_max : float
        Maximum spot price for grid
    T : float
        Time to maturity
    r : float
        Risk-free rate
    sigma : float
        Volatility
    payoff_fn : Callable
        Terminal payoff function taking spot array
    N_space : int
        Number of spatial grid points
    N_time : int
        Number of time steps
    q : float
        Dividend yield
    boundary_condition : str
        Type of boundary condition ("dirichlet" or "neumann")
    theta : float
        Theta parameter for implicit scheme (0.5 = Crank-Nicolson, 1.0 = fully implicit)

    Returns
    -------
    tuple[Array, Array, Array]
        (spot_grid, time_grid, value_grid)
    """
    # Create grids
    S = jnp.linspace(S_min, S_max, N_space)
    dt = T / N_time
    dS = (S_max - S_min) / (N_space - 1)

    # Initialize value grid with terminal condition
    V = payoff_fn(S)

    # Build finite difference matrices
    # Using central differences for space, theta-scheme for time
    a = 0.5 * sigma**2 * S**2 / dS**2 - 0.5 * (r - q) * S / dS
    b = -sigma**2 * S**2 / dS**2 - r
    c = 0.5 * sigma**2 * S**2 / dS**2 + 0.5 * (r - q) * S / dS

    # Build tridiagonal matrices
    # Explicit part: (1 + (1-theta)*dt*L)
    # Implicit part: (1 - theta*dt*L)
    diag_exp = 1.0 + (1.0 - theta) * dt * b[1:-1]
    lower_exp = (1.0 - theta) * dt * a[2:]
    upper_exp = (1.0 - theta) * dt * c[:-2]

    diag_imp = 1.0 - theta * dt * b[1:-1]
    lower_imp = -theta * dt * a[2:]
    upper_imp = -theta * dt * c[:-2]

    # Time stepping (backward in time)
    for _ in range(N_time):
        # Apply explicit operator to interior points
        V_interior = V[1:-1]
        rhs = (
            diag_exp * V_interior
            + lower_exp * V[2:]
            + upper_exp * V[:-2]
        )

        # Solve implicit system: A * V_new = rhs
        # Using Thomas algorithm for tridiagonal system
        V_new = thomas_algorithm(lower_imp, diag_imp, upper_imp, rhs)

        # Update V with boundary conditions
        V = V.at[1:-1].set(V_new)

        # Apply boundary conditions
        if boundary_condition == "dirichlet":
            # V(0, t) = 0, V(S_max, t) = S_max - K*exp(-r*t) for call
            pass  # Keep boundaries as initialized
        elif boundary_condition == "neumann":
            # dV/dS = 0 at boundaries
            V = V.at[0].set(V[1])
            V = V.at[-1].set(V[-2])

    time_grid = jnp.linspace(0, T, N_time + 1)
    return S, time_grid, V


def thomas_algorithm(
    lower: Array, diag: Array, upper: Array, rhs: Array
) -> Array:
    """Solve tridiagonal system using Thomas algorithm.

    Parameters
    ----------
    lower : Array
        Lower diagonal (length n-1)
    diag : Array
        Main diagonal (length n)
    upper : Array
        Upper diagonal (length n-1)
    rhs : Array
        Right hand side (length n)

    Returns
    -------
    Array
        Solution vector
    """
    n = len(diag)

    # Forward sweep
    c_prime = jnp.zeros(n - 1)
    d_prime = jnp.zeros(n)

    c_prime = c_prime.at[0].set(upper[0] / diag[0])
    d_prime = d_prime.at[0].set(rhs[0] / diag[0])

    for i in range(1, n - 1):
        denom = diag[i] - lower[i - 1] * c_prime[i - 1]
        c_prime = c_prime.at[i].set(upper[i] / denom)
        d_prime = d_prime.at[i].set((rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom)

    # Last row
    denom = diag[n - 1] - lower[n - 2] * c_prime[n - 2]
    d_prime = d_prime.at[n - 1].set((rhs[n - 1] - lower[n - 2] * d_prime[n - 2]) / denom)

    # Back substitution
    x = jnp.zeros(n)
    x = x.at[n - 1].set(d_prime[n - 1])

    for i in range(n - 2, -1, -1):
        x = x.at[i].set(d_prime[i] - c_prime[i] * x[i + 1])

    return x


def pde_2d_solve(
    S1_min: float,
    S1_max: float,
    S2_min: float,
    S2_max: float,
    T: float,
    r: float,
    sigma1: float,
    sigma2: float,
    rho: float,
    payoff_fn: Callable[[Array, Array], Array],
    N_S1: int = 50,
    N_S2: int = 50,
    N_time: int = 50,
    q1: float = 0.0,
    q2: float = 0.0,
) -> tuple[Array, Array, Array, Array]:
    """2D PDE solver for two-asset options.

    Solves 2D Black-Scholes PDE for basket/spread options.

    Parameters
    ----------
    S1_min, S1_max : float
        Price range for first asset
    S2_min, S2_max : float
        Price range for second asset
    T : float
        Time to maturity
    r : float
        Risk-free rate
    sigma1, sigma2 : float
        Volatilities of the two assets
    rho : float
        Correlation between assets
    payoff_fn : Callable
        Terminal payoff function taking (S1_grid, S2_grid)
    N_S1, N_S2 : int
        Number of spatial grid points for each asset
    N_time : int
        Number of time steps
    q1, q2 : float
        Dividend yields

    Returns
    -------
    tuple[Array, Array, Array, Array]
        (S1_grid, S2_grid, time_grid, value_grid)
    """
    # Create spatial grids
    S1 = jnp.linspace(S1_min, S1_max, N_S1)
    S2 = jnp.linspace(S2_min, S2_max, N_S2)
    S1_grid, S2_grid = jnp.meshgrid(S1, S2, indexing="ij")

    # Time grid
    dt = T / N_time
    dS1 = (S1_max - S1_min) / (N_S1 - 1)
    dS2 = (S2_max - S2_min) / (N_S2 - 1)

    # Initialize with terminal payoff
    V = payoff_fn(S1_grid, S2_grid)

    # Simplified ADI (Alternating Direction Implicit) method
    # For now, use explicit method for simplicity
    for _ in range(N_time):
        V_new = jnp.zeros_like(V)

        # Interior points
        for i in range(1, N_S1 - 1):
            for j in range(1, N_S2 - 1):
                # Second derivatives
                d2V_dS1 = (V[i + 1, j] - 2 * V[i, j] + V[i - 1, j]) / dS1**2
                d2V_dS2 = (V[i, j + 1] - 2 * V[i, j] + V[i, j - 1]) / dS2**2

                # Cross derivative
                d2V_dS1dS2 = (
                    V[i + 1, j + 1] - V[i + 1, j - 1] - V[i - 1, j + 1] + V[i - 1, j - 1]
                ) / (4 * dS1 * dS2)

                # First derivatives
                dV_dS1 = (V[i + 1, j] - V[i - 1, j]) / (2 * dS1)
                dV_dS2 = (V[i, j + 1] - V[i, j - 1]) / (2 * dS2)

                # PDE terms
                drift1 = (r - q1) * S1_grid[i, j] * dV_dS1
                drift2 = (r - q2) * S2_grid[i, j] * dV_dS2
                diff1 = 0.5 * sigma1**2 * S1_grid[i, j] ** 2 * d2V_dS1
                diff2 = 0.5 * sigma2**2 * S2_grid[i, j] ** 2 * d2V_dS2
                cross = rho * sigma1 * sigma2 * S1_grid[i, j] * S2_grid[i, j] * d2V_dS1dS2

                # Explicit Euler step
                V_new = V_new.at[i, j].set(
                    V[i, j] + dt * (drift1 + drift2 + diff1 + diff2 + cross - r * V[i, j])
                )

        # Boundary conditions (simple Dirichlet - keep as terminal)
        V_new = V_new.at[0, :].set(V[0, :])
        V_new = V_new.at[-1, :].set(V[-1, :])
        V_new = V_new.at[:, 0].set(V[:, 0])
        V_new = V_new.at[:, -1].set(V[:, -1])

        V = V_new

    time_grid = jnp.linspace(0, T, N_time + 1)
    return S1, S2, time_grid, V


def price_european_pde_1d(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    is_call: bool = True,
    q: float = 0.0,
    N_space: int = 200,
    N_time: int = 200,
) -> float:
    """Price European option using 1D PDE solver.

    Parameters
    ----------
    S0 : float
        Initial spot price
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    sigma : float
        Volatility
    is_call : bool
        True for call, False for put
    q : float
        Dividend yield
    N_space : int
        Number of spatial grid points
    N_time : int
        Number of time steps

    Returns
    -------
    float
        Option price
    """
    # Set grid boundaries
    S_min = 0.0
    S_max = 3.0 * K

    # Terminal payoff
    def payoff(S):
        if is_call:
            return jnp.maximum(S - K, 0.0)
        else:
            return jnp.maximum(K - S, 0.0)

    S_grid, _, V = pde_1d_solve(
        S_min, S_max, T, r, sigma, payoff, N_space, N_time, q
    )

    # Interpolate to get value at S0
    return float(jnp.interp(S0, S_grid, V))


__all__ = [
    "pde_1d_solve",
    "pde_2d_solve",
    "thomas_algorithm",
    "price_european_pde_1d",
]
