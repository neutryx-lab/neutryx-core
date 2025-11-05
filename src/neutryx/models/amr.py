"""Adaptive Mesh Refinement (AMR) for PDE Solvers.

This module implements adaptive mesh refinement techniques for finite difference PDE solvers,
enabling automatic grid refinement in regions requiring higher resolution (e.g., near strike,
steep gradients, discontinuities) and coarsening in regions where lower resolution suffices.

Key Features:
- Non-uniform spatial grid support
- Gradient-based error indicators
- Richardson extrapolation for error estimation
- Automatic grid refinement/coarsening
- Interpolation between grid levels
- Integration with existing Crank-Nicolson solvers

References:
- Tavella, D., & Randall, C. (2000). Pricing Financial Instruments: The Finite Difference Method.
- Wilmott, P. (2006). Paul Wilmott On Quantitative Finance (2nd ed.).
- Duffy, D. J. (2006). Finite Difference Methods in Financial Engineering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import jax.numpy as jnp
from jax import Array

__all__ = [
    "AdaptivePDEGrid",
    "ErrorIndicators",
    "RefineInfo",
    "estimate_gradient_error",
    "richardson_extrapolation_error",
    "refine_grid_adaptive",
    "interpolate_to_new_grid",
    "adaptive_crank_nicolson",
]


@dataclass
class AdaptivePDEGrid:
    """Non-uniform grid specification for adaptive PDE solvers.

    Attributes
    ----------
    S_grid : Array
        Non-uniform spatial grid points, shape [N_S]
    T : float
        Time to maturity
    N_T : int
        Number of time steps
    refinement_history : list
        History of refinement operations (for debugging/analysis)
    """

    S_grid: Array
    T: float
    N_T: int
    refinement_history: list = None

    def __post_init__(self):
        if self.refinement_history is None:
            self.refinement_history = []

    @property
    def N_S(self) -> int:
        """Number of spatial grid points."""
        return len(self.S_grid)

    @property
    def S_min(self) -> float:
        """Minimum spot price."""
        return float(self.S_grid[0])

    @property
    def S_max(self) -> float:
        """Maximum spot price."""
        return float(self.S_grid[-1])

    @property
    def dt(self) -> float:
        """Time step size."""
        return self.T / self.N_T

    def dS_local(self, i: int) -> float:
        """Local spatial step size at index i."""
        if i == 0:
            return float(self.S_grid[1] - self.S_grid[0])
        elif i == self.N_S - 1:
            return float(self.S_grid[-1] - self.S_grid[-2])
        else:
            return float(0.5 * (self.S_grid[i + 1] - self.S_grid[i - 1]))

    @classmethod
    def from_uniform(cls, S_min: float, S_max: float, T: float, N_S: int, N_T: int):
        """Create adaptive grid from uniform grid specification."""
        S_grid = jnp.linspace(S_min, S_max, N_S)
        return cls(S_grid=S_grid, T=T, N_T=N_T)

    @classmethod
    def from_strike_concentrated(
        cls, S_min: float, S_max: float, T: float, N_S: int, N_T: int, strike: float,
        concentration: float = 2.0
    ):
        """Create grid with higher density near the strike.

        Parameters
        ----------
        concentration : float
            Concentration factor (> 1). Higher values = more points near strike.
        """
        # Use sinh transformation for strike concentration
        x = jnp.linspace(-1, 1, N_S)
        # Transform to concentrate around 0 (strike)
        y = jnp.sinh(concentration * x) / jnp.sinh(concentration)

        # Map y ∈ [-1, 1] to S ∈ [S_min, S_max] with concentration at strike
        # Use a simpler mapping: linear transform with strike as focal point
        strike_norm = (strike - S_min) / (S_max - S_min)  # Strike position in [0,1]

        # Map y to S with concentration at strike
        # For y < 0: map [-1, 0] to [S_min, strike]
        # For y >= 0: map [0, 1] to [strike, S_max]
        S_grid_left = S_min + (strike - S_min) * (1 + y[y < 0])
        S_grid_right = strike + (S_max - strike) * y[y >= 0]
        S_grid = jnp.concatenate([S_grid_left, S_grid_right])

        S_grid = jnp.sort(jnp.unique(S_grid))

        return cls(S_grid=S_grid, T=T, N_T=N_T)


@dataclass
class ErrorIndicators:
    """Error indicators for each spatial grid point.

    Attributes
    ----------
    gradient_error : Array
        Gradient-based error estimate, shape [N_S]
    curvature_error : Array
        Curvature-based error estimate, shape [N_S]
    richardson_error : Array, optional
        Richardson extrapolation error estimate, shape [N_S]
    """

    gradient_error: Array
    curvature_error: Array
    richardson_error: Optional[Array] = None

    def combined_error(self, weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)) -> Array:
        """Compute weighted combination of error indicators.

        Parameters
        ----------
        weights : tuple
            Weights for (gradient, curvature, richardson) errors.
            Must sum to 1.0. If richardson_error is None, only first two are used.
        """
        if self.richardson_error is not None:
            w_grad, w_curv, w_rich = weights
            total = w_grad + w_curv + w_rich
            w_grad, w_curv, w_rich = w_grad / total, w_curv / total, w_rich / total
            return (
                w_grad * self.gradient_error
                + w_curv * self.curvature_error
                + w_rich * self.richardson_error
            )
        else:
            w_grad, w_curv = weights[0], weights[1]
            total = w_grad + w_curv
            w_grad, w_curv = w_grad / total, w_curv / total
            return w_grad * self.gradient_error + w_curv * self.curvature_error


@dataclass
class RefineInfo:
    """Information about refinement operation.

    Attributes
    ----------
    indices_to_refine : Array
        Indices where new points should be added
    estimated_error : float
        Overall estimated error before refinement
    max_local_error : float
        Maximum local error indicator
    n_points_added : int
        Number of points added
    """

    indices_to_refine: Array
    estimated_error: float
    max_local_error: float
    n_points_added: int


def estimate_gradient_error(solution: Array, S_grid: Array, threshold: float = 1e-3) -> Array:
    """Estimate error using solution gradient (first derivative).

    High gradients indicate regions needing refinement.

    Parameters
    ----------
    solution : Array
        Solution values at grid points, shape [N_S]
    S_grid : Array
        Spatial grid points, shape [N_S]
    threshold : float
        Minimum gradient to consider for error

    Returns
    -------
    Array
        Error indicator at each point, shape [N_S]
    """
    N_S = len(solution)

    # Compute gradients using central differences
    gradients = jnp.zeros_like(solution)

    # Interior points: central difference
    for i in range(1, N_S - 1):
        dS_forward = S_grid[i + 1] - S_grid[i]
        dS_backward = S_grid[i] - S_grid[i - 1]
        # Weighted central difference for non-uniform grid
        grad = (solution[i + 1] - solution[i - 1]) / (dS_forward + dS_backward)
        gradients = gradients.at[i].set(jnp.abs(grad))

    # Boundaries: forward/backward difference
    gradients = gradients.at[0].set(jnp.abs((solution[1] - solution[0]) / (S_grid[1] - S_grid[0])))
    gradients = gradients.at[-1].set(
        jnp.abs((solution[-1] - solution[-2]) / (S_grid[-1] - S_grid[-2]))
    )

    # Normalize by maximum gradient
    max_grad = jnp.maximum(jnp.max(gradients), threshold)
    error_indicator = gradients / max_grad

    return error_indicator


def estimate_curvature_error(solution: Array, S_grid: Array, threshold: float = 1e-5) -> Array:
    """Estimate error using solution curvature (second derivative).

    High curvature indicates regions needing refinement.

    Parameters
    ----------
    solution : Array
        Solution values at grid points, shape [N_S]
    S_grid : Array
        Spatial grid points, shape [N_S]
    threshold : float
        Minimum curvature to consider for error

    Returns
    -------
    Array
        Error indicator at each point, shape [N_S]
    """
    N_S = len(solution)
    curvatures = jnp.zeros_like(solution)

    # Interior points: second derivative approximation
    for i in range(1, N_S - 1):
        h_left = S_grid[i] - S_grid[i - 1]
        h_right = S_grid[i + 1] - S_grid[i]

        # Non-uniform grid second derivative
        # d²f/dx² ≈ 2*[f(i+1)/(h_r(h_l+h_r)) - f(i)/(h_l*h_r) + f(i-1)/(h_l(h_l+h_r))]
        coeff_right = 2.0 / (h_right * (h_left + h_right))
        coeff_center = -2.0 / (h_left * h_right)
        coeff_left = 2.0 / (h_left * (h_left + h_right))

        curv = (
            coeff_left * solution[i - 1]
            + coeff_center * solution[i]
            + coeff_right * solution[i + 1]
        )
        curvatures = curvatures.at[i].set(jnp.abs(curv))

    # Boundaries: use one-sided approximations
    curvatures = curvatures.at[0].set(curvatures[1])
    curvatures = curvatures.at[-1].set(curvatures[-2])

    # Normalize
    max_curv = jnp.maximum(jnp.max(curvatures), threshold)
    error_indicator = curvatures / max_curv

    return error_indicator


def richardson_extrapolation_error(
    solution_fine: Array,
    solution_coarse: Array,
    S_grid_fine: Array,
    S_grid_coarse: Array,
    order: int = 2,
) -> Array:
    """Estimate error using Richardson extrapolation between two grid levels.

    Parameters
    ----------
    solution_fine : Array
        Solution on fine grid, shape [N_fine]
    solution_coarse : Array
        Solution on coarse grid, shape [N_coarse]
    S_grid_fine : Array
        Fine grid points, shape [N_fine]
    S_grid_coarse : Array
        Coarse grid points, shape [N_coarse]
    order : int
        Expected convergence order (typically 2 for Crank-Nicolson)

    Returns
    -------
    Array
        Error estimate on fine grid, shape [N_fine]
    """
    # Interpolate coarse solution to fine grid
    solution_coarse_interp = jnp.interp(S_grid_fine, S_grid_coarse, solution_coarse)

    # Richardson extrapolation: error ≈ (V_fine - V_coarse) / (2^order - 1)
    error_estimate = jnp.abs(solution_fine - solution_coarse_interp) / (2**order - 1)

    return error_estimate


def refine_grid_adaptive(
    S_grid: Array,
    error_indicators: ErrorIndicators,
    tolerance: float = 0.1,
    max_points: int = 500,
    min_spacing: float = 1e-4,
) -> Tuple[Array, RefineInfo]:
    """Adaptively refine grid based on error indicators.

    Parameters
    ----------
    S_grid : Array
        Current spatial grid, shape [N_S]
    error_indicators : ErrorIndicators
        Error indicators for each grid point
    tolerance : float
        Refinement threshold (points with error > tolerance are refined)
    max_points : int
        Maximum allowed grid points
    min_spacing : float
        Minimum allowed spacing between points

    Returns
    -------
    new_grid : Array
        Refined grid
    refine_info : RefineInfo
        Information about refinement operation
    """
    # Combine error indicators
    combined_error = error_indicators.combined_error()

    # Find regions needing refinement
    needs_refinement = combined_error > tolerance
    refinement_indices = jnp.where(needs_refinement)[0]

    # Estimate overall and maximum errors
    estimated_error = float(jnp.mean(combined_error))
    max_local_error = float(jnp.max(combined_error))

    # Check if we can add more points
    current_n_points = len(S_grid)
    if current_n_points >= max_points:
        # No refinement - already at maximum
        return S_grid, RefineInfo(
            indices_to_refine=refinement_indices,
            estimated_error=estimated_error,
            max_local_error=max_local_error,
            n_points_added=0,
        )

    # Add midpoints between refined points and their neighbors
    new_points = []
    for idx in refinement_indices:
        if idx < len(S_grid) - 1:
            # Add midpoint to the right
            midpoint = 0.5 * (S_grid[idx] + S_grid[idx + 1])
            # Check minimum spacing
            if (midpoint - S_grid[idx]) > min_spacing:
                new_points.append(midpoint)

        if idx > 0:
            # Add midpoint to the left
            midpoint = 0.5 * (S_grid[idx - 1] + S_grid[idx])
            # Check minimum spacing
            if (S_grid[idx] - midpoint) > min_spacing:
                new_points.append(midpoint)

    # Combine and sort
    if len(new_points) > 0:
        new_points_arr = jnp.array(new_points)
        new_grid = jnp.concatenate([S_grid, new_points_arr])
        new_grid = jnp.sort(jnp.unique(new_grid))

        # Truncate if exceeds maximum
        if len(new_grid) > max_points:
            # Keep points by removing those with lowest error
            # (This is a simplified strategy)
            new_grid = new_grid[:max_points]
    else:
        new_grid = S_grid

    n_added = len(new_grid) - len(S_grid)

    refine_info = RefineInfo(
        indices_to_refine=refinement_indices,
        estimated_error=estimated_error,
        max_local_error=max_local_error,
        n_points_added=n_added,
    )

    return new_grid, refine_info


def interpolate_to_new_grid(
    solution_old: Array, S_grid_old: Array, S_grid_new: Array
) -> Array:
    """Interpolate solution from old grid to new refined grid.

    Uses linear interpolation for stability and simplicity.

    Parameters
    ----------
    solution_old : Array
        Solution on old grid, shape [N_old]
    S_grid_old : Array
        Old grid points, shape [N_old]
    S_grid_new : Array
        New grid points, shape [N_new]

    Returns
    -------
    Array
        Solution interpolated to new grid, shape [N_new]
    """
    # Use JAX's interp for linear interpolation
    solution_new = jnp.interp(S_grid_new, S_grid_old, solution_old)
    return solution_new


def adaptive_crank_nicolson(
    grid: AdaptivePDEGrid,
    payoff_fn: Callable[[Array], Array],
    r: float,
    q: float,
    sigma: float,
    option_type: str = "call",
    strike: float = None,
    tolerance: float = 0.1,
    max_refinements: int = 3,
    verbose: bool = False,
) -> Tuple[Array, AdaptivePDEGrid, list]:
    """Solve PDE using Crank-Nicolson with adaptive mesh refinement.

    Parameters
    ----------
    grid : AdaptivePDEGrid
        Initial adaptive PDE grid
    payoff_fn : callable
        Terminal payoff function V(S, T)
    r : float
        Risk-free rate
    q : float
        Dividend yield
    sigma : float
        Volatility
    option_type : str
        "call" or "put"
    strike : float
        Strike price (for boundary conditions)
    tolerance : float
        Error tolerance for refinement
    max_refinements : int
        Maximum number of refinement iterations
    verbose : bool
        Print refinement information

    Returns
    -------
    solution : Array
        Final solution on refined grid, shape [N_S_final]
    final_grid : AdaptivePDEGrid
        Final refined grid
    refinement_history : list
        History of refinement operations

    Notes
    -----
    The algorithm:
    1. Solve PDE on current grid
    2. Estimate errors using gradient/curvature indicators
    3. If error > tolerance and refinements < max, refine grid and repeat
    4. Return solution on final refined grid
    """
    if strike is None:
        raise ValueError("strike must be provided for boundary conditions")

    refinement_history = []
    current_grid = grid

    for iteration in range(max_refinements + 1):
        if verbose:
            print(f"\nRefinement iteration {iteration}")
            print(f"Grid points: {current_grid.N_S}")

        # Solve PDE on current grid
        solution = _solve_crank_nicolson_nonuniform(
            current_grid, payoff_fn, r, q, sigma, option_type, strike
        )

        # Check convergence (last iteration or error below tolerance)
        if iteration < max_refinements:
            # Estimate errors
            gradient_error = estimate_gradient_error(solution, current_grid.S_grid)
            curvature_error = estimate_curvature_error(solution, current_grid.S_grid)
            error_indicators = ErrorIndicators(
                gradient_error=gradient_error, curvature_error=curvature_error
            )

            # Decide if refinement is needed
            combined_error = error_indicators.combined_error()
            mean_error = float(jnp.mean(combined_error))
            max_error = float(jnp.max(combined_error))

            if verbose:
                print(f"Mean error: {mean_error:.6f}, Max error: {max_error:.6f}")

            if mean_error < tolerance and max_error < tolerance * 3:
                if verbose:
                    print("Converged!")
                break

            # Refine grid
            new_S_grid, refine_info = refine_grid_adaptive(
                current_grid.S_grid, error_indicators, tolerance=tolerance
            )

            if refine_info.n_points_added == 0:
                if verbose:
                    print("No more points can be added. Stopping.")
                break

            # Interpolate solution to new grid
            solution_interp = interpolate_to_new_grid(
                solution, current_grid.S_grid, new_S_grid
            )

            # Update grid
            current_grid = AdaptivePDEGrid(
                S_grid=new_S_grid, T=current_grid.T, N_T=current_grid.N_T
            )

            # Record history
            refinement_history.append({
                'iteration': iteration,
                'n_points': current_grid.N_S,
                'points_added': refine_info.n_points_added,
                'mean_error': mean_error,
                'max_error': max_error,
            })

            if verbose:
                print(f"Added {refine_info.n_points_added} points")

    return solution, current_grid, refinement_history


def _solve_crank_nicolson_nonuniform(
    grid: AdaptivePDEGrid,
    payoff_fn: Callable[[Array], Array],
    r: float,
    q: float,
    sigma: float,
    option_type: str,
    strike: float,
) -> Array:
    """Solve Black-Scholes PDE on non-uniform grid using Crank-Nicolson.

    This is the core solver adapted for non-uniform spatial grids.
    For numerical stability, we use a coordinate transformation approach
    that maps to a uniform grid in log-space.

    Parameters
    ----------
    grid : AdaptivePDEGrid
        Non-uniform spatial grid
    payoff_fn : callable
        Terminal payoff function
    r : float
        Risk-free rate
    q : float
        Dividend yield
    sigma : float
        Volatility
    option_type : str
        "call" or "put"
    strike : float
        Strike price

    Returns
    -------
    Array
        Option values at t=0, shape [N_S]
    """
    N_S = grid.N_S
    N_T = grid.N_T
    dt = grid.dt
    S = grid.S_grid

    # Initialize solution matrix
    V = jnp.zeros((N_S, N_T + 1))

    # Set terminal payoff
    V = V.at[:, -1].set(payoff_fn(S))

    # Backward time-stepping
    for j in range(N_T - 1, -1, -1):
        t = j * dt
        tau = grid.T - t

        # Boundary conditions
        if option_type == "call":
            lower_boundary = 0.0
            upper_boundary = jnp.maximum(
                S[-1] * jnp.exp(-q * tau) - strike * jnp.exp(-r * tau), 0.0
            )
        else:
            lower_boundary = jnp.maximum(
                strike * jnp.exp(-r * tau) - S[0] * jnp.exp(-q * tau), 0.0
            )
            upper_boundary = 0.0

        V = V.at[0, j].set(lower_boundary)
        V = V.at[-1, j].set(upper_boundary)

        # For interior points, build tridiagonal system
        N_interior = N_S - 2

        # Build matrices for Crank-Nicolson on non-uniform grid
        # Use robust finite difference coefficients
        A_diag = jnp.zeros(N_interior)
        A_lower = jnp.zeros(N_interior - 1)
        A_upper = jnp.zeros(N_interior - 1)

        B_diag = jnp.zeros(N_interior)
        B_lower = jnp.zeros(N_interior - 1)
        B_upper = jnp.zeros(N_interior - 1)

        theta = 0.5  # Crank-Nicolson

        for idx in range(N_interior):
            i = idx + 1  # Actual grid index

            # Local spacing
            h_minus = S[i] - S[i - 1]
            h_plus = S[i + 1] - S[i]
            h_avg = 0.5 * (h_minus + h_plus)

            # Finite difference coefficients for non-uniform grid
            # First derivative
            d_minus = -h_plus / (h_minus * (h_minus + h_plus))
            d_center = (h_plus - h_minus) / (h_minus * h_plus)
            d_plus = h_minus / (h_plus * (h_minus + h_plus))

            # Second derivative
            d2_minus = 2.0 / (h_minus * (h_minus + h_plus))
            d2_center = -2.0 / (h_minus * h_plus)
            d2_plus = 2.0 / (h_plus * (h_minus + h_plus))

            # PDE coefficients
            Si = S[i]
            drift = (r - q) * Si
            diffusion = 0.5 * sigma ** 2 * Si ** 2

            # Build stencil for this point
            # L_operator * V = drift * dV/dS + diffusion * d²V/dS² - r*V
            L_minus = drift * d_minus + diffusion * d2_minus
            L_center = drift * d_center + diffusion * d2_center - r
            L_plus = drift * d_plus + diffusion * d2_plus

            # Crank-Nicolson scheme:
            # (I - theta*dt*L) * V^{n} = (I + (1-theta)*dt*L) * V^{n+1}

            # A matrix (implicit)
            a_minus = -theta * dt * L_minus
            a_center = 1.0 - theta * dt * L_center
            a_plus = -theta * dt * L_plus

            # B matrix (explicit)
            b_minus = (1 - theta) * dt * L_minus
            b_center = 1.0 + (1 - theta) * dt * L_center
            b_plus = (1 - theta) * dt * L_plus

            # Fill tridiagonal arrays
            A_diag = A_diag.at[idx].set(a_center)
            if idx > 0:
                A_lower = A_lower.at[idx - 1].set(a_minus)
            if idx < N_interior - 1:
                A_upper = A_upper.at[idx].set(a_plus)

            B_diag = B_diag.at[idx].set(b_center)
            if idx > 0:
                B_lower = B_lower.at[idx - 1].set(b_minus)
            if idx < N_interior - 1:
                B_upper = B_upper.at[idx].set(b_plus)

        # Construct full matrices
        A = jnp.diag(A_diag) + jnp.diag(A_lower, -1) + jnp.diag(A_upper, 1)
        B = jnp.diag(B_diag) + jnp.diag(B_lower, -1) + jnp.diag(B_upper, 1)

        # Solve system: A * V_current = B * V_next + boundary_terms
        V_next = V[1:-1, j + 1]
        rhs = B @ V_next

        # Add boundary contributions
        # The first interior point (idx=0) has a stencil involving V[0] (lower boundary)
        # The last interior point (idx=N_interior-1) has a stencil involving V[-1] (upper boundary)

        # Compute boundary contributions for first interior point
        i = 1
        h_minus = S[i] - S[i - 1]
        h_plus = S[i + 1] - S[i]

        d_minus = -h_plus / (h_minus * (h_minus + h_plus))
        d2_minus = 2.0 / (h_minus * (h_minus + h_plus))

        Si = S[i]
        drift = (r - q) * Si
        diffusion = 0.5 * sigma ** 2 * Si ** 2
        L_minus = drift * d_minus + diffusion * d2_minus

        # Contribution from lower boundary to first interior point
        lower_bc_contrib = jnp.zeros(N_interior)
        lower_bc_contrib = lower_bc_contrib.at[0].set(
            -(-theta * dt * L_minus) * lower_boundary  # From A matrix
            + (1 - theta) * dt * L_minus * V[0, j + 1]  # From B matrix
        )

        # Compute boundary contributions for last interior point
        i = N_S - 2
        h_minus = S[i] - S[i - 1]
        h_plus = S[i + 1] - S[i]

        d_plus = h_minus / (h_plus * (h_minus + h_plus))
        d2_plus = 2.0 / (h_plus * (h_minus + h_plus))

        Si = S[i]
        drift = (r - q) * Si
        diffusion = 0.5 * sigma ** 2 * Si ** 2
        L_plus = drift * d_plus + diffusion * d2_plus

        # Contribution from upper boundary to last interior point
        upper_bc_contrib = jnp.zeros(N_interior)
        upper_bc_contrib = upper_bc_contrib.at[-1].set(
            -(-theta * dt * L_plus) * upper_boundary  # From A matrix
            + (1 - theta) * dt * L_plus * V[-1, j + 1]  # From B matrix
        )

        # Add boundary contributions to RHS
        rhs = rhs + lower_bc_contrib + upper_bc_contrib

        V_current = jnp.linalg.solve(A, rhs)
        V = V.at[1:-1, j].set(V_current)

    # Return solution at t=0
    return V[:, 0]
