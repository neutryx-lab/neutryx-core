"""Regularization and stability techniques for model calibration.

This module provides regularization methods to ensure stable and well-behaved
model calibration in quantitative finance:

1. Tikhonov Regularization: Penalizes large parameter values
2. L1/L2 Penalties: Sparsity and smoothness inducing penalties
3. Arbitrage-Free Constraints: Ensures no calendar/butterfly arbitrage
4. Smoothness Penalties: Enforces smooth volatility surfaces

These techniques are essential for:
- Local volatility calibration
- Implied volatility surface construction
- Stochastic volatility parameter estimation
- Interest rate model calibration

Key challenges addressed:
- Ill-posed inverse problems
- Overfitting to noisy market data
- Numerical instability
- Arbitrage violations in calibrated surfaces

References
----------
Tikhonov, A. N., & Arsenin, V. Y. (1977). "Solutions of Ill-posed Problems."
Winston & Sons.

Cont, R., & Da Fonseca, J. (2002). "Dynamics of implied volatility surfaces."
Quantitative Finance, 2(1), 45-60.

Andersen, L., & Brotherton-Ratcliffe, R. (1998). "The equity option volatility
smile: an implicit finite-difference approach." The Journal of Computational
Finance, 1(2), 5-37.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import grad, hessian


@dataclass
class TikhonovRegularizer:
    """Tikhonov (L2) regularization for stable parameter estimation.

    Adds penalty term: λ ||Γ(θ - θ₀)||²₂

    where:
        - λ: regularization strength
        - Γ: regularization matrix (often identity or derivative operator)
        - θ: parameters to estimate
        - θ₀: prior/target parameters

    Attributes
    ----------
    lambda_reg : float
        Regularization strength (must be non-negative)
    prior_params : Optional[Array]
        Prior/target parameter values. If None, regularizes toward zero.
    regularization_matrix : Optional[Array]
        Regularization matrix Γ. If None, uses identity (standard L2).
    """
    lambda_reg: float
    prior_params: Optional[jnp.ndarray] = None
    regularization_matrix: Optional[jnp.ndarray] = None

    def __post_init__(self):
        """Validate regularization parameters."""
        if self.lambda_reg < 0:
            raise ValueError(f"lambda_reg must be non-negative, got {self.lambda_reg}")

    def penalty(self, params: jnp.ndarray) -> float:
        """Compute Tikhonov penalty for given parameters.

        Parameters
        ----------
        params : Array
            Parameter vector

        Returns
        -------
        float
            Penalty value λ ||Γ(θ - θ₀)||²₂
        """
        # Deviation from prior
        if self.prior_params is not None:
            deviation = params - self.prior_params
        else:
            deviation = params

        # Apply regularization matrix
        if self.regularization_matrix is not None:
            deviation = self.regularization_matrix @ deviation

        # L2 penalty
        penalty_value = self.lambda_reg * jnp.sum(deviation ** 2)

        return penalty_value

    def gradient(self, params: jnp.ndarray) -> jnp.ndarray:
        """Compute gradient of Tikhonov penalty.

        Parameters
        ----------
        params : Array
            Parameter vector

        Returns
        -------
        Array
            Gradient of penalty with respect to parameters
        """
        return grad(self.penalty)(params)

    def regularized_objective(
        self,
        objective_fn: Callable[[jnp.ndarray], float],
    ) -> Callable[[jnp.ndarray], float]:
        """Create regularized objective function.

        Parameters
        ----------
        objective_fn : Callable
            Original objective function (e.g., pricing error)

        Returns
        -------
        Callable
            Regularized objective: objective + penalty
        """
        def regularized_fn(params: jnp.ndarray) -> float:
            return objective_fn(params) + self.penalty(params)

        return regularized_fn


@dataclass
class L1Regularizer:
    """L1 (Lasso) regularization for sparse parameter estimation.

    Adds penalty term: λ ||θ||₁ = λ Σᵢ |θᵢ|

    L1 regularization induces sparsity by driving some parameters to exactly zero.

    Attributes
    ----------
    lambda_reg : float
        Regularization strength (must be non-negative)
    weights : Optional[Array]
        Element-wise weights for different parameters. If None, uses uniform weights.
    """
    lambda_reg: float
    weights: Optional[jnp.ndarray] = None

    def __post_init__(self):
        """Validate regularization parameters."""
        if self.lambda_reg < 0:
            raise ValueError(f"lambda_reg must be non-negative, got {self.lambda_reg}")

    def penalty(self, params: jnp.ndarray) -> float:
        """Compute L1 penalty for given parameters.

        Parameters
        ----------
        params : Array
            Parameter vector

        Returns
        -------
        float
            Penalty value λ ||θ||₁
        """
        if self.weights is not None:
            weighted_params = params * self.weights
        else:
            weighted_params = params

        penalty_value = self.lambda_reg * jnp.sum(jnp.abs(weighted_params))

        return penalty_value

    def regularized_objective(
        self,
        objective_fn: Callable[[jnp.ndarray], float],
    ) -> Callable[[jnp.ndarray], float]:
        """Create regularized objective function.

        Parameters
        ----------
        objective_fn : Callable
            Original objective function

        Returns
        -------
        Callable
            Regularized objective: objective + penalty
        """
        def regularized_fn(params: jnp.ndarray) -> float:
            return objective_fn(params) + self.penalty(params)

        return regularized_fn


@dataclass
class ElasticNetRegularizer:
    """Elastic Net regularization combining L1 and L2 penalties.

    Adds penalty term: λ₁ ||θ||₁ + λ₂ ||θ||²₂

    Combines sparsity (L1) with stability (L2).

    Attributes
    ----------
    lambda_l1 : float
        L1 regularization strength
    lambda_l2 : float
        L2 regularization strength
    """
    lambda_l1: float
    lambda_l2: float

    def __post_init__(self):
        """Validate regularization parameters."""
        if self.lambda_l1 < 0:
            raise ValueError(f"lambda_l1 must be non-negative, got {self.lambda_l1}")
        if self.lambda_l2 < 0:
            raise ValueError(f"lambda_l2 must be non-negative, got {self.lambda_l2}")

    def penalty(self, params: jnp.ndarray) -> float:
        """Compute Elastic Net penalty.

        Parameters
        ----------
        params : Array
            Parameter vector

        Returns
        -------
        float
            Combined L1 + L2 penalty
        """
        l1_penalty = self.lambda_l1 * jnp.sum(jnp.abs(params))
        l2_penalty = self.lambda_l2 * jnp.sum(params ** 2)

        return l1_penalty + l2_penalty

    def regularized_objective(
        self,
        objective_fn: Callable[[jnp.ndarray], float],
    ) -> Callable[[jnp.ndarray], float]:
        """Create regularized objective function."""
        def regularized_fn(params: jnp.ndarray) -> float:
            return objective_fn(params) + self.penalty(params)

        return regularized_fn


class SmoothnessRegularizer:
    """Smoothness penalties for volatility surfaces.

    Penalizes roughness in local volatility or implied volatility surfaces
    by adding penalties on derivatives:

    - First-order: Penalizes large gradients
    - Second-order: Penalizes curvature (preferred for smoothness)

    This prevents oscillations and ensures realistic volatility surfaces.
    """

    def __init__(
        self,
        lambda_reg: float,
        order: int = 2,
        direction: str = "both",
    ):
        """Initialize smoothness regularizer.

        Parameters
        ----------
        lambda_reg : float
            Regularization strength
        order : int, optional
            Derivative order (1 or 2, default: 2)
        direction : str, optional
            Regularization direction: "strike", "maturity", or "both"
        """
        if lambda_reg < 0:
            raise ValueError(f"lambda_reg must be non-negative, got {lambda_reg}")
        if order not in [1, 2]:
            raise ValueError(f"order must be 1 or 2, got {order}")
        if direction not in ["strike", "maturity", "both"]:
            raise ValueError(f"direction must be 'strike', 'maturity', or 'both'")

        self.lambda_reg = lambda_reg
        self.order = order
        self.direction = direction

    def penalty_1d(self, values: jnp.ndarray) -> float:
        """Compute smoothness penalty for 1D array.

        Parameters
        ----------
        values : Array
            1D array of values (e.g., volatilities along strike axis)

        Returns
        -------
        float
            Smoothness penalty
        """
        if self.order == 1:
            # First-order: sum of squared differences
            diff = jnp.diff(values)
            penalty = jnp.sum(diff ** 2)
        else:
            # Second-order: sum of squared second differences
            diff2 = jnp.diff(values, n=2)
            penalty = jnp.sum(diff2 ** 2)

        return float(self.lambda_reg * penalty)

    def penalty_2d(self, surface: jnp.ndarray) -> float:
        """Compute smoothness penalty for 2D surface.

        Parameters
        ----------
        surface : Array
            2D array of shape [n_maturities, n_strikes]

        Returns
        -------
        float
            Total smoothness penalty
        """
        penalty = 0.0

        # Penalty along strike direction
        if self.direction in ["strike", "both"]:
            for i in range(surface.shape[0]):
                penalty += self.penalty_1d(surface[i, :])

        # Penalty along maturity direction
        if self.direction in ["maturity", "both"]:
            for j in range(surface.shape[1]):
                penalty += self.penalty_1d(surface[:, j])

        return penalty

    def regularized_objective(
        self,
        objective_fn: Callable[[jnp.ndarray], float],
        surface_shape: tuple[int, int],
    ) -> Callable[[jnp.ndarray], float]:
        """Create regularized objective for surface calibration.

        Parameters
        ----------
        objective_fn : Callable
            Original objective function taking flattened surface
        surface_shape : tuple
            Shape of volatility surface (n_maturities, n_strikes)

        Returns
        -------
        Callable
            Regularized objective
        """
        def regularized_fn(params_flat: jnp.ndarray) -> float:
            # Reshape to surface
            surface = params_flat.reshape(surface_shape)

            # Original objective
            obj_value = objective_fn(params_flat)

            # Smoothness penalty
            smooth_penalty = self.penalty_2d(surface)

            return obj_value + smooth_penalty

        return regularized_fn


class ArbitrageFreeConstraints:
    """Arbitrage-free constraints for volatility surfaces.

    Enforces no-arbitrage conditions:
    1. Calendar spread arbitrage: σ²(T)·T must be increasing in T
    2. Butterfly arbitrage: Local volatility must be positive
    3. Call price monotonicity: C(K) decreasing in K

    These are soft constraints implemented as penalty functions.
    """

    def __init__(
        self,
        lambda_calendar: float = 1000.0,
        lambda_butterfly: float = 1000.0,
        epsilon: float = 1e-6,
    ):
        """Initialize arbitrage-free constraint enforcer.

        Parameters
        ----------
        lambda_calendar : float, optional
            Penalty strength for calendar spread violations
        lambda_butterfly : float, optional
            Penalty strength for butterfly violations
        epsilon : float, optional
            Small constant for numerical stability
        """
        self.lambda_calendar = lambda_calendar
        self.lambda_butterfly = lambda_butterfly
        self.epsilon = epsilon

    def calendar_spread_penalty(
        self,
        total_variance: jnp.ndarray,
    ) -> float:
        """Penalize calendar spread arbitrage violations.

        Total variance σ²(T)·T must be non-decreasing in T.

        Parameters
        ----------
        total_variance : Array
            Total variance at each maturity (sorted by maturity)

        Returns
        -------
        float
            Penalty for violations
        """
        # Compute differences
        diffs = jnp.diff(total_variance)

        # Penalize negative differences (violations)
        violations = jnp.maximum(-diffs, 0.0)

        penalty = self.lambda_calendar * jnp.sum(violations ** 2)

        return penalty

    def butterfly_penalty(
        self,
        call_prices: jnp.ndarray,
        strikes: jnp.ndarray,
    ) -> float:
        """Penalize butterfly arbitrage violations.

        Call prices must be convex in strike: C''(K) ≥ 0

        Parameters
        ----------
        call_prices : Array
            Call option prices (sorted by strike)
        strikes : Array
            Strike prices

        Returns
        -------
        float
            Penalty for violations
        """
        # Compute discrete second derivative
        # C''(K) ≈ [C(K+h) - 2C(K) + C(K-h)] / h²

        if len(strikes) < 3:
            return 0.0

        # Assume uniform grid for simplicity
        dK = strikes[1] - strikes[0]

        # Second differences
        second_diff = jnp.diff(call_prices, n=2) / (dK ** 2)

        # Penalize negative second derivatives
        violations = jnp.maximum(-second_diff, 0.0)

        penalty = self.lambda_butterfly * jnp.sum(violations ** 2)

        return penalty

    def positive_density_penalty(
        self,
        call_prices: jnp.ndarray,
        strikes: jnp.ndarray,
        discount_factor: float = 1.0,
    ) -> float:
        """Penalize negative risk-neutral density.

        Risk-neutral density: q(K) = e^{rT} ∂²C/∂K²

        Parameters
        ----------
        call_prices : Array
            Call option prices
        strikes : Array
            Strike prices
        discount_factor : float, optional
            Discount factor e^{-rT}

        Returns
        -------
        float
            Penalty for negative density
        """
        if len(strikes) < 3:
            return 0.0

        dK = strikes[1] - strikes[0]

        # Digital payoff approximation (risk-neutral density)
        second_diff = jnp.diff(call_prices, n=2) / (dK ** 2)
        density = second_diff / discount_factor

        # Penalize negative density
        violations = jnp.maximum(-density, 0.0)

        penalty = self.lambda_butterfly * jnp.sum(violations ** 2)

        return penalty

    def total_penalty(
        self,
        implied_vols: jnp.ndarray,
        maturities: jnp.ndarray,
        strikes: jnp.ndarray,
        spot: float,
        rate: float,
    ) -> float:
        """Compute total arbitrage penalty for implied vol surface.

        Parameters
        ----------
        implied_vols : Array
            Implied volatility surface [n_maturities, n_strikes]
        maturities : Array
            Maturity times
        strikes : Array
            Strike prices
        spot : float
            Spot price
        rate : float
            Risk-free rate

        Returns
        -------
        float
            Total arbitrage penalty
        """
        total_pen = 0.0

        # Calendar spread check
        # Use ATM variance for each maturity
        atm_idx = jnp.argmin(jnp.abs(strikes - spot))
        atm_vols = implied_vols[:, atm_idx]
        total_variance = atm_vols ** 2 * maturities

        total_pen += self.calendar_spread_penalty(total_variance)

        # Butterfly check for each maturity
        for i, T in enumerate(maturities):
            vols_at_T = implied_vols[i, :]

            # Convert to call prices (simplified Black-Scholes)
            from scipy.stats import norm

            discount = jnp.exp(-rate * T)
            forward = spot * jnp.exp(rate * T)

            call_prices = []
            for j, K in enumerate(strikes):
                vol = vols_at_T[j]
                d1 = (jnp.log(forward / K) + 0.5 * vol ** 2 * T) / (vol * jnp.sqrt(T))
                d2 = d1 - vol * jnp.sqrt(T)

                call_price = discount * (forward * norm.cdf(float(d1)) - K * norm.cdf(float(d2)))
                call_prices.append(call_price)

            call_prices = jnp.array(call_prices)
            total_pen += self.butterfly_penalty(call_prices, strikes)

        return total_pen


def create_difference_matrix(n: int, order: int = 1) -> jnp.ndarray:
    """Create finite difference matrix for derivatives.

    Used in Tikhonov regularization with derivative operators.

    Parameters
    ----------
    n : int
        Dimension of parameter vector
    order : int, optional
        Derivative order (1 or 2)

    Returns
    -------
    Array
        Finite difference matrix of shape [(n-order), n]

    Examples
    --------
    >>> D1 = create_difference_matrix(5, order=1)
    >>> D1.shape
    (4, 5)
    >>> # D1 @ x computes first differences of x
    """
    if order == 1:
        # First difference matrix
        D = jnp.eye(n)[:-1] - jnp.eye(n)[1:]
        return -D  # Standard convention

    elif order == 2:
        # Second difference matrix
        D1 = create_difference_matrix(n, order=1)
        D2 = create_difference_matrix(n - 1, order=1)
        return D2 @ D1

    else:
        raise ValueError(f"order must be 1 or 2, got {order}")


__all__ = [
    "TikhonovRegularizer",
    "L1Regularizer",
    "ElasticNetRegularizer",
    "SmoothnessRegularizer",
    "ArbitrageFreeConstraints",
    "create_difference_matrix",
]
