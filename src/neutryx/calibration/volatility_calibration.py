"""Volatility surface calibration with regularization.

Provides utilities for calibrating local and implied volatility surfaces
from market option prices using regularization techniques to ensure:
- Smoothness of the resulting surface
- Absence of arbitrage opportunities
- Numerical stability

Common applications:
- Building implied volatility surfaces from market quotes
- Calibrating local volatility models (Dupire)
- Constructing risk-neutral densities
- Model-free volatility surface interpolation/extrapolation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import grad
import scipy.optimize

from .regularization import (
    SmoothnessRegularizer,
    ArbitrageFreeConstraints,
    TikhonovRegularizer,
)


@dataclass
class VolatilitySurfaceCalibration:
    """Calibrate volatility surface to market option prices.

    Uses regularization to ensure smooth, arbitrage-free surfaces.

    Attributes
    ----------
    maturities : Array
        Option maturities (in years)
    strikes : Array
        Strike prices
    market_prices : Array
        Market option prices [n_maturities, n_strikes]
    spot : float
        Current spot price
    rate : float
        Risk-free rate
    smoothness_lambda : float, optional
        Smoothness regularization strength (default: 0.01)
    arbitrage_lambda : float, optional
        Arbitrage constraint penalty strength (default: 100.0)
    """
    maturities: jnp.ndarray
    strikes: jnp.ndarray
    market_prices: jnp.ndarray
    spot: float
    rate: float
    smoothness_lambda: float = 0.01
    arbitrage_lambda: float = 100.0

    def __post_init__(self):
        """Validate inputs."""
        self.maturities = jnp.array(self.maturities)
        self.strikes = jnp.array(self.strikes)
        self.market_prices = jnp.array(self.market_prices)

        expected_shape = (len(self.maturities), len(self.strikes))
        if self.market_prices.shape != expected_shape:
            raise ValueError(
                f"market_prices shape {self.market_prices.shape} must match "
                f"(n_maturities, n_strikes) = {expected_shape}"
            )

    def pricing_error(
        self,
        implied_vols: jnp.ndarray,
        pricing_fn: Callable,
    ) -> float:
        """Compute pricing error between model and market.

        Parameters
        ----------
        implied_vols : Array
            Implied volatilities [n_maturities, n_strikes]
        pricing_fn : Callable
            Function to price options given (S, K, T, r, sigma)

        Returns
        -------
        float
            Mean squared pricing error
        """
        model_prices = jnp.zeros_like(self.market_prices)

        for i, T in enumerate(self.maturities):
            for j, K in enumerate(self.strikes):
                sigma = implied_vols[i, j]
                model_price = pricing_fn(
                    S=self.spot,
                    K=K,
                    T=T,
                    r=self.rate,
                    sigma=sigma,
                )
                model_prices = model_prices.at[i, j].set(model_price)

        # Mean squared error
        errors = model_prices - self.market_prices
        mse = jnp.mean(errors ** 2)

        return float(mse)

    def calibrate(
        self,
        pricing_fn: Callable,
        initial_vols: Optional[jnp.ndarray] = None,
        method: str = "L-BFGS-B",
        max_iter: int = 1000,
    ) -> dict:
        """Calibrate volatility surface to market prices.

        Parameters
        ----------
        pricing_fn : Callable
            Option pricing function
        initial_vols : Array, optional
            Initial volatility guess. If None, uses 0.2 (20%)
        method : str, optional
            Optimization method (default: L-BFGS-B)
        max_iter : int, optional
            Maximum iterations

        Returns
        -------
        dict
            Calibration results with keys:
            - 'implied_vols': Calibrated volatility surface
            - 'pricing_error': Final pricing error
            - 'smoothness_penalty': Final smoothness penalty
            - 'arbitrage_penalty': Final arbitrage penalty
            - 'success': Whether optimization succeeded
        """
        # Initialize
        if initial_vols is None:
            initial_vols = 0.2 * jnp.ones_like(self.market_prices)

        shape = self.market_prices.shape

        # Create regularizers
        smoothness_reg = SmoothnessRegularizer(
            lambda_reg=self.smoothness_lambda,
            order=2,
            direction="both",
        )

        arbitrage_constraints = ArbitrageFreeConstraints(
            lambda_calendar=self.arbitrage_lambda,
            lambda_butterfly=self.arbitrage_lambda,
        )

        # Objective function
        def objective(vols_flat: jnp.ndarray) -> float:
            vols = vols_flat.reshape(shape)

            # Pricing error
            price_error = self.pricing_error(vols, pricing_fn)

            # Smoothness penalty
            smooth_penalty = smoothness_reg.penalty_2d(vols)

            # Arbitrage penalty
            arb_penalty = arbitrage_constraints.total_penalty(
                vols,
                self.maturities,
                self.strikes,
                self.spot,
                self.rate,
            )

            return price_error + smooth_penalty + arb_penalty

        # Optimize
        result = scipy.optimize.minimize(
            fun=objective,
            x0=initial_vols.flatten(),
            method=method,
            options={'maxiter': max_iter},
            bounds=[(0.01, 2.0)] * len(initial_vols.flatten()),  # Reasonable vol bounds
        )

        # Extract results
        calibrated_vols = result.x.reshape(shape)

        return {
            'implied_vols': calibrated_vols,
            'pricing_error': self.pricing_error(calibrated_vols, pricing_fn),
            'smoothness_penalty': smoothness_reg.penalty_2d(calibrated_vols),
            'arbitrage_penalty': arbitrage_constraints.total_penalty(
                calibrated_vols,
                self.maturities,
                self.strikes,
                self.spot,
                self.rate,
            ),
            'success': result.success,
            'message': result.message,
            'iterations': result.nit,
        }


@dataclass
class LocalVolatilityCalibration:
    """Calibrate local volatility surface using Dupire's formula with regularization.

    Local volatility σ_loc(K, T) is computed from implied volatilities using:

        σ²_loc(K, T) = (∂C/∂T + rK∂C/∂K) / (½K²∂²C/∂K²)

    Regularization ensures numerical stability of derivatives.

    Attributes
    ----------
    maturities : Array
        Option maturities
    strikes : Array
        Strike prices
    implied_vols : Array
        Implied volatility surface [n_maturities, n_strikes]
    spot : float
        Spot price
    rate : float
        Risk-free rate
    dividend_yield : float, optional
        Continuous dividend yield (default: 0)
    smoothness_lambda : float, optional
        Smoothness regularization for derivatives (default: 0.1)
    """
    maturities: jnp.ndarray
    strikes: jnp.ndarray
    implied_vols: jnp.ndarray
    spot: float
    rate: float
    dividend_yield: float = 0.0
    smoothness_lambda: float = 0.1

    def __post_init__(self):
        """Validate inputs."""
        self.maturities = jnp.array(self.maturities)
        self.strikes = jnp.array(self.strikes)
        self.implied_vols = jnp.array(self.implied_vols)

    def compute_local_volatility(self) -> jnp.ndarray:
        """Compute local volatility surface using Dupire's formula.

        Returns
        -------
        Array
            Local volatility surface [n_maturities, n_strikes]

        Notes
        -----
        Uses finite differences with regularization to compute derivatives
        stably from the implied volatility surface.
        """
        from scipy.stats import norm

        n_T, n_K = self.implied_vols.shape
        local_vols = jnp.zeros_like(self.implied_vols)

        # Grid spacings
        dT = jnp.diff(self.maturities).mean() if n_T > 1 else 0.1
        dK = jnp.diff(self.strikes).mean() if n_K > 1 else self.spot * 0.01

        for i in range(n_T):
            for j in range(n_K):
                T = self.maturities[i]
                K = self.strikes[j]
                sigma_impl = self.implied_vols[i, j]

                # Black-Scholes price and derivatives
                forward = self.spot * jnp.exp((self.rate - self.dividend_yield) * T)
                d1 = (jnp.log(forward / K) + 0.5 * sigma_impl ** 2 * T) / (sigma_impl * jnp.sqrt(T))
                d2 = d1 - sigma_impl * jnp.sqrt(T)

                discount = jnp.exp(-self.rate * T)
                C = discount * (forward * norm.cdf(float(d1)) - K * norm.cdf(float(d2)))

                # Derivatives using finite differences
                # ∂C/∂T
                if i < n_T - 1:
                    sigma_next = self.implied_vols[i + 1, j]
                    T_next = self.maturities[i + 1]
                    d1_next = (jnp.log(forward / K) + 0.5 * sigma_next ** 2 * T_next) / (sigma_next * jnp.sqrt(T_next))
                    d2_next = d1_next - sigma_next * jnp.sqrt(T_next)
                    C_next = jnp.exp(-self.rate * T_next) * (
                        forward * norm.cdf(float(d1_next)) - K * norm.cdf(float(d2_next))
                    )
                    dC_dT = (C_next - C) / (T_next - T)
                else:
                    dC_dT = 0.0

                # ∂C/∂K (using vega)
                vega = discount * forward * norm.pdf(float(d1)) * jnp.sqrt(T)
                if j < n_K - 1 and j > 0:
                    # Central difference
                    dC_dK = (self.implied_vols[i, j + 1] - self.implied_vols[i, j - 1]) / (2 * dK) * vega
                else:
                    dC_dK = -discount * norm.cdf(float(d2))  # Analytical

                # ∂²C/∂K² (gamma)
                gamma = discount * norm.pdf(float(d1)) / (K * sigma_impl * jnp.sqrt(T))

                # Dupire formula
                numerator = dC_dT + self.rate * K * dC_dK
                denominator = 0.5 * K * K * gamma

                # Regularization: avoid division by very small numbers
                epsilon = 1e-8
                sigma_loc_sq = numerator / jnp.maximum(denominator, epsilon)

                # Ensure positive and bounded
                sigma_loc = jnp.sqrt(jnp.clip(sigma_loc_sq, 0.01, 4.0))

                local_vols = local_vols.at[i, j].set(sigma_loc)

        # Apply smoothness regularization
        smoothness_reg = SmoothnessRegularizer(
            lambda_reg=self.smoothness_lambda,
            order=2,
            direction="both",
        )

        # Smooth the local vol surface
        # (In practice, would optimize to minimize penalty while staying close to Dupire values)
        # For now, return direct Dupire calculation

        return local_vols


__all__ = [
    "VolatilitySurfaceCalibration",
    "LocalVolatilityCalibration",
]
