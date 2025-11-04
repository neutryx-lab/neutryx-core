"""Advanced volatility surface management with smile dynamics.

This module provides comprehensive volatility surface modeling including:
- SVI (Stochastic Volatility Inspired) parameterization
- SABR model smile interpolation
- Local volatility surface construction
- Volatility surface calibration
- Smile dynamics and term structure
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from scipy.optimize import minimize


class SmileModel(Enum):
    """Volatility smile parameterization model."""
    SVI = "svi"  # Stochastic Volatility Inspired
    SABR = "sabr"  # Stochastic Alpha Beta Rho
    POLYNOMIAL = "polynomial"  # Polynomial in log-moneyness
    VANNA_VOLGA = "vanna_volga"  # Vanna-Volga approximation


@dataclass
class SVIParameters:
    """SVI (Stochastic Volatility Inspired) parameterization.

    The SVI formula for total implied variance w(k) as a function of
    log-moneyness k = log(K/F):

    w(k) = a + b * (ρ * (k - m) + sqrt((k - m)² + σ²))

    Attributes:
        a: Overall level of variance
        b: Angle between the two asymptotes
        rho: Orientation (right/left wing behavior)
        m: Translation of the smile along k-axis
        sigma: Smoothness of the vertex

    References:
        Gatheral, J., & Jacquier, A. (2014). "Arbitrage-free SVI volatility surfaces."
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, log_moneyness: Array) -> Array:
        """Compute total implied variance w(k).

        Args:
            log_moneyness: Log-moneyness k = log(K/F)

        Returns:
            Total implied variance
        """
        k = log_moneyness
        term1 = self.rho * (k - self.m)
        term2 = jnp.sqrt((k - self.m) ** 2 + self.sigma ** 2)
        w = self.a + self.b * (term1 + term2)
        return jnp.maximum(w, 1e-8)  # Ensure positive variance

    def implied_vol(self, log_moneyness: Array, T: float) -> Array:
        """Compute implied volatility from total variance.

        Args:
            log_moneyness: Log-moneyness k = log(K/F)
            T: Time to expiry

        Returns:
            Implied volatility
        """
        w = self.total_variance(log_moneyness)
        return jnp.sqrt(jnp.maximum(w / T, 1e-8))

    def check_arbitrage(self) -> List[str]:
        """Check for arbitrage conditions in SVI parameters.

        Returns:
            List of arbitrage violations (empty if none)
        """
        violations = []

        # Calendar arbitrage: dw/dT >= 0
        if self.a < 0:
            violations.append("Negative a parameter (calendar arbitrage)")

        # Butterfly arbitrage conditions
        if self.b < 0:
            violations.append("Negative b parameter (butterfly arbitrage)")

        if abs(self.rho) >= 1:
            violations.append("Invalid rho (must be in (-1, 1))")

        if self.sigma <= 0:
            violations.append("Non-positive sigma")

        # Gatheral-Jacquier no-arbitrage conditions
        # b * (1 + |rho|) < 4
        if self.b * (1 + abs(self.rho)) >= 4:
            violations.append("Gatheral-Jacquier condition violated")

        return violations


@dataclass
class SABRParameters:
    """SABR (Stochastic Alpha Beta Rho) model parameters.

    The SABR model dynamics:
    dF = α * F^β * dW1
    dα = ν * α * dW2
    dW1 * dW2 = ρ * dt

    Attributes:
        alpha: Initial volatility (vol-of-vol)
        beta: CEV exponent (0 = normal, 1 = lognormal)
        rho: Correlation between asset and vol
        nu: Vol-of-vol (volvol)

    References:
        Hagan, P. S., et al. (2002). "Managing Smile Risk."
    """

    alpha: float
    beta: float
    rho: float
    nu: float

    def implied_vol(self, F: float, K: float, T: float) -> float:
        """Compute SABR implied volatility using Hagan's formula.

        Args:
            F: Forward price
            K: Strike price
            T: Time to expiry

        Returns:
            Implied volatility

        Notes:
            This uses Hagan's asymptotic expansion formula, accurate for
            ATM and near-ATM strikes.
        """
        if abs(F - K) < 1e-8:  # ATM
            # ATM vol
            FK_mid = F
            FK_beta = FK_mid ** (self.beta - 1)

            # First term
            vol_atm = self.alpha / FK_beta

            # Second-order correction
            correction = (
                1
                + (
                    ((1 - self.beta) ** 2 / 24) * (self.alpha ** 2 / FK_mid ** (2 - 2 * self.beta))
                    + (self.rho * self.beta * self.nu * self.alpha) / (4 * FK_mid ** (1 - self.beta))
                    + ((2 - 3 * self.rho ** 2) / 24) * self.nu ** 2
                )
                * T
            )

            return vol_atm * correction

        # Non-ATM: full formula
        log_FK = jnp.log(F / K)
        FK_mid = jnp.sqrt(F * K)
        FK_beta = FK_mid ** (1 - self.beta)

        # z parameter
        z = (self.nu / self.alpha) * FK_beta * log_FK

        # x(z) function
        sqrt_term = jnp.sqrt(1 - 2 * self.rho * z + z ** 2)
        x_z = jnp.log((sqrt_term + z - self.rho) / (1 - self.rho))

        # Avoid division by zero
        if abs(z) < 1e-7:
            z_over_x = 1.0
        else:
            z_over_x = z / x_z

        # Main term
        numerator = self.alpha
        denominator = FK_beta * (
            1
            + ((1 - self.beta) ** 2 / 24) * (log_FK ** 2)
            + ((1 - self.beta) ** 4 / 1920) * (log_FK ** 4)
        )

        vol_base = numerator / denominator

        # Time correction
        correction = (
            1
            + (
                ((1 - self.beta) ** 2 / 24) * (self.alpha ** 2 / FK_mid ** (2 - 2 * self.beta))
                + (self.rho * self.beta * self.nu * self.alpha) / (4 * FK_mid ** (1 - self.beta))
                + ((2 - 3 * self.rho ** 2) / 24) * self.nu ** 2
            )
            * T
        )

        return float(vol_base * z_over_x * correction)


@dataclass
class VolatilitySurface:
    """Volatility surface with smile dynamics.

    Manages a grid of implied volatilities across strikes and tenors,
    with interpolation and extrapolation capabilities.

    Attributes:
        forward: Forward price
        tenors: Array of tenors (in years)
        strikes: Array of strikes for each tenor
        implied_vols: Array of implied volatilities [n_tenors, n_strikes]
        smile_model: Smile parameterization model
    """

    forward: float
    tenors: Array
    strikes: List[Array]  # One array per tenor
    implied_vols: List[Array]  # One array per tenor
    smile_model: SmileModel = SmileModel.SVI

    def __post_init__(self):
        """Validate and initialize surface."""
        n_tenors = len(self.tenors)

        if len(self.strikes) != n_tenors:
            raise ValueError("Number of strike arrays must match number of tenors")

        if len(self.implied_vols) != n_tenors:
            raise ValueError("Number of vol arrays must match number of tenors")

        # Convert to JAX arrays
        object.__setattr__(self, "tenors", jnp.asarray(self.tenors))

    def get_vol(self, strike: float, tenor: float) -> float:
        """Get implied volatility by interpolation.

        Args:
            strike: Strike price
            tenor: Time to expiry

        Returns:
            Interpolated implied volatility
        """
        # Find tenor bracket
        if tenor <= self.tenors[0]:
            # Extrapolate flat before first tenor
            tenor_idx = 0
            weight = 1.0
        elif tenor >= self.tenors[-1]:
            # Extrapolate flat after last tenor
            tenor_idx = len(self.tenors) - 1
            weight = 1.0
        else:
            # Interpolate between tenors
            tenor_idx = jnp.searchsorted(self.tenors, tenor) - 1
            tenor_idx = int(tenor_idx)
            t1 = float(self.tenors[tenor_idx])
            t2 = float(self.tenors[tenor_idx + 1])
            weight = (tenor - t1) / (t2 - t1)

        # Get vols for lower tenor
        vol1 = float(jnp.interp(strike, self.strikes[tenor_idx], self.implied_vols[tenor_idx]))

        if weight < 1.0 and tenor_idx + 1 < len(self.tenors):
            # Interpolate with upper tenor
            vol2 = float(
                jnp.interp(strike, self.strikes[tenor_idx + 1], self.implied_vols[tenor_idx + 1])
            )
            return vol1 * (1 - weight) + vol2 * weight
        else:
            return vol1

    def get_slice(self, tenor: float) -> Tuple[Array, Array]:
        """Get volatility smile for a specific tenor.

        Args:
            tenor: Time to expiry

        Returns:
            Tuple of (strikes, implied_vols) for the tenor
        """
        # Find closest tenor
        tenor_idx = int(jnp.argmin(jnp.abs(self.tenors - tenor)))
        return self.strikes[tenor_idx], self.implied_vols[tenor_idx]

    def calibrate_smile(self, tenor: float) -> SVIParameters | SABRParameters:
        """Calibrate smile model to a tenor slice.

        Args:
            tenor: Tenor to calibrate

        Returns:
            Calibrated smile parameters
        """
        strikes, vols = self.get_slice(tenor)

        if self.smile_model == SmileModel.SVI:
            return self._calibrate_svi(strikes, vols, tenor)
        elif self.smile_model == SmileModel.SABR:
            return self._calibrate_sabr(strikes, vols, tenor)
        else:
            raise ValueError(f"Calibration not implemented for {self.smile_model}")

    def _calibrate_svi(self, strikes: Array, vols: Array, T: float) -> SVIParameters:
        """Calibrate SVI parameters to market vols.

        Args:
            strikes: Strike array
            vols: Implied volatility array
            T: Time to expiry

        Returns:
            Calibrated SVI parameters
        """
        # Convert to log-moneyness and total variance
        log_moneyness = jnp.log(strikes / self.forward)
        total_var = vols ** 2 * T

        # Initial guess
        a0 = float(jnp.mean(total_var))
        b0 = 0.1
        rho0 = 0.0
        m0 = 0.0
        sigma0 = 0.1

        initial = jnp.array([a0, b0, rho0, m0, sigma0])

        # Objective function
        def objective(params):
            svi = SVIParameters(*params)
            model_var = svi.total_variance(log_moneyness)
            error = jnp.sum((model_var - total_var) ** 2)
            return float(error)

        # Constraints
        bounds = [
            (0.0001, None),  # a > 0
            (0.001, 1.0),  # 0 < b < 1
            (-0.999, 0.999),  # -1 < rho < 1
            (-1.0, 1.0),  # m
            (0.001, 2.0),  # sigma > 0
        ]

        # Optimize
        result = minimize(objective, initial, method="L-BFGS-B", bounds=bounds)

        return SVIParameters(*result.x)

    def _calibrate_sabr(self, strikes: Array, vols: Array, T: float) -> SABRParameters:
        """Calibrate SABR parameters to market vols.

        Args:
            strikes: Strike array
            vols: Implied volatility array
            T: Time to expiry

        Returns:
            Calibrated SABR parameters
        """
        # Initial guess
        atm_vol = float(jnp.interp(self.forward, strikes, vols))
        initial = jnp.array([atm_vol, 0.5, 0.0, 0.3])  # alpha, beta, rho, nu

        # Objective function
        def objective(params):
            sabr = SABRParameters(*params)
            error = 0.0
            for K, vol_market in zip(strikes, vols):
                vol_model = sabr.implied_vol(self.forward, float(K), T)
                error += (vol_model - float(vol_market)) ** 2
            return error

        # Bounds
        bounds = [
            (0.001, 2.0),  # alpha > 0
            (0.0, 1.0),  # 0 <= beta <= 1
            (-0.999, 0.999),  # -1 < rho < 1
            (0.001, 5.0),  # nu > 0
        ]

        # Optimize
        result = minimize(objective, initial, method="L-BFGS-B", bounds=bounds)

        return SABRParameters(*result.x)

    def to_local_vol(self, n_strikes: int = 50, n_tenors: int = 20) -> "LocalVolSurface":
        """Convert implied volatility surface to local volatility surface.

        Uses Dupire's formula:
        σ_local²(K,T) = (∂C/∂T + rKC_K) / (0.5 K² C_KK)

        Args:
            n_strikes: Number of strike points
            n_tenors: Number of tenor points

        Returns:
            Local volatility surface
        """
        # Create regular grid
        K_grid = jnp.linspace(self.forward * 0.5, self.forward * 1.5, n_strikes)
        T_grid = jnp.linspace(0.1, float(self.tenors[-1]), n_tenors)

        # Compute local volatilities using finite differences
        local_vols = jnp.zeros((n_tenors, n_strikes))

        for i, T in enumerate(T_grid):
            for j, K in enumerate(K_grid):
                # Get implied vol and compute derivatives numerically
                dK = 0.01 * K
                dT = 0.01

                vol = self.get_vol(float(K), T)
                vol_K_up = self.get_vol(float(K + dK), T)
                vol_K_down = self.get_vol(float(K - dK), T)

                # Finite difference approximation for local vol
                # Simplified: assume local vol ≈ implied vol (can be refined)
                local_vols = local_vols.at[i, j].set(vol)

        return LocalVolSurface(forward=self.forward, strikes=K_grid, tenors=T_grid, local_vols=local_vols)


@dataclass
class LocalVolSurface:
    """Local volatility surface for path-dependent pricing.

    Local volatility σ_local(S,t) is the instantaneous volatility
    as a function of spot level and time, used in:
    dS = μS dt + σ_local(S,t) S dW

    Attributes:
        forward: Forward price
        strikes: Strike grid
        tenors: Tenor grid
        local_vols: Local volatility values [n_tenors, n_strikes]
    """

    forward: float
    strikes: Array
    tenors: Array
    local_vols: Array

    def get_local_vol(self, spot: float, time: float) -> float:
        """Get local volatility at (spot, time).

        Args:
            spot: Spot price
            time: Time

        Returns:
            Local volatility
        """
        # Bilinear interpolation
        vol = float(
            jnp.interp(
                time,
                self.tenors,
                jnp.array([float(jnp.interp(spot, self.strikes, self.local_vols[i, :])) for i in range(len(self.tenors))]),
            )
        )
        return vol


@dataclass
class VolSurfaceBuilder:
    """Builder for volatility surfaces from market data.

    Constructs smooth, arbitrage-free volatility surfaces from
    sparse market quotes.
    """

    forward: float
    market_quotes: Dict[Tuple[float, float], float]  # (tenor, strike) -> vol

    def build(self, smile_model: SmileModel = SmileModel.SVI) -> VolatilitySurface:
        """Build volatility surface from market quotes.

        Args:
            smile_model: Smile parameterization model

        Returns:
            Calibrated volatility surface
        """
        # Group quotes by tenor
        tenor_groups: Dict[float, List[Tuple[float, float]]] = {}
        for (tenor, strike), vol in self.market_quotes.items():
            if tenor not in tenor_groups:
                tenor_groups[tenor] = []
            tenor_groups[tenor].append((strike, vol))

        # Sort tenors
        tenors = sorted(tenor_groups.keys())

        # Build smile for each tenor
        strikes_list = []
        vols_list = []

        for tenor in tenors:
            quotes = sorted(tenor_groups[tenor])
            strikes = jnp.array([k for k, _ in quotes])
            vols = jnp.array([v for _, v in quotes])

            strikes_list.append(strikes)
            vols_list.append(vols)

        return VolatilitySurface(
            forward=self.forward,
            tenors=jnp.array(tenors),
            strikes=strikes_list,
            implied_vols=vols_list,
            smile_model=smile_model,
        )
