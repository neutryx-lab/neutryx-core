"""Wrong-way risk (WWR) modeling for CVA calculations.

Wrong-way risk occurs when exposure to a counterparty is adversely correlated
with the credit quality of that counterparty. This module provides tools to
model and quantify WWR in XVA calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array


class WWRType(Enum):
    """Type of wrong-way risk."""
    GENERAL = "general"  # Systemic correlation (macro factors)
    SPECIFIC = "specific"  # Direct dependence (e.g., credit default swap on own name)


@dataclass
class WWRParameters:
    """Parameters for wrong-way risk modeling.

    Attributes:
        correlation: Correlation between exposure and credit quality (-1 to 1)
        wwr_type: Type of wrong-way risk
        recovery_correlation: Correlation affecting recovery rates
        jump_correlation: Correlation in jump-to-default scenarios
    """
    correlation: float = 0.0
    wwr_type: WWRType = WWRType.GENERAL
    recovery_correlation: float = 0.0
    jump_correlation: float = 0.0

    def __post_init__(self):
        if not -1.0 <= self.correlation <= 1.0:
            raise ValueError("Correlation must be between -1 and 1")
        if not -1.0 <= self.recovery_correlation <= 1.0:
            raise ValueError("Recovery correlation must be between -1 and 1")


def simulate_correlated_defaults(
    key: jax.random.KeyArray,
    exposure_paths: Array,
    hazard_rate: float,
    correlation: float,
    T: float,
    dt: Optional[float] = None
) -> Tuple[Array, Array]:
    """Simulate default times correlated with exposure paths.

    Args:
        key: JAX random key
        exposure_paths: Simulated exposure paths [n_paths, n_steps]
        hazard_rate: Base hazard rate (intensity)
        correlation: Correlation between exposure and default intensity
        T: Time horizon
        dt: Time step (inferred if None)

    Returns:
        Tuple of (default_times, default_indicators)
        - default_times: Time of default for each path [n_paths]
        - default_indicators: Binary indicator matrix [n_paths, n_steps]

    Notes:
        Uses a stochastic hazard rate model:
        λ(t) = λ_0 × exp(ρ × Z(t))

        where Z(t) is a standardized exposure level.
    """
    n_paths, n_steps = exposure_paths.shape

    if dt is None:
        dt = T / n_steps

    # Standardize exposure to mean 0, std 1 for each time step
    exposure_mean = jnp.mean(exposure_paths, axis=0, keepdims=True)
    exposure_std = jnp.std(exposure_paths, axis=0, keepdims=True) + 1e-8
    standardized_exposure = (exposure_paths - exposure_mean) / exposure_std

    # Correlated hazard rate: λ(t) = λ_0 * exp(ρ * Z(t))
    # For small ρ: λ(t) ≈ λ_0 * (1 + ρ * Z(t))
    hazard_multiplier = jnp.exp(correlation * standardized_exposure)
    hazard_rates = hazard_rate * hazard_multiplier

    # Simulate defaults using hazard rates
    key, subkey = jax.random.split(key)
    uniforms = jax.random.uniform(subkey, shape=(n_paths, n_steps))

    # Probability of default in each time step: 1 - exp(-λ * dt)
    pd_step = 1.0 - jnp.exp(-hazard_rates * dt)

    # Default indicators
    defaults = uniforms < pd_step

    # Find first default time for each path
    default_indices = jnp.argmax(defaults, axis=1)

    # If no default occurred, set to n_steps
    no_default = ~jnp.any(defaults, axis=1)
    default_indices = jnp.where(no_default, n_steps, default_indices)

    default_times = default_indices * dt

    # Create default indicator matrix (1 from default onwards)
    time_grid = jnp.arange(n_steps)
    default_indicators = (time_grid[None, :] >= default_indices[:, None]).astype(jnp.float32)

    return default_times, default_indicators


def cva_with_wwr(
    key: jax.random.KeyArray,
    exposure_paths: Array,
    df_t: Array,
    hazard_rate: float,
    lgd: float,
    wwr_params: WWRParameters,
    T: float
) -> Tuple[float, float, Array]:
    """Calculate CVA accounting for wrong-way risk.

    Args:
        key: JAX random key
        exposure_paths: Monte Carlo exposure paths [n_paths, n_steps]
        df_t: Discount factors [n_steps]
        hazard_rate: Base hazard rate
        lgd: Loss Given Default
        wwr_params: Wrong-way risk parameters
        T: Time horizon

    Returns:
        Tuple of (CVA_wwr, CVA_no_wwr, exposure_at_default)
        - CVA_wwr: CVA with wrong-way risk
        - CVA_no_wwr: CVA without wrong-way risk (for comparison)
        - exposure_at_default: Average exposure conditional on default

    Notes:
        CVA_WWR accounts for the correlation between exposure and default.
        The difference (CVA_WWR - CVA_no_wwr) quantifies the WWR charge.
    """
    n_paths, n_steps = exposure_paths.shape
    dt = T / n_steps

    # Simulate correlated defaults
    default_times, default_indicators = simulate_correlated_defaults(
        key,
        exposure_paths,
        hazard_rate,
        wwr_params.correlation,
        T,
        dt
    )

    # Exposure at default (first default time for each path)
    default_indices = jnp.round(default_times / dt).astype(jnp.int32)
    default_indices = jnp.clip(default_indices, 0, n_steps - 1)

    exposure_at_default = exposure_paths[jnp.arange(n_paths), default_indices]

    # Calculate CVA with WWR
    # For each path that defaults, compute discounted loss
    discount_at_default = df_t[default_indices]
    defaulted = default_times < T
    losses = jnp.where(defaulted, exposure_at_default * lgd * discount_at_default, 0.0)

    cva_wwr = float(jnp.mean(losses))

    # CVA without WWR (using EPE)
    epe_t = jnp.maximum(jnp.mean(exposure_paths, axis=0), 0.0)

    # Convert hazard rate to marginal PD
    pd_t = 1.0 - jnp.exp(-hazard_rate * jnp.arange(1, n_steps + 1) * dt)
    dPD = jnp.diff(jnp.concatenate([jnp.array([0.0]), pd_t]))

    cva_no_wwr = float((df_t * epe_t * dPD * lgd).sum())

    return cva_wwr, cva_no_wwr, exposure_at_default


@dataclass
class GaussianCopulaWWR:
    """Gaussian copula model for wrong-way risk.

    Models the joint distribution of exposure and default using a Gaussian copula,
    allowing for flexible correlation structures.

    Attributes:
        correlation_matrix: Correlation matrix between risk factors
        n_factors: Number of common risk factors
    """

    correlation_matrix: Optional[Array] = None
    n_factors: int = 2

    def simulate_joint(
        self,
        key: jax.random.KeyArray,
        n_paths: int,
        n_steps: int,
        exposure_model: Callable,
        credit_model: Callable,
        **model_params
    ) -> Tuple[Array, Array]:
        """Simulate joint exposure and credit scenarios using copula.

        Args:
            key: JAX random key
            n_paths: Number of Monte Carlo paths
            n_steps: Number of time steps
            exposure_model: Function to generate exposure from risk factors
            credit_model: Function to generate credit metrics from risk factors
            **model_params: Additional parameters for models

        Returns:
            Tuple of (exposure_paths, credit_metrics)

        Notes:
            Uses a factor model approach:
            Z_exposure = ρ_1 * F_1 + ρ_2 * F_2 + ε_exposure
            Z_credit = β_1 * F_1 + β_2 * F_2 + ε_credit

            where F_i are common factors and ε are idiosyncratic shocks.
        """
        if self.correlation_matrix is None:
            # Default: moderate positive correlation
            rho = 0.3
            corr = jnp.array([[1.0, rho], [rho, 1.0]])
        else:
            corr = self.correlation_matrix

        # Generate correlated normal random variables
        key1, key2 = jax.random.split(key)

        # Common factors
        factors = jax.random.normal(key1, (n_paths, n_steps, self.n_factors))

        # Idiosyncratic shocks
        key_exp, key_cred = jax.random.split(key2)
        eps_exposure = jax.random.normal(key_exp, (n_paths, n_steps))
        eps_credit = jax.random.normal(key_cred, (n_paths, n_steps))

        # Apply correlation structure
        # Simplified: use first two columns of correlation matrix
        rho_exp = jnp.sqrt(corr[0, 0])
        rho_cred_common = corr[1, 0]
        rho_cred_idio = jnp.sqrt(1.0 - rho_cred_common**2)

        # Correlated risk factors
        z_exposure = rho_exp * factors[:, :, 0] + jnp.sqrt(1.0 - rho_exp**2) * eps_exposure
        z_credit = rho_cred_common * factors[:, :, 0] + rho_cred_idio * eps_credit

        # Transform to exposure and credit metrics
        exposure_paths = exposure_model(z_exposure, **model_params)
        credit_metrics = credit_model(z_credit, **model_params)

        return exposure_paths, credit_metrics


def wwr_adjustment_factor(
    correlation: float,
    volatility_exposure: float,
    volatility_spread: float
) -> float:
    """Calculate WWR adjustment factor for CVA.

    This provides a first-order approximation for the WWR effect on CVA.

    Args:
        correlation: Correlation between exposure and credit spread
        volatility_exposure: Volatility of exposure
        volatility_spread: Volatility of credit spread

    Returns:
        WWR adjustment multiplier (CVA_wwr ≈ CVA × adjustment)

    Notes:
        Based on the formula:
        Adjustment ≈ exp(ρ × σ_E × σ_S)

        where:
        - ρ is the correlation
        - σ_E is exposure volatility
        - σ_S is credit spread volatility

        For negative correlation (right-way risk), adjustment < 1.0
        For positive correlation (wrong-way risk), adjustment > 1.0
    """
    # First-order Taylor expansion
    adjustment = 1.0 + correlation * volatility_exposure * volatility_spread

    # More accurate: exponential form
    adjustment_exp = jnp.exp(correlation * volatility_exposure * volatility_spread)

    # Use exponential for larger adjustments, linear for small
    threshold = 0.1
    magnitude = jnp.abs(correlation * volatility_exposure * volatility_spread)

    final_adjustment = jnp.where(
        magnitude < threshold,
        adjustment,
        adjustment_exp
    )

    return float(final_adjustment)


def specific_wwr_multiplier(
    exposure_to_reference: float,
    total_exposure: float,
    jump_given_default: float = 1.0
) -> float:
    """Calculate specific wrong-way risk multiplier.

    Specific WWR occurs when the exposure is directly tied to the
    counterparty's creditworthiness (e.g., CDS on counterparty's own name,
    put options on counterparty's stock).

    Args:
        exposure_to_reference: Exposure directly tied to counterparty
        total_exposure: Total exposure to counterparty
        jump_given_default: Jump in exposure given counterparty default

    Returns:
        Multiplier for CVA calculation

    Notes:
        For specific WWR, the exposure jumps upon default:
        E_default = E_current × jump_multiplier

        Typical ranges:
        - CDS on own name: jump ~10-20x
        - Put option on own stock: jump ~2-5x
    """
    if total_exposure <= 0:
        return 1.0

    # Fraction of exposure with specific WWR
    wwr_fraction = exposure_to_reference / total_exposure

    # Weighted multiplier
    multiplier = (1.0 - wwr_fraction) + wwr_fraction * jump_given_default

    return float(multiplier)


@dataclass
class WWREngine:
    """Comprehensive wrong-way risk calculation engine.

    Combines multiple WWR modeling approaches:
    - General WWR via correlation modeling
    - Specific WWR via jump-to-default scenarios
    - Copula-based joint simulation
    """

    general_wwr_params: WWRParameters
    specific_wwr_exposure: float = 0.0
    specific_wwr_jump: float = 1.0

    def calculate_cva_adjustment(
        self,
        key: jax.random.KeyArray,
        exposure_paths: Array,
        df_t: Array,
        hazard_rate: float,
        lgd: float,
        T: float
    ) -> dict:
        """Calculate comprehensive CVA with WWR adjustments.

        Args:
            key: JAX random key
            exposure_paths: Monte Carlo exposure paths
            df_t: Discount factors
            hazard_rate: Hazard rate
            lgd: Loss Given Default
            T: Time horizon

        Returns:
            Dictionary with CVA components and WWR adjustments
        """
        # Base CVA with general WWR
        cva_wwr, cva_base, exp_at_default = cva_with_wwr(
            key,
            exposure_paths,
            df_t,
            hazard_rate,
            lgd,
            self.general_wwr_params,
            T
        )

        # Specific WWR adjustment
        total_exposure = jnp.mean(exposure_paths)
        specific_multiplier = specific_wwr_multiplier(
            self.specific_wwr_exposure,
            total_exposure,
            self.specific_wwr_jump
        )

        # Combined CVA
        cva_total = cva_wwr * specific_multiplier

        # WWR charge
        wwr_charge = cva_total - cva_base

        return {
            "cva_base": float(cva_base),
            "cva_general_wwr": float(cva_wwr),
            "cva_total": float(cva_total),
            "wwr_charge": float(wwr_charge),
            "general_wwr_impact": float(cva_wwr - cva_base),
            "specific_wwr_impact": float(cva_total - cva_wwr),
            "specific_wwr_multiplier": float(specific_multiplier),
            "avg_exposure_at_default": float(jnp.mean(exp_at_default)),
        }
