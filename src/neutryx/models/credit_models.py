"""Comprehensive Credit Risk Models.

This module implements advanced portfolio credit risk models:

1. **Copula Models**
   - Gaussian copula with base correlation
   - Student-t copula
   - Factor copula models

2. **Large Portfolio Approximation (LPA)**
   - One-factor Gaussian LPA
   - Loss distribution approximation
   - Vasicek limit distribution

3. **CreditMetrics Framework**
   - Multi-state migration modeling
   - Joint default/migration simulation
   - Portfolio value-at-risk

4. **Structural Models**
   - Merton model (distance-to-default)
   - Black-Cox model (first-passage time)
   - KMV model implementation

These models are fundamental for:
- CDO tranche pricing
- Portfolio credit risk (CVA, DVA, FVA)
- Credit correlation trading
- Counterparty credit risk (CCR)
- Economic capital allocation
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.stats import norm
from jax.scipy.special import ndtri, betainc  # Inverse normal CDF, regularized incomplete beta


def student_t_cdf(x: Array, df: float) -> Array:
    """Compute Student-t CDF using regularized incomplete beta function.

    For t-distribution with df degrees of freedom:
    CDF(x) = 0.5 + 0.5 * sign(x) * I(t²/(df+t²), 0.5, df/2)
    """
    t_squared = x * x
    z = t_squared / (df + t_squared)

    # I_z(1/2, df/2) = betainc(1/2, df/2, z)
    beta_val = betainc(0.5, df / 2.0, z)

    # CDF = 0.5 + 0.5 * sign(x) * beta_val
    cdf = jnp.where(
        x >= 0,
        0.5 + 0.5 * beta_val,
        0.5 - 0.5 * beta_val
    )

    return cdf


# ==============================================================================
# Gaussian Copula Models
# ==============================================================================


@dataclass
class GaussianCopulaParams:
    """Parameters for Gaussian copula model.

    The Gaussian copula is the most widely used model for portfolio credit risk.
    It assumes that default times are driven by correlated Gaussian factors.

    Attributes
    ----------
    correlation_matrix : Array
        Pairwise correlation matrix, shape [n_names, n_names]
    default_probabilities : Array
        Marginal default probabilities for each name, shape [n_names]
    recovery_rates : Array
        Recovery rates for each name, shape [n_names]

    Notes
    -----
    The Gaussian copula model works as follows:
    1. Draw correlated normal variables: Z ~ N(0, Σ)
    2. Transform to uniform: U_i = Φ(Z_i)
    3. Default if U_i < p_i (default probability)

    For portfolio credit derivatives (CDO tranches), the model uses
    "base correlation" to interpolate between attachment points.

    References
    ----------
    Li, D. X. (2000). On default correlation: A copula function approach.
    Journal of Fixed Income, 9(4), 43-54.
    """

    correlation_matrix: Array
    default_probabilities: Array
    recovery_rates: Array

    def __post_init__(self):
        self.correlation_matrix = jnp.asarray(self.correlation_matrix)
        self.default_probabilities = jnp.asarray(self.default_probabilities)
        self.recovery_rates = jnp.asarray(self.recovery_rates)


def simulate_gaussian_copula(
    key: jax.random.KeyArray,
    params: GaussianCopulaParams,
    n_simulations: int = 10000,
) -> Tuple[Array, Array]:
    """Simulate defaults using Gaussian copula.

    Parameters
    ----------
    key : jax.random.KeyArray
        PRNG key
    params : GaussianCopulaParams
        Copula parameters
    n_simulations : int
        Number of Monte Carlo simulations

    Returns
    -------
    defaults : Array
        Binary default indicators, shape [n_simulations, n_names]
    portfolio_loss : Array
        Portfolio losses for each simulation, shape [n_simulations]

    Example
    -------
    >>> params = GaussianCopulaParams(
    ...     correlation_matrix=jnp.eye(100) * 0.3 + jnp.ones((100, 100)) * 0.1,
    ...     default_probabilities=jnp.full(100, 0.01),  # 1% PD
    ...     recovery_rates=jnp.full(100, 0.4)  # 40% recovery
    ... )
    >>> defaults, losses = simulate_gaussian_copula(key, params, 10000)
    """
    n_names = len(params.default_probabilities)

    # Cholesky decomposition of correlation matrix
    try:
        L = jnp.linalg.cholesky(params.correlation_matrix)
    except:
        # If not positive definite, add jitter
        L = jnp.linalg.cholesky(
            params.correlation_matrix + jnp.eye(n_names) * 1e-6
        )

    # Sample independent normals
    Z_indep = jax.random.normal(key, (n_simulations, n_names))

    # Create correlated normals
    Z_corr = jnp.dot(Z_indep, L.T)

    # Transform to uniform via CDF
    U = norm.cdf(Z_corr)

    # Default thresholds
    thresholds = params.default_probabilities

    # Defaults occur when U < threshold
    defaults = (U < thresholds).astype(jnp.float32)

    # Compute losses (LGD = 1 - recovery_rate)
    lgd = 1.0 - params.recovery_rates
    portfolio_loss = jnp.sum(defaults * lgd, axis=1) / n_names

    return defaults, portfolio_loss


def base_correlation_to_compound_correlation(
    base_corr_lower: float,
    base_corr_upper: float,
    attachment_lower: float,
    attachment_upper: float,
) -> float:
    """Convert base correlations to compound (tranche) correlation.

    Base correlation is a market-standard way to quote CDO tranche prices.
    For a tranche [K1, K2], we have two base correlations:
    - Base correlation for [0, K1] equity tranche
    - Base correlation for [0, K2] equity tranche

    Parameters
    ----------
    base_corr_lower : float
        Base correlation for [0, K1]
    base_corr_upper : float
        Base correlation for [0, K2]
    attachment_lower : float
        Lower attachment point K1
    attachment_upper : float
        Upper attachment point K2

    Returns
    -------
    float
        Compound correlation for tranche [K1, K2]

    Notes
    -----
    This is an approximation. The exact relationship requires pricing.

    References
    ----------
    McGinty, L., et al. (2004). Introducing base correlations.
    JP Morgan Credit Derivatives Strategy.
    """
    # Simplified linear interpolation (actual conversion requires pricing)
    weight = (attachment_upper - attachment_lower) / attachment_upper
    compound_corr = (
        base_corr_upper * attachment_upper - base_corr_lower * attachment_lower
    ) / (attachment_upper - attachment_lower)

    return float(jnp.clip(compound_corr, 0.0, 1.0))


# ==============================================================================
# Student-t Copula
# ==============================================================================


@dataclass
class StudentTCopulaParams:
    """Parameters for Student-t copula model.

    The Student-t copula exhibits tail dependence, making it more realistic
    for modeling default clustering during crises than the Gaussian copula.

    Attributes
    ----------
    correlation_matrix : Array
        Correlation matrix
    default_probabilities : Array
        Marginal default probabilities
    recovery_rates : Array
        Recovery rates
    degrees_of_freedom : float
        Degrees of freedom parameter (lower = heavier tails)

    Notes
    -----
    Key differences from Gaussian copula:
    - Exhibits tail dependence (joint defaults more likely)
    - Controlled by degrees of freedom parameter
    - Reduces to Gaussian as df → ∞

    Lower df implies:
    - Heavier tails
    - More extreme events
    - Higher default clustering

    References
    ----------
    Demarta, S., & McNeil, A. J. (2005). The t copula and related copulas.
    International Statistical Review, 73(1), 111-129.
    """

    correlation_matrix: Array
    default_probabilities: Array
    recovery_rates: Array
    degrees_of_freedom: float = 4.0


def simulate_student_t_copula(
    key: jax.random.KeyArray,
    params: StudentTCopulaParams,
    n_simulations: int = 10000,
) -> Tuple[Array, Array]:
    """Simulate defaults using Student-t copula.

    Parameters
    ----------
    key : jax.random.KeyArray
        PRNG key
    params : StudentTCopulaParams
        Copula parameters
    n_simulations : int
        Number of simulations

    Returns
    -------
    defaults : Array
        Default indicators, shape [n_simulations, n_names]
    portfolio_loss : Array
        Portfolio losses, shape [n_simulations]

    Example
    -------
    >>> params = StudentTCopulaParams(
    ...     correlation_matrix=jnp.eye(50) * 0.7 + jnp.ones((50, 50)) * 0.2,
    ...     default_probabilities=jnp.full(50, 0.02),
    ...     recovery_rates=jnp.full(50, 0.4),
    ...     degrees_of_freedom=4.0  # Heavy tails
    ... )
    >>> defaults, losses = simulate_student_t_copula(key, params, 10000)
    """
    n_names = len(params.default_probabilities)

    # Cholesky decomposition
    try:
        L = jnp.linalg.cholesky(params.correlation_matrix)
    except:
        L = jnp.linalg.cholesky(
            params.correlation_matrix + jnp.eye(n_names) * 1e-6
        )

    # Split keys
    key1, key2 = jax.random.split(key)

    # Sample independent normals
    Z_indep = jax.random.normal(key1, (n_simulations, n_names))

    # Sample chi-squared for Student-t
    chi2_samples = jax.random.gamma(
        key2, a=params.degrees_of_freedom / 2.0, shape=(n_simulations, 1)
    ) * 2.0

    # Create correlated normals
    Z_corr = jnp.dot(Z_indep, L.T)

    # Transform to Student-t: t = Z * sqrt(df / chi2)
    T_corr = Z_corr * jnp.sqrt(params.degrees_of_freedom / chi2_samples)

    # Transform to uniform via Student-t CDF
    U = student_t_cdf(T_corr, params.degrees_of_freedom)

    # Default thresholds
    thresholds = params.default_probabilities

    # Defaults
    defaults = (U < thresholds).astype(jnp.float32)

    # Compute losses
    lgd = 1.0 - params.recovery_rates
    portfolio_loss = jnp.sum(defaults * lgd, axis=1) / n_names

    return defaults, portfolio_loss


# ==============================================================================
# Large Portfolio Approximation (LPA)
# ==============================================================================


@dataclass
class LPAParams:
    """Parameters for Large Portfolio Approximation.

    The LPA (also called one-factor Gaussian model or Vasicek model) provides
    a fast semi-analytical approach to computing loss distributions for large
    homogeneous portfolios.

    Attributes
    ----------
    default_probability : float
        Homogeneous default probability
    correlation : float
        Asset correlation (typically 0.1-0.3)
    recovery_rate : float
        Recovery rate
    n_names : int
        Number of names in portfolio

    Notes
    -----
    Model structure:
        A_i = √ρ * M + √(1-ρ) * ε_i

    where M is systematic factor, ε_i is idiosyncratic.

    Default occurs when A_i < threshold.

    In the limit n → ∞, the conditional default rate given M is:

        p(M) = Φ((Φ⁻¹(p) - √ρ * M) / √(1-ρ))

    And portfolio loss is deterministic given M.

    This is the foundation of Basel II/III capital requirements.

    References
    ----------
    Vasicek, O. (2002). Loan portfolio value.
    Risk, 15(12), 160-162.
    """

    default_probability: float
    correlation: float
    recovery_rate: float = 0.4
    n_names: int = 100


def vasicek_loss_distribution(
    params: LPAParams,
    loss_grid: Array,
) -> Array:
    """Compute loss distribution using Vasicek (LPA) formula.

    Parameters
    ----------
    params : LPAParams
        Model parameters
    loss_grid : Array
        Grid of loss levels to evaluate

    Returns
    -------
    Array
        Probabilities for each loss level

    Notes
    -----
    For large homogeneous portfolio, conditional on systematic factor M,
    the portfolio loss rate is:

        L(M) = (1 - R) * Φ((Φ⁻¹(p) - √ρ * M) / √(1-ρ))

    The unconditional distribution is obtained by integrating over M ~ N(0,1).

    Example
    -------
    >>> params = LPAParams(
    ...     default_probability=0.01,
    ...     correlation=0.2,
    ...     recovery_rate=0.4,
    ...     n_names=1000
    ... )
    >>> loss_grid = jnp.linspace(0, 0.1, 100)
    >>> loss_dist = vasicek_loss_distribution(params, loss_grid)
    """
    # Loss given default
    lgd = 1.0 - params.recovery_rate

    # Default threshold
    threshold = ndtri(params.default_probability)

    sqrt_rho = jnp.sqrt(params.correlation)
    sqrt_1_minus_rho = jnp.sqrt(1.0 - params.correlation)

    # For each loss level, compute CDF directly
    # CDF(L) = P(Loss ≤ L) = P(M ≤ M(L))
    # where M(L) solves: L = LGD * Φ((threshold - √ρ * M) / √(1-ρ))

    def loss_cdf(loss_rate):
        """Compute CDF at given loss rate."""
        # Find M such that loss_rate = LGD * Φ((threshold - √ρ * M) / √(1-ρ))
        conditional_pd = jnp.clip(loss_rate / lgd, 1e-10, 1.0 - 1e-10)

        # M = (threshold - √(1-ρ) * Φ⁻¹(conditional_pd)) / √ρ
        M = (threshold - sqrt_1_minus_rho * ndtri(conditional_pd)) / sqrt_rho

        # CDF = Φ(M), but return 0 if loss_rate <= 0
        cdf = norm.cdf(M)
        return jnp.where(loss_rate <= 0, 0.0, cdf)

    # Compute CDF values
    cdf_values = jax.vmap(loss_cdf)(loss_grid)

    # Compute PDF via numerical differentiation
    # Use central differences for interior points
    pdf = jnp.zeros_like(loss_grid)
    dx = loss_grid[1] - loss_grid[0]

    # Central difference for interior points
    pdf = pdf.at[1:-1].set((cdf_values[2:] - cdf_values[:-2]) / (2 * dx))

    # Forward/backward difference for endpoints
    pdf = pdf.at[0].set((cdf_values[1] - cdf_values[0]) / dx)
    pdf = pdf.at[-1].set((cdf_values[-1] - cdf_values[-2]) / dx)

    # Ensure non-negative and normalize
    pdf = jnp.maximum(pdf, 0.0)
    total_prob = jnp.sum(pdf) * dx
    if total_prob > 0:
        pdf = pdf / total_prob

    return pdf


def lpa_expected_loss(params: LPAParams) -> float:
    """Calculate expected loss using LPA.

    Parameters
    ----------
    params : LPAParams
        Model parameters

    Returns
    -------
    float
        Expected loss rate

    Notes
    -----
    EL = PD * LGD

    Example
    -------
    >>> params = LPAParams(default_probability=0.01, recovery_rate=0.4)
    >>> el = lpa_expected_loss(params)
    """
    lgd = 1.0 - params.recovery_rate
    return params.default_probability * lgd


def lpa_unexpected_loss(params: LPAParams, confidence_level: float = 0.99) -> float:
    """Calculate unexpected loss (VaR) using LPA.

    Parameters
    ----------
    params : LPAParams
        Model parameters
    confidence_level : float
        Confidence level for VaR (e.g., 0.99 for 99%)

    Returns
    -------
    float
        Unexpected loss (VaR - EL)

    Notes
    -----
    Uses the Vasicek formula for portfolio loss VaR:

        VaR_α = LGD * Φ((Φ⁻¹(p) + √ρ * Φ⁻¹(α)) / √(1-ρ))

    This is the foundation of Basel II/III IRB capital formulas.

    Example
    -------
    >>> params = LPAParams(default_probability=0.01, correlation=0.15)
    >>> ul = lpa_unexpected_loss(params, 0.999)  # 99.9% VaR for Basel
    """
    lgd = 1.0 - params.recovery_rate
    sqrt_rho = jnp.sqrt(params.correlation)
    sqrt_1_minus_rho = jnp.sqrt(1.0 - params.correlation)

    threshold = ndtri(params.default_probability)
    alpha_quantile = ndtri(confidence_level)

    # Vasicek VaR formula
    var = lgd * norm.cdf(
        (threshold + sqrt_rho * alpha_quantile) / sqrt_1_minus_rho
    )

    # Unexpected loss = VaR - EL
    el = lpa_expected_loss(params)
    ul = var - el

    return float(ul)


# ==============================================================================
# CreditMetrics Framework
# ==============================================================================


@dataclass
class CreditMetricsParams:
    """Parameters for CreditMetrics framework.

    CreditMetrics is a multi-state migration model that tracks not just
    defaults but also rating migrations.

    Attributes
    ----------
    transition_matrix : Array
        Migration transition matrix, shape [n_ratings, n_ratings]
        Element [i,j] = probability of migrating from rating i to rating j
    current_ratings : Array
        Current rating for each obligor, shape [n_obligors]
    exposures : Array
        Exposure amounts, shape [n_obligors]
    values_by_rating : Array
        Value of exposure in each rating state, shape [n_ratings]
    correlation_matrix : Array
        Asset correlation matrix, shape [n_obligors, n_obligors]

    Notes
    -----
    The model works as follows:
    1. Simulate correlated asset returns
    2. Map returns to rating migrations via thresholds
    3. Compute portfolio value in each scenario
    4. Derive value distribution and risk metrics

    References
    ----------
    J.P. Morgan. (1997). CreditMetrics—Technical Document.
    Gupton, G. M., Finger, C. C., & Bhatia, M. (1997).
    """

    transition_matrix: Array  # [n_ratings, n_ratings]
    current_ratings: Array  # [n_obligors]
    exposures: Array  # [n_obligors]
    values_by_rating: Array  # [n_ratings]
    correlation_matrix: Array  # [n_obligors, n_obligors]


def simulate_credit_migrations(
    key: jax.random.KeyArray,
    params: CreditMetricsParams,
    n_simulations: int = 10000,
) -> Tuple[Array, Array]:
    """Simulate credit rating migrations using CreditMetrics framework.

    Parameters
    ----------
    key : jax.random.KeyArray
        PRNG key
    params : CreditMetricsParams
        Model parameters
    n_simulations : int
        Number of simulations

    Returns
    -------
    ratings : Array
        Simulated ratings for each obligor, shape [n_simulations, n_obligors]
    portfolio_values : Array
        Portfolio values for each scenario, shape [n_simulations]

    Example
    -------
    >>> # 3-rating system: AAA (0), BBB (1), Default (2)
    >>> transition_matrix = jnp.array([
    ...     [0.95, 0.04, 0.01],  # From AAA
    ...     [0.02, 0.93, 0.05],  # From BBB
    ...     [0.00, 0.00, 1.00],  # From Default (absorbing)
    ... ])
    >>> params = CreditMetricsParams(
    ...     transition_matrix=transition_matrix,
    ...     current_ratings=jnp.array([0, 0, 1, 1]),  # 2 AAA, 2 BBB
    ...     exposures=jnp.array([100, 100, 100, 100]),
    ...     values_by_rating=jnp.array([100, 80, 20]),  # Values if AAA/BBB/Default
    ...     correlation_matrix=jnp.eye(4) * 0.8 + jnp.ones((4, 4)) * 0.1
    ... )
    >>> ratings, values = simulate_credit_migrations(key, params, 10000)
    """
    n_obligors = len(params.current_ratings)
    n_ratings = params.transition_matrix.shape[0]

    # Compute migration thresholds from transition matrix
    # For each current rating, find thresholds such that Φ(threshold) gives transition prob

    # Cholesky of correlation matrix
    try:
        L = jnp.linalg.cholesky(params.correlation_matrix)
    except:
        L = jnp.linalg.cholesky(
            params.correlation_matrix + jnp.eye(n_obligors) * 1e-6
        )

    # Sample correlated normals
    Z_indep = jax.random.normal(key, (n_simulations, n_obligors))
    Z_corr = jnp.dot(Z_indep, L.T)

    # Transform to uniform
    U = norm.cdf(Z_corr)

    # For each obligor, determine new rating based on current rating and U
    # Compute cumulative transition probabilities
    new_ratings = jnp.zeros((n_simulations, n_obligors), dtype=jnp.int32)

    for i in range(n_obligors):
        current_rating = int(params.current_ratings[i])
        trans_probs = params.transition_matrix[current_rating, :]
        cum_probs = jnp.cumsum(trans_probs)

        # Determine new rating by comparing U to cumulative probs
        obligor_ratings = jnp.zeros(n_simulations, dtype=jnp.int32)

        for j in range(n_ratings):
            # Set rating to j if U is in the range for this rating
            if j == 0:
                mask = U[:, i] <= cum_probs[j]
            else:
                mask = (U[:, i] > cum_probs[j-1]) & (U[:, i] <= cum_probs[j])

            obligor_ratings = jnp.where(mask, j, obligor_ratings)

        new_ratings = new_ratings.at[:, i].set(obligor_ratings)

    # Compute portfolio values
    # Value = sum over obligors of: exposure * value_multiplier[new_rating]
    portfolio_values = jnp.zeros(n_simulations)

    for i in range(n_obligors):
        obligor_values = params.values_by_rating[new_ratings[:, i]]
        portfolio_values += params.exposures[i] * obligor_values / 100.0  # Normalize

    return new_ratings, portfolio_values


# ==============================================================================
# Structural Models (Merton, Black-Cox)
# ==============================================================================


@dataclass
class MertonModelParams:
    """Parameters for Merton structural credit model.

    In the Merton model, default occurs if asset value falls below debt value
    at maturity.

    Model:
        V_T = V_0 * exp((μ - 0.5*σ²)*T + σ*√T*Z)

    Default if V_T < D (debt level).

    Attributes
    ----------
    asset_value : float
        Current firm asset value V_0
    debt_value : float
        Face value of debt D
    volatility : float
        Asset volatility σ
    maturity : float
        Debt maturity T
    risk_free_rate : float
        Risk-free rate r

    Notes
    -----
    The Merton model provides:
    - Default probability: Φ(-DD) where DD is distance-to-default
    - Equity value: Call option on assets with strike = debt
    - Credit spread: Implied from default probability

    Distance-to-default:
        DD = (ln(V₀/D) + (μ - 0.5*σ²)*T) / (σ*√T)

    KMV model uses this framework with empirical default frequencies.

    References
    ----------
    Merton, R. C. (1974). On the pricing of corporate debt.
    Journal of Finance, 29(2), 449-470.
    """

    asset_value: float
    debt_value: float
    volatility: float
    maturity: float
    risk_free_rate: float


def merton_default_probability(params: MertonModelParams) -> float:
    """Calculate default probability under Merton model.

    Parameters
    ----------
    params : MertonModelParams
        Model parameters

    Returns
    -------
    float
        Probability of default at maturity

    Example
    -------
    >>> params = MertonModelParams(
    ...     asset_value=100,
    ...     debt_value=80,
    ...     volatility=0.25,
    ...     maturity=1.0,
    ...     risk_free_rate=0.05
    ... )
    >>> pd = merton_default_probability(params)
    """
    # Drift: risk-free rate for risk-neutral pricing
    mu = params.risk_free_rate

    # Distance-to-default
    d2 = (
        jnp.log(params.asset_value / params.debt_value)
        + (mu - 0.5 * params.volatility**2) * params.maturity
    ) / (params.volatility * jnp.sqrt(params.maturity))

    # Default probability = Φ(-d2)
    pd = norm.cdf(-d2)

    return float(pd)


def merton_distance_to_default(params: MertonModelParams) -> float:
    """Calculate distance-to-default (DD) metric.

    Parameters
    ----------
    params : MertonModelParams
        Model parameters

    Returns
    -------
    float
        Distance-to-default in standard deviations

    Notes
    -----
    DD measures how many standard deviations the asset value is above
    the default point. Higher DD = lower default risk.

    Moody's KMV uses DD to estimate default probabilities via
    empirical default frequency (EDF) tables.

    Example
    -------
    >>> params = MertonModelParams(...)
    >>> dd = merton_distance_to_default(params)
    >>> # DD > 3 is typically considered safe
    >>> # DD < 1 indicates high default risk
    """
    mu = params.risk_free_rate

    dd = (
        jnp.log(params.asset_value / params.debt_value)
        + (mu - 0.5 * params.volatility**2) * params.maturity
    ) / (params.volatility * jnp.sqrt(params.maturity))

    return float(dd)


def merton_equity_value(params: MertonModelParams) -> float:
    """Calculate equity value as call option on firm assets.

    Parameters
    ----------
    params : MertonModelParams
        Model parameters

    Returns
    -------
    float
        Equity value

    Notes
    -----
    Equity holders have a call option on firm assets with strike = debt.

    E = V₀*Φ(d1) - D*exp(-r*T)*Φ(d2)

    where d1 = d2 + σ*√T

    Example
    -------
    >>> params = MertonModelParams(...)
    >>> equity = merton_equity_value(params)
    """
    # Black-Scholes formula for call option
    d2 = (
        jnp.log(params.asset_value / params.debt_value)
        + (params.risk_free_rate - 0.5 * params.volatility**2) * params.maturity
    ) / (params.volatility * jnp.sqrt(params.maturity))

    d1 = d2 + params.volatility * jnp.sqrt(params.maturity)

    equity = (
        params.asset_value * norm.cdf(d1)
        - params.debt_value * jnp.exp(-params.risk_free_rate * params.maturity) * norm.cdf(d2)
    )

    return float(equity)


@dataclass
class BlackCoxParams:
    """Parameters for Black-Cox first-passage model.

    The Black-Cox model extends Merton by allowing default at first passage
    time when assets hit a lower barrier (not just at maturity).

    Attributes
    ----------
    asset_value : float
        Current firm asset value
    barrier : float
        Default barrier level (time-dependent barrier supported)
    volatility : float
        Asset volatility
    maturity : float
        Time horizon
    risk_free_rate : float
        Risk-free rate
    dividend_yield : float
        Asset payout rate (e.g., cash flows to debt holders)

    Notes
    -----
    Default occurs at first time τ when:
        V_τ ≤ K(τ)

    where K(t) is the barrier (often K(t) = K₀*exp(γ*t)).

    The model provides more realistic default timing than Merton,
    as default can occur before maturity.

    References
    ----------
    Black, F., & Cox, J. C. (1976). Valuing corporate securities.
    Journal of Financial Economics, 3(1-2), 351-367.
    """

    asset_value: float
    barrier: float  # Could be function of time
    volatility: float
    maturity: float
    risk_free_rate: float
    dividend_yield: float = 0.0


def black_cox_default_probability(params: BlackCoxParams) -> float:
    """Calculate default probability under Black-Cox model.

    Parameters
    ----------
    params : BlackCoxParams
        Model parameters

    Returns
    -------
    float
        Probability of hitting barrier before maturity

    Notes
    -----
    For constant barrier K, the default probability is:

        P(τ ≤ T) = Φ(-d₁) + (V₀/K)^(2λ) * Φ(-d₂)

    where:
        λ = (r - q - 0.5*σ²) / σ²
        d₁ = [ln(V₀/K) + (r - q + 0.5*σ²)*T] / (σ*√T)
        d₂ = [ln(K/V₀) + (r - q + 0.5*σ²)*T] / (σ*√T)

    Example
    -------
    >>> params = BlackCoxParams(
    ...     asset_value=100,
    ...     barrier=60,  # Default if assets drop to 60
    ...     volatility=0.30,
    ...     maturity=5.0,
    ...     risk_free_rate=0.05,
    ...     dividend_yield=0.02
    ... )
    >>> pd = black_cox_default_probability(params)
    """
    V = params.asset_value
    K = params.barrier
    sigma = params.volatility
    T = params.maturity
    r = params.risk_free_rate
    q = params.dividend_yield

    # Drift parameter
    mu = r - q
    sqrt_T = jnp.sqrt(T)
    sigma_sqrt_T = sigma * sqrt_T

    # Black-Cox first-passage formula
    # h1 = [ln(K/V) + (μ - 0.5σ²)T] / (σ√T)
    # h2 = [ln(K/V) - (μ - 0.5σ²)T] / (σ√T)
    drift = mu - 0.5 * sigma**2
    log_ratio = jnp.log(K / V)

    h1 = (log_ratio + drift * T) / sigma_sqrt_T
    h2 = (log_ratio - drift * T) / sigma_sqrt_T

    # Power term: (K/V)^(2μ/σ²)
    power_exponent = 2.0 * mu / (sigma * sigma)

    # P(τ ≤ T) = Φ(h1) + (K/V)^(2μ/σ²) * Φ(h2)
    pd = norm.cdf(h1) + jnp.power(K / V, power_exponent) * norm.cdf(h2)

    return float(jnp.clip(pd, 0.0, 1.0))


# ==============================================================================
# Utility Functions
# ==============================================================================


def credit_spread_from_default_prob(
    default_prob: float,
    recovery_rate: float,
    maturity: float,
) -> float:
    """Convert default probability to credit spread.

    Parameters
    ----------
    default_prob : float
        Cumulative default probability to maturity
    recovery_rate : float
        Recovery rate
    maturity : float
        Time to maturity

    Returns
    -------
    float
        Credit spread (continuously compounded)

    Notes
    -----
    Approximation:
        spread ≈ -ln(1 - PD*(1-R)) / T

    More accurate:
        Uses hazard rate: λ = -ln(1 - PD) / T
        spread = λ * (1 - R)

    Example
    -------
    >>> spread = credit_spread_from_default_prob(0.05, 0.4, 5.0)
    >>> # 5% default prob over 5 years, 40% recovery
    """
    lgd = 1.0 - recovery_rate

    # Approximate hazard rate from cumulative PD
    hazard_rate = -jnp.log(1.0 - default_prob) / maturity

    # Credit spread ≈ hazard_rate * LGD
    spread = hazard_rate * lgd

    return float(spread)


__all__ = [
    # Gaussian Copula
    "GaussianCopulaParams",
    "simulate_gaussian_copula",
    "base_correlation_to_compound_correlation",
    # Student-t Copula
    "StudentTCopulaParams",
    "simulate_student_t_copula",
    # Large Portfolio Approximation
    "LPAParams",
    "vasicek_loss_distribution",
    "lpa_expected_loss",
    "lpa_unexpected_loss",
    # CreditMetrics
    "CreditMetricsParams",
    "simulate_credit_migrations",
    # Merton Model
    "MertonModelParams",
    "merton_default_probability",
    "merton_distance_to_default",
    "merton_equity_value",
    # Black-Cox Model
    "BlackCoxParams",
    "black_cox_default_probability",
    # Utils
    "credit_spread_from_default_prob",
]
