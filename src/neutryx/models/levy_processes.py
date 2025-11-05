"""Advanced Lévy Processes for Equity Modeling.

This module implements time-changed Lévy processes including:
1. Normal Inverse Gaussian (NIG)
2. CGMY (Carr-Geman-Madan-Yor)
3. Generalized time-changed Lévy framework

These models are pure jump processes with infinite activity, capturing
heavy tails, skewness, and kurtosis in return distributions.

References
----------
- Barndorff-Nielsen, O. E. (1997). Normal inverse Gaussian distributions and
  stochastic volatility modelling. Scandinavian Journal of statistics, 24(1), 1-13.
- Carr, P., Geman, H., Madan, D. B., & Yor, M. (2002). The fine structure of asset
  returns: An empirical investigation. The Journal of Business, 75(2), 305-333.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.stats import norm

from ..core.engine import MCConfig
from ..core.utils.math import compute_option_payoff, discount_payoff


# ==============================================================================
# Normal Inverse Gaussian (NIG) Process
# ==============================================================================


@dataclass
class NIGParams:
    """Parameters for Normal Inverse Gaussian (NIG) Lévy process.

    The NIG distribution is obtained by subordinating Brownian motion with drift
    by an inverse Gaussian process. It provides excellent fit to empirical return
    distributions with heavy tails and skewness.

    Attributes
    ----------
    alpha : float
        Tail heaviness parameter (alpha > 0). Larger values = lighter tails
    beta : float
        Skewness parameter (-alpha < beta < alpha)
        - beta > 0: positive skewness (right tail heavier)
        - beta < 0: negative skewness (left tail heavier)
        - beta = 0: symmetric
    delta : float
        Scale parameter (delta > 0). Controls the steepness of the peak
    mu : float
        Location parameter (drift)

    Notes
    -----
    Constraint: alpha^2 > beta^2 must hold.

    The NIG characteristic function is:
    φ(u) = exp(iuμ + δ[√(α² - β²) - √(α² - (β + iu)²)])

    Example parameter sets:
    - Symmetric: alpha=15, beta=0, delta=0.5, mu=0
    - Left skewed (typical equity): alpha=15, beta=-5, delta=0.5, mu=0
    """

    alpha: float  # Tail heaviness (> 0)
    beta: float  # Skewness parameter
    delta: float  # Scale parameter (> 0)
    mu: float = 0.0  # Location parameter

    def __post_init__(self):
        """Validate NIG parameters."""
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        if self.delta <= 0:
            raise ValueError(f"delta must be positive, got {self.delta}")
        if self.alpha**2 <= self.beta**2:
            raise ValueError(
                f"Must have alpha^2 > beta^2, got alpha={self.alpha}, beta={self.beta}"
            )

    def mean_correction(self) -> float:
        """Drift correction to make process a martingale."""
        gamma = jnp.sqrt(self.alpha**2 - self.beta**2)
        return self.delta * self.beta / gamma

    def variance_per_unit_time(self) -> float:
        """Variance of NIG process per unit time."""
        gamma = jnp.sqrt(self.alpha**2 - self.beta**2)
        return self.delta * self.alpha**2 / (gamma**3)


def _nig_increments(
    key: jax.random.KeyArray,
    params: NIGParams,
    dt: float,
    shape: tuple[int, int],
    dtype: jnp.dtype,
) -> Array:
    """Generate NIG process increments using inverse Gaussian subordination.

    NIG is constructed as: X(t) = β*IG(t) + √IG(t) * Z
    where IG(t) is Inverse Gaussian and Z is standard normal.

    Parameters
    ----------
    key : jax.random.KeyArray
        PRNG key
    params : NIGParams
        NIG parameters
    dt : float
        Time step
    shape : tuple[int, int]
        Shape (paths, steps)
    dtype : jnp.dtype
        Data type

    Returns
    -------
    Array
        NIG increments
    """
    key_ig, key_norm = jax.random.split(key)

    # Inverse Gaussian parameters scaled by dt
    # For NIG subordinator: IG(δ*dt/γ, δ*dt) where γ = √(α²-β²)
    # This gives E[T] = δ*dt/γ and matches the NIG mean: μ + δβ/γ
    gamma = jnp.sqrt(params.alpha**2 - params.beta**2)
    ig_mean = params.delta * dt / gamma
    ig_shape = params.delta * dt  # λ parameter

    # Generate inverse Gaussian variates using Michael-Schucany-Haas algorithm
    # IG(μ, λ) generation
    n_samples = shape[0] * shape[1]
    normals = jax.random.normal(key_norm, (n_samples,), dtype=dtype)
    uniforms = jax.random.uniform(key_ig, (n_samples,), dtype=dtype)

    # Generate IG using MSH algorithm
    y = normals**2
    x = ig_mean + (ig_mean**2 * y) / (2 * ig_shape) - (ig_mean / (2 * ig_shape)) * jnp.sqrt(
        4 * ig_mean * ig_shape * y + ig_mean**2 * y**2
    )

    # Accept/reject step
    z = uniforms
    ig_samples = jnp.where(z <= ig_mean / (ig_mean + x), x, ig_mean**2 / x)
    ig_samples = ig_samples.reshape(shape)

    # Generate correlated normals for the Brownian component
    key_norm2 = jax.random.fold_in(key_norm, 1)
    Z = jax.random.normal(key_norm2, shape, dtype=dtype)

    # NIG increments: μ*dt + β*IG + √IG * Z (accounting for time change)
    nig_inc = params.mu * dt + params.beta * ig_samples + jnp.sqrt(jnp.maximum(ig_samples, 1e-10)) * Z

    return nig_inc


def simulate_nig(
    key: jax.random.KeyArray,
    S0: float,
    T: float,
    r: float,
    q: float,
    params: NIGParams,
    cfg: MCConfig,
) -> Array:
    """Simulate asset price paths under Normal Inverse Gaussian model.

    Parameters
    ----------
    key : jax.random.KeyArray
        PRNG key
    S0 : float
        Initial asset price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    q : float
        Dividend yield
    params : NIGParams
        NIG model parameters
    cfg : MCConfig
        Monte Carlo configuration

    Returns
    -------
    Array
        Simulated paths of shape [paths, steps+1]

    Notes
    -----
    The model is:
    S(t) = S(0) * exp((r - q - ω)t + X(t))

    where X(t) is a NIG Lévy process and ω is the martingale correction:
    ω = -log(φ(-i)) where φ is the characteristic function

    Example
    -------
    >>> params = NIGParams(alpha=15.0, beta=-5.0, delta=0.5, mu=0.0)
    >>> key = jax.random.PRNGKey(0)
    >>> cfg = MCConfig(base_paths=10000, steps=252, dtype=jnp.float32)
    >>> paths = simulate_nig(key, 100.0, 1.0, 0.05, 0.01, params, cfg)
    """
    dtype = cfg.dtype
    dt = T / cfg.steps

    # Martingale correction (convexity adjustment)
    # omega such that E[exp(X(1))] = exp(ω)
    # For NIG: E[exp(X)] = exp(μ + δ[√(α²-β²) - √(α²-(β+1)²)])
    # So: ω = μ + δ[√(α²-β²) - √(α²-(β+1)²)]
    gamma = jnp.sqrt(params.alpha**2 - params.beta**2)
    gamma_shifted = jnp.sqrt(params.alpha**2 - (params.beta + 1) ** 2)
    omega = params.mu + params.delta * (gamma - gamma_shifted)

    # Drift per time step
    drift = (r - q - omega) * dt

    # Generate NIG increments
    nig_increments = _nig_increments(key, params, dt, (cfg.base_paths, cfg.steps), dtype)

    # Add drift
    increments = drift + nig_increments

    if cfg.antithetic:
        # Antithetic variates: -X(t) is also NIG with β -> -β
        anti_key = jax.random.fold_in(key, 12345)
        params_anti = NIGParams(
            alpha=params.alpha, beta=-params.beta, delta=params.delta, mu=-params.mu
        )
        # Compute martingale correction for antithetic params
        gamma_anti = jnp.sqrt(params_anti.alpha**2 - params_anti.beta**2)
        gamma_shifted_anti = jnp.sqrt(params_anti.alpha**2 - (params_anti.beta + 1) ** 2)
        omega_anti = params_anti.mu + params_anti.delta * (gamma_anti - gamma_shifted_anti)
        drift_anti = (r - q - omega_anti) * dt

        nig_anti = _nig_increments(anti_key, params_anti, dt, (cfg.base_paths, cfg.steps), dtype)
        increments = jnp.concatenate([increments, drift_anti + nig_anti], axis=0)

    # Build log price paths
    log_S0 = jnp.log(jnp.asarray(S0, dtype=dtype))
    total_paths = increments.shape[0]
    cum_returns = jnp.cumsum(increments, axis=1)
    log_paths = jnp.concatenate(
        [jnp.full((total_paths, 1), log_S0, dtype=dtype), log_S0 + cum_returns],
        axis=1,
    )

    return jnp.exp(log_paths)


def price_vanilla_nig_mc(
    key: jax.random.KeyArray,
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    params: NIGParams,
    cfg: MCConfig,
    kind: str = "call",
) -> float:
    """Price vanilla European option under NIG model using Monte Carlo.

    Parameters
    ----------
    key : jax.random.KeyArray
        PRNG key
    S0 : float
        Initial asset price
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    q : float
        Dividend yield
    params : NIGParams
        NIG model parameters
    cfg : MCConfig
        Monte Carlo configuration
    kind : str, optional
        "call" or "put"

    Returns
    -------
    float
        Option price
    """
    paths = simulate_nig(key, S0, T, r, q, params, cfg)
    ST = paths[:, -1]

    payoffs = compute_option_payoff(ST, K, kind)
    discounted = discount_payoff(payoffs, r, T)
    return float(discounted.mean())


def nig_characteristic_function(
    u: Array | complex, t: float, params: NIGParams
) -> Array | complex:
    """Characteristic function of NIG process at time t.

    φ(u; t) = exp(iuμt + δt[√(α² - β²) - √(α² - (β + iu)²)])

    Parameters
    ----------
    u : Array or complex
        Frequency parameter
    t : float
        Time
    params : NIGParams
        NIG parameters

    Returns
    -------
    Array or complex
        Characteristic function value
    """
    gamma = jnp.sqrt(params.alpha**2 - params.beta**2)
    gamma_shifted = jnp.sqrt(params.alpha**2 - (params.beta + 1j * u) ** 2)
    return jnp.exp(1j * u * params.mu * t + params.delta * t * (gamma - gamma_shifted))


# ==============================================================================
# CGMY (Carr-Geman-Madan-Yor) Process
# ==============================================================================


@dataclass
class CGMYParams:
    """Parameters for CGMY Lévy process.

    The CGMY process is a tempered stable process with separate control over
    small and large jumps. It generalizes the VG process.

    Attributes
    ----------
    C : float
        Overall jump activity (C > 0). Higher C = more jumps
    G : float
        Rate of exponential decay of negative jumps (G > 0)
    M : float
        Rate of exponential decay of positive jumps (M > 0)
    Y : float
        Fine structure parameter (-∞ < Y < 2)
        - Y = 0: finite activity (Kou-like)
        - 0 < Y < 1: infinite activity, finite variation
        - Y = 1: infinite activity, infinite variation (VG-like)
        - 1 < Y < 2: infinite activity, infinite variation (more active)

    Notes
    -----
    The Lévy density is:
    k(x) = C * exp(-G|x|) / |x|^(1+Y)  for x < 0
    k(x) = C * exp(-Mx) / x^(1+Y)      for x > 0

    Constraints:
    - C, G, M > 0
    - Y < 2 for finite variance
    - G, M > 1 when Y ≥ 1 for exponential moments to exist

    Example parameter sets:
    - Moderate activity: C=1.0, G=10.0, M=10.0, Y=0.5
    - High activity (VG-like): C=1.0, G=5.0, M=5.0, Y=1.0
    - Symmetric: G = M
    """

    C: float  # Jump activity
    G: float  # Decay rate for negative jumps
    M: float  # Decay rate for positive jumps
    Y: float  # Fine structure parameter

    def __post_init__(self):
        """Validate CGMY parameters."""
        if self.C <= 0:
            raise ValueError(f"C must be positive, got {self.C}")
        if self.G <= 0:
            raise ValueError(f"G must be positive, got {self.G}")
        if self.M <= 0:
            raise ValueError(f"M must be positive, got {self.M}")
        if self.Y >= 2:
            raise ValueError(f"Y must be < 2 for finite variance, got {self.Y}")
        if self.Y >= 1 and (self.G <= 1 or self.M <= 1):
            raise ValueError(
                f"When Y >= 1, need G > 1 and M > 1 for exp moments. Got G={self.G}, M={self.M}"
            )


def _cgmy_small_jumps(
    key: jax.random.KeyArray,
    params: CGMYParams,
    dt: float,
    shape: tuple[int, int],
    dtype: jnp.dtype,
    jump_threshold: float = 0.01,
) -> Array:
    """Generate small CGMY jumps using series representation.

    For |x| < ε, use series representation and compound Poisson approximation.
    """
    # Approximate small jump activity
    if params.Y < 0:
        # Finite activity - use compound Poisson
        # Lambda = C * [(ε^(-Y) - G^Y) / (G*Y) + ...]
        # Simplified for small jumps
        lam = params.C * dt * 2.0 * jump_threshold ** (-params.Y)
    else:
        # Infinite activity - approximate using diffusion
        # For Y close to 0, behaves like diffusion
        lam = params.C * dt * 10.0  # Heuristic

    # Generate compound Poisson jumps
    key_pois, key_jump = jax.random.split(key)

    n_jumps = jax.random.poisson(key_pois, lam, shape=shape)

    # Jump sizes - mixture of exponentials
    key_dir, key_size = jax.random.split(key_jump)
    jump_dirs = jax.random.bernoulli(key_dir, 0.5, shape=shape)  # Direction
    exp_samples = jax.random.exponential(key_size, shape=shape)

    # Scale jumps
    jump_scale = jump_threshold * 0.5
    jump_sizes = jnp.where(jump_dirs, jump_scale * exp_samples, -jump_scale * exp_samples)

    total_jumps = (n_jumps.astype(dtype) * jump_sizes).astype(dtype)

    return total_jumps


def _cgmy_large_jumps(
    key: jax.random.KeyArray,
    params: CGMYParams,
    dt: float,
    shape: tuple[int, int],
    dtype: jnp.dtype,
    jump_threshold: float = 0.01,
    n_terms: int = 50,
) -> Array:
    """Generate large CGMY jumps using series representation.

    Uses truncated Lévy series representation for large jumps.
    """
    key_gamma, key_exp_pos, key_exp_neg = jax.random.split(key, 3)

    # Generate arrival times using gamma process
    # For large jumps, use explicit series
    jumps = jnp.zeros(shape, dtype=dtype)

    # Positive jumps: use tempered stable representation
    # Generate exponential jumps with rate M
    for _ in range(n_terms // 2):
        key_exp_pos = jax.random.fold_in(key_exp_pos, _)
        exp_jumps = jax.random.exponential(key_exp_pos, shape=shape) / params.M

        # Only keep jumps larger than threshold
        large_jumps = jnp.where(exp_jumps > jump_threshold, exp_jumps, 0.0)
        jumps += large_jumps

    # Negative jumps
    for _ in range(n_terms // 2):
        key_exp_neg = jax.random.fold_in(key_exp_neg, _ + 1000)
        exp_jumps = jax.random.exponential(key_exp_neg, shape=shape) / params.G

        large_jumps = jnp.where(exp_jumps > jump_threshold, exp_jumps, 0.0)
        jumps -= large_jumps

    # Scale by activity and dt
    jumps *= params.C * dt

    return jumps


def simulate_cgmy(
    key: jax.random.KeyArray,
    S0: float,
    T: float,
    r: float,
    q: float,
    params: CGMYParams,
    cfg: MCConfig,
) -> Array:
    """Simulate asset price paths under CGMY model.

    Parameters
    ----------
    key : jax.random.KeyArray
        PRNG key
    S0 : float
        Initial asset price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    q : float
        Dividend yield
    params : CGMYParams
        CGMY model parameters
    cfg : MCConfig
        Monte Carlo configuration

    Returns
    -------
    Array
        Simulated paths of shape [paths, steps+1]

    Notes
    -----
    The implementation uses a hybrid approach:
    1. Small jumps: compound Poisson approximation
    2. Large jumps: truncated series representation

    For production use, consider more sophisticated algorithms:
    - Rosinski's rejection algorithm
    - CGMY characteristic function inversion
    - Fourier methods

    Example
    -------
    >>> params = CGMYParams(C=1.0, G=10.0, M=10.0, Y=0.5)
    >>> key = jax.random.PRNGKey(0)
    >>> cfg = MCConfig(base_paths=10000, steps=252, dtype=jnp.float32)
    >>> paths = simulate_cgmy(key, 100.0, 1.0, 0.05, 0.01, params, cfg)
    """
    dtype = cfg.dtype
    dt = T / cfg.steps

    # Martingale correction using characteristic function
    # ω such that E[exp(X(1))] = exp(ω)
    # For CGMY: ω = C*Γ(-Y)[(M-1)^Y - M^Y + (G+1)^Y - G^Y]
    from jax.scipy.special import gammaln

    if params.Y != 0:
        gamma_factor = jnp.exp(gammaln(-params.Y))
        omega = params.C * gamma_factor * (
            (params.M - 1) ** params.Y
            - params.M**params.Y
            + (params.G + 1) ** params.Y
            - params.G**params.Y
        )
    else:
        # Y = 0 limit
        omega = params.C * (jnp.log(params.M / (params.M - 1)) + jnp.log(params.G / (params.G + 1)))

    drift = (r - q - omega) * dt

    # Split key for small and large jumps
    key_small, key_large = jax.random.split(key)

    # Generate jumps
    small_jumps = _cgmy_small_jumps(key_small, params, dt, (cfg.base_paths, cfg.steps), dtype)
    large_jumps = _cgmy_large_jumps(key_large, params, dt, (cfg.base_paths, cfg.steps), dtype)

    increments = drift + small_jumps + large_jumps

    if cfg.antithetic:
        # For CGMY, antithetic means swapping G and M
        params_anti = CGMYParams(C=params.C, G=params.M, M=params.G, Y=params.Y)
        key_small_anti = jax.random.fold_in(key_small, 9999)
        key_large_anti = jax.random.fold_in(key_large, 9999)

        small_anti = _cgmy_small_jumps(
            key_small_anti, params_anti, dt, (cfg.base_paths, cfg.steps), dtype
        )
        large_anti = _cgmy_large_jumps(
            key_large_anti, params_anti, dt, (cfg.base_paths, cfg.steps), dtype
        )
        increments = jnp.concatenate([increments, drift + small_anti + large_anti], axis=0)

    # Build log price paths
    log_S0 = jnp.log(jnp.asarray(S0, dtype=dtype))
    total_paths = increments.shape[0]
    cum_returns = jnp.cumsum(increments, axis=1)
    log_paths = jnp.concatenate(
        [jnp.full((total_paths, 1), log_S0, dtype=dtype), log_S0 + cum_returns],
        axis=1,
    )

    return jnp.exp(log_paths)


def price_vanilla_cgmy_mc(
    key: jax.random.KeyArray,
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    params: CGMYParams,
    cfg: MCConfig,
    kind: str = "call",
) -> float:
    """Price vanilla European option under CGMY model using Monte Carlo.

    Parameters
    ----------
    key : jax.random.KeyArray
        PRNG key
    S0 : float
        Initial asset price
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    q : float
        Dividend yield
    params : CGMYParams
        CGMY model parameters
    cfg : MCConfig
        Monte Carlo configuration
    kind : str, optional
        "call" or "put"

    Returns
    -------
    float
        Option price
    """
    paths = simulate_cgmy(key, S0, T, r, q, params, cfg)
    ST = paths[:, -1]

    payoffs = compute_option_payoff(ST, K, kind)
    discounted = discount_payoff(payoffs, r, T)
    return float(discounted.mean())


def cgmy_characteristic_function(
    u: Array | complex, t: float, params: CGMYParams
) -> Array | complex:
    """Characteristic function of CGMY process at time t.

    φ(u; t) = exp(t * C * Γ(-Y) * [(M - iu)^Y - M^Y + (G + iu)^Y - G^Y])

    Parameters
    ----------
    u : Array or complex
        Frequency parameter
    t : float
        Time
    params : CGMYParams
        CGMY parameters

    Returns
    -------
    Array or complex
        Characteristic function value
    """
    from jax.scipy.special import gammaln

    gamma_factor = jnp.exp(gammaln(-params.Y))

    term1 = (params.M - 1j * u) ** params.Y - params.M**params.Y
    term2 = (params.G + 1j * u) ** params.Y - params.G**params.Y

    exponent = t * params.C * gamma_factor * (term1 + term2)

    return jnp.exp(exponent)


__all__ = [
    # NIG
    "NIGParams",
    "simulate_nig",
    "price_vanilla_nig_mc",
    "nig_characteristic_function",
    # CGMY
    "CGMYParams",
    "simulate_cgmy",
    "price_vanilla_cgmy_mc",
    "cgmy_characteristic_function",
]
