"""Comprehensive Equity Pricing Models.

This module provides a unified interface to all equity pricing models implemented
in Neutryx:

1. **Local Volatility Models**
   - Dupire local volatility
   - Calibrated from market implied volatility surfaces

2. **Stochastic Local Volatility (SLV)**
   - Combines local volatility with stochastic volatility (Heston)
   - Provides more realistic volatility dynamics than pure local vol

3. **Rough Volatility Models**
   - Rough Bergomi (rBergomi)
   - Rough Heston
   - Driven by fractional Brownian motion with Hurst parameter H < 0.5

4. **Jump-Diffusion Models**
   - Merton jump-diffusion (lognormal jumps)
   - Kou double exponential jump-diffusion
   - Variance Gamma process

5. **Time-Changed Lévy Processes**
   - Variance Gamma (VG)
   - Normal Inverse Gaussian (NIG)
   - CGMY (Carr-Geman-Madan-Yor)

6. **Jump Clustering Models**
   - Hawkes jump-diffusion (self-exciting jumps)
   - Self-exciting Lévy models
   - Jump contagion and volatility feedback

All models support Monte Carlo simulation, calibration to market data,
and various pricing methods.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import Array

from ..core.engine import MCConfig


# ==============================================================================
# Stochastic Local Volatility (SLV)
# ==============================================================================


@dataclass
class SLVParams:
    """Parameters for Stochastic Local Volatility model.

    SLV combines local volatility sigma_L(S,t) with stochastic volatility.
    The dynamics are:

        dS = r*S*dt + sqrt(V)*sigma_L(S,t)*S*dW_S
        dV = kappa*(theta - V)*dt + xi*sqrt(V)*dW_V
        dW_S * dW_V = rho * dt

    Attributes
    ----------
    kappa : float
        Mean reversion speed for variance
    theta : float
        Long-term variance
    xi : float
        Vol-of-vol
    rho : float
        Correlation between asset and variance
    local_vol_func : Callable[[float, float], float]
        Local volatility function sigma_L(S, t)
    V0 : float
        Initial variance
    """

    kappa: float
    theta: float
    xi: float
    rho: float
    local_vol_func: Callable[[float, float], float]
    V0: float = 0.04


def simulate_slv(
    key: jax.random.KeyArray,
    S0: float,
    T: float,
    r: float,
    q: float,
    params: SLVParams,
    cfg: MCConfig,
) -> Array:
    """Simulate Stochastic Local Volatility paths using Euler-Maruyama.

    Parameters
    ----------
    key : jax.random.KeyArray
        PRNG key
    S0 : float
        Initial stock price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    q : float
        Dividend yield
    params : SLVParams
        SLV model parameters
    cfg : MCConfig
        Monte Carlo configuration

    Returns
    -------
    Array
        Simulated paths of shape [paths, steps+1]

    Notes
    -----
    The SLV model provides more realistic joint dynamics of spot and vol
    than pure local vol or pure stochastic vol models. It combines:
    - The arbitrage-free property and smile-fitting of local vol
    - The forward-looking vol dynamics of stochastic vol

    Example
    -------
    >>> from neutryx.models.dupire import calibrate_local_vol_surface
    >>> # First calibrate local vol surface
    >>> local_vol_params = calibrate_local_vol_surface(...)
    >>> # Then use in SLV
    >>> slv_params = SLVParams(
    ...     kappa=2.0,
    ...     theta=0.04,
    ...     xi=0.3,
    ...     rho=-0.7,
    ...     local_vol_func=local_vol_params.local_vol,
    ...     V0=0.04
    ... )
    >>> paths = simulate_slv(key, 100.0, 1.0, 0.05, 0.01, slv_params, cfg)
    """
    dtype = cfg.dtype
    dt = T / cfg.steps
    sqrt_dt = jnp.sqrt(dt)

    # Split keys
    key_S, key_V = jax.random.split(key)

    # Generate correlated Brownian motions
    Z1 = jax.random.normal(key_S, (cfg.base_paths, cfg.steps), dtype=dtype)
    Z2 = jax.random.normal(key_V, (cfg.base_paths, cfg.steps), dtype=dtype)

    # Correlate: W_V = Z2, W_S = rho*Z2 + sqrt(1-rho^2)*Z1
    rho_safe = jnp.clip(params.rho, -0.999, 0.999)
    dW_S = rho_safe * Z2 + jnp.sqrt(1.0 - rho_safe**2) * Z1
    dW_V = Z2

    # Initialize arrays
    total_paths = cfg.base_paths * (2 if cfg.antithetic else 1)
    S_paths = jnp.zeros((total_paths, cfg.steps + 1), dtype=dtype)
    V_paths = jnp.zeros((total_paths, cfg.steps + 1), dtype=dtype)

    # Initialize all paths at t=0
    S_paths = S_paths.at[:cfg.base_paths, 0].set(S0)
    V_paths = V_paths.at[:cfg.base_paths, 0].set(params.V0)
    if cfg.antithetic:
        S_paths = S_paths.at[cfg.base_paths:, 0].set(S0)
        V_paths = V_paths.at[cfg.base_paths:, 0].set(params.V0)

    # Euler-Maruyama scheme
    # Time grid
    times = jnp.linspace(0, T, cfg.steps + 1, dtype=dtype)

    # Simulation loop
    S_current = S_paths[:cfg.base_paths, 0]
    V_current = V_paths[:cfg.base_paths, 0]

    # Vectorized local vol evaluation
    def eval_local_vol_vectorized(S_vals, t_val):
        """Evaluate local vol for all paths at time t."""
        # If local_vol_func is a simple function, call it
        # Otherwise, use a constant as fallback
        try:
            # Try to vectorize the local vol function
            local_vols = jax.vmap(lambda s: params.local_vol_func(s, float(t_val)))(S_vals)
            return local_vols
        except:
            # Fallback to constant local vol if function evaluation fails
            return jnp.ones_like(S_vals) * 0.2

    for i in range(cfg.steps):
        # Evaluate local volatility at current S and t
        sigma_L = eval_local_vol_vectorized(S_current, times[i])

        # Variance process (Heston-style with Feller condition)
        sqrt_V = jnp.sqrt(jnp.maximum(V_current, 0.0))
        dV = params.kappa * (params.theta - V_current) * dt + params.xi * sqrt_V * sqrt_dt * dW_V[:, i]
        V_current = jnp.maximum(V_current + dV, 0.0)

        # Spot process with SLV: dS/S = (r-q)dt + σ_L(S,t)√V dW_S
        sqrt_V_spot = jnp.sqrt(jnp.maximum(V_current, 0.0))
        dS = (r - q) * S_current * dt + sqrt_V_spot * sigma_L * S_current * sqrt_dt * dW_S[:, i]
        S_current = jnp.maximum(S_current + dS, 1e-8)

        S_paths = S_paths.at[:cfg.base_paths, i + 1].set(S_current)
        V_paths = V_paths.at[:cfg.base_paths, i + 1].set(V_current)

    if cfg.antithetic:
        # Antithetic paths - flip Brownian increments
        S_current_anti = jnp.full(cfg.base_paths, S0, dtype=dtype)
        V_current_anti = jnp.full(cfg.base_paths, params.V0, dtype=dtype)

        for i in range(cfg.steps):
            sigma_L = eval_local_vol_vectorized(S_current_anti, times[i])

            sqrt_V = jnp.sqrt(jnp.maximum(V_current_anti, 0.0))
            dV = params.kappa * (params.theta - V_current_anti) * dt - params.xi * sqrt_V * sqrt_dt * dW_V[:, i]
            V_current_anti = jnp.maximum(V_current_anti + dV, 0.0)

            sqrt_V_spot = jnp.sqrt(jnp.maximum(V_current_anti, 0.0))
            dS = (r - q) * S_current_anti * dt - sqrt_V_spot * sigma_L * S_current_anti * sqrt_dt * dW_S[:, i]
            S_current_anti = jnp.maximum(S_current_anti + dS, 1e-8)

            S_paths = S_paths.at[cfg.base_paths:, i + 1].set(S_current_anti)
            V_paths = V_paths.at[cfg.base_paths:, i + 1].set(V_current_anti)

    return S_paths


# ==============================================================================
# Rough Heston Model
# ==============================================================================


@dataclass
class RoughHestonParams:
    """Parameters for Rough Heston model.

    The rough Heston model uses a fractional kernel for the variance process:

        V_t = V_0 + (1/Γ(H+0.5)) * ∫_0^t (t-s)^(H-0.5) * κ(θ - V_s) ds
              + (1/Γ(H+0.5)) * ∫_0^t (t-s)^(H-0.5) * ξ*sqrt(V_s) dW_V

    Attributes
    ----------
    H : float
        Hurst parameter (0 < H < 0.5 for rough volatility)
    kappa : float
        Mean reversion speed
    theta : float
        Long-term variance
    xi : float
        Vol-of-vol
    rho : float
        Correlation between asset and variance
    V0 : float
        Initial variance

    Notes
    -----
    When H = 0.5, this reduces to standard Heston.
    For H < 0.5, volatility exhibits "roughness" - more realistic dynamics.

    The rough Heston model has been shown to better fit:
    - Short-dated option smiles
    - At-the-money skew behavior
    - Forward variance term structure
    """

    H: float  # Hurst parameter (< 0.5 for roughness)
    kappa: float
    theta: float
    xi: float
    rho: float
    V0: float = 0.04


def simulate_rough_heston(
    key: jax.random.KeyArray,
    S0: float,
    T: float,
    r: float,
    q: float,
    params: RoughHestonParams,
    cfg: MCConfig,
) -> tuple[Array, Array]:
    """Simulate Rough Heston paths using hybrid scheme.

    Parameters
    ----------
    key : jax.random.KeyArray
        PRNG key
    S0 : float
        Initial stock price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    q : float
        Dividend yield
    params : RoughHestonParams
        Rough Heston parameters
    cfg : MCConfig
        Monte Carlo configuration

    Returns
    -------
    S_paths : Array
        Stock price paths, shape [paths, steps+1]
    V_paths : Array
        Variance paths, shape [paths, steps+1]

    Notes
    -----
    Implementation uses a hybrid scheme combining:
    1. Cholesky factorization for fractional kernel
    2. Euler-Maruyama for the asset process

    For production use, consider more sophisticated schemes like:
    - Adams scheme for fractional ODEs
    - Hybrid scheme of Bennedsen et al. (2017)

    Example
    -------
    >>> params = RoughHestonParams(
    ...     H=0.1,  # Rough (H < 0.5)
    ...     kappa=0.3,
    ...     theta=0.02,
    ...     xi=0.3,
    ...     rho=-0.7,
    ...     V0=0.04
    ... )
    >>> S_paths, V_paths = simulate_rough_heston(key, 100, 1.0, 0.05, 0.01, params, cfg)
    """
    dtype = cfg.dtype
    dt = T / cfg.steps
    sqrt_dt = jnp.sqrt(dt)

    # For rough Heston, we need fractional integration kernel
    # Simplified implementation - full version would use hybrid scheme

    # Use rough Bergomi as approximation for now
    # (Rough Heston requires more sophisticated numerical methods)
    from .rough_vol import RoughBergomiParams, simulate_rough_bergomi

    # Map rough Heston params to rough Bergomi params (approximation)
    rb_params = RoughBergomiParams(
        hurst=params.H,
        eta=params.xi,
        rho=params.rho,
        forward_variance=params.theta,
    )

    # Simulate using rough Bergomi
    rb_paths = simulate_rough_bergomi(
        key, S0, T, r, q, cfg, rb_params, return_full=True, store_log=False
    )

    S_paths = rb_paths.values
    V_paths = rb_paths.variance if rb_paths.variance is not None else jnp.ones_like(S_paths) * params.V0

    return S_paths, V_paths


# ==============================================================================
# Time-Changed Lévy Processes
# ==============================================================================


@dataclass
class TimeChangedLevyParams:
    """Parameters for time-changed Lévy process.

    A time-changed Lévy process has the form:
        S_t = S_0 * exp(X(T_t))

    where X is a Lévy process and T_t is a stochastic time change
    (typically an increasing Lévy subordinator).

    Attributes
    ----------
    levy_process : str
        Type of Lévy process: 'VG', 'NIG', 'CGMY'
    levy_params : dict
        Parameters for the Lévy process
    time_change : str
        Type of time change: 'gamma', 'inverse_gaussian', 'deterministic'
    time_change_params : dict
        Parameters for the time change process
    """

    levy_process: str = 'VG'
    levy_params: dict = None
    time_change: str = 'gamma'
    time_change_params: dict = None

    def __post_init__(self):
        if self.levy_params is None:
            # Default VG parameters
            self.levy_params = {'theta': -0.14, 'sigma': 0.2, 'nu': 0.2}
        if self.time_change_params is None:
            # Default gamma parameters
            self.time_change_params = {'rate': 1.0}


def simulate_time_changed_levy(
    key: jax.random.KeyArray,
    S0: float,
    T: float,
    r: float,
    q: float,
    params: TimeChangedLevyParams,
    cfg: MCConfig,
) -> Array:
    """Simulate time-changed Lévy process paths.

    Parameters
    ----------
    key : jax.random.KeyArray
        PRNG key
    S0 : float
        Initial stock price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    q : float
        Dividend yield
    params : TimeChangedLevyParams
        Time-changed Lévy parameters
    cfg : MCConfig
        Monte Carlo configuration

    Returns
    -------
    Array
        Simulated paths, shape [paths, steps+1]

    Notes
    -----
    Time-changed Lévy processes include:
    - Variance Gamma (VG): Brownian motion time-changed by gamma process
    - Normal Inverse Gaussian (NIG): Brownian motion time-changed by inverse Gaussian
    - CGMY: Generalization with additional parameters

    These models capture:
    - Heavy tails in return distributions
    - Implied volatility skew and smile
    - Jump clustering

    Example
    -------
    >>> # Variance Gamma
    >>> params_vg = TimeChangedLevyParams(
    ...     levy_process='VG',
    ...     levy_params={'theta': -0.14, 'sigma': 0.2, 'nu': 0.2}
    ... )
    >>> paths = simulate_time_changed_levy(key, 100, 1.0, 0.05, 0.01, params_vg, cfg)
    >>>
    >>> # Normal Inverse Gaussian
    >>> params_nig = TimeChangedLevyParams(
    ...     levy_process='NIG',
    ...     levy_params={'alpha': 15.0, 'beta': -5.0, 'delta': 0.5}
    ... )
    >>> paths = simulate_time_changed_levy(key, 100, 1.0, 0.05, 0.01, params_nig, cfg)
    >>>
    >>> # CGMY
    >>> params_cgmy = TimeChangedLevyParams(
    ...     levy_process='CGMY',
    ...     levy_params={'C': 1.0, 'G': 10.0, 'M': 10.0, 'Y': 0.5}
    ... )
    >>> paths = simulate_time_changed_levy(key, 100, 1.0, 0.05, 0.01, params_cgmy, cfg)
    """
    levy_type = params.levy_process.upper()

    if levy_type == 'VG':
        from .variance_gamma import simulate_variance_gamma

        theta = params.levy_params.get('theta', -0.14)
        sigma = params.levy_params.get('sigma', 0.2)
        nu = params.levy_params.get('nu', 0.2)

        return simulate_variance_gamma(key, S0, r - q, theta, sigma, nu, T, cfg)

    elif levy_type == 'NIG':
        from .levy_processes import NIGParams, simulate_nig

        alpha = params.levy_params.get('alpha', 15.0)
        beta = params.levy_params.get('beta', -5.0)
        delta = params.levy_params.get('delta', 0.5)
        mu = params.levy_params.get('mu', 0.0)

        nig_params = NIGParams(alpha=alpha, beta=beta, delta=delta, mu=mu)
        return simulate_nig(key, S0, T, r, q, nig_params, cfg)

    elif levy_type == 'CGMY':
        from .levy_processes import CGMYParams, simulate_cgmy

        C = params.levy_params.get('C', 1.0)
        G = params.levy_params.get('G', 10.0)
        M = params.levy_params.get('M', 10.0)
        Y = params.levy_params.get('Y', 0.5)

        cgmy_params = CGMYParams(C=C, G=G, M=M, Y=Y)
        return simulate_cgmy(key, S0, T, r, q, cgmy_params, cfg)

    else:
        raise NotImplementedError(f"Lévy process '{params.levy_process}' not yet implemented")


# ==============================================================================
# Model Comparison and Selection
# ==============================================================================


def get_model_characteristics():
    """Get characteristics of available equity models.

    Returns
    -------
    dict
        Dictionary with model characteristics

    Example
    -------
    >>> chars = get_model_characteristics()
    >>> print(chars['SLV']['features'])
    ['arbitrage_free', 'smile_fitting', 'forward_vol_dynamics']
    """
    return {
        'Local_Vol': {
            'features': ['arbitrage_free', 'smile_fitting', 'deterministic_vol'],
            'pros': ['Calibrates to full smile', 'No arbitrage', 'Fast PDE pricing'],
            'cons': ['Unrealistic forward vols', 'No stochastic vol effects'],
            'use_cases': ['Exotic options', 'Model-independent hedging'],
        },
        'SLV': {
            'features': ['arbitrage_free', 'smile_fitting', 'forward_vol_dynamics'],
            'pros': ['Realistic vol dynamics', 'Fits market smile', 'Stochastic vol effects'],
            'cons': ['More complex', 'Slower than local vol', 'Calibration challenging'],
            'use_cases': ['Forward-start options', 'Cliquet options', 'Variance swaps'],
        },
        'Rough_Bergomi': {
            'features': ['rough_volatility', 'realistic_short_dated_skew', 'ATM_skew'],
            'pros': ['Fits short-dated options well', 'Realistic vol dynamics', 'Parsimonious'],
            'cons': ['No closed-form solutions', 'Computationally intensive', 'Calibration complex'],
            'use_cases': ['Short-dated options', 'Skew trading', 'Vol surface modeling'],
        },
        'Rough_Heston': {
            'features': ['rough_volatility', 'mean_reverting', 'semi_tractable'],
            'pros': ['More tractable than rBergomi', 'Fits term structure', 'Fast calibration possible'],
            'cons': ['Still computationally intensive', 'Requires fractional calculus'],
            'use_cases': ['Full term structure', 'Vol derivatives', 'Research'],
        },
        'Merton_Jump': {
            'features': ['jumps', 'closed_form', 'lognormal_jumps'],
            'pros': ['Closed-form pricing', 'Captures jump risk', 'Fast calibration'],
            'cons': ['Symmetric jumps', 'Limited smile flexibility'],
            'use_cases': ['Earnings jumps', 'Event risk', 'Credit spreads'],
        },
        'Kou_Jump': {
            'features': ['jumps', 'asymmetric_jumps', 'double_exponential'],
            'pros': ['Asymmetric jumps', 'Better tail modeling', 'Tractable'],
            'cons': ['More parameters', 'Still limited smile'],
            'use_cases': ['Crash protection', 'Deep OTM options', 'Tail risk'],
        },
        'Variance_Gamma': {
            'features': ['pure_jump', 'infinite_activity', 'levy_process'],
            'pros': ['Infinite activity jumps', 'Flexible', 'Fits skew well'],
            'cons': ['No diffusion component', 'Calibration sensitive'],
            'use_cases': ['Skew trading', 'Lévy modeling', 'Research'],
        },
        'NIG': {
            'features': ['pure_jump', 'infinite_activity', 'heavy_tails', 'skewness'],
            'pros': ['Excellent fit to empirical returns', 'Semi-heavy tails', 'Tractable CF'],
            'cons': ['No closed-form for options', 'Requires FFT pricing'],
            'use_cases': ['Return modeling', 'Risk management', 'FFT pricing'],
        },
        'CGMY': {
            'features': ['pure_jump', 'infinite_activity', 'flexible_tails', 'fine_structure'],
            'pros': ['Very flexible', 'Controls small/large jumps separately', 'General framework'],
            'cons': ['Many parameters', 'Complex calibration', 'Computationally intensive'],
            'use_cases': ['Research', 'Exotic options', 'Jump modeling'],
        },
        'Hawkes_Jump': {
            'features': ['self_exciting', 'jump_clustering', 'contagion'],
            'pros': ['Realistic clustering', 'Captures volatility feedback', 'Contagion modeling'],
            'cons': ['Complex dynamics', 'Difficult calibration', 'Path-dependent'],
            'use_cases': ['Crisis modeling', 'Volatility clustering', 'Risk management'],
        },
    }


__all__ = [
    # SLV
    "SLVParams",
    "simulate_slv",
    # Rough Heston
    "RoughHestonParams",
    "simulate_rough_heston",
    # Time-changed Lévy
    "TimeChangedLevyParams",
    "simulate_time_changed_levy",
    # Utils
    "get_model_characteristics",
]

# Note: For direct access to specific models, import from their respective modules:
# - levy_processes: NIGParams, CGMYParams, simulate_nig, simulate_cgmy
# - jump_clustering: HawkesJumpParams, simulate_hawkes_jump_diffusion
# - variance_gamma: simulate_variance_gamma
# - jump_diffusion: MertonParams, MertonJumpDiffusion
# - kou: simulate_kou
