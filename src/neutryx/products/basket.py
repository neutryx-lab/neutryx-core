"""Basket option pricing for multi-asset options.

Basket options depend on a portfolio or basket of underlying assets.
Common types include worst-of, best-of, average, and rainbow options.

This module provides both:
1. Product classes for integration with pricing engines
2. Standalone payoff functions for custom implementations
"""
from dataclasses import dataclass

import jax.numpy as jnp

from neutryx.core.engine import Array
from .base import Product


def worst_of_call_payoff(ST_basket: Array, K: float) -> Array:
    """Worst-of call payoff: max(min(S1_T, S2_T, ...) - K, 0).

    Parameters
    ----------
    ST_basket : Array
        Terminal prices for each asset, shape [paths, n_assets]
    K : float
        Strike price

    Returns
    -------
    Array
        Payoff for each path
    """
    worst = jnp.min(ST_basket, axis=1)
    return jnp.maximum(worst - K, 0.0)


def best_of_call_payoff(ST_basket: Array, K: float) -> Array:
    """Best-of call payoff: max(max(S1_T, S2_T, ...) - K, 0).

    Parameters
    ----------
    ST_basket : Array
        Terminal prices for each asset, shape [paths, n_assets]
    K : float
        Strike price

    Returns
    -------
    Array
        Payoff for each path
    """
    best = jnp.max(ST_basket, axis=1)
    return jnp.maximum(best - K, 0.0)


def worst_of_put_payoff(ST_basket: Array, K: float) -> Array:
    """Worst-of put payoff: max(K - min(S1_T, S2_T, ...), 0).

    Parameters
    ----------
    ST_basket : Array
        Terminal prices for each asset, shape [paths, n_assets]
    K : float
        Strike price

    Returns
    -------
    Array
        Payoff for each path
    """
    worst = jnp.min(ST_basket, axis=1)
    return jnp.maximum(K - worst, 0.0)


def best_of_put_payoff(ST_basket: Array, K: float) -> Array:
    """Best-of put payoff: max(K - max(S1_T, S2_T, ...), 0).

    Parameters
    ----------
    ST_basket : Array
        Terminal prices for each asset, shape [paths, n_assets]
    K : float
        Strike price

    Returns
    -------
    Array
        Payoff for each path
    """
    best = jnp.max(ST_basket, axis=1)
    return jnp.maximum(K - best, 0.0)


def average_basket_call_payoff(ST_basket: Array, K: float, weights: Array = None) -> Array:
    """Average basket call payoff: max(weighted_avg(S1_T, S2_T, ...) - K, 0).

    Parameters
    ----------
    ST_basket : Array
        Terminal prices for each asset, shape [paths, n_assets]
    K : float
        Strike price
    weights : Array, optional
        Weight for each asset. If None, equal weights are used.

    Returns
    -------
    Array
        Payoff for each path
    """
    if weights is None:
        weights = jnp.ones(ST_basket.shape[1]) / ST_basket.shape[1]
    else:
        weights = jnp.asarray(weights)
        weights = weights / weights.sum()  # Normalize

    avg = jnp.dot(ST_basket, weights)
    return jnp.maximum(avg - K, 0.0)


def rainbow_max_call_payoff(ST_basket: Array, K1: float, K2: float) -> Array:
    """Rainbow option: max(S1_T - K1, S2_T - K2, ..., 0).

    Parameters
    ----------
    ST_basket : Array
        Terminal prices for each asset, shape [paths, n_assets]
    K1 : float
        Strike for first asset
    K2 : float
        Strike for second asset (can be extended to more)

    Returns
    -------
    Array
        Payoff for each path
    """
    # For simplicity, assume 2 assets
    payoff1 = ST_basket[:, 0] - K1
    payoff2 = ST_basket[:, 1] - K2
    return jnp.maximum(jnp.maximum(payoff1, payoff2), 0.0)


def basket_option_mc(paths_basket: Array, payoff_fn, K: float, r: float, T: float,
                     **payoff_kwargs) -> float:
    """Generic Monte Carlo pricing for basket options.

    Parameters
    ----------
    paths_basket : Array
        Simulated price paths for basket, shape [paths, steps+1, n_assets]
    payoff_fn : callable
        Payoff function that takes (ST_basket, K, **kwargs) and returns payoffs
    K : float
        Strike price
    r : float
        Risk-free rate
    T : float
        Time to maturity
    **payoff_kwargs
        Additional keyword arguments to pass to the payoff function

    Returns
    -------
    float
        Basket option price
    """
    ST_basket = paths_basket[:, -1, :]
    payoffs = payoff_fn(ST_basket, K, **payoff_kwargs)
    discount = jnp.exp(-r * T)
    return float((discount * payoffs).mean())


def worst_of_call_mc(paths_basket: Array, K: float, r: float, T: float) -> float:
    """Monte Carlo pricing for worst-of call option.

    Parameters
    ----------
    paths_basket : Array
        Simulated price paths for basket, shape [paths, steps+1, n_assets]
    K : float
        Strike price
    r : float
        Risk-free rate
    T : float
        Time to maturity

    Returns
    -------
    float
        Worst-of call option price
    """
    return basket_option_mc(paths_basket, worst_of_call_payoff, K, r, T)


def best_of_call_mc(paths_basket: Array, K: float, r: float, T: float) -> float:
    """Monte Carlo pricing for best-of call option.

    Parameters
    ----------
    paths_basket : Array
        Simulated price paths for basket, shape [paths, steps+1, n_assets]
    K : float
        Strike price
    r : float
        Risk-free rate
    T : float
        Time to maturity

    Returns
    -------
    float
        Best-of call option price
    """
    return basket_option_mc(paths_basket, best_of_call_payoff, K, r, T)


def average_basket_call_mc(paths_basket: Array, K: float, r: float, T: float,
                           weights: Array = None) -> float:
    """Monte Carlo pricing for average basket call option.

    Parameters
    ----------
    paths_basket : Array
        Simulated price paths for basket, shape [paths, steps+1, n_assets]
    K : float
        Strike price
    r : float
        Risk-free rate
    T : float
        Time to maturity
    weights : Array, optional
        Weight for each asset

    Returns
    -------
    float
        Average basket call option price
    """
    return basket_option_mc(paths_basket, average_basket_call_payoff, K, r, T, weights=weights)


# Multi-asset simulation helper
def simulate_correlated_gbm(key, S0_basket, mu_basket, sigma_basket, correlation_matrix,
                           T, steps, paths):
    """Simulate correlated GBM paths for multiple assets.

    This implementation uses jax.lax.scan for efficient JIT compilation and
    optimal performance on accelerators.

    Parameters
    ----------
    key : jax.random.KeyArray
        Random key
    S0_basket : Array
        Initial prices for each asset, shape [n_assets]
    mu_basket : Array
        Drift for each asset, shape [n_assets]
    sigma_basket : Array
        Volatility for each asset, shape [n_assets]
    correlation_matrix : Array
        Correlation matrix, shape [n_assets, n_assets]
    T : float
        Time to maturity
    steps : int
        Number of time steps
    paths : int
        Number of Monte Carlo paths

    Returns
    -------
    Array
        Simulated paths, shape [paths, steps+1, n_assets]
    """
    import jax
    from jax import lax

    n_assets = len(S0_basket)
    dt = T / steps

    # Cholesky decomposition for correlation
    L = jnp.linalg.cholesky(correlation_matrix)

    # Generate independent normals
    normals = jax.random.normal(key, (paths, steps, n_assets))

    # Apply correlation
    correlated_normals = jnp.einsum('ptn,nm->ptm', normals, L)

    # Precompute drift term
    drift = (mu_basket - 0.5 * sigma_basket ** 2) * dt
    diffusion_scale = sigma_basket * jnp.sqrt(dt)

    # Initial log prices
    log_S0 = jnp.log(S0_basket)

    # Scan function for time stepping
    def step_fn(log_S_prev, t_idx):
        """Single time step of GBM simulation."""
        diffusion = diffusion_scale * correlated_normals[:, t_idx, :]
        log_S_next = log_S_prev + drift + diffusion
        return log_S_next, log_S_next

    # Run simulation using scan
    _, log_paths_scan = lax.scan(step_fn, log_S0, jnp.arange(steps))

    # Combine initial state with scanned results
    # log_paths_scan shape: [steps, paths, n_assets]
    # Need to transpose to [paths, steps, n_assets] and prepend initial values
    log_paths_scan = jnp.transpose(log_paths_scan, (1, 0, 2))  # [paths, steps, n_assets]

    # Prepend initial log prices
    log_S0_expanded = jnp.broadcast_to(log_S0, (paths, 1, n_assets))
    log_paths = jnp.concatenate([log_S0_expanded, log_paths_scan], axis=1)

    return jnp.exp(log_paths)


# ============================================================================
# Product Classes for Multi-Asset Options
# ============================================================================


@dataclass
class WorstOfCall(Product):
    """Worst-of call option on a basket of assets.

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Time to maturity
    n_assets : int
        Number of assets in the basket

    Notes
    -----
    Payoff = max(min(S1_T, S2_T, ..., Sn_T) - K, 0)

    For multi-asset products, the path should be the terminal prices
    of all assets, shape: [n_assets]
    """

    K: float
    T: float
    n_assets: int

    def payoff_terminal(self, spots: Array) -> Array:
        """Compute payoff from terminal spot prices.

        Parameters
        ----------
        spots : Array
            Terminal prices for each asset, shape [n_assets]

        Returns
        -------
        Array
            Payoff value
        """
        spots = jnp.asarray(spots)
        worst = jnp.min(spots)
        return jnp.maximum(worst - self.K, 0.0)


@dataclass
class WorstOfPut(Product):
    """Worst-of put option on a basket of assets.

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Time to maturity
    n_assets : int
        Number of assets in the basket

    Notes
    -----
    Payoff = max(K - min(S1_T, S2_T, ..., Sn_T), 0)
    """

    K: float
    T: float
    n_assets: int

    def payoff_terminal(self, spots: Array) -> Array:
        spots = jnp.asarray(spots)
        worst = jnp.min(spots)
        return jnp.maximum(self.K - worst, 0.0)


@dataclass
class BestOfCall(Product):
    """Best-of call option on a basket of assets.

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Time to maturity
    n_assets : int
        Number of assets in the basket

    Notes
    -----
    Payoff = max(max(S1_T, S2_T, ..., Sn_T) - K, 0)
    """

    K: float
    T: float
    n_assets: int

    def payoff_terminal(self, spots: Array) -> Array:
        spots = jnp.asarray(spots)
        best = jnp.max(spots)
        return jnp.maximum(best - self.K, 0.0)


@dataclass
class BestOfPut(Product):
    """Best-of put option on a basket of assets.

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Time to maturity
    n_assets : int
        Number of assets in the basket

    Notes
    -----
    Payoff = max(K - max(S1_T, S2_T, ..., Sn_T), 0)
    """

    K: float
    T: float
    n_assets: int

    def payoff_terminal(self, spots: Array) -> Array:
        spots = jnp.asarray(spots)
        best = jnp.max(spots)
        return jnp.maximum(self.K - best, 0.0)


@dataclass
class AverageBasketCall(Product):
    """Call option on weighted average of basket assets.

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Time to maturity
    weights : Array or None
        Weight for each asset. If None, equal weights are used.
    n_assets : int
        Number of assets in the basket

    Notes
    -----
    Payoff = max(weighted_avg(S1_T, S2_T, ...) - K, 0)
    """

    K: float
    T: float
    n_assets: int
    weights: Array = None

    def payoff_terminal(self, spots: Array) -> Array:
        spots = jnp.asarray(spots)
        if self.weights is None:
            weights = jnp.ones(self.n_assets) / self.n_assets
        else:
            weights = jnp.asarray(self.weights)
            weights = weights / weights.sum()  # Normalize

        avg = jnp.dot(spots, weights)
        return jnp.maximum(avg - self.K, 0.0)


@dataclass
class AverageBasketPut(Product):
    """Put option on weighted average of basket assets.

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Time to maturity
    weights : Array or None
        Weight for each asset
    n_assets : int
        Number of assets in the basket

    Notes
    -----
    Payoff = max(K - weighted_avg(S1_T, S2_T, ...), 0)
    """

    K: float
    T: float
    n_assets: int
    weights: Array = None

    def payoff_terminal(self, spots: Array) -> Array:
        spots = jnp.asarray(spots)
        if self.weights is None:
            weights = jnp.ones(self.n_assets) / self.n_assets
        else:
            weights = jnp.asarray(self.weights)
            weights = weights / weights.sum()

        avg = jnp.dot(spots, weights)
        return jnp.maximum(self.K - avg, 0.0)


@dataclass
class SpreadOption(Product):
    """Spread option between two assets.

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Time to maturity
    is_call : bool
        True for call, False for put

    Notes
    -----
    Payoff = max(S1_T - S2_T - K, 0) for call
    Payoff = max(K - (S1_T - S2_T), 0) for put

    Common in commodity markets (crack spreads, spark spreads).
    """

    K: float
    T: float
    is_call: bool = True

    def payoff_terminal(self, spots: Array) -> Array:
        """Compute payoff from two asset prices.

        Parameters
        ----------
        spots : Array
            Terminal prices [S1_T, S2_T]
        """
        spots = jnp.asarray(spots)
        spread = spots[0] - spots[1]
        if self.is_call:
            return jnp.maximum(spread - self.K, 0.0)
        else:
            return jnp.maximum(self.K - spread, 0.0)


@dataclass
class RainbowOption(Product):
    """Rainbow option (best/worst of N assets with individual strikes).

    Parameters
    ----------
    strikes : Array
        Strike for each asset
    T : float
        Time to maturity
    option_type : str
        'best_of' or 'worst_of'
    is_call : bool
        True for call, False for put
    n_assets : int
        Number of assets

    Notes
    -----
    Rainbow options provide exposure to the best or worst performing
    asset in a basket.

    For best_of call: Payoff = max(max_i(S_i - K_i), 0)
    For worst_of call: Payoff = max(min_i(S_i - K_i), 0)
    """

    strikes: Array
    T: float
    option_type: str = "best_of"  # 'best_of' or 'worst_of'
    is_call: bool = True
    n_assets: int = 2

    def payoff_terminal(self, spots: Array) -> Array:
        spots = jnp.asarray(spots)
        strikes = jnp.asarray(self.strikes)

        if self.is_call:
            individual_payoffs = spots - strikes
        else:
            individual_payoffs = strikes - spots

        if self.option_type == "best_of":
            payoff = jnp.max(individual_payoffs)
        else:  # worst_of
            payoff = jnp.min(individual_payoffs)

        return jnp.maximum(payoff, 0.0)


__all__ = [
    # Product classes
    "WorstOfCall",
    "WorstOfPut",
    "BestOfCall",
    "BestOfPut",
    "AverageBasketCall",
    "AverageBasketPut",
    "SpreadOption",
    "RainbowOption",
    # Payoff functions
    "worst_of_call_payoff",
    "best_of_call_payoff",
    "worst_of_put_payoff",
    "best_of_put_payoff",
    "average_basket_call_payoff",
    "rainbow_max_call_payoff",
    # Unified pricing function
    "basket_option_mc",
    # Convenience wrappers
    "worst_of_call_mc",
    "best_of_call_mc",
    "average_basket_call_mc",
    # Simulation
    "simulate_correlated_gbm",
]
