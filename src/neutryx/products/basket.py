"""Basket option pricing for multi-asset options.

Basket options depend on a portfolio or basket of underlying assets.
Common types include worst-of, best-of, average, and rainbow options.
"""
import jax.numpy as jnp

from neutryx.core.engine import Array


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
    # Get terminal values for all assets
    ST_basket = paths_basket[:, -1, :]
    payoffs = worst_of_call_payoff(ST_basket, K)
    discount = jnp.exp(-r * T)
    return float((discount * payoffs).mean())


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
    ST_basket = paths_basket[:, -1, :]
    payoffs = best_of_call_payoff(ST_basket, K)
    discount = jnp.exp(-r * T)
    return float((discount * payoffs).mean())


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
    ST_basket = paths_basket[:, -1, :]
    payoffs = average_basket_call_payoff(ST_basket, K, weights)
    discount = jnp.exp(-r * T)
    return float((discount * payoffs).mean())


# Multi-asset simulation helper
def simulate_correlated_gbm(key, S0_basket, mu_basket, sigma_basket, correlation_matrix,
                           T, steps, paths):
    """Simulate correlated GBM paths for multiple assets.

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

    n_assets = len(S0_basket)
    dt = T / steps

    # Cholesky decomposition for correlation
    L = jnp.linalg.cholesky(correlation_matrix)

    # Generate independent normals
    normals = jax.random.normal(key, (paths, steps, n_assets))

    # Apply correlation
    correlated_normals = jnp.einsum('ptn,nm->ptm', normals, L)

    # Initialize paths
    log_paths = jnp.zeros((paths, steps + 1, n_assets))
    log_S0 = jnp.log(S0_basket)
    log_paths = log_paths.at[:, 0, :].set(log_S0)

    # Simulate paths
    for t in range(steps):
        drift = (mu_basket - 0.5 * sigma_basket ** 2) * dt
        diffusion = sigma_basket * jnp.sqrt(dt) * correlated_normals[:, t, :]
        log_paths = log_paths.at[:, t + 1, :].set(log_paths[:, t, :] + drift + diffusion)

    return jnp.exp(log_paths)


__all__ = [
    "worst_of_call_payoff",
    "best_of_call_payoff",
    "worst_of_put_payoff",
    "best_of_put_payoff",
    "average_basket_call_payoff",
    "rainbow_max_call_payoff",
    "worst_of_call_mc",
    "best_of_call_mc",
    "average_basket_call_mc",
    "simulate_correlated_gbm",
]
