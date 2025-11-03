"""Trinomial tree models for option pricing.

This module implements trinomial tree methods for pricing American and European options.
"""
from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
from jax import Array


def build_trinomial_tree(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    N: int,
    q: float = 0.0,
) -> tuple[Array, Array, float, float, float]:
    """Build a trinomial tree for stock price evolution.

    Parameters
    ----------
    S0 : float
        Initial stock price
    r : float
        Risk-free rate
    sigma : float
        Volatility
    T : float
        Time to maturity
    N : int
        Number of time steps
    q : float
        Dividend yield

    Returns
    -------
    tuple[Array, Array, float, float, float]
        (stock_tree, prob_up, prob_mid, prob_down, dt)
    """
    dt = T / N
    u = jnp.exp(sigma * jnp.sqrt(2 * dt))
    d = 1 / u
    m = 1.0  # Middle node stays the same

    # Risk-neutral probabilities
    dx = jnp.log(u)
    nu = r - q - 0.5 * sigma**2

    pu = 0.5 * ((sigma**2 * dt + nu**2 * dt**2) / dx**2 + nu * dt / dx)
    pd = 0.5 * ((sigma**2 * dt + nu**2 * dt**2) / dx**2 - nu * dt / dx)
    pm = 1.0 - pu - pd

    # Build stock price tree
    # Tree has shape where at step i, we have 2*i+1 possible values
    max_nodes = 2 * N + 1
    stock_tree = jnp.zeros((N + 1, max_nodes))

    # Initialize first step
    stock_tree = stock_tree.at[0, N].set(S0)

    # Build forward
    for i in range(1, N + 1):
        for j in range(max_nodes):
            if j < N - i or j > N + i:
                continue
            if j == 0:
                stock_tree = stock_tree.at[i, j].set(stock_tree[i - 1, j + 1] * d)
            elif j == max_nodes - 1:
                stock_tree = stock_tree.at[i, j].set(stock_tree[i - 1, j - 1] * u)
            else:
                # Middle nodes can come from up, middle, or down
                if j > 0 and stock_tree[i - 1, j - 1] > 0:
                    stock_tree = stock_tree.at[i, j].set(stock_tree[i - 1, j - 1] * m)
                elif j > 1 and stock_tree[i - 1, j - 2] > 0:
                    stock_tree = stock_tree.at[i, j].set(stock_tree[i - 1, j - 2] * d)
                elif j < max_nodes - 1 and stock_tree[i - 1, j + 1] > 0:
                    stock_tree = stock_tree.at[i, j].set(stock_tree[i - 1, j + 1] * u)

    # Simplified: recombining tree
    for i in range(N + 1):
        center = N
        for k in range(-i, i + 1):
            idx = center + k
            if k > 0:
                stock_tree = stock_tree.at[i, idx].set(S0 * (u ** k))
            elif k < 0:
                stock_tree = stock_tree.at[i, idx].set(S0 * (d ** (-k)))
            else:
                stock_tree = stock_tree.at[i, idx].set(S0)

    return stock_tree, pu, pm, pd, dt


def price_trinomial_tree(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
    exercise_type: Literal["european", "american"] = "european",
    N: int = 100,
    q: float = 0.0,
) -> float:
    """Price option using trinomial tree.

    Parameters
    ----------
    S0 : float
        Initial stock price
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    sigma : float
        Volatility
    option_type : str
        "call" or "put"
    exercise_type : str
        "european" or "american"
    N : int
        Number of time steps
    q : float
        Dividend yield

    Returns
    -------
    float
        Option price
    """
    dt = T / N
    u = jnp.exp(sigma * jnp.sqrt(2 * dt))
    d = 1 / u

    # Risk-neutral probabilities
    dx = jnp.log(u)
    nu = r - q - 0.5 * sigma**2

    pu = 0.5 * ((sigma**2 * dt + nu**2 * dt**2) / dx**2 + nu * dt / dx)
    pd = 0.5 * ((sigma**2 * dt + nu**2 * dt**2) / dx**2 - nu * dt / dx)
    pm = 1.0 - pu - pd

    # Discount factor
    df = jnp.exp(-r * dt)

    # Build stock prices at each node (recombining tree)
    # At step i, we have 2*i+1 nodes
    max_nodes = 2 * N + 1

    # Terminal payoffs
    option_values = jnp.zeros(max_nodes)

    # Calculate terminal stock prices and payoffs
    for j in range(max_nodes):
        k = j - N  # Displacement from center
        S_T = S0 * (u ** k)
        if option_type == "call":
            option_values = option_values.at[j].set(jnp.maximum(S_T - K, 0.0))
        else:
            option_values = option_values.at[j].set(jnp.maximum(K - S_T, 0.0))

    # Backward induction
    for i in range(N - 1, -1, -1):
        new_values = jnp.zeros(max_nodes)

        for j in range(max_nodes):
            k = j - N
            # Check if this node is active at this time step
            if abs(k) > i:
                continue

            # Continuation value
            continuation = df * (pu * option_values[j + 1] + pm * option_values[j] + pd * option_values[j - 1])

            if exercise_type == "american":
                # Early exercise value
                S_current = S0 * (u ** k)
                if option_type == "call":
                    exercise_value = jnp.maximum(S_current - K, 0.0)
                else:
                    exercise_value = jnp.maximum(K - S_current, 0.0)

                new_values = new_values.at[j].set(jnp.maximum(continuation, exercise_value))
            else:
                new_values = new_values.at[j].set(continuation)

        option_values = new_values

    return float(option_values[N])


def price_european_trinomial(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    is_call: bool = True,
    N: int = 100,
    q: float = 0.0,
) -> float:
    """Price European option using trinomial tree.

    Parameters
    ----------
    S0 : float
        Initial stock price
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    sigma : float
        Volatility
    is_call : bool
        True for call, False for put
    N : int
        Number of time steps
    q : float
        Dividend yield

    Returns
    -------
    float
        European option price
    """
    option_type = "call" if is_call else "put"
    return price_trinomial_tree(S0, K, T, r, sigma, option_type, "european", N, q)


def price_american_trinomial(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    is_call: bool = True,
    N: int = 100,
    q: float = 0.0,
) -> float:
    """Price American option using trinomial tree.

    Parameters
    ----------
    S0 : float
        Initial stock price
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    sigma : float
        Volatility
    is_call : bool
        True for call, False for put
    N : int
        Number of time steps
    q : float
        Dividend yield

    Returns
    -------
    float
        American option price
    """
    option_type = "call" if is_call else "put"
    return price_trinomial_tree(S0, K, T, r, sigma, option_type, "american", N, q)


__all__ = [
    "build_trinomial_tree",
    "price_trinomial_tree",
    "price_european_trinomial",
    "price_american_trinomial",
]
