"""Internal utility functions for market data operations.

This module contains common mathematical operations and helper functions
used across market curves, FX, and volatility surfaces.
"""

from __future__ import annotations

import jax.numpy as jnp


def df_to_zero_rate(discount_factor: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
    """Convert discount factor to zero rate.

    Parameters
    ----------
    discount_factor : jnp.ndarray
        Discount factor(s)
    time : jnp.ndarray
        Time to maturity

    Returns
    -------
    jnp.ndarray
        Zero rate(s)
    """
    # Avoid division by zero for very small times
    t_safe = jnp.where(time > 1e-10, time, 1e-10)
    return -jnp.log(discount_factor) / t_safe


def compute_forward_rate(
    df1: jnp.ndarray,
    df2: jnp.ndarray,
    t1: float,
    t2: float
) -> jnp.ndarray:
    """Compute forward rate between two times.

    Parameters
    ----------
    df1 : jnp.ndarray
        Discount factor at time t1
    df2 : jnp.ndarray
        Discount factor at time t2
    t1 : float
        Start time
    t2 : float
        End time

    Returns
    -------
    jnp.ndarray
        Forward rate from t1 to t2
    """
    dt = jnp.maximum(t2 - t1, 1e-10)
    return -jnp.log(df2 / df1) / dt


def validate_curve_inputs(times: jnp.ndarray, rates: jnp.ndarray) -> None:
    """Validate curve input arrays.

    Parameters
    ----------
    times : jnp.ndarray
        Time points
    rates : jnp.ndarray
        Rate values

    Raises
    ------
    ValueError
        If inputs are invalid
    """
    if len(times) != len(rates):
        raise ValueError(
            f"times and rates must have same length: "
            f"got {len(times)} times and {len(rates)} rates"
        )

    if len(times) == 0:
        raise ValueError("times and rates must not be empty")


def ensure_array(x: jnp.ndarray | float | list) -> jnp.ndarray:
    """Convert input to JAX array.

    Parameters
    ----------
    x : array-like or scalar
        Input to convert

    Returns
    -------
    jnp.ndarray
        JAX array
    """
    return jnp.asarray(x)
