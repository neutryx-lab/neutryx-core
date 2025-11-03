"""Longstaff-Schwartz American option pricing with JAX.

This module provides production-ready JAX implementations of the Longstaff-Schwartz
Monte Carlo method for pricing American options with early exercise.
"""
import jax
import jax.numpy as jnp
from jax import lax


def american_put_lsm(ST_paths, K, r, dt):
    """Price American put option using Longstaff-Schwartz algorithm.

    Parameters
    ----------
    ST_paths : Array
        Simulated price paths of shape [paths, steps+1]
    K : float
        Strike price
    r : float
        Risk-free rate
    dt : float
        Time step size

    Returns
    -------
    float
        Present value of the American put option

    Notes
    -----
    This implementation uses jax.lax.fori_loop for JIT compatibility and
    optimal performance on GPU/TPU hardware.
    """
    paths, steps_plus = ST_paths.shape
    steps = steps_plus - 1

    # Immediate payoff matrix for put
    payoff = jnp.maximum(K - ST_paths, 0.0)

    # Initialize continuation value as terminal payoff
    V = payoff[:, -1]

    # Backward induction using fori_loop
    def backward_step(t_idx, V_current):
        # Map t_idx (0, 1, ..., steps-2) to actual time (steps-1, steps-2, ..., 1)
        t = steps - 1 - t_idx

        # Find in-the-money paths
        immediate = payoff[:, t]
        itm_mask = immediate > 1e-10  # Small threshold for numerical stability

        # Conditional regression on ITM paths only
        X = ST_paths[:, t]
        Y = V_current * jnp.exp(-r * dt)

        # Build regression matrix: [1, S, S^2]
        A = jnp.stack([jnp.ones_like(X), X, X * X], axis=1)

        # Solve least squares with regularization
        # Using solve instead of lstsq for better JIT compatibility
        ATA = A.T @ (A * itm_mask[:, None])
        ATY = A.T @ (Y * itm_mask)

        # Add small regularization for numerical stability
        regularization = 1e-8 * jnp.eye(3)
        beta = jnp.linalg.solve(ATA + regularization, ATY)

        # Continuation value estimate
        continuation = A @ beta

        # Exercise decision: exercise if immediate > continuation
        exercise = (immediate > continuation) & itm_mask

        # Update continuation value
        V_new = jnp.where(exercise, immediate, V_current * jnp.exp(-r * dt))

        return V_new

    # Run backward induction
    V_final = lax.fori_loop(0, steps - 1, backward_step, V)

    # Discount from t=1 to t=0
    V_0 = jnp.maximum(payoff[:, 0], V_final * jnp.exp(-r * dt))

    return V_0.mean()


def american_call_lsm(ST_paths, K, r, dt):
    """Price American call option using Longstaff-Schwartz algorithm.

    Parameters
    ----------
    ST_paths : Array
        Simulated price paths of shape [paths, steps+1]
    K : float
        Strike price
    r : float
        Risk-free rate
    dt : float
        Time step size

    Returns
    -------
    float
        Present value of the American call option
    """
    paths, steps_plus = ST_paths.shape
    steps = steps_plus - 1

    # Immediate payoff matrix for call
    payoff = jnp.maximum(ST_paths - K, 0.0)

    # Initialize continuation value as terminal payoff
    V = payoff[:, -1]

    # Backward induction
    def backward_step(t_idx, V_current):
        t = steps - 1 - t_idx

        immediate = payoff[:, t]
        itm_mask = immediate > 1e-10

        X = ST_paths[:, t]
        Y = V_current * jnp.exp(-r * dt)

        A = jnp.stack([jnp.ones_like(X), X, X * X], axis=1)

        ATA = A.T @ (A * itm_mask[:, None])
        ATY = A.T @ (Y * itm_mask)

        regularization = 1e-8 * jnp.eye(3)
        beta = jnp.linalg.solve(ATA + regularization, ATY)

        continuation = A @ beta

        exercise = (immediate > continuation) & itm_mask

        V_new = jnp.where(exercise, immediate, V_current * jnp.exp(-r * dt))

        return V_new

    V_final = lax.fori_loop(0, steps - 1, backward_step, V)

    V_0 = jnp.maximum(payoff[:, 0], V_final * jnp.exp(-r * dt))

    return V_0.mean()


# Maintain backward compatibility
def american_option_lsm(ST_paths, K, r, dt, kind="put"):
    """Price American option using Longstaff-Schwartz algorithm.

    Parameters
    ----------
    ST_paths : Array
        Simulated price paths of shape [paths, steps+1]
    K : float
        Strike price
    r : float
        Risk-free rate
    dt : float
        Time step size
    kind : str
        Option type: "put" or "call"

    Returns
    -------
    float
        Present value of the American option
    """
    if kind == "put":
        return american_put_lsm(ST_paths, K, r, dt)
    elif kind == "call":
        return american_call_lsm(ST_paths, K, r, dt)
    else:
        raise ValueError(f"Unknown option kind: {kind}. Must be 'put' or 'call'.")
