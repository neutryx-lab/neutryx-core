"""Exposure calculation utilities for credit valuation adjustments.

This module provides functions for computing expected positive exposure (EPE)
and other exposure metrics used in counterparty credit risk assessment and
XVA calculations.
"""
import jax.numpy as jnp


def epe(paths, K, is_call=True):
    """Calculate Expected Positive Exposure (EPE) for an option.

    EPE represents the average positive mark-to-market value of a position,
    weighted by the probability of default. It is a key input for CVA
    (Credit Valuation Adjustment) calculations.

    Parameters
    ----------
    paths : Array
        Simulated price paths of shape [paths, steps+1]
    K : float
        Strike price
    is_call : bool, optional
        If True, compute EPE for call option. If False, for put option.
        Default is True.

    Returns
    -------
    float
        Expected positive exposure

    Notes
    -----
    This simplified implementation computes EPE at maturity only. A full
    implementation would compute exposure profiles across all time steps.
    """
    ST = paths[:, -1]
    payoff = jnp.maximum(ST - K, 0.0) if is_call else jnp.maximum(K - ST, 0.0)
    return payoff.mean()
