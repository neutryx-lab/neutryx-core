"""Utility functions for XVA and credit risk calculations.

This module provides helper functions for converting between different
representations of credit risk metrics, such as hazard rates and default
probabilities.
"""
import jax.numpy as jnp


def hazard_to_pd(lambda_t, times):
    """Convert hazard rate to cumulative default probability.

    Transforms a constant hazard rate (intensity) into a cumulative
    probability of default over specified time horizons.

    Parameters
    ----------
    lambda_t : float or Array
        Hazard rate (default intensity) per unit time.
        Can be scalar or array matching the shape of times
    times : Array
        Time horizons at which to compute default probabilities

    Returns
    -------
    Array
        Cumulative default probabilities at each time horizon

    Notes
    -----
    The cumulative default probability is given by:
        PD(t) = 1 - exp(-λ * t)

    where λ is the hazard rate.

    This formula assumes a constant hazard rate. For time-varying hazard
    rates, the integral of the hazard rate function should be used instead:
        PD(t) = 1 - exp(-∫[0,t] λ(s) ds)

    Examples
    --------
    >>> hazard_to_pd(0.05, jnp.array([1.0, 2.0, 5.0]))
    Array([0.04877, 0.09516, 0.22120], dtype=float32)
    """
    return 1.0 - jnp.exp(-lambda_t * times)
