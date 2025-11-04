"""Lookback option payoffs.

Lookback options are path-dependent options where the payoff depends on the
maximum or minimum price reached during the option's life.

Types:
- Floating Strike: The strike is determined by the path (min for calls, max for puts)
- Fixed Strike: The strike is predetermined, payoff based on max/min price
"""
from dataclasses import dataclass

import jax.numpy as jnp

from .base import PathProduct


@dataclass
class LookbackFloatStrikeCall(PathProduct):
    """Floating-strike lookback call option.

    Parameters
    ----------
    T : float
        Time to maturity

    Notes
    -----
    The holder receives the difference between the terminal price and
    the minimum price observed during the option's life.

    Payoff = S_T - min(S_t)

    This is equivalent to a call with strike = minimum price along the path.
    """

    T: float

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        path = jnp.asarray(path)
        return path[-1] - path.min()


@dataclass
class LookbackFloatStrikePut(PathProduct):
    """Floating-strike lookback put option.

    Parameters
    ----------
    T : float
        Time to maturity

    Notes
    -----
    The holder receives the difference between the maximum price observed
    during the option's life and the terminal price.

    Payoff = max(S_t) - S_T

    This is equivalent to a put with strike = maximum price along the path.
    """

    T: float

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        path = jnp.asarray(path)
        return path.max() - path[-1]


@dataclass
class LookbackFixedStrikeCall(PathProduct):
    """Fixed-strike lookback call option.

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Time to maturity

    Notes
    -----
    The holder receives the payoff based on the maximum price observed
    during the option's life.

    Payoff = max(max(S_t) - K, 0)

    The holder can effectively "buy" at the strike and "sell" at the
    highest price observed during the option's life.
    """

    K: float
    T: float

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        path = jnp.asarray(path)
        max_price = path.max()
        return jnp.maximum(max_price - self.K, 0.0)


@dataclass
class LookbackFixedStrikePut(PathProduct):
    """Fixed-strike lookback put option.

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Time to maturity

    Notes
    -----
    The holder receives the payoff based on the minimum price observed
    during the option's life.

    Payoff = max(K - min(S_t), 0)

    The holder can effectively "sell" at the strike and "buy" at the
    lowest price observed during the option's life.
    """

    K: float
    T: float

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        path = jnp.asarray(path)
        min_price = path.min()
        return jnp.maximum(self.K - min_price, 0.0)


@dataclass
class LookbackPartialFixedStrikeCall(PathProduct):
    """Partial lookback call with observation window.

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Time to maturity
    observation_start : float
        Start of observation window (as fraction of total time)

    Notes
    -----
    Similar to fixed-strike lookback, but only observes prices during
    a portion of the option's life.

    Useful for reducing premium costs while maintaining lookback features.
    """

    K: float
    T: float
    observation_start: float = 0.0  # Default to full observation

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        path = jnp.asarray(path)
        # Determine observation window
        n = len(path)
        start_idx = int(self.observation_start * (n - 1))
        observed_path = path[start_idx:]
        max_price = observed_path.max()
        return jnp.maximum(max_price - self.K, 0.0)


@dataclass
class LookbackPartialFixedStrikePut(PathProduct):
    """Partial lookback put with observation window.

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Time to maturity
    observation_start : float
        Start of observation window (as fraction of total time)

    Notes
    -----
    Similar to fixed-strike lookback put, but only observes prices during
    a portion of the option's life.
    """

    K: float
    T: float
    observation_start: float = 0.0

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        path = jnp.asarray(path)
        n = len(path)
        start_idx = int(self.observation_start * (n - 1))
        observed_path = path[start_idx:]
        min_price = observed_path.min()
        return jnp.maximum(self.K - min_price, 0.0)


__all__ = [
    "LookbackFloatStrikeCall",
    "LookbackFloatStrikePut",
    "LookbackFixedStrikeCall",
    "LookbackFixedStrikePut",
    "LookbackPartialFixedStrikeCall",
    "LookbackPartialFixedStrikePut",
]

