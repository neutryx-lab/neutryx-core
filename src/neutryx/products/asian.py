"""Asian option payoffs.

Asian options are path-dependent options where the payoff depends on the average
price of the underlying asset over the option's life.

Types:
- Arithmetic Average: Uses arithmetic mean of prices
- Geometric Average: Uses geometric mean of prices
- Fixed Strike: Strike is predetermined
- Floating Strike: Strike is the average price, payoff based on terminal price
"""
from dataclasses import dataclass

import jax.numpy as jnp

from .base import PathProduct


@dataclass
class AsianArithmetic(PathProduct):
    """Arithmetic-average Asian option (fixed strike).

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Time to maturity
    is_call : bool
        True for call, False for put (default: True)

    Notes
    -----
    Payoff = max(Arithmetic_Avg(S) - K, 0) for call
    Payoff = max(K - Arithmetic_Avg(S), 0) for put
    """

    K: float
    T: float
    is_call: bool = True

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        path = jnp.asarray(path)
        avg = path.mean()
        intrinsic = avg - self.K if self.is_call else self.K - avg
        return jnp.maximum(intrinsic, 0.0)


@dataclass
class AsianGeometric(PathProduct):
    """Geometric-average Asian option (fixed strike).

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Time to maturity
    is_call : bool
        True for call, False for put (default: True)

    Notes
    -----
    Payoff = max(Geometric_Avg(S) - K, 0) for call
    Payoff = max(K - Geometric_Avg(S), 0) for put

    The geometric average is calculated as:
        G = (S_1 * S_2 * ... * S_n)^(1/n) = exp(mean(log(S_i)))

    Geometric Asian options are typically cheaper than arithmetic Asian options
    due to the averaging property of geometric means.
    """

    K: float
    T: float
    is_call: bool = True

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        path = jnp.asarray(path)
        # Geometric mean = exp(mean(log(prices)))
        # Add small epsilon to avoid log(0)
        log_prices = jnp.log(jnp.maximum(path, 1e-10))
        geometric_avg = jnp.exp(log_prices.mean())
        intrinsic = geometric_avg - self.K if self.is_call else self.K - geometric_avg
        return jnp.maximum(intrinsic, 0.0)


@dataclass
class AsianArithmeticFloatingStrike(PathProduct):
    """Arithmetic-average Asian option with floating strike.

    Parameters
    ----------
    T : float
        Time to maturity
    is_call : bool
        True for call, False for put (default: True)

    Notes
    -----
    For floating strike Asian options, the strike is the average price
    and the payoff is based on the terminal price:

    Payoff = max(S_T - Arithmetic_Avg(S), 0) for call
    Payoff = max(Arithmetic_Avg(S) - S_T, 0) for put
    """

    T: float
    is_call: bool = True

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        path = jnp.asarray(path)
        avg = path.mean()
        terminal = path[-1]
        intrinsic = terminal - avg if self.is_call else avg - terminal
        return jnp.maximum(intrinsic, 0.0)


@dataclass
class AsianGeometricFloatingStrike(PathProduct):
    """Geometric-average Asian option with floating strike.

    Parameters
    ----------
    T : float
        Time to maturity
    is_call : bool
        True for call, False for put (default: True)

    Notes
    -----
    For floating strike Asian options, the strike is the geometric average
    and the payoff is based on the terminal price:

    Payoff = max(S_T - Geometric_Avg(S), 0) for call
    Payoff = max(Geometric_Avg(S) - S_T, 0) for put
    """

    T: float
    is_call: bool = True

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        path = jnp.asarray(path)
        log_prices = jnp.log(jnp.maximum(path, 1e-10))
        geometric_avg = jnp.exp(log_prices.mean())
        terminal = path[-1]
        intrinsic = terminal - geometric_avg if self.is_call else geometric_avg - terminal
        return jnp.maximum(intrinsic, 0.0)


__all__ = [
    "AsianArithmetic",
    "AsianGeometric",
    "AsianArithmeticFloatingStrike",
    "AsianGeometricFloatingStrike",
]

