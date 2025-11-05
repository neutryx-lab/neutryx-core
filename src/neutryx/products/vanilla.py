"""Standard vanilla payoffs."""
from dataclasses import dataclass

import jax.numpy as jnp

from .base import Product, PathProduct


@dataclass
class European(Product):
    """European vanilla option payoff.

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Time to maturity in years
    is_call : bool
        True for call option, False for put option

    Notes
    -----
    European options can only be exercised at maturity.
    Payoff:
    - Call: max(S_T - K, 0)
    - Put: max(K - S_T, 0)
    """

    K: float
    T: float
    is_call: bool = True

    def payoff_terminal(self, spot: jnp.ndarray) -> jnp.ndarray:
        spot = jnp.asarray(spot)
        intrinsic = spot - self.K if self.is_call else self.K - spot
        return jnp.maximum(intrinsic, 0.0)


@dataclass
class American(PathProduct):
    """American vanilla option payoff (requires LSM pricing).

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Time to maturity in years
    is_call : bool
        True for call option, False for put option

    Notes
    -----
    American options can be exercised at any time up to maturity.
    This class provides the immediate exercise payoff at each time step.

    For pricing, use the Longstaff-Schwartz Monte Carlo (LSM) engine
    which performs backward induction to find optimal exercise policy.

    Immediate exercise payoff:
    - Call: max(S_t - K, 0)
    - Put: max(K - S_t, 0)

    Example
    -------
    >>> from neutryx.products import American
    >>> from neutryx.engines.american import american_put_lsm
    >>> import jax.numpy as jnp
    >>> # Create American put
    >>> put = American(K=100.0, T=1.0, is_call=False)
    >>> # Generate paths (example)
    >>> paths = jnp.array([[100, 95, 90], [100, 105, 110]])
    >>> # Price using LSM
    >>> price = american_put_lsm(paths, K=100.0, r=0.05, dt=0.5)
    """

    K: float
    T: float
    is_call: bool = True

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        """Compute immediate exercise value at terminal time.

        For American options, this is used with LSM backward induction
        to determine optimal exercise strategy.
        """
        path = jnp.asarray(path)
        spot = path[-1]  # Terminal value
        intrinsic = spot - self.K if self.is_call else self.K - spot
        return jnp.maximum(intrinsic, 0.0)

    def immediate_exercise(self, spot: jnp.ndarray) -> jnp.ndarray:
        """Compute immediate exercise value at any time.

        Parameters
        ----------
        spot : Array
            Spot prices (can be entire path or single value)

        Returns
        -------
        Array
            Immediate exercise values
        """
        spot = jnp.asarray(spot)
        intrinsic = spot - self.K if self.is_call else self.K - spot
        return jnp.maximum(intrinsic, 0.0)

