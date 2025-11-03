"""Barrier option payoffs and pricing.

Barrier options are path-dependent options where the payoff depends on whether
the underlying asset price reaches a certain barrier level during the option's life.

Types:
- Knock-Out: Option becomes worthless if barrier is hit
- Knock-In: Option activates only if barrier is hit
- Single Barrier: One barrier level (up or down)
- Double Barrier: Two barrier levels (upper and lower)
"""
from dataclasses import dataclass

import jax.numpy as jnp

from .base import PathProduct
from ._utils import (
    ensure_array,
    extract_terminal,
    vanilla_payoff,
    check_barrier_hit_up,
    check_barrier_hit_down,
)


@dataclass
class UpAndOutCall(PathProduct):
    """Up-and-out barrier call option.

    A call option that becomes worthless if the underlying price
    reaches or exceeds the upper barrier during the option's life.

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Time to maturity
    B : float
        Upper barrier level (B > S0)
    """

    K: float
    T: float
    B: float

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        path = ensure_array(path)
        hit = check_barrier_hit_up(path, self.B)
        ST = extract_terminal(path)
        intrinsic = vanilla_payoff(ST, self.K, is_call=True)
        return jnp.where(hit, 0.0, intrinsic)


@dataclass
class UpAndOutPut(PathProduct):
    """Up-and-out barrier put option.

    A put option that becomes worthless if the underlying price
    reaches or exceeds the upper barrier.

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Time to maturity
    B : float
        Upper barrier level (B > S0)
    """

    K: float
    T: float
    B: float

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        path = ensure_array(path)
        hit = check_barrier_hit_up(path, self.B)
        ST = extract_terminal(path)
        intrinsic = vanilla_payoff(ST, self.K, is_call=False)
        return jnp.where(hit, 0.0, intrinsic)


@dataclass
class DownAndOutCall(PathProduct):
    """Down-and-out barrier call option.

    A call option that becomes worthless if the underlying price
    falls to or below the lower barrier.

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Time to maturity
    B : float
        Lower barrier level (B < S0)
    """

    K: float
    T: float
    B: float

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        path = ensure_array(path)
        hit = check_barrier_hit_down(path, self.B)
        ST = extract_terminal(path)
        intrinsic = vanilla_payoff(ST, self.K, is_call=True)
        return jnp.where(hit, 0.0, intrinsic)


@dataclass
class DownAndOutPut(PathProduct):
    """Down-and-out barrier put option.

    A put option that becomes worthless if the underlying price
    falls to or below the lower barrier.

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Time to maturity
    B : float
        Lower barrier level (B < S0)
    """

    K: float
    T: float
    B: float

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        path = ensure_array(path)
        hit = check_barrier_hit_down(path, self.B)
        ST = extract_terminal(path)
        intrinsic = vanilla_payoff(ST, self.K, is_call=False)
        return jnp.where(hit, 0.0, intrinsic)


@dataclass
class UpAndInCall(PathProduct):
    """Up-and-in barrier call option.

    A call option that activates (becomes standard call) only if
    the underlying price reaches or exceeds the upper barrier.

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Time to maturity
    B : float
        Upper barrier level (B > S0)
    """

    K: float
    T: float
    B: float

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        path = ensure_array(path)
        hit = check_barrier_hit_up(path, self.B)
        ST = extract_terminal(path)
        intrinsic = vanilla_payoff(ST, self.K, is_call=True)
        return jnp.where(hit, intrinsic, 0.0)


@dataclass
class UpAndInPut(PathProduct):
    """Up-and-in barrier put option.

    A put option that activates only if the underlying price
    reaches or exceeds the upper barrier.

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Time to maturity
    B : float
        Upper barrier level (B > S0)
    """

    K: float
    T: float
    B: float

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        path = ensure_array(path)
        hit = check_barrier_hit_up(path, self.B)
        ST = extract_terminal(path)
        intrinsic = vanilla_payoff(ST, self.K, is_call=False)
        return jnp.where(hit, intrinsic, 0.0)


@dataclass
class DownAndInCall(PathProduct):
    """Down-and-in barrier call option.

    A call option that activates only if the underlying price
    falls to or below the lower barrier.

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Time to maturity
    B : float
        Lower barrier level (B < S0)
    """

    K: float
    T: float
    B: float

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        path = ensure_array(path)
        hit = check_barrier_hit_down(path, self.B)
        ST = extract_terminal(path)
        intrinsic = vanilla_payoff(ST, self.K, is_call=True)
        return jnp.where(hit, intrinsic, 0.0)


@dataclass
class DownAndInPut(PathProduct):
    """Down-and-in barrier put option.

    A put option that activates only if the underlying price
    falls to or below the lower barrier.

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Time to maturity
    B : float
        Lower barrier level (B < S0)
    """

    K: float
    T: float
    B: float

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        path = ensure_array(path)
        hit = check_barrier_hit_down(path, self.B)
        ST = extract_terminal(path)
        intrinsic = vanilla_payoff(ST, self.K, is_call=False)
        return jnp.where(hit, intrinsic, 0.0)


@dataclass
class DoubleBarrierCall(PathProduct):
    """Double barrier call option (knock-out).

    A call option that becomes worthless if the underlying price
    reaches either the upper or lower barrier.

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Time to maturity
    B_lower : float
        Lower barrier level (B_lower < S0)
    B_upper : float
        Upper barrier level (B_upper > S0)
    """

    K: float
    T: float
    B_lower: float
    B_upper: float

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        path = ensure_array(path)
        hit_upper = check_barrier_hit_up(path, self.B_upper)
        hit_lower = check_barrier_hit_down(path, self.B_lower)
        hit_either = hit_upper | hit_lower
        ST = extract_terminal(path)
        intrinsic = vanilla_payoff(ST, self.K, is_call=True)
        return jnp.where(hit_either, 0.0, intrinsic)


@dataclass
class DoubleBarrierPut(PathProduct):
    """Double barrier put option (knock-out).

    A put option that becomes worthless if the underlying price
    reaches either the upper or lower barrier.

    Parameters
    ----------
    K : float
        Strike price
    T : float
        Time to maturity
    B_lower : float
        Lower barrier level (B_lower < S0)
    B_upper : float
        Upper barrier level (B_upper > S0)
    """

    K: float
    T: float
    B_lower: float
    B_upper: float

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        path = ensure_array(path)
        hit_upper = check_barrier_hit_up(path, self.B_upper)
        hit_lower = check_barrier_hit_down(path, self.B_lower)
        hit_either = hit_upper | hit_lower
        ST = extract_terminal(path)
        intrinsic = vanilla_payoff(ST, self.K, is_call=False)
        return jnp.where(hit_either, 0.0, intrinsic)


__all__ = [
    "UpAndOutCall",
    "UpAndOutPut",
    "DownAndOutCall",
    "DownAndOutPut",
    "UpAndInCall",
    "UpAndInPut",
    "DownAndInCall",
    "DownAndInPut",
    "DoubleBarrierCall",
    "DoubleBarrierPut",
]
