"""Pathwise differentiation utilities for American Monte Carlo (AMC).

The routines defined here build on top of the core Monte Carlo engine and
provide a high level interface that produces prices together with
first-order Greeks in a single differentiation pass.  The implementation
leans on JAX's automatic differentiation capabilities while keeping the
path simulation explicit which in turn allows deterministic regression
tests for a fixed random seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Iterable, Mapping, Sequence

import jax
import jax.numpy as jnp

from neutryx.core.engine import MCConfig

Array = jnp.ndarray


@dataclass(frozen=True)
class AMCInputs:
    """Parameter bundle used for AMC pathwise differentiation."""

    S0: float
    r: float
    q: float
    sigma: float
    T: float


@dataclass(frozen=True)
class PathwisePayoff:
    """Container describing a payoff evaluated on simulated paths."""

    name: str
    fn: Callable[[Array, AMCInputs], Array]

    def __call__(self, paths: Array, params: AMCInputs) -> Array:
        return self.fn(paths, params)


@dataclass
class PathwiseResult:
    """Holds price and Greeks obtained from a pathwise run."""

    price: float
    delta: float
    rho: float
    dividend: float
    vega: float

    def as_dict(self) -> Mapping[str, float]:
        return {
            "price": self.price,
            "delta": self.delta,
            "rho": self.rho,
            "dividend": self.dividend,
            "vega": self.vega,
        }


def _simulate_from_normals(normals: Array, params: AMCInputs, cfg: MCConfig) -> Array:
    """Generate GBM paths from pre-sampled standard normal increments."""

    dtype = cfg.dtype
    normals = jnp.asarray(normals, dtype=dtype)

    dt = params.T / cfg.steps
    drift = (params.r - params.q - 0.5 * params.sigma * params.sigma) * dt
    vol = params.sigma * jnp.sqrt(dt)

    increments = drift + vol * normals
    if cfg.antithetic:
        increments = jnp.concatenate([increments, drift + vol * (-normals)], axis=0)

    log_S0 = jnp.log(jnp.asarray(params.S0, dtype=dtype))
    cumulative = jnp.cumsum(increments, axis=1)
    log_paths = jnp.concatenate(
        [jnp.full((increments.shape[0], 1), log_S0, dtype=dtype), log_S0 + cumulative],
        axis=1,
    )
    return jnp.exp(log_paths)


def european_call(strike: float) -> PathwisePayoff:
    """European call payoff discounted at maturity."""

    def _fn(paths: Array, params: AMCInputs) -> Array:
        ST = paths[:, -1]
        payoff = jnp.maximum(ST - strike, 0.0)
        disc = jnp.exp(-params.r * params.T)
        return disc * payoff

    return PathwisePayoff(name=f"european_call_{strike:g}", fn=_fn)


def european_put(strike: float) -> PathwisePayoff:
    """European put payoff discounted at maturity."""

    def _fn(paths: Array, params: AMCInputs) -> Array:
        ST = paths[:, -1]
        payoff = jnp.maximum(strike - ST, 0.0)
        disc = jnp.exp(-params.r * params.T)
        return disc * payoff

    return PathwisePayoff(name=f"european_put_{strike:g}", fn=_fn)


def asian_arithmetic_call(strike: float) -> PathwisePayoff:
    """Arithmetic Asian call payoff using the full path average."""

    def _fn(paths: Array, params: AMCInputs) -> Array:
        avg = paths[:, 1:].mean(axis=1)
        payoff = jnp.maximum(avg - strike, 0.0)
        disc = jnp.exp(-params.r * params.T)
        return disc * payoff

    return PathwisePayoff(name=f"asian_call_{strike:g}", fn=_fn)


def asian_arithmetic_put(strike: float) -> PathwisePayoff:
    """Arithmetic Asian put payoff using the full path average."""

    def _fn(paths: Array, params: AMCInputs) -> Array:
        avg = paths[:, 1:].mean(axis=1)
        payoff = jnp.maximum(strike - avg, 0.0)
        disc = jnp.exp(-params.r * params.T)
        return disc * payoff

    return PathwisePayoff(name=f"asian_put_{strike:g}", fn=_fn)


def _lsm_regression(matrix: Array, targets: Array, mask: Array) -> Array:
    """Solve the least squares regression used in LSM with masking."""

    weight = mask.astype(matrix.dtype)

    def _solve(_: None) -> Array:
        sqrt_w = jnp.sqrt(weight)[:, None]
        design = matrix * sqrt_w
        rhs = targets * sqrt_w.squeeze()
        gram = design.T @ design
        ridge = 1e-8 * jnp.eye(gram.shape[0], dtype=matrix.dtype)
        beta = jnp.linalg.solve(gram + ridge, design.T @ rhs)
        return beta

    return jax.lax.cond(
        jnp.any(mask),
        _solve,
        lambda _: jnp.zeros((matrix.shape[1],), dtype=matrix.dtype),
        operand=None,
    )


def american_put_lsm(strike: float) -> PathwisePayoff:
    """Least-Squares Monte Carlo payoff for an American put."""

    def _fn(paths: Array, params: AMCInputs) -> Array:
        paths = jnp.asarray(paths)
        steps = paths.shape[1] - 1
        dt = params.T / steps
        payoff = jnp.maximum(strike - paths, 0.0)

        cashflow = payoff[:, -1]

        for t in range(steps - 1, 0, -1):
            St = paths[:, t]
            immediate = payoff[:, t]
            discounted_cf = cashflow * jnp.exp(-params.r * dt)

            basis = jnp.stack([jnp.ones_like(St), St, St * St], axis=1)
            mask = immediate > 0
            beta = _lsm_regression(basis, discounted_cf, mask)
            continuation_value = jnp.sum(basis * beta, axis=1)
            exercise = (immediate >= continuation_value) & mask

            cashflow = jnp.where(exercise, immediate, discounted_cf)

        pv_paths = cashflow * jnp.exp(-params.r * dt)
        immediate0 = payoff[:, 0]
        return jnp.maximum(pv_paths, immediate0)

    return PathwisePayoff(name=f"american_put_{strike:g}", fn=_fn)


def _price_vector(
    param_vec: Array,
    normals: Array,
    cfg: MCConfig,
    payoffs: Sequence[PathwisePayoff],
    maturity: float,
) -> Array:
    params = AMCInputs(
        S0=param_vec[0],
        r=param_vec[1],
        q=param_vec[2],
        sigma=param_vec[3],
        T=maturity,
    )
    paths = _simulate_from_normals(normals, params, cfg)
    payoff_stack = jnp.stack([payoff(paths, params) for payoff in payoffs])
    return payoff_stack.mean(axis=1)


def pathwise_price_and_greeks(
    key: jax.random.KeyArray,
    params: AMCInputs,
    cfg: MCConfig,
    payoffs: Iterable[PathwisePayoff],
) -> Mapping[str, PathwiseResult]:
    """Return price and Greeks for multiple payoffs from a single simulation.

    Parameters
    ----------
    key:
        PRNG key controlling the standard normal samples.
    params:
        Parameters describing the GBM dynamics.
    cfg:
        Monte Carlo configuration controlling discretisation and sampling.
    payoffs:
        Iterable of :class:`PathwisePayoff` definitions.  The payoff function
        must return *discounted* path-wise cash flows so that taking the mean
        yields the present value.
    """

    payoffs = tuple(payoffs)
    if not payoffs:
        raise ValueError("At least one payoff must be supplied.")

    normals = jax.random.normal(key, (cfg.base_paths, cfg.steps), dtype=cfg.dtype)

    param_vec = jnp.asarray([params.S0, params.r, params.q, params.sigma], dtype=cfg.dtype)

    price_fn = partial(
        _price_vector,
        normals=normals,
        cfg=cfg,
        payoffs=payoffs,
        maturity=float(params.T),
    )

    values = price_fn(param_vec)
    jac_rows = jax.jacrev(price_fn)(param_vec)

    results = {}
    for idx, payoff in enumerate(payoffs):
        price = float(values[idx])
        grads = jac_rows[idx]
        results[payoff.name] = PathwiseResult(
            price=price,
            delta=float(grads[0]),
            rho=float(grads[1]),
            dividend=float(grads[2]),
            vega=float(grads[3]),
        )

    return results


__all__ = [
    "AMCInputs",
    "PathwisePayoff",
    "PathwiseResult",
    "american_put_lsm",
    "asian_arithmetic_call",
    "asian_arithmetic_put",
    "european_call",
    "european_put",
    "pathwise_price_and_greeks",
]

