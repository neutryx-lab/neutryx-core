"""Unified short-rate interest rate models with analytic solvers.

This module provides a thin abstraction layer above the existing two-factor
Gaussian (G2++) and Quasi-Gaussian (QG) implementations.  The goal is to offer
an SDE-centric interface – similar in spirit to :mod:`equity_models` – that
exposes drift/diffusion dynamics, analytic term-structure solvers and Monte
Carlo helpers in a single place.

Two concrete model classes are provided:

``G2PPInterestRateModel``
    Wrapper around :class:`~neutryx.models.g2pp.G2PPParams` offering
    instantaneous drift/diffusion, closed-form bond pricing and path
    generation.

``QuasiGaussianInterestRateModel``
    Companion wrapper for :class:`~neutryx.models.quasi_gaussian.QuasiGaussianParams`
    that retains compatibility with time-dependent mean-reversion/volatility
    functions while exposing the same ergonomic interface.

Both classes inherit from :class:`~neutryx.models.sde.SDE`, enabling reuse of
the stochastic calculus utilities across the library.  Analytical bond pricing
is delegated to the underlying model implementations to avoid duplicating
complex formulae while keeping an explicit, documented API for calibration and
testing purposes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from .g2pp import G2PPParams, zero_coupon_bond_price as g2pp_zero_coupon
from .g2pp import simulate_paths as g2pp_simulate_paths
from .quasi_gaussian import (
    QuasiGaussianParams,
    zero_coupon_bond_price as qg_zero_coupon,
    simulate_paths as qg_simulate_paths,
)
from .sde import SDE

ArrayLike = Array


def _cholesky_from_correlation(rho: float) -> Array:
    """Return the Cholesky factor for a 2×2 correlation matrix.

    The helper is shared by both models to construct diffusion matrices with a
    consistent convention.
    """

    rho_clipped = jnp.clip(rho, -0.999, 0.999)
    return jnp.array([[1.0, 0.0], [rho_clipped, jnp.sqrt(1.0 - rho_clipped**2)]])


def _affine_variance_term(
    a: float,
    b: float,
    sigma_x: float,
    sigma_y: float,
    rho: float,
    tau: float,
) -> float:
    """Integrated variance of the short rate for constant-parameter OU factors.

    This recovers the well-known expressions used in the G2++ model and is
    reused in :meth:`G2PPInterestRateModel.conditional_moments` for analytical
    variance evaluation.
    """

    var_x = (sigma_x**2 / (2.0 * a)) * (1.0 - jnp.exp(-2.0 * a * tau))
    var_y = (sigma_y**2 / (2.0 * b)) * (1.0 - jnp.exp(-2.0 * b * tau))
    cov_xy = (
        rho
        * sigma_x
        * sigma_y
        / (a + b)
        * (1.0 - jnp.exp(-(a + b) * tau))
    )

    # Convert factor variances into short-rate variance for r = x + y + phi
    return var_x + var_y + 2.0 * cov_xy


@dataclass
class InterestRateModel(SDE):
    """Common interface for short-rate models."""

    def zero_coupon_bond_price(
        self,
        maturity: float,
        *,
        t: float = 0.0,
        state: Optional[Sequence[float]] = None,
    ) -> float:
        """Analytical zero-coupon bond price ``P(t, maturity)``."""

        raise NotImplementedError

    def bond_yield(
        self,
        maturity: float,
        *,
        t: float = 0.0,
        state: Optional[Sequence[float]] = None,
    ) -> float:
        """Continuously-compounded yield implied by :meth:`zero_coupon_bond_price`."""

        price = self.zero_coupon_bond_price(maturity, t=t, state=state)
        tau = jnp.maximum(maturity - t, 1e-12)
        return -jnp.log(price) / tau

    def simulate_paths(
        self,
        *,
        horizon: float,
        steps: int,
        paths: int,
        key: Array,
    ) -> Tuple[Array, Array]:
        """Simulate short-rate paths returning the factor trajectories."""

        raise NotImplementedError


@dataclass
class G2PPInterestRateModel(InterestRateModel):
    """Two-factor Gaussian short-rate model with analytic bond pricing."""

    params: G2PPParams

    # ------------------------------------------------------------------
    # SDE interface
    # ------------------------------------------------------------------
    def drift(self, t: float, state: ArrayLike) -> Array:
        del t  # Time-homogeneous
        x, y = state
        return jnp.array([
            -self.params.a * x,
            -self.params.b * y,
        ])

    def diffusion(self, t: float, state: ArrayLike) -> Array:
        del t, state  # Time-homogeneous
        chol = _cholesky_from_correlation(self.params.rho)
        vol = jnp.diag(jnp.array([self.params.sigma_x, self.params.sigma_y]))
        return vol @ chol

    # ------------------------------------------------------------------
    # Analytic utilities
    # ------------------------------------------------------------------
    def zero_coupon_bond_price(
        self,
        maturity: float,
        *,
        t: float = 0.0,
        state: Optional[Sequence[float]] = None,
    ) -> float:
        x_t, y_t = (state if state is not None else (self.params.x0, self.params.y0))
        return g2pp_zero_coupon(self.params, maturity, x_t=x_t, y_t=y_t, t=t)

    def conditional_moments(
        self,
        *,
        t: float,
        horizon: float,
        state: Optional[Sequence[float]] = None,
    ) -> Tuple[float, float]:
        """Return ``(mean, variance)`` of ``r(t+horizon)`` conditional on ``state``."""

        x_t, y_t = (state if state is not None else (self.params.x0, self.params.y0))
        tau = jnp.maximum(horizon, 0.0)
        exp_x = x_t * jnp.exp(-self.params.a * tau)
        exp_y = y_t * jnp.exp(-self.params.b * tau)

        phi = self.params.phi_fn(t + tau) if self.params.phi_fn is not None else 0.0
        mean = phi + exp_x + exp_y

        variance = _affine_variance_term(
            self.params.a,
            self.params.b,
            self.params.sigma_x,
            self.params.sigma_y,
            self.params.rho,
            tau,
        )
        return mean, variance

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def simulate_paths(
        self,
        *,
        horizon: float,
        steps: int,
        paths: int,
        key: Array,
    ) -> Tuple[Array, Array]:
        r_paths, x_paths, y_paths = g2pp_simulate_paths(
            self.params,
            T=horizon,
            n_steps=steps,
            n_paths=paths,
            key=key,
        )
        return r_paths, jnp.stack([x_paths, y_paths], axis=-1)


@dataclass
class QuasiGaussianInterestRateModel(InterestRateModel):
    """Quasi-Gaussian short-rate model with time-dependent parameters."""

    params: QuasiGaussianParams

    def drift(self, t: float, state: ArrayLike) -> Array:
        x, y = state
        return jnp.array([
            -self.params.alpha_fn(float(t)) * x,
            -self.params.beta_fn(float(t)) * y,
        ])

    def diffusion(self, t: float, state: ArrayLike) -> Array:
        del state
        sigma_x = self.params.sigma_x_fn(float(t))
        sigma_y = self.params.sigma_y_fn(float(t))
        rho = self.params.get_rho(float(t))
        chol = _cholesky_from_correlation(rho)
        vol = jnp.diag(jnp.array([sigma_x, sigma_y]))
        return vol @ chol

    def zero_coupon_bond_price(
        self,
        maturity: float,
        *,
        t: float = 0.0,
        state: Optional[Sequence[float]] = None,
    ) -> float:
        x_t, y_t = (state if state is not None else (self.params.x0, self.params.y0))
        return qg_zero_coupon(self.params, maturity, x_t=x_t, y_t=y_t, t=t)

    def simulate_paths(
        self,
        *,
        horizon: float,
        steps: int,
        paths: int,
        key: Array,
    ) -> Tuple[Array, Array]:
        r_paths, x_paths, y_paths = qg_simulate_paths(
            self.params,
            T=horizon,
            n_steps=steps,
            n_paths=paths,
            key=key,
        )
        return r_paths, jnp.stack([x_paths, y_paths], axis=-1)


def create_interest_rate_model(
    model: str,
    params: G2PPParams | QuasiGaussianParams,
) -> InterestRateModel:
    """Factory returning the appropriate interest-rate model wrapper."""

    model_lower = model.lower()
    if model_lower in {"g2pp", "g2++", "gaussian"}:
        if not isinstance(params, G2PPParams):
            raise TypeError("G2PP model requires G2PPParams")
        return G2PPInterestRateModel(params)
    if model_lower in {"quasi_gaussian", "qg", "quasi-gaussian"}:
        if not isinstance(params, QuasiGaussianParams):
            raise TypeError("Quasi-Gaussian model requires QuasiGaussianParams")
        return QuasiGaussianInterestRateModel(params)
    raise ValueError(f"Unsupported interest-rate model '{model}'")


__all__ = [
    "InterestRateModel",
    "G2PPInterestRateModel",
    "QuasiGaussianInterestRateModel",
    "create_interest_rate_model",
]
