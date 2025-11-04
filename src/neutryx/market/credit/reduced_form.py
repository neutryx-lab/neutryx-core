"""Reduced-form (intensity-based) credit risk models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import jax.numpy as jnp

from .hazard import HazardRateCurve

Array = jnp.ndarray


class DiscountCurve(Protocol):
    """Protocol for discount curves compatible with JAX arrays."""

    def __call__(self, t: Array) -> Array:  # pragma: no cover - structural typing only
        ...


def _as_array(values: Sequence[float] | Array) -> Array:
    arr = jnp.asarray(values)
    if arr.ndim == 0:
        arr = arr[None]
    return arr


def _integrated_hazard(hazard: float | HazardRateCurve, t: Array) -> Array:
    if isinstance(hazard, HazardRateCurve):
        return hazard.integrated_hazard(t)
    return hazard * t


def _survival_probability(hazard: float | HazardRateCurve, t: Array) -> Array:
    if isinstance(hazard, HazardRateCurve):
        return hazard.survival_probability(t)
    return jnp.exp(-hazard * t)


@dataclass
class ReducedFormModel:
    """Base utilities for intensity-based credit models."""

    hazard: float | HazardRateCurve
    recovery_rate: float
    risk_free_rate: float | DiscountCurve

    def _discount_factor(self, t: Array) -> Array:
        if callable(self.risk_free_rate):  # term-structure aware
            return jnp.asarray(self.risk_free_rate(t))
        return jnp.exp(-float(self.risk_free_rate) * t)

    def survival_probability(self, t: Sequence[float] | Array) -> Array:
        t_arr = _as_array(t)
        return _survival_probability(self.hazard, t_arr)

    def default_probability(self, t: Sequence[float] | Array) -> Array:
        return 1.0 - self.survival_probability(t)

    def zero_coupon_price(self, maturity: float) -> Array:
        maturity_arr = jnp.asarray(maturity)
        hazard_term = (1.0 - self.recovery_rate) * _integrated_hazard(self.hazard, maturity_arr)
        discount = self._discount_factor(maturity_arr)
        return discount * jnp.exp(-hazard_term)

    def credit_spread(self, maturity: float) -> Array:
        price = self.zero_coupon_price(maturity)
        risk_free = self._discount_factor(jnp.asarray(maturity))
        return -jnp.log(jnp.clip(price / risk_free, 1e-12)) / maturity

    def coupon_bond_price(
        self,
        maturity: float,
        coupon_rate: float,
        frequency: int = 2,
        face_value: float = 100.0,
    ) -> Array:
        dt = 1.0 / frequency
        num_payments = int(jnp.round(maturity * frequency))
        times = jnp.arange(1, num_payments + 1) * dt
        coupon_cf = face_value * coupon_rate / frequency
        discount_factors = jnp.asarray([self.zero_coupon_price(float(t)) for t in times])
        pv_coupons = jnp.sum(coupon_cf * discount_factors)
        pv_principal = face_value * self.zero_coupon_price(float(maturity))
        return pv_coupons + pv_principal


@dataclass
class JarrowTurnbullModel(ReducedFormModel):
    """Jarrow–Turnbull reduced-form credit model with liquidity premium."""

    liquidity_spread: float = 0.0

    def zero_coupon_price(self, maturity: float) -> Array:
        base_price = super().zero_coupon_price(maturity)
        liquidity_adjustment = jnp.exp(-self.liquidity_spread * maturity)
        return base_price * liquidity_adjustment


@dataclass
class DuffieSingletonModel(ReducedFormModel):
    """Duffie–Singleton model with jump-to-default recovery of market value."""

    def default_density(self, t: Sequence[float] | Array) -> Array:
        t_arr = _as_array(t)
        survival = self.survival_probability(t_arr)
        integrated = _integrated_hazard(self.hazard, t_arr)
        # Derivative of survival probability with respect to t under FRMV
        lambdas = jnp.gradient(integrated, t_arr, edge_order=2)
        return lambdas * jnp.exp(-integrated)

    def loss_leg_pv(
        self,
        payment_times: Sequence[float],
        discount_curve: DiscountCurve | None = None,
    ) -> Array:
        payment_arr = _as_array(payment_times)
        discount = self._discount_factor(payment_arr)
        if discount_curve is not None:
            discount = jnp.asarray(discount_curve(payment_arr))
        survival = self.survival_probability(payment_arr)
        prev_survival = jnp.concatenate([jnp.array([1.0]), survival[:-1]])
        default_probs = prev_survival - survival
        return (1.0 - self.recovery_rate) * jnp.sum(discount * default_probs)


def calibrate_constant_intensity(spread: float, recovery_rate: float) -> float:
    """Calibrate constant default intensity from an observed credit spread."""

    if not (0.0 <= recovery_rate < 1.0):
        raise ValueError("Recovery rate must be in [0, 1).")
    if spread < 0:
        raise ValueError("Spread must be non-negative.")
    return spread / max(1.0 - recovery_rate, 1e-12)
