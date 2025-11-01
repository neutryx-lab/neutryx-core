"""Credit default swap valuation utilities."""
from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp

from .hazard import HazardRateCurve, _ensure_array, cds_premium_leg, cds_protection_leg


def cds_pv(
    spread: float,
    payment_times: Sequence[float],
    discount_factors: Sequence[float],
    hazard_curve: HazardRateCurve,
    recovery: float = 0.4,
    accrual_fractions: Sequence[float] | None = None,
) -> jnp.ndarray:
    """Present value of a CDS given a hazard-rate curve."""

    payment_times = _ensure_array(payment_times)
    discount_factors = _ensure_array(discount_factors)
    if accrual_fractions is None:
        accrual_fractions = jnp.diff(jnp.concatenate([jnp.array([0.0]), payment_times]))
    else:
        accrual_fractions = _ensure_array(accrual_fractions)

    if (
        payment_times.shape[0] != discount_factors.shape[0]
        or payment_times.shape[0] != accrual_fractions.shape[0]
    ):
        raise ValueError("Payment times, discount factors, and accrual fractions must align.")

    survival = hazard_curve.survival_probability(payment_times)
    premium = cds_premium_leg(spread, payment_times, discount_factors, survival, accrual_fractions)
    protection = cds_protection_leg(payment_times, discount_factors, survival, recovery)
    return premium - protection


def cds_par_spread(
    payment_times: Sequence[float],
    discount_factors: Sequence[float],
    hazard_curve: HazardRateCurve,
    recovery: float = 0.4,
    accrual_fractions: Sequence[float] | None = None,
) -> jnp.ndarray:
    """Compute the par spread that sets CDS PV to zero."""

    payment_times = _ensure_array(payment_times)
    discount_factors = _ensure_array(discount_factors)
    if accrual_fractions is None:
        accrual_fractions = jnp.diff(jnp.concatenate([jnp.array([0.0]), payment_times]))
    else:
        accrual_fractions = _ensure_array(accrual_fractions)

    survival = hazard_curve.survival_probability(payment_times)
    premium_denominator = cds_premium_leg(
        1.0, payment_times, discount_factors, survival, accrual_fractions
    )
    protection = cds_protection_leg(payment_times, discount_factors, survival, recovery)
    return jnp.where(premium_denominator > 0, protection / premium_denominator, 0.0)
