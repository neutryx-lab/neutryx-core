"""Hazard-rate models and calibration utilities for credit curves."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import jax
import jax.numpy as jnp
from scipy import optimize

ArrayLike = jnp.ndarray


@dataclass
class HazardRateCurve:
    """Piecewise-constant hazard rate term structure.

    Parameters
    ----------
    maturities: Sequence[float]
        Increasing maturities (in years) marking the end of each hazard interval.
    intensities: Sequence[float]
        Default intensities (per year) assumed constant over each interval.
    """

    maturities: ArrayLike
    intensities: ArrayLike

    def __post_init__(self) -> None:
        maturities = jnp.asarray(self.maturities)
        intensities = jnp.asarray(self.intensities)
        if maturities.ndim != 1:
            raise ValueError("Maturities must be a 1D array.")
        if intensities.ndim != 1:
            raise ValueError("Intensities must be a 1D array.")
        if maturities.shape[0] != intensities.shape[0]:
            raise ValueError("Maturities and intensities must have the same length.")
        if not jnp.all(maturities[1:] >= maturities[:-1]):
            raise ValueError("Maturities must be non-decreasing.")
        if jnp.any(intensities < 0):
            raise ValueError("Hazard intensities must be non-negative.")
        self.maturities = maturities
        self.intensities = intensities

    @property
    def intervals(self) -> tuple[ArrayLike, ArrayLike]:
        """Return start and end times for each hazard interval."""

        starts = jnp.concatenate([jnp.array([0.0]), self.maturities[:-1]])
        ends = self.maturities
        return starts, ends

    def value(self, t: ArrayLike) -> ArrayLike:
        """Return hazard intensity λ(t) using piecewise-constant interpolation."""

        t_arr = jnp.asarray(t)
        return jnp.interp(
            t_arr,
            self.maturities,
            self.intensities,
            left=float(self.intensities[0]),
            right=float(self.intensities[-1]),
        )

    def __call__(self, t: ArrayLike) -> ArrayLike:
        return self.value(t)

    def integrated_hazard(self, t: ArrayLike) -> ArrayLike:
        r"""Compute the integrated hazard :math:`\int_0^t \lambda(s) ds`.

        Works for scalar or vector inputs. Values beyond the final
        maturity assume the last hazard intensity persists.
        """

        starts, ends = self.intervals
        lambdas = self.intensities

        def _integrated_single(x: float) -> float:
            x = jnp.asarray(x)
            lengths = jnp.clip(x - starts, 0.0, ends - starts)
            base = jnp.sum(lambdas * lengths)
            tail = lambdas[-1] * jnp.maximum(x - ends[-1], 0.0)
            return base + tail

        t = jnp.asarray(t)
        if t.ndim == 0:
            return _integrated_single(float(t))
        return jax.vmap(_integrated_single)(t)

    def survival_probability(self, t: ArrayLike) -> ArrayLike:
        """Return survival probability :math:`P(\tau > t)` for times ``t``."""

        return jnp.exp(-self.integrated_hazard(t))

    def default_probability(self, t: ArrayLike) -> ArrayLike:
        r"""Return default probability :math:`P(\tau \le t)` for times ``t``."""

        return 1.0 - self.survival_probability(t)

    def with_intensity(self, index: int, value: float) -> "HazardRateCurve":
        """Return a new curve with a single intensity replaced."""

        intensities = self.intensities.at[index].set(value)
        return HazardRateCurve(self.maturities, intensities)


@dataclass
class SurvivalProbabilityCurve:
    """Piecewise-defined survival probability curve."""

    maturities: ArrayLike
    survival_probabilities: ArrayLike

    def __post_init__(self) -> None:
        maturities = jnp.asarray(self.maturities)
        surv = jnp.asarray(self.survival_probabilities)

        if maturities.ndim != 1:
            raise ValueError("Maturities must be a 1D array.")
        if surv.ndim != 1:
            raise ValueError("Survival probabilities must be a 1D array.")
        if maturities.shape[0] != surv.shape[0]:
            raise ValueError("Maturities and survival probabilities must have the same length.")
        if maturities.shape[0] == 0:
            raise ValueError("At least one maturity is required.")
        if not bool(jnp.all(maturities[1:] >= maturities[:-1])):
            raise ValueError("Maturities must be non-decreasing.")
        if not bool(jnp.all((surv >= 0.0) & (surv <= 1.0))):
            raise ValueError("Survival probabilities must be between 0 and 1.")
        if not bool(jnp.all(surv[1:] <= surv[:-1] + 1e-12)):
            raise ValueError("Survival probabilities must be non-increasing.")

        self.maturities = maturities
        self.survival_probabilities = surv

    def value(self, t: ArrayLike) -> ArrayLike:
        return self.survival_probability(t)

    def __call__(self, t: ArrayLike) -> ArrayLike:
        return self.value(t)

    def survival_probability(self, t: ArrayLike) -> ArrayLike:
        t_arr = jnp.asarray(t)
        return jnp.interp(
            t_arr,
            self.maturities,
            self.survival_probabilities,
            left=1.0,
            right=float(self.survival_probabilities[-1]),
        )

    def default_probability(self, t: ArrayLike) -> ArrayLike:
        return 1.0 - self.survival_probability(t)


def _ensure_array(values: Iterable[float]) -> jnp.ndarray:
    arr = jnp.asarray(list(values))
    if arr.ndim != 1:
        raise ValueError("Input must be one-dimensional.")
    return arr


def cds_premium_leg(
    spread: float,
    payment_times: Sequence[float],
    discount_factors: Sequence[float],
    survival_prob: Sequence[float],
    accrual_fractions: Sequence[float],
) -> jnp.ndarray:
    """Present value of the CDS premium leg."""

    payment_times = _ensure_array(payment_times)
    discount_factors = _ensure_array(discount_factors)
    survival_prob = _ensure_array(survival_prob)
    accrual_fractions = _ensure_array(accrual_fractions)

    return spread * jnp.sum(discount_factors * survival_prob * accrual_fractions)


def cds_protection_leg(
    payment_times: Sequence[float],
    discount_factors: Sequence[float],
    survival_prob: Sequence[float],
    recovery: float,
) -> jnp.ndarray:
    """Present value of the CDS protection leg."""

    payment_times = _ensure_array(payment_times)
    discount_factors = _ensure_array(discount_factors)
    survival_prob = _ensure_array(survival_prob)

    survival_prev = jnp.concatenate([jnp.array([1.0]), survival_prob[:-1]])
    default_probs = survival_prev - survival_prob
    return (1.0 - recovery) * jnp.sum(discount_factors * default_probs)


def calibrate_piecewise_hazard(
    payment_times: Sequence[float],
    discount_factors: Sequence[float],
    market_spreads: Sequence[float],
    recovery: float = 0.4,
    accrual_fractions: Sequence[float] | None = None,
    hazard_bounds: tuple[float, float] = (1e-6, 5.0),
) -> HazardRateCurve:
    """Calibrate a piecewise-constant hazard curve from CDS par spreads.

    Parameters
    ----------
    payment_times:
        CDS payment times in years.
    discount_factors:
        Discount factors for each payment time.
    market_spreads:
        Observed market par spreads quoted as decimal (e.g. 0.01 for 100 bps).
    recovery:
        Assumed recovery rate.
    accrual_fractions:
        Accrual fractions for each interval. When ``None`` they are inferred
        from successive payment times.
    hazard_bounds:
        Bracketing interval used for solving each hazard rate.
    """

    payment_times = _ensure_array(payment_times)
    discount_factors = _ensure_array(discount_factors)
    market_spreads = _ensure_array(market_spreads)

    if accrual_fractions is None:
        accrual_fractions = jnp.diff(jnp.concatenate([jnp.array([0.0]), payment_times]))
    else:
        accrual_fractions = _ensure_array(accrual_fractions)

    if (
        payment_times.shape[0] != discount_factors.shape[0]
        or payment_times.shape[0] != market_spreads.shape[0]
        or payment_times.shape[0] != accrual_fractions.shape[0]
    ):
        raise ValueError("Input arrays must have identical length.")

    calibrated_intensities: list[float] = []

    for i in range(payment_times.shape[0]):
        times = payment_times[: i + 1]
        dfs = discount_factors[: i + 1]
        accruals = accrual_fractions[: i + 1]
        spread = float(market_spreads[i])

        def objective(lam: float) -> float:
            intensities = jnp.asarray(calibrated_intensities + [lam])
            curve = HazardRateCurve(times, intensities)
            survival = curve.survival_probability(times)
            premium = cds_premium_leg(spread, times, dfs, survival, accruals)
            protection = cds_protection_leg(times, dfs, survival, recovery)
            return float(premium - protection)

        lam = optimize.brentq(objective, *hazard_bounds)
        calibrated_intensities.append(lam)

    return HazardRateCurve(payment_times, jnp.asarray(calibrated_intensities))
