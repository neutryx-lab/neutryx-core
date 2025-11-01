"""Structural default models and calibration helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp
from jax.scipy.stats import norm
from scipy import optimize


@dataclass
class MertonModel:
    """Classic Merton structural credit model."""

    asset_value: float
    debt: float
    maturity: float
    asset_vol: float
    rate: float
    payout: float = 0.0

    def _d1d2(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        sigma_sqrt_t = self.asset_vol * jnp.sqrt(self.maturity)
        if sigma_sqrt_t <= 0:
            raise ValueError("Asset volatility must be positive.")
        log_ratio = jnp.log(self.asset_value / self.debt)
        drift = (self.rate - self.payout + 0.5 * self.asset_vol**2) * self.maturity
        d1 = (log_ratio + drift) / sigma_sqrt_t
        d2 = d1 - sigma_sqrt_t
        return d1, d2

    def survival_probability(self) -> jnp.ndarray:
        """Probability firm value exceeds debt at maturity."""

        _, d2 = self._d1d2()
        return norm.cdf(d2)

    def default_probability(self) -> jnp.ndarray:
        return 1.0 - self.survival_probability()

    def equity_value(self) -> jnp.ndarray:
        d1, d2 = self._d1d2()
        disc_asset = self.asset_value * jnp.exp(-self.payout * self.maturity)
        disc_debt = self.debt * jnp.exp(-self.rate * self.maturity)
        return disc_asset * norm.cdf(d1) - disc_debt * norm.cdf(d2)

    def debt_value(self) -> jnp.ndarray:
        return self.asset_value - self.equity_value()

    def credit_spread(self) -> jnp.ndarray:
        debt_val = self.debt_value()
        risk_free = self.debt * jnp.exp(-self.rate * self.maturity)
        spread = -jnp.log(jnp.clip(debt_val / risk_free, 1e-12)) / self.maturity
        return spread

    def distance_to_default(self) -> jnp.ndarray:
        """Number of asset standard deviations away from default barrier."""

        _, d2 = self._d1d2()
        return d2


def calibrate_merton_from_equity(
    debt: float,
    maturity: float,
    rate: float,
    equity_value: float,
    equity_vol: float,
    payout: float = 0.0,
) -> MertonModel:
    """Recover asset value and volatility from equity inputs."""

    if equity_value <= 0:
        raise ValueError("Equity value must be positive.")
    if equity_vol <= 0:
        raise ValueError("Equity volatility must be positive.")

    def residuals(x: jnp.ndarray) -> jnp.ndarray:
        asset_value, asset_vol = x
        model = MertonModel(asset_value, debt, maturity, asset_vol, rate, payout)
        equity_model = model.equity_value()
        d1, _ = model._d1d2()
        implied_equity_vol = asset_vol * asset_value * norm.cdf(d1) / equity_model
        return jnp.array([
            equity_model - equity_value,
            implied_equity_vol - equity_vol,
        ])

    initial_asset = debt + equity_value
    initial_sigma = equity_vol * equity_value / initial_asset
    guess = jnp.array([initial_asset, jnp.maximum(initial_sigma, 1e-4)])

    result = optimize.root(lambda x: residuals(jnp.array(x)), guess, method="hybr")
    if not result.success:
        raise RuntimeError("Merton calibration failed: " + result.message)

    calibrated_asset, calibrated_sigma = result.x
    return MertonModel(calibrated_asset, debt, maturity, calibrated_sigma, rate, payout)


@dataclass
class BlackCoxModel:
    """Black–Cox first-passage structural default model."""

    asset_value: float
    debt: float
    barrier: float
    maturity: float
    asset_vol: float
    rate: float
    payout: float = 0.0
    recovery_rate: float = 0.4

    def survival_probability(self) -> jnp.ndarray:
        if self.asset_vol <= 0:
            raise ValueError("Asset volatility must be positive.")
        log_ratio = jnp.log(self.asset_value / self.barrier)
        drift = self.rate - self.payout - 0.5 * self.asset_vol**2
        sigma = self.asset_vol
        sqrt_term = sigma * jnp.sqrt(self.maturity)
        drift_term = drift * self.maturity
        term1 = norm.cdf((log_ratio + drift_term) / sqrt_term)
        exponent = -2.0 * drift * log_ratio / (sigma**2)
        term2 = jnp.exp(exponent) * norm.cdf((-log_ratio + drift_term) / sqrt_term)
        survival = term1 - term2
        return jnp.clip(survival, 0.0, 1.0)

    def default_probability(self) -> jnp.ndarray:
        return 1.0 - self.survival_probability()

    def credit_spread(self) -> jnp.ndarray:
        survival = self.survival_probability()
        effective_recovery = self.recovery_rate + (1.0 - self.recovery_rate) * survival
        debt_price = self.debt * jnp.exp(-self.rate * self.maturity) * effective_recovery
        risk_free = self.debt * jnp.exp(-self.rate * self.maturity)
        spread = -jnp.log(jnp.clip(debt_price / risk_free, 1e-12)) / self.maturity
        return spread


def calibrate_black_cox_barrier(
    asset_value: float,
    debt: float,
    maturity: float,
    asset_vol: float,
    rate: float,
    target_default_prob: float,
    payout: float = 0.0,
    recovery_rate: float = 0.4,
    barrier_bounds: tuple[float, float] = (1e-3, 0.99),
) -> BlackCoxModel:
    """Calibrate barrier level to match a target default probability."""

    if not (0.0 < target_default_prob < 1.0):
        raise ValueError("Target default probability must lie in (0, 1).")

    lower_frac, upper_frac = barrier_bounds
    lower = lower_frac * debt
    upper = upper_frac * debt

    def objective(barrier: float) -> float:
        model = BlackCoxModel(
            asset_value,
            debt,
            barrier,
            maturity,
            asset_vol,
            rate,
            payout,
            recovery_rate,
        )
        return float(model.default_probability() - target_default_prob)

    barrier = optimize.brentq(objective, lower, upper)
    return BlackCoxModel(
        asset_value,
        debt,
        barrier,
        maturity,
        asset_vol,
        rate,
        payout,
        recovery_rate,
    )
