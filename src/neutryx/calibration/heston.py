"""Heston model calibration controller."""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import jax
import jax.numpy as jnp
import optax

from neutryx.solver.heston import HestonParams, heston_call_price

from . import losses
from .base import CalibrationController, ParameterSpec
from .constraints import positive, symmetric

Array = jnp.ndarray


def default_parameter_specs() -> Mapping[str, ParameterSpec]:
    return {
        "v0": ParameterSpec(0.04, positive()),
        "kappa": ParameterSpec(1.5, positive()),
        "theta": ParameterSpec(0.04, positive()),
        "sigma": ParameterSpec(0.3, positive()),
        "rho": ParameterSpec(-0.5, symmetric(0.999)),
    }


def generate_heston_market_data(
    spot: float,
    strikes: Array,
    maturities: Array,
    params: HestonParams,
    *,
    rate: float = 0.0,
    dividend: float = 0.0,
    noise_std: float = 0.0,
    key: Optional[jax.Array] = None,
    weights: Optional[Array] = None,
) -> Mapping[str, Array]:
    strikes = jnp.asarray(strikes, dtype=jnp.float64)
    maturities = jnp.asarray(maturities, dtype=jnp.float64)
    prices = jax.vmap(lambda k, t: heston_call_price(spot, k, t, rate, dividend, params))(strikes, maturities)
    if noise_std > 0:
        if key is None:
            raise ValueError("Random key must be provided when noise_std > 0")
        prices = prices + noise_std * jax.random.normal(key, shape=prices.shape, dtype=prices.dtype)
    data: Dict[str, Array] = {
        "spot": jnp.asarray(spot, dtype=jnp.float64),
        "strikes": strikes,
        "maturities": maturities,
        "rates": jnp.asarray(rate, dtype=jnp.float64),
        "dividends": jnp.asarray(dividend, dtype=jnp.float64),
        "target_prices": prices,
    }
    if weights is not None:
        data = {**data, "weights": jnp.asarray(weights, dtype=jnp.float64)}
    return data


class HestonCalibrationController(CalibrationController):
    """Calibration routine for Heston parameters."""

    def __init__(
        self,
        parameter_specs: Optional[Mapping[str, ParameterSpec]] = None,
        loss_fn=losses.mean_squared_error,
        optimizer: Optional[optax.GradientTransformation] = None,
        max_steps: int = 200,
        tol: float = 1e-7,
        penalty_weight: float = 10.0,
    ) -> None:
        if parameter_specs is None:
            parameter_specs = default_parameter_specs()
        if optimizer is None:
            optimizer = optax.adam(5e-3)

        def penalty_fn(params: Mapping[str, Array], _: Mapping[str, Array]) -> Array:
            feller_violation = jnp.maximum(0.0, params["sigma"] ** 2 - 2.0 * params["kappa"] * params["theta"])
            return penalty_weight * (feller_violation ** 2)

        super().__init__(
            parameter_specs,
            loss_fn,
            optimizer,
            penalty_fn=penalty_fn,
            max_steps=max_steps,
            tol=tol,
        )

    def _prepare_market_data(self, market_data: Mapping[str, Array]) -> Mapping[str, Array]:
        data = super()._prepare_market_data(market_data)
        prepared = {
            "spot": jnp.asarray(data["spot"], dtype=self.dtype),
            "strikes": jnp.asarray(data["strikes"], dtype=self.dtype),
            "maturities": jnp.asarray(data["maturities"], dtype=self.dtype),
            "rates": jnp.asarray(data["rates"], dtype=self.dtype),
            "dividends": jnp.asarray(data["dividends"], dtype=self.dtype),
            "target_prices": jnp.asarray(data["target_prices"], dtype=self.dtype),
        }
        if "weights" in data:
            prepared["weights"] = jnp.asarray(data["weights"], dtype=self.dtype)
        return prepared

    def _target_observables(self, market_data: Mapping[str, Array]) -> Array:
        return market_data["target_prices"]

    def _model_observables(
        self, params: Mapping[str, Array], market_data: Mapping[str, Array]
    ) -> Array:
        heston_params = HestonParams(
            v0=params["v0"],
            kappa=params["kappa"],
            theta=params["theta"],
            sigma=params["sigma"],
            rho=params["rho"],
        )
        spot = market_data["spot"]
        strikes = market_data["strikes"]
        maturities = market_data["maturities"]
        rate = market_data["rates"]
        dividend = market_data["dividends"]
        return jax.vmap(lambda k, t: heston_call_price(spot, k, t, rate, dividend, heston_params))(strikes, maturities)
