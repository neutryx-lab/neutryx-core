"""Simplified stochastic-local-volatility (SLV) calibration controller."""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import jax
import jax.numpy as jnp
import optax

from . import losses
from .base import CalibrationController, ParameterSpec
from .constraints import bounded, identity, positive

Array = jnp.ndarray


def default_parameter_specs() -> Mapping[str, ParameterSpec]:
    return {
        "base_vol": ParameterSpec(0.2, positive()),
        "local_slope": ParameterSpec(0.0, identity()),
        "local_curvature": ParameterSpec(0.1, identity()),
        "mixing": ParameterSpec(0.3, bounded(0.0, 0.999)),
        "time_decay": ParameterSpec(0.05, positive()),
    }


def slv_implied_volatility(
    forward: Array,
    strike: Array,
    maturity: Array,
    params: Mapping[str, Array],
) -> Array:
    log_moneyness = jnp.log(strike / forward)
    local_component = jnp.exp(
        params["local_slope"] * log_moneyness + params["local_curvature"] * log_moneyness**2
    )
    stochastic_component = jnp.sqrt(1.0 + params["mixing"] * maturity)
    term_structure = jnp.exp(params["time_decay"] * maturity)
    return params["base_vol"] * local_component * stochastic_component * term_structure


def generate_slv_market_data(
    forward: float,
    strikes: Array,
    maturities: Array,
    params: Mapping[str, float],
    *,
    noise_std: float = 0.0,
    key: Optional[jax.Array] = None,
    weights: Optional[Array] = None,
) -> Mapping[str, Array]:
    strikes = jnp.asarray(strikes, dtype=jnp.float64)
    maturities = jnp.asarray(maturities, dtype=jnp.float64)
    param_arrays = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in params.items()}
    vols = slv_implied_volatility(forward, strikes, maturities, param_arrays)
    if noise_std > 0:
        if key is None:
            raise ValueError("Random key must be provided when noise_std > 0")
        vols = vols + noise_std * jax.random.normal(key, shape=vols.shape, dtype=vols.dtype)
    data: Dict[str, Array] = {
        "forward": jnp.asarray(forward, dtype=jnp.float64),
        "strikes": strikes,
        "maturities": maturities,
        "target_vols": vols,
    }
    if weights is not None:
        data = {**data, "weights": jnp.asarray(weights, dtype=jnp.float64)}
    return data


class SLVCalibrationController(CalibrationController):
    """Calibration controller for a stylised SLV surface."""

    def __init__(
        self,
        parameter_specs: Optional[Mapping[str, ParameterSpec]] = None,
        loss_fn=losses.mean_squared_error,
        optimizer: Optional[optax.GradientTransformation] = None,
        max_steps: int = 250,
        tol: float = 1e-8,
    ) -> None:
        if parameter_specs is None:
            parameter_specs = default_parameter_specs()
        if optimizer is None:
            optimizer = optax.adam(2e-2)
        super().__init__(parameter_specs, loss_fn, optimizer, max_steps=max_steps, tol=tol)

    def _prepare_market_data(self, market_data: Mapping[str, Array]) -> Mapping[str, Array]:
        data = super()._prepare_market_data(market_data)
        prepared = {
            "forward": jnp.asarray(data["forward"], dtype=self.dtype),
            "strikes": jnp.asarray(data["strikes"], dtype=self.dtype),
            "maturities": jnp.asarray(data["maturities"], dtype=self.dtype),
            "target_vols": jnp.asarray(data["target_vols"], dtype=self.dtype),
        }
        if "weights" in data:
            prepared["weights"] = jnp.asarray(data["weights"], dtype=self.dtype)
        return prepared

    def _target_observables(self, market_data: Mapping[str, Array]) -> Array:
        return market_data["target_vols"]

    def _model_observables(
        self, params: Mapping[str, Array], market_data: Mapping[str, Array]
    ) -> Array:
        forward = market_data["forward"]
        strikes = market_data["strikes"]
        maturities = market_data["maturities"]
        return slv_implied_volatility(forward, strikes, maturities, params)
