"""SABR model calibration controller."""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import jax
import jax.numpy as jnp
import optax

from neutryx.models.sabr import SABRParams, hagan_implied_vol

from . import losses
from .base import CalibrationController, ParameterSpec
from .constraints import bounded, positive_with_upper, symmetric

Array = jnp.ndarray


def default_parameter_specs() -> Mapping[str, ParameterSpec]:
    return {
        "alpha": ParameterSpec(0.2, positive_with_upper(1e-4, 3.0)),
        "beta": ParameterSpec(0.5, bounded(0.0, 0.999)),
        "rho": ParameterSpec(-0.1, symmetric(0.999)),
        "nu": ParameterSpec(0.5, positive_with_upper(1e-4, 3.0)),
    }


def generate_sabr_market_data(
    forward: float,
    strikes: Array,
    maturities: Array,
    params: SABRParams,
    *,
    noise_std: float = 0.0,
    key: Optional[jax.Array] = None,
    weights: Optional[Array] = None,
) -> Mapping[str, Array]:
    strikes = jnp.asarray(strikes, dtype=jnp.float64)
    maturities = jnp.asarray(maturities, dtype=jnp.float64)
    vols = jax.vmap(lambda k, t: hagan_implied_vol(forward, k, t, params))(strikes, maturities)
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


class SABRCalibrationController(CalibrationController):
    """Calibration routine for SABR parameters."""

    def __init__(
        self,
        parameter_specs: Optional[Mapping[str, ParameterSpec]] = None,
        loss_fn=losses.mean_squared_error,
        optimizer: Optional[optax.GradientTransformation] = None,
        max_steps: int = 300,
        tol: float = 1e-8,
    ) -> None:
        if parameter_specs is None:
            parameter_specs = default_parameter_specs()
        if optimizer is None:
            optimizer = optax.chain(optax.clip(1.0), optax.adam(1e-2))
        super().__init__(parameter_specs, loss_fn, optimizer, max_steps=max_steps, tol=tol)

    def _prepare_market_data(self, market_data: Mapping[str, Array]) -> Mapping[str, Array]:
        data = super()._prepare_market_data(market_data)
        return {
            "forward": jnp.asarray(data["forward"], dtype=self.dtype),
            "strikes": jnp.asarray(data["strikes"], dtype=self.dtype),
            "maturities": jnp.asarray(data["maturities"], dtype=self.dtype),
            "target_vols": jnp.asarray(data["target_vols"], dtype=self.dtype),
            **({"weights": jnp.asarray(data["weights"], dtype=self.dtype)} if "weights" in data else {}),
        }

    def _target_observables(self, market_data: Mapping[str, Array]) -> Array:
        return market_data["target_vols"]

    def _model_observables(
        self, params: Mapping[str, Array], market_data: Mapping[str, Array]
    ) -> Array:
        sabr_params = SABRParams(
            alpha=params["alpha"],
            beta=params["beta"],
            rho=params["rho"],
            nu=params["nu"],
        )
        forward = market_data["forward"]
        strikes = market_data["strikes"]
        maturities = market_data["maturities"]
        vols = jax.vmap(lambda k, t: hagan_implied_vol(forward, k, t, sabr_params))(strikes, maturities)
        vols = jnp.where(jnp.isfinite(vols), vols, 1.0)
        return jnp.clip(vols, 1e-4, 5.0)
