"""Common infrastructure for model calibration controllers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


@dataclass
class ParameterTransform:
    """Bidirectional transform used to enforce parameter constraints."""

    forward: Callable[[Array], Array]
    inverse: Callable[[Array], Array]

    def apply(self, value: Array) -> Array:
        return self.forward(value)

    def invert(self, value: Array) -> Array:
        return self.inverse(value)


@dataclass
class ParameterSpec:
    """Specification for a calibrated parameter."""

    init: float
    transform: ParameterTransform


@dataclass
class CalibrationResult:
    """Container for calibration outputs."""

    params: Dict[str, float]
    loss_history: Iterable[float]
    converged: bool
    iterations: int


class CalibrationController:
    """Base class implementing gradient-based calibration workflow."""

    def __init__(
        self,
        parameter_specs: Mapping[str, ParameterSpec],
        loss_fn: Callable[..., Array],
        optimizer: optax.GradientTransformation,
        penalty_fn: Optional[Callable[[Mapping[str, Array], Mapping[str, Array]], Array]] = None,
        max_steps: int = 400,
        tol: float = 1e-8,
        dtype: jnp.dtype = jnp.float64,
    ) -> None:
        self.parameter_specs = dict(parameter_specs)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.penalty_fn = penalty_fn
        self.max_steps = int(max_steps)
        self.tol = float(tol)
        self.dtype = dtype

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------
    def _prepare_market_data(self, market_data: Mapping[str, Any]) -> Mapping[str, Array]:
        """Convert raw calibration inputs into JAX arrays."""

        converted: Dict[str, Any] = {}
        for key, value in market_data.items():
            if isinstance(value, (int, float)):
                converted[key] = jnp.asarray(value, dtype=self.dtype)
            elif isinstance(value, (list, tuple)):
                converted[key] = jnp.asarray(value, dtype=self.dtype)
            elif isinstance(value, jnp.ndarray):
                converted[key] = jnp.asarray(value, dtype=self.dtype)
            elif isinstance(value, np.ndarray):
                converted[key] = jnp.asarray(value, dtype=self.dtype)
            else:
                converted[key] = value
        return converted

    def _target_observables(self, market_data: Mapping[str, Array]) -> Array:
        raise NotImplementedError

    def _model_observables(
        self, params: Mapping[str, Array], market_data: Mapping[str, Array]
    ) -> Array:
        raise NotImplementedError

    # ------------------------------------------------------------------
    def _initial_theta(self) -> Dict[str, Array]:
        theta: Dict[str, Array] = {}
        for name, spec in self.parameter_specs.items():
            init_value = jnp.asarray(spec.init, dtype=self.dtype)
            theta[name] = spec.transform.invert(init_value)
        return theta

    def _constrain_params(self, theta: Mapping[str, Array]) -> Dict[str, Array]:
        return {name: spec.transform.apply(theta[name]) for name, spec in self.parameter_specs.items()}

    def calibrate(self, market_data: Mapping[str, Any]) -> CalibrationResult:
        """Run gradient-based calibration with autodiff support."""

        prepared_data = self._prepare_market_data(market_data)
        theta = self._initial_theta()
        opt_state = self.optimizer.init(theta)

        def objective(current_theta: Mapping[str, Array]) -> Array:
            constrained = self._constrain_params(current_theta)
            predicted = self._model_observables(constrained, prepared_data)
            target = self._target_observables(prepared_data)
            weights = prepared_data.get("weights")
            loss = self.loss_fn(predicted, target, weights=weights, params=constrained, market_data=prepared_data)
            if self.penalty_fn is not None:
                loss = loss + self.penalty_fn(constrained, prepared_data)
            return loss

        loss_and_grad = jax.jit(jax.value_and_grad(objective))

        loss_history: list[float] = []
        converged = False
        prev_loss: Optional[float] = None

        for step in range(1, self.max_steps + 1):
            value, grad = loss_and_grad(theta)
            updates, opt_state = self.optimizer.update(grad, opt_state, theta)
            theta_candidate = optax.apply_updates(theta, updates)
            theta = jax.tree_util.tree_map(
                lambda new, old: jnp.where(jnp.isfinite(new), jnp.clip(new, -1e6, 1e6), old),
                theta_candidate,
                theta,
            )

            loss_value = float(jax.device_get(value))
            loss_history.append(loss_value)

            if prev_loss is not None and step > 5 and abs(prev_loss - loss_value) < self.tol:
                converged = True
                iterations = step
                break

            prev_loss = loss_value
        else:
            iterations = self.max_steps

        constrained_params = {
            name: float(jax.device_get(jnp.nan_to_num(value, nan=0.0)))
            for name, value in self._constrain_params(theta).items()
        }

        return CalibrationResult(
            params=constrained_params,
            loss_history=loss_history,
            converged=converged,
            iterations=iterations,
        )
