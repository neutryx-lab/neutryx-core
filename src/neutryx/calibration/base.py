"""Common infrastructure for model calibration controllers."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

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


@dataclass
class ParetoSolution:
    """Single solution belonging to a Pareto front."""

    params: Dict[str, float]
    objectives: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ParetoFront:
    """Collection of Pareto optimal (or candidate) solutions."""

    solutions: list[ParetoSolution] = field(default_factory=list)

    def __iter__(self):
        return iter(self.solutions)

    def add(self, solution: ParetoSolution) -> None:
        self.solutions.append(solution)

    def objective_names(self) -> Sequence[str]:
        if not self.solutions:
            return []
        return tuple(self.solutions[0].objectives.keys())

    def nondominated(self) -> list[ParetoSolution]:
        """Return non-dominated solutions (assumes minimisation)."""

        if not self.solutions:
            return []

        names = self.objective_names()
        nondominated: list[ParetoSolution] = []
        for candidate in self.solutions:
            dominated = False
            for challenger in self.solutions:
                if challenger is candidate:
                    continue
                less_or_equal = all(
                    challenger.objectives[name] <= candidate.objectives[name]
                    for name in names
                )
                strictly_better = any(
                    challenger.objectives[name] < candidate.objectives[name]
                    for name in names
                )
                if less_or_equal and strictly_better:
                    dominated = True
                    break
            if not dominated:
                nondominated.append(candidate)
        return nondominated


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
        objective_vector_fn: Optional[
            Callable[[Mapping[str, Array], Mapping[str, Array]], Mapping[str, Array] | Array]
        ] = None,
        pareto_solver: str = "epsilon_constraint",
        objective_penalty_scale: float = 1e3,
    ) -> None:
        self.parameter_specs = dict(parameter_specs)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.penalty_fn = penalty_fn
        self.max_steps = int(max_steps)
        self.tol = float(tol)
        self.dtype = dtype
        self.objective_vector_fn = objective_vector_fn
        self.pareto_solver = pareto_solver
        self.objective_penalty_scale = float(objective_penalty_scale)

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
        result = self._run_optimization(prepared_data, penalty_override=None)

        return CalibrationResult(
            params=result["constrained_params"],
            loss_history=result["loss_history"],
            converged=result["converged"],
            iterations=result["iterations"],
        )

    # ------------------------------------------------------------------
    # Multi-objective utilities
    # ------------------------------------------------------------------
    def _run_optimization(
        self,
        prepared_data: Mapping[str, Array],
        penalty_override: Optional[Callable[[Mapping[str, Array], Mapping[str, Array]], Array]],
    ) -> Dict[str, Any]:
        theta = self._initial_theta()
        opt_state = self.optimizer.init(theta)

        combined_penalty: Optional[Callable[[Mapping[str, Array], Mapping[str, Array]], Array]]
        if penalty_override is None:
            combined_penalty = self.penalty_fn
        elif self.penalty_fn is None:
            combined_penalty = penalty_override
        else:
            def combined_penalty(params: Mapping[str, Array], data: Mapping[str, Array]) -> Array:
                return self.penalty_fn(params, data) + penalty_override(params, data)

        def objective(current_theta: Mapping[str, Array]) -> Array:
            constrained = self._constrain_params(current_theta)
            predicted = self._model_observables(constrained, prepared_data)
            target = self._target_observables(prepared_data)
            weights = prepared_data.get("weights")
            loss = self.loss_fn(
                predicted,
                target,
                weights=weights,
                params=constrained,
                market_data=prepared_data,
            )
            if combined_penalty is not None:
                loss = loss + combined_penalty(constrained, prepared_data)
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

        constrained_theta = self._constrain_params(theta)
        constrained_params = {
            name: float(jax.device_get(jnp.nan_to_num(value, nan=0.0)))
            for name, value in constrained_theta.items()
        }

        return {
            "params": theta,
            "constrained_params": constrained_params,
            "loss_history": loss_history,
            "converged": converged,
            "iterations": iterations,
            "final_loss": loss_history[-1] if loss_history else np.nan,
            "constrained_theta": constrained_theta,
        }

    def _objective_vector(
        self, params: Mapping[str, Array], market_data: Mapping[str, Array]
    ) -> Dict[str, Array]:
        if self.objective_vector_fn is None:
            raise ValueError("objective_vector_fn must be provided for multi-objective calibration")

        vector = self.objective_vector_fn(params, market_data)
        if isinstance(vector, Mapping):
            converted = {
                str(name): jnp.asarray(value, dtype=self.dtype)
                for name, value in vector.items()
            }
        else:
            array = jnp.atleast_1d(jnp.asarray(vector, dtype=self.dtype))
            converted = {f"objective_{idx}": array[idx] for idx in range(array.shape[0])}
        return converted

    def _objective_values_as_float(
        self, params: Mapping[str, Array], market_data: Mapping[str, Array]
    ) -> Dict[str, float]:
        vector = self._objective_vector(params, market_data)
        return {
            name: float(jax.device_get(jnp.nan_to_num(value, nan=np.inf)))
            for name, value in vector.items()
        }

    def solve_pareto_front(
        self,
        market_data: Mapping[str, Any],
        epsilon_grid: Optional[Mapping[str, Sequence[float]]] = None,
        primary_objective: Optional[str] = None,
    ) -> ParetoFront:
        """Solve a multi-objective calibration problem using Pareto optimisation."""

        if self.objective_vector_fn is None:
            raise ValueError("Multi-objective optimisation requires objective_vector_fn")

        prepared_data = self._prepare_market_data(market_data)
        baseline = self._run_optimization(prepared_data, penalty_override=None)
        base_objectives = self._objective_values_as_float(baseline["constrained_theta"], prepared_data)

        objective_names = list(base_objectives.keys())
        if not objective_names:
            raise ValueError("objective_vector_fn must produce at least one objective")

        if primary_objective is None:
            primary_objective = objective_names[0]
        if primary_objective not in base_objectives:
            raise ValueError(f"Unknown primary objective '{primary_objective}'")

        constraint_objectives = [name for name in objective_names if name != primary_objective]

        if self.pareto_solver != "epsilon_constraint":
            raise NotImplementedError(f"Unsupported Pareto solver: {self.pareto_solver}")

        if constraint_objectives and epsilon_grid is None:
            epsilon_grid = {}
            for name in constraint_objectives:
                reference = base_objectives[name]
                width = abs(reference) if reference != 0 else 1.0
                epsilon_grid[name] = (
                    np.linspace(reference - 0.5 * width, reference + 0.5 * width, num=3).tolist()
                )
        elif epsilon_grid is None:
            epsilon_grid = {}
        else:
            missing = [name for name in constraint_objectives if name not in epsilon_grid]
            if missing:
                raise ValueError(
                    "Epsilon grid missing objectives: " + ", ".join(missing)
                )
            epsilon_grid = {name: list(values) for name, values in epsilon_grid.items()}
            for name, values in epsilon_grid.items():
                if not values:
                    raise ValueError(f"Epsilon grid for objective '{name}' must not be empty")

        pareto_front = ParetoFront()

        if not constraint_objectives:
            objective_values = self._objective_values_as_float(
                baseline["constrained_theta"], prepared_data
            )
            pareto_front.add(
                ParetoSolution(
                    params=baseline["constrained_params"],
                    objectives=objective_values,
                    metadata={"epsilon": {}},
                )
            )
            return pareto_front

        epsilon_sequences = [epsilon_grid[name] for name in constraint_objectives]
        for epsilon_values in product(*epsilon_sequences):
            epsilon_targets = dict(zip(constraint_objectives, epsilon_values))

            def epsilon_penalty(
                params: Mapping[str, Array], data: Mapping[str, Array]
            ) -> Array:
                objectives = self._objective_vector(params, data)
                penalties = []
                for name, epsilon in epsilon_targets.items():
                    diff = objectives[name] - epsilon
                    penalties.append(jnp.where(diff > 0, diff**2, 0.0))
                if not penalties:
                    return jnp.asarray(0.0, dtype=self.dtype)
                stacked = jnp.stack(penalties)
                return self.objective_penalty_scale * jnp.sum(stacked)

            result = self._run_optimization(prepared_data, penalty_override=epsilon_penalty)
            objective_values = self._objective_values_as_float(result["constrained_theta"], prepared_data)
            pareto_front.add(
                ParetoSolution(
                    params=result["constrained_params"],
                    objectives=objective_values,
                    metadata={
                        "epsilon": epsilon_targets,
                        "loss": result["final_loss"],
                        "converged": result["converged"],
                    },
                )
            )

        return pareto_front
