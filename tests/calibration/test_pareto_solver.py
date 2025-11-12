"""Tests for Pareto-based multi-objective calibration utilities."""

from __future__ import annotations

from typing import Mapping

import jax.numpy as jnp
import optax
import pytest

from neutryx.calibration.base import (
    CalibrationController,
    ParameterSpec,
    ParameterTransform,
)


class QuadraticCalibration(CalibrationController):
    """Simple calibration problem with quadratic objectives."""

    def _target_observables(self, market_data: Mapping[str, jnp.ndarray]):
        return jnp.asarray([market_data["target"]], dtype=self.dtype)

    def _model_observables(
        self, params: Mapping[str, jnp.ndarray], market_data: Mapping[str, jnp.ndarray]
    ):
        return jnp.asarray([params["theta"]], dtype=self.dtype)


@pytest.fixture
def controller() -> QuadraticCalibration:
    transform = ParameterTransform(forward=lambda x: x, inverse=lambda x: x)
    spec = ParameterSpec(init=0.0, transform=transform)

    def loss_fn(predicted, target, weights=None, params=None, market_data=None):
        residual = predicted - target
        return jnp.mean(residual**2)

    def objective_vector(params, market_data):
        theta = params["theta"]
        target = market_data["target"]
        residual = theta - target
        return {
            "mse": residual**2,
            "magnitude": theta**2,
        }

    optimizer = optax.adam(learning_rate=0.1)

    return QuadraticCalibration(
        parameter_specs={"theta": spec},
        loss_fn=loss_fn,
        optimizer=optimizer,
        max_steps=200,
        tol=1e-10,
        objective_vector_fn=objective_vector,
        objective_penalty_scale=2e3,
    )


def test_pareto_solver_generates_front(controller: QuadraticCalibration):
    market_data = {"target": jnp.asarray(2.0)}

    front = controller.solve_pareto_front(
        market_data,
        epsilon_grid={"magnitude": [0.01, 1.0, 4.0]},
        primary_objective="mse",
    )

    assert len(front.solutions) == 3

    magnitudes = [solution.objectives["magnitude"] for solution in front.solutions]
    epsilons = [solution.metadata["epsilon"]["magnitude"] for solution in front.solutions]

    for magnitude, epsilon in zip(magnitudes, epsilons):
        assert magnitude <= epsilon + 1e-2

    nondominated = front.nondominated()
    assert nondominated
    for solution in nondominated:
        assert solution in front.solutions


def test_multiobjective_requires_vector(controller: QuadraticCalibration):
    market_data = {"target": jnp.asarray(1.0)}

    with pytest.raises(ValueError):
        base = QuadraticCalibration(
            parameter_specs=controller.parameter_specs,
            loss_fn=controller.loss_fn,
            optimizer=controller.optimizer,
        )
        base.solve_pareto_front(market_data)
