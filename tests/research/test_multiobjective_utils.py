"""Tests for research utilities built around Pareto analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neutryx.calibration.base import ParetoFront, ParetoSolution
from neutryx.research.multiobjective import (
    integrate_model_selection,
    pareto_front_to_dataframe,
    rank_pareto_solutions,
    select_preferred_solution,
)


def build_sample_front() -> ParetoFront:
    front = ParetoFront()
    front.add(
        ParetoSolution(
            params={"theta": 0.5},
            objectives={"mse": 0.4, "magnitude": 0.25},
            metadata={"model": "model_a", "label": "A"},
        )
    )
    front.add(
        ParetoSolution(
            params={"theta": 1.0},
            objectives={"mse": 0.2, "magnitude": 1.0},
            metadata={"model": "model_b", "label": "B"},
        )
    )
    front.add(
        ParetoSolution(
            params={"theta": 1.5},
            objectives={"mse": 0.1, "magnitude": 2.25},
            metadata={"model": "model_c", "label": "C"},
        )
    )
    return front


def test_dataframe_conversion_flattens_metadata():
    front = build_sample_front()
    df = pareto_front_to_dataframe(front)

    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {"solution_id", "param_theta", "objective_mse", "meta_model"}
    assert len(df) == 3


def test_rank_and_selection_apply_preferences():
    front = build_sample_front()
    weights = {"mse": 0.7, "magnitude": 0.3}

    ranked = rank_pareto_solutions(front, weights)
    assert ranked[0].score <= ranked[-1].score

    selected = select_preferred_solution(front, weights)
    assert selected is ranked[0].solution


def test_rank_requires_non_zero_weight():
    front = build_sample_front()

    with pytest.raises(ValueError):
        rank_pareto_solutions(front, {"mse": 0.0, "magnitude": 0.0})


def test_integration_with_model_selection_scores():
    front = build_sample_front()
    df = integrate_model_selection(front, {"model_a": 10.0, "model_c": 5.0})

    assert "model_information_criterion" in df.columns
    assert np.isnan(df.loc[df["meta_model"] == "model_b", "model_information_criterion"]).all()
