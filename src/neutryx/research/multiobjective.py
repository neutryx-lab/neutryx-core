"""Visualization and analysis utilities for multi-objective calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from neutryx.calibration.base import ParetoFront, ParetoSolution


@dataclass
class RankedSolution:
    """Pareto solution with an aggregated preference score."""

    solution: ParetoSolution
    score: float


def pareto_front_to_dataframe(pareto_front: ParetoFront) -> pd.DataFrame:
    """Convert a Pareto front into a tabular representation."""

    rows = []
    for idx, solution in enumerate(pareto_front, start=1):
        row: dict[str, float | str | int | bool] = {"solution_id": idx}
        for name, value in solution.params.items():
            row[f"param_{name}"] = value
        for name, value in solution.objectives.items():
            row[f"objective_{name}"] = value
        if solution.metadata:
            for meta_key, meta_value in solution.metadata.items():
                column = f"meta_{meta_key}"
                if isinstance(meta_value, Mapping):
                    for inner_key, inner_value in meta_value.items():
                        row[f"{column}_{inner_key}"] = inner_value
                else:
                    row[column] = meta_value
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["solution_id"])

    return pd.DataFrame(rows)


def _objective_matrix(pareto_front: ParetoFront) -> Tuple[np.ndarray, Sequence[str]]:
    names = tuple(pareto_front.objective_names())
    if not names:
        return np.empty((0, 0)), names

    data = []
    for solution in pareto_front:
        data.append([solution.objectives[name] for name in names])
    return np.asarray(data, dtype=float), names


def rank_pareto_solutions(
    pareto_front: ParetoFront,
    weights: Mapping[str, float],
    normalise: bool = True,
) -> list[RankedSolution]:
    """Rank Pareto solutions according to a weighted preference vector."""

    matrix, names = _objective_matrix(pareto_front)
    if matrix.size == 0:
        return []

    weight_vector = np.array([weights.get(name, 0.0) for name in names], dtype=float)
    if not np.any(weight_vector):
        raise ValueError("At least one objective weight must be non-zero")

    values = matrix.copy()
    if normalise:
        min_vals = values.min(axis=0)
        max_vals = values.max(axis=0)
        span = np.where(max_vals - min_vals > 1e-12, max_vals - min_vals, 1.0)
        values = (values - min_vals) / span

    scores = values @ weight_vector

    ranked = [
        RankedSolution(solution=solution, score=float(score))
        for solution, score in zip(pareto_front, scores)
    ]

    ranked.sort(key=lambda item: item.score)
    return ranked


def select_preferred_solution(
    pareto_front: ParetoFront,
    weights: Mapping[str, float],
    normalise: bool = True,
) -> Optional[ParetoSolution]:
    """Select the most preferred solution according to the weights."""

    ranked = rank_pareto_solutions(pareto_front, weights, normalise=normalise)
    if not ranked:
        return None
    return ranked[0].solution


def plot_pareto_front(
    pareto_front: ParetoFront,
    x_objective: str,
    y_objective: str,
    ax=None,
    annotate: bool = False,
    **scatter_kwargs,
):
    """Visualise a 2D projection of the Pareto front."""

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for Pareto visualisation") from exc

    if ax is None:
        _, ax = plt.subplots()

    xs: list[float] = []
    ys: list[float] = []
    labels: list[str] = []
    for idx, solution in enumerate(pareto_front, start=1):
        xs.append(solution.objectives.get(x_objective, np.nan))
        ys.append(solution.objectives.get(y_objective, np.nan))
        labels.append(solution.metadata.get("label", f"S{idx}") if solution.metadata else f"S{idx}")

    scatter_defaults = {"s": 60, "c": "tab:blue"}
    scatter_defaults.update(scatter_kwargs)
    ax.scatter(xs, ys, **scatter_defaults)
    ax.set_xlabel(x_objective)
    ax.set_ylabel(y_objective)
    ax.set_title("Pareto Front")

    if annotate:
        for label, x_val, y_val in zip(labels, xs, ys):
            ax.annotate(label, (x_val, y_val))

    return ax


def integrate_model_selection(
    pareto_front: ParetoFront,
    model_scores: Mapping[str, float],
    score_name: str = "information_criterion",
) -> pd.DataFrame:
    """Combine Pareto solutions with model selection scores."""

    df = pareto_front_to_dataframe(pareto_front)
    if df.empty or not model_scores:
        return df

    model_column = "meta_model"
    if model_column not in df.columns:
        return df

    df[f"model_{score_name}"] = df[model_column].map(model_scores)
    return df


__all__ = [
    "RankedSolution",
    "pareto_front_to_dataframe",
    "rank_pareto_solutions",
    "select_preferred_solution",
    "plot_pareto_front",
    "integrate_model_selection",
]
