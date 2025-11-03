"""Diagnostics utilities for model calibration workflows."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


@dataclass
class ResidualPlotData:
    """Container for calibration residuals that can be plotted or exported.

    Attributes:
        coordinates: Mapping of coordinate names (e.g. strike, maturity) to arrays
            describing the option grid the calibration was performed on.
        residuals: Residual array (model - market).
    """

    coordinates: Mapping[str, jnp.ndarray]
    residuals: jnp.ndarray

    def to_dataframe(self):
        """Return a tidy :class:`pandas.DataFrame` with residual information."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - dependency is part of the project
            raise RuntimeError("pandas is required to export residual data") from exc

        data: Dict[str, jnp.ndarray] = {
            name: jnp.asarray(value).reshape(-1) for name, value in self.coordinates.items()
        }
        data["residual"] = jnp.asarray(self.residuals).reshape(-1)
        return pd.DataFrame(data)

    def plot(self, ax=None, *, cmap: str = "coolwarm"):
        """Plot residuals using Matplotlib.

        The first two coordinates are interpreted as x/y axes. If only a single
        coordinate is provided, the residuals are plotted against that coordinate.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("matplotlib is required for plotting residuals") from exc

        df = self.to_dataframe()
        coord_names = list(self.coordinates.keys())
        if not coord_names:
            raise ValueError("At least one coordinate is required for plotting")

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        if len(coord_names) >= 2:
            x_key, y_key = coord_names[:2]
            scatter = ax.scatter(df[x_key], df[y_key], c=df["residual"], cmap=cmap)
            ax.set_xlabel(x_key)
            ax.set_ylabel(y_key)
        else:
            x_key = coord_names[0]
            scatter = ax.scatter(df[x_key], df["residual"], c=df["residual"], cmap=cmap)
            ax.set_xlabel(x_key)
            ax.set_ylabel("residual")

        ax.set_title("Calibration residuals")
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Residual")
        return ax


@dataclass
class IdentifiabilityMetrics:
    """Identifiability diagnostics derived from the model Jacobian."""

    condition_number: float
    parameter_std: Mapping[str, float]
    correlation_matrix: jnp.ndarray
    fisher_information: jnp.ndarray
    parameter_names: Tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> Dict[str, Any]:
        """Return a dictionary summary of the identifiability metrics."""
        return {
            "condition_number": self.condition_number,
            "parameter_std": dict(self.parameter_std),
            "parameter_names": self.parameter_names,
        }


@dataclass
class CalibrationDiagnostics:
    """Aggregate diagnostics produced after a calibration run."""

    residuals: jnp.ndarray
    mse: float
    mae: float
    max_abs_error: float
    predicted: jnp.ndarray
    residual_plot: Optional[ResidualPlotData] = None
    identifiability: Optional[IdentifiabilityMetrics] = None

    def summary(self) -> Dict[str, float]:
        """Return key error statistics in a dictionary format."""
        return {
            "mse": self.mse,
            "mae": self.mae,
            "max_abs_error": self.max_abs_error,
        }


def build_residual_plot_data(
    coordinates: Mapping[str, jnp.ndarray] | None,
    residuals: jnp.ndarray,
) -> Optional[ResidualPlotData]:
    """Construct a :class:`ResidualPlotData` instance for plotting.

    Args:
        coordinates: Mapping of axis name to calibration coordinates. The arrays must
            be broadcast-compatible with the residuals.
        residuals: Residual array from calibration (model - market).

    Returns:
        ResidualPlotData if coordinates are provided, otherwise ``None``.
    """

    if not coordinates:
        return None

    residuals_arr = jnp.asarray(residuals)
    processed: Dict[str, jnp.ndarray] = {}
    for name, value in coordinates.items():
        arr = jnp.asarray(value)
        if arr.shape != residuals_arr.shape:
            try:
                arr = jnp.broadcast_to(arr, residuals_arr.shape)
            except ValueError as exc:
                raise ValueError(
                    f"Coordinate '{name}' with shape {arr.shape} is not compatible with "
                    f"residuals of shape {residuals_arr.shape}"
                ) from exc
        processed[name] = arr

    return ResidualPlotData(coordinates=processed, residuals=residuals_arr)


def compute_identifiability_metrics(
    jacobian: jnp.ndarray,
    residuals: jnp.ndarray,
    parameter_names: Sequence[str],
    *,
    ridge: float = 1e-8,
) -> IdentifiabilityMetrics:
    """Compute identifiability diagnostics from a Jacobian matrix.

    Args:
        jacobian: Jacobian matrix of model outputs with respect to parameters with
            shape (n_observations, n_parameters).
        residuals: Residual vector used to approximate the observation noise.
        parameter_names: Sequence of parameter names corresponding to the Jacobian
            columns.
        ridge: Diagonal regularisation added to the Fisher information matrix to
            ensure numerical stability.
    """

    jacobian = jnp.asarray(jacobian)
    if jacobian.ndim != 2:
        raise ValueError("Jacobian must be a 2-D array")

    n_obs, n_params = jacobian.shape
    if n_params != len(tuple(parameter_names)):
        raise ValueError("Number of parameter names must match Jacobian columns")

    fisher = jacobian.T @ jacobian
    fisher = fisher + ridge * jnp.eye(fisher.shape[0])
    cond_number = float(jnp.linalg.cond(fisher))

    residual_vector = jnp.asarray(residuals).reshape(-1)
    noise_var = jnp.maximum(jnp.mean(residual_vector**2), 1e-12)
    covariance = noise_var * jnp.linalg.pinv(fisher)
    std = jnp.sqrt(jnp.maximum(jnp.diag(covariance), 0.0))

    parameter_std = {name: float(value) for name, value in zip(parameter_names, std)}
    denom = jnp.outer(std, std)
    correlation = jnp.where(denom > 0, covariance / denom, 0.0)

    return IdentifiabilityMetrics(
        condition_number=cond_number,
        parameter_std=parameter_std,
        correlation_matrix=correlation,
        fisher_information=fisher,
        parameter_names=tuple(parameter_names),
    )


def _infer_parameter_names(params: Any, size: int) -> Tuple[str, ...]:
    if is_dataclass(params):
        return tuple(field.name for field in fields(params))
    if isinstance(params, Mapping):
        return tuple(str(k) for k in params.keys())
    return tuple(f"param_{i}" for i in range(size))


def _as_pytree(params: Any) -> Tuple[Any, Callable[[Any], Any]]:
    """Return a pytree representation of ``params`` and a rebuild callable."""

    if is_dataclass(params):
        field_names = [field.name for field in fields(params)]
        base_tree = {name: getattr(params, name) for name in field_names}

        def rebuild(structured: Any) -> Any:
            if is_dataclass(structured):
                return structured
            if isinstance(structured, Mapping):
                return type(params)(**structured)
            raise TypeError("Unexpected structure returned by unravel_pytree")

        return base_tree, rebuild

    if isinstance(params, Mapping):
        return params, lambda structured: structured

    return params, lambda structured: structured


def generate_calibration_diagnostics(
    model_fn: Callable[..., jnp.ndarray],
    params: Any,
    observed: jnp.ndarray,
    *,
    coordinates: Mapping[str, jnp.ndarray] | None = None,
    parameter_names: Sequence[str] | None = None,
    compute_identifiability: bool = True,
    ridge: float = 1e-8,
) -> CalibrationDiagnostics:
    """Generate diagnostics for a calibration run.

    Args:
        model_fn: Callable returning model prices for the provided parameters. The
            callable must accept the calibrated parameters as the first argument and
            keyword arguments for each entry in ``coordinates``.
        params: Calibrated parameters (dataclass, mapping or pytree).
        observed: Array with market observations used during calibration.
        coordinates: Optional mapping of coordinate names to arrays used to build
            residual plots (e.g. strike/maturity grids).
        parameter_names: Optional names for parameters. If ``None`` the names are
            inferred from ``params`` when possible.
        compute_identifiability: Whether to compute Jacobian-based identifiability
            metrics. The computation can be expensive for large models.
        ridge: Diagonal regularisation added to the Fisher information matrix.
    """

    coords = dict(coordinates or {})
    observed_arr = jnp.asarray(observed)
    predictions = jnp.asarray(model_fn(params, **coords))
    residuals = predictions - observed_arr

    residual_flat = residuals.reshape(-1)
    mse = float(jnp.mean(residual_flat**2))
    mae = float(jnp.mean(jnp.abs(residual_flat)))
    max_abs = float(jnp.max(jnp.abs(residual_flat)))

    residual_plot = build_residual_plot_data(coords, residuals)

    identifiability: Optional[IdentifiabilityMetrics] = None
    if compute_identifiability:
        pytree_params, rebuild = _as_pytree(params)
        flat_params, unravel = ravel_pytree(pytree_params)
        size = int(flat_params.shape[0])
        inferred_names = tuple(parameter_names) if parameter_names else _infer_parameter_names(params, size)

        def flat_model(flat_theta):
            structured = unravel(flat_theta)
            model_values = model_fn(rebuild(structured), **coords)
            return jnp.asarray(model_values).reshape(-1)

        jac = jax.jacfwd(flat_model)(flat_params)
        jac_matrix = jnp.asarray(jac).reshape(size, -1).T
        identifiability = compute_identifiability_metrics(
            jac_matrix, residual_flat, inferred_names, ridge=ridge
        )

    return CalibrationDiagnostics(
        residuals=residuals,
        mse=mse,
        mae=mae,
        max_abs_error=max_abs,
        predicted=predictions,
        residual_plot=residual_plot,
        identifiability=identifiability,
    )
