"""Experiment tracking utilities.

This module provides a minimal abstraction over experiment trackers so that
calibration/training scripts can emit metrics to either Weights & Biases or
MLflow without taking a hard dependency on either package.  Importing happens
lazily which keeps the base installation lightweight while still allowing
projects that have either client installed to benefit from structured logging.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional


class TrackingError(RuntimeError):
    """Raised when a configured tracker cannot be initialised."""


@dataclass
class TrackingConfig:
    """Configuration toggles for experiment tracking.

    Attributes
    ----------
    enable:
        Global flag allowing tracking to be switched on/off without changing
        the rest of the configuration.
    provider:
        Tracker backend to use.  Supported values are ``"wandb"``,
        ``"mlflow"`` and ``"none"``.
    project:
        Project/experiment identifier used by the backend.  For MLflow this is
        mapped to the experiment name.
    experiment_name:
        Display name for the run.  If omitted the calibrator may generate a
        reasonable default (e.g. ``"heston-calibration"``).
    tags:
        Optional iterable of tags to attach to the run.
    log_every:
        Default logging cadence used by helpers; individual call-sites can
        override this value.
    run_kwargs:
        Additional keyword arguments forwarded to the backend specific
        ``init``/``start_run`` calls.
    """

    enable: bool = False
    provider: str = "none"
    project: Optional[str] = None
    experiment_name: Optional[str] = None
    tags: Optional[Iterable[str]] = None
    log_every: int = 10
    run_kwargs: Dict[str, Any] = field(default_factory=dict)

    def is_enabled(self) -> bool:
        return self.enable and self.provider.lower() not in {"", "none"}


class BaseTracker:
    """Lightweight tracker interface used by calibration routines."""

    def __init__(self, config: TrackingConfig):
        self.config = config

    # pylint: disable=unused-argument
    def log_params(self, params: Mapping[str, Any]) -> None:
        """Record configuration parameters for the run."""

    def log_metrics(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        """Record numeric metrics."""

    def finish(self) -> None:
        """Close/flush the tracker."""


class NoOpTracker(BaseTracker):
    """Tracker implementation that silently ignores all calls."""

    def __init__(self, config: Optional[TrackingConfig] = None):
        super().__init__(config or TrackingConfig())


class WandbTracker(BaseTracker):
    """Tracker that proxies calls to :mod:`wandb`."""

    def __init__(self, config: TrackingConfig):
        super().__init__(config)
        try:
            import wandb  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise TrackingError("wandb is not installed") from exc

        init_kwargs = dict(config.run_kwargs)
        if config.project:
            init_kwargs.setdefault("project", config.project)
        if config.experiment_name:
            init_kwargs.setdefault("name", config.experiment_name)
        if config.tags:
            init_kwargs.setdefault("tags", list(config.tags))

        self._wandb = wandb
        self._run = wandb.init(**init_kwargs)

    def log_params(self, params: Mapping[str, Any]) -> None:  # pragma: no cover - optional dependency
        if self._run is None:
            return
        self._wandb.config.update(dict(params), allow_val_change=True)

    def log_metrics(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:  # pragma: no cover
        if self._run is None:
            return
        self._wandb.log(dict(metrics), step=step)

    def finish(self) -> None:  # pragma: no cover - optional dependency
        if self._run is not None:
            self._wandb.finish()
            self._run = None


class MLflowTracker(BaseTracker):
    """Tracker that proxies calls to :mod:`mlflow`."""

    def __init__(self, config: TrackingConfig):
        super().__init__(config)
        try:
            import mlflow  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise TrackingError("mlflow is not installed") from exc

        self._mlflow = mlflow
        run_kwargs = dict(config.run_kwargs)
        tags = list(config.tags) if config.tags else None

        if config.project:
            mlflow.set_experiment(config.project)

        self._active_run = mlflow.start_run(run_name=config.experiment_name, tags=tags, **run_kwargs)

    def log_params(self, params: Mapping[str, Any]) -> None:  # pragma: no cover - optional dependency
        self._mlflow.log_params(_to_serialisable(params))

    def log_metrics(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:  # pragma: no cover
        metrics_dict = _to_serialisable(metrics)
        if step is None:
            self._mlflow.log_metrics(metrics_dict)
        else:
            for key, value in metrics_dict.items():
                self._mlflow.log_metric(key, value, step=step)

    def finish(self) -> None:  # pragma: no cover - optional dependency
        if self._active_run is not None:
            self._mlflow.end_run()
            self._active_run = None


def create_tracker(config: Optional[TrackingConfig | Mapping[str, Any]]) -> BaseTracker:
    """Instantiate an appropriate tracker from configuration."""

    if config is None:
        return NoOpTracker()

    if not isinstance(config, TrackingConfig):
        config = TrackingConfig(**dict(config))

    if not config.is_enabled():
        return NoOpTracker(config)

    provider = config.provider.lower()
    if provider == "wandb":
        return WandbTracker(config)
    if provider == "mlflow":
        return MLflowTracker(config)
    raise TrackingError(f"Unsupported tracking provider: {config.provider}")


@contextmanager
def tracker_context(config: Optional[TrackingConfig | Mapping[str, Any]]) -> BaseTracker:
    """Context manager that yields a tracker and ensures cleanup."""

    tracker = create_tracker(config)
    try:
        yield tracker
    finally:
        tracker.finish()


def calibration_metric_template(
    loss: Any,
    params: Mapping[str, Any],
    prefix: str = "calibration",
) -> Dict[str, Any]:
    """Standardised dictionary structure for calibration metrics."""

    metrics: Dict[str, Any] = {f"{prefix}/loss": _to_float(loss)}
    for key, value in params.items():
        metrics[f"{prefix}/params/{key}"] = _to_float(value)
    return metrics


def calibration_param_template(
    params: Mapping[str, Any],
    prefix: str = "calibration",
) -> Dict[str, Any]:
    """Standardised dictionary for logging calibration hyper-parameters."""

    return {f"{prefix}.{key}": _to_float(value) for key, value in params.items()}


def _to_float(value: Any) -> float:
    """Best-effort conversion of numeric tensors to Python floats."""

    if isinstance(value, (int, float)):
        return float(value)
    try:
        import jax.numpy as jnp  # type: ignore[import-not-found]

        if isinstance(value, jnp.ndarray):
            return float(value.item())
    except Exception:  # pragma: no cover - fallback for missing jax or errors
        pass
    try:
        return float(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError(f"Cannot convert value {value!r} to float") from exc


def _to_serialisable(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert mapping values to serialisable Python types."""

    return {key: _to_float(value) for key, value in mapping.items()}


@contextmanager
def resolve_tracker(
    tracker: Optional[BaseTracker],
    config: Optional[TrackingConfig | Mapping[str, Any]],
) -> BaseTracker:
    """Context manager that reuses a provided tracker or creates a new one."""

    if tracker is not None:
        yield tracker
        return

    with tracker_context(config) as ctx:
        yield ctx

