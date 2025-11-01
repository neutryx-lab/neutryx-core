"""Exposure simulation utilities for the XVA engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import jax
import jax.numpy as jnp

from neutryx.core.engine import MCPaths

Array = jnp.ndarray


@dataclass(frozen=True)
class XVAScenario:
    """Container describing a risk scenario used in XVA simulations."""

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def with_overrides(self, **kwargs: Any) -> "XVAScenario":
        """Return a new scenario with parameter overrides."""

        updated: dict[str, Any] = dict(self.params)
        updated.update(kwargs)
        return XVAScenario(name=self.name, params=updated, weight=self.weight, metadata=self.metadata)


@dataclass
class ExposureResult:
    """Exposure summary for a single scenario."""

    scenario: XVAScenario
    times: Array
    expected_positive: Array
    expected_negative: Array
    net_exposure: Array
    pathwise: Array
    params: Mapping[str, Any]

    def validate(self) -> None:
        """Validate internal array dimensions."""

        length = self.times.shape[0]
        for arr in (self.expected_positive, self.expected_negative, self.net_exposure):
            if arr.shape[0] != length:
                raise ValueError("ExposureResult arrays must align with the provided time grid.")
        if self.pathwise.shape[1] != length:
            raise ValueError("Pathwise exposures must align with the provided time grid.")


@dataclass
class ExposureCube:
    """Collection of exposure results across multiple scenarios."""

    profiles: Sequence[ExposureResult]

    def __post_init__(self) -> None:
        if not self.profiles:
            raise ValueError("ExposureCube requires at least one ExposureResult.")
        lengths = {tuple(profile.times.tolist()) for profile in self.profiles}
        if len(lengths) != 1:
            raise ValueError("All exposure profiles must share the same time grid.")
        for profile in self.profiles:
            profile.validate()

    @property
    def times(self) -> Array:
        return self.profiles[0].times

    def scenario_names(self) -> list[str]:
        return [profile.scenario.name for profile in self.profiles]

    def _weights(self) -> Array:
        weights = jnp.array([max(profile.scenario.weight, 0.0) for profile in self.profiles])
        total = weights.sum()
        if total == 0:
            raise ValueError("At least one scenario must have a positive weight.")
        return weights / total

    def expected_positive_matrix(self) -> Array:
        return jnp.stack([profile.expected_positive for profile in self.profiles], axis=0)

    def expected_negative_matrix(self) -> Array:
        return jnp.stack([profile.expected_negative for profile in self.profiles], axis=0)

    def net_exposure_matrix(self) -> Array:
        return jnp.stack([profile.net_exposure for profile in self.profiles], axis=0)

    def aggregate_expected_positive(self) -> Array:
        weights = self._weights()[:, None]
        return (weights * self.expected_positive_matrix()).sum(axis=0)

    def aggregate_expected_negative(self) -> Array:
        weights = self._weights()[:, None]
        return (weights * self.expected_negative_matrix()).sum(axis=0)

    def aggregate_net_exposure(self) -> Array:
        weights = self._weights()[:, None]
        return (weights * self.net_exposure_matrix()).sum(axis=0)

    def aggregate_pathwise(self) -> Array:
        weights = self._weights()[:, None, None]
        pathwise = jnp.stack([profile.pathwise for profile in self.profiles], axis=0)
        return (weights * pathwise).sum(axis=0)


ExposureGenerator = Callable[[jax.Array, Mapping[str, Any]], Array | MCPaths]
ExposureFn = Callable[[Array, Mapping[str, Any]], Array]


class ExposureSimulator:
    """Run exposure simulations across multiple scenarios."""

    def __init__(
        self,
        path_generator: ExposureGenerator,
        exposure_fn: ExposureFn,
        *,
        time_grid: Array | None = None,
    ) -> None:
        self._path_generator = path_generator
        self._exposure_fn = exposure_fn
        self._time_grid = time_grid

    def simulate(
        self,
        key: jax.Array,
        base_params: MutableMapping[str, Any],
        scenarios: Iterable[XVAScenario],
    ) -> ExposureCube:
        profiles: list[ExposureResult] = []
        for idx, scenario in enumerate(scenarios):
            params: dict[str, Any] = dict(base_params)
            params.update(scenario.params)
            scenario_key = jax.random.fold_in(key, idx)
            raw_paths = self._path_generator(scenario_key, params)
            paths, times = self._extract_paths_and_times(raw_paths, params)
            exposures = self._compute_pathwise_exposure(paths, params)
            expected_positive = jnp.maximum(exposures, 0.0).mean(axis=0)
            expected_negative = jnp.maximum(-exposures, 0.0).mean(axis=0)
            net_exposure = exposures.mean(axis=0)
            profile = ExposureResult(
                scenario=scenario,
                times=times,
                expected_positive=expected_positive,
                expected_negative=expected_negative,
                net_exposure=net_exposure,
                pathwise=exposures,
                params=params,
            )
            profile.validate()
            profiles.append(profile)
        return ExposureCube(profiles)

    def _extract_paths_and_times(
        self, raw_paths: Array | MCPaths, params: Mapping[str, Any]
    ) -> tuple[Array, Array]:
        if isinstance(raw_paths, MCPaths):
            return raw_paths.values, raw_paths.times
        paths = jnp.asarray(raw_paths)
        if paths.ndim != 2:
            raise ValueError("Path generator must return a 2D array of shape [paths, time].")
        if self._time_grid is not None:
            times = self._time_grid
        else:
            horizon = float(params.get("T", 1.0))
            steps = paths.shape[1] - 1
            times = jnp.linspace(0.0, horizon, steps + 1)
        return paths, times

    def _compute_pathwise_exposure(self, paths: Array, params: Mapping[str, Any]) -> Array:
        exposures = jnp.asarray(self._exposure_fn(paths, params))
        if exposures.shape != paths.shape:
            raise ValueError("Exposure function must return an array with the same shape as the paths.")
        return exposures
