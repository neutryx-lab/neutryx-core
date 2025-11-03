"""
Exposure simulation utilities for the XVA engine.

Enhanced with MarketDataEnvironment support for scenario-based risk analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Sequence

import jax
import jax.numpy as jnp

from neutryx.core.engine import MCPaths

Array = jnp.ndarray

# Import market data types (optional, for type hints)
try:
    from neutryx.market.environment import MarketDataEnvironment
    from neutryx.valuations.scenarios.bumpers import MarketScenario
    _HAS_MARKET_ENV = True
except ImportError:
    MarketDataEnvironment = Any  # type: ignore
    MarketScenario = Any  # type: ignore
    _HAS_MARKET_ENV = False


@dataclass(frozen=True)
class XVAScenario:
    """
    Container describing a risk scenario used in XVA simulations.

    Supports both legacy parameter-based scenarios and new MarketScenario-based
    scenarios that apply systematic market data shocks.

    Attributes:
        name: Scenario identifier
        params: Legacy parameter dict (for backward compatibility)
        weight: Scenario weight for aggregation (default 1.0)
        metadata: Additional scenario metadata
        market_scenario: Optional MarketScenario for environment-based shocks

    Example (legacy):
        >>> scenario = XVAScenario(
        ...     name="vol_up_10pct",
        ...     params={"volatility": 0.22}  # 20% vol -> 22% vol
        ... )

    Example (new):
        >>> from neutryx.valuations.scenarios import MarketScenario, CurveBumper
        >>> market_shock = MarketScenario(
        ...     name="rates_up_50bps",
        ...     curve_shocks={
        ...         ('discount', 'USD'): lambda c: CurveBumper.parallel_shift(c, 50)
        ...     }
        ... )
        >>> scenario = XVAScenario(
        ...     name="rates_up_50bps",
        ...     market_scenario=market_shock
        ... )
    """

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    market_scenario: Optional[MarketScenario] = None

    def with_overrides(self, **kwargs: Any) -> "XVAScenario":
        """Return a new scenario with parameter overrides."""

        updated: dict[str, Any] = dict(self.params)
        updated.update(kwargs)
        return XVAScenario(
            name=self.name,
            params=updated,
            weight=self.weight,
            metadata=self.metadata,
            market_scenario=self.market_scenario
        )

    def apply_to_environment(
        self,
        base_env: MarketDataEnvironment
    ) -> MarketDataEnvironment:
        """
        Apply scenario shocks to a base market environment.

        Args:
            base_env: Base market data environment

        Returns:
            Shocked environment

        Raises:
            ValueError: If no market_scenario is configured
        """
        if self.market_scenario is None:
            raise ValueError(
                f"Scenario '{self.name}' has no market_scenario configured. "
                "Cannot apply to environment."
            )
        return self.market_scenario.apply(base_env)


@dataclass
class ExposureResult:
    """
    Exposure summary for a single scenario.

    Attributes:
        scenario: XVA scenario definition
        times: Time grid for exposure profile
        expected_positive: Expected positive exposure (EPE)
        expected_negative: Expected negative exposure (ENE)
        net_exposure: Net expected exposure
        pathwise: Pathwise exposures (n_paths × n_times)
        params: Scenario parameters (legacy)
        market_env: Market data environment for this scenario (optional)
    """

    scenario: XVAScenario
    times: Array
    expected_positive: Array
    expected_negative: Array
    net_exposure: Array
    pathwise: Array
    params: Mapping[str, Any]
    market_env: Optional[MarketDataEnvironment] = None

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

# Environment-aware versions
EnvExposureGenerator = Callable[[jax.Array, MarketDataEnvironment, Mapping[str, Any]], Array | MCPaths]
EnvExposureFn = Callable[[Array, MarketDataEnvironment, Mapping[str, Any]], Array]


class ExposureSimulator:
    """
    Run exposure simulations across multiple scenarios.

    Supports both legacy parameter-based scenarios and new MarketDataEnvironment-based
    scenarios for comprehensive risk analysis.

    Examples:
        Legacy (parameter-based):
        >>> simulator = ExposureSimulator(
        ...     path_generator=my_path_gen,
        ...     exposure_fn=my_exposure_fn
        ... )
        >>> cube = simulator.simulate(key, base_params, scenarios)

        Environment-based:
        >>> simulator = ExposureSimulator(
        ...     path_generator=my_env_path_gen,
        ...     exposure_fn=my_env_exposure_fn,
        ...     base_market_env=market_env
        ... )
        >>> cube = simulator.simulate_with_environment(key, scenarios)
    """

    def __init__(
        self,
        path_generator: ExposureGenerator | EnvExposureGenerator,
        exposure_fn: ExposureFn | EnvExposureFn,
        *,
        time_grid: Array | None = None,
        base_market_env: Optional[MarketDataEnvironment] = None,
    ) -> None:
        """
        Initialize exposure simulator.

        Args:
            path_generator: Function to generate price paths
            exposure_fn: Function to compute exposure from paths
            time_grid: Optional fixed time grid for simulations
            base_market_env: Optional base market environment for scenario-based sims
        """
        self._path_generator = path_generator
        self._exposure_fn = exposure_fn
        self._time_grid = time_grid
        self._base_market_env = base_market_env

    def simulate(
        self,
        key: jax.Array,
        base_params: MutableMapping[str, Any],
        scenarios: Iterable[XVAScenario],
    ) -> ExposureCube:
        """
        Simulate exposures using legacy parameter-based approach.

        Args:
            key: JAX random key
            base_params: Base parameters dict
            scenarios: XVA scenarios with parameter overrides

        Returns:
            ExposureCube with all scenario results
        """
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

    def simulate_with_environment(
        self,
        key: jax.Array,
        scenarios: Iterable[XVAScenario],
        base_params: Optional[Mapping[str, Any]] = None,
    ) -> ExposureCube:
        """
        Simulate exposures using MarketDataEnvironment-based approach.

        Each scenario applies market data shocks to the base environment,
        enabling systematic risk analysis across rates, volatilities, and FX.

        Args:
            key: JAX random key
            scenarios: XVA scenarios with market_scenario configured
            base_params: Optional additional parameters (non-market data)

        Returns:
            ExposureCube with all scenario results

        Raises:
            ValueError: If base_market_env not configured

        Example:
            >>> from neutryx.valuations.scenarios import MarketScenario, CurveBumper
            >>> # Define market scenarios
            >>> rates_up = MarketScenario(
            ...     name="rates_up_100bps",
            ...     curve_shocks={
            ...         ('discount', 'USD'): lambda c: CurveBumper.parallel_shift(c, 100)
            ...     }
            ... )
            >>> scenarios = [
            ...     XVAScenario(name="base", market_scenario=None),
            ...     XVAScenario(name="rates_up", market_scenario=rates_up),
            ... ]
            >>> cube = simulator.simulate_with_environment(key, scenarios)
        """
        if self._base_market_env is None:
            raise ValueError(
                "simulate_with_environment requires base_market_env to be set. "
                "Pass base_market_env when initializing ExposureSimulator."
            )

        profiles: list[ExposureResult] = []
        params = dict(base_params) if base_params else {}

        for idx, scenario in enumerate(scenarios):
            # Apply market scenario shocks to base environment
            if scenario.market_scenario is not None:
                market_env = scenario.apply_to_environment(self._base_market_env)
            else:
                # Base scenario: use base environment
                market_env = self._base_market_env

            # Merge scenario params with base params
            scenario_params = dict(params)
            scenario_params.update(scenario.params)

            # Generate paths with environment
            scenario_key = jax.random.fold_in(key, idx)
            raw_paths = self._path_generator(scenario_key, market_env, scenario_params)
            paths, times = self._extract_paths_and_times(raw_paths, scenario_params)

            # Compute exposures with environment
            exposures = self._compute_pathwise_exposure_env(paths, market_env, scenario_params)

            # Aggregate exposure metrics
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
                params=scenario_params,
                market_env=market_env,
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
        """Compute exposures using legacy parameter-based exposure function."""
        exposures = jnp.asarray(self._exposure_fn(paths, params))
        if exposures.shape != paths.shape:
            raise ValueError("Exposure function must return an array with the same shape as the paths.")
        return exposures

    def _compute_pathwise_exposure_env(
        self,
        paths: Array,
        market_env: MarketDataEnvironment,
        params: Mapping[str, Any]
    ) -> Array:
        """Compute exposures using environment-aware exposure function."""
        exposures = jnp.asarray(self._exposure_fn(paths, market_env, params))
        if exposures.shape != paths.shape:
            raise ValueError("Exposure function must return an array with the same shape as the paths.")
        return exposures
