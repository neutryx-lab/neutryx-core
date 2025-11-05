"""Joint calibration framework for multi-instrument, cross-asset, and time-dependent fitting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp
import optax

from .base import CalibrationController, CalibrationResult, ParameterSpec
from .losses import mean_squared_error

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


@dataclass
class InstrumentSpec:
    """Specification for a single instrument in multi-instrument calibration.

    Attributes:
        name: Identifier for the instrument
        pricing_fn: Function that takes (params, market_data) -> prices
        target_prices: Observed market prices
        weights: Optional weights for each observation
        market_data: Additional data needed for pricing (strikes, maturities, etc.)
    """

    name: str
    pricing_fn: Callable[[Mapping[str, Array], Mapping[str, Array]], Array]
    target_prices: Array
    weights: Optional[Array] = None
    market_data: Optional[Mapping[str, Array]] = None


@dataclass
class AssetClassSpec:
    """Specification for a single asset class in cross-asset calibration.

    Attributes:
        name: Asset class identifier (e.g., 'equity', 'fx', 'rates')
        parameter_specs: Parameter specifications for this asset class
        instruments: List of instruments for this asset class
        shared_params: List of parameter names shared across asset classes
    """

    name: str
    parameter_specs: Mapping[str, ParameterSpec]
    instruments: List[InstrumentSpec]
    shared_params: Optional[List[str]] = None


@dataclass
class TimeSegment:
    """Time segment for piecewise constant parameters.

    Attributes:
        start_time: Segment start time
        end_time: Segment end time
        parameter_specs: Parameters for this segment
    """

    start_time: float
    end_time: float
    parameter_specs: Mapping[str, ParameterSpec]


class MultiInstrumentCalibrator(CalibrationController):
    """Calibrate a single model to multiple instruments simultaneously.

    This calibrator fits one set of model parameters to multiple instruments
    (e.g., options at different strikes/maturities, caps, floors, swaptions).

    Example:
        Calibrate SABR to both ATM options and OTM options with different weights:

        instruments = [
            InstrumentSpec(
                name="atm_options",
                pricing_fn=lambda p, d: price_options(p, d['strikes'], d['maturities']),
                target_prices=atm_prices,
                weights=jnp.ones(len(atm_prices)) * 2.0,  # Higher weight on ATM
                market_data={'strikes': atm_strikes, 'maturities': atm_mats}
            ),
            InstrumentSpec(
                name="otm_options",
                pricing_fn=lambda p, d: price_options(p, d['strikes'], d['maturities']),
                target_prices=otm_prices,
                weights=jnp.ones(len(otm_prices)),
                market_data={'strikes': otm_strikes, 'maturities': otm_mats}
            ),
        ]

        calibrator = MultiInstrumentCalibrator(
            parameter_specs=sabr_specs,
            instruments=instruments,
        )

        result = calibrator.calibrate({})
    """

    def __init__(
        self,
        parameter_specs: Mapping[str, ParameterSpec],
        instruments: List[InstrumentSpec],
        loss_fn=mean_squared_error,
        optimizer: Optional[optax.GradientTransformation] = None,
        penalty_fn: Optional[Callable] = None,
        max_steps: int = 500,
        tol: float = 1e-8,
    ):
        """Initialize multi-instrument calibrator.

        Args:
            parameter_specs: Model parameter specifications
            instruments: List of instruments to calibrate to
            loss_fn: Loss function (default: mean squared error)
            optimizer: Optax optimizer (default: Adam with gradient clipping)
            penalty_fn: Optional penalty/regularization function
            max_steps: Maximum optimization steps
            tol: Convergence tolerance
        """
        if optimizer is None:
            optimizer = optax.chain(optax.clip(1.0), optax.adam(1e-2))

        super().__init__(
            parameter_specs=parameter_specs,
            loss_fn=loss_fn,
            optimizer=optimizer,
            penalty_fn=penalty_fn,
            max_steps=max_steps,
            tol=tol,
        )

        self.instruments = instruments

    def _target_observables(self, market_data: Mapping[str, Array]) -> Array:
        """Concatenate all target prices from all instruments."""
        targets = []
        for instrument in self.instruments:
            targets.append(instrument.target_prices)
        return jnp.concatenate(targets)

    def _model_observables(
        self, params: Mapping[str, Array], market_data: Mapping[str, Array]
    ) -> Array:
        """Compute model prices for all instruments."""
        predictions = []
        for instrument in self.instruments:
            instr_data = instrument.market_data if instrument.market_data else {}
            pred = instrument.pricing_fn(params, instr_data)
            predictions.append(pred)
        return jnp.concatenate(predictions)

    def _prepare_market_data(self, market_data: Mapping[str, Any]) -> Mapping[str, Array]:
        """Prepare market data with instrument weights."""
        data = super()._prepare_market_data(market_data)

        # Concatenate weights from all instruments
        weights = []
        for instrument in self.instruments:
            if instrument.weights is not None:
                weights.append(instrument.weights)
            else:
                # Default weight of 1.0 for each observation
                weights.append(jnp.ones(len(instrument.target_prices)))

        data["weights"] = jnp.concatenate(weights)
        return data


class CrossAssetCalibrator:
    """Calibrate models across multiple asset classes with shared parameters.

    This calibrator fits parameters across different asset classes, allowing
    some parameters to be shared (e.g., correlation parameters) while others
    are asset-class specific.

    Example:
        Calibrate FX and equity volatility models with shared correlation:

        fx_spec = AssetClassSpec(
            name='fx',
            parameter_specs={'vol_fx': ParameterSpec(0.15, positive())},
            instruments=[fx_option_spec],
            shared_params=['correlation']
        )

        equity_spec = AssetClassSpec(
            name='equity',
            parameter_specs={'vol_eq': ParameterSpec(0.25, positive())},
            instruments=[equity_option_spec],
            shared_params=['correlation']
        )

        shared_specs = {'correlation': ParameterSpec(-0.5, symmetric(0.999))}

        calibrator = CrossAssetCalibrator(
            asset_classes=[fx_spec, equity_spec],
            shared_parameter_specs=shared_specs,
        )

        result = calibrator.calibrate({})
    """

    def __init__(
        self,
        asset_classes: List[AssetClassSpec],
        shared_parameter_specs: Optional[Mapping[str, ParameterSpec]] = None,
        loss_fn=mean_squared_error,
        optimizer: Optional[optax.GradientTransformation] = None,
        penalty_fn: Optional[Callable] = None,
        max_steps: int = 600,
        tol: float = 1e-8,
    ):
        """Initialize cross-asset calibrator.

        Args:
            asset_classes: List of asset class specifications
            shared_parameter_specs: Specifications for shared parameters
            loss_fn: Loss function
            optimizer: Optax optimizer
            penalty_fn: Optional penalty function
            max_steps: Maximum optimization steps
            tol: Convergence tolerance
        """
        if optimizer is None:
            optimizer = optax.chain(optax.clip(1.0), optax.adam(1e-2))

        self.asset_classes = asset_classes
        self.shared_parameter_specs = shared_parameter_specs or {}
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.penalty_fn = penalty_fn
        self.max_steps = max_steps
        self.tol = tol
        self.dtype = jnp.float64

        # Build combined parameter specs
        self._build_parameter_structure()

    def _build_parameter_structure(self):
        """Build combined parameter structure with namespacing."""
        self.all_parameter_specs: Dict[str, ParameterSpec] = {}

        # Add shared parameters
        for name, spec in self.shared_parameter_specs.items():
            self.all_parameter_specs[f"shared_{name}"] = spec

        # Add asset-class specific parameters
        for asset_class in self.asset_classes:
            for name, spec in asset_class.parameter_specs.items():
                param_name = f"{asset_class.name}_{name}"
                self.all_parameter_specs[param_name] = spec

    def _initial_theta(self) -> Dict[str, Array]:
        """Initialize unconstrained parameters."""
        theta: Dict[str, Array] = {}
        for name, spec in self.all_parameter_specs.items():
            init_value = jnp.asarray(spec.init, dtype=self.dtype)
            theta[name] = spec.transform.invert(init_value)
        return theta

    def _constrain_params(self, theta: Mapping[str, Array]) -> Dict[str, Array]:
        """Apply parameter constraints."""
        return {
            name: spec.transform.apply(theta[name])
            for name, spec in self.all_parameter_specs.items()
        }

    def _extract_asset_params(
        self, params: Mapping[str, Array], asset_class_name: str
    ) -> Dict[str, Array]:
        """Extract parameters for a specific asset class."""
        asset_params: Dict[str, Array] = {}

        # Add shared parameters
        for name in self.shared_parameter_specs:
            asset_params[name] = params[f"shared_{name}"]

        # Add asset-specific parameters
        prefix = f"{asset_class_name}_"
        for param_name in params:
            if param_name.startswith(prefix):
                clean_name = param_name[len(prefix):]
                asset_params[clean_name] = params[param_name]

        return asset_params

    def calibrate(self, market_data: Optional[Mapping[str, Any]] = None) -> CalibrationResult:
        """Run cross-asset calibration.

        Args:
            market_data: Optional additional market data

        Returns:
            CalibrationResult with converged parameters
        """
        theta = self._initial_theta()
        opt_state = self.optimizer.init(theta)

        def objective(current_theta: Mapping[str, Array]) -> Array:
            """Compute total loss across all asset classes."""
            constrained = self._constrain_params(current_theta)
            total_loss = 0.0

            # Compute loss for each asset class
            for asset_class in self.asset_classes:
                # Extract parameters for this asset class
                asset_params = self._extract_asset_params(constrained, asset_class.name)

                # Compute loss for each instrument in this asset class
                for instrument in asset_class.instruments:
                    instr_data = instrument.market_data if instrument.market_data else {}
                    predicted = instrument.pricing_fn(asset_params, instr_data)
                    target = instrument.target_prices
                    weights = instrument.weights

                    loss = self.loss_fn(
                        predicted, target, weights=weights, params=asset_params,
                        market_data=instr_data
                    )
                    total_loss = total_loss + loss

            # Add penalty if provided
            if self.penalty_fn is not None:
                total_loss = total_loss + self.penalty_fn(constrained, market_data or {})

            return total_loss

        loss_and_grad = jax.jit(jax.value_and_grad(objective))

        loss_history: List[float] = []
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


class TimeDependentCalibrator(CalibrationController):
    """Calibrate time-dependent parameters (piecewise constant or smooth).

    This calibrator fits parameters that vary over time, useful for:
    - Term structure of volatility
    - Time-varying correlation
    - Local volatility surfaces

    Supports two modes:
    1. Piecewise constant: Different parameter values in each time segment
    2. Smooth parametric: Parameters follow a smooth function (e.g., polynomial)

    Example:
        Piecewise constant volatility calibration:

        segments = [
            TimeSegment(0.0, 0.5, {'vol': ParameterSpec(0.2, positive())}),
            TimeSegment(0.5, 1.0, {'vol': ParameterSpec(0.25, positive())}),
            TimeSegment(1.0, 2.0, {'vol': ParameterSpec(0.22, positive())}),
        ]

        calibrator = TimeDependentCalibrator(
            time_segments=segments,
            pricing_fn=lambda params, t, data: price_with_vol(params['vol'], t, data),
        )

        result = calibrator.calibrate(market_data)
    """

    def __init__(
        self,
        time_segments: List[TimeSegment],
        pricing_fn: Callable[[Mapping[str, Array], Array, Mapping[str, Array]], Array],
        target_prices: Array,
        observation_times: Array,
        weights: Optional[Array] = None,
        loss_fn=mean_squared_error,
        optimizer: Optional[optax.GradientTransformation] = None,
        penalty_fn: Optional[Callable] = None,
        max_steps: int = 500,
        tol: float = 1e-8,
        smoothness_penalty: float = 0.0,
    ):
        """Initialize time-dependent calibrator.

        Args:
            time_segments: List of time segments with parameter specs
            pricing_fn: Function (params, time, market_data) -> price
            target_prices: Observed market prices
            observation_times: Time points for each observation
            weights: Optional observation weights
            loss_fn: Loss function
            optimizer: Optax optimizer
            penalty_fn: Optional penalty function
            max_steps: Maximum optimization steps
            tol: Convergence tolerance
            smoothness_penalty: Penalty on parameter jumps between segments
        """
        if optimizer is None:
            optimizer = optax.chain(optax.clip(1.0), optax.adam(1e-2))

        # Build combined parameter specs with time-segment prefixes
        combined_specs: Dict[str, ParameterSpec] = {}
        for i, segment in enumerate(time_segments):
            for param_name, spec in segment.parameter_specs.items():
                combined_specs[f"t{i}_{param_name}"] = spec

        super().__init__(
            parameter_specs=combined_specs,
            loss_fn=loss_fn,
            optimizer=optimizer,
            penalty_fn=penalty_fn,
            max_steps=max_steps,
            tol=tol,
        )

        self.time_segments = time_segments
        self.pricing_fn = pricing_fn
        self.target_prices = target_prices
        self.observation_times = observation_times
        self.weights_array = weights
        self.smoothness_penalty = smoothness_penalty

        # Pre-compute segment indices for each observation (JAX-compatible)
        self._segment_indices = self._compute_segment_indices()

    def _compute_segment_indices(self) -> List[int]:
        """Pre-compute segment indices for each observation time."""
        indices = []
        for obs_time in self.observation_times:
            t = float(obs_time)
            for i, segment in enumerate(self.time_segments):
                if segment.start_time <= t < segment.end_time:
                    indices.append(i)
                    break
            else:
                # If time is beyond last segment, use last segment
                indices.append(len(self.time_segments) - 1)
        return indices

    def _get_segment_index(self, time: float) -> int:
        """Find the time segment index for a given time."""
        for i, segment in enumerate(self.time_segments):
            if segment.start_time <= time < segment.end_time:
                return i
        # Return last segment for times beyond the last segment
        return len(self.time_segments) - 1

    def _extract_segment_params(
        self, params: Mapping[str, Array], segment_idx: int
    ) -> Dict[str, Array]:
        """Extract parameters for a specific time segment."""
        segment_params: Dict[str, Array] = {}
        prefix = f"t{segment_idx}_"

        for param_name in params:
            if param_name.startswith(prefix):
                clean_name = param_name[len(prefix):]
                segment_params[clean_name] = params[param_name]

        return segment_params

    def _target_observables(self, market_data: Mapping[str, Array]) -> Array:
        """Return target prices."""
        return self.target_prices

    def _model_observables(
        self, params: Mapping[str, Array], market_data: Mapping[str, Array]
    ) -> Array:
        """Compute model prices using time-dependent parameters."""
        predictions = []

        # Use pre-computed segment indices (not traced by JAX)
        for i, obs_time in enumerate(self.observation_times):
            segment_idx = self._segment_indices[i]  # Use pre-computed index
            segment_params = self._extract_segment_params(params, segment_idx)

            # Price using segment-specific parameters
            price = self.pricing_fn(segment_params, obs_time, market_data)
            predictions.append(price)

        return jnp.array(predictions)

    def _prepare_market_data(self, market_data: Mapping[str, Any]) -> Mapping[str, Array]:
        """Prepare market data with weights."""
        data = super()._prepare_market_data(market_data)

        if self.weights_array is not None:
            data["weights"] = self.weights_array

        return data

    def _smoothness_penalty_fn(self, params: Mapping[str, Array]) -> Array:
        """Penalize large jumps in parameters between adjacent segments."""
        if self.smoothness_penalty <= 0:
            return 0.0

        penalty = 0.0

        # For each parameter name, compute jumps between segments
        param_names = set()
        for segment in self.time_segments:
            param_names.update(segment.parameter_specs.keys())

        for param_name in param_names:
            for i in range(len(self.time_segments) - 1):
                try:
                    current_val = params[f"t{i}_{param_name}"]
                    next_val = params[f"t{i+1}_{param_name}"]
                    jump = (next_val - current_val) ** 2
                    penalty = penalty + jump
                except KeyError:
                    # Parameter doesn't exist in this segment
                    continue

        return self.smoothness_penalty * penalty

    def calibrate(self, market_data: Mapping[str, Any]) -> CalibrationResult:
        """Run time-dependent calibration."""
        # Add smoothness penalty to base penalty
        original_penalty = self.penalty_fn

        def combined_penalty(params: Mapping[str, Array], data: Mapping[str, Array]) -> Array:
            penalty = self._smoothness_penalty_fn(params)
            if original_penalty is not None:
                penalty = penalty + original_penalty(params, data)
            return penalty

        self.penalty_fn = combined_penalty
        result = super().calibrate(market_data)
        self.penalty_fn = original_penalty  # Restore original

        return result


__all__ = [
    "MultiInstrumentCalibrator",
    "CrossAssetCalibrator",
    "TimeDependentCalibrator",
    "InstrumentSpec",
    "AssetClassSpec",
    "TimeSegment",
]
