"""Calibration utilities."""

from .base import (
    CalibrationController,
    CalibrationResult,
    ParameterSpec,
    ParameterTransform,
    ParetoFront,
    ParetoSolution,
)
from .constraints import (
    g2pp_parameter_specs,
    quasi_gaussian_parameter_specs,
)
from .losses import (
    g2pp_zero_curve_loss,
    quasi_gaussian_zero_curve_loss,
)
from .diagnostics import (
    CalibrationDiagnostics,
    IdentifiabilityMetrics,
    ResidualPlotData,
    build_residual_plot_data,
    compute_identifiability_metrics,
    generate_calibration_diagnostics,
)
from .joint_calibration import (
    AssetClassSpec,
    CrossAssetCalibrator,
    InstrumentSpec,
    MultiInstrumentCalibrator,
    TimeDependentCalibrator,
    TimeSegment,
)
from .model_selection import (
    InformationCriterion,
    ModelFit,
    compute_aic,
    compute_bic,
    compute_aicc,
    compute_hqic,
    compute_information_criterion,
    ModelComparison,
    compare_models,
    CrossValidationResult,
    k_fold_split,
    time_series_split,
    cross_validate,
    LocalSensitivity,
    GlobalSensitivity,
    compute_local_sensitivity,
    sobol_indices,
)
from .bayesian_model_averaging import (
    BayesianModelAveraging,
    BMAResult,
    ModelWeights,
    WeightingScheme,
    compute_stacking_weights,
    compute_weights_from_ic,
    pseudo_bma_weights,
)

__all__ = [
    "AssetClassSpec",
    "CalibrationController",
    "CalibrationDiagnostics",
    "CalibrationResult",
    "ParetoFront",
    "ParetoSolution",
    "CrossAssetCalibrator",
    "IdentifiabilityMetrics",
    "InstrumentSpec",
    "MultiInstrumentCalibrator",
    "ParameterSpec",
    "ParameterTransform",
    "g2pp_parameter_specs",
    "quasi_gaussian_parameter_specs",
    "g2pp_zero_curve_loss",
    "quasi_gaussian_zero_curve_loss",
    "ResidualPlotData",
    "TimeDependentCalibrator",
    "TimeSegment",
    "build_residual_plot_data",
    "compute_identifiability_metrics",
    "generate_calibration_diagnostics",
    # Model Selection
    "InformationCriterion",
    "ModelFit",
    "compute_aic",
    "compute_bic",
    "compute_aicc",
    "compute_hqic",
    "compute_information_criterion",
    "ModelComparison",
    "compare_models",
    # Cross-Validation
    "CrossValidationResult",
    "k_fold_split",
    "time_series_split",
    "cross_validate",
    # Sensitivity Analysis
    "LocalSensitivity",
    "GlobalSensitivity",
    "compute_local_sensitivity",
    "sobol_indices",
    # Bayesian Model Averaging
    "BayesianModelAveraging",
    "BMAResult",
    "ModelWeights",
    "WeightingScheme",
    "compute_stacking_weights",
    "compute_weights_from_ic",
    "pseudo_bma_weights",
]
