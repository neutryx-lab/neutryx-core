"""Calibration utilities."""

from .base import (
    CalibrationController,
    CalibrationResult,
    ParameterSpec,
    ParameterTransform,
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

__all__ = [
    "AssetClassSpec",
    "CalibrationController",
    "CalibrationDiagnostics",
    "CalibrationResult",
    "CrossAssetCalibrator",
    "IdentifiabilityMetrics",
    "InstrumentSpec",
    "MultiInstrumentCalibrator",
    "ParameterSpec",
    "ParameterTransform",
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
]
