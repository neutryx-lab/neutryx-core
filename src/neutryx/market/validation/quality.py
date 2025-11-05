"""Data quality checker and metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List
import logging

from .validators import ValidationResult, ValidationSeverity

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """
    Data quality metrics.

    Attributes:
        total_records: Total number of records processed
        valid_records: Number of valid records
        warning_records: Number of records with warnings
        error_records: Number of records with errors
        critical_records: Number of records with critical issues
        quality_score: Overall quality score (0-1)
        start_time: When quality assessment started
        end_time: When quality assessment ended
    """
    total_records: int = 0
    valid_records: int = 0
    warning_records: int = 0
    error_records: int = 0
    critical_records: int = 0
    quality_score: float = 1.0
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None

    def update(self, result: ValidationResult):
        """Update metrics with validation result."""
        self.total_records += 1

        if result.passed:
            self.valid_records += 1
        else:
            if result.severity == ValidationSeverity.WARNING:
                self.warning_records += 1
            elif result.severity == ValidationSeverity.ERROR:
                self.error_records += 1
            elif result.severity == ValidationSeverity.CRITICAL:
                self.critical_records += 1

        # Calculate quality score
        if self.total_records > 0:
            # Weight: critical=3, error=2, warning=1
            penalty = (
                self.critical_records * 3 +
                self.error_records * 2 +
                self.warning_records * 1
            )
            max_penalty = self.total_records * 3
            self.quality_score = 1.0 - (penalty / max_penalty) if max_penalty > 0 else 1.0

    def finalize(self):
        """Finalize metrics."""
        self.end_time = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "warning_records": self.warning_records,
            "error_records": self.error_records,
            "critical_records": self.critical_records,
            "quality_score": self.quality_score,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }


@dataclass
class QualityReport:
    """
    Data quality report.

    Attributes:
        metrics: Quality metrics
        validation_results: List of validation results
        issues_summary: Summary of issues by severity
        recommendations: Recommendations for improvement
    """
    metrics: QualityMetrics
    validation_results: List[ValidationResult] = field(default_factory=list)
    issues_summary: Dict[str, int] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def generate_summary(self):
        """Generate issues summary."""
        self.issues_summary = {
            "critical": 0,
            "error": 0,
            "warning": 0,
            "info": 0,
        }

        for result in self.validation_results:
            if not result.passed:
                severity_key = result.severity.value
                self.issues_summary[severity_key] = self.issues_summary.get(severity_key, 0) + 1

    def generate_recommendations(self):
        """Generate recommendations based on validation results."""
        self.recommendations = []

        if self.metrics.critical_records > 0:
            self.recommendations.append(
                f"CRITICAL: {self.metrics.critical_records} records have critical issues. "
                "Immediate attention required."
            )

        if self.metrics.error_records > 0:
            self.recommendations.append(
                f"ERROR: {self.metrics.error_records} records have errors. "
                "Review data source and connection."
            )

        if self.metrics.warning_records > self.metrics.total_records * 0.1:
            self.recommendations.append(
                f"WARNING: {self.metrics.warning_records} records have warnings "
                f"({self.metrics.warning_records / self.metrics.total_records:.1%}). "
                "Consider adjusting validation thresholds or investigating data source."
            )

        if self.metrics.quality_score < 0.9:
            self.recommendations.append(
                f"Quality score is {self.metrics.quality_score:.1%}. "
                "Recommend enabling additional data sources for cross-validation."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metrics": self.metrics.to_dict(),
            "issues_summary": self.issues_summary,
            "recommendations": self.recommendations,
        }


class DataQualityChecker:
    """
    Data quality checker for market data.

    Performs comprehensive quality checks on market data and generates
    detailed quality reports.
    """

    def __init__(self):
        """Initialize quality checker."""
        self.metrics = QualityMetrics()
        self.results: List[ValidationResult] = []

    def check(self, result: ValidationResult):
        """
        Check a validation result.

        Args:
            result: Validation result to check
        """
        self.results.append(result)
        self.metrics.update(result)

        # Log significant issues
        if not result.passed:
            if result.severity == ValidationSeverity.CRITICAL:
                logger.error(f"[{result.validator_name}] {result.message}")
            elif result.severity == ValidationSeverity.ERROR:
                logger.warning(f"[{result.validator_name}] {result.message}")
            elif result.severity == ValidationSeverity.WARNING:
                logger.info(f"[{result.validator_name}] {result.message}")

    def generate_report(self) -> QualityReport:
        """
        Generate quality report.

        Returns:
            QualityReport
        """
        self.metrics.finalize()

        report = QualityReport(
            metrics=self.metrics,
            validation_results=self.results,
        )

        report.generate_summary()
        report.generate_recommendations()

        return report

    def reset(self):
        """Reset quality checker."""
        self.metrics = QualityMetrics()
        self.results = []
