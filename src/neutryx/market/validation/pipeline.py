"""Validation pipeline for market data."""

from __future__ import annotations

from typing import Any, Dict, List
import logging

from .validators import BaseValidator, ValidationResult
from .quality import DataQualityChecker, QualityReport

logger = logging.getLogger(__name__)


class ValidationPipeline:
    """
    Validation pipeline for market data.

    Orchestrates multiple validators and generates quality reports.

    Example:
        >>> from neutryx.market.validation import (
        ...     ValidationPipeline,
        ...     PriceRangeValidator,
        ...     SpreadValidator,
        ...     VolumeValidator
        ... )
        >>>
        >>> pipeline = ValidationPipeline()
        >>> pipeline.add_validator(PriceRangeValidator(min_price=0, max_price=1000))
        >>> pipeline.add_validator(SpreadValidator(max_spread_pct=0.05))
        >>> pipeline.add_validator(VolumeValidator())
        >>>
        >>> data = {
        ...     "price": 150.25,
        ...     "bid": 150.20,
        ...     "ask": 150.30,
        ...     "volume": 1000000
        ... }
        >>> results = pipeline.validate(data)
        >>> report = pipeline.get_quality_report()
    """

    def __init__(self):
        """Initialize validation pipeline."""
        self.validators: List[BaseValidator] = []
        self.quality_checker = DataQualityChecker()

    def add_validator(self, validator: BaseValidator):
        """
        Add validator to pipeline.

        Args:
            validator: Validator to add
        """
        self.validators.append(validator)
        logger.info(f"Added validator: {validator.name}")

    def validate(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """
        Validate data through all validators.

        Args:
            data: Data to validate

        Returns:
            List of validation results
        """
        results = []

        for validator in self.validators:
            try:
                result = validator.validate(data)
                results.append(result)
                self.quality_checker.check(result)
            except Exception as e:
                logger.error(f"Error in validator {validator.name}: {e}")
                # Continue with other validators

        return results

    def validate_batch(self, data_list: List[Dict[str, Any]]) -> List[List[ValidationResult]]:
        """
        Validate batch of data.

        Args:
            data_list: List of data records to validate

        Returns:
            List of validation results for each record
        """
        all_results = []

        for data in data_list:
            results = self.validate(data)
            all_results.append(results)

        return all_results

    def get_quality_report(self) -> QualityReport:
        """
        Get quality report.

        Returns:
            QualityReport
        """
        return self.quality_checker.generate_report()

    def reset(self):
        """Reset pipeline and quality checker."""
        self.quality_checker.reset()

    def get_failed_validations(self) -> List[ValidationResult]:
        """
        Get all failed validations.

        Returns:
            List of failed validation results
        """
        return [r for r in self.quality_checker.results if not r.passed]

    def get_quality_score(self) -> float:
        """
        Get current quality score.

        Returns:
            Quality score (0-1)
        """
        return self.quality_checker.metrics.quality_score
