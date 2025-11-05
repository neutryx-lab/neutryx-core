"""Tests for market data validation."""

import pytest
from datetime import datetime
from neutryx.market.validation import (
    ValidationPipeline,
    PriceRangeValidator,
    SpreadValidator,
    VolumeValidator,
    ValidationSeverity
)


def test_price_range_validator():
    """Test price range validation."""
    validator = PriceRangeValidator(min_price=0, max_price=1000)

    # Valid price
    result = validator.validate({"price": 150.25})
    assert result.passed is True

    # Negative price
    result = validator.validate({"price": -10})
    assert result.passed is False
    assert result.severity == ValidationSeverity.ERROR

    # Price too high
    result = validator.validate({"price": 2000})
    assert result.passed is False
    assert result.severity == ValidationSeverity.WARNING


def test_spread_validator():
    """Test spread validation."""
    validator = SpreadValidator(max_spread_pct=0.05)

    # Valid spread
    result = validator.validate({"bid": 100.00, "ask": 100.50, "price": 100.25})
    assert result.passed is True

    # Negative spread (bid > ask)
    result = validator.validate({"bid": 100.50, "ask": 100.00})
    assert result.passed is False
    assert result.severity == ValidationSeverity.CRITICAL

    # Spread too wide
    result = validator.validate({"bid": 100.00, "ask": 120.00})
    assert result.passed is False
    assert result.severity == ValidationSeverity.WARNING


def test_volume_validator():
    """Test volume validation."""
    validator = VolumeValidator()

    # Valid volume
    result = validator.validate({"volume": 1000000})
    assert result.passed is True

    # Negative volume
    result = validator.validate({"volume": -100})
    assert result.passed is False
    assert result.severity == ValidationSeverity.ERROR


def test_validation_pipeline():
    """Test validation pipeline."""
    pipeline = ValidationPipeline()
    pipeline.add_validator(PriceRangeValidator(min_price=0, max_price=1000))
    pipeline.add_validator(SpreadValidator(max_spread_pct=0.05))

    # Valid data
    data = {
        "price": 150.25,
        "bid": 150.20,
        "ask": 150.30,
        "volume": 1000000
    }

    results = pipeline.validate(data)
    assert len(results) == 2
    assert all(r.passed for r in results)

    # Invalid data (negative spread)
    invalid_data = {
        "price": 150.25,
        "bid": 150.30,
        "ask": 150.20
    }

    results = pipeline.validate(invalid_data)
    assert any(not r.passed for r in results)

    # Get quality report
    report = pipeline.get_quality_report()
    assert report.metrics.total_records > 0
    assert 0.0 <= report.metrics.quality_score <= 1.0


def test_quality_report_generation():
    """Test quality report generation."""
    pipeline = ValidationPipeline()
    # Use validator without jump detection
    pipeline.add_validator(PriceRangeValidator(min_price=0, max_jump_pct=10.0))

    # Process multiple records
    for price in [100, 150, -10, 200]:  # One invalid (negative)
        pipeline.validate({"price": price})

    report = pipeline.get_quality_report()

    assert report.metrics.total_records == 4
    assert report.metrics.error_records >= 1  # At least the negative price
    assert report.metrics.quality_score < 1.0
    assert len(report.recommendations) > 0
