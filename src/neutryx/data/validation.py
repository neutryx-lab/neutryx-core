"""Data validation and quality control utilities for market data feeds.

Provides a flexible rule-based validation engine that can be attached to
real-time market data feeds and integrations. The validator computes
quality flags, attaches structured metadata about detected issues, and
supports custom validation rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import logging

from neutryx.market.data_models import DataQuality, MarketDataPoint
from neutryx.data.security_master import (
    SecurityInactiveError,
    SecurityMaster,
    SecurityNotFoundError,
)


logger = logging.getLogger(__name__)


class Severity(Enum):
    """Severity of a validation issue."""

    INFO = auto()
    WARNING = auto()
    ERROR = auto()

    @property
    def weight(self) -> int:
        """Monotonic weight used for comparisons."""
        return {
            Severity.INFO: 0,
            Severity.WARNING: 1,
            Severity.ERROR: 2,
        }[self]


@dataclass
class ValidationIssue:
    """Represents a validation issue detected by a rule."""

    rule: str
    message: str
    severity: Severity = Severity.ERROR
    field: Optional[str] = None
    value: Any = None

    def to_metadata(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation for metadata."""
        return {
            "rule": self.rule,
            "message": self.message,
            "severity": self.severity.name.lower(),
            "field": self.field,
            "value": self.value,
        }


@dataclass
class ValidationResult:
    """Aggregated result of running validation rules for a data point."""

    is_valid: bool
    quality: DataQuality
    issues: List[ValidationIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ValidationRule:
    """Base class for validation rules."""

    name: str = "rule"
    severity: Severity = Severity.ERROR

    def evaluate(self, data_point: MarketDataPoint) -> Sequence[ValidationIssue]:
        """Evaluate the rule against the data point."""
        raise NotImplementedError


class RequiredFieldRule(ValidationRule):
    """Ensures that required fields are populated."""

    name = "required_field"

    def __init__(
        self,
        fields: Sequence[str],
        *,
        allow_zero: bool = True,
        severity: Severity = Severity.ERROR,
    ):
        self.fields = list(fields)
        self.allow_zero = allow_zero
        self.severity = severity

    def evaluate(self, data_point: MarketDataPoint) -> Sequence[ValidationIssue]:
        issues: List[ValidationIssue] = []

        for field in self.fields:
            value = getattr(data_point, field, None)

            if value is None:
                issues.append(
                    ValidationIssue(
                        rule=self.name,
                        message=f"Missing required field '{field}'",
                        field=field,
                        severity=self.severity,
                    )
                )
            elif not self.allow_zero and value == 0:
                issues.append(
                    ValidationIssue(
                        rule=self.name,
                        message=f"Field '{field}' must be non-zero",
                        field=field,
                        value=value,
                        severity=self.severity,
                    )
                )

        return issues


class RangeRule(ValidationRule):
    """Validates that a numeric field lies within an inclusive range."""

    name = "range"

    def __init__(
        self,
        field: str,
        *,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        severity: Severity = Severity.ERROR,
    ):
        self.field = field
        self.minimum = minimum
        self.maximum = maximum
        self.severity = severity

    def evaluate(self, data_point: MarketDataPoint) -> Sequence[ValidationIssue]:
        value = getattr(data_point, self.field, None)
        issues: List[ValidationIssue] = []

        if value is None:
            return issues

        if self.minimum is not None and value < self.minimum:
            issues.append(
                ValidationIssue(
                    rule=self.name,
                    message=f"Field '{self.field}' below minimum {self.minimum}",
                    field=self.field,
                    value=value,
                    severity=self.severity,
                )
            )

        if self.maximum is not None and value > self.maximum:
            issues.append(
                ValidationIssue(
                    rule=self.name,
                    message=f"Field '{self.field}' above maximum {self.maximum}",
                    field=self.field,
                    value=value,
                    severity=self.severity,
                )
            )

        return issues


class StalenessRule(ValidationRule):
    """Validates that data is recent."""

    name = "stale_data"

    def __init__(
        self,
        max_age: timedelta,
        *,
        severity: Severity = Severity.WARNING,
    ):
        self.max_age = max_age
        self.severity = severity

    def evaluate(self, data_point: MarketDataPoint) -> Sequence[ValidationIssue]:
        timestamp = getattr(data_point, "timestamp", None)
        if timestamp is None:
            return [
                ValidationIssue(
                    rule=self.name,
                    message="Data point missing timestamp",
                    field="timestamp",
                    severity=self.severity,
                )
            ]

        now = datetime.utcnow()
        if now - timestamp > self.max_age:
            return [
                ValidationIssue(
                    rule=self.name,
                    message=f"Data older than {self.max_age.total_seconds()} seconds",
                    field="timestamp",
                    value=timestamp.isoformat(),
                    severity=self.severity,
                )
            ]

        return []


class SecurityReferenceRule(ValidationRule):
    """Ensure market data references securities in the master."""

    name = "security_master_reference"

    def __init__(
        self,
        security_master: SecurityMaster,
        *,
        identifier_fields: Sequence[str] = ("security_id", "ticker", "isin", "cusip"),
        attach_metadata: bool = True,
    ):
        self.security_master = security_master
        self.identifier_fields = list(identifier_fields)
        self.attach_metadata = attach_metadata

    def _identifier_candidates(
        self, data_point: MarketDataPoint
    ) -> List[Tuple[str, str]]:
        metadata: Dict[str, Any] = getattr(data_point, "metadata", {}) or {}
        candidates: List[Tuple[str, str]] = []

        for field in self.identifier_fields:
            value: Optional[str]
            if field == "security_id":
                value = metadata.get("security_id")
            else:
                value = getattr(data_point, field, None) or metadata.get(field)

            if value:
                candidates.append((field, str(value)))

        # Maintain input order but remove duplicates
        seen: set[Tuple[str, str]] = set()
        unique_candidates: List[Tuple[str, str]] = []
        for candidate in candidates:
            key = (candidate[0], candidate[1].upper())
            if key not in seen:
                seen.add(key)
                unique_candidates.append(candidate)

        return unique_candidates

    def evaluate(self, data_point: MarketDataPoint) -> Sequence[ValidationIssue]:
        candidates = self._identifier_candidates(data_point)
        as_of_date = None
        timestamp = getattr(data_point, "timestamp", None)
        if hasattr(timestamp, "date"):
            as_of_date = timestamp.date()

        metadata: Dict[str, Any] = getattr(data_point, "metadata", {}) or {}

        for identifier_type, identifier_value in candidates:
            lookup_type = identifier_type if identifier_type != "security_id" else "security_id"
            try:
                record = self.security_master.lookup(
                    identifier_value,
                    identifier_type=lookup_type,
                    as_of=as_of_date,
                    require_active=False,
                )
            except SecurityNotFoundError:
                continue
            except SecurityInactiveError:
                return [
                    ValidationIssue(
                        rule=self.name,
                        message=(
                            f"Security '{identifier_value}' inactive for identifier "
                            f"'{identifier_type}'"
                        ),
                        field=identifier_type,
                        value=identifier_value,
                        severity=Severity.ERROR,
                    )
                ]

            if as_of_date and not record.is_active(as_of=as_of_date):
                return [
                    ValidationIssue(
                        rule=self.name,
                        message=(
                            f"Security '{record.security_id}' inactive on {as_of_date.isoformat()}"
                        ),
                        field=identifier_type,
                        value=identifier_value,
                        severity=Severity.ERROR,
                    )
                ]

            if self.attach_metadata:
                enriched = dict(metadata)
                enriched.setdefault("security_master", record.to_summary())
                enriched.setdefault(
                    "security_master_lookup",
                    {"identifier_type": identifier_type, "identifier_value": identifier_value},
                )
                data_point.metadata = enriched
                metadata = enriched

            return []

        return [
            ValidationIssue(
                rule=self.name,
                message="Security reference not found in security master",
                severity=Severity.ERROR,
            )
        ]


class DataValidator:
    """Rule-based validator for market data points."""

    def __init__(
        self,
        rules: Optional[Iterable[ValidationRule]] = None,
        *,
        quality_overrides: Optional[Dict[Severity, DataQuality]] = None,
        attach_metadata: bool = True,
        security_master: Optional[SecurityMaster] = None,
        security_identifier_fields: Sequence[str] = (
            "security_id",
            "ticker",
            "isin",
            "cusip",
        ),
    ):
        self._rules: List[ValidationRule] = list(rules) if rules else []
        self._attach_metadata = attach_metadata
        self._security_master = security_master
        self._quality_overrides = {
            Severity.INFO: None,
            Severity.WARNING: DataQuality.INDICATIVE,
            Severity.ERROR: DataQuality.STALE,
        }

        if quality_overrides:
            self._quality_overrides.update(quality_overrides)

        if self._security_master is not None:
            security_rule = SecurityReferenceRule(
                self._security_master,
                identifier_fields=security_identifier_fields,
                attach_metadata=attach_metadata,
            )
            self._rules.insert(0, security_rule)

    def add_rule(self, rule: ValidationRule) -> None:
        """Register a validation rule."""
        self._rules.append(rule)

    def validate(self, data_point: MarketDataPoint) -> ValidationResult:
        """Validate a market data point and return the result."""
        issues: List[ValidationIssue] = []

        for rule in self._rules:
            try:
                issues.extend(rule.evaluate(data_point))
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Validation rule '%s' failed: %s", rule.name, exc)
                issues.append(
                    ValidationIssue(
                        rule=rule.name,
                        message=f"Rule execution error: {exc}",
                        severity=Severity.ERROR,
                    )
                )

        if not issues:
            return ValidationResult(
                is_valid=True,
                quality=data_point.quality,
                issues=[],
                metadata=data_point.metadata,
            )

        worst_issue = max(issues, key=lambda issue: issue.severity.weight)
        override_quality = self._quality_overrides.get(worst_issue.severity)
        quality = override_quality or data_point.quality

        metadata = dict(data_point.metadata)
        if self._attach_metadata:
            metadata.setdefault("quality_issues", [])
            for issue in issues:
                metadata["quality_issues"].append(issue.to_metadata())

        data_point.metadata = metadata
        data_point.quality = quality

        return ValidationResult(
            is_valid=worst_issue.severity != Severity.ERROR,
            quality=quality,
            issues=issues,
            metadata=metadata,
        )


__all__ = [
    "DataValidator",
    "Severity",
    "ValidationIssue",
    "ValidationResult",
    "ValidationRule",
    "RequiredFieldRule",
    "RangeRule",
    "StalenessRule",
    "SecurityReferenceRule",
]
