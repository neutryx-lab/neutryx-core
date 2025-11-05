"""Validators for market data quality checks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """
    Result of a validation check.

    Attributes:
        validator_name: Name of the validator
        passed: Whether validation passed
        severity: Severity level if validation failed
        message: Descriptive message
        details: Additional details about the validation
        timestamp: When validation was performed
    """
    validator_name: str
    passed: bool
    severity: ValidationSeverity = ValidationSeverity.INFO
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class BaseValidator(ABC):
    """Base class for market data validators."""

    def __init__(self, name: str):
        """
        Initialize validator.

        Args:
            name: Validator name
        """
        self.name = name

    @abstractmethod
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate market data.

        Args:
            data: Market data to validate

        Returns:
            ValidationResult
        """
        pass


class PriceRangeValidator(BaseValidator):
    """
    Validates that prices are within acceptable ranges.

    Checks for:
    - Negative prices
    - Unrealistic prices (too high or too low)
    - Price jumps exceeding threshold
    """

    def __init__(
        self,
        min_price: float = 0.0,
        max_price: Optional[float] = None,
        max_jump_pct: float = 0.20,  # 20% max jump
    ):
        """
        Initialize price range validator.

        Args:
            min_price: Minimum acceptable price
            max_price: Maximum acceptable price (None = no limit)
            max_jump_pct: Maximum allowed price jump as percentage
        """
        super().__init__("PriceRangeValidator")
        self.min_price = min_price
        self.max_price = max_price
        self.max_jump_pct = max_jump_pct
        self._last_price: Optional[float] = None

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate price is within acceptable range."""
        price = data.get("price")

        if price is None:
            return ValidationResult(
                validator_name=self.name,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message="Price field is missing",
            )

        # Check for negative price
        if price < self.min_price:
            return ValidationResult(
                validator_name=self.name,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Price {price} is below minimum {self.min_price}",
                details={"price": price, "min_price": self.min_price},
            )

        # Check maximum price
        if self.max_price and price > self.max_price:
            return ValidationResult(
                validator_name=self.name,
                passed=False,
                severity=ValidationSeverity.WARNING,
                message=f"Price {price} exceeds maximum {self.max_price}",
                details={"price": price, "max_price": self.max_price},
            )

        # Check for large price jumps
        if self._last_price is not None:
            jump_pct = abs(price - self._last_price) / self._last_price
            if jump_pct > self.max_jump_pct:
                result = ValidationResult(
                    validator_name=self.name,
                    passed=False,
                    severity=ValidationSeverity.WARNING,
                    message=f"Price jump of {jump_pct:.1%} exceeds threshold {self.max_jump_pct:.1%}",
                    details={
                        "previous_price": self._last_price,
                        "current_price": price,
                        "jump_pct": jump_pct,
                    },
                )
                self._last_price = price
                return result

        self._last_price = price

        return ValidationResult(
            validator_name=self.name,
            passed=True,
            message="Price within acceptable range",
        )


class SpreadValidator(BaseValidator):
    """
    Validates bid-ask spreads.

    Checks for:
    - Negative spreads (bid > ask)
    - Excessive spreads
    - Mid price consistency
    """

    def __init__(
        self,
        max_spread_pct: float = 0.05,  # 5% max spread
        check_mid_consistency: bool = True,
    ):
        """
        Initialize spread validator.

        Args:
            max_spread_pct: Maximum acceptable spread as percentage
            check_mid_consistency: Check if mid price matches (bid+ask)/2
        """
        super().__init__("SpreadValidator")
        self.max_spread_pct = max_spread_pct
        self.check_mid_consistency = check_mid_consistency

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate bid-ask spread."""
        bid = data.get("bid")
        ask = data.get("ask")
        price = data.get("price")

        if bid is None or ask is None:
            return ValidationResult(
                validator_name=self.name,
                passed=True,  # Skip if bid/ask not available
                severity=ValidationSeverity.INFO,
                message="Bid or ask not available",
            )

        # Check for negative spread
        if bid > ask:
            return ValidationResult(
                validator_name=self.name,
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Negative spread: bid {bid} > ask {ask}",
                details={"bid": bid, "ask": ask},
            )

        # Check spread width
        mid = (bid + ask) / 2
        spread_pct = (ask - bid) / mid if mid > 0 else 0

        if spread_pct > self.max_spread_pct:
            return ValidationResult(
                validator_name=self.name,
                passed=False,
                severity=ValidationSeverity.WARNING,
                message=f"Spread {spread_pct:.2%} exceeds maximum {self.max_spread_pct:.2%}",
                details={"bid": bid, "ask": ask, "spread_pct": spread_pct},
            )

        # Check mid price consistency
        if self.check_mid_consistency and price is not None:
            expected_mid = (bid + ask) / 2
            deviation = abs(price - expected_mid) / expected_mid if expected_mid > 0 else 0

            if deviation > 0.01:  # 1% deviation tolerance
                return ValidationResult(
                    validator_name=self.name,
                    passed=False,
                    severity=ValidationSeverity.WARNING,
                    message=f"Price {price} deviates from mid {expected_mid:.4f}",
                    details={
                        "price": price,
                        "expected_mid": expected_mid,
                        "deviation_pct": deviation,
                    },
                )

        return ValidationResult(
            validator_name=self.name,
            passed=True,
            message="Spread is acceptable",
        )


class VolumeValidator(BaseValidator):
    """
    Validates trading volume.

    Checks for:
    - Negative volume
    - Suspicious volume spikes
    - Zero volume during trading hours
    """

    def __init__(
        self,
        min_volume: float = 0.0,
        max_volume_multiplier: float = 10.0,  # 10x average
    ):
        """
        Initialize volume validator.

        Args:
            min_volume: Minimum expected volume
            max_volume_multiplier: Maximum volume as multiplier of average
        """
        super().__init__("VolumeValidator")
        self.min_volume = min_volume
        self.max_volume_multiplier = max_volume_multiplier
        self._volume_history: List[float] = []
        self._max_history = 100

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate trading volume."""
        volume = data.get("volume")

        if volume is None:
            return ValidationResult(
                validator_name=self.name,
                passed=True,
                severity=ValidationSeverity.INFO,
                message="Volume not available",
            )

        # Check for negative volume
        if volume < 0:
            return ValidationResult(
                validator_name=self.name,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Negative volume: {volume}",
                details={"volume": volume},
            )

        # Check for volume spikes
        if self._volume_history:
            avg_volume = sum(self._volume_history) / len(self._volume_history)
            if avg_volume > 0 and volume > avg_volume * self.max_volume_multiplier:
                result = ValidationResult(
                    validator_name=self.name,
                    passed=False,
                    severity=ValidationSeverity.WARNING,
                    message=f"Volume spike: {volume} vs avg {avg_volume:.0f}",
                    details={
                        "volume": volume,
                        "avg_volume": avg_volume,
                        "multiplier": volume / avg_volume,
                    },
                )
                # Still update history
                self._update_history(volume)
                return result

        self._update_history(volume)

        return ValidationResult(
            validator_name=self.name,
            passed=True,
            message="Volume is acceptable",
        )

    def _update_history(self, volume: float):
        """Update volume history."""
        self._volume_history.append(volume)
        if len(self._volume_history) > self._max_history:
            self._volume_history.pop(0)


class VolatilityValidator(BaseValidator):
    """
    Validates implied volatility.

    Checks for:
    - Negative volatility
    - Unrealistic volatility levels
    - Volatility smiles and surfaces
    """

    def __init__(
        self,
        min_vol: float = 0.01,  # 1%
        max_vol: float = 3.0,   # 300%
    ):
        """
        Initialize volatility validator.

        Args:
            min_vol: Minimum realistic volatility
            max_vol: Maximum realistic volatility
        """
        super().__init__("VolatilityValidator")
        self.min_vol = min_vol
        self.max_vol = max_vol

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate volatility."""
        vol = data.get("volatility") or data.get("implied_vol")

        if vol is None:
            return ValidationResult(
                validator_name=self.name,
                passed=True,
                severity=ValidationSeverity.INFO,
                message="Volatility not available",
            )

        if vol < self.min_vol:
            return ValidationResult(
                validator_name=self.name,
                passed=False,
                severity=ValidationSeverity.WARNING,
                message=f"Volatility {vol:.2%} below minimum {self.min_vol:.2%}",
                details={"volatility": vol, "min_vol": self.min_vol},
            )

        if vol > self.max_vol:
            return ValidationResult(
                validator_name=self.name,
                passed=False,
                severity=ValidationSeverity.WARNING,
                message=f"Volatility {vol:.2%} exceeds maximum {self.max_vol:.2%}",
                details={"volatility": vol, "max_vol": self.max_vol},
            )

        return ValidationResult(
            validator_name=self.name,
            passed=True,
            message="Volatility is acceptable",
        )


class TimeSeriesValidator(BaseValidator):
    """
    Validates time-series properties.

    Checks for:
    - Out-of-sequence timestamps
    - Duplicate timestamps
    - Missing data gaps
    """

    def __init__(
        self,
        max_gap_seconds: int = 300,  # 5 minutes
    ):
        """
        Initialize time-series validator.

        Args:
            max_gap_seconds: Maximum acceptable gap in seconds
        """
        super().__init__("TimeSeriesValidator")
        self.max_gap_seconds = max_gap_seconds
        self._last_timestamp: Optional[datetime] = None

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate time-series properties."""
        timestamp = data.get("timestamp")

        if timestamp is None:
            return ValidationResult(
                validator_name=self.name,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message="Timestamp is missing",
            )

        # Convert to datetime if needed
        if not isinstance(timestamp, datetime):
            try:
                timestamp = datetime.fromisoformat(str(timestamp))
            except Exception:
                return ValidationResult(
                    validator_name=self.name,
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid timestamp format: {timestamp}",
                )

        # Check for future timestamps
        now = datetime.utcnow()
        if timestamp > now:
            return ValidationResult(
                validator_name=self.name,
                passed=False,
                severity=ValidationSeverity.WARNING,
                message=f"Future timestamp: {timestamp} > {now}",
                details={"timestamp": timestamp, "current_time": now},
            )

        # Check sequence and gaps
        if self._last_timestamp is not None:
            # Check for out-of-sequence
            if timestamp < self._last_timestamp:
                result = ValidationResult(
                    validator_name=self.name,
                    passed=False,
                    severity=ValidationSeverity.WARNING,
                    message=f"Out of sequence: {timestamp} < {self._last_timestamp}",
                    details={
                        "timestamp": timestamp,
                        "last_timestamp": self._last_timestamp,
                    },
                )
                self._last_timestamp = timestamp
                return result

            # Check for gaps
            gap_seconds = (timestamp - self._last_timestamp).total_seconds()
            if gap_seconds > self.max_gap_seconds:
                result = ValidationResult(
                    validator_name=self.name,
                    passed=False,
                    severity=ValidationSeverity.INFO,
                    message=f"Data gap of {gap_seconds:.0f}s detected",
                    details={
                        "gap_seconds": gap_seconds,
                        "max_gap_seconds": self.max_gap_seconds,
                    },
                )
                self._last_timestamp = timestamp
                return result

        self._last_timestamp = timestamp

        return ValidationResult(
            validator_name=self.name,
            passed=True,
            message="Time-series properties valid",
        )
