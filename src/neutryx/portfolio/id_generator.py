"""Trade and entity ID generation system.

Provides systematic ID generation for trades, counterparties, and other entities.
"""
from __future__ import annotations

import re
import uuid
from collections import defaultdict
from datetime import date
from enum import Enum
from threading import Lock
from typing import Dict, Optional, Pattern, Set

from pydantic import BaseModel, Field


class IDPattern(Enum):
    """Predefined ID generation patterns."""

    SEQUENTIAL = "sequential"
    DATE_SEQUENTIAL = "date_sequential"
    UUID = "uuid"
    CUSTOM = "custom"


class IDGeneratorConfig(BaseModel):
    """Configuration for ID generator."""

    pattern: IDPattern = Field(default=IDPattern.DATE_SEQUENTIAL)
    prefix: str = Field(default="TRD")
    separator: str = Field(default="-")
    sequence_width: int = Field(default=4, ge=1)
    date_format: str = Field(default="%Y%m%d")
    custom_pattern: Optional[str] = Field(default=None)
    validate_uniqueness: bool = Field(default=True)


class TradeIDGenerator:
    """Generates unique trade identifiers with configurable patterns."""

    def __init__(self, config: Optional[IDGeneratorConfig] = None):
        self.config = config or IDGeneratorConfig()
        self._counters: Dict[str, int] = defaultdict(int)
        self._generated_ids: Set[str] = set()
        self._lock = Lock()

    def generate(self, reference_date: Optional[date] = None) -> str:
        """Generate a new trade ID."""
        with self._lock:
            if self.config.pattern == IDPattern.SEQUENTIAL:
                trade_id = self._generate_sequential()
            elif self.config.pattern == IDPattern.DATE_SEQUENTIAL:
                trade_id = self._generate_date_sequential(reference_date)
            elif self.config.pattern == IDPattern.UUID:
                trade_id = self._generate_uuid()
            elif self.config.pattern == IDPattern.CUSTOM:
                trade_id = self._generate_custom(reference_date)
            else:
                raise ValueError(f"Unknown pattern: {self.config.pattern}")

            if self.config.validate_uniqueness:
                self._generated_ids.add(trade_id)

            return trade_id

    def _generate_sequential(self) -> str:
        counter_key = "global"
        self._counters[counter_key] += 1
        seq = str(self._counters[counter_key]).zfill(self.config.sequence_width)
        return f"{self.config.prefix}{self.config.separator}{seq}"

    def _generate_date_sequential(self, reference_date: Optional[date] = None) -> str:
        ref_date = reference_date or date.today()
        date_str = ref_date.strftime(self.config.date_format)
        counter_key = f"date_{date_str}"
        self._counters[counter_key] += 1
        seq = str(self._counters[counter_key]).zfill(self.config.sequence_width)
        return f"{self.config.prefix}{self.config.separator}{date_str}{self.config.separator}{seq}"

    def _generate_uuid(self) -> str:
        uuid_str = str(uuid.uuid4())
        return f"{self.config.prefix}{self.config.separator}{uuid_str}"

    def _generate_custom(self, reference_date: Optional[date] = None) -> str:
        if not self.config.custom_pattern:
            raise ValueError("Custom pattern not configured")

        ref_date = reference_date or date.today()
        date_str = ref_date.strftime(self.config.date_format)

        counter_key = f"custom_{date_str}" if "{date}" in self.config.custom_pattern else "custom_global"
        self._counters[counter_key] += 1
        seq = self._counters[counter_key]

        uuid_str = str(uuid.uuid4()) if "{uuid}" in self.config.custom_pattern else ""

        try:
            trade_id = self.config.custom_pattern.format(
                prefix=self.config.prefix, date=date_str, seq=seq, uuid=uuid_str
            )
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid custom pattern: {e}") from e

        return trade_id

    def is_valid(self, trade_id: str) -> bool:
        """Check if an ID matches the configured pattern."""
        if not trade_id:
            return False

        if self.config.validate_uniqueness and trade_id in self._generated_ids:
            return True

        pattern = self._get_validation_pattern()
        return bool(pattern.match(trade_id))

    def _get_validation_pattern(self) -> Pattern:
        sep = re.escape(self.config.separator)
        prefix = re.escape(self.config.prefix)

        if self.config.pattern == IDPattern.SEQUENTIAL:
            return re.compile(rf"^{prefix}{sep}\d{{{self.config.sequence_width}}}$")
        elif self.config.pattern == IDPattern.DATE_SEQUENTIAL:
            return re.compile(rf"^{prefix}{sep}\d{{8}}{sep}\d{{{self.config.sequence_width}}}$")
        elif self.config.pattern == IDPattern.UUID:
            uuid_pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
            return re.compile(rf"^{prefix}{sep}{uuid_pattern}$")
        else:
            return re.compile(rf"^{prefix}.*$")

    def is_unique(self, trade_id: str) -> bool:
        """Check if an ID is unique."""
        return trade_id not in self._generated_ids

    def reset_counters(self, pattern: Optional[str] = None) -> None:
        """Reset sequence counters."""
        with self._lock:
            if pattern:
                keys_to_reset = [k for k in self._counters.keys() if pattern in k]
                for key in keys_to_reset:
                    self._counters[key] = 0
            else:
                self._counters.clear()

    def get_counter(self, counter_key: Optional[str] = None) -> int:
        """Get current counter value."""
        key = counter_key or "global"
        return self._counters.get(key, 0)


class EntityIDGenerator:
    """Generator for entity IDs (counterparties, books, desks, etc.)."""

    def __init__(self, config: Optional[IDGeneratorConfig] = None):
        self._generator = TradeIDGenerator(config)

    def generate(self, reference_date: Optional[date] = None) -> str:
        """Generate a new entity ID."""
        return self._generator.generate(reference_date)

    def is_valid(self, entity_id: str) -> bool:
        """Validate entity ID format."""
        return self._generator.is_valid(entity_id)

    def is_unique(self, entity_id: str) -> bool:
        """Check if entity ID is unique."""
        return self._generator.is_unique(entity_id)


def create_trade_id_generator(prefix: str = "TRD", pattern: IDPattern = IDPattern.DATE_SEQUENTIAL) -> TradeIDGenerator:
    """Create a trade ID generator."""
    config = IDGeneratorConfig(prefix=prefix, pattern=pattern)
    return TradeIDGenerator(config)


def create_counterparty_id_generator() -> EntityIDGenerator:
    """Create a counterparty ID generator."""
    config = IDGeneratorConfig(prefix="CP", pattern=IDPattern.SEQUENTIAL, sequence_width=4)
    return EntityIDGenerator(config)


def create_book_id_generator() -> EntityIDGenerator:
    """Create a book ID generator."""
    config = IDGeneratorConfig(prefix="BK", pattern=IDPattern.DATE_SEQUENTIAL, sequence_width=6)
    return EntityIDGenerator(config)


def create_desk_id_generator() -> EntityIDGenerator:
    """Create a desk ID generator."""
    config = IDGeneratorConfig(prefix="DSK", pattern=IDPattern.SEQUENTIAL, sequence_width=3)
    return EntityIDGenerator(config)


__all__ = [
    "IDPattern",
    "IDGeneratorConfig",
    "TradeIDGenerator",
    "EntityIDGenerator",
    "create_trade_id_generator",
    "create_counterparty_id_generator",
    "create_book_id_generator",
    "create_desk_id_generator",
]
