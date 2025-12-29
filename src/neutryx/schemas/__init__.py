"""Neutryx Schema Registry - Single Source of Truth for all data models.

This package provides centralized Pydantic schemas for the entire Neutryx platform,
enabling Schema-Driven Development (SDD) with:
- Type safety through Pydantic validation
- Automatic JSON Schema generation
- OpenAPI specification generation
- TypeScript type generation
- Schema versioning and migration

Usage:
    from neutryx.schemas import Trade, Party, TradeStatus, ProductType
    from neutryx.schemas.core import NeutryxSchema
    from neutryx.schemas.trading import RFQ, Quote
"""

from neutryx.schemas.core.base import NeutryxSchema, VersionedSchema, AuditedSchema
from neutryx.schemas.core.enums import (
    TradeStatus,
    ProductType,
    SettlementType,
    AssetClass,
    DataQuality,
    CreditRating,
    EntityType,
    MessageType,
    DayCountConvention,
    BusinessDayConvention,
    CurrencyCode,
)
from neutryx.schemas.core.identifiers import Party, Money

__all__ = [
    # Base classes
    "NeutryxSchema",
    "VersionedSchema",
    "AuditedSchema",
    # Core enums
    "TradeStatus",
    "ProductType",
    "SettlementType",
    "AssetClass",
    "DataQuality",
    "CreditRating",
    "EntityType",
    "MessageType",
    "DayCountConvention",
    "BusinessDayConvention",
    "CurrencyCode",
    # Core models
    "Party",
    "Money",
]

__version__ = "1.0.0"
