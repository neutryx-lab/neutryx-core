"""Base schema classes with versioning and metadata support.

This module provides the foundational classes for all Neutryx schemas,
enabling Schema-Driven Development with built-in versioning, validation,
and JSON Schema generation capabilities.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


class NeutryxSchema(BaseModel):
    """Base class for all Neutryx schemas with versioning support.

    All domain models should inherit from this class to ensure consistent
    validation, serialization, and schema generation behavior.

    Class Attributes:
        __schema_version__: Semantic version of this schema definition
        __schema_domain__: Domain category (e.g., 'trading', 'clearing', 'market')
        __schema_deprecated__: Whether this schema is deprecated
        __schema_deprecated_in__: Version when deprecation was introduced
        __schema_removed_in__: Version when this schema will be removed

    Example:
        >>> class MyModel(NeutryxSchema):
        ...     __schema_version__ = "1.0.0"
        ...     __schema_domain__ = "trading"
        ...     name: str
        ...     value: float
    """

    model_config = ConfigDict(
        # Strict validation - disallow extra fields
        extra="forbid",
        # Validate on assignment, not just initialization
        validate_assignment=True,
        # Use enum values for serialization
        use_enum_values=False,
        # JSON Schema configuration
        json_schema_extra={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
        },
        # Serialization options
        ser_json_timedelta="float",
        ser_json_bytes="base64",
        # Allow population by field name
        populate_by_name=True,
    )

    # Schema version metadata (class-level)
    __schema_version__: ClassVar[str] = "1.0.0"
    __schema_domain__: ClassVar[str] = "core"
    __schema_deprecated__: ClassVar[bool] = False
    __schema_deprecated_in__: ClassVar[Optional[str]] = None
    __schema_removed_in__: ClassVar[Optional[str]] = None

    @classmethod
    def schema_info(cls) -> Dict[str, Any]:
        """Return schema metadata for registry and documentation.

        Returns:
            Dictionary containing schema name, version, domain, and deprecation info.
        """
        return {
            "name": cls.__name__,
            "version": cls.__schema_version__,
            "domain": cls.__schema_domain__,
            "deprecated": cls.__schema_deprecated__,
            "deprecated_in": cls.__schema_deprecated_in__,
            "removed_in": cls.__schema_removed_in__,
            "module": cls.__module__,
        }

    @classmethod
    def json_schema(cls, mode: str = "serialization") -> Dict[str, Any]:
        """Generate JSON Schema with Neutryx extensions.

        Args:
            mode: Either 'validation' or 'serialization'

        Returns:
            JSON Schema dictionary with custom metadata.
        """
        schema = cls.model_json_schema(mode=mode)
        schema["$id"] = f"https://schemas.neutryx.tech/{cls.__name__}.json"
        schema["x-schema-version"] = cls.__schema_version__
        schema["x-schema-domain"] = cls.__schema_domain__
        return schema


class VersionedSchema(NeutryxSchema):
    """Schema with instance-level version tracking for data migration.

    Use this base class when you need to track which schema version was used
    to create a specific instance, enabling future data migrations.
    """

    # Use PrivateAttr for private fields in Pydantic v2
    _instance_schema_version: Optional[str] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Set schema version after initialization."""
        if self._instance_schema_version is None:
            self._instance_schema_version = self.__schema_version__

    @property
    def instance_schema_version(self) -> str:
        """Get the schema version used to create this instance."""
        return self._instance_schema_version or self.__schema_version__


class AuditedSchema(VersionedSchema):
    """Schema with audit trail fields for compliance tracking.

    Use this base class for entities that require audit logging,
    such as trades, confirmations, and regulatory reports.
    """

    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when this record was created",
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last update",
    )
    created_by: Optional[str] = Field(
        default=None,
        description="User or system that created this record",
    )
    updated_by: Optional[str] = Field(
        default=None,
        description="User or system that last updated this record",
    )


class ImmutableSchema(NeutryxSchema):
    """Immutable schema - instances cannot be modified after creation.

    Use this for value objects and reference data that should not change.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
    )
