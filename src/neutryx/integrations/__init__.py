"""External library integrations and adapters.

This module provides integrations with external libraries and data formats:

- **FpML**: Financial Products Markup Language (ISO 20022) support
- **Pandas**: Integration with pandas DataFrames
- **QuantLib**: Bindings to QuantLib C++ library
- **FFI**: Foreign Function Interfaces for performance-critical code
- **Eigen**: Linear algebra operations via Eigen C++ library

All integrations are designed to be optional and can be used independently.
"""

from __future__ import annotations

from .databases import (
    DatabaseConfig,
    DatabaseConnector,
    DatabaseConnectorError,
    InMemoryConnector,
    MongoConnector,
    PostgresConnector,
    TimescaleConnector,
)

__all__ = [
    "DatabaseConfig",
    "DatabaseConnector",
    "DatabaseConnectorError",
    "PostgresConnector",
    "TimescaleConnector",
    "MongoConnector",
    "InMemoryConnector",
]
