"""Tests for market data storage backends."""

import pytest
from datetime import datetime
from neutryx.market.storage import StorageConfig, StorageType


def test_storage_config():
    """Test storage configuration."""
    config = StorageConfig(
        storage_type=StorageType.POSTGRESQL,
        host="localhost",
        port=5432,
        database="test_db"
    )

    assert config.storage_type == StorageType.POSTGRESQL
    assert config.host == "localhost"
    assert config.port == 5432
    assert config.database == "test_db"


def test_storage_config_defaults():
    """Test storage configuration defaults."""
    config = StorageConfig(storage_type=StorageType.MONGODB)

    assert config.connection_pool_size == 10
    assert config.connection_timeout == 30
    assert config.ssl_enabled is False


def test_storage_type_enum():
    """Test storage type enum."""
    assert StorageType.POSTGRESQL.value == "postgresql"
    assert StorageType.MONGODB.value == "mongodb"
    assert StorageType.TIMESCALEDB.value == "timescaledb"
