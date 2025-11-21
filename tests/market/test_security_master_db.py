"""Tests for database-backed security master."""

import pytest
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch
import json

from neutryx.market.storage.security_master import (
    SecurityIdentifierType,
    SecurityIdentifier,
    CorporateActionType,
    CorporateActionEvent,
)

try:
    from neutryx.market.storage.security_master_db import (
        SecurityMasterDatabase,
        SecurityMasterDBConfig,
    )
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not ASYNCPG_AVAILABLE, reason="asyncpg not available"
)


@pytest.fixture
def db_config():
    """Create test database configuration."""
    return SecurityMasterDBConfig(
        host="localhost",
        port=5432,
        database="test_security_master",
        user="test_user",
        password="test_password",
    )


@pytest.fixture
def sample_identifiers():
    """Create sample security identifiers."""
    return [
        SecurityIdentifier(
            id_type=SecurityIdentifierType.ISIN,
            value="US0378331005",
            primary=True
        ),
        SecurityIdentifier(
            id_type=SecurityIdentifierType.CUSIP,
            value="037833100",
            primary=False
        ),
        SecurityIdentifier(
            id_type=SecurityIdentifierType.TICKER,
            value="AAPL",
            primary=False
        ),
    ]


class TestSecurityMasterDatabaseConfig:
    """Tests for database configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SecurityMasterDBConfig()

        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "neutryx_security_master"
        assert config.pool_min_size == 5
        assert config.pool_max_size == 20

    def test_custom_config(self):
        """Test custom configuration."""
        config = SecurityMasterDBConfig(
            host="db.example.com",
            port=5433,
            database="custom_db",
            user="custom_user",
            password="custom_pass",
            pool_min_size=10,
            pool_max_size=50,
        )

        assert config.host == "db.example.com"
        assert config.port == 5433
        assert config.database == "custom_db"
        assert config.user == "custom_user"
        assert config.password == "custom_pass"
        assert config.pool_min_size == 10
        assert config.pool_max_size == 50


@pytest.mark.asyncio
class TestSecurityMasterDatabase:
    """Tests for database-backed security master."""

    async def test_initialization(self, db_config):
        """Test database initialization."""
        db = SecurityMasterDatabase(db_config)

        assert db.config == db_config
        assert db._pool is None

    async def test_connect(self, db_config):
        """Test database connection."""
        db = SecurityMasterDatabase(db_config)

        with patch('asyncpg.create_pool', new_callable=AsyncMock) as mock_pool:
            mock_pool.return_value = MagicMock()

            await db.connect()

            assert db._pool is not None
            mock_pool.assert_called_once()

    async def test_disconnect(self, db_config):
        """Test database disconnection."""
        db = SecurityMasterDatabase(db_config)

        mock_pool = AsyncMock()
        db._pool = mock_pool

        await db.disconnect()

        mock_pool.close.assert_called_once()
        assert db._pool is None

    async def test_not_connected_error(self, db_config):
        """Test error when operations attempted without connection."""
        db = SecurityMasterDatabase(db_config)

        with pytest.raises(RuntimeError, match="Not connected to database"):
            await db.get_security("TEST001")

    async def test_register_security(self, db_config, sample_identifiers):
        """Test security registration."""
        db = SecurityMasterDatabase(db_config)

        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_transaction = AsyncMock()

        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.transaction.return_value.__aenter__.return_value = mock_transaction
        mock_conn.fetchrow.return_value = {
            "security_id": "AAPL",
            "name": "Apple Inc.",
            "asset_class": "equity"
        }
        mock_conn.fetch.return_value = []

        db._pool = mock_pool

        # Mock get_security to return a complete record
        with patch.object(db, 'get_security', new_callable=AsyncMock) as mock_get:
            mock_security = MagicMock()
            mock_security.security_id = "AAPL"
            mock_get.return_value = mock_security

            result = await db.register_security(
                security_id="AAPL",
                name="Apple Inc.",
                asset_class="equity",
                identifiers=sample_identifiers,
                attributes={"exchange": "NASDAQ"},
            )

            assert result.security_id == "AAPL"
            mock_get.assert_called_once_with("AAPL")

    async def test_get_security(self, db_config):
        """Test retrieving security by ID."""
        db = SecurityMasterDatabase(db_config)

        mock_pool = AsyncMock()
        mock_conn = AsyncMock()

        # Mock main security record
        mock_conn.fetchrow.return_value = {
            "security_id": "AAPL",
            "name": "Apple Inc.",
            "asset_class": "equity"
        }

        # Mock identifiers
        mock_conn.fetch.side_effect = [
            [
                {
                    "identifier_type": "isin",
                    "identifier_value": "US0378331005",
                    "is_primary": True
                }
            ],
            [
                {
                    "version": 1,
                    "effective_date": date.today(),
                    "attributes": json.dumps({"exchange": "NASDAQ"})
                }
            ],
            []  # No corporate actions
        ]

        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        db._pool = mock_pool

        result = await db.get_security("AAPL")

        assert result is not None
        assert result.security_id == "AAPL"
        assert result.name == "Apple Inc."
        assert result.asset_class == "equity"
        assert len(result.identifiers) == 1
        assert SecurityIdentifierType.ISIN in result.identifiers

    async def test_get_security_not_found(self, db_config):
        """Test retrieving non-existent security."""
        db = SecurityMasterDatabase(db_config)

        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = None

        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        db._pool = mock_pool

        result = await db.get_security("NONEXISTENT")

        assert result is None

    async def test_get_security_by_identifier(self, db_config):
        """Test retrieving security by identifier."""
        db = SecurityMasterDatabase(db_config)

        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"security_id": "AAPL"}

        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        db._pool = mock_pool

        with patch.object(db, 'get_security', new_callable=AsyncMock) as mock_get:
            mock_security = MagicMock()
            mock_security.security_id = "AAPL"
            mock_get.return_value = mock_security

            result = await db.get_security_by_identifier(
                "US0378331005",
                SecurityIdentifierType.ISIN
            )

            assert result.security_id == "AAPL"

    async def test_update_security(self, db_config):
        """Test updating security attributes."""
        db = SecurityMasterDatabase(db_config)

        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_transaction = AsyncMock()

        # Mock latest version
        mock_conn.fetchrow.return_value = {
            "version": 1,
            "attributes": json.dumps({"exchange": "NASDAQ"})
        }

        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.transaction.return_value.__aenter__.return_value = mock_transaction
        db._pool = mock_pool

        result = await db.update_security(
            "AAPL",
            {"market_cap": 2500000000000},
            effective_date=date(2025, 1, 1)
        )

        assert result.version == 2
        assert result.effective_date == date(2025, 1, 1)
        assert "market_cap" in result.attributes
        assert result.attributes["market_cap"] == 2500000000000

    async def test_add_identifier(self, db_config):
        """Test adding identifier to security."""
        db = SecurityMasterDatabase(db_config)

        mock_pool = AsyncMock()
        mock_conn = AsyncMock()

        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        db._pool = mock_pool

        identifier = SecurityIdentifier(
            id_type=SecurityIdentifierType.FIGI,
            value="BBG000B9XRY4",
            primary=False
        )

        await db.add_identifier("AAPL", identifier)

        # Verify execute was called
        assert mock_conn.execute.called

    async def test_apply_corporate_action_dividend(self, db_config):
        """Test applying dividend corporate action."""
        db = SecurityMasterDatabase(db_config)

        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_transaction = AsyncMock()

        # Mock latest version with existing dividends
        mock_conn.fetchrow.return_value = {
            "attributes": json.dumps({"dividends": []})
        }

        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.transaction.return_value.__aenter__.return_value = mock_transaction
        db._pool = mock_pool

        with patch.object(db, 'update_security', new_callable=AsyncMock) as mock_update:
            event = CorporateActionEvent(
                action_type=CorporateActionType.DIVIDEND,
                effective_date=date(2025, 3, 15),
                description="Quarterly dividend",
                details={
                    "amount": 0.25,
                    "currency": "USD",
                    "pay_date": "2025-03-31"
                }
            )

            await db.apply_corporate_action("AAPL", event)

            # Verify corporate action was recorded
            assert mock_conn.execute.called

    async def test_apply_corporate_action_split(self, db_config):
        """Test applying stock split corporate action."""
        db = SecurityMasterDatabase(db_config)

        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_transaction = AsyncMock()

        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.transaction.return_value.__aenter__.return_value = mock_transaction
        db._pool = mock_pool

        with patch.object(db, 'update_security', new_callable=AsyncMock) as mock_update:
            event = CorporateActionEvent(
                action_type=CorporateActionType.SPLIT,
                effective_date=date(2025, 6, 1),
                description="2-for-1 stock split",
                details={"split_ratio": 2.0}
            )

            await db.apply_corporate_action("AAPL", event)

            # Verify update_security was called with split_ratio
            mock_update.assert_called_once()
            call_args = mock_update.call_args
            assert "split_ratio" in call_args[0][1]

    async def test_list_securities(self, db_config):
        """Test listing securities with filtering."""
        db = SecurityMasterDatabase(db_config)

        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {"security_id": "AAPL"},
            {"security_id": "MSFT"},
        ]

        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        db._pool = mock_pool

        with patch.object(db, 'get_security', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = [
                MagicMock(security_id="AAPL"),
                MagicMock(security_id="MSFT"),
            ]

            result = await db.list_securities(
                asset_class="equity",
                active_only=True,
                limit=10,
                offset=0
            )

            assert len(result) == 2
            assert result[0].security_id == "AAPL"
            assert result[1].security_id == "MSFT"

    async def test_search_securities(self, db_config):
        """Test searching securities by name."""
        db = SecurityMasterDatabase(db_config)

        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {"security_id": "AAPL"},
        ]

        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        db._pool = mock_pool

        with patch.object(db, 'get_security', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = MagicMock(
                security_id="AAPL",
                name="Apple Inc."
            )

            result = await db.search_securities(
                "Apple",
                search_fields=["name"]
            )

            assert len(result) == 1
            assert result[0].security_id == "AAPL"


class TestIntegration:
    """Integration tests for security master database."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, db_config, sample_identifiers):
        """Test complete security lifecycle."""
        db = SecurityMasterDatabase(db_config)

        # Mock the entire database
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_transaction = AsyncMock()

        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.transaction.return_value.__aenter__.return_value = mock_transaction
        db._pool = mock_pool

        # Mock responses for different operations
        security_record = {
            "security_id": "AAPL",
            "name": "Apple Inc.",
            "asset_class": "equity"
        }

        mock_conn.fetchrow.side_effect = [
            security_record,  # For get_security main record
            {"version": 1, "attributes": json.dumps({})},  # For update
            security_record,  # For get after update
        ]

        mock_conn.fetch.side_effect = [
            [],  # identifiers
            [],  # versions
            [],  # corporate actions
            [],  # identifiers after update
            [],  # versions after update
            [],  # corporate actions after update
        ]

        # 1. Register
        with patch.object(db, 'get_security', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = MagicMock(security_id="AAPL")

            security = await db.register_security(
                security_id="AAPL",
                name="Apple Inc.",
                asset_class="equity",
                identifiers=sample_identifiers,
            )

            assert security.security_id == "AAPL"

        # 2. Update
        version = await db.update_security(
            "AAPL",
            {"market_cap": 2500000000000}
        )

        assert version.version == 2

        # 3. Add identifier
        new_identifier = SecurityIdentifier(
            id_type=SecurityIdentifierType.FIGI,
            value="BBG000B9XRY4"
        )

        await db.add_identifier("AAPL", new_identifier)

        # Verify all operations completed
        assert mock_conn.execute.called
