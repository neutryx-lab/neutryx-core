"""Database-backed Security Master with PostgreSQL persistence.

This module provides a centralized security master database with:
- Persistent storage in PostgreSQL
- ISIN/CUSIP/SEDOL cross-reference
- Real-time reference data updates
- Corporate actions processing
- Version history tracking
- High-performance querying
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Dict, List, Optional, Any, Tuple
import json
import logging

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

from .security_master import (
    SecurityIdentifierType,
    SecurityIdentifier,
    CorporateActionType,
    CorporateActionEvent,
    SecurityVersion,
    SecurityRecord,
)

logger = logging.getLogger(__name__)


@dataclass
class SecurityMasterDBConfig:
    """Configuration for database-backed security master."""

    host: str = "localhost"
    port: int = 5432
    database: str = "neutryx_security_master"
    user: str = "neutryx"
    password: str = ""
    pool_min_size: int = 5
    pool_max_size: int = 20
    command_timeout: float = 60.0


class SecurityMasterDatabase:
    """
    Database-backed security master service with PostgreSQL persistence.

    Features:
    ---------
    - Persistent storage of security reference data
    - Multi-identifier cross-reference (ISIN, CUSIP, SEDOL, FIGI, Ticker)
    - Version history with temporal queries
    - Corporate actions tracking
    - Real-time updates
    - High-performance indexing
    - ACID transactions

    Schema:
    -------
    - securities: Main security records
    - security_identifiers: Cross-reference table
    - security_versions: Attribute history
    - corporate_actions: Event log
    """

    def __init__(self, config: SecurityMasterDBConfig):
        if not ASYNCPG_AVAILABLE:
            raise ImportError(
                "asyncpg is required for SecurityMasterDatabase. "
                "Install with: pip install asyncpg"
            )

        self.config = config
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        """Establish database connection pool."""
        if self._pool is not None:
            logger.warning("Connection pool already exists")
            return

        try:
            self._pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                min_size=self.config.pool_min_size,
                max_size=self.config.pool_max_size,
                command_timeout=self.config.command_timeout,
            )
            logger.info(
                f"Connected to security master database at {self.config.host}:{self.config.port}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    async def disconnect(self) -> None:
        """Close database connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("Disconnected from security master database")

    async def initialize_schema(self) -> None:
        """Create database schema if it doesn't exist."""
        if self._pool is None:
            raise RuntimeError("Not connected to database")

        schema_sql = """
        -- Main securities table
        CREATE TABLE IF NOT EXISTS securities (
            security_id VARCHAR(50) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            asset_class VARCHAR(50) NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            active BOOLEAN NOT NULL DEFAULT TRUE
        );

        -- Security identifiers cross-reference
        CREATE TABLE IF NOT EXISTS security_identifiers (
            id SERIAL PRIMARY KEY,
            security_id VARCHAR(50) NOT NULL REFERENCES securities(security_id) ON DELETE CASCADE,
            identifier_type VARCHAR(20) NOT NULL,
            identifier_value VARCHAR(50) NOT NULL,
            is_primary BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(identifier_type, identifier_value)
        );

        -- Security version history
        CREATE TABLE IF NOT EXISTS security_versions (
            id SERIAL PRIMARY KEY,
            security_id VARCHAR(50) NOT NULL REFERENCES securities(security_id) ON DELETE CASCADE,
            version INTEGER NOT NULL,
            effective_date DATE NOT NULL,
            attributes JSONB NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(security_id, version)
        );

        -- Corporate actions log
        CREATE TABLE IF NOT EXISTS corporate_actions (
            id SERIAL PRIMARY KEY,
            security_id VARCHAR(50) NOT NULL REFERENCES securities(security_id) ON DELETE CASCADE,
            action_type VARCHAR(50) NOT NULL,
            effective_date DATE NOT NULL,
            description TEXT NOT NULL,
            details JSONB,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        -- Indexes for performance
        CREATE INDEX IF NOT EXISTS idx_security_identifiers_security_id
            ON security_identifiers(security_id);
        CREATE INDEX IF NOT EXISTS idx_security_identifiers_type_value
            ON security_identifiers(identifier_type, identifier_value);
        CREATE INDEX IF NOT EXISTS idx_security_versions_security_id
            ON security_versions(security_id);
        CREATE INDEX IF NOT EXISTS idx_security_versions_effective_date
            ON security_versions(effective_date);
        CREATE INDEX IF NOT EXISTS idx_corporate_actions_security_id
            ON corporate_actions(security_id);
        CREATE INDEX IF NOT EXISTS idx_corporate_actions_effective_date
            ON corporate_actions(effective_date);
        """

        async with self._pool.acquire() as conn:
            await conn.execute(schema_sql)
            logger.info("Database schema initialized")

    async def register_security(
        self,
        security_id: str,
        name: str,
        asset_class: str,
        identifiers: List[SecurityIdentifier],
        attributes: Optional[Dict[str, Any]] = None,
        effective_date: Optional[date] = None,
    ) -> SecurityRecord:
        """
        Register a new security in the database.

        Parameters:
        -----------
        security_id : str
            Unique security identifier
        name : str
            Security name
        asset_class : str
            Asset class (equity, bond, derivative, etc.)
        identifiers : List[SecurityIdentifier]
            List of security identifiers (ISIN, CUSIP, etc.)
        attributes : Dict[str, Any], optional
            Additional security attributes
        effective_date : date, optional
            Effective date for initial version (defaults to today)

        Returns:
        --------
        SecurityRecord
            The created security record
        """
        if self._pool is None:
            raise RuntimeError("Not connected to database")

        effective_date = effective_date or date.today()
        attributes = attributes or {}

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Insert main security record
                await conn.execute(
                    """
                    INSERT INTO securities (security_id, name, asset_class)
                    VALUES ($1, $2, $3)
                    """,
                    security_id,
                    name,
                    asset_class,
                )

                # Insert identifiers
                for identifier in identifiers:
                    await conn.execute(
                        """
                        INSERT INTO security_identifiers
                        (security_id, identifier_type, identifier_value, is_primary)
                        VALUES ($1, $2, $3, $4)
                        """,
                        security_id,
                        identifier.id_type.value,
                        identifier.value,
                        identifier.primary,
                    )

                # Insert initial version
                await conn.execute(
                    """
                    INSERT INTO security_versions
                    (security_id, version, effective_date, attributes)
                    VALUES ($1, $2, $3, $4)
                    """,
                    security_id,
                    1,
                    effective_date,
                    json.dumps(attributes),
                )

        logger.info(f"Registered security: {security_id}")

        # Return constructed record
        return await self.get_security(security_id)

    async def get_security(self, security_id: str) -> Optional[SecurityRecord]:
        """Retrieve security by ID."""
        if self._pool is None:
            raise RuntimeError("Not connected to database")

        async with self._pool.acquire() as conn:
            # Get main record
            row = await conn.fetchrow(
                """
                SELECT security_id, name, asset_class
                FROM securities
                WHERE security_id = $1
                """,
                security_id,
            )

            if row is None:
                return None

            # Get identifiers
            identifier_rows = await conn.fetch(
                """
                SELECT identifier_type, identifier_value, is_primary
                FROM security_identifiers
                WHERE security_id = $1
                """,
                security_id,
            )

            identifiers = {}
            for id_row in identifier_rows:
                id_type = SecurityIdentifierType(id_row["identifier_type"])
                identifier = SecurityIdentifier(
                    id_type=id_type,
                    value=id_row["identifier_value"],
                    primary=id_row["is_primary"],
                )
                identifiers[id_type] = identifier

            # Get versions
            version_rows = await conn.fetch(
                """
                SELECT version, effective_date, attributes
                FROM security_versions
                WHERE security_id = $1
                ORDER BY effective_date, version
                """,
                security_id,
            )

            versions = [
                SecurityVersion(
                    version=v["version"],
                    effective_date=v["effective_date"],
                    attributes=json.loads(v["attributes"]),
                )
                for v in version_rows
            ]

            # Get corporate actions
            action_rows = await conn.fetch(
                """
                SELECT action_type, effective_date, description, details
                FROM corporate_actions
                WHERE security_id = $1
                ORDER BY effective_date
                """,
                security_id,
            )

            corporate_actions = [
                CorporateActionEvent(
                    action_type=CorporateActionType(a["action_type"]),
                    effective_date=a["effective_date"],
                    description=a["description"],
                    details=json.loads(a["details"]) if a["details"] else {},
                )
                for a in action_rows
            ]

            return SecurityRecord(
                security_id=row["security_id"],
                name=row["name"],
                asset_class=row["asset_class"],
                identifiers=identifiers,
                versions=versions,
                corporate_actions=corporate_actions,
            )

    async def get_security_by_identifier(
        self, identifier_value: str, identifier_type: SecurityIdentifierType
    ) -> Optional[SecurityRecord]:
        """Retrieve security by any identifier (ISIN, CUSIP, SEDOL, etc.)."""
        if self._pool is None:
            raise RuntimeError("Not connected to database")

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT security_id
                FROM security_identifiers
                WHERE identifier_type = $1 AND UPPER(identifier_value) = UPPER($2)
                """,
                identifier_type.value,
                identifier_value,
            )

            if row is None:
                return None

            return await self.get_security(row["security_id"])

    async def update_security(
        self,
        security_id: str,
        updates: Dict[str, Any],
        effective_date: Optional[date] = None,
    ) -> SecurityVersion:
        """Update security attributes with new version."""
        if self._pool is None:
            raise RuntimeError("Not connected to database")

        effective_date = effective_date or date.today()

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Get latest version
                latest_row = await conn.fetchrow(
                    """
                    SELECT version, attributes
                    FROM security_versions
                    WHERE security_id = $1
                    ORDER BY version DESC
                    LIMIT 1
                    """,
                    security_id,
                )

                if latest_row is None:
                    raise ValueError(f"Security {security_id} not found")

                # Merge attributes
                current_attrs = json.loads(latest_row["attributes"])
                current_attrs.update(updates)
                new_version = latest_row["version"] + 1

                # Insert new version
                await conn.execute(
                    """
                    INSERT INTO security_versions
                    (security_id, version, effective_date, attributes)
                    VALUES ($1, $2, $3, $4)
                    """,
                    security_id,
                    new_version,
                    effective_date,
                    json.dumps(current_attrs),
                )

                # Update securities table timestamp
                await conn.execute(
                    """
                    UPDATE securities
                    SET updated_at = CURRENT_TIMESTAMP
                    WHERE security_id = $1
                    """,
                    security_id,
                )

        logger.info(f"Updated security {security_id} to version {new_version}")

        return SecurityVersion(
            version=new_version,
            effective_date=effective_date,
            attributes=current_attrs,
        )

    async def add_identifier(
        self, security_id: str, identifier: SecurityIdentifier
    ) -> None:
        """Add a new identifier to an existing security."""
        if self._pool is None:
            raise RuntimeError("Not connected to database")

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO security_identifiers
                (security_id, identifier_type, identifier_value, is_primary)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (identifier_type, identifier_value) DO NOTHING
                """,
                security_id,
                identifier.id_type.value,
                identifier.value,
                identifier.primary,
            )

        logger.info(
            f"Added identifier {identifier.id_type.value}:{identifier.value} to {security_id}"
        )

    async def apply_corporate_action(
        self, security_id: str, event: CorporateActionEvent
    ) -> None:
        """Record and apply corporate action."""
        if self._pool is None:
            raise RuntimeError("Not connected to database")

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Record action
                await conn.execute(
                    """
                    INSERT INTO corporate_actions
                    (security_id, action_type, effective_date, description, details)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    security_id,
                    event.action_type.value,
                    event.effective_date,
                    event.description,
                    json.dumps(event.details),
                )

                # Apply updates based on action type
                updates: Dict[str, Any] = {}

                if event.action_type == CorporateActionType.SYMBOL_CHANGE:
                    new_ticker = event.details.get("new_ticker")
                    if new_ticker:
                        updates["ticker"] = new_ticker
                        # Update identifier
                        await conn.execute(
                            """
                            UPDATE security_identifiers
                            SET identifier_value = $1
                            WHERE security_id = $2 AND identifier_type = 'ticker'
                            """,
                            new_ticker,
                            security_id,
                        )

                elif event.action_type == CorporateActionType.SPLIT:
                    if "split_ratio" in event.details:
                        updates["split_ratio"] = event.details["split_ratio"]

                elif event.action_type == CorporateActionType.DIVIDEND:
                    # Get current dividends
                    latest = await conn.fetchrow(
                        """
                        SELECT attributes FROM security_versions
                        WHERE security_id = $1
                        ORDER BY version DESC LIMIT 1
                        """,
                        security_id,
                    )
                    if latest:
                        attrs = json.loads(latest["attributes"])
                        dividends = attrs.get("dividends", [])
                        dividends.append(
                            {
                                "amount": event.details.get("amount"),
                                "currency": event.details.get("currency"),
                                "pay_date": str(event.details.get("pay_date")),
                            }
                        )
                        updates["dividends"] = dividends

                if updates:
                    await self.update_security(security_id, updates, event.effective_date)

        logger.info(f"Applied corporate action {event.action_type.value} to {security_id}")

    async def list_securities(
        self,
        asset_class: Optional[str] = None,
        active_only: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> List[SecurityRecord]:
        """List securities with optional filtering."""
        if self._pool is None:
            raise RuntimeError("Not connected to database")

        query = "SELECT security_id FROM securities WHERE 1=1"
        params = []
        param_idx = 1

        if asset_class:
            query += f" AND asset_class = ${param_idx}"
            params.append(asset_class)
            param_idx += 1

        if active_only:
            query += f" AND active = ${param_idx}"
            params.append(True)
            param_idx += 1

        query += f" LIMIT ${param_idx} OFFSET ${param_idx + 1}"
        params.extend([limit, offset])

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        securities = []
        for row in rows:
            security = await self.get_security(row["security_id"])
            if security:
                securities.append(security)

        return securities

    async def search_securities(
        self,
        search_term: str,
        search_fields: List[str] = ["name", "security_id"],
    ) -> List[SecurityRecord]:
        """Search securities by name or identifier."""
        if self._pool is None:
            raise RuntimeError("Not connected to database")

        # Build search query
        conditions = []
        params = []

        for i, field in enumerate(search_fields, 1):
            if field in ["name", "security_id"]:
                conditions.append(f"UPPER({field}) LIKE ${i}")
                params.append(f"%{search_term.upper()}%")

        query = f"""
            SELECT DISTINCT security_id
            FROM securities
            WHERE {' OR '.join(conditions)}
            LIMIT 50
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        securities = []
        for row in rows:
            security = await self.get_security(row["security_id"])
            if security:
                securities.append(security)

        return securities
