"""Client repository for managing counterparties and CSA agreements.

This module provides repository implementations for:
- Counterparty (client) management
- CSA (Credit Support Annex) agreement storage
- Master agreement relationships
- Query capabilities for client-related data
"""
from __future__ import annotations

import asyncio
from datetime import date
from typing import Any, Dict, List, Optional

from neutryx.portfolio.contracts.counterparty import Counterparty, CounterpartyCredit, CreditRating, EntityType
from neutryx.portfolio.contracts.csa import (
    CSA,
    CollateralTerms,
    ThresholdTerms,
    EligibleCollateral,
    CollateralType,
    ValuationFrequency,
    DisputeResolution,
)
from neutryx.portfolio.repository import CounterpartyRepository
from neutryx.integrations.databases.base import DatabaseConfig


class CSARepository:
    """Abstract repository for CSA agreement persistence."""

    def save(self, csa: CSA) -> None:
        """Save a CSA agreement."""
        raise NotImplementedError

    def find_by_id(self, csa_id: str) -> Optional[CSA]:
        """Find a CSA by ID."""
        raise NotImplementedError

    def find_by_counterparty(self, counterparty_id: str) -> List[CSA]:
        """Find CSAs involving a specific counterparty."""
        raise NotImplementedError

    def find_by_parties(self, party_a_id: str, party_b_id: str) -> Optional[CSA]:
        """Find CSA between two specific parties."""
        raise NotImplementedError

    def find_all(self) -> List[CSA]:
        """Find all CSA agreements."""
        raise NotImplementedError

    def delete(self, csa_id: str) -> bool:
        """Delete a CSA agreement."""
        raise NotImplementedError


class PostgresCounterpartyRepository(CounterpartyRepository):
    """PostgreSQL-based counterparty repository.

    Provides persistent storage of counterparty information with:
    - Full credit profile storage
    - LEI indexing for fast lookups
    - Parent-child entity relationships
    - Support for bank and clearinghouse flags

    Attributes
    ----------
    config : DatabaseConfig
        Database connection configuration
    schema : str
        PostgreSQL schema name (default: "clients")
    """

    def __init__(
        self,
        config: DatabaseConfig,
        schema: str = "clients",
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        """Initialize PostgreSQL counterparty repository.

        Parameters
        ----------
        config : DatabaseConfig
            Database connection configuration
        schema : str
            PostgreSQL schema name
        loop : asyncio.AbstractEventLoop, optional
            Event loop for async operations
        """
        self.config = config
        self.schema = schema
        self._pool = None
        self._loop = loop
        self._connected = False

    async def connect(self) -> bool:
        """Establish database connection and ensure schema exists."""
        if self._connected:
            return True

        try:
            import asyncpg
        except ImportError as exc:
            raise RuntimeError(
                "asyncpg is required for PostgreSQL repository. "
                "Install with `pip install asyncpg`."
            ) from exc

        dsn = self._build_dsn()
        self._pool = await asyncpg.create_pool(
            dsn=dsn,
            timeout=self.config.connect_timeout,
            **self.config.options,
        )

        await self._ensure_schema()
        self._connected = True
        return True

    async def disconnect(self) -> None:
        """Close database connection."""
        if self._pool:
            await self._pool.close()
            self._pool = None
        self._connected = False

    async def _ensure_schema(self) -> None:
        """Create schema and tables if they don't exist."""
        async with self._pool.acquire() as conn:
            # Create schema
            await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")

            # Create counterparties table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.counterparties (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    lei TEXT UNIQUE,
                    jurisdiction TEXT,
                    parent_id TEXT,
                    is_bank BOOLEAN DEFAULT FALSE,
                    is_clearinghouse BOOLEAN DEFAULT FALSE,
                    -- Credit attributes
                    credit_rating TEXT,
                    internal_rating TEXT,
                    lgd NUMERIC,
                    recovery_rate NUMERIC,
                    credit_spread_bps NUMERIC,
                    -- Metadata
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)

            # Create indexes
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_counterparties_lei
                ON {self.schema}.counterparties (lei)
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_counterparties_entity_type
                ON {self.schema}.counterparties (entity_type)
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_counterparties_parent
                ON {self.schema}.counterparties (parent_id)
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_counterparties_is_bank
                ON {self.schema}.counterparties (is_bank)
            """)

    def save(self, counterparty: Counterparty) -> None:
        """Save a counterparty (synchronous wrapper)."""
        loop = self._loop or asyncio.get_event_loop()
        loop.run_until_complete(self.save_async(counterparty))

    async def save_async(self, counterparty: Counterparty) -> None:
        """Save a counterparty asynchronously."""
        await self.connect()

        async with self._pool.acquire() as conn:
            credit = counterparty.credit
            await conn.execute(
                f"""
                INSERT INTO {self.schema}.counterparties (
                    id, name, entity_type, lei, jurisdiction, parent_id,
                    is_bank, is_clearinghouse, credit_rating, internal_rating,
                    lgd, recovery_rate, credit_spread_bps
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13
                )
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    entity_type = EXCLUDED.entity_type,
                    lei = EXCLUDED.lei,
                    jurisdiction = EXCLUDED.jurisdiction,
                    parent_id = EXCLUDED.parent_id,
                    is_bank = EXCLUDED.is_bank,
                    is_clearinghouse = EXCLUDED.is_clearinghouse,
                    credit_rating = EXCLUDED.credit_rating,
                    internal_rating = EXCLUDED.internal_rating,
                    lgd = EXCLUDED.lgd,
                    recovery_rate = EXCLUDED.recovery_rate,
                    credit_spread_bps = EXCLUDED.credit_spread_bps,
                    updated_at = NOW()
                """,
                counterparty.id,
                counterparty.name,
                counterparty.entity_type.value,
                counterparty.lei,
                counterparty.jurisdiction,
                counterparty.parent_id,
                counterparty.is_bank,
                counterparty.is_clearinghouse,
                credit.rating.value if credit and credit.rating else None,
                credit.internal_rating if credit else None,
                credit.lgd if credit else None,
                credit.recovery_rate if credit else None,
                credit.credit_spread_bps if credit else None,
            )

    def find_by_id(self, counterparty_id: str) -> Optional[Counterparty]:
        """Find a counterparty by ID (synchronous wrapper)."""
        loop = self._loop or asyncio.get_event_loop()
        return loop.run_until_complete(self.find_by_id_async(counterparty_id))

    async def find_by_id_async(self, counterparty_id: str) -> Optional[Counterparty]:
        """Find a counterparty by ID asynchronously."""
        await self.connect()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM {self.schema}.counterparties WHERE id = $1",
                counterparty_id,
            )

            if row:
                return self._row_to_counterparty(row)
            return None

    def find_by_lei(self, lei: str) -> Optional[Counterparty]:
        """Find a counterparty by LEI (synchronous wrapper)."""
        loop = self._loop or asyncio.get_event_loop()
        return loop.run_until_complete(self.find_by_lei_async(lei))

    async def find_by_lei_async(self, lei: str) -> Optional[Counterparty]:
        """Find a counterparty by LEI asynchronously."""
        await self.connect()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM {self.schema}.counterparties WHERE lei = $1",
                lei,
            )

            if row:
                return self._row_to_counterparty(row)
            return None

    def find_all(self) -> List[Counterparty]:
        """Find all counterparties (synchronous wrapper)."""
        loop = self._loop or asyncio.get_event_loop()
        return loop.run_until_complete(self.find_all_async())

    async def find_all_async(self) -> List[Counterparty]:
        """Find all counterparties asynchronously."""
        await self.connect()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM {self.schema}.counterparties ORDER BY name"
            )

            return [self._row_to_counterparty(row) for row in rows]

    async def find_banks_async(self) -> List[Counterparty]:
        """Find all bank counterparties asynchronously."""
        await self.connect()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM {self.schema}.counterparties WHERE is_bank = TRUE ORDER BY name"
            )

            return [self._row_to_counterparty(row) for row in rows]

    def _build_dsn(self) -> str:
        """Build PostgreSQL DSN connection string."""
        if self.config.dsn:
            return self.config.dsn

        if not self.config.database:
            raise ValueError("Database name is required for PostgreSQL repository")

        host = self.config.host or "localhost"
        port = self.config.port or 5432
        user = self.config.user or ""
        password = self.config.password or ""

        auth = ""
        if user:
            from urllib.parse import quote_plus

            auth = quote_plus(user)
            if password:
                auth += f":{quote_plus(password)}"
            auth += "@"

        return f"postgresql://{auth}{host}:{port}/{self.config.database}"

    def _row_to_counterparty(self, row: Any) -> Counterparty:
        """Convert database row to Counterparty object."""
        # Build credit profile if data exists
        credit = None
        if row["lgd"] is not None or row["credit_rating"] is not None:
            credit = CounterpartyCredit(
                rating=CreditRating(row["credit_rating"]) if row["credit_rating"] else None,
                internal_rating=row["internal_rating"],
                lgd=float(row["lgd"]) if row["lgd"] is not None else 0.6,
                recovery_rate=float(row["recovery_rate"]) if row["recovery_rate"] is not None else None,
                credit_spread_bps=float(row["credit_spread_bps"]) if row["credit_spread_bps"] is not None else None,
            )

        return Counterparty(
            id=row["id"],
            name=row["name"],
            entity_type=EntityType(row["entity_type"]),
            lei=row["lei"],
            jurisdiction=row["jurisdiction"],
            credit=credit,
            parent_id=row["parent_id"],
            is_bank=row["is_bank"],
            is_clearinghouse=row["is_clearinghouse"],
        )


class PostgresCSARepository(CSARepository):
    """PostgreSQL-based CSA agreement repository.

    Provides persistent storage of CSA agreements with:
    - Threshold and MTA terms
    - Collateral specifications
    - Eligible collateral with haircuts
    - Bilateral party relationships

    Attributes
    ----------
    config : DatabaseConfig
        Database connection configuration
    schema : str
        PostgreSQL schema name (default: "clients")
    """

    def __init__(
        self,
        config: DatabaseConfig,
        schema: str = "clients",
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        """Initialize PostgreSQL CSA repository.

        Parameters
        ----------
        config : DatabaseConfig
            Database connection configuration
        schema : str
            PostgreSQL schema name
        loop : asyncio.AbstractEventLoop, optional
            Event loop for async operations
        """
        self.config = config
        self.schema = schema
        self._pool = None
        self._loop = loop
        self._connected = False

    async def connect(self) -> bool:
        """Establish database connection and ensure schema exists."""
        if self._connected:
            return True

        try:
            import asyncpg
        except ImportError as exc:
            raise RuntimeError(
                "asyncpg is required for PostgreSQL repository. "
                "Install with `pip install asyncpg`."
            ) from exc

        dsn = self._build_dsn()
        self._pool = await asyncpg.create_pool(
            dsn=dsn,
            timeout=self.config.connect_timeout,
            **self.config.options,
        )

        await self._ensure_schema()
        self._connected = True
        return True

    async def disconnect(self) -> None:
        """Close database connection."""
        if self._pool:
            await self._pool.close()
            self._pool = None
        self._connected = False

    async def _ensure_schema(self) -> None:
        """Create schema and tables if they don't exist."""
        async with self._pool.acquire() as conn:
            # Create schema
            await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")

            # Create CSA agreements table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.csa_agreements (
                    id TEXT PRIMARY KEY,
                    party_a_id TEXT NOT NULL,
                    party_b_id TEXT NOT NULL,
                    effective_date TEXT NOT NULL,
                    -- Threshold terms
                    threshold_party_a NUMERIC DEFAULT 0,
                    threshold_party_b NUMERIC DEFAULT 0,
                    mta_party_a NUMERIC DEFAULT 0,
                    mta_party_b NUMERIC DEFAULT 0,
                    independent_amount_party_a NUMERIC DEFAULT 0,
                    independent_amount_party_b NUMERIC DEFAULT 0,
                    rounding NUMERIC DEFAULT 100000,
                    -- Collateral terms
                    base_currency TEXT NOT NULL,
                    valuation_frequency TEXT DEFAULT 'Daily',
                    valuation_time TEXT,
                    dispute_threshold NUMERIC DEFAULT 0,
                    dispute_resolution TEXT DEFAULT 'MarketQuotation',
                    -- Flags
                    initial_margin_required BOOLEAN DEFAULT FALSE,
                    variation_margin_required BOOLEAN DEFAULT TRUE,
                    rehypothecation_allowed BOOLEAN DEFAULT FALSE,
                    segregation_required BOOLEAN DEFAULT FALSE,
                    -- Metadata
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(party_a_id, party_b_id)
                )
            """)

            # Create eligible collateral table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.eligible_collateral (
                    id SERIAL PRIMARY KEY,
                    csa_id TEXT NOT NULL REFERENCES {self.schema}.csa_agreements(id) ON DELETE CASCADE,
                    collateral_type TEXT NOT NULL,
                    currency TEXT,
                    haircut NUMERIC DEFAULT 0,
                    concentration_limit NUMERIC,
                    rating_threshold TEXT,
                    maturity_max_years NUMERIC,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)

            # Create indexes
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_csa_party_a
                ON {self.schema}.csa_agreements (party_a_id)
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_csa_party_b
                ON {self.schema}.csa_agreements (party_b_id)
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_eligible_collateral_csa
                ON {self.schema}.eligible_collateral (csa_id)
            """)

    def save(self, csa: CSA) -> None:
        """Save a CSA agreement (synchronous wrapper)."""
        loop = self._loop or asyncio.get_event_loop()
        loop.run_until_complete(self.save_async(csa))

    async def save_async(self, csa: CSA) -> None:
        """Save a CSA agreement asynchronously."""
        await self.connect()

        async with self._pool.acquire() as conn:
            # Start transaction
            async with conn.transaction():
                # Save CSA main record
                await conn.execute(
                    f"""
                    INSERT INTO {self.schema}.csa_agreements (
                        id, party_a_id, party_b_id, effective_date,
                        threshold_party_a, threshold_party_b,
                        mta_party_a, mta_party_b,
                        independent_amount_party_a, independent_amount_party_b,
                        rounding, base_currency, valuation_frequency,
                        valuation_time, dispute_threshold, dispute_resolution,
                        initial_margin_required, variation_margin_required,
                        rehypothecation_allowed, segregation_required
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                        $11, $12, $13, $14, $15, $16, $17, $18, $19, $20
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        party_a_id = EXCLUDED.party_a_id,
                        party_b_id = EXCLUDED.party_b_id,
                        effective_date = EXCLUDED.effective_date,
                        threshold_party_a = EXCLUDED.threshold_party_a,
                        threshold_party_b = EXCLUDED.threshold_party_b,
                        mta_party_a = EXCLUDED.mta_party_a,
                        mta_party_b = EXCLUDED.mta_party_b,
                        independent_amount_party_a = EXCLUDED.independent_amount_party_a,
                        independent_amount_party_b = EXCLUDED.independent_amount_party_b,
                        rounding = EXCLUDED.rounding,
                        base_currency = EXCLUDED.base_currency,
                        valuation_frequency = EXCLUDED.valuation_frequency,
                        valuation_time = EXCLUDED.valuation_time,
                        dispute_threshold = EXCLUDED.dispute_threshold,
                        dispute_resolution = EXCLUDED.dispute_resolution,
                        initial_margin_required = EXCLUDED.initial_margin_required,
                        variation_margin_required = EXCLUDED.variation_margin_required,
                        rehypothecation_allowed = EXCLUDED.rehypothecation_allowed,
                        segregation_required = EXCLUDED.segregation_required,
                        updated_at = NOW()
                    """,
                    csa.id,
                    csa.party_a_id,
                    csa.party_b_id,
                    csa.effective_date,
                    csa.threshold_terms.threshold_party_a,
                    csa.threshold_terms.threshold_party_b,
                    csa.threshold_terms.mta_party_a,
                    csa.threshold_terms.mta_party_b,
                    csa.threshold_terms.independent_amount_party_a,
                    csa.threshold_terms.independent_amount_party_b,
                    csa.threshold_terms.rounding,
                    csa.collateral_terms.base_currency,
                    csa.collateral_terms.valuation_frequency.value,
                    csa.collateral_terms.valuation_time,
                    csa.collateral_terms.dispute_threshold,
                    csa.collateral_terms.dispute_resolution.value,
                    csa.initial_margin_required,
                    csa.variation_margin_required,
                    csa.rehypothecation_allowed,
                    csa.segregation_required,
                )

                # Delete existing eligible collateral
                await conn.execute(
                    f"DELETE FROM {self.schema}.eligible_collateral WHERE csa_id = $1",
                    csa.id,
                )

                # Save eligible collateral
                for ec in csa.collateral_terms.eligible_collateral:
                    await conn.execute(
                        f"""
                        INSERT INTO {self.schema}.eligible_collateral (
                            csa_id, collateral_type, currency, haircut,
                            concentration_limit, rating_threshold, maturity_max_years
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                        """,
                        csa.id,
                        ec.collateral_type.value,
                        ec.currency,
                        ec.haircut,
                        ec.concentration_limit,
                        ec.rating_threshold,
                        ec.maturity_max_years,
                    )

    def find_by_id(self, csa_id: str) -> Optional[CSA]:
        """Find a CSA by ID (synchronous wrapper)."""
        loop = self._loop or asyncio.get_event_loop()
        return loop.run_until_complete(self.find_by_id_async(csa_id))

    async def find_by_id_async(self, csa_id: str) -> Optional[CSA]:
        """Find a CSA by ID asynchronously."""
        await self.connect()

        async with self._pool.acquire() as conn:
            # Fetch CSA main record
            row = await conn.fetchrow(
                f"SELECT * FROM {self.schema}.csa_agreements WHERE id = $1",
                csa_id,
            )

            if not row:
                return None

            # Fetch eligible collateral
            collateral_rows = await conn.fetch(
                f"SELECT * FROM {self.schema}.eligible_collateral WHERE csa_id = $1",
                csa_id,
            )

            return self._rows_to_csa(row, collateral_rows)

    def find_by_counterparty(self, counterparty_id: str) -> List[CSA]:
        """Find CSAs involving a specific counterparty (synchronous wrapper)."""
        loop = self._loop or asyncio.get_event_loop()
        return loop.run_until_complete(self.find_by_counterparty_async(counterparty_id))

    async def find_by_counterparty_async(self, counterparty_id: str) -> List[CSA]:
        """Find CSAs involving a specific counterparty asynchronously."""
        await self.connect()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT * FROM {self.schema}.csa_agreements
                WHERE party_a_id = $1 OR party_b_id = $1
                """,
                counterparty_id,
            )

            csas = []
            for row in rows:
                collateral_rows = await conn.fetch(
                    f"SELECT * FROM {self.schema}.eligible_collateral WHERE csa_id = $1",
                    row["id"],
                )
                csas.append(self._rows_to_csa(row, collateral_rows))

            return csas

    def find_by_parties(self, party_a_id: str, party_b_id: str) -> Optional[CSA]:
        """Find CSA between two specific parties (synchronous wrapper)."""
        loop = self._loop or asyncio.get_event_loop()
        return loop.run_until_complete(self.find_by_parties_async(party_a_id, party_b_id))

    async def find_by_parties_async(self, party_a_id: str, party_b_id: str) -> Optional[CSA]:
        """Find CSA between two specific parties asynchronously."""
        await self.connect()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT * FROM {self.schema}.csa_agreements
                WHERE (party_a_id = $1 AND party_b_id = $2)
                   OR (party_a_id = $2 AND party_b_id = $1)
                """,
                party_a_id,
                party_b_id,
            )

            if not row:
                return None

            collateral_rows = await conn.fetch(
                f"SELECT * FROM {self.schema}.eligible_collateral WHERE csa_id = $1",
                row["id"],
            )

            return self._rows_to_csa(row, collateral_rows)

    def find_all(self) -> List[CSA]:
        """Find all CSA agreements (synchronous wrapper)."""
        loop = self._loop or asyncio.get_event_loop()
        return loop.run_until_complete(self.find_all_async())

    async def find_all_async(self) -> List[CSA]:
        """Find all CSA agreements asynchronously."""
        await self.connect()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM {self.schema}.csa_agreements"
            )

            csas = []
            for row in rows:
                collateral_rows = await conn.fetch(
                    f"SELECT * FROM {self.schema}.eligible_collateral WHERE csa_id = $1",
                    row["id"],
                )
                csas.append(self._rows_to_csa(row, collateral_rows))

            return csas

    def delete(self, csa_id: str) -> bool:
        """Delete a CSA agreement (synchronous wrapper)."""
        loop = self._loop or asyncio.get_event_loop()
        return loop.run_until_complete(self.delete_async(csa_id))

    async def delete_async(self, csa_id: str) -> bool:
        """Delete a CSA agreement asynchronously."""
        await self.connect()

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM {self.schema}.csa_agreements WHERE id = $1",
                csa_id,
            )

            return result != "DELETE 0"

    def _build_dsn(self) -> str:
        """Build PostgreSQL DSN connection string."""
        if self.config.dsn:
            return self.config.dsn

        if not self.config.database:
            raise ValueError("Database name is required for PostgreSQL repository")

        host = self.config.host or "localhost"
        port = self.config.port or 5432
        user = self.config.user or ""
        password = self.config.password or ""

        auth = ""
        if user:
            from urllib.parse import quote_plus

            auth = quote_plus(user)
            if password:
                auth += f":{quote_plus(password)}"
            auth += "@"

        return f"postgresql://{auth}{host}:{port}/{self.config.database}"

    def _rows_to_csa(self, main_row: Any, collateral_rows: List[Any]) -> CSA:
        """Convert database rows to CSA object."""
        # Build eligible collateral list
        eligible_collateral = []
        for row in collateral_rows:
            eligible_collateral.append(
                EligibleCollateral(
                    collateral_type=CollateralType(row["collateral_type"]),
                    currency=row["currency"],
                    haircut=float(row["haircut"]) if row["haircut"] is not None else 0.0,
                    concentration_limit=float(row["concentration_limit"]) if row["concentration_limit"] is not None else None,
                    rating_threshold=row["rating_threshold"],
                    maturity_max_years=float(row["maturity_max_years"]) if row["maturity_max_years"] is not None else None,
                )
            )

        # Build threshold terms
        threshold_terms = ThresholdTerms(
            threshold_party_a=float(main_row["threshold_party_a"]) if main_row["threshold_party_a"] is not None else 0.0,
            threshold_party_b=float(main_row["threshold_party_b"]) if main_row["threshold_party_b"] is not None else 0.0,
            mta_party_a=float(main_row["mta_party_a"]) if main_row["mta_party_a"] is not None else 0.0,
            mta_party_b=float(main_row["mta_party_b"]) if main_row["mta_party_b"] is not None else 0.0,
            independent_amount_party_a=float(main_row["independent_amount_party_a"]) if main_row["independent_amount_party_a"] is not None else 0.0,
            independent_amount_party_b=float(main_row["independent_amount_party_b"]) if main_row["independent_amount_party_b"] is not None else 0.0,
            rounding=float(main_row["rounding"]) if main_row["rounding"] is not None else 100000.0,
        )

        # Build collateral terms
        collateral_terms = CollateralTerms(
            base_currency=main_row["base_currency"],
            valuation_frequency=ValuationFrequency(main_row["valuation_frequency"]),
            valuation_time=main_row["valuation_time"],
            dispute_threshold=float(main_row["dispute_threshold"]) if main_row["dispute_threshold"] is not None else 0.0,
            dispute_resolution=DisputeResolution(main_row["dispute_resolution"]),
            eligible_collateral=eligible_collateral,
        )

        # Build CSA
        return CSA(
            id=main_row["id"],
            party_a_id=main_row["party_a_id"],
            party_b_id=main_row["party_b_id"],
            effective_date=main_row["effective_date"],
            threshold_terms=threshold_terms,
            collateral_terms=collateral_terms,
            initial_margin_required=main_row["initial_margin_required"],
            variation_margin_required=main_row["variation_margin_required"],
            rehypothecation_allowed=main_row["rehypothecation_allowed"],
            segregation_required=main_row["segregation_required"],
        )


__all__ = [
    "CSARepository",
    "PostgresCounterpartyRepository",
    "PostgresCSARepository",
]
