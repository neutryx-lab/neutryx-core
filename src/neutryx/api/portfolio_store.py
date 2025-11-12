"""Portfolio store abstractions for the REST API layer.

The REST endpoints previously relied on an in-memory dictionary scoped to the
FastAPI application instance.  That approach made it impossible to persist
registered portfolios across application restarts and introduced race
conditions when multiple clients attempted to read/write simultaneously.

This module introduces a small service layer responsible for persisting
portfolios.  Different storage backends can be selected at runtime (e.g.
in-memory for tests, filesystem for lightweight persistence, or future
extensions for database/cache backends).
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Protocol

from neutryx.portfolio.portfolio import Portfolio


class PortfolioStore(Protocol):
    """Abstract interface describing persistence operations for portfolios."""

    def save_portfolio(self, portfolio: Portfolio) -> None:
        """Persist or update a portfolio instance."""

    def get_portfolio(self, portfolio_id: str) -> Portfolio | None:
        """Retrieve a portfolio by identifier, returning ``None`` if missing."""

    def delete_portfolio(self, portfolio_id: str) -> None:
        """Remove a stored portfolio."""

    def list_portfolios(self) -> Iterable[str]:
        """Return an iterable of known portfolio identifiers."""


class InMemoryPortfolioStore(PortfolioStore):
    """Thread-safe in-memory store used for development and testing."""

    def __init__(self) -> None:
        self._portfolios: Dict[str, Portfolio] = {}
        self._lock = threading.RLock()

    def save_portfolio(self, portfolio: Portfolio) -> None:
        with self._lock:
            # Store a deep copy to prevent accidental mutation by callers.
            self._portfolios[portfolio.name] = portfolio.model_copy(deep=True)

    def get_portfolio(self, portfolio_id: str) -> Portfolio | None:
        with self._lock:
            portfolio = self._portfolios.get(portfolio_id)
            return None if portfolio is None else portfolio.model_copy(deep=True)

    def delete_portfolio(self, portfolio_id: str) -> None:
        with self._lock:
            self._portfolios.pop(portfolio_id, None)

    def list_portfolios(self) -> Iterable[str]:
        with self._lock:
            return list(self._portfolios.keys())


class FileSystemPortfolioStore(PortfolioStore):
    """Persist portfolios to a JSON file on the local filesystem."""

    def __init__(self, path: Path) -> None:
        self._path = path.expanduser().resolve()
        self._lock = threading.RLock()
        self._portfolios: Dict[str, Portfolio] = {}
        self._initialised = False
        self._load()

    def _load(self) -> None:
        with self._lock:
            if self._initialised:
                return
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if self._path.exists():
                try:
                    raw = json.loads(self._path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    raw = {}
                if isinstance(raw, dict):
                    for portfolio_id, payload in raw.items():
                        try:
                            self._portfolios[portfolio_id] = Portfolio.model_validate(payload)
                        except Exception:
                            # Skip corrupt entries but continue loading the rest
                            continue
            self._initialised = True

    def _flush(self) -> None:
        data = {
            portfolio_id: portfolio.model_dump(mode="json")
            for portfolio_id, portfolio in self._portfolios.items()
        }
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self._path)

    def save_portfolio(self, portfolio: Portfolio) -> None:
        with self._lock:
            self._portfolios[portfolio.name] = portfolio.model_copy(deep=True)
            self._flush()

    def get_portfolio(self, portfolio_id: str) -> Portfolio | None:
        with self._lock:
            portfolio = self._portfolios.get(portfolio_id)
            return None if portfolio is None else portfolio.model_copy(deep=True)

    def delete_portfolio(self, portfolio_id: str) -> None:
        with self._lock:
            if portfolio_id in self._portfolios:
                del self._portfolios[portfolio_id]
                self._flush()

    def list_portfolios(self) -> Iterable[str]:
        with self._lock:
            return list(self._portfolios.keys())


@dataclass
class PortfolioStoreSettings:
    """Configuration for selecting a portfolio store backend."""

    backend: str = "memory"
    filesystem_path: Path | None = None

    @classmethod
    def from_env(cls) -> "PortfolioStoreSettings":
        backend = os.getenv("NEUTRYX_PORTFOLIO_STORE", "memory").strip().lower() or "memory"
        raw_path = os.getenv("NEUTRYX_PORTFOLIO_STORE_PATH")
        path = Path(raw_path).expanduser() if raw_path else None
        return cls(backend=backend, filesystem_path=path)


def create_portfolio_store(settings: PortfolioStoreSettings | None = None) -> PortfolioStore:
    """Create a portfolio store instance based on the provided settings."""

    resolved_settings = settings or PortfolioStoreSettings.from_env()
    backend = resolved_settings.backend

    if backend in {"memory", "inmemory", "in-memory"}:
        return InMemoryPortfolioStore()
    if backend in {"filesystem", "file", "fs"}:
        path = resolved_settings.filesystem_path
        if path is None:
            default_dir = Path.home() / ".cache" / "neutryx"
            path = default_dir / "portfolios.json"
        return FileSystemPortfolioStore(path)

    raise ValueError(f"Unsupported portfolio store backend: {backend}")


__all__ = [
    "PortfolioStore",
    "InMemoryPortfolioStore",
    "FileSystemPortfolioStore",
    "PortfolioStoreSettings",
    "create_portfolio_store",
]
