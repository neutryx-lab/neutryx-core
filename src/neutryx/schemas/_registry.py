"""Schema registry for discovery, validation, and metadata.

The registry provides centralized access to all Neutryx schemas,
enabling:
- Schema discovery by name or domain
- Version tracking and compatibility checking
- Schema metadata access
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Dict, Iterator, List, Optional, Set, Type

from pydantic import BaseModel


class SchemaRegistry:
    """Central registry for all Neutryx Pydantic schemas.

    Singleton pattern ensures consistent access across the application.

    Example:
        >>> registry = SchemaRegistry.instance()
        >>> trade_schema = registry.get("Trade")
        >>> trading_schemas = registry.by_domain("trading")
    """

    _instance: Optional[SchemaRegistry] = None
    _schemas: Dict[str, Type[BaseModel]] = {}
    _by_domain: Dict[str, Set[str]] = {}
    _initialized: bool = False

    def __new__(cls) -> SchemaRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def instance(cls) -> SchemaRegistry:
        """Get the singleton registry instance."""
        if cls._instance is None:
            cls._instance = cls()
        if not cls._initialized:
            cls._instance._discover_schemas()
            cls._initialized = True
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (mainly for testing)."""
        cls._instance = None
        cls._schemas = {}
        cls._by_domain = {}
        cls._initialized = False

    def _discover_schemas(self) -> None:
        """Discover and register all schemas in the schemas package."""
        import neutryx.schemas as schemas_pkg

        # Walk through all submodules
        for _importer, modname, _ispkg in pkgutil.walk_packages(
            schemas_pkg.__path__, prefix="neutryx.schemas."
        ):
            # Skip private modules and codegen
            if "._" in modname or "_codegen" in modname:
                continue

            try:
                module = importlib.import_module(modname)
            except ImportError:
                continue

            # Find all BaseModel subclasses in the module
            for name in dir(module):
                obj = getattr(module, name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, BaseModel)
                    and obj is not BaseModel
                    and obj.__module__ == module.__name__
                ):
                    self.register(obj)

    def register(self, schema_cls: Type[BaseModel]) -> None:
        """Register a schema class.

        Args:
            schema_cls: The Pydantic model class to register
        """
        name = schema_cls.__name__

        # Avoid re-registration
        if name in self._schemas:
            return

        self._schemas[name] = schema_cls

        # Index by domain if available
        domain = getattr(schema_cls, "__schema_domain__", "unknown")
        if domain not in self._by_domain:
            self._by_domain[domain] = set()
        self._by_domain[domain].add(name)

    def get(self, name: str) -> Optional[Type[BaseModel]]:
        """Get a schema by name.

        Args:
            name: The schema class name

        Returns:
            The schema class or None if not found
        """
        return self._schemas.get(name)

    def by_domain(self, domain: str) -> List[Type[BaseModel]]:
        """Get all schemas in a domain.

        Args:
            domain: Domain name (e.g., 'trading', 'clearing')

        Returns:
            List of schema classes in the domain
        """
        names = self._by_domain.get(domain, set())
        return [self._schemas[n] for n in names if n in self._schemas]

    def all_schemas(self) -> List[Type[BaseModel]]:
        """Get all registered schemas.

        Returns:
            List of all registered schema classes
        """
        return list(self._schemas.values())

    def all_names(self) -> List[str]:
        """Get names of all registered schemas.

        Returns:
            List of schema names
        """
        return list(self._schemas.keys())

    def domains(self) -> List[str]:
        """Get all registered domains.

        Returns:
            List of domain names
        """
        return list(self._by_domain.keys())

    def schema_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a schema.

        Args:
            name: The schema class name

        Returns:
            Dictionary with schema metadata or None
        """
        schema_cls = self.get(name)
        if schema_cls is None:
            return None

        if hasattr(schema_cls, "schema_info"):
            return schema_cls.schema_info()

        return {
            "name": name,
            "version": getattr(schema_cls, "__schema_version__", "unknown"),
            "domain": getattr(schema_cls, "__schema_domain__", "unknown"),
            "module": schema_cls.__module__,
        }

    def __iter__(self) -> Iterator[Type[BaseModel]]:
        """Iterate over all registered schemas."""
        return iter(self._schemas.values())

    def __len__(self) -> int:
        """Return count of registered schemas."""
        return len(self._schemas)

    def __contains__(self, name: str) -> bool:
        """Check if a schema is registered."""
        return name in self._schemas


def get_schema(name: str) -> Optional[Type[BaseModel]]:
    """Convenience function to get a schema by name.

    Args:
        name: Schema class name

    Returns:
        The schema class or None
    """
    return SchemaRegistry.instance().get(name)


def list_schemas(domain: Optional[str] = None) -> List[str]:
    """List registered schema names.

    Args:
        domain: Optional domain filter

    Returns:
        List of schema names
    """
    registry = SchemaRegistry.instance()
    if domain:
        return [s.__name__ for s in registry.by_domain(domain)]
    return registry.all_names()
