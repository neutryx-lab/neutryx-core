"""Infrastructure for optional native extensions.

The project frequently talks to native libraries that might not be present on every
installation (for example, QuantLib or custom Eigen-based kernels).  The :class:`OptionalExtension`
helper defined here wraps the import lifecycle, keeps track of errors, and exposes a consistent
interface to consumers.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable, Optional

LOGGER = logging.getLogger(__name__)

FallbackFactory = Callable[[str, Optional[Exception]], Any]

__all__ = [
    "OptionalExtension",
    "UnavailableExtension",
    "default_fallback",
]


class UnavailableExtension:
    """Light-weight proxy returned when a native extension is missing.

    Attempting to access any attribute on the proxy raises a descriptive error message.  The
    object evaluates to ``False`` so it can be used in ``if`` statements for quick availability
    checks.
    """

    def __init__(self, import_path: str, error: Optional[Exception]) -> None:
        self.import_path = import_path
        self.error = error

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return False

    def __getattr__(self, item: str) -> Any:
        raise RuntimeError(self._message()) from self.error

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<UnavailableExtension {self.import_path!r}>"

    def _message(self) -> str:
        if self.error is None:
            return f"Native extension '{self.import_path}' is not available."
        return (
            "Native extension '{name}' could not be imported: {error_type}: {error}".format(
                name=self.import_path,
                error_type=type(self.error).__name__,
                error=self.error,
            )
        )


def default_fallback(import_path: str, error: Optional[Exception]) -> UnavailableExtension:
    """Default factory returning an :class:`UnavailableExtension`."""

    return UnavailableExtension(import_path, error)


@dataclass
class OptionalExtension:
    """Encapsulates the lifecycle of an optional extension module."""

    display_name: str
    import_path: str
    fallback_factory: FallbackFactory = default_fallback
    eager: bool = False

    _module: Optional[Any] = field(default=None, init=False)
    _fallback: Optional[Any] = field(default=None, init=False)
    _error: Optional[Exception] = field(default=None, init=False)
    _available: Optional[bool] = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.eager:
            self.load()

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def load(self) -> Any:
        """Ensure the extension has been attempted to load.

        Returns the loaded module if available, otherwise the fallback proxy.
        """

        if self._available is not None:
            return self._module if self._available else self._fallback

        try:
            LOGGER.debug("Importing optional extension '%s'", self.import_path)
            module = importlib.import_module(self.import_path)
        except Exception as exc:  # pragma: no cover - exercised in tests
            self._error = exc
            self._available = False
            self._fallback = self.fallback_factory(self.import_path, exc)
            LOGGER.debug(
                "Failed to import optional extension '%s': %s: %s",
                self.import_path,
                type(exc).__name__,
                exc,
            )
            return self._fallback

        self._module = module
        self._available = True
        return module

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """Return ``True`` if the extension module successfully imported."""

        self.load()
        return bool(self._available)

    def get_module(self) -> Optional[Any]:
        """Return the imported module or ``None`` if unavailable."""

        self.load()
        return self._module if self._available else None

    def get_fallback(self) -> Optional[Any]:
        """Return the fallback proxy when the extension is not available."""

        self.load()
        return self._fallback if not self._available else None

    def get(self) -> Any:
        """Return the imported module if available, otherwise the fallback proxy."""

        self.load()
        return self._module if self._available else self._fallback

    def get_error(self) -> Optional[Exception]:
        """Return the error raised during import, if any."""

        self.load()
        return self._error

    def require(self) -> Any:
        """Return the imported module or raise a descriptive :class:`RuntimeError`."""

        module = self.get_module()
        if module is not None:
            return module
        fallback = self.get_fallback()
        if isinstance(fallback, UnavailableExtension):
            raise RuntimeError(fallback._message()) from self._error
        raise RuntimeError(
            f"Extension '{self.display_name}' is unavailable."
        )

    # ------------------------------------------------------------------
    # User facing helpers
    # ------------------------------------------------------------------
    def describe(self) -> str:
        """Return a human friendly description of the extension state."""

        if self.is_available():
            module = self.get_module()
            if isinstance(module, ModuleType):
                return f"{self.display_name} extension loaded from {module.__file__}"
            return f"{self.display_name} extension loaded."
        if self._error is None:
            return f"{self.display_name} extension is not available."
        return (
            f"{self.display_name} extension failed to import: "
            f"{type(self._error).__name__}: {self._error}"
        )
