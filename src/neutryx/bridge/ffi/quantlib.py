"""Bindings for the QuantLib optional native extension."""

from __future__ import annotations

import os
from typing import Any, Optional

from .base import OptionalExtension, UnavailableExtension

_QUANTLIB_MODULE = os.getenv("NEUTRYX_QUANTLIB_MODULE", "QuantLib")

_quantlib_extension = OptionalExtension(
    display_name="QuantLib",
    import_path=_QUANTLIB_MODULE,
)

__all__ = [
    "load_quantlib",
    "get_quantlib_module",
    "quantlib_available",
    "require_quantlib",
    "describe_quantlib",
]


def load_quantlib(prefer_fallback: bool = False) -> Any:
    """Return the QuantLib module or a stub when not available.

    Parameters
    ----------
    prefer_fallback:
        When set to ``True`` the fallback proxy is preferred when available.  This is mostly useful
        for testing and diagnostics.
    """

    module = _quantlib_extension.get()
    if prefer_fallback and not isinstance(module, UnavailableExtension):
        return _quantlib_extension.get_fallback() or module
    return module


def get_quantlib_module() -> Optional[Any]:
    """Return the loaded QuantLib module when available."""

    return _quantlib_extension.get_module()


def quantlib_available() -> bool:
    """Return ``True`` when the QuantLib native extension is importable."""

    return _quantlib_extension.is_available()


def require_quantlib() -> Any:
    """Return the QuantLib module or raise a descriptive error."""

    return _quantlib_extension.require()


def describe_quantlib() -> str:
    """Return a human readable description of the QuantLib import status."""

    return _quantlib_extension.describe()
