"""Bindings for optional Eigen-backed kernels."""

from __future__ import annotations

import os
from typing import Any, Optional

from .base import OptionalExtension

_EIGEN_MODULE = os.getenv("NEUTRYX_EIGEN_MODULE", "neutryx_ext.eigen")

_eigen_extension = OptionalExtension(
    display_name="Eigen kernels",
    import_path=_EIGEN_MODULE,
)

__all__ = [
    "load_eigen_kernels",
    "eigen_available",
    "get_eigen_module",
    "require_eigen",
    "describe_eigen",
]


def load_eigen_kernels() -> Any:
    """Return the Eigen kernel module or a stub when not available."""

    return _eigen_extension.get()


def eigen_available() -> bool:
    """Return ``True`` if the Eigen kernels module imported successfully."""

    return _eigen_extension.is_available()


def get_eigen_module() -> Optional[Any]:
    """Return the actual Eigen module when available."""

    return _eigen_extension.get_module()


def require_eigen() -> Any:
    """Return the Eigen module or raise a descriptive error."""

    return _eigen_extension.require()


def describe_eigen() -> str:
    """Return a human readable description of the Eigen import status."""

    return _eigen_extension.describe()
