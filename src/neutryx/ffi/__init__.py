"""Compatibility shim for optional native bindings."""

from __future__ import annotations

import importlib
import pkgutil
import sys
from types import ModuleType

from neutryx.bridge import ffi as _ffi
from neutryx.bridge.ffi import *  # noqa: F401,F403

__all__ = getattr(_ffi, "__all__", [])
__path__ = getattr(_ffi, "__path__", [])  # type: ignore[assignment]


def _register_submodules(source: ModuleType, alias: str) -> None:
    """Register submodules so ``import neutryx.ffi.foo`` resolves correctly."""
    if not hasattr(source, "__path__"):
        return
    for module_info in pkgutil.iter_modules(source.__path__):  # type: ignore[attr-defined]
        target_name = f"{alias}.{module_info.name}"
        if target_name in sys.modules:
            continue
        sys.modules[target_name] = importlib.import_module(f"{source.__name__}.{module_info.name}")


_register_submodules(_ffi, __name__)


def __getattr__(name: str):
    return getattr(_ffi, name)


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(dir(_ffi)))
