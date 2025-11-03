"""Compatibility shim for pricing modules.

Projects importing :mod:`neutryx.pricing` expect the public pricing API to be
available from this namespace.  The canonical implementation now lives in
``neutryx.core.pricing`` so we forward the exports to preserve the contract.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from types import ModuleType

from neutryx.core import pricing as _pricing
from neutryx.core.pricing import *  # noqa: F401,F403

__all__ = getattr(_pricing, "__all__", [])
__path__ = getattr(_pricing, "__path__", [])  # type: ignore[assignment]


def _register_submodules(source: ModuleType, alias: str) -> None:
    if not hasattr(source, "__path__"):
        return
    for module_info in pkgutil.iter_modules(source.__path__):  # type: ignore[attr-defined]
        target_name = f"{alias}.{module_info.name}"
        if target_name in sys.modules:
            continue
        sys.modules[target_name] = importlib.import_module(f"{source.__name__}.{module_info.name}")


_register_submodules(_pricing, __name__)


def __getattr__(name: str):
    return getattr(_pricing, name)


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(dir(_pricing)))
