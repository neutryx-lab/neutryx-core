"""Compatibility shim for calibration utilities."""

from __future__ import annotations

import importlib
import pkgutil
import sys
from types import ModuleType

from neutryx.solver import calibration as _calibration
from neutryx.solver.calibration import *  # noqa: F401,F403

__all__ = getattr(_calibration, "__all__", [])
__path__ = getattr(_calibration, "__path__", [])  # type: ignore[assignment]


def _register_submodules(source: ModuleType, alias: str) -> None:
    if not hasattr(source, "__path__"):
        return
    for module_info in pkgutil.iter_modules(source.__path__):  # type: ignore[attr-defined]
        target_name = f"{alias}.{module_info.name}"
        if target_name in sys.modules:
            continue
        sys.modules[target_name] = importlib.import_module(f"{source.__name__}.{module_info.name}")


_register_submodules(_calibration, __name__)


def __getattr__(name: str):
    return getattr(_calibration, name)


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(dir(_calibration)))
