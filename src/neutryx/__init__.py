"""Neutryx core package."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Dict

__all__ = [
    "api",
    "calibration",
    "calibrate",
    "core",
    "market",
    "models",
    "portfolio",
    "products",
    "risk",
    "xva",
    "contracts",
    "utils",
    "credit",
    "cli",
]

_MODULE_ALIASES: Dict[str, str] = {
    "api": "neutryx.api",
    "calibration": "neutryx.calibration",
    "calibrate": "neutryx.calibration",
    "core": "neutryx.core",
    "market": "neutryx.market",
    "models": "neutryx.models",
    "portfolio": "neutryx.portfolio",
    "products": "neutryx.products",
    "risk": "neutryx.valuations.risk",
    "xva": "neutryx.valuations.xva",
    "contracts": "neutryx.portfolio.contracts",
    "utils": "neutryx.core.utils",
    "credit": "neutryx.market.credit",
    "cli": "neutryx.core.cli",
}


class _MissingDependencyModule(ModuleType):
    """Placeholder for modules that require optional dependencies."""

    def __init__(self, name: str, error: ModuleNotFoundError) -> None:
        super().__init__(name)
        self.__dict__["__missing_dependency_error__"] = error

    def __getattr__(self, attr: str):  # type: ignore[override]
        error: ModuleNotFoundError = self.__dict__["__missing_dependency_error__"]
        raise ModuleNotFoundError(
            f"Module '{self.__name__}' could not be imported because of a missing "
            f"optional dependency: {error}."
        ) from error


def __getattr__(name: str):
    if name in _MODULE_ALIASES:
        module_name = _MODULE_ALIASES[name]
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised in optional setups
            placeholder = _MissingDependencyModule(module_name, exc)
            globals()[name] = placeholder
            return placeholder
        else:
            globals()[name] = module
            return module
    raise AttributeError(f"module 'neutryx' has no attribute '{name}'")


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))


__version__ = "0.1.0"
