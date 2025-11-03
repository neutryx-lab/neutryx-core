"""Compatibility shim for configuration access.

This re-exports the public configuration helpers that previously lived under
``neutryx.config`` while the implementation now resides in
``neutryx.core.config``.  Keeping the shim allows older imports to remain
functional during the package reorganisation.
"""

from __future__ import annotations

from neutryx.core import config as _config
from neutryx.core.config import ConfigDict, get_config, get_default_config, init_environment

__all__ = ["ConfigDict", "get_config", "get_default_config", "init_environment"]


def __getattr__(name: str):
    return getattr(_config, name)


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(dir(_config)))
