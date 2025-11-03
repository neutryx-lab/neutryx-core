"""Compatibility shim for optional native bindings."""

from __future__ import annotations

from neutryx.bridge import ffi as _ffi
from neutryx.bridge.ffi import *  # noqa: F401,F403

__all__ = getattr(_ffi, "__all__", [])
