"""Compatibility shim for Foreign Function Interfaces.

This module re-exports the FFI functionality that has been moved to
``neutryx.integrations.ffi``.  Keeping the shim allows older imports to remain
functional during the package reorganisation.
"""

from __future__ import annotations

from neutryx.integrations.ffi import *  # noqa: F401,F403

__all__ = []
