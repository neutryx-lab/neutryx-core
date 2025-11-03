"""Compatibility shim for credit risk helpers."""

from __future__ import annotations

from neutryx.market import credit as _credit
from neutryx.market.credit import *  # noqa: F401,F403

__all__ = getattr(_credit, "__all__", [])
