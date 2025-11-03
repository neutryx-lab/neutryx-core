"""Compatibility shim for pricing modules.

Projects importing :mod:`neutryx.pricing` expect the public pricing API to be
available from this namespace.  The canonical implementation now lives in
``neutryx.core.pricing`` so we forward the exports to preserve the contract.
"""

from __future__ import annotations

from neutryx.core import pricing as _pricing
from neutryx.core.pricing import *  # noqa: F401,F403

__all__ = getattr(_pricing, "__all__", [])
