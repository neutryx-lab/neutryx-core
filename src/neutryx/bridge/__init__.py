"""Compatibility shim for external library integrations.

This module re-exports the integration functionality that has been moved to
``neutryx.integrations``.  Keeping the shim allows older imports to remain
functional during the package reorganisation.
"""

from __future__ import annotations

# Re-export from new location
from neutryx.integrations import fpml

__all__ = ["fpml"]
