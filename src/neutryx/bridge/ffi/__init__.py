"""Optional native extension bindings.

This module exposes helpers to load optional native extensions that back high performance
kernels used by the Neutryx laboratory.  The bindings are intentionally defensive: if the
corresponding shared libraries are not available we fall back to light-weight stub objects
that provide rich error messages while keeping the rest of the code base importable.
"""

from __future__ import annotations

from .base import OptionalExtension
from . import quantlib, eigen, performance

__all__ = [
    "OptionalExtension",
    "quantlib",
    "eigen",
    "performance",
]
