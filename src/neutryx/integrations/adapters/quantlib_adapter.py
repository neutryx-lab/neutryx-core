"""Bridge utilities for the optional QuantLib dependency."""

from __future__ import annotations

from neutryx.integrations.ffi import quantlib

# ``ql`` mirrors the historical interface: it is ``None`` when QuantLib is not available to keep the
# import side-effect free for downstream projects.  Consumers that want richer diagnostics can use
# the helpers exposed by :mod:`neutryx.ffi.quantlib` directly.
ql = quantlib.get_quantlib_module()
