"""Core utilities for computation and precision management."""

from __future__ import annotations

from . import solvers  # noqa: F401
from .math import *  # noqa: F401, F403
from .parallel import *  # noqa: F401, F403
from .precision import *  # noqa: F401, F403
from .registry import *  # noqa: F401, F403
from .types import *  # noqa: F401, F403

__all__ = [
    # math utilities
    "logsumexp",
    # numerical solvers
    "solvers",
    # precision
    "apply_loss_scaling",
    "canonicalize_dtype",
    "get_compute_dtype",
    "get_loss_scale",
    "undo_loss_scaling",
    # parallel
    "ParallelConfig",
    "ParallelExecutable",
    "compile_parallel",
    # registry
    "Registry",
    # types
    "OptionSpec",
]
