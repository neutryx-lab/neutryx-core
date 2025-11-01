"""Profiling utilities for Neutryx Lab.

This package exposes helpers that make it easier to instrument JAX
workloads and collect structured kernel timing information.  The
entry-point for most users is :class:`~tools.profiling.kernel_profiler.KernelProfiler`.
"""

from .kernel_profiler import KernelEvent, KernelProfiler

__all__ = ["KernelEvent", "KernelProfiler"]
