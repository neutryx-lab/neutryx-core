"""Compatibility shim for batch computation engine.

This module re-exports the batch computation functions that have been moved to
``neutryx.data``.  Keeping the shim allows older imports to remain
functional during the package reorganisation.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from types import ModuleType

from neutryx import data as _data

# Re-export all batch computation functions from neutryx.data
from neutryx.data import (
    ChunkedSimConfig,
    aggregate_by_index,
    aggregate_to_counterparties,
    compute_concentration_metrics,
    compute_counterparty_exposures,
    compute_gross_exposure,
    compute_netting_factors,
    compute_portfolio_delta_batch,
    compute_portfolio_pv,
    compute_top_n_counterparties,
    estimate_optimal_chunk_size,
    load_chunked_simulation,
    price_option_chunked,
    price_portfolio_batch,
    price_vanilla_options_batch,
    simulate_gbm_chunked,
)

__all__ = [
    # Batch pricing
    "price_portfolio_batch",
    "price_vanilla_options_batch",
    "compute_portfolio_pv",
    "compute_portfolio_delta_batch",
    # Aggregation
    "aggregate_by_index",
    "aggregate_to_counterparties",
    "compute_counterparty_exposures",
    "compute_netting_factors",
    "compute_gross_exposure",
    "compute_top_n_counterparties",
    "compute_concentration_metrics",
    # Chunked simulation
    "ChunkedSimConfig",
    "simulate_gbm_chunked",
    "load_chunked_simulation",
    "price_option_chunked",
    "estimate_optimal_chunk_size",
]


def _register_submodules(source: ModuleType, alias: str) -> None:
    if not hasattr(source, "__path__"):
        return
    for module_info in pkgutil.iter_modules(source.__path__):  # type: ignore[attr-defined]
        target_name = f"{alias}.{module_info.name}"
        if target_name in sys.modules:
            continue
        sys.modules[target_name] = importlib.import_module(f"{source.__name__}.{module_info.name}")


_register_submodules(_data, __name__)


def __getattr__(name: str):
    return getattr(_data, name)


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(dir(_data)))
