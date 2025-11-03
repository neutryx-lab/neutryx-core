"""High-performance batch computation engine.

This module provides vectorized pricing and risk calculation functions optimized
for processing 10K+ trades simultaneously on CPU and GPU.

Key Features:
- **Batch Pricing**: Price entire portfolios in single JIT-compiled kernels
- **Vectorized Greeks**: Compute sensitivities for all trades in parallel
- **Sparse Aggregation**: Efficient counterparty/netting set aggregation
- **CPU-First Optimization**: SIMD-friendly operations, cache-aware algorithms

Design Principles:
- All functions are JAX JIT-compilable
- vmap/pmap for automatic parallelization
- Minimize branches and loops in hot paths
- Pre-computed market data grids (no runtime interpolation)
"""

from neutryx.compute.aggregation import (
    aggregate_by_index,
    aggregate_to_counterparties,
    compute_concentration_metrics,
    compute_counterparty_exposures,
    compute_gross_exposure,
    compute_netting_factors,
    compute_top_n_counterparties,
)
from neutryx.compute.batch_pricing import (
    compute_portfolio_delta_batch,
    compute_portfolio_pv,
    price_portfolio_batch,
    price_vanilla_options_batch,
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
]
