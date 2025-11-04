"""High-performance data structures and batch computation engine.

This module provides JAX-optimized data structures and computation functions
for efficient portfolio and market data processing:

- **Struct-of-Arrays (SoA)**: Portfolio data optimized for vectorized operations
- **JAX PyTrees**: Market data structures that JIT-compile efficiently
- **Index Mappings**: Efficient categorical data encoding
- **Data Converters**: Migration utilities from legacy formats
- **Batch Pricing**: Price entire portfolios in single JIT-compiled kernels
- **Vectorized Greeks**: Compute sensitivities for all trades in parallel
- **Sparse Aggregation**: Efficient counterparty/netting set aggregation

Design Principles
-----------------
1. **CPU-first optimization**: SIMD-friendly memory layouts
2. **JAX-native**: All structures are pytree-compatible
3. **Zero-copy where possible**: Memory-mapped and shared arrays
4. **Batch-oriented**: Operations on 10K+ trades simultaneously
"""

from neutryx.data.aggregation import (
    aggregate_by_index,
    aggregate_to_counterparties,
    compute_concentration_metrics,
    compute_counterparty_exposures,
    compute_gross_exposure,
    compute_netting_factors,
    compute_top_n_counterparties,
)
from neutryx.data.batch_pricing import (
    compute_portfolio_delta_batch,
    compute_portfolio_pv,
    price_portfolio_batch,
    price_vanilla_options_batch,
)
from neutryx.data.chunked_simulation import (
    ChunkedSimConfig,
    estimate_optimal_chunk_size,
    load_chunked_simulation,
    price_option_chunked,
    simulate_gbm_chunked,
)
from neutryx.data.converters import (
    batch_to_trade_dicts,
    pydantic_portfolio_to_batch,
    trades_to_portfolio_batch,
)
from neutryx.data.grid_builder import (
    build_market_data_grid_from_environment,
    build_market_data_grid_simple,
    build_time_grid,
    estimate_grid_memory_size,
)
from neutryx.data.indices import (
    IndexMapping,
    build_index_mapping,
    create_sparse_aggregation_matrix,
)
from neutryx.data.market_grid import MarketDataGrid
from neutryx.data.portfolio_batch import PortfolioBatch, TradeArrays
from neutryx.data.validation import (
    DataValidator,
    RangeRule,
    RequiredFieldRule,
    Severity,
    StalenessRule,
    ValidationIssue,
    ValidationResult,
)

__all__ = [
    # Index mappings
    "IndexMapping",
    "build_index_mapping",
    "create_sparse_aggregation_matrix",
    # Portfolio data structures
    "PortfolioBatch",
    "TradeArrays",
    # Market data structures
    "MarketDataGrid",
    # Grid builders
    "build_time_grid",
    "build_market_data_grid_simple",
    "build_market_data_grid_from_environment",
    "estimate_grid_memory_size",
    # Converters
    "trades_to_portfolio_batch",
    "pydantic_portfolio_to_batch",
    "batch_to_trade_dicts",
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
    # Validation
    "DataValidator",
    "RangeRule",
    "RequiredFieldRule",
    "Severity",
    "StalenessRule",
    "ValidationIssue",
    "ValidationResult",
]
