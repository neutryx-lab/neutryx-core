"""High-performance data structures for batch processing.

This module provides JAX-optimized data structures for efficient portfolio
and market data processing:

- **Struct-of-Arrays (SoA)**: Portfolio data optimized for vectorized operations
- **JAX PyTrees**: Market data structures that JIT-compile efficiently
- **Index Mappings**: Efficient categorical data encoding
- **Data Converters**: Migration utilities from legacy formats

Design Principles
-----------------
1. **CPU-first optimization**: SIMD-friendly memory layouts
2. **JAX-native**: All structures are pytree-compatible
3. **Zero-copy where possible**: Memory-mapped and shared arrays
4. **Batch-oriented**: Operations on 10K+ trades simultaneously
"""

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
]
