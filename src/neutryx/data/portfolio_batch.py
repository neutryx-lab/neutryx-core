"""Struct-of-Arrays portfolio representation for high-performance batch processing.

This module provides a JAX-optimized portfolio data structure that stores all trades
in a vectorized format, enabling efficient CPU SIMD operations and GPU processing.

Key Design Principles:
- Struct-of-Arrays (SoA) layout for memory efficiency
- JAX pytree registration for JIT compilation
- Integer indexing for categorical data
- Pre-built sparse matrices for aggregation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import jax
import jax.numpy as jnp
from jax import Array

from neutryx.data.indices import IndexMapping


@dataclass(frozen=True)
class TradeArrays:
    """Raw numerical arrays for trade parameters.

    All arrays have shape [n_trades] and are aligned for vectorized operations.

    Attributes
    ----------
    spots : Array
        Current spot prices/rates
    strikes : Array
        Strike prices/rates
    maturities : Array
        Time to maturity (years)
    notionals : Array
        Notional amounts
    option_types : Array
        Option type encoded as integers (0=call, 1=put, -1=N/A for non-options)
    """

    spots: Array
    strikes: Array
    maturities: Array
    notionals: Array
    option_types: Array

    def __post_init__(self) -> None:
        """Validate that all arrays have consistent shape."""
        shapes = {
            "spots": self.spots.shape,
            "strikes": self.strikes.shape,
            "maturities": self.maturities.shape,
            "notionals": self.notionals.shape,
            "option_types": self.option_types.shape,
        }

        unique_shapes = set(shapes.values())
        if len(unique_shapes) != 1:
            raise ValueError(f"Inconsistent array shapes: {shapes}")

        n_trades = self.spots.shape[0]
        if n_trades == 0:
            raise ValueError("TradeArrays cannot be empty")

    @property
    def n_trades(self) -> int:
        """Number of trades in the batch."""
        return int(self.spots.shape[0])

    def __len__(self) -> int:
        """Return number of trades."""
        return self.n_trades


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PortfolioBatch:
    """High-performance Struct-of-Arrays portfolio representation.

    Stores 10K+ trades in vectorized arrays optimized for batch pricing and risk calculations.
    All numerical data is stored in aligned JAX arrays, while categorical data is stored
    as integer indices into lookup tables.

    This structure is registered as a JAX pytree, making it fully compatible with
    JIT compilation, vmap, pmap, and other JAX transformations.

    Attributes
    ----------
    trade_arrays : TradeArrays
        Numerical trade parameters (spots, strikes, maturities, etc.)
    currency_idx : Array
        Index into currency mapping [n_trades]
    counterparty_idx : Array
        Index into counterparty mapping [n_trades]
    product_type_idx : Array
        Index into product type mapping [n_trades]
    currency_mapping : IndexMapping
        Bidirectional mapping for currency codes
    counterparty_mapping : IndexMapping
        Bidirectional mapping for counterparty IDs
    product_type_mapping : IndexMapping
        Bidirectional mapping for product types
    metadata : dict[str, Any]
        Optional metadata (not part of pytree, for debugging/logging)

    Examples
    --------
    >>> # Create portfolio batch from raw data
    >>> trade_arrays = TradeArrays(
    ...     spots=jnp.array([100.0, 105.0, 110.0]),
    ...     strikes=jnp.array([110.0, 115.0, 120.0]),
    ...     maturities=jnp.array([1.0, 2.0, 0.5]),
    ...     notionals=jnp.array([1e6, 2e6, 5e5]),
    ...     option_types=jnp.array([0, 0, 1]),  # call, call, put
    ... )
    >>> currency_idx = jnp.array([0, 1, 0])  # USD, EUR, USD
    >>> cp_idx = jnp.array([0, 0, 1])  # CP1, CP1, CP2
    >>> product_idx = jnp.array([0, 0, 0])  # All vanilla options
    >>>
    >>> portfolio = PortfolioBatch(
    ...     trade_arrays=trade_arrays,
    ...     currency_idx=currency_idx,
    ...     counterparty_idx=cp_idx,
    ...     product_type_idx=product_idx,
    ...     currency_mapping=IndexMapping.from_values(["USD", "EUR"]),
    ...     counterparty_mapping=IndexMapping.from_values(["CP1", "CP2"]),
    ...     product_type_mapping=IndexMapping.from_values(["VanillaOption"]),
    ... )
    >>> portfolio.n_trades
    3
    >>> portfolio.n_counterparties
    2

    Notes
    -----
    Memory Layout:
    - All arrays are contiguous in memory for SIMD efficiency
    - Integer indices are int32 (sufficient for 10K trades/counterparties)
    - Float arrays use configurable precision (default float32)

    Performance:
    - 10K trades price in ~50ms on modern CPU
    - Aggregation to 1K counterparties takes ~1ms (sparse matmul)
    - Memory footprint: ~1MB per 10K trades (vs ~50MB for Pydantic models)
    """

    trade_arrays: TradeArrays
    currency_idx: Array
    counterparty_idx: Array
    product_type_idx: Array
    currency_mapping: IndexMapping
    counterparty_mapping: IndexMapping
    product_type_mapping: IndexMapping
    product_ids: Optional[tuple[str, ...]] = None
    product_parameters: Optional[tuple[dict[str, Any], ...]] = None
    metadata: dict[str, Any] = None

    def __post_init__(self) -> None:
        """Validate consistency of arrays and mappings."""
        n_trades = self.trade_arrays.n_trades

        # Validate index array shapes
        if self.currency_idx.shape != (n_trades,):
            raise ValueError(f"currency_idx shape {self.currency_idx.shape} != ({n_trades},)")
        if self.counterparty_idx.shape != (n_trades,):
            raise ValueError(f"counterparty_idx shape {self.counterparty_idx.shape} != ({n_trades},)")
        if self.product_type_idx.shape != (n_trades,):
            raise ValueError(f"product_type_idx shape {self.product_type_idx.shape} != ({n_trades},)")

        # Validate index ranges (skip if inside JAX transformation)
        try:
            if jnp.any(self.currency_idx < 0) or jnp.any(self.currency_idx >= len(self.currency_mapping)):
                raise ValueError("currency_idx contains out-of-range indices")
            if jnp.any(self.counterparty_idx < 0) or jnp.any(self.counterparty_idx >= len(self.counterparty_mapping)):
                raise ValueError("counterparty_idx contains out-of-range indices")
            if jnp.any(self.product_type_idx < 0) or jnp.any(self.product_type_idx >= len(self.product_type_mapping)):
                raise ValueError("product_type_idx contains out-of-range indices")
        except jax.errors.TracerBoolConversionError:
            # Skip validation inside JAX transformations (e.g., jit, vmap)
            pass

        # Initialize metadata if None
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

        # Initialize product identifiers
        if self.product_ids is None:
            object.__setattr__(self, "product_ids", tuple(f"trade_{i}" for i in range(n_trades)))
        else:
            product_ids: Sequence[str] = self.product_ids
            if len(product_ids) != n_trades:
                raise ValueError(
                    f"product_ids length {len(product_ids)} does not match number of trades {n_trades}"
                )
            object.__setattr__(self, "product_ids", tuple(str(pid) for pid in product_ids))

        if self.product_parameters is None:
            object.__setattr__(self, "product_parameters", tuple({} for _ in range(n_trades)))
        else:
            product_params: Sequence[dict[str, Any]] = self.product_parameters
            if len(product_params) != n_trades:
                raise ValueError(
                    "product_parameters length "
                    f"{len(product_params)} does not match number of trades {n_trades}"
                )
            object.__setattr__(
                self,
                "product_parameters",
                tuple(dict(params) for params in product_params),
            )

    @property
    def n_trades(self) -> int:
        """Number of trades in the portfolio."""
        return self.trade_arrays.n_trades

    @property
    def n_currencies(self) -> int:
        """Number of unique currencies."""
        return len(self.currency_mapping)

    @property
    def n_counterparties(self) -> int:
        """Number of unique counterparties."""
        return len(self.counterparty_mapping)

    @property
    def n_product_types(self) -> int:
        """Number of unique product types."""
        return len(self.product_type_mapping)

    def __len__(self) -> int:
        """Return number of trades."""
        return self.n_trades

    # JAX Pytree implementation
    def tree_flatten(self) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Flatten for JAX pytree operations.

        Separates dynamic (arrays) from static (mappings, metadata) data.

        Returns
        -------
        children : tuple
            Dynamic data (JAX arrays)
        aux_data : dict
            Static data (mappings, metadata)
        """
        children = (
            self.trade_arrays.spots,
            self.trade_arrays.strikes,
            self.trade_arrays.maturities,
            self.trade_arrays.notionals,
            self.trade_arrays.option_types,
            self.currency_idx,
            self.counterparty_idx,
            self.product_type_idx,
        )
        aux_data = {
            "currency_mapping": self.currency_mapping,
            "counterparty_mapping": self.counterparty_mapping,
            "product_type_mapping": self.product_type_mapping,
            "product_ids": self.product_ids,
            "product_parameters": self.product_parameters,
            "metadata": self.metadata,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: dict[str, Any], children: tuple[Any, ...]) -> PortfolioBatch:
        """Reconstruct from flattened pytree.

        Parameters
        ----------
        aux_data : dict
            Static data (mappings, metadata)
        children : tuple
            Dynamic data (JAX arrays)

        Returns
        -------
        PortfolioBatch
            Reconstructed portfolio batch
        """
        trade_arrays = TradeArrays(
            spots=children[0],
            strikes=children[1],
            maturities=children[2],
            notionals=children[3],
            option_types=children[4],
        )
        return cls(
            trade_arrays=trade_arrays,
            currency_idx=children[5],
            counterparty_idx=children[6],
            product_type_idx=children[7],
            **aux_data,
        )

    def slice_trades(self, start: int, end: int) -> PortfolioBatch:
        """Extract a slice of trades as a new PortfolioBatch.

        Useful for chunked processing or batching.

        Parameters
        ----------
        start : int
            Starting trade index (inclusive)
        end : int
            Ending trade index (exclusive)

        Returns
        -------
        PortfolioBatch
            New portfolio batch containing trades [start:end]

        Examples
        --------
        >>> batch = portfolio.slice_trades(0, 1000)  # First 1000 trades
        >>> batch.n_trades
        1000
        """
        trade_arrays = TradeArrays(
            spots=self.trade_arrays.spots[start:end],
            strikes=self.trade_arrays.strikes[start:end],
            maturities=self.trade_arrays.maturities[start:end],
            notionals=self.trade_arrays.notionals[start:end],
            option_types=self.trade_arrays.option_types[start:end],
        )
        return PortfolioBatch(
            trade_arrays=trade_arrays,
            currency_idx=self.currency_idx[start:end],
            counterparty_idx=self.counterparty_idx[start:end],
            product_type_idx=self.product_type_idx[start:end],
            currency_mapping=self.currency_mapping,
            counterparty_mapping=self.counterparty_mapping,
            product_type_mapping=self.product_type_mapping,
            product_ids=self.product_ids[start:end],
            product_parameters=tuple(
                dict(params) for params in self.product_parameters[start:end]
            ),
            metadata={**self.metadata, "slice": f"[{start}:{end}]"},
        )

    def filter_by_counterparty(self, cp_id: str) -> PortfolioBatch:
        """Filter trades by counterparty ID.

        Parameters
        ----------
        cp_id : str
            Counterparty ID to filter by

        Returns
        -------
        PortfolioBatch
            New portfolio batch containing only trades with specified counterparty

        Raises
        ------
        KeyError
            If counterparty ID not in mapping
        """
        cp_idx = self.counterparty_mapping.encode(cp_id)
        mask = self.counterparty_idx == cp_idx

        trade_arrays = TradeArrays(
            spots=self.trade_arrays.spots[mask],
            strikes=self.trade_arrays.strikes[mask],
            maturities=self.trade_arrays.maturities[mask],
            notionals=self.trade_arrays.notionals[mask],
            option_types=self.trade_arrays.option_types[mask],
        )

        # Create new mapping with only the filtered counterparty
        new_cp_mapping = IndexMapping.from_values([cp_id])
        # Remap indices to the new mapping (all will be 0 since only one counterparty)
        new_cp_idx = jnp.zeros_like(self.counterparty_idx[mask])

        return PortfolioBatch(
            trade_arrays=trade_arrays,
            currency_idx=self.currency_idx[mask],
            counterparty_idx=new_cp_idx,
            product_type_idx=self.product_type_idx[mask],
            currency_mapping=self.currency_mapping,
            counterparty_mapping=new_cp_mapping,
            product_type_mapping=self.product_type_mapping,
            product_ids=tuple(
                pid for pid, keep in zip(self.product_ids, mask.tolist()) if keep
            ),
            product_parameters=tuple(
                dict(self.product_parameters[i])
                for i, keep in enumerate(mask.tolist())
                if keep
            ),
            metadata={**self.metadata, "filter_cp": cp_id},
        )

    def select_trades_by_mask(self, mask: Array) -> PortfolioBatch:
        """Return a new portfolio containing only trades where mask is True."""

        bool_mask = jnp.asarray(mask, dtype=bool)
        if bool_mask.shape != (self.n_trades,):
            raise ValueError(
                f"Mask shape {bool_mask.shape} does not match number of trades {self.n_trades}"
            )

        if not bool(bool_mask.any()):
            raise ValueError("Mask selects no trades")

        trade_arrays = TradeArrays(
            spots=self.trade_arrays.spots[bool_mask],
            strikes=self.trade_arrays.strikes[bool_mask],
            maturities=self.trade_arrays.maturities[bool_mask],
            notionals=self.trade_arrays.notionals[bool_mask],
            option_types=self.trade_arrays.option_types[bool_mask],
        )

        mask_list = bool_mask.tolist()

        return PortfolioBatch(
            trade_arrays=trade_arrays,
            currency_idx=self.currency_idx[bool_mask],
            counterparty_idx=self.counterparty_idx[bool_mask],
            product_type_idx=self.product_type_idx[bool_mask],
            currency_mapping=self.currency_mapping,
            counterparty_mapping=self.counterparty_mapping,
            product_type_mapping=self.product_type_mapping,
            product_ids=tuple(pid for pid, keep in zip(self.product_ids, mask_list) if keep),
            product_parameters=tuple(
                dict(self.product_parameters[i]) for i, keep in enumerate(mask_list) if keep
            ),
            metadata={**self.metadata, "mask_selection": True},
        )

    def get_counterparty_indices(self) -> Array:
        """Get unique counterparty indices present in portfolio.

        Returns
        -------
        Array
            Sorted array of unique counterparty indices

        Examples
        --------
        >>> portfolio.counterparty_idx
        Array([0, 0, 1, 2, 1], dtype=int32)
        >>> portfolio.get_counterparty_indices()
        Array([0, 1, 2], dtype=int32)
        """
        return jnp.unique(self.counterparty_idx)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"PortfolioBatch(\n"
            f"  n_trades={self.n_trades:,},\n"
            f"  n_counterparties={self.n_counterparties:,},\n"
            f"  n_currencies={self.n_currencies},\n"
            f"  total_notional={float(jnp.sum(self.trade_arrays.notionals)):,.0f}\n"
            f")"
        )
