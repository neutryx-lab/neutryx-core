"""Efficient categorical data indexing for batch operations.

Provides utilities to map categorical data (currency codes, counterparty IDs,
product types) to integer indices for fast array-based lookups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Sequence

import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class IndexMapping:
    """Bidirectional mapping between categorical values and integer indices.

    Used to convert string-based identifiers (e.g., "USD", "COUNTERPARTY_123")
    to integer indices for efficient array operations.

    Attributes
    ----------
    value_to_idx : dict[Hashable, int]
        Map from categorical value to integer index
    idx_to_value : tuple[Hashable, ...]
        Tuple of values indexed by their position

    Examples
    --------
    >>> mapping = IndexMapping.from_values(["USD", "EUR", "GBP"])
    >>> mapping.encode("USD")
    0
    >>> mapping.encode("EUR")
    1
    >>> mapping.decode(jnp.array([0, 2, 1]))
    ['USD', 'GBP', 'EUR']
    >>> mapping.encode_batch(["EUR", "USD", "EUR"])
    Array([1, 0, 1], dtype=int32)
    """

    value_to_idx: dict[Hashable, int]
    idx_to_value: tuple[Hashable, ...]

    def __post_init__(self) -> None:
        """Validate consistency between forward and reverse mappings."""
        if len(self.value_to_idx) != len(self.idx_to_value):
            raise ValueError(
                f"Inconsistent mapping sizes: {len(self.value_to_idx)} != {len(self.idx_to_value)}"
            )

        for idx, value in enumerate(self.idx_to_value):
            if self.value_to_idx.get(value) != idx:
                raise ValueError(f"Inconsistent mapping for {value}: {self.value_to_idx[value]} != {idx}")

    @classmethod
    def from_values(cls, values: Sequence[Hashable]) -> IndexMapping:
        """Create mapping from a sequence of unique categorical values.

        Parameters
        ----------
        values : Sequence[Hashable]
            Sequence of categorical values (will be deduplicated in order of first appearance)

        Returns
        -------
        IndexMapping
            Bidirectional index mapping

        Examples
        --------
        >>> mapping = IndexMapping.from_values(["EUR", "USD", "EUR", "GBP"])
        >>> len(mapping)
        3
        >>> mapping.encode("EUR")
        0
        >>> mapping.encode("USD")
        1
        >>> mapping.encode("GBP")
        2
        """
        # Preserve insertion order while deduplicating
        seen = set()
        unique_values = []
        for val in values:
            if val not in seen:
                seen.add(val)
                unique_values.append(val)

        value_to_idx = {val: idx for idx, val in enumerate(unique_values)}
        idx_to_value = tuple(unique_values)
        return cls(value_to_idx=value_to_idx, idx_to_value=idx_to_value)

    def encode(self, value: Hashable) -> int:
        """Convert a single categorical value to its integer index.

        Parameters
        ----------
        value : Hashable
            Categorical value to encode

        Returns
        -------
        int
            Integer index

        Raises
        ------
        KeyError
            If value is not in the mapping
        """
        return self.value_to_idx[value]

    def encode_batch(self, values: Sequence[Hashable]) -> Array:
        """Convert a sequence of categorical values to integer indices.

        Parameters
        ----------
        values : Sequence[Hashable]
            Sequence of categorical values

        Returns
        -------
        Array
            JAX array of integer indices (int32)

        Examples
        --------
        >>> mapping = IndexMapping.from_values(["A", "B", "C"])
        >>> mapping.encode_batch(["B", "A", "C", "B"])
        Array([1, 0, 2, 1], dtype=int32)  # B=1, A=0, C=2
        """
        indices = [self.value_to_idx[val] for val in values]
        return jnp.array(indices, dtype=jnp.int32)

    def decode(self, indices: Array | Sequence[int]) -> list[Hashable]:
        """Convert integer indices back to categorical values.

        Parameters
        ----------
        indices : Array | Sequence[int]
            Integer indices to decode

        Returns
        -------
        list[Hashable]
            List of categorical values

        Examples
        --------
        >>> mapping = IndexMapping.from_values(["X", "Y", "Z"])
        >>> mapping.decode([2, 0, 1])
        ['Z', 'X', 'Y']  # 0=X, 1=Y, 2=Z
        """
        if isinstance(indices, Array):
            indices = indices.tolist()
        return [self.idx_to_value[int(idx)] for idx in indices]

    def __len__(self) -> int:
        """Return the number of unique categorical values."""
        return len(self.idx_to_value)

    def __contains__(self, value: Hashable) -> bool:
        """Check if a categorical value is in the mapping."""
        return value in self.value_to_idx

    def __repr__(self) -> str:
        """Return string representation."""
        return f"IndexMapping({len(self)} values: {list(self.idx_to_value[:5])}{'...' if len(self) > 5 else ''})"


def build_index_mapping(values: Sequence[Hashable], name: str | None = None) -> IndexMapping:
    """Build an index mapping from a sequence of categorical values.

    Convenience function that wraps IndexMapping.from_values with better error messages.

    Parameters
    ----------
    values : Sequence[Hashable]
        Categorical values to index
    name : str, optional
        Descriptive name for the mapping (used in error messages)

    Returns
    -------
    IndexMapping
        Bidirectional index mapping

    Raises
    ------
    ValueError
        If values sequence is empty

    Examples
    --------
    >>> currencies = ["USD", "EUR", "GBP", "USD", "JPY"]
    >>> mapping = build_index_mapping(currencies, name="currency")
    >>> len(mapping)
    4
    """
    if not values:
        raise ValueError(f"Cannot build index mapping{f' for {name}' if name else ''}: empty values sequence")

    try:
        return IndexMapping.from_values(values)
    except Exception as exc:
        raise ValueError(
            f"Failed to build index mapping{f' for {name}' if name else ''}: {exc}"
        ) from exc


def create_sparse_aggregation_matrix(
    source_indices: Array,
    num_sources: int,
    num_targets: int,
) -> tuple[Array, Array, tuple[int, int]]:
    """Create a sparse aggregation matrix in COO format.

    Creates a sparse matrix that aggregates values from source indices to target indices.
    Useful for counterparty or netting set aggregation.

    Parameters
    ----------
    source_indices : Array
        Integer array mapping each source to a target (shape: [num_sources])
    num_sources : int
        Total number of source items (e.g., trades)
    num_targets : int
        Total number of target items (e.g., counterparties)

    Returns
    -------
    data : Array
        Values for sparse matrix (all ones)
    indices : Array
        COO indices (shape: [num_sources, 2])
    shape : tuple[int, int]
        Shape of the sparse matrix (num_targets, num_sources)

    Examples
    --------
    >>> # 5 trades mapping to 2 counterparties
    >>> trade_to_cp = jnp.array([0, 1, 0, 1, 0])
    >>> data, indices, shape = create_sparse_aggregation_matrix(trade_to_cp, 5, 2)
    >>> shape
    (2, 5)
    >>> # Can be used with jax.experimental.sparse for aggregation
    """
    source_ids = jnp.arange(num_sources, dtype=jnp.int32)

    # COO format: (row_indices, col_indices)
    row_indices = source_indices  # Target indices
    col_indices = source_ids       # Source indices

    indices = jnp.stack([row_indices, col_indices], axis=1)
    data = jnp.ones(num_sources, dtype=jnp.float32)
    shape = (num_targets, num_sources)

    return data, indices, shape
