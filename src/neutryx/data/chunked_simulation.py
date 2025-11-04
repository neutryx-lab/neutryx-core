"""Chunked Monte Carlo simulation for large-scale path generation.

Handles 100K+ paths × 600 steps simulations by chunking the paths dimension
and streaming results to storage, avoiding memory overflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from jax import Array

from neutryx.io.base import DataStore


@dataclass
class ChunkedSimConfig:
    """Configuration for chunked simulation.

    Attributes
    ----------
    total_paths : int
        Total number of paths to simulate
    chunk_size : int
        Number of paths per chunk (balance memory vs overhead)
    steps : int
        Number of time steps
    dtype : str
        Data type for simulation ("float32" or "float64")
    stream_to_storage : bool
        If True, stream chunks directly to storage
    storage_key_prefix : str
        Prefix for storage keys when streaming

    Examples
    --------
    >>> # 100K paths in 10K path chunks
    >>> config = ChunkedSimConfig(
    ...     total_paths=100_000,
    ...     chunk_size=10_000,
    ...     steps=600,
    ...     dtype="float32",
    ... )
    >>> config.n_chunks
    10
    """

    total_paths: int
    chunk_size: int
    steps: int
    dtype: str = "float32"
    stream_to_storage: bool = False
    storage_key_prefix: str = "simulation_chunk"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.total_paths <= 0:
            raise ValueError("total_paths must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_size > self.total_paths:
            raise ValueError("chunk_size cannot exceed total_paths")
        if self.steps <= 0:
            raise ValueError("steps must be positive")

    @property
    def n_chunks(self) -> int:
        """Number of chunks to process."""
        return int(jnp.ceil(self.total_paths / self.chunk_size))

    @property
    def memory_per_chunk_mb(self) -> float:
        """Estimated memory per chunk in MB."""
        dtype_size = 4 if self.dtype == "float32" else 8
        bytes_per_chunk = self.chunk_size * (self.steps + 1) * dtype_size
        return bytes_per_chunk / (1024**2)

    @property
    def total_memory_mb(self) -> float:
        """Estimated total memory if not chunked (MB)."""
        dtype_size = 4 if self.dtype == "float32" else 8
        total_bytes = self.total_paths * (self.steps + 1) * dtype_size
        return total_bytes / (1024**2)


@jax.jit
def _simulate_gbm_chunk(
    key: Array,
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    steps: int,
    paths: int,
    dtype: jnp.dtype,
) -> Array:
    """Simulate a single chunk of GBM paths.

    Parameters
    ----------
    key : Array
        JAX random key
    S0 : float
        Initial spot price
    mu : float
        Drift (risk-free rate - dividend yield)
    sigma : float
        Volatility
    T : float
        Time horizon in years
    steps : int
        Number of time steps
    paths : int
        Number of paths in this chunk
    dtype : jnp.dtype
        Data type

    Returns
    -------
    Array
        Simulated paths [paths, steps+1]
    """
    dt = T / steps
    sqrt_dt = jnp.sqrt(dt)

    # Generate random increments
    dW = jax.random.normal(key, shape=(paths, steps), dtype=dtype) * sqrt_dt

    # Drift adjustment
    drift = (mu - 0.5 * sigma**2) * dt

    # Cumulative sum of log returns
    log_returns = drift + sigma * dW
    log_prices = jnp.cumsum(log_returns, axis=1)

    # Convert to prices
    prices = S0 * jnp.exp(log_prices)

    # Prepend initial value
    initial_prices = jnp.full((paths, 1), S0, dtype=dtype)
    paths_with_initial = jnp.concatenate([initial_prices, prices], axis=1)

    return paths_with_initial


def simulate_gbm_chunked(
    key: Array,
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    config: ChunkedSimConfig,
    store: Optional[DataStore] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Optional[Array]:
    """Simulate GBM paths in chunks to handle large-scale simulations.

    Parameters
    ----------
    key : Array
        JAX random key
    S0 : float
        Initial spot price
    mu : float
        Drift (risk-free rate - dividend yield)
    sigma : float
        Volatility
    T : float
        Time horizon in years
    config : ChunkedSimConfig
        Chunking configuration
    store : DataStore, optional
        Storage backend for streaming chunks (required if stream_to_storage=True)
    progress_callback : callable, optional
        Callback function(chunk_idx, n_chunks) for progress reporting

    Returns
    -------
    Array or None
        If stream_to_storage=False, returns full paths array [total_paths, steps+1].
        If stream_to_storage=True, returns None (chunks saved to store).

    Examples
    --------
    >>> from neutryx.io import create_cache_store
    >>>
    >>> # Simulate 100K paths without storage (fits in memory)
    >>> config = ChunkedSimConfig(total_paths=100_000, chunk_size=10_000, steps=600)
    >>> key = jax.random.PRNGKey(42)
    >>> paths = simulate_gbm_chunked(key, S0=100.0, mu=0.05, sigma=0.2, T=5.0, config=config)
    >>> paths.shape
    (100000, 601)
    >>>
    >>> # Simulate 1M paths with streaming to storage
    >>> store = create_cache_store("/tmp/sim_cache")
    >>> config = ChunkedSimConfig(
    ...     total_paths=1_000_000,
    ...     chunk_size=50_000,
    ...     steps=600,
    ...     stream_to_storage=True,
    ... )
    >>> simulate_gbm_chunked(key, 100.0, 0.05, 0.2, 5.0, config, store=store)
    >>> # Chunks saved to storage as "simulation_chunk_0", "simulation_chunk_1", ...

    Notes
    -----
    Memory Management:
    - Chunk size trades off memory usage vs computation overhead
    - Recommended: 10K-50K paths per chunk for CPU
    - Each chunk is JIT-compiled once and reused

    Performance:
    - 100K paths × 600 steps × float32: ~230MB total, ~23MB per chunk (10K paths)
    - Chunking overhead: ~10-20% vs monolithic (due to loop and key splitting)
    - Streaming to storage: Adds I/O time but enables unlimited scale
    """
    dtype = jnp.float32 if config.dtype == "float32" else jnp.float64

    if config.stream_to_storage and store is None:
        raise ValueError("store must be provided when stream_to_storage=True")

    chunks_list = [] if not config.stream_to_storage else None

    for chunk_idx in range(config.n_chunks):
        # Determine chunk size (last chunk may be smaller)
        start_idx = chunk_idx * config.chunk_size
        end_idx = min(start_idx + config.chunk_size, config.total_paths)
        chunk_paths = end_idx - start_idx

        # Split random key for this chunk
        chunk_key = jax.random.fold_in(key, chunk_idx)

        # Simulate chunk
        chunk = _simulate_gbm_chunk(
            key=chunk_key,
            S0=S0,
            mu=mu,
            sigma=sigma,
            T=T,
            steps=config.steps,
            paths=chunk_paths,
            dtype=dtype,
        )

        # Store or accumulate
        if config.stream_to_storage:
            chunk_key = f"{config.storage_key_prefix}_{chunk_idx}"
            store.save_array(
                chunk_key,
                chunk,
                metadata={
                    "chunk_idx": chunk_idx,
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "S0": float(S0),
                    "mu": float(mu),
                    "sigma": float(sigma),
                    "T": float(T),
                    "steps": config.steps,
                },
            )
        else:
            chunks_list.append(chunk)

        # Progress callback
        if progress_callback is not None:
            progress_callback(chunk_idx + 1, config.n_chunks)

    # Concatenate chunks if not streaming
    if not config.stream_to_storage:
        full_paths = jnp.concatenate(chunks_list, axis=0)
        return full_paths

    return None


def load_chunked_simulation(
    store: DataStore,
    key_prefix: str,
    n_chunks: int,
) -> Array:
    """Load a chunked simulation from storage.

    Parameters
    ----------
    store : DataStore
        Storage backend containing chunks
    key_prefix : str
        Prefix used when saving chunks
    n_chunks : int
        Number of chunks to load

    Returns
    -------
    Array
        Concatenated paths array

    Examples
    --------
    >>> from neutryx.io import create_cache_store
    >>> store = create_cache_store("/tmp/sim_cache")
    >>> paths = load_chunked_simulation(store, "simulation_chunk", n_chunks=10)
    """
    chunks = []
    for chunk_idx in range(n_chunks):
        chunk_key = f"{key_prefix}_{chunk_idx}"
        chunk = store.load_array(chunk_key)
        chunks.append(chunk)

    return jnp.concatenate(chunks, axis=0)


def estimate_optimal_chunk_size(
    total_paths: int,
    steps: int,
    available_memory_gb: float = 4.0,
    dtype: str = "float32",
    safety_factor: float = 0.5,
) -> int:
    """Estimate optimal chunk size based on available memory.

    Parameters
    ----------
    total_paths : int
        Total number of paths to simulate
    steps : int
        Number of time steps
    available_memory_gb : float, optional
        Available memory in GB (default: 4.0)
    dtype : str, optional
        Data type (default: "float32")
    safety_factor : float, optional
        Safety factor (0-1) to avoid OOM (default: 0.5 = use 50% of available)

    Returns
    -------
    int
        Recommended chunk size

    Examples
    --------
    >>> # 4GB RAM available, simulate 100K paths × 600 steps
    >>> chunk_size = estimate_optimal_chunk_size(100_000, 600, available_memory_gb=4.0)
    >>> chunk_size
    20000  # Approximately
    """
    dtype_size = 4 if dtype == "float32" else 8
    bytes_per_path = (steps + 1) * dtype_size

    # Available bytes for paths (after safety factor)
    available_bytes = available_memory_gb * (1024**3) * safety_factor

    # Paths per chunk
    optimal_chunk_size = int(available_bytes / bytes_per_path)

    # Round to nearest 1000 for nice numbers
    optimal_chunk_size = max(1000, (optimal_chunk_size // 1000) * 1000)

    # Don't exceed total paths
    optimal_chunk_size = min(optimal_chunk_size, total_paths)

    return optimal_chunk_size


@jax.jit
def process_chunk_payoff(
    chunk_paths: Array,
    payoff_fn: Callable[[Array], Array],
) -> Array:
    """Apply payoff function to a chunk of paths.

    Parameters
    ----------
    chunk_paths : Array
        Paths for this chunk [chunk_paths, steps+1]
    payoff_fn : callable
        Payoff function applied to each path

    Returns
    -------
    Array
        Payoffs [chunk_paths]

    Notes
    -----
    Uses vmap for efficient vectorization across paths in the chunk.
    """
    return jax.vmap(payoff_fn)(chunk_paths)


def price_option_chunked(
    store: DataStore,
    key_prefix: str,
    n_chunks: int,
    payoff_fn: Callable[[Array], Array],
    discount_factor: float,
) -> float:
    """Price an option from chunked simulation results.

    Loads chunks one at a time, applies payoff, and aggregates.
    Memory-efficient for large simulations.

    Parameters
    ----------
    store : DataStore
        Storage containing path chunks
    key_prefix : str
        Chunk key prefix
    n_chunks : int
        Number of chunks
    payoff_fn : callable
        Payoff function
    discount_factor : float
        Discount factor

    Returns
    -------
    float
        Option price

    Examples
    --------
    >>> def call_payoff(path):
    ...     return jnp.maximum(path[-1] - 100.0, 0.0)
    >>>
    >>> price = price_option_chunked(
    ...     store=store,
    ...     key_prefix="simulation_chunk",
    ...     n_chunks=20,
    ...     payoff_fn=call_payoff,
    ...     discount_factor=0.95,
    ... )
    """
    total_payoff = 0.0
    total_paths = 0

    for chunk_idx in range(n_chunks):
        # Load chunk
        chunk_key = f"{key_prefix}_{chunk_idx}"
        chunk_paths = store.load_array(chunk_key)
        chunk_size = chunk_paths.shape[0]

        # Compute payoffs for this chunk
        chunk_payoffs = process_chunk_payoff(chunk_paths, payoff_fn)

        # Accumulate
        total_payoff += float(jnp.sum(chunk_payoffs))
        total_paths += chunk_size

    # Average and discount
    mean_payoff = total_payoff / total_paths
    price = mean_payoff * discount_factor

    return price
