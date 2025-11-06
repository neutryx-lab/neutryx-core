"""Example of chunked Monte Carlo simulation for large-scale path generation.

This example demonstrates:
1. Simulating 100K paths without memory overflow
2. Streaming results to storage
3. Memory-efficient option pricing from chunks
"""

import jax
import jax.numpy as jnp

from neutryx.data import (
    ChunkedSimConfig,
    estimate_optimal_chunk_size,
    price_option_chunked,
    simulate_gbm_chunked,
)
from neutryx.io import create_cache_store


def call_payoff(path: jnp.ndarray) -> float:
    """Call option payoff function.

    Parameters
    ----------
    path : Array
        Simulated price path [steps+1]

    Returns
    -------
    float
        Payoff (max(S_T - K, 0))
    """
    K = 100.0  # Strike
    S_T = path[-1]  # Terminal price
    return jnp.maximum(S_T - K, 0.0)


def main():
    """Run chunked simulation example."""
    print("=" * 80)
    print("Chunked Monte Carlo Simulation Example")
    print("=" * 80)

    # Parameters
    S0 = 100.0  # Initial spot
    mu = 0.05  # Drift (r - q)
    sigma = 0.20  # Volatility
    T = 1.0  # Maturity
    r = 0.05  # Risk-free rate

    # Step 1: Estimate optimal chunk size
    print("\n1. Estimating optimal chunk size...")
    total_paths = 100_000
    steps = 600
    available_memory_gb = 4.0  # Assume 4GB available

    optimal_chunk = estimate_optimal_chunk_size(
        total_paths=total_paths,
        steps=steps,
        available_memory_gb=available_memory_gb,
        dtype="float32",
        safety_factor=0.5,
    )

    print(f"   ✓ Optimal chunk size: {optimal_chunk:,} paths")
    print(f"     - Total paths: {total_paths:,}")
    print(f"     - Time steps: {steps}")
    print(f"     - Available memory: {available_memory_gb:.1f} GB")

    # Step 2: Configure chunked simulation
    print("\n2. Configuring chunked simulation...")
    config = ChunkedSimConfig(
        total_paths=total_paths,
        chunk_size=10_000,  # 10K paths per chunk
        steps=steps,
        dtype="float32",
        stream_to_storage=True,
        storage_key_prefix="sim_chunk",
    )

    print(f"   ✓ Configuration created")
    print(f"     - Number of chunks: {config.n_chunks}")
    print(f"     - Memory per chunk: {config.memory_per_chunk_mb:.1f} MB")
    print(f"     - Total memory (if not chunked): {config.total_memory_mb:.1f} MB")
    print(f"     - Memory savings: {(1 - config.memory_per_chunk_mb / config.total_memory_mb):.1%}")

    # Step 3: Create storage backend
    print("\n3. Creating storage backend...")
    store = create_cache_store("/tmp/neutryx_sim_example")
    print(f"   ✓ Cache store created at /tmp/neutryx_sim_example")

    # Step 4: Run chunked simulation
    print("\n4. Running chunked simulation...")

    def progress_callback(chunk_idx, n_chunks):
        """Print progress."""
        pct = chunk_idx / n_chunks * 100
        print(f"   Progress: [{chunk_idx}/{n_chunks}] {pct:.0f}%", end="\r")

    key = jax.random.PRNGKey(42)

    import time

    start = time.perf_counter()

    simulate_gbm_chunked(
        key=key,
        S0=S0,
        mu=mu,
        sigma=sigma,
        T=T,
        config=config,
        store=store,
        progress_callback=progress_callback,
    )

    elapsed = time.perf_counter() - start
    print(f"\n   ✓ Simulation complete in {elapsed:.2f}s")
    print(f"     - Throughput: {total_paths / elapsed:.0f} paths/second")

    # Step 5: Price option from chunks
    print("\n5. Pricing call option from stored chunks...")
    discount_factor = jnp.exp(-r * T)

    start = time.perf_counter()

    option_price = price_option_chunked(
        store=store,
        key_prefix="sim_chunk",
        n_chunks=config.n_chunks,
        payoff_fn=call_payoff,
        discount_factor=float(discount_factor),
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"   ✓ Option priced in {elapsed_ms:.1f}ms")
    print(f"     - Call price: ${option_price:.4f}")
    print(f"     - Parameters: S0={S0}, K=100, T={T}, r={r}, σ={sigma}")

    # Step 6: Cleanup
    print("\n6. Cleaning up storage...")
    store.clear()
    store.close()
    print(f"   ✓ Storage cleared")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("\n💡 Key Takeaways:")
    print("   - Chunking enables 100K+ paths without memory overflow")
    print("   - Streaming to storage allows unlimited scale")
    print(f"   - Memory footprint: {config.memory_per_chunk_mb:.0f}MB vs {config.total_memory_mb:.0f}MB (unchunked)")
    print("=" * 80)


if __name__ == "__main__":
    main()
