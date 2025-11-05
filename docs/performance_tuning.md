# Performance Tuning Guide

This guide provides comprehensive strategies for optimizing Neutryx Core performance across different scenarios and hardware configurations.

## Table of Contents

1. [Quick Wins](#quick-wins)
2. [JAX Optimization](#jax-optimization)
3. [Monte Carlo Optimization](#monte-carlo-optimization)
4. [GPU Acceleration](#gpu-acceleration)
5. [Memory Optimization](#memory-optimization)
6. [Distributed Computing](#distributed-computing)
7. [Profiling and Benchmarking](#profiling-and-benchmarking)
8. [Production Optimization](#production-optimization)

## Quick Wins

### 1. Enable JIT Compilation

Always use `@jax.jit` for frequently called functions:

```python
import jax
import jax.numpy as jnp
from neutryx.models.bs import price

# Bad - No JIT compilation
def price_portfolio(spots, strikes, maturities):
    return [price(S, K, T, 0.05, 0.02, 0.20, "call")
            for S, K, T in zip(spots, strikes, maturities)]

# Good - JIT compiled with vmap
@jax.jit
def price_portfolio_optimized(spots, strikes, maturities):
    return jax.vmap(lambda S, K, T: price(S, K, T, 0.05, 0.02, 0.20, "call"))(
        spots, strikes, maturities
    )

# Benchmark
spots = jnp.ones(1000) * 100.0
strikes = jnp.ones(1000) * 100.0
maturities = jnp.ones(1000) * 1.0

# First call (includes compilation)
%timeit price_portfolio_optimized(spots, strikes, maturities).block_until_ready()
# Subsequent calls (compiled)
# Result: ~100x faster than uncompiled version
```

**Performance Impact**: 10-100x speedup

### 2. Use Vectorization with `vmap`

Replace loops with vectorized operations:

```python
# Bad - Python loop
def calculate_greeks_loop(spots):
    deltas = []
    for S in spots:
        delta = jax.grad(lambda x: price(x, 100, 1.0, 0.05, 0.02, 0.20, "call"))(S)
        deltas.append(delta)
    return jnp.array(deltas)

# Good - Vectorized with vmap
@jax.jit
def calculate_greeks_vmap(spots):
    price_fn = lambda S: price(S, 100, 1.0, 0.05, 0.02, 0.20, "call")
    return jax.vmap(jax.grad(price_fn))(spots)

# Benchmark
spots = jnp.linspace(80, 120, 1000)
%timeit calculate_greeks_vmap(spots).block_until_ready()
# Result: ~50x faster than loop version
```

**Performance Impact**: 10-50x speedup

### 3. Batch Operations

Process multiple items simultaneously:

```python
# Bad - One at a time
prices = []
for option in options:
    price = pricing_engine.price(option)
    prices.append(price)

# Good - Batch processing
prices = pricing_engine.price_batch(options)

# Implementation
@jax.jit
def price_batch(spots, strikes, maturities, config):
    """Batch price multiple options"""
    return jax.vmap(
        lambda S, K, T: price(S, K, T, config.r, config.q, config.sigma, "call")
    )(spots, strikes, maturities)
```

**Performance Impact**: 5-20x speedup

### 4. Use Lower Precision When Appropriate

Float32 is significantly faster than float64:

```python
# Configure JAX to use float32 by default
jax.config.update("jax_enable_x64", False)

# Explicit conversion
spots_f32 = spots.astype(jnp.float32)
prices_f32 = price_batch(spots_f32, strikes_f32, maturities_f32, config)

# Memory usage: 50% reduction
# Speed: 2-3x faster on GPU
```

**Performance Impact**: 2-3x speedup, 50% memory reduction

## JAX Optimization

### JIT Compilation Best Practices

#### 1. Mark Static Arguments

Use `static_argnums` for arguments that won't change:

```python
@partial(jax.jit, static_argnums=(4,))  # option_type is static
def price_option(S, K, T, r, option_type):
    if option_type == "call":
        return price_call(S, K, T, r)
    else:
        return price_put(S, K, T, r)
```

#### 2. Avoid Python Control Flow

Use JAX control flow primitives:

```python
# Bad - Python if statement
def price(S, K, T, option_type):
    if option_type == "call":  # Causes recompilation
        return jnp.maximum(S - K, 0)
    else:
        return jnp.maximum(K - S, 0)

# Good - JAX where
@jax.jit
def price_optimized(S, K, T, is_call):
    call_payoff = jnp.maximum(S - K, 0)
    put_payoff = jnp.maximum(K - S, 0)
    return jax.lax.cond(is_call, lambda: call_payoff, lambda: put_payoff)

# Even better - Pure JAX operations
@jax.jit
def price_best(S, K, T, is_call):
    payoff = jnp.maximum(S - K, 0) * is_call + jnp.maximum(K - S, 0) * (1 - is_call)
    return payoff
```

#### 3. Minimize Data Transfer

Keep computations on device:

```python
# Bad - Transfer between CPU and GPU
for i in range(iterations):
    x = jnp.array([...])  # CPU -> GPU
    y = compute(x)
    result = np.array(y)  # GPU -> CPU
    print(result)

# Good - Keep on device
x = jnp.array([...])  # Single transfer
for i in range(iterations):
    x = compute(x)  # Stays on GPU
result = np.array(x)  # Single transfer back
```

### Automatic Differentiation Optimization

#### Forward vs. Reverse Mode

```python
# Reverse mode (adjoint) - efficient for many inputs, few outputs
# Use for: Greeks calculation (∇f where f: R^n → R)
@jax.jit
def calculate_delta_gamma(S, K, T, r, sigma):
    price_fn = lambda S: bs_price(S, K, T, r, 0.0, sigma, "call")
    delta = jax.grad(price_fn)(S)
    gamma = jax.grad(jax.grad(price_fn))(S)
    return delta, gamma

# Forward mode - efficient for few inputs, many outputs
# Use for: Jacobian calculation (Jf where f: R → R^n)
@jax.jit
def calculate_price_surface(S, strikes, maturities):
    price_fn = lambda K, T: bs_price(S, K, T, 0.05, 0.02, 0.20, "call")
    return jax.jacfwd(price_fn)(strikes, maturities)
```

#### Efficient Hessian Calculation

```python
# Bad - Nested grad (slow)
def hessian_slow(f, x):
    return jax.grad(jax.grad(f))(x)

# Good - Use hessian function
def hessian_fast(f, x):
    return jax.hessian(f)(x)

# Even better - Use Hessian-vector product for large problems
def hvp(f, x, v):
    """Hessian-vector product without computing full Hessian"""
    return jax.grad(lambda x: jnp.vdot(jax.grad(f)(x), v))(x)
```

## Monte Carlo Optimization

### Variance Reduction Techniques

#### 1. Antithetic Variates

```python
@jax.jit
def monte_carlo_antithetic(key, S0, K, T, r, sigma, n_paths):
    """Monte Carlo with antithetic variates"""
    # Generate standard normal samples
    key1, key2 = jax.random.split(key)
    Z = jax.random.normal(key1, (n_paths // 2,))

    # Antithetic pairs
    dt = T
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * jnp.sqrt(dt)

    S1 = S0 * jnp.exp(drift + diffusion * Z)
    S2 = S0 * jnp.exp(drift - diffusion * Z)  # Antithetic

    # Combined payoff
    payoff1 = jnp.maximum(S1 - K, 0)
    payoff2 = jnp.maximum(S2 - K, 0)
    payoff = jnp.concatenate([payoff1, payoff2])

    return jnp.mean(payoff) * jnp.exp(-r * T)

# Variance reduction: ~2x (variance halved)
```

#### 2. Control Variates

```python
@jax.jit
def monte_carlo_control_variate(key, S0, K, T, r, sigma, n_paths):
    """Monte Carlo with control variate (geometric average)"""
    paths = simulate_gbm(key, S0, r, sigma, T, n_paths, steps=252)

    # Arithmetic average (target)
    arithmetic_avg = jnp.mean(paths, axis=1)
    target_payoff = jnp.maximum(arithmetic_avg - K, 0)

    # Geometric average (control variate - has closed form)
    geometric_avg = jnp.exp(jnp.mean(jnp.log(paths), axis=1))
    control_payoff = jnp.maximum(geometric_avg - K, 0)

    # Analytical value of geometric asian
    analytical_control = geometric_asian_price(S0, K, T, r, sigma)

    # Control variate adjustment
    c = -jnp.cov(target_payoff, control_payoff)[0, 1] / jnp.var(control_payoff)
    adjusted_payoff = target_payoff + c * (control_payoff - analytical_control)

    return jnp.mean(adjusted_payoff) * jnp.exp(-r * T)

# Variance reduction: ~5-10x
```

#### 3. Quasi-Monte Carlo (Low-Discrepancy Sequences)

```python
from scipy.stats import qmc

def quasi_monte_carlo(S0, K, T, r, sigma, n_paths):
    """Quasi-Monte Carlo with Sobol sequences"""
    # Generate Sobol sequence
    sampler = qmc.Sobol(d=1, scramble=True)
    uniform = sampler.random(n_paths)

    # Convert to normal via inverse CDF
    from scipy.stats import norm
    Z = norm.ppf(uniform).flatten()

    # Price path
    dt = T
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * jnp.sqrt(dt)
    ST = S0 * jnp.exp(drift + diffusion * Z)

    # Payoff
    payoff = jnp.maximum(ST - K, 0)
    return jnp.mean(payoff) * jnp.exp(-r * T)

# Convergence: O(1/N) vs O(1/√N) for standard MC
```

### Optimized Path Generation

```python
@jax.jit
def simulate_gbm_optimized(key, S0, mu, sigma, T, n_paths, steps):
    """Optimized GBM simulation"""
    dt = T / steps

    # Pre-compute constants
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * jnp.sqrt(dt)

    # Generate all random numbers at once
    dW = jax.random.normal(key, (n_paths, steps)) * diffusion

    # Cumulative sum for log-price
    log_returns = drift + dW
    log_S = jnp.cumsum(log_returns, axis=1)

    # Convert to price
    paths = S0 * jnp.exp(log_S)

    return paths

# ~3x faster than iterative approach
```

## GPU Acceleration

### Device Selection

```python
# Check available devices
devices = jax.devices()
print(f"Available devices: {devices}")

# Force GPU
jax.config.update('jax_platform_name', 'gpu')

# Manual device placement
from jax import device_put

# Transfer to specific GPU
x_gpu = device_put(x, devices[0])
y_gpu = compute(x_gpu)
```

### Multi-GPU with `pmap`

```python
@jax.pmap
def price_on_multiple_gpus(keys, spots, strikes, maturities):
    """Distribute computation across GPUs"""
    return jax.vmap(lambda S, K, T: price(S, K, T, 0.05, 0.02, 0.20, "call"))(
        spots, strikes, maturities
    )

# Split data across devices
n_devices = jax.device_count()
spots_per_device = jnp.array_split(spots, n_devices)
strikes_per_device = jnp.array_split(strikes, n_devices)
maturities_per_device = jnp.array_split(maturities, n_devices)

# Generate keys for each device
keys = jax.random.split(jax.random.PRNGKey(42), n_devices)

# Execute on all GPUs
prices = price_on_multiple_gpus(
    keys,
    jnp.array(spots_per_device),
    jnp.array(strikes_per_device),
    jnp.array(maturities_per_device)
)

# Flatten results
all_prices = prices.flatten()
```

### GPU Memory Management

```python
# Clear GPU cache
from jax.lib import xla_bridge
xla_bridge.get_backend().clear_cache()

# Process in chunks to avoid OOM
def process_large_batch(data, chunk_size=10000):
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        result = process_chunk(chunk)
        results.append(result)
        # Clear intermediate results
        del result
    return jnp.concatenate(results)
```

## Memory Optimization

### Efficient Array Operations

```python
# Bad - Creates intermediate arrays
def bad_calculation(x, y, z):
    temp1 = x * y
    temp2 = temp1 + z
    temp3 = jnp.sqrt(temp2)
    return temp3

# Good - Fused operations
@jax.jit
def good_calculation(x, y, z):
    return jnp.sqrt(x * y + z)  # Single fused kernel

# Memory savings: 2x less memory, 3x faster
```

### Gradient Checkpointing

```python
from jax.checkpoint import checkpoint

@checkpoint
def expensive_forward_pass(x):
    """Checkpoint to trade compute for memory"""
    # Instead of storing all intermediate activations,
    # recompute them during backward pass
    return complex_computation(x)

# Memory usage: O(√N) instead of O(N)
# Compute time: ~1.3x slower
```

### Memory-Efficient Monte Carlo

```python
def memory_efficient_mc(key, S0, K, T, r, sigma, total_paths, batch_size=10000):
    """Process Monte Carlo in batches to save memory"""
    n_batches = total_paths // batch_size
    prices = []

    for i in range(n_batches):
        key, subkey = jax.random.split(key)
        paths = simulate_gbm(subkey, S0, r, sigma, T, batch_size, 252)
        ST = paths[:, -1]
        payoff = jnp.maximum(ST - K, 0)
        price = jnp.mean(payoff)
        prices.append(price)

    return jnp.mean(jnp.array(prices)) * jnp.exp(-r * T)
```

## Distributed Computing

### Cluster Setup with Ray

```python
import ray

ray.init(address="auto")  # Connect to cluster

@ray.remote
def price_portfolio_chunk(options_chunk, market_data):
    """Price a chunk of portfolio on remote worker"""
    from neutryx.models.bs import price
    return [price(opt, market_data) for opt in options_chunk]

# Distribute work
portfolio_chunks = np.array_split(portfolio, 100)
futures = [price_portfolio_chunk.remote(chunk, market_data)
           for chunk in portfolio_chunks]

# Collect results
prices = ray.get(futures)
all_prices = np.concatenate(prices)
```

### Dask for Large-Scale Computation

```python
import dask.array as da
from dask.distributed import Client

client = Client()

# Create large array distributed across cluster
spots = da.from_delayed([...], shape=(1_000_000,), dtype=float)
strikes = da.from_delayed([...], shape=(1_000_000,), dtype=float)

# Compute in parallel
def price_chunk(S, K):
    from neutryx.models.bs import price
    return price(S, K, 1.0, 0.05, 0.02, 0.20, "call")

prices = da.map_blocks(price_chunk, spots, strikes, dtype=float)
result = prices.compute()
```

## Profiling and Benchmarking

### JAX Profiling

```python
# Enable profiling
with jax.profiler.trace("/tmp/jax-trace", create_perfetto_trace=True):
    result = expensive_computation(data)

# View trace at https://ui.perfetto.dev
```

### Python Profiling

```python
import cProfile
import pstats

# Profile code
profiler = cProfile.Profile()
profiler.enable()

result = price_portfolio(portfolio)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumtime')
stats.print_stats(20)
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    large_array = jnp.zeros((10000, 10000))
    result = compute(large_array)
    return result
```

### Benchmarking Utilities

```python
import time
import jax

def benchmark(fn, *args, n_iterations=100, warmup=10):
    """Benchmark a JAX function"""
    # Warmup (includes compilation)
    for _ in range(warmup):
        result = fn(*args)
        result.block_until_ready()

    # Actual benchmark
    times = []
    for _ in range(n_iterations):
        start = time.time()
        result = fn(*args)
        result.block_until_ready()
        times.append(time.time() - start)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }

# Usage
stats = benchmark(price_portfolio_optimized, spots, strikes, maturities)
print(f"Mean: {stats['mean']*1000:.2f}ms ± {stats['std']*1000:.2f}ms")
```

## Production Optimization

### Connection Pooling

```python
from contextlib import asynccontextmanager
import asyncpg

class DatabasePool:
    def __init__(self, dsn, min_size=10, max_size=20):
        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size
        self.pool = None

    async def initialize(self):
        self.pool = await asyncpg.create_pool(
            self.dsn,
            min_size=self.min_size,
            max_size=self.max_size
        )

    @asynccontextmanager
    async def acquire(self):
        async with self.pool.acquire() as connection:
            yield connection
```

### Caching Strategy

```python
from functools import lru_cache
import redis

# In-memory cache
@lru_cache(maxsize=1000)
def get_market_data(date, instrument):
    return fetch_from_database(date, instrument)

# Distributed cache
class RedisCache:
    def __init__(self, redis_client):
        self.redis = redis_client

    async def get_or_compute(self, key, compute_fn, ttl=3600):
        # Try cache first
        cached = await self.redis.get(key)
        if cached:
            return pickle.loads(cached)

        # Compute and cache
        value = await compute_fn()
        await self.redis.setex(key, ttl, pickle.dumps(value))
        return value
```

### Batch API Requests

```python
from fastapi import FastAPI
from typing import List

app = FastAPI()

@app.post("/price/batch")
async def price_batch(requests: List[PricingRequest]):
    """Batch endpoint for multiple pricing requests"""
    # Extract parameters
    spots = jnp.array([r.spot for r in requests])
    strikes = jnp.array([r.strike for r in requests])
    maturities = jnp.array([r.maturity for r in requests])

    # Batch pricing
    prices = price_portfolio_optimized(spots, strikes, maturities)

    return [{"request_id": r.id, "price": float(p)}
            for r, p in zip(requests, prices)]
```

## Performance Checklist

- [ ] Enable JIT compilation for hot paths
- [ ] Use `vmap` instead of Python loops
- [ ] Batch operations when possible
- [ ] Use float32 for GPU computations
- [ ] Implement variance reduction for MC
- [ ] Profile and identify bottlenecks
- [ ] Optimize data transfers (CPU ↔ GPU)
- [ ] Use connection pooling for databases
- [ ] Implement caching for expensive computations
- [ ] Use batch APIs for multiple requests
- [ ] Monitor memory usage and optimize
- [ ] Consider distributed computing for large workloads

## Common Performance Pitfalls

1. **Not using JIT**: Always use `@jax.jit` for repeated calculations
2. **Python loops**: Replace with `vmap` or `lax.scan`
3. **Unnecessary data transfers**: Keep data on GPU
4. **Recompilation**: Use `static_argnums` appropriately
5. **Float64 on GPU**: Use float32 when precision allows
6. **Ignoring profiling**: Always profile before optimizing

## Next Steps

- [Architecture Guide](architecture.md) - Understand system design
- [Troubleshooting Guide](troubleshooting.md) - Debug performance issues
- [API Reference](api_reference.md) - Complete API documentation
