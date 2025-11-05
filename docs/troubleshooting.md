# Troubleshooting Guide

Common issues and solutions when using Neutryx Core.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [JAX and GPU Issues](#jax-and-gpu-issues)
3. [Pricing and Model Issues](#pricing-and-model-issues)
4. [Performance Issues](#performance-issues)
5. [Memory Issues](#memory-issues)
6. [Market Data Issues](#market-data-issues)
7. [API and Integration Issues](#api-and-integration-issues)
8. [Debugging Tips](#debugging-tips)

## Installation Issues

### Issue: pip install fails with dependency conflicts

**Symptoms**:
```
ERROR: Cannot install neutryx-core because these package versions have conflicting dependencies
```

**Solution**:
```bash
# Create a fresh virtual environment
python -m venv .venv_clean
source .venv_clean/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install neutryx with specific versions
pip install -r requirements.txt --no-cache-dir

# If still failing, try installing dependencies one by one
pip install jax==0.4.26
pip install jaxlib==0.4.26
pip install -e .
```

### Issue: JAX installation fails on specific platform

**Symptoms**:
```
ERROR: Could not find a version that satisfies the requirement jaxlib
```

**Solution**:
```bash
# For CUDA 12.x
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# For CUDA 11.x
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# For CPU only
pip install --upgrade "jax[cpu]"

# For Mac with Apple Silicon
pip install --upgrade jax[metal]
```

### Issue: Import errors after installation

**Symptoms**:
```python
ImportError: cannot import name 'MCConfig' from 'neutryx.core.engine'
```

**Solution**:
```bash
# Ensure editable install
pip install -e .

# Check PYTHONPATH
export PYTHONPATH=/path/to/neutryx-core/src:$PYTHONPATH

# Verify installation
python -c "import neutryx; print(neutryx.__file__)"
```

## JAX and GPU Issues

### Issue: JAX not using GPU

**Symptoms**:
```python
import jax
print(jax.devices())  # Output: [CpuDevice(id=0)]
```

**Diagnosis**:
```python
# Check JAX configuration
print(jax.default_backend())  # Should be 'gpu' or 'cuda'

# Check CUDA availability
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
```

**Solution**:
```bash
# Install CUDA-enabled JAX
pip uninstall jax jaxlib
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Force GPU backend
export JAX_PLATFORM_NAME=gpu

# Verify GPU is detected
nvidia-smi
```

### Issue: CUDA out of memory

**Symptoms**:
```
XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate ...
```

**Solution**:
```python
# Solution 1: Reduce batch size
config = MCConfig(steps=252, paths=50_000)  # Instead of 100_000

# Solution 2: Process in chunks
def process_in_chunks(data, chunk_size=10000):
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        result = process(chunk)
        results.append(result)
    return jnp.concatenate(results)

# Solution 3: Use float32 instead of float64
jax.config.update("jax_enable_x64", False)

# Solution 4: Clear GPU cache
from jax.lib import xla_bridge
xla_bridge.get_backend().clear_cache()

# Solution 5: Limit GPU memory
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'  # Use 80% of GPU memory
```

### Issue: JAX compilation is slow

**Symptoms**:
First call to JIT-compiled function takes very long.

**Solution**:
```python
# Expected behavior - first call includes compilation
@jax.jit
def price_option(S, K, T, r, sigma):
    return bs_price(S, K, T, r, 0.0, sigma, "call")

# First call (slow - includes compilation)
price1 = price_option(100.0, 100.0, 1.0, 0.05, 0.20)

# Subsequent calls (fast - uses compiled version)
price2 = price_option(100.0, 105.0, 1.0, 0.05, 0.20)

# To avoid recompilation, use static arguments
@partial(jax.jit, static_argnums=(5,))
def price_option_static(S, K, T, r, sigma, option_type):
    # option_type won't trigger recompilation
    ...

# Pre-compile with dummy data
_ = price_option(100.0, 100.0, 1.0, 0.05, 0.20)
```

## Pricing and Model Issues

### Issue: Prices don't match benchmarks

**Symptoms**:
Calculated prices differ from expected values.

**Diagnosis**:
```python
from neutryx.models.bs import price

# Check parameters
S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.20
call_price = price(S, K, T, r, q, sigma, "call")
print(f"Call price: {call_price}")

# Verify against Black-Scholes formula manually
from scipy.stats import norm
import numpy as np

d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)
call_bs = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
print(f"Expected: {call_bs}")
```

**Common Causes**:
1. **Day count convention**: Ensure T is in years
2. **Dividend yield**: Check if q is continuous or discrete
3. **Volatility**: Verify if using decimal (0.20) or percentage (20)
4. **Interest rates**: Ensure continuous compounding

**Solution**:
```python
# Correct parameter format
S = 100.0           # Spot price
K = 100.0           # Strike
T = 1.0             # Years (not days!)
r = 0.05            # 5% annual rate (not 5!)
q = 0.02            # 2% dividend yield
sigma = 0.20        # 20% volatility (not 0.20%!)

# Verify with put-call parity
call = price(S, K, T, r, q, sigma, "call")
put = price(S, K, T, r, q, sigma, "put")
parity = call - put - (S*np.exp(-q*T) - K*np.exp(-r*T))
print(f"Put-call parity error: {abs(parity):.10f}")  # Should be ~0
```

### Issue: Monte Carlo convergence is slow

**Symptoms**:
MC estimates have high variance or don't converge.

**Solution**:
```python
# Solution 1: Increase paths
config = MCConfig(steps=252, paths=100_000)  # More paths

# Solution 2: Use antithetic variates
config = MCConfig(steps=252, paths=50_000, antithetic=True)

# Solution 3: Implement control variates
# See performance_tuning.md for details

# Solution 4: Use quasi-Monte Carlo (Sobol sequences)
from neutryx.core.qmc import sobol_sequence
paths = sobol_monte_carlo(S0, K, T, r, sigma, n_paths=10_000)

# Check convergence
path_counts = [1000, 5000, 10000, 50000, 100000]
prices = [mc_price(n) for n in path_counts]
plt.plot(path_counts, prices)
plt.axhline(y=analytical_price, color='r', linestyle='--')
plt.xlabel('Number of paths')
plt.ylabel('Price')
plt.show()
```

### Issue: Calibration fails or gives unstable results

**Symptoms**:
```python
CalibrationError: Optimization did not converge
```

**Solution**:
```python
from neutryx.calibration.heston import calibrate_heston

# Solution 1: Better initial guess
initial_params = {
    "kappa": 2.0,      # Mean reversion speed
    "theta": 0.04,     # Long-term variance
    "sigma": 0.3,      # Vol of vol
    "rho": -0.7,       # Correlation
    "v0": 0.04         # Initial variance
}

# Solution 2: Add constraints
from neutryx.calibration.constraints import ParameterConstraints
constraints = ParameterConstraints(
    kappa=(0.1, 10.0),
    theta=(0.01, 0.5),
    sigma=(0.01, 2.0),
    rho=(-0.99, 0.99),
    v0=(0.01, 0.5)
)

# Solution 3: Regularization
from neutryx.calibration.regularization import TikhonovRegularization
regularizer = TikhonovRegularization(lambda_=0.01)

# Solution 4: Use robust optimizer
calibrated = calibrate_heston(
    market_data,
    initial_params,
    constraints=constraints,
    regularizer=regularizer,
    optimizer="lbfgs",  # Instead of "adam"
    max_iterations=1000
)

# Solution 5: Scale parameters
# Normalize parameters to similar scales
def scale_params(params):
    return {
        "kappa": params["kappa"] / 10.0,
        "theta": params["theta"] / 0.1,
        "sigma": params["sigma"],
        "rho": params["rho"],
        "v0": params["v0"] / 0.1
    }
```

## Performance Issues

### Issue: Pricing is slower than expected

**Diagnosis**:
```python
import time

# Benchmark
start = time.time()
result = price_option(100.0, 100.0, 1.0, 0.05, 0.20)
result.block_until_ready()  # Important for JAX!
elapsed = time.time() - start
print(f"Time: {elapsed*1000:.2f}ms")
```

**Common Causes**:
1. Not using JIT compilation
2. Not using `block_until_ready()`
3. Python loops instead of vmap
4. Unnecessary data transfers

**Solution**:
```python
# 1. Use JIT
@jax.jit
def price_option(S, K, T, r, sigma):
    return bs_price(S, K, T, r, 0.0, sigma, "call")

# 2. Use vmap for batches
@jax.jit
def price_batch(spots, strikes, maturities):
    return jax.vmap(price_option)(spots, strikes, maturities)

# 3. Keep data on GPU
spots_gpu = jnp.array(spots)  # Transfer once
results = []
for _ in range(1000):
    result = price_option(spots_gpu, ...)  # No transfer
results_cpu = np.array(results)  # Transfer back once

# 4. Profile to find bottlenecks
with jax.profiler.trace("/tmp/jax-trace"):
    result = expensive_computation(data)
# View at https://ui.perfetto.dev
```

### Issue: High memory usage

**Diagnosis**:
```python
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1024**2:.2f} MB")

# Check GPU memory
nvidia-smi
```

**Solution**:
```python
# Process in chunks
def process_large_dataset(data, chunk_size=10000):
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        result = process(chunk)
        results.append(result)
        del chunk, result  # Explicit cleanup
    return jnp.concatenate(results)

# Use generators
def data_generator(total_size, batch_size):
    for i in range(0, total_size, batch_size):
        yield generate_batch(batch_size)

# Use float32
jax.config.update("jax_enable_x64", False)
```

## Memory Issues

### Issue: Memory leak in long-running process

**Symptoms**:
Memory usage increases over time and doesn't decrease.

**Diagnosis**:
```python
import gc
import tracemalloc

tracemalloc.start()

# Run your code
for i in range(100):
    result = compute(data)

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

for stat in top_stats[:10]:
    print(stat)
```

**Solution**:
```python
# Explicit garbage collection
import gc

for i in range(iterations):
    result = compute(data)
    # Process result
    del result
    gc.collect()

# Clear JAX cache periodically
from jax.lib import xla_bridge
xla_bridge.get_backend().clear_cache()

# Use context managers
with jax.disable_jit():
    result = debug_computation(data)
```

## Market Data Issues

### Issue: Connection to Bloomberg fails

**Symptoms**:
```
BloombergConnectionError: Failed to connect to Bloomberg API
```

**Solution**:
```python
# Check Bloomberg service is running
# Windows: Check if "bbcomm.exe" is running
# Linux: Check Bloomberg Terminal is active

# Verify connection settings
config = BloombergConfig(
    host="localhost",
    port=8194,  # Default Bloomberg API port
    timeout=30
)

# Test connection
adapter = BloombergAdapter(config)
try:
    await adapter.connect()
    print("Connected successfully")
except Exception as e:
    print(f"Connection failed: {e}")
    # Check firewall settings
    # Check Bloomberg Terminal is logged in
```

### Issue: Missing or stale market data

**Symptoms**:
Data is not updating or is missing.

**Solution**:
```python
# Check subscription status
manager = FeedManager(adapters=[adapter])
await manager.start()

subscriptions = await manager.get_subscriptions()
print(f"Active subscriptions: {subscriptions}")

# Verify data freshness
latest_tick = await storage.get_latest_tick("AAPL")
age = datetime.now() - latest_tick.timestamp
print(f"Data age: {age.total_seconds()} seconds")

# Force refresh
await manager.resubscribe(["AAPL", "MSFT"])

# Check data quality
quality_report = await validation_pipeline.get_quality_report()
print(f"Failed validations: {quality_report['failed_count']}")
```

## API and Integration Issues

### Issue: REST API returns 500 errors

**Symptoms**:
```
HTTP 500: Internal Server Error
```

**Diagnosis**:
```python
# Check API logs
tail -f /var/log/neutryx/api.log

# Enable debug mode
import logging
logging.basicConfig(level=logging.DEBUG)

# Test endpoint directly
import requests

response = requests.post(
    "http://localhost:8000/price",
    json={"spot": 100, "strike": 100, "maturity": 1.0},
    timeout=30
)
print(response.status_code)
print(response.json())
```

**Solution**:
```python
# Add error handling
from fastapi import HTTPException

@app.post("/price")
async def price_option(request: PricingRequest):
    try:
        result = pricing_service.price(request)
        return {"price": float(result)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Pricing error: {e}")
        raise HTTPException(status_code=500, detail="Internal error")

# Add request validation
from pydantic import BaseModel, validator

class PricingRequest(BaseModel):
    spot: float
    strike: float
    maturity: float

    @validator('spot', 'strike')
    def positive(cls, v):
        if v <= 0:
            raise ValueError('Must be positive')
        return v

    @validator('maturity')
    def valid_maturity(cls, v):
        if not 0 < v <= 30:
            raise ValueError('Maturity must be between 0 and 30 years')
        return v
```

### Issue: gRPC connection refused

**Symptoms**:
```
grpc._channel._InactiveRpcError: Connect Failed
```

**Solution**:
```python
# Check server is running
ps aux | grep grpc

# Start gRPC server
python -m neutryx.api.grpc.server --port 50051

# Test connection
import grpc
channel = grpc.insecure_channel('localhost:50051')
try:
    grpc.channel_ready_future(channel).result(timeout=10)
    print("Connected")
except grpc.FutureTimeoutError:
    print("Connection timeout")

# Check firewall
telnet localhost 50051
```

## Debugging Tips

### Enable Verbose Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Neutryx logger
logger = logging.getLogger('neutryx')
logger.setLevel(logging.DEBUG)
```

### Use JAX Debug Mode

```python
# Enable NaN checking
jax.config.update("jax_debug_nans", True)

# Enable infs checking
jax.config.update("jax_debug_infs", True)

# Disable JIT for debugging
with jax.disable_jit():
    result = problematic_function(data)
```

### Inspect Intermediate Values

```python
# Add debug prints
@jax.jit
def debug_price(S, K, T, r, sigma):
    d1 = calculate_d1(S, K, T, r, sigma)
    jax.debug.print("d1: {}", d1)

    d2 = d1 - sigma * jnp.sqrt(T)
    jax.debug.print("d2: {}", d2)

    price = calculate_price(d1, d2)
    return price
```

### Check Array Shapes

```python
def debug_shapes(**arrays):
    """Print shapes of all arrays"""
    for name, arr in arrays.items():
        print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")

# Usage
debug_shapes(spots=spots, strikes=strikes, prices=prices)
```

## Getting Help

If you're still experiencing issues:

1. **Search existing issues**: [GitHub Issues](https://github.com/neutryx-lab/neutryx-core/issues)
2. **Check documentation**: [docs.neutryx.com](https://neutryx-lab.github.io/neutryx-core)
3. **Ask the community**: [GitHub Discussions](https://github.com/neutryx-lab/neutryx-core/discussions)
4. **Report a bug**: [New Issue](https://github.com/neutryx-lab/neutryx-core/issues/new)

When reporting issues, please include:
- Neutryx version: `pip show neutryx-core`
- JAX version: `pip show jax jaxlib`
- Python version: `python --version`
- OS and platform: `uname -a`
- Minimal reproducible example
- Full error traceback

## Next Steps

- [Getting Started](getting_started.md) - Setup and basics
- [Performance Tuning](performance_tuning.md) - Optimize performance
- [API Reference](api_reference.md) - Complete API docs
