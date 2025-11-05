# Getting Started with Neutryx Core

This guide will help you get up and running with Neutryx Core quickly. Whether you're a quantitative analyst, risk manager, or developer, this guide provides everything you need to start pricing derivatives and managing risk.

## Prerequisites

- Python 3.10 or higher
- Basic knowledge of Python and NumPy
- Understanding of quantitative finance concepts (recommended)
- 8GB+ RAM (16GB+ recommended for large-scale simulations)
- GPU/TPU (optional, for accelerated computations)

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/neutryx-lab/neutryx-core.git
cd neutryx-core

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Install Neutryx in editable mode
pip install -e ".[dev]"
```

### Optional Dependencies

```bash
# For market data integration
pip install asyncpg motor timescaledb

# For QuantLib integration
pip install -e ".[quantlib]"

# For Eigen bindings
pip install -e ".[eigen]"

# Install all optional features
pip install -e ".[native,marketdata,dev]"
```

### Verify Installation

```bash
# Run tests to verify installation
pytest -q

# Check JAX installation
python -c "import jax; print(f'JAX version: {jax.__version__}')"

# Verify GPU support (if available)
python -c "import jax; print(f'Devices: {jax.devices()}')"
```

## Your First Pricing Example

### Example 1: Black-Scholes European Option

Let's price a European call option using the Black-Scholes model:

```python
import jax
import jax.numpy as jnp
from neutryx.models.bs import price as bs_price, greeks as bs_greeks

# Market parameters
spot = 100.0          # Current stock price
strike = 100.0        # Strike price
maturity = 1.0        # Time to maturity (years)
risk_free = 0.05      # Risk-free rate
dividend = 0.02       # Dividend yield
volatility = 0.20     # Volatility

# Price the option
call_price = bs_price(spot, strike, maturity, risk_free, dividend, volatility, "call")
put_price = bs_price(spot, strike, maturity, risk_free, dividend, volatility, "put")

print(f"Call Price: ${call_price:.4f}")
print(f"Put Price:  ${put_price:.4f}")

# Calculate Greeks
delta, gamma, vega, theta, rho = bs_greeks(
    spot, strike, maturity, risk_free, dividend, volatility, "call"
)

print(f"\nGreeks:")
print(f"  Delta: {delta:.4f}")
print(f"  Gamma: {gamma:.4f}")
print(f"  Vega:  {vega:.4f}")
print(f"  Theta: {theta:.4f}")
print(f"  Rho:   {rho:.4f}")
```

### Example 2: Monte Carlo Simulation

Price an option using Monte Carlo simulation:

```python
import jax
import jax.numpy as jnp
from neutryx.core.engine import MCConfig, simulate_gbm, present_value

# Setup
key = jax.random.PRNGKey(42)
S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
q = 0.02
sigma = 0.20

# Configure Monte Carlo
config = MCConfig(
    steps=252,        # Daily timesteps
    paths=100_000,    # Number of paths
    seed=42
)

# Simulate paths
paths = simulate_gbm(key, S0, r - q, sigma, T, config)
ST = paths[:, -1]  # Terminal values

# Calculate payoffs and price
call_payoff = jnp.maximum(ST - K, 0.0)
call_price = present_value(call_payoff, jnp.array(T), r)

put_payoff = jnp.maximum(K - ST, 0.0)
put_price = present_value(put_payoff, jnp.array(T), r)

print(f"Monte Carlo Call Price: ${float(call_price):.4f}")
print(f"Monte Carlo Put Price:  ${float(put_price):.4f}")

# Calculate standard error
call_std = jnp.std(call_payoff * jnp.exp(-r * T)) / jnp.sqrt(config.paths)
print(f"Standard Error: ${float(call_std):.4f}")
```

### Example 3: Path-Dependent Options

Price an Asian option with geometric averaging:

```python
import jax
import jax.numpy as jnp
from neutryx.core.engine import MCConfig, simulate_gbm, present_value

# Setup
key = jax.random.PRNGKey(42)
S0, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.20

# Simulate paths
config = MCConfig(steps=252, paths=100_000)
paths = simulate_gbm(key, S0, r - q, sigma, T, config)

# Geometric average
avg = jnp.exp(jnp.mean(jnp.log(paths), axis=1))
payoff = jnp.maximum(avg - K, 0.0)
price = present_value(payoff, jnp.array(T), r)

print(f"Asian Option Price: ${float(price):.4f}")
```

### Example 4: Multi-Asset Products

Price equity and commodity derivatives:

```python
from neutryx.products.equity import equity_forward_price, variance_swap_value
from neutryx.products.commodity import commodity_forward_price

# Equity forward
eq_forward = equity_forward_price(
    spot=100.0,
    maturity=1.0,
    risk_free_rate=0.05,
    dividend_yield=0.02
)
print(f"Equity Forward: ${eq_forward:.2f}")

# Commodity forward with convenience yield
commodity_fwd = commodity_forward_price(
    spot=50.0,
    maturity=1.0,
    risk_free_rate=0.05,
    storage_cost=0.02,
    convenience_yield=0.03
)
print(f"Commodity Forward: ${commodity_fwd:.2f}")

# Variance swap
var_swap = variance_swap_value(
    realized_var=0.04,
    strike_var=0.045,
    notional=1_000_000,
    days_remaining=180,
    total_days=252
)
print(f"Variance Swap Value: ${var_swap:,.2f}")
```

## Core Concepts

### JAX-First Design

Neutryx Core is built on JAX, which provides:

- **Just-In-Time (JIT) compilation**: Significant performance improvements
- **Automatic differentiation**: Accurate Greeks and sensitivities
- **GPU/TPU acceleration**: Seamless hardware acceleration
- **Vectorization**: Efficient batch operations

```python
import jax

# JIT compilation for faster execution
@jax.jit
def price_option(S, K, T, r, sigma):
    from neutryx.models.bs import price
    return price(S, K, T, r, 0.0, sigma, "call")

# Automatic differentiation for Greeks
delta_fn = jax.grad(price_option, argnums=0)
delta = delta_fn(100.0, 100.0, 1.0, 0.05, 0.20)
print(f"Delta: {delta:.4f}")
```

### Configuration Management

Use YAML configuration for reproducible pricing:

```yaml
# config/pricing.yaml
monte_carlo:
  steps: 252
  paths: 100000
  seed: 42

market:
  risk_free_rate: 0.05
  dividend_yield: 0.02

models:
  black_scholes:
    volatility: 0.20
  heston:
    kappa: 2.0
    theta: 0.04
    sigma: 0.3
    rho: -0.7
    v0: 0.04
```

Load and use configuration:

```python
from neutryx.config.loader import load_config

config = load_config("config/pricing.yaml")
mc_config = MCConfig(**config["monte_carlo"])
```

### Working with Market Data

#### Real-Time Market Data Feed

```python
from neutryx.market.adapters import BloombergAdapter, BloombergConfig
from neutryx.market.storage import TimescaleDBStorage, TimescaleDBConfig
from neutryx.market.feeds import FeedManager

# Configure Bloomberg
bloomberg_config = BloombergConfig(
    adapter_name="bloomberg",
    host="localhost",
    port=8194
)

# Configure storage
storage_config = TimescaleDBConfig(
    host="localhost",
    database="market_data",
    compression_enabled=True
)

# Setup feed manager
adapter = BloombergAdapter(bloomberg_config)
storage = TimescaleDBStorage(storage_config)
manager = FeedManager(adapters=[adapter], storage=storage)

# Subscribe to real-time data
await manager.start()
await manager.subscribe("equity", ["AAPL", "MSFT", "GOOGL"])
```

### Risk Management

#### Calculate Value at Risk (VaR)

```python
from neutryx.risk.var import historical_var, monte_carlo_var, parametric_var

# Historical simulation VaR
returns = jnp.array([...])  # Historical returns
var_95 = historical_var(returns, confidence=0.95)
print(f"Historical VaR (95%): ${var_95:,.2f}")

# Monte Carlo VaR
positions = jnp.array([100000, 200000, 150000])
var_mc = monte_carlo_var(positions, returns, confidence=0.95, num_simulations=10000)
print(f"Monte Carlo VaR (95%): ${var_mc:,.2f}")

# Parametric VaR
mean_return = jnp.mean(returns)
std_return = jnp.std(returns)
var_param = parametric_var(positions, mean_return, std_return, confidence=0.95)
print(f"Parametric VaR (95%): ${var_param:,.2f}")
```

#### Position Limits and Pre-Trade Controls

```python
from neutryx.risk.limits import PositionLimits, LimitChecker

# Define limits
limits = PositionLimits(
    notional_limit=10_000_000,
    var_limit=500_000,
    concentration_limit=0.25,
    issuer_exposure_limit=2_000_000
)

# Check proposed trade
checker = LimitChecker(limits)
proposed_trade = {
    "notional": 1_000_000,
    "var": 50_000,
    "issuer": "AAPL",
    "position_size": 0.15
}

result = checker.check_trade(proposed_trade)
if result.approved:
    print("Trade approved")
else:
    print(f"Trade rejected: {result.reason}")
```

### Model Calibration

Calibrate the Heston model to market data:

```python
from neutryx.calibration.heston import calibrate_heston
from neutryx.calibration.model_selection import select_model_aic, cross_validate

# Market data (strikes, maturities, implied vols)
market_data = {
    "strikes": jnp.array([90, 95, 100, 105, 110]),
    "maturities": jnp.array([0.25, 0.5, 1.0]),
    "implied_vols": jnp.array([...])
}

# Calibrate model
params, diagnostics = calibrate_heston(
    market_data,
    initial_guess={"kappa": 2.0, "theta": 0.04, "sigma": 0.3, "rho": -0.7, "v0": 0.04}
)

print(f"Calibrated parameters: {params}")
print(f"Calibration RMSE: {diagnostics['rmse']:.6f}")

# Model selection using AIC
aic_score = select_model_aic(params, market_data, num_params=5)
print(f"AIC Score: {aic_score:.2f}")
```

## Next Steps

Now that you've completed the basics, explore these advanced topics:

1. **[Tutorials](tutorials.md)** - Hands-on tutorials for specific use cases
2. **[Product Pricing Guide](products.md)** - Multi-asset class product coverage
3. **[Risk Management Guide](risk/index.md)** - Comprehensive risk analytics
4. **[XVA Framework](valuations_comprehensive.md)** - CVA, DVA, FVA, and more
5. **[Model Calibration](calibration/index.md)** - Advanced calibration techniques
6. **[API Reference](api_reference.md)** - Complete API documentation
7. **[Performance Tuning](performance_tuning.md)** - Optimization strategies
8. **[Architecture Guide](architecture.md)** - System design and patterns

## Common Pitfalls

### 1. JAX Pure Functions

JAX requires pure functions (no side effects):

```python
# Bad - side effects
def bad_pricing(S):
    global counter
    counter += 1  # Side effect!
    return S * some_factor

# Good - pure function
def good_pricing(S, counter):
    return S * some_factor, counter + 1
```

### 2. Array Shapes

Always check array shapes for broadcasting:

```python
# Check shapes
print(f"Spot shape: {S.shape}")
print(f"Strike shape: {K.shape}")

# Reshape if needed
S = S.reshape(-1, 1)
K = K.reshape(1, -1)
```

### 3. GPU Memory

Monitor GPU memory for large simulations:

```python
import jax

# Check available devices
print(jax.devices())

# Monitor memory
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
```

## Getting Help

- **Documentation**: [https://neutryx-lab.github.io/neutryx-core](https://neutryx-lab.github.io/neutryx-core)
- **GitHub Issues**: [https://github.com/neutryx-lab/neutryx-core/issues](https://github.com/neutryx-lab/neutryx-core/issues)
- **Examples**: Browse the `demos/` and `examples/` directories
- **API Reference**: [docs/api_reference.md](api_reference.md)

## What's Next?

After completing this guide, you should be able to:

- Price basic and exotic derivatives
- Run Monte Carlo simulations
- Calculate Greeks and sensitivities
- Manage risk with VaR and limits
- Calibrate models to market data
- Integrate real-time market data feeds

Continue your journey with our comprehensive tutorials and documentation!
