# Equity Pricing Models - Complete Guide

Neutryx provides a comprehensive suite of equity pricing models covering the full spectrum from simple Black-Scholes to advanced rough volatility and jump-diffusion models.

## Model Overview

| Model | Type | Key Features | Best For |
|-------|------|--------------|----------|
| **Black-Scholes** | Closed-form | Constant volatility | Vanilla options, quick pricing |
| **Local Volatility (Dupire)** | PDE/Monte Carlo | Arbitrage-free smile fitting | Exotic options, model hedging |
| **Stochastic Local Volatility** | Monte Carlo | SV + Local vol combined | Forward-start, cliquet options |
| **Heston** | Semi-closed-form | Stochastic volatility | Vol surface calibration |
| **SABR** | Closed-form approx | Stochastic vol beta | Swaption markets |
| **Rough Bergomi** | Monte Carlo | Fractional BM, rough vol | Short-dated skew |
| **Rough Heston** | Monte Carlo | Rough vol + mean reversion | Full term structure |
| **Merton Jump-Diffusion** | Closed-form | Lognormal jumps | Earnings events |
| **Kou Jump-Diffusion** | Monte Carlo | Asymmetric jumps | Crash protection |
| **Variance Gamma** | Monte Carlo/FFT | Pure jump Lévy | Skew trading |

---

## 1. Local Volatility Models

### Dupire Local Volatility

**Mathematical Framework:**
```
dS = r*S*dt + σ_L(S,t)*S*dW
```

where σ_L(S,t) is calibrated from market prices using Dupire's formula:

```
σ²_L(K,T) = (∂C/∂T + (r-q)K*∂C/∂K + q*C) / (0.5*K²*∂²C/∂K²)
```

**Usage:**
```python
from neutryx.models import DupireParams, dupire_formula, calibrate_local_vol_surface

# Calibrate from market data
local_vol_params = calibrate_local_vol_surface(
    strikes=strikes,
    maturities=maturities,
    market_call_prices=prices,
    S0=100.0,
    r=0.05,
    q=0.01
)

# Price exotic option using Monte Carlo
from neutryx.models.dupire import DupireSDE
sde = DupireSDE(mu=0.04, params=local_vol_params)
# ... simulate and price
```

**Key Properties:**
- ✅ Arbitrage-free by construction
- ✅ Perfectly calibrates to vanilla option prices
- ❌ Unrealistic forward volatility dynamics
- ❌ Predicts zero correlation between spot and vol

**Best For:**
- Exotic derivatives pricing
- Model-independent hedging
- First-generation exotics (barriers, Asians, lookbacks)

---

## 2. Stochastic Local Volatility (SLV)

**Mathematical Framework:**
```
dS = r*S*dt + √V*σ_L(S,t)*S*dW_S
dV = κ(θ - V)*dt + ξ√V*dW_V
dW_S * dW_V = ρ*dt
```

Combines the arbitrage-free property of local vol with realistic vol dynamics.

**Usage:**
```python
from neutryx.models import SLVParams, simulate_slv

slv_params = SLVParams(
    kappa=2.0,          # Mean reversion speed
    theta=0.04,         # Long-term variance (20% vol)
    xi=0.3,             # Vol-of-vol
    rho=-0.7,           # Spot-vol correlation
    local_vol_func=local_vol_surface.value,  # From Dupire calibration
    V0=0.04
)

# Simulate paths
paths = simulate_slv(key, S0=100, T=1.0, r=0.05, q=0.01, params=slv_params, cfg=mc_config)
```

**Key Properties:**
- ✅ Fits vanilla option smile (like local vol)
- ✅ Realistic forward vol dynamics (like stochastic vol)
- ✅ Non-zero spot-vol correlation
- ❌ Complex calibration procedure
- ❌ Slower than pure local vol

**Best For:**
- Forward-start options
- Cliquet options
- Variance swaps
- Second-generation exotics

---

## 3. Rough Volatility Models

### Rough Bergomi (rBergomi)

**Mathematical Framework:**
```
dS = r*S*dt + √V_t*S*dW_S
V_t = ξ(t) * exp(η*W^H_t - 0.5*η²*t^(2H))
```

where W^H is fractional Brownian motion with Hurst parameter H < 0.5.

**Usage:**
```python
from neutryx.models import RoughBergomiParams, simulate_rough_bergomi

params = RoughBergomiParams(
    hurst=0.1,                  # H < 0.5 for roughness
    eta=1.9,                    # Vol-of-vol
    rho=-0.9,                   # Spot-vol correlation
    forward_variance=0.04       # Can be function of time
)

paths = simulate_rough_bergomi(
    key, S0=100, T=1.0, r=0.05, q=0.01,
    cfg=mc_config, params=params,
    return_full=True
)
```

**Key Properties:**
- ✅ Excellent fit to short-dated options
- ✅ Realistic ATM skew behavior
- ✅ Parsimonious (few parameters)
- ❌ No closed-form solutions
- ❌ Computationally intensive

**Best For:**
- Short-dated option pricing
- Vol surface modeling
- Research and calibration studies

### Rough Heston

**Mathematical Framework:**

Similar to rough Bergomi but with mean-reverting variance:
```
V_t = V_0 + fractional_integral(κ(θ - V_s) + ξ√V_s*dW_V)
```

**Usage:**
```python
from neutryx.models import RoughHestonParams, simulate_rough_heston

params = RoughHestonParams(
    H=0.1,              # Hurst < 0.5
    kappa=0.3,          # Mean reversion
    theta=0.02,         # Long-term variance
    xi=0.3,             # Vol-of-vol
    rho=-0.7,           # Correlation
    V0=0.04
)

S_paths, V_paths = simulate_rough_heston(
    key, S0=100, T=1.0, r=0.05, q=0.01,
    params=params, cfg=mc_config
)
```

**Key Properties:**
- ✅ More tractable than rBergomi
- ✅ Mean-reverting variance
- ✅ Better long-dated performance
- ❌ Still requires fractional calculus
- ❌ Complex numerical schemes

**Best For:**
- Full term structure fitting
- Vol derivatives (variance swaps, vol swaps)
- Hybrid models

---

## 4. Jump-Diffusion Models

### Merton Jump-Diffusion

**Mathematical Framework:**
```
dS = (r-q-λκ)*S*dt + σ*S*dW + S*dJ
```

where J is a compound Poisson process with lognormal jumps.

**Usage:**
```python
from neutryx.models.jump_diffusion import (
    MertonParams, merton_jump_price, calibrate_merton
)

params = MertonParams(
    sigma=0.15,         # Diffusion volatility
    lam=2.0,            # Jump intensity (2 jumps/year)
    mu_jump=-0.05,      # Mean log-jump size
    sigma_jump=0.10     # Jump size volatility
)

# Closed-form pricing
price = merton_jump_price(
    S0=100, K=100, T=1.0, r=0.05, q=0.01,
    sigma=params.sigma, lam=params.lam,
    mu_jump=params.mu_jump, sigma_jump=params.sigma_jump,
    kind='call'
)

# Calibrate to market
calibrated = calibrate_merton(
    S0, strikes, maturities, market_prices,
    r=0.05, q=0.01
)
```

**Key Properties:**
- ✅ Closed-form solutions (fast pricing)
- ✅ Captures jump risk
- ✅ Easy calibration
- ❌ Symmetric jumps only
- ❌ Limited smile flexibility

**Best For:**
- Earnings announcements
- Event-driven strategies
- Credit spread modeling

### Kou Double Exponential Jump-Diffusion

**Mathematical Framework:**

Asymmetric jumps with double exponential distribution:
```
P(jump = y) = p*η₁*exp(-η₁*y)*I(y>0) + (1-p)*η₂*exp(η₂*y)*I(y<0)
```

**Usage:**
```python
from neutryx.models.kou import simulate_kou, price_vanilla_kou_mc

paths = simulate_kou(
    key, S0=100, mu=0.04, sigma=0.15,
    lam=5.0,            # Jump rate
    p=0.4,              # Prob of up-jump
    eta1=25.0,          # Up-jump rate (> 1)
    eta2=10.0,          # Down-jump rate
    T=1.0, cfg=mc_config
)
```

**Key Properties:**
- ✅ Asymmetric jumps (up ≠ down)
- ✅ Better tail modeling
- ✅ Captures leverage effect
- ❌ More parameters to calibrate
- ❌ No closed-form (MC or PDE needed)

**Best For:**
- Crash protection strategies
- Deep OTM options
- Tail risk hedging

### Variance Gamma

**Mathematical Framework:**

Pure jump Lévy process (no diffusion):
```
X_t = θ*G_t + σ*W(G_t)
```

where G_t is a Gamma process (subordinator).

**Usage:**
```python
from neutryx.models.variance_gamma import (
    simulate_variance_gamma,
    price_vanilla_vg_mc,
    vg_characteristic_function
)

paths = simulate_variance_gamma(
    key, S0=100, mu=0.04,
    theta=-0.14,        # Drift
    sigma=0.20,         # Volatility
    nu=0.20,            # Variance rate (kurtosis)
    T=1.0, cfg=mc_config
)

# FFT pricing via characteristic function
char_func = vg_characteristic_function(u, t=1.0, theta=-0.14, sigma=0.20, nu=0.20)
```

**Key Properties:**
- ✅ Infinite activity (many small jumps)
- ✅ Flexible skew and kurtosis
- ✅ Tractable characteristic function
- ❌ No diffusion component
- ❌ Can be unstable for extreme parameters

**Best For:**
- Volatility skew trading
- Lévy process research
- FFT-based pricing

---

## 5. Time-Changed Lévy Processes

**Framework:**

General family of processes:
```
S_t = S_0 * exp(X(T_t))
```

where X is a Lévy process and T_t is a stochastic clock.

**Usage:**
```python
from neutryx.models import TimeChangedLevyParams, simulate_time_changed_levy

params = TimeChangedLevyParams(
    levy_process='VG',
    levy_params={'theta': -0.14, 'sigma': 0.2, 'nu': 0.2},
    time_change='gamma',
    time_change_params={'rate': 1.0}
)

paths = simulate_time_changed_levy(
    key, S0=100, T=1.0, r=0.05, q=0.01,
    params=params, cfg=mc_config
)
```

**Supported Models:**
- Variance Gamma (VG) ✅
- Normal Inverse Gaussian (NIG) 🚧
- CGMY process 🚧

**Key Properties:**
- ✅ Very flexible jump structure
- ✅ Captures clustering
- ✅ Heavy tails
- ❌ Complex calibration
- ❌ Many parameters

---

## Model Comparison

### Computational Performance

| Model | Pricing Speed | Calibration | Memory |
|-------|---------------|-------------|--------|
| Black-Scholes | ⚡⚡⚡⚡ | ⚡⚡⚡⚡ | ⚡⚡⚡⚡ |
| Local Vol (PDE) | ⚡⚡⚡ | ⚡⚡⚡ | ⚡⚡⚡ |
| Heston | ⚡⚡⚡ | ⚡⚡ | ⚡⚡⚡ |
| SLV | ⚡⚡ | ⚡ | ⚡⚡ |
| Rough Bergomi | ⚡ | ⚡ | ⚡⚡ |
| Merton Jump | ⚡⚡⚡⚡ | ⚡⚡⚡ | ⚡⚡⚡⚡ |
| Kou Jump | ⚡⚡ | ⚡⚡ | ⚡⚡⚡ |
| Variance Gamma | ⚡⚡ | ⚡⚡ | ⚡⚡⚡ |

### Market Fit Quality

| Model | Vanilla Smile | Forward Smile | Short-dated Skew | Tail Behavior |
|-------|---------------|---------------|------------------|---------------|
| Black-Scholes | ❌ | ❌ | ❌ | ❌ |
| Local Vol | ✅ | ⚠️ | ⚠️ | ⚠️ |
| SLV | ✅ | ✅ | ✅ | ⚠️ |
| Heston | ✅ | ✅ | ⚠️ | ⚠️ |
| Rough Bergomi | ✅ | ✅ | ✅ | ⚠️ |
| Rough Heston | ✅ | ✅ | ✅ | ⚠️ |
| Merton Jump | ⚠️ | ⚠️ | ⚠️ | ✅ |
| Kou Jump | ⚠️ | ⚠️ | ⚠️ | ✅ |
| Variance Gamma | ✅ | ✅ | ✅ | ✅ |

---

## Choosing the Right Model

### By Use Case

**Vanilla Options:**
- Short-dated (< 3M): Rough Bergomi, SLV
- Medium-term (3M-1Y): Heston, SLV
- Long-dated (> 1Y): Local Vol, Heston

**Exotic Options:**
- Barriers: Local Vol, SLV
- Asians: Local Vol, Heston
- Cliquets/Forward-starts: SLV
- Lookbacks: Local Vol, Jump models

**Volatility Products:**
- Variance swaps: SLV, Rough Heston
- Volatility swaps: Heston, SLV
- Vol options: Heston, Rough volatility

**Event Risk:**
- Earnings: Merton Jump
- Crashes: Kou Jump
- Tail hedging: Kou, Variance Gamma

### By Market Regime

**Low Volatility:**
- Use: Local Vol, Heston
- Avoid: Jump models (over-parameterized)

**High Volatility:**
- Use: Jump models, SLV
- Avoid: Simple models (underestimate risk)

**Crisis/Crash:**
- Use: Kou, Variance Gamma
- Avoid: Local Vol (no jumps)

---

## Advanced Topics

### Model Calibration

All models support calibration to market data:

```python
# Example: Calibrate Merton model
from neutryx.models.jump_diffusion import calibrate_merton

calibrated_params = calibrate_merton(
    S0=100,
    strikes=market_strikes,
    maturities=market_maturities,
    market_prices=market_calls,
    r=0.05,
    q=0.01,
    initial=MertonParams(sigma=0.2, lam=2.0, mu_jump=-0.05, sigma_jump=0.1),
    n_iterations=200
)
```

### Model Risk Management

Compare prices across models:

```python
from neutryx.models import get_model_characteristics

# Get model characteristics
chars = get_model_characteristics()

# Price with multiple models
prices = {
    'local_vol': price_with_local_vol(...),
    'heston': price_with_heston(...),
    'slv': price_with_slv(...),
    'merton': price_with_merton(...),
}

# Model risk metric: spread across models
model_risk = max(prices.values()) - min(prices.values())
```

### Hybrid Models

Combine different model features:

```python
# SLV = Local Vol + Stochastic Vol
slv = SLVParams(
    local_vol_func=dupire_surface.value,
    kappa=heston_params.kappa,
    theta=heston_params.theta,
    xi=heston_params.xi,
    rho=heston_params.rho
)

# Jump-diffusion + Stochastic Vol (Bates model)
# Available via Heston + Jump extensions
```

---

## Performance Optimization

### GPU Acceleration

All models support JAX's GPU acceleration:

```python
import jax
# Automatic GPU usage if available
paths = simulate_rough_bergomi(key, ..., cfg=cfg)  # Runs on GPU if available
```

### Batch Pricing

Vectorize over strikes/maturities:

```python
# Price entire surface at once
prices = jax.vmap(jax.vmap(price_func))(strikes_grid, maturities_grid)
```

### Just-In-Time Compilation

```python
from jax import jit

# JIT compile for speed
@jit
def fast_pricer(params):
    return simulate_and_price(params)
```

---

## References

**Local Volatility:**
- Dupire, B. (1994). Pricing with a smile. Risk, 7(1), 18-20.

**Stochastic Volatility:**
- Heston, S. L. (1993). A closed-form solution for options with stochastic volatility.

**Rough Volatility:**
- Bayer, C., Friz, P., & Gatheral, J. (2016). Pricing under rough volatility.
- El Euch, O., & Rosenbaum, M. (2018). Perfect hedging in rough Heston models.

**Jump-Diffusion:**
- Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous.
- Kou, S. G. (2002). A jump-diffusion model for option pricing.

**Lévy Processes:**
- Madan, D. B., Carr, P., & Chang, E. C. (1998). The Variance Gamma process.
- Carr, P., & Wu, L. (2004). Time-changed Lévy processes and option pricing.

---

## Support and Contributing

For issues, questions, or contributions:
- GitHub: https://github.com/neutryx-lab/neutryx-core
- Documentation: https://docs.neutryx.com
- Examples: `examples/models/` directory
