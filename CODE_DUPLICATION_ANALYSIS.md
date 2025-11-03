# Code Duplication Analysis: neutryx-core/src/neutryx/models

## Overview
This analysis identifies concrete patterns of code duplication across the models module that could be extracted into reusable utility functions.

---

## 1. REPEATED CODE PATTERNS: Parameter Validation

### Pattern: __post_init__ Validation for Positive Parameters

**Files affected:** vasicek.py, cir.py, hull_white.py

**Vasicek (lines 51-56):**
```python
def __post_init__(self):
    """Validate parameters."""
    if self.a <= 0:
        raise ValueError(f"Mean reversion speed a must be positive, got {self.a}")
    if self.sigma <= 0:
        raise ValueError(f"Volatility sigma must be positive, got {self.sigma}")
```

**CIR (lines 53-62):**
```python
def __post_init__(self):
    """Validate parameters."""
    if self.a <= 0:
        raise ValueError(f"Mean reversion speed a must be positive, got {self.a}")
    if self.b <= 0:
        raise ValueError(f"Long-term mean b must be positive, got {self.b}")
    if self.sigma <= 0:
        raise ValueError(f"Volatility sigma must be positive, got {self.sigma}")
    if self.r0 <= 0:
        raise ValueError(f"Initial rate r0 must be positive, got {self.r0}")
```

**HullWhite (lines 53-58):**
```python
def __post_init__(self):
    """Validate parameters."""
    if self.a <= 0:
        raise ValueError(f"Mean reversion speed a must be positive, got {self.a}")
    if self.sigma <= 0:
        raise ValueError(f"Volatility sigma must be positive, got {self.sigma}")
```

**Duplication Factor:** 3 implementations of very similar validation logic
**Extractable Utility:** `validate_positive_param(name: str, value: float)`

---

## 2. REPEATED SIMULATION PATTERNS: Path Simulation Structure

### Pattern A: Multiple Paths via vmap + Key Splitting

**Vasicek.simulate_paths() (lines 200-235):**
```python
def simulate_paths(
    params: VasicekParams,
    T: float,
    n_steps: int,
    n_paths: int,
    key: jax.random.KeyArray,
    method: str = "exact"
) -> jnp.ndarray:
    keys = jax.random.split(key, n_paths)

    def sim_single_path(k):
        return simulate_path(params, T, n_steps, k, method=method)

    return jax.vmap(sim_single_path)(keys)
```

**CIR.simulate_paths() (lines 249-289):**
```python
def simulate_paths(
    params: CIRParams,
    T: float,
    n_steps: int,
    n_paths: int,
    key: jax.random.KeyArray,
    method: str = "euler"
) -> jnp.ndarray:
    keys = jax.random.split(key, n_paths)

    def sim_single_path(k):
        return simulate_path(params, T, n_steps, k, method=method)

    return jax.vmap(sim_single_path)(keys)
```

**HullWhite.simulate_paths() (lines 216-251):**
```python
def simulate_paths(
    params: HullWhiteParams,
    T: float,
    n_steps: int,
    n_paths: int,
    key: jax.random.KeyArray,
    method: str = "exact"
) -> jnp.ndarray:
    keys = jax.random.split(key, n_paths)

    def sim_single_path(k):
        return simulate_path(params, T, n_steps, k, method=method)

    return jax.vmap(sim_single_path)(keys)
```

**Duplication Factor:** 3 identical implementations
**Extractable Utility:** `generic_vmap_simulate_paths(simulate_fn, key, n_paths)`

### Pattern B: Initial Rate Path Prepending

**Vasicek (line 197):**
```python
return jnp.concatenate([jnp.array([r0]), r_path])
```

**CIR (line 246):**
```python
return jnp.concatenate([jnp.array([r0]), r_path])
```

**HullWhite (line 213):**
```python
return jnp.concatenate([jnp.array([r0]), r_path])
```

**Duplication Factor:** 3 identical implementations
**Extractable Utility:** `prepend_initial_value(initial, path)`

---

## 3. REPEATED NUMERICAL PATTERNS: Discount Factor Calculations

### Pattern: Discount Factor in Pricing Functions

**variance_gamma.py (line 161):**
```python
discount = jnp.exp(-r * T)
return float((discount * payoffs).mean())
```

**kou.py (line 186):**
```python
discount = jnp.exp(-r * T)
return float((discount * payoffs).mean())
```

**rough_vol.py (lines 204-205):**
```python
discount = jnp.exp(-r * T)
return discount * payoff.mean(axis=0)
```

**jump_diffusion.py (line 84):**
```python
discounted = jnp.exp(-r * T) * weights * payoff
```

**Duplication Factor:** 4+ pricing functions
**Extractable Utility:** `discount_payoff(payoffs, rate, maturity)` or `present_value(payoffs, rate, maturity)`

---

## 4. OPTION PAYOFF CALCULATIONS: Repeated Pattern

### Pattern: Call/Put Payoff Computation

**variance_gamma.py (lines 154-159):**
```python
if kind == "call":
    payoffs = jnp.maximum(ST - K, 0.0)
elif kind == "put":
    payoffs = jnp.maximum(K - ST, 0.0)
else:
    raise ValueError(f"Unknown option kind: {kind}")
```

**kou.py (lines 179-184):**
```python
if kind == "call":
    payoffs = jnp.maximum(ST - K, 0.0)
elif kind == "put":
    payoffs = jnp.maximum(K - ST, 0.0)
else:
    raise ValueError(f"Unknown option kind: {kind}")
```

**jump_diffusion.py (lines 102-103):**
```python
if T <= 0:
    return jnp.maximum(S0 - K, 0.0)
# ... later for put:
return call_price - parity
```

**bs.py (lines 18-21):**
```python
if kind == "call":
    return jnp.exp(-q*T)*S*norm.cdf(d1) - jnp.exp(-r*T)*K*norm.cdf(d2)
else:
    return jnp.exp(-r*T)*K*norm.cdf(-d2) - jnp.exp(-q*T)*S*norm.cdf(-d1)
```

**Duplication Factor:** 4+ implementations
**Extractable Utility:** `compute_payoff(spot_prices, strike, kind: str)`

---

## 5. SIMILAR ARRAY INITIALIZATION PATTERNS

### Pattern A: Log-transformed Price Initialization

**variance_gamma.py (lines 94-100):**
```python
log_S0 = jnp.log(jnp.asarray(S0, dtype=dtype))
total_paths = increments.shape[0]
cum_returns = jnp.cumsum(increments, axis=1)
log_paths = jnp.concatenate(
    [jnp.full((total_paths, 1), log_S0, dtype=dtype), log_S0 + cum_returns],
    axis=1,
)
return jnp.exp(log_paths)
```

**kou.py (lines 113-120):**
```python
log_S0 = jnp.log(jnp.asarray(S0, dtype=dtype))
total_paths = increments.shape[0]
cum_returns = jnp.cumsum(increments, axis=1)
log_paths = jnp.concatenate(
    [jnp.full((total_paths, 1), log_S0, dtype=dtype), log_S0 + cum_returns],
    axis=1,
)
return jnp.exp(log_paths)
```

**rough_vol.py (lines 149-154):**
```python
log_S0 = jnp.log(S0)
cumulative = jnp.cumsum(log_increments, axis=1)
log_paths = jnp.concatenate(
    [jnp.full((log_increments.shape[0], 1), log_S0, dtype=cfg.dtype), log_S0 + cumulative],
    axis=1,
)
price_paths = jnp.exp(log_paths)
```

**Duplication Factor:** 3 near-identical patterns
**Extractable Utility:** `build_log_paths(S0, increments, dtype) -> prices`

---

## 6. REPEATED BOND PRICING PATTERNS

### Pattern: Zero-Coupon Bond Prices

**vasicek.py (lines 88-99):**
```python
a, b, sigma, r0 = params.a, params.b, params.sigma, params.r0

B_T = (1.0 - jnp.exp(-a * T)) / a
term1 = (B_T - T) * (a * a * b - sigma * sigma / 2.0) / (a * a)
term2 = -sigma * sigma * B_T * B_T / (4.0 * a)
A_T = jnp.exp(term1 + term2)

return A_T * jnp.exp(-B_T * r0)
```

**cir.py (lines 105-127):**
```python
a, b, sigma, r0 = params.a, params.b, params.sigma, params.r0

gamma = jnp.sqrt(a * a + 2.0 * sigma * sigma)
exp_gamma_T = jnp.exp(gamma * T)
denom = (gamma + a) * (exp_gamma_T - 1.0) + 2.0 * gamma
B_T = 2.0 * (exp_gamma_T - 1.0) / denom
exponent = 2.0 * a * b / (sigma * sigma)
numerator = 2.0 * gamma * jnp.exp((a + gamma) * T / 2.0)
A_T = jnp.power(numerator / denom, exponent)

return A_T * jnp.exp(-B_T * r0)
```

**hull_white.py (lines 91-114):**
```python
a, sigma, r0 = params.a, params.sigma, params.r0

B_T = (1.0 - jnp.exp(-a * T)) / a
V_T = (sigma * sigma / (2.0 * a * a)) * (B_T - T) - \
      (sigma * sigma / (4.0 * a)) * B_T * B_T

if forward_curve_fn is None:
    mean_level = r0
    A_T = jnp.exp((B_T - T) * mean_level + V_T)
else:
    f0 = forward_curve_fn(0.0)
    A_T = jnp.exp((B_T - T) * f0 + V_T)

return A_T * jnp.exp(-B_T * r0)
```

**Duplication Factor:** 3 similar bond pricing formulas
**Extractable Utility:** `compute_bond_price(a, b, sigma, r0, T)`

---

## 7. REPEATED YIELD CURVE COMPUTATION

### Pattern: Identical yield_curve() implementations

**vasicek.py (lines 120-124):**
```python
def yield_curve(params: VasicekParams, maturities: jnp.ndarray) -> jnp.ndarray:
    def yield_at_maturity(T):
        P_T = zero_coupon_bond_price(params, T)
        return -jnp.log(P_T) / T

    return jax.vmap(yield_at_maturity)(maturities)
```

**cir.py (lines 145-149):**
```python
def yield_curve(params: CIRParams, maturities: jnp.ndarray) -> jnp.ndarray:
    def yield_at_maturity(T):
        P_T = zero_coupon_bond_price(params, T)
        return -jnp.log(P_T) / T

    return jax.vmap(yield_at_maturity)(maturities)
```

**Duplication Factor:** 2 identical implementations
**Extractable Utility:** Generic `compute_yield_curve(bond_price_fn, maturities)`

---

## 8. RANDOM NUMBER GENERATION PATTERNS

### Pattern: Key Splitting for Multiple Random Components

**kou.py (lines 76-95):**
```python
key_norm, key_pois, key_jump_dir, key_jump_size = jax.random.split(key, 4)

normals = jax.random.normal(key_norm, (cfg.base_paths, cfg.steps), dtype=dtype)
jump_counts = jax.random.poisson(key_pois, lam=lam * dt,
                                 shape=(cfg.base_paths, cfg.steps))
jump_directions = jax.random.bernoulli(key_jump_dir, p=p,
                                      shape=(cfg.base_paths, cfg.steps))
exp_up = jax.random.exponential(key_jump_size,
                                shape=(cfg.base_paths, cfg.steps)) / eta1
```

**rough_vol.py (lines 112-119):**
```python
key_z1, key_z2 = jax.random.split(key)

z1 = jax.random.normal(key_z1, (base_paths, cfg.steps), dtype=cfg.dtype)
z2 = jax.random.normal(key_z2, (base_paths, cfg.steps), dtype=cfg.dtype)

if cfg.antithetic:
    z1 = jnp.concatenate([z1, -z1], axis=0)
    z2 = jnp.concatenate([z2, -z2], axis=0)
```

**variance_gamma.py (lines 68-77):**
```python
key_gamma, key_norm = jax.random.split(key)

gamma_increments = jax.random.gamma(
    key_gamma, a=shape, shape=(cfg.base_paths, cfg.steps)
) * scale

normals = jax.random.normal(key_norm, (cfg.base_paths, cfg.steps), dtype=dtype)
```

**Duplication Factor:** 3+ variations of key splitting patterns
**Extractable Utility:** Structured key management utilities

---

## 9. NUMERICAL STABILITY PATTERNS

### Pattern: Safe Square Root with Clipping

**rough_vol.py (line 142):**
```python
sqrt_variance = jnp.sqrt(jnp.maximum(variance_paths[:, :-1], 1e-12))
```

**cir.py (line 216):**
```python
r_t_pos = jnp.maximum(r_t, 1e-10)
```

**jump_diffusion.py (line 76):**
```python
sqrt_var = jnp.sqrt(jnp.maximum(var, 1e-16))
```

**bs.py (lines 10-11):**
```python
vol = jnp.maximum(sigma, 1e-12)
sqrtT = jnp.sqrt(jnp.maximum(T, 1e-12))
```

**Duplication Factor:** 4+ implementations of safe square root
**Extractable Utility:** `safe_sqrt(x, min_val=1e-12)` and `safe_param(param, min_val, param_name)`

---

## 10. IMPLIED VOL / CALIBRATION PATTERNS

### Pattern: Bisection-based rootfinding (bs.py, implicit in calibration)

**bs.py (lines 55-64):**
```python
def implied_vol(S, K, T, r, q, price_target, kind="call", tol=1e-8, max_iter=100):
    lo, hi = 1e-6, 5.0
    for _ in range(max_iter):
        mid = 0.5*(lo+hi)
        p = price(S,K,T,r,q,mid,kind)
        lo, hi = (lo, mid) if p > price_target else (mid, hi)
        if jnp.abs(p - price_target) < tol:
            return mid
    return 0.5*(lo+hi)
```

**jump_diffusion.py (lines 142-205):**
```python
def calibrate_merton(...):
    # Uses optax for optimization instead of bisection
    opt = optax.adam(lr)
    ...
    for _ in range(n_iterations):
        _, grads = jax.value_and_grad(loss)(params_dict)
        updates, opt_state_new = opt.update(grads, opt_state, params_dict)
```

**Duplication Factor:** Different implementations of same concept
**Extractable Utility:** Unified calibration framework

---

## Summary of Extractable Utilities

| Utility Function | Files | Occurrences | Benefit |
|------------------|-------|-------------|---------|
| `validate_positive_param(name, value)` | 3 | 5+ | Consolidate parameter validation |
| `generic_vmap_simulate_paths(simulate_fn, key, n_paths)` | 3 | 3 | Reduce duplicate simulation logic |
| `prepend_initial_value(initial, path)` | 3 | 3 | Array concatenation pattern |
| `discount_payoff(payoffs, rate, maturity)` | 4+ | 4+ | Standardize discounting |
| `compute_payoff(spot, strike, kind)` | 4+ | 8+ | Payoff calculation |
| `build_log_paths(S0, increments, dtype)` | 3 | 3 | Log-normal price paths |
| `compute_bond_price(params, T)` | 3 | 3 | Bond pricing |
| `compute_yield_curve(bond_price_fn, maturities)` | 2 | 2 | Yield calculation |
| `safe_sqrt(x, min_val=1e-12)` | 4+ | 4+ | Numerical stability |
| `safe_parameter(value, min_val, name)` | 4+ | 4+ | Bounds checking |

---

## Code Duplication Severity Score

**High Priority (3+ duplications, moderate complexity):**
- Payoff calculations (8 occurrences across 4+ files)
- Discount factors (4 occurrences across 4+ files)
- Simulate_paths pattern (3 identical implementations)
- Initial value prepending (3 identical implementations)

**Medium Priority (2-3 duplications):**
- Parameter validation (3 files, 5+ checks)
- Bond pricing formulas (3 files, model-specific)
- Yield curve computation (2 identical implementations)

**Lower Priority (specialized, but repetitive):**
- Safe square root patterns
- Log-path building
- Key splitting strategies

---

## Recommended Refactoring Approach

1. **Phase 1 - High Impact, Low Risk:**
   - Extract `compute_payoff()` - reduces 8 similar blocks
   - Extract `discount_payoff()` - consolidates 4 pricing functions
   - Extract `safe_sqrt()` and `safe_parameter()` - improves robustness

2. **Phase 2 - Medium Impact:**
   - Extract parameter validation helpers
   - Generic `vmap_simulate_paths()` wrapper
   - `prepend_initial_value()` helper

3. **Phase 3 - Structural:**
   - Unified bond pricing interface
   - Generic yield curve computation
   - Centralized calibration patterns
