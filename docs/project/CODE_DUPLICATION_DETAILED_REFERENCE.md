# Detailed Code Duplication Reference
## neutryx-core/src/neutryx/models

This document provides exact line-by-line references for code duplication patterns that can be extracted into utilities.

---

## 1. OPTION PAYOFF COMPUTATION

### Duplication Instance 1.1: kou.py lines 179-184
**File:** `/workspaces/neutryx-core/src/neutryx/models/kou.py`
**Function:** `price_vanilla_kou_mc()`
**Lines:** 179-184

```python
if kind == "call":
    payoffs = jnp.maximum(ST - K, 0.0)
elif kind == "put":
    payoffs = jnp.maximum(K - ST, 0.0)
else:
    raise ValueError(f"Unknown option kind: {kind}")
```

### Duplication Instance 1.2: variance_gamma.py lines 154-159
**File:** `/workspaces/neutryx-core/src/neutryx/models/variance_gamma.py`
**Function:** `price_vanilla_vg_mc()`
**Lines:** 154-159

```python
if kind == "call":
    payoffs = jnp.maximum(ST - K, 0.0)
elif kind == "put":
    payoffs = jnp.maximum(K - ST, 0.0)
else:
    raise ValueError(f"Unknown option kind: {kind}")
```

**STATUS:** IDENTICAL - Direct copy-paste duplication

### Duplication Instance 1.3: rough_vol.py lines 241
**File:** `/workspaces/neutryx-core/src/neutryx/models/rough_vol.py`
**Function:** `price_european_call_mc()` (second definition, lines 208-243)
**Lines:** 241

```python
payoffs = jnp.maximum(ST - K, 0.0)
```

**STATUS:** PARTIAL - Call payoff only, inline, no error handling

### Duplication Instance 1.4: jump_diffusion.py lines 102-103
**File:** `/workspaces/neutryx-core/src/neutryx/models/jump_diffusion.py`
**Function:** `merton_jump_call()`
**Lines:** 102-103

```python
if T <= 0:
    return jnp.maximum(S0 - K, 0.0)
```

**STATUS:** PARTIAL - Call payoff inline, with early return condition

---

## 2. DISCOUNT FACTOR & PRESENT VALUE

### Duplication Instance 2.1: variance_gamma.py lines 161-162
**File:** `/workspaces/neutryx-core/src/neutryx/models/variance_gamma.py`
**Function:** `price_vanilla_vg_mc()`
**Lines:** 161-162

```python
discount = jnp.exp(-r * T)
return float((discount * payoffs).mean())
```

### Duplication Instance 2.2: kou.py lines 186-187
**File:** `/workspaces/neutryx-core/src/neutryx/models/kou.py`
**Function:** `price_vanilla_kou_mc()`
**Lines:** 186-187

```python
discount = jnp.exp(-r * T)
return float((discount * payoffs).mean())
```

**STATUS:** IDENTICAL - Direct copy-paste, different models

### Duplication Instance 2.3: rough_vol.py lines 204-205
**File:** `/workspaces/neutryx-core/src/neutryx/models/rough_vol.py`
**Function:** `price_european_call_mc()` (second definition)
**Lines:** 204-205

```python
discount = jnp.exp(-r * T)
return discount * payoff.mean(axis=0)
```

**STATUS:** SIMILAR - Same discount calculation, different return type

### Duplication Instance 2.4: jump_diffusion.py line 84
**File:** `/workspaces/neutryx-core/src/neutryx/models/jump_diffusion.py`
**Function:** `_lognormal_call_terms()`
**Line:** 84

```python
discounted = jnp.exp(-r * T) * weights * payoff
```

**STATUS:** SIMILAR - Discount component in larger expression

---

## 3. SIMULATE_PATHS GENERIC PATTERN

### Duplication Instance 3.1: vasicek.py lines 200-235
**File:** `/workspaces/neutryx-core/src/neutryx/models/vasicek.py`
**Function:** `simulate_paths()`
**Lines:** 200-235

```python
def simulate_paths(
    params: VasicekParams,
    T: float,
    n_steps: int,
    n_paths: int,
    key: jax.random.KeyArray,
    method: str = "exact"
) -> jnp.ndarray:
    """Simulate multiple paths of the Vasicek short rate process."""
    keys = jax.random.split(key, n_paths)

    def sim_single_path(k):
        return simulate_path(params, T, n_steps, k, method=method)

    return jax.vmap(sim_single_path)(keys)
```

### Duplication Instance 3.2: cir.py lines 249-289
**File:** `/workspaces/neutryx-core/src/neutryx/models/cir.py`
**Function:** `simulate_paths()`
**Lines:** 249-289

```python
def simulate_paths(
    params: CIRParams,
    T: float,
    n_steps: int,
    n_paths: int,
    key: jax.random.KeyArray,
    method: str = "euler"  # Different default
) -> jnp.ndarray:
    """Simulate multiple paths of the CIR short rate process."""
    keys = jax.random.split(key, n_paths)

    def sim_single_path(k):
        return simulate_path(params, T, n_steps, k, method=method)

    return jax.vmap(sim_single_path)(keys)
```

### Duplication Instance 3.3: hull_white.py lines 216-251
**File:** `/workspaces/neutryx-core/src/neutryx/models/hull_white.py`
**Function:** `simulate_paths()`
**Lines:** 216-251

```python
def simulate_paths(
    params: HullWhiteParams,
    T: float,
    n_steps: int,
    n_paths: int,
    key: jax.random.KeyArray,
    method: str = "exact"
) -> jnp.ndarray:
    """Simulate multiple paths of the Hull-White short rate process."""
    keys = jax.random.split(key, n_paths)

    def sim_single_path(k):
        return simulate_path(params, T, n_steps, k, method=method)

    return jax.vmap(sim_single_path)(keys)
```

**STATUS:** IDENTICAL STRUCTURE - Different parameter types, same logic
**CORE DUPLICATION (lines 284-289 / 246-251 / 230-234):**
```python
keys = jax.random.split(key, n_paths)

def sim_single_path(k):
    return simulate_path(params, T, n_steps, k, method=method)

return jax.vmap(sim_single_path)(keys)
```

---

## 4. LOG-NORMAL PATH CONSTRUCTION

### Duplication Instance 4.1: variance_gamma.py lines 94-101
**File:** `/workspaces/neutryx-core/src/neutryx/models/variance_gamma.py`
**Function:** `simulate_variance_gamma()`
**Lines:** 94-101

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

### Duplication Instance 4.2: kou.py lines 113-121
**File:** `/workspaces/neutryx-core/src/neutryx/models/kou.py`
**Function:** `simulate_kou()`
**Lines:** 113-121

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

**STATUS:** IDENTICAL CODE - Direct copy between variance_gamma and kou

### Duplication Instance 4.3: rough_vol.py lines 149-155
**File:** `/workspaces/neutryx-core/src/neutryx/models/rough_vol.py`
**Function:** `simulate_rough_bergomi()`
**Lines:** 149-155

```python
log_S0 = jnp.log(S0)  # Note: No asarray wrapping
cumulative = jnp.cumsum(log_increments, axis=1)
log_paths = jnp.concatenate(
    [jnp.full((log_increments.shape[0], 1), log_S0, dtype=cfg.dtype), log_S0 + cumulative],
    axis=1,
)
price_paths = jnp.exp(log_paths)
```

**STATUS:** SIMILAR - Same pattern, minor dtype handling differences

---

## 5. INITIAL VALUE PATH PREPENDING

### Duplication Instance 5.1: vasicek.py line 197
**File:** `/workspaces/neutryx-core/src/neutryx/models/vasicek.py`
**Function:** `simulate_path()`
**Line:** 197

```python
return jnp.concatenate([jnp.array([r0]), r_path])
```

### Duplication Instance 5.2: cir.py line 246
**File:** `/workspaces/neutryx-core/src/neutryx/models/cir.py`
**Function:** `simulate_path()`
**Line:** 246

```python
return jnp.concatenate([jnp.array([r0]), r_path])
```

### Duplication Instance 5.3: hull_white.py line 213
**File:** `/workspaces/neutryx-core/src/neutryx/models/hull_white.py`
**Function:** `simulate_path()`
**Line:** 213

```python
return jnp.concatenate([jnp.array([r0]), r_path])
```

**STATUS:** IDENTICAL - 3 identical single-line implementations
**CONSOLIDATION BENEFIT:** High readability improvement

---

## 6. PARAMETER VALIDATION PATTERNS

### Duplication Instance 6.1: vasicek.py lines 51-56
**File:** `/workspaces/neutryx-core/src/neutryx/models/vasicek.py`
**Class:** `VasicekParams`
**Method:** `__post_init__()`
**Lines:** 51-56

```python
def __post_init__(self):
    """Validate parameters."""
    if self.a <= 0:
        raise ValueError(f"Mean reversion speed a must be positive, got {self.a}")
    if self.sigma <= 0:
        raise ValueError(f"Volatility sigma must be positive, got {self.sigma}")
```

### Duplication Instance 6.2: cir.py lines 53-62
**File:** `/workspaces/neutryx-core/src/neutryx/models/cir.py`
**Class:** `CIRParams`
**Method:** `__post_init__()`
**Lines:** 53-62

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

### Duplication Instance 6.3: hull_white.py lines 53-58
**File:** `/workspaces/neutryx-core/src/neutryx/models/hull_white.py`
**Class:** `HullWhiteParams`
**Method:** `__post_init__()`
**Lines:** 53-58

```python
def __post_init__(self):
    """Validate parameters."""
    if self.a <= 0:
        raise ValueError(f"Mean reversion speed a must be positive, got {self.a}")
    if self.sigma <= 0:
        raise ValueError(f"Volatility sigma must be positive, got {self.sigma}")
```

**STATUS:** REPETITIVE - Same validation pattern across 3 dataclasses
**PATTERN ANALYSIS:**
- All validate: `a` (mean reversion speed) > 0
- All validate: `sigma` (volatility) > 0
- Additional checks vary per model

---

## 7. YIELD CURVE COMPUTATION

### Duplication Instance 7.1: vasicek.py lines 120-124
**File:** `/workspaces/neutryx-core/src/neutryx/models/vasicek.py`
**Function:** `yield_curve()`
**Lines:** 120-124

```python
def yield_curve(params: VasicekParams, maturities: jnp.ndarray) -> jnp.ndarray:
    """Calculate zero-coupon yield curve under Vasicek model."""
    def yield_at_maturity(T):
        P_T = zero_coupon_bond_price(params, T)
        return -jnp.log(P_T) / T

    return jax.vmap(yield_at_maturity)(maturities)
```

### Duplication Instance 7.2: cir.py lines 145-149
**File:** `/workspaces/neutryx-core/src/neutryx/models/cir.py`
**Function:** `yield_curve()`
**Lines:** 145-149

```python
def yield_curve(params: CIRParams, maturities: jnp.ndarray) -> jnp.ndarray:
    """Calculate zero-coupon yield curve under CIR model."""
    def yield_at_maturity(T):
        P_T = zero_coupon_bond_price(params, T)
        return -jnp.log(P_T) / T

    return jax.vmap(yield_at_maturity)(maturities)
```

**STATUS:** IDENTICAL LOGIC - Only parameter type differs (VasicekParams vs CIRParams)
**CORE DUPLICATION:**
```python
def yield_at_maturity(T):
    P_T = zero_coupon_bond_price(params, T)
    return -jnp.log(P_T) / T
return jax.vmap(yield_at_maturity)(maturities)
```

---

## 8. BOND PRICE CALCULATIONS

### Pattern Analysis: Bond Price Formula Structure

All three models follow pattern: `P(0,T) = A(T) * exp(-B(T) * r0)`

**Vasicek** (lines 88-99):
```python
a, b, sigma, r0 = params.a, params.b, params.sigma, params.r0
B_T = (1.0 - jnp.exp(-a * T)) / a
term1 = (B_T - T) * (a * a * b - sigma * sigma / 2.0) / (a * a)
term2 = -sigma * sigma * B_T * B_T / (4.0 * a)
A_T = jnp.exp(term1 + term2)
return A_T * jnp.exp(-B_T * r0)
```

**CIR** (lines 105-127):
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

**HullWhite** (lines 91-114):
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

**STATUS:** MODEL-SPECIFIC but structurally similar
**SHARED PATTERN:**
1. Extract model parameters
2. Calculate time-dependent B(T)
3. Calculate time-dependent A(T) with model-specific formula
4. Return discounted bond price

---

## 9. SAFE SQUARE ROOT WITH CLIPPING

### Duplication Instance 9.1: bs.py lines 10-11
**File:** `/workspaces/neutryx-core/src/neutryx/models/bs.py`
**Function:** `_d1d2()`
**Lines:** 10-11

```python
vol = jnp.maximum(sigma, 1e-12)
sqrtT = jnp.sqrt(jnp.maximum(T, 1e-12))
```

### Duplication Instance 9.2: jump_diffusion.py line 76
**File:** `/workspaces/neutryx-core/src/neutryx/models/jump_diffusion.py`
**Function:** `_lognormal_call_terms()`
**Line:** 76

```python
sqrt_var = jnp.sqrt(jnp.maximum(var, 1e-16))
```

### Duplication Instance 9.3: cir.py line 216
**File:** `/workspaces/neutryx-core/src/neutryx/models/cir.py`
**Function:** `simulate_path()` - Euler method
**Line:** 216

```python
r_t_pos = jnp.maximum(r_t, 1e-10)
```

### Duplication Instance 9.4: rough_vol.py line 142
**File:** `/workspaces/neutryx-core/src/neutryx/models/rough_vol.py`
**Function:** `simulate_rough_bergomi()`
**Line:** 142

```python
sqrt_variance = jnp.sqrt(jnp.maximum(variance_paths[:, :-1], 1e-12))
```

**STATUS:** REPEATED PATTERN
**PATTERN:** `safe_value = jnp.maximum(value, min_threshold)`
**MIN VALUES USED:** 1e-10, 1e-12, 1e-16
**CONSOLIDATION:** Could extract to `safe_sqrt(x, min_val=1e-12)`

---

## 10. DUPLICATE FUNCTION DEFINITIONS

### Instance 10.1: rough_vol.py - Duplicate price_european_call_mc()
**File:** `/workspaces/neutryx-core/src/neutryx/models/rough_vol.py`
**Lines:** 189-205 AND 208-242

```python
# First definition (lines 189-205)
def price_european_call_mc(
    key: jax.random.KeyArray,
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    cfg: MCConfig,
    params: RoughBergomiParams,
) -> Array:
    """Monte Carlo price of a European call under Rough Bergomi dynamics."""
    paths = simulate_rough_bergomi(key, S0, T, r, q, cfg, params)
    ST = paths[:, -1]
    payoff = jnp.maximum(ST - K, 0.0)
    discount = jnp.exp(-r * T)
    return discount * payoff.mean(axis=0)


# Second definition (lines 208-242) - OVERWRITES FIRST
def price_european_call_mc(
    key: jax.random.KeyArray,
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    cfg: MCConfig,
    params: RoughBergomiParams,
) -> Array:
    """Price European call under rough Bergomi via Monte Carlo."""
    from ..core.engine import present_value
    
    paths = simulate_rough_bergomi(
        key, S0, T, r, q, cfg, params, return_full=False
    )
    ST = paths[:, -1]
    payoffs = jnp.maximum(ST - K, 0.0)
    return present_value(payoffs, jnp.array(T), r)
```

**STATUS:** CRITICAL - Function is defined twice with same signature
**ISSUE:** Second definition overwrites first; unclear intent
**RESOLUTION:** One definition should be removed or refactored

---

## SUMMARY TABLE: All Duplication Instances

| Category | File 1 | Lines | File 2 | Lines | Status | Severity |
|----------|--------|-------|--------|-------|--------|----------|
| Payoff | kou.py | 179-184 | variance_gamma.py | 154-159 | IDENTICAL | HIGH |
| Discount | variance_gamma.py | 161-162 | kou.py | 186-187 | IDENTICAL | HIGH |
| Simulate_paths | vasicek.py | 200-235 | cir.py | 249-289 | IDENTICAL LOGIC | HIGH |
| Simulate_paths | vasicek.py | 200-235 | hull_white.py | 216-251 | IDENTICAL LOGIC | HIGH |
| Log-paths | variance_gamma.py | 94-101 | kou.py | 113-121 | IDENTICAL | HIGH |
| Prepend init | vasicek.py | 197 | cir.py | 246 | IDENTICAL | HIGH |
| Prepend init | vasicek.py | 197 | hull_white.py | 213 | IDENTICAL | HIGH |
| Yield curve | vasicek.py | 120-124 | cir.py | 145-149 | IDENTICAL | MEDIUM |
| Validation | vasicek.py | 51-56 | cir.py | 53-62 | REPETITIVE | MEDIUM |
| Safe sqrt | bs.py | 10-11 | jump_diffusion.py | 76 | PATTERN | MEDIUM |
| Duplicate def | rough_vol.py | 189-205 | rough_vol.py | 208-242 | OVERWRITE | CRITICAL |

