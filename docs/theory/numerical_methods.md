# Numerical Methods: Theoretical Background

This document provides rigorous mathematical foundations for numerical methods used in derivatives pricing and risk management.

## Table of Contents

1. [Monte Carlo Methods](#monte-carlo-methods)
2. [Variance Reduction Techniques](#variance-reduction-techniques)
3. [Quasi-Monte Carlo Methods](#quasi-monte-carlo-methods)
4. [Multi-Level Monte Carlo](#multi-level-monte-carlo)
5. [American Options: Longstaff-Schwartz](#american-options-longstaff-schwartz)
6. [PDE Methods](#pde-methods)
7. [Fourier Methods](#fourier-methods)
8. [Greeks Computation](#greeks-computation)

---

## Monte Carlo Methods

### Fundamental Theorem

**Monte Carlo pricing** is based on the **Law of Large Numbers** and **risk-neutral pricing**:

$$
V_0 = e^{-rT} \mathbb{E}^{\mathbb{Q}}[g(S_T)] \approx e^{-rT} \frac{1}{N} \sum_{i=1}^N g(S_T^{(i)})
$$

where $S_T^{(i)}$ are **independent samples** from the risk-neutral distribution.

**Convergence**: By the **Central Limit Theorem**:

$$
\hat{V}_N \overset{d}{\longrightarrow} \mathcal{N}\left(V_0, \frac{\sigma_V^2}{N}\right)
$$

**Convergence rate**: $O(N^{-1/2})$ (independent of dimension - key advantage)

**Reference**: [Glasserman, 2004], Chapter 2; [Shreve, 2004], Chapter 8

**Implementation**: `src/neutryx/engines/mc.py`

### Standard Error

The **standard error** of the Monte Carlo estimator is:

$$
\text{SE}(\hat{V}_N) = \frac{\hat{\sigma}_V}{\sqrt{N}} = \frac{1}{\sqrt{N}} \sqrt{\frac{1}{N-1} \sum_{i=1}^N (g(S_T^{(i)}) - \hat{V}_N)^2}
$$

**95% confidence interval**:

$$
\hat{V}_N \pm 1.96 \cdot \text{SE}(\hat{V}_N)
$$

**To reduce SE by factor of 10**: Requires $100\times$ more samples (expensive!)

**Solution**: Variance reduction techniques

**Reference**: [Glasserman, 2004], Section 2.3

### Path Simulation

#### Discretization Schemes

For SDE $dX_t = \mu(X_t, t) \, dt + \sigma(X_t, t) \, dW_t$, discretize on $[0, T]$ with time step $\Delta t = T/M$:

**Euler-Maruyama Scheme**:

$$
X_{t+\Delta t} = X_t + \mu(X_t, t) \Delta t + \sigma(X_t, t) \sqrt{\Delta t} \, Z
$$

where $Z \sim \mathcal{N}(0, 1)$.

**Properties**:
- **Strong convergence**: $O(\Delta t^{1/2})$ (path-wise error)
- **Weak convergence**: $O(\Delta t)$ (distribution/expectation error)

**Milstein Scheme** (higher-order):

$$
X_{t+\Delta t} = X_t + \mu \Delta t + \sigma \sqrt{\Delta t} Z + \frac{1}{2} \sigma \frac{\partial \sigma}{\partial x} (Z^2 - 1) \Delta t
$$

**Properties**:
- **Strong convergence**: $O(\Delta t)$
- **Weak convergence**: $O(\Delta t)$

**Reference**: [Glasserman, 2004], Chapter 6; [Kloeden & Platen, 1992]

#### Exact Simulation

For some models, **exact sampling** is possible (no discretization error):

1. **Geometric Brownian Motion**:
   $$
   S_T = S_0 \exp\left((r - \sigma^2/2)T + \sigma\sqrt{T} Z\right)
   $$

2. **Vasicek/Ornstein-Uhlenbeck**:
   $$
   r_T = r_0 e^{-\kappa T} + \theta(1 - e^{-\kappa T}) + \sigma\sqrt{\frac{1-e^{-2\kappa T}}{2\kappa}} Z
   $$

3. **CIR**: Non-central chi-squared distribution

**Advantage**: Eliminates discretization bias, faster (fewer time steps)

**Reference**: [Glasserman, 2004], Section 3.3

---

## Variance Reduction Techniques

### Antithetic Variates

**Idea**: For every path using $Z \sim \mathcal{N}(0, 1)$, also simulate using $-Z$.

**Estimator**:

$$
\hat{V}_{\text{AV}} = \frac{1}{2N} \sum_{i=1}^N \left(g(S_T(Z_i)) + g(S_T(-Z_i))\right)
$$

**Variance reduction**: If $g$ is monotonic in $Z$:

$$
\text{Cov}(g(S_T(Z)), g(S_T(-Z))) < 0 \implies \text{Var}(\hat{V}_{\text{AV}}) < \text{Var}(\hat{V}_{\text{naive}})
$$

**Typical reduction**: 30-50% for European options

**Reference**: [Glasserman, 2004], Section 4.1; [Clewlow & Carverhill, 1994]

**Implementation**: `src/neutryx/engines/variance_reduction.py:antithetic_variates()`

### Control Variates

**Idea**: Use a correlated variable $Y$ with known expectation $\mathbb{E}[Y] = \mu_Y$.

**Estimator**:

$$
\hat{V}_{\text{CV}} = \hat{V} - \beta(\hat{Y} - \mu_Y)
$$

**Optimal coefficient**:

$$
\beta^* = \frac{\text{Cov}(V, Y)}{\text{Var}(Y)}
$$

**Variance reduction**:

$$
\text{Var}(\hat{V}_{\text{CV}}) = \text{Var}(\hat{V})(1 - \rho^2)
$$

where $\rho = \text{Corr}(V, Y)$ is the correlation.

**Example**: Pricing Asian option

- $V$: Asian option (unknown)
- $Y$: Geometric Asian option (analytical formula available)
- Often $\rho > 0.9$ → variance reduction $> 80\%$

**Reference**: [Glasserman, 2004], Section 4.2; [Lavenberg & Welch, 1981]

**Implementation**: `src/neutryx/engines/variance_reduction.py:control_variate()`

### Importance Sampling

**Idea**: Sample from a different distribution $\mathbb{Q}$ to reduce variance in the tail.

**Estimator**:

$$
\mathbb{E}^{\mathbb{P}}[X] = \mathbb{E}^{\mathbb{Q}}\left[X \frac{d\mathbb{P}}{d\mathbb{Q}}\right]
$$

**Optimal measure**: $\mathbb{Q}^*$ such that $X \frac{d\mathbb{P}}{d\mathbb{Q}}$ is constant (zero-variance).

**Application**: Deep out-of-the-money options, rare events (VaR)

**Example**: For OTM call, shift drift to increase probability of $S_T > K$.

**Radon-Nikodym derivative** (Girsanov):

$$
\frac{d\mathbb{P}}{d\mathbb{Q}} = \exp\left(-\int_0^T \theta_s \, dW_s - \frac{1}{2}\int_0^T \theta_s^2 \, ds\right)
$$

**Reference**: [Glasserman, 2004], Chapter 5; [Glasserman et al., 1999]

**Implementation**: `src/neutryx/engines/variance_reduction.py:importance_sampling()`

### Stratified Sampling

**Idea**: Partition sample space into strata, sample proportionally from each.

**Estimator**:

$$
\hat{V}_{\text{strat}} = \sum_{j=1}^K p_j \hat{V}_j
$$

where $p_j$ is the probability of stratum $j$, and $\hat{V}_j$ is the estimate from stratum $j$.

**Variance**:

$$
\text{Var}(\hat{V}_{\text{strat}}) = \sum_{j=1}^K p_j^2 \frac{\sigma_j^2}{n_j}
$$

**Optimal allocation** (Neyman):

$$
n_j \propto p_j \sigma_j
$$

**Application**: Digital options, barrier options

**Reference**: [Glasserman, 2004], Section 4.6

**Implementation**: `src/neutryx/engines/variance_reduction.py:stratified_sampling()`

### Moment Matching

**Idea**: Adjust samples to match theoretical moments (mean, variance).

**Procedure**:
1. Generate samples $Z_1, \ldots, Z_N \sim \mathcal{N}(0, 1)$
2. Transform: $Z_i' = \frac{Z_i - \bar{Z}}{\hat{\sigma}_Z}$ (force mean=0, variance=1)

**Benefit**: Eliminates sampling error in first two moments (especially helpful for small $N$).

**Reference**: [Glasserman, 2004], Section 4.5

**Implementation**: `src/neutryx/engines/variance_reduction.py:moment_matching()`

### Conditional Monte Carlo

**Idea**: Condition on a subset of random variables to reduce variance.

**Formula**:

$$
\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X | Y]]
$$

Since $\text{Var}(\mathbb{E}[X | Y]) \leq \text{Var}(X)$, estimating $\mathbb{E}[X | Y]$ is more efficient.

**Example**: Asian option

Condition on geometric mean (analytical), reduce variance of arithmetic mean estimation.

**Reference**: [Glasserman, 2004], Section 4.3; [Curran, 1994]

**Implementation**: `src/neutryx/engines/variance_reduction.py:conditional_mc()`

---

## Quasi-Monte Carlo Methods

### Motivation

**Monte Carlo** uses **pseudo-random** numbers → $O(N^{-1/2})$ convergence.

**Quasi-Monte Carlo (QMC)** uses **low-discrepancy sequences** → $O((\log N)^d / N)$ convergence (for smooth integrands).

**Key idea**: Fill space more uniformly than random points.

**Reference**: [Glasserman, 2004], Chapter 5; [Niederreiter, 1992]

**Implementation**: `src/neutryx/engines/qmc.py`

### Discrepancy

**Discrepancy** measures how uniformly points $\{x_1, \ldots, x_N\}$ fill $[0,1]^d$:

$$
D_N^* = \sup_{B \subseteq [0,1]^d} \left| \frac{\#\{x_i \in B\}}{N} - \text{Vol}(B) \right|
$$

**Koksma-Hlawka inequality**:

$$
\left| \frac{1}{N} \sum_{i=1}^N f(x_i) - \int_{[0,1]^d} f(x) \, dx \right| \leq V(f) D_N^*
$$

where $V(f)$ is the **variation** of $f$ (Hardy-Krause sense).

**Low-discrepancy sequences**: $D_N^* = O((\log N)^d / N)$

**Reference**: [Niederreiter, 1992]; [Glasserman, 2004], Section 5.1

### Sobol Sequences

**Sobol sequences** are **$(t,s)$-sequences** with excellent discrepancy properties.

**Construction**:
- Based on digital nets and direction numbers
- Use Gray code for efficient generation
- **Scrambling** (Owen, 1995) improves convergence

**Properties**:
- Dimension: Up to $d = 1000+$ (practical limit)
- Discrepancy: $D_N^* = O((\log N)^d / N)$
- **Effective dimensions**: Finance problems often have low effective dimension

**Reference**: [Sobol, 1967]; [Glasserman, 2004], Section 5.2

**Implementation**: `src/neutryx/engines/qmc.py:sobol_sequence()`

### Halton Sequences

**Halton sequences** use **van der Corput sequences** in different prime bases.

**Construction**: For base $b$, reflect digits of $n$ around decimal point.

**Example**: Base 2

| $n$ | Binary | Halton |
|-----|--------|--------|
| 1   | 1      | 0.1 = 0.5 |
| 2   | 10     | 0.01 = 0.25 |
| 3   | 11     | 0.11 = 0.75 |
| 4   | 100    | 0.001 = 0.125 |

**Multidimensional**: Use different primes for each dimension.

**Disadvantage**: Correlation between dimensions for high $d$ (use scrambling).

**Reference**: [Halton, 1960]; [Glasserman, 2004], Section 5.2

**Implementation**: `src/neutryx/engines/qmc.py:halton_sequence()`

### Randomized QMC

**Randomized QMC** combines benefits of QMC (fast convergence) and MC (error estimates):

1. Generate QMC sequence
2. Apply **random shift** (modulo 1)
3. Repeat $M$ times to estimate variance

**Convergence**: $O(N^{-3/2})$ for smooth problems (better than both QMC and MC alone!)

**Reference**: [Owen, 1995]; [L'Ecuyer & Lemieux, 2002]

---

## Multi-Level Monte Carlo

### Motivation

For path-dependent options, we discretize with $M$ time steps → **discretization bias** $O(M^{-1})$.

**Problem**: Fine discretization (large $M$) is expensive.

**Multi-Level MC (MLMC)** [Giles, 2008]: Use multiple levels of refinement, allocate samples optimally.

**Reference**: [Giles, 2008]; [Glasserman, 2004], Section 8.4

**Implementation**: `src/neutryx/engines/qmc.py:multilevel_mc()`

### Telescoping Sum

**Idea**: Estimate expectation at finest level $L$ as a telescoping sum:

$$
\mathbb{E}[V_L] = \mathbb{E}[V_0] + \sum_{\ell=1}^L \mathbb{E}[V_\ell - V_{\ell-1}]
$$

where $V_\ell$ is the estimator with $M_\ell = 2^\ell$ time steps.

**Key insight**: $\text{Var}(V_\ell - V_{\ell-1}) \ll \text{Var}(V_\ell)$ (successive levels are highly correlated).

**MLMC estimator**:

$$
\hat{V}_{\text{MLMC}} = \frac{1}{N_0} \sum_{i=1}^{N_0} V_0^{(i)} + \sum_{\ell=1}^L \frac{1}{N_\ell} \sum_{i=1}^{N_\ell} (V_\ell^{(i)} - V_{\ell-1}^{(i)})
$$

**Coupled paths**: Use the same Brownian increments for $V_\ell$ and $V_{\ell-1}$ (maximizes correlation).

**Reference**: [Giles, 2008], Section 2

### Complexity Analysis

**Standard MC** to achieve MSE $\varepsilon^2$:
- Need $M = O(\varepsilon^{-1})$ time steps (discretization)
- Need $N = O(\varepsilon^{-2})$ samples (variance)
- **Cost**: $O(\varepsilon^{-3})$

**MLMC** with optimal allocation:
- **Cost**: $O(\varepsilon^{-2} (\log \varepsilon)^2)$ (Lipschitz payoffs)
- **Cost**: $O(\varepsilon^{-2})$ (smooth payoffs)

**Speedup**: Factor of $\varepsilon^{-1}$ or more!

**Optimal sample allocation**:

$$
N_\ell \propto \frac{\sqrt{\text{Var}(V_\ell - V_{\ell-1})}}{\sqrt{\text{Cost}(V_\ell)}}
$$

**Reference**: [Giles, 2008], Theorem 1

### Applications

MLMC is highly effective for:
- Path-dependent options (Asians, barriers)
- SDEs with small diffusion coefficient
- PDEs with stochastic coefficients

**Extension**: Multi-index MC (multiple discretization parameters)

**Reference**: [Giles, 2015]

---

## American Options: Longstaff-Schwartz

### Problem Formulation

**American option**: Holder can exercise at any time $\tau \in [0, T]$.

**Value**:

$$
V_0 = \sup_{\tau \in \mathcal{T}} \mathbb{E}^{\mathbb{Q}}[e^{-r\tau} g(S_\tau)]
$$

where $\mathcal{T}$ is the set of stopping times.

**Dynamic programming** (Bellman):

$$
V_t = \max(g(S_t), e^{-r\Delta t} \mathbb{E}^{\mathbb{Q}}[V_{t+\Delta t} | S_t])
$$

**Continuation value**: $C(S_t) = e^{-r\Delta t} \mathbb{E}^{\mathbb{Q}}[V_{t+\Delta t} | S_t]$

**Optimal exercise**: Exercise if $g(S_t) \geq C(S_t)$

**Challenge**: $C(S_t)$ is unknown → estimate via regression.

**Reference**: [Longstaff & Schwartz, 2001]; [Glasserman, 2004], Section 8.6

**Implementation**: `src/neutryx/engines/longstaff_schwartz.py`

### LSM Algorithm

**Longstaff-Schwartz (LSM)** algorithm [Longstaff & Schwartz, 2001]:

1. **Simulate paths**: Generate $N$ paths $\{S_t^{(i)}\}_{i=1}^N$ forward in time.

2. **Backward induction**: Start at maturity $T$, work backward:
   - At time $t_m$:
     - **In-the-money paths**: Paths where $g(S_t^{(i)}) > 0$
     - **Regress**: $V_{t_{m+1}}^{(i)} \sim \sum_{k=1}^K \beta_k \phi_k(S_t^{(i)})$ (basis functions $\phi_k$)
     - **Continuation value**: $C(S_t^{(i)}) = \sum_{k=1}^K \hat{\beta}_k \phi_k(S_t^{(i)})$
     - **Decision**: Exercise if $g(S_t^{(i)}) \geq C(S_t^{(i)})$

3. **Forward pass**: Compute payoff along each path using optimal exercise times.

4. **Estimate**: $\hat{V}_0 = e^{-r\bar{\tau}} \bar{g}(S_{\bar{\tau}})$ (average over paths)

**Reference**: [Longstaff & Schwartz, 2001]; [Glasserman, 2004], Section 8.6

### Basis Functions

Common choices for $\phi_k(S)$:

1. **Power polynomials**: $1, S, S^2, S^3, \ldots$
2. **Laguerre polynomials**: $L_0(S), L_1(S), L_2(S), \ldots$
   $$
   L_n(x) = \frac{e^x}{n!} \frac{d^n}{dx^n}(x^n e^{-x})
   $$
3. **Hermite polynomials**: $H_0(S), H_1(S), H_2(S), \ldots$
4. **Chebyshev polynomials**: $T_0(S), T_1(S), T_2(S), \ldots$

**Guideline**: 3-5 basis functions typically sufficient.

**Reference**: [Longstaff & Schwartz, 2001], Section 2

### Convergence and Bias

**High bias**: $\hat{V}_{\text{LSM}} \leq V_{\text{true}}$ (suboptimal exercise)

**Bias correction** [Andersen & Broadie, 2004]:
- Compute **upper bound** via dual formulation
- Bracket true value: $\text{Lower bound} \leq V \leq \text{Upper bound}$

**Convergence**: As $N, K \to \infty$, $\hat{V}_{\text{LSM}} \to V$

**Reference**: [Longstaff & Schwartz, 2001]; [Andersen & Broadie, 2004]

---

## PDE Methods

### Black-Scholes PDE

Recall the **Black-Scholes PDE**:

$$
\frac{\partial V}{\partial t} + r S \frac{\partial V}{\partial S} + \frac{\sigma^2 S^2}{2} \frac{\partial^2 V}{\partial S^2} - r V = 0
$$

**Terminal condition**: $V(S, T) = g(S)$

**Boundary conditions**: Depend on option type (e.g., $V(0, t) = 0$ for call)

**Reference**: [Wilmott et al., 1995]; [Tavella & Randall, 2000]

**Implementation**: `src/neutryx/models/pde.py`

### Finite Difference Discretization

**Grid**:
- Space: $S_{\min} = S_0, S_1, \ldots, S_N = S_{\max}$
- Time: $t_0 = 0, t_1, \ldots, t_M = T$

**Finite difference approximations**:

$$
\frac{\partial V}{\partial S} \approx \frac{V_{i+1,j} - V_{i-1,j}}{2\Delta S} \quad \text{(central)}
$$

$$
\frac{\partial^2 V}{\partial S^2} \approx \frac{V_{i+1,j} - 2V_{i,j} + V_{i-1,j}}{(\Delta S)^2}
$$

$$
\frac{\partial V}{\partial t} \approx \frac{V_{i,j+1} - V_{i,j}}{\Delta t} \quad \text{(backward)}
$$

**Reference**: [Wilmott et al., 1995], Chapter 10

### Explicit Euler Scheme

**Discretize in time** (forward difference):

$$
\frac{V_{i,j+1} - V_{i,j}}{\Delta t} = r S_i \frac{V_{i+1,j} - V_{i-1,j}}{2\Delta S} + \frac{\sigma^2 S_i^2}{2} \frac{V_{i+1,j} - 2V_{i,j} + V_{i-1,j}}{(\Delta S)^2} - r V_{i,j}
$$

**Explicit update**:

$$
V_{i,j+1} = a_i V_{i-1,j} + b_i V_{i,j} + c_i V_{i+1,j}
$$

**Stability**: Requires $\Delta t \leq \frac{(\Delta S)^2}{\sigma^2 S_{\max}^2}$ (CFL condition) - **very restrictive**!

**Advantage**: Simple, no matrix inversion

**Disadvantage**: Conditionally stable (small time steps required)

**Reference**: [Wilmott et al., 1995], Section 10.4

### Implicit Euler Scheme

**Discretize** using backward differences (implicit):

$$
\frac{V_{i,j+1} - V_{i,j}}{\Delta t} = r S_i \frac{V_{i+1,j+1} - V_{i-1,j+1}}{2\Delta S} + \frac{\sigma^2 S_i^2}{2} \frac{V_{i+1,j+1} - 2V_{i,j+1} + V_{i-1,j+1}}{(\Delta S)^2} - r V_{i,j+1}
$$

**Matrix form**: $A \mathbf{V}_{j+1} = \mathbf{V}_j$

where $A$ is a **tridiagonal matrix** (Thomas algorithm: $O(N)$).

**Stability**: **Unconditionally stable** (can use large time steps)

**Disadvantage**: Requires matrix solve at each step

**Reference**: [Wilmott et al., 1995], Section 10.5; [Tavella & Randall, 2000]

**Implementation**: `src/neutryx/models/pde.py:solve_pde_implicit()`

### Crank-Nicolson Scheme

**Crank-Nicolson** averages explicit and implicit (theta-scheme with $\theta = 1/2$):

$$
\frac{V_{i,j+1} - V_{i,j}}{\Delta t} = \frac{1}{2}\left[\mathcal{L}V_{i,j} + \mathcal{L}V_{i,j+1}\right]
$$

where $\mathcal{L}$ is the spatial operator.

**Properties**:
- **Second-order accurate** in both time and space: $O(\Delta t^2, \Delta S^2)$
- **Unconditionally stable**
- **No oscillations** (for smooth payoffs)

**Standard for European options**

**Reference**: [Wilmott et al., 1995], Section 10.6; [Tavella & Randall, 2000]

**Implementation**: `src/neutryx/models/pde.py:crank_nicolson()`

### American Options (Free Boundary)

**Constraint**: $V(S, t) \geq g(S)$ (immediate exercise value)

**Discretization**: At each time step, enforce:

$$
V_{i,j} = \max(g(S_i), V_{i,j}^{\text{PDE}})
$$

**Projected SOR**: Solve PDE subject to constraint using successive over-relaxation.

**Reference**: [Wilmott et al., 1995], Chapter 11; [Tavella & Randall, 2000], Chapter 5

**Implementation**: `src/neutryx/models/pde.py:american_option_pde()`

### Multidimensional PDEs

For **basket options** or **multi-asset** derivatives:

$$
\frac{\partial V}{\partial t} + \frac{1}{2} \sum_{i,j} \rho_{ij} \sigma_i \sigma_j S_i S_j \frac{\partial^2 V}{\partial S_i \partial S_j} + \cdots = 0
$$

**Curse of dimensionality**: Grid size grows exponentially ($N^d$ for $d$ assets).

**ADI (Alternating Direction Implicit)**: Splits multi-D problem into sequence of 1D problems.

**Complexity**: $O(MN^d)$ instead of $O(MN^{2d})$ (huge savings)

**Reference**: [Wilmott et al., 1995], Chapter 12; [Tavella & Randall, 2000], Chapter 6

**Implementation**: `src/neutryx/models/pde.py:adi_2d()`

---

## Fourier Methods

### Carr-Madan Formula

**Carr and Madan (1999)** derived an FFT-based pricing formula using the **characteristic function**.

**Modified call price** (with damping $\alpha > 0$):

$$
C_\alpha(k) = e^{\alpha k} C(K), \quad k = \log K
$$

**Fourier transform**:

$$
\psi_T(v) = \int_{-\infty}^\infty e^{ivk} C_\alpha(k) \, dk = \frac{e^{-rT} \phi_T(v - (\alpha+1)i)}{(\alpha + iv)(\alpha + 1 + iv)}
$$

where $\phi_T(u) = \mathbb{E}[e^{iu \log S_T}]$ is the characteristic function.

**Inverse Fourier transform**:

$$
C(K) = \frac{e^{-\alpha k}}{\pi} \int_0^\infty e^{-ivk} \psi_T(v) \, dv
$$

**FFT**: Discretize and apply FFT → $O(N \log N)$ for $N$ strikes.

**Reference**: [Carr & Madan, 1999]; [Gatheral, 2006], Chapter 2

**Implementation**: `src/neutryx/engines/fourier.py:carr_madan_fft()`

### COS Method

**Fang and Oosterlee (2008)** developed the **Fourier-cosine (COS) method** with superior stability.

**Idea**: Expand the density in cosine series:

$$
f(x) \approx \sum_{k=0}^{N-1} A_k \cos\left(k\pi \frac{x-a}{b-a}\right)
$$

**Option price**:

$$
V = \sum_{k=0}^{N-1} F_k(a, b) \operatorname{Re}\left[\phi_T\left(\frac{k\pi}{b-a}\right)\right]
$$

where $F_k$ are known analytically for European options.

**Truncation**: $[a, b]$ chosen based on cumulants:

$$
a = \mathbb{E}[\log S_T] - L \sqrt{\text{Var}(\log S_T)}, \quad b = \mathbb{E}[\log S_T] + L \sqrt{\text{Var}(\log S_T)}
$$

($L = 10$ typically)

**Advantages**:
- Fewer evaluations than Carr-Madan
- Better numerical stability
- Easier to implement

**Reference**: [Fang & Oosterlee, 2008]; [Fang & Oosterlee, 2009]

**Implementation**: `src/neutryx/engines/fourier.py:cos_method()`

### Applicability

Fourier methods work for **any model with known characteristic function**:
- Heston, SABR (via numerical integration)
- Merton, Kou, Variance Gamma
- Affine jump-diffusions
- Time-changed Lévy processes

**Limitation**: Requires smooth characteristic function (no path-dependence in payoff).

**Reference**: [Carr & Madan, 1999]; [Fang & Oosterlee, 2008]

---

## Greeks Computation

### Finite Differences (Bump-and-Revalue)

**Delta** (first derivative):

$$
\Delta \approx \frac{V(S_0 + h) - V(S_0 - h)}{2h} \quad \text{(central difference)}
$$

**Gamma** (second derivative):

$$
\Gamma \approx \frac{V(S_0 + h) - 2V(S_0) + V(S_0 - h)}{h^2}
$$

**Choice of $h$**: Balance truncation error ($O(h^2)$) and rounding error ($O(\varepsilon/h^2)$).

**Optimal $h$**: $h \approx \varepsilon^{1/3} S_0$ (where $\varepsilon$ is machine precision).

**Disadvantages**:
- Multiple pricing evaluations (expensive for MC)
- Noisy for MC (requires variance reduction)

**Reference**: [Glasserman, 2004], Chapter 7

### Pathwise Derivatives

**Pathwise method** differentiates the **simulated path** with respect to parameters.

**Assumption**: Payoff $g(S_T)$ is **Lipschitz continuous** (not discontinuous).

**Formula**:

$$
\frac{\partial}{\partial S_0} \mathbb{E}[g(S_T)] = \mathbb{E}\left[\frac{\partial g}{\partial S_T} \frac{\partial S_T}{\partial S_0}\right]
$$

**For GBM**:

$$
\frac{\partial S_T}{\partial S_0} = \frac{S_T}{S_0}
$$

**Delta**:

$$
\Delta = e^{-rT} \mathbb{E}\left[g'(S_T) \frac{S_T}{S_0}\right]
$$

**Advantages**:
- **Unbiased** (if payoff is smooth)
- **Low variance** (same simulation as pricing)
- **No additional samples** required

**Disadvantages**:
- Requires smooth payoff (fails for digitals, barriers)

**Reference**: [Glasserman, 2004], Section 7.2; [Broadie & Glasserman, 1996]

**Implementation**: `src/neutryx/engines/pathwise.py`

### Likelihood Ratio Method (LRM)

**LRM** (also called **score function method**) works for **discontinuous payoffs**.

**Idea**: Differentiate the **density** instead of the payoff.

$$
\frac{\partial}{\partial \theta} \mathbb{E}[g(S_T)] = \mathbb{E}\left[g(S_T) \frac{\partial \log f(S_T; \theta)}{\partial \theta}\right]
$$

where $f(S_T; \theta)$ is the density of $S_T$.

**Score function**: $\psi(\theta) = \frac{\partial \log f}{\partial \theta}$

**Example (GBM)**: For $\theta = \sigma$ (vega):

$$
\psi(\sigma) = \frac{1}{\sigma}\left[\frac{(\log(S_T/S_0) - (r - \sigma^2/2)T)^2}{\sigma^2 T} - 1\right]
$$

**Advantages**:
- Works for discontinuous payoffs (digitals, barriers)
- Unbiased

**Disadvantages**:
- **High variance** (weights can be large)
- Requires knowledge of density

**Reference**: [Glasserman, 2004], Section 7.3; [Broadie & Glasserman, 1996]

### Automatic Differentiation (JAX)

**JAX** provides **automatic differentiation** (autodiff) via:

1. **Forward mode** (tangent): Efficient for $\mathbb{R} \to \mathbb{R}^m$
2. **Reverse mode** (adjoint): Efficient for $\mathbb{R}^n \to \mathbb{R}$

**Greeks via autodiff**:

```python
import jax
delta = jax.grad(price_function, argnums=0)(S0)  # ∂V/∂S0
gamma = jax.grad(jax.grad(price_function))(S0)  # ∂²V/∂S0²
```

**Advantages**:
- **Exact derivatives** (up to numerical precision)
- **Automatic** (no manual derivation)
- **Fast** (same complexity as function evaluation, up to small constant)

**Disadvantages**:
- Requires differentiable code (no if/else based on computed values)
- Memory overhead for reverse mode

**Reference**: [Bradbury et al., 2018]; [Griewank & Walther, 2008]

**Implementation**: All Greeks in `src/neutryx/valuations/greeks/` use JAX autodiff

---

## Summary

This document covered:

1. **Monte Carlo**: Fundamental MC, convergence, standard error
2. **Variance Reduction**: Antithetic, control variates, importance sampling, stratified, moment matching, conditional MC
3. **Quasi-Monte Carlo**: Low-discrepancy sequences (Sobol, Halton), randomized QMC
4. **MLMC**: Telescoping estimator, optimal allocation, complexity reduction
5. **LSM**: Regression-based American option pricing, basis functions
6. **PDE**: Finite differences, Crank-Nicolson, ADI for multi-asset
7. **Fourier**: Carr-Madan FFT, COS method, characteristic functions
8. **Greeks**: Finite differences, pathwise, LRM, automatic differentiation

All methods are rigorously implemented in Neutryx with extensive validation.

**Next**: [Calibration Theory](calibration_theory.md) | [Risk Metrics Theory](risk_theory.md)

---

**References**: See [Bibliography](../references.md) for complete citations.
