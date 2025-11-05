# Model Reference Compendium

Comprehensive reference documentation for all pricing models implemented in Neutryx, with mathematical specifications, parameters, implementation details, and validation notes.

**For theoretical foundations and mathematical derivations, see [Pricing Models Theory](../../theory/pricing_models.md).**

---

## Table of Contents

- [Equity Models](#equity-models)
  - [Black-Scholes-Merton](#black-scholes-merton)
  - [Heston Stochastic Volatility](#heston-stochastic-volatility)
  - [SABR Model](#sabr-model)
  - [Rough Bergomi](#rough-bergomi)
  - [Local Volatility (Dupire)](#local-volatility-dupire)
  - [Jump-Diffusion Models](#jump-diffusion-models)
- [Interest Rate Models](#interest-rate-models)
  - [Vasicek](#vasicek)
  - [Cox-Ingersoll-Ross (CIR)](#cox-ingersoll-ross-cir)
  - [Hull-White](#hull-white)
- [References](#references)

---

## Equity Models

### Black-Scholes-Merton

#### Overview

The foundational model for European option pricing, developed by Black, Scholes, and Merton in 1973.

**Nobel Prize**: Scholes and Merton, 1997

**Key References**:
- [Black & Scholes, 1973] "The Pricing of Options and Corporate Liabilities"
- [Merton, 1973] "Theory of Rational Option Pricing"
- See [Theory: Black-Scholes-Merton Model](../../theory/pricing_models.md#black-scholes-merton-model)

#### Model Specification

Under risk-neutral measure $\mathbb{Q}$:

$$
dS_t = r S_t \, dt + \sigma S_t \, dW_t^{\mathbb{Q}}
$$

**Parameters**:
- $S_t$: Stock price at time $t$
- $r$: Risk-free interest rate (constant)
- $\sigma$: Volatility (constant)
- $W_t^{\mathbb{Q}$: Brownian motion under risk-neutral measure

**Solution** (Geometric Brownian Motion):

$$
S_t = S_0 \exp\left(\left(r - \frac{\sigma^2}{2}\right)t + \sigma W_t^{\mathbb{Q}}\right)
$$

#### Option Pricing Formulas

**European Call**:

$$
C(S_0, K, T) = S_0 \Phi(d_1) - K e^{-rT} \Phi(d_2)
$$

**European Put**:

$$
P(S_0, K, T) = K e^{-rT} \Phi(-d_2) - S_0 \Phi(-d_1)
$$

where:

$$
d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}
$$

and $\Phi(\cdot)$ is the standard normal CDF.

**Put-Call Parity**:

$$
C - P = S_0 - K e^{-rT}
$$

#### Greeks (Analytical)

| Greek | Formula | Description |
|-------|---------|-------------|
| **Delta** | $\Phi(d_1)$ (call), $-\Phi(-d_1)$ (put) | Price sensitivity to underlying |
| **Gamma** | $\frac{\phi(d_1)}{S_0 \sigma \sqrt{T}}$ | Curvature (convexity) |
| **Vega** | $S_0 \phi(d_1) \sqrt{T}$ | Sensitivity to volatility |
| **Theta** | $-\frac{S_0 \phi(d_1) \sigma}{2\sqrt{T}} - r K e^{-rT} \Phi(d_2)$ | Time decay |
| **Rho** | $K T e^{-rT} \Phi(d_2)$ | Interest rate sensitivity |

where $\phi(\cdot)$ is the standard normal PDF.

#### Implementation

**File**: `src/neutryx/models/bs.py`

**Key Functions**:
- `price_european_call(S, K, T, r, sigma)`: Analytical call price
- `price_european_put(S, K, T, r, sigma)`: Analytical put price
- `implied_volatility(price, S, K, T, r, option_type)`: IV solver (bisection)
- `delta(S, K, T, r, sigma, option_type)`: Delta computation
- `gamma(S, K, T, r, sigma)`: Gamma computation

**Validation**:
- Analytical formulas verified against Hull textbook examples
- Put-call parity enforced to machine precision
- Greeks validated via finite difference comparison

#### Model Limitations

1. **Constant volatility**: Market exhibits volatility smile/skew
2. **Log-normal distribution**: Underestimates tail risk (fat tails empirically observed)
3. **Continuous trading**: Ignores transaction costs and discrete hedging
4. **No jumps**: Cannot capture gap risk or earnings surprises

**Extensions**: See Heston, SABR (stochastic volatility), Merton/Kou (jumps), Local Volatility (smile fitting)

---

### Heston Stochastic Volatility

#### Overview

The Heston model (1993) is the most popular stochastic volatility model with semi-analytical solutions via characteristic functions.

**Key References**:
- [Heston, 1993] "A Closed-Form Solution for Options with Stochastic Volatility"
- [Gatheral, 2006] *The Volatility Surface*, Chapter 3
- See [Theory: Heston Model](../../theory/pricing_models.md#heston-model)

#### Model Specification

$$
\begin{align}
dS_t &= r S_t \, dt + \sqrt{v_t} S_t \, dW_t^S \\
dv_t &= \kappa(\theta - v_t) \, dt + \sigma_v \sqrt{v_t} \, dW_t^v \\
\langle dW_t^S, dW_t^v \rangle &= \rho \, dt
\end{align}
$$

**Parameters**:

| Parameter | Symbol | Range | Description |
|-----------|--------|-------|-------------|
| Initial variance | $v_0$ | $> 0$ | Spot variance (e.g., 0.04 = 20% vol) |
| Mean-reversion speed | $\kappa$ | $> 0$ | Speed of reversion to $\theta$ |
| Long-term variance | $\theta$ | $> 0$ | Equilibrium variance level |
| Volatility of volatility | $\sigma_v$ | $> 0$ | Vol-of-vol parameter |
| Correlation | $\rho$ | $[-1, 1]$ | Price-vol correlation (typically negative) |

**Feller Condition**: To ensure $v_t > 0$ almost surely:

$$
2\kappa\theta > \sigma_v^2
$$

If violated, variance can reach zero (boundary attainable).

#### Characteristic Function

The characteristic function of $\log S_T$ is:

$$
\phi_T(u) = \mathbb{E}^{\mathbb{Q}}[e^{i u \log S_T}] = \exp\left(C(\tau, u) + D(\tau, u) v_t + i u \log S_t\right)
$$

where $\tau = T - t$ and $C, D$ solve Riccati ODEs with analytical solutions.

**Application**: Use Carr-Madan FFT or COS method for fast pricing across many strikes.

#### Simulation Schemes

**Euler Scheme** (simple, can produce negative variance):

$$
v_{t+\Delta t} = v_t + \kappa(\theta - v_t) \Delta t + \sigma_v \sqrt{v_t^+} \sqrt{\Delta t} Z^v
$$

**Quadratic-Exponential (QE) Scheme** [Andersen, 2008] (recommended):
- Preserves non-negativity
- Maintains first two moments exactly
- Adaptive: quadratic for small $v$, exponential for large $v$

**Implementation**: QE scheme is default in `simulate()` method.

#### Calibration

**Target**: Implied volatility surface (strikes × maturities)

**Loss Function** (vega-weighted IV error):

$$
\mathcal{L}(\boldsymbol{\theta}) = \sum_{i,j} w_{ij} \left(\sigma_{ij}^{\text{market}} - \sigma_{ij}^{\text{Heston}}(\boldsymbol{\theta})\right)^2
$$

**Typical Parameter Ranges** (equity):
- $v_0 \in [0.01, 0.1]$ (10%-30% spot vol)
- $\kappa \in [0.5, 5]$ (half-life of 0.14-1.4 years)
- $\theta \in [0.01, 0.1]$ (10%-30% long-term vol)
- $\sigma_v \in [0.1, 1]$ (vol-of-vol)
- $\rho \in [-0.9, -0.5]$ (negative for equities, leverage effect)

**Optimizer**: Adam (via optax) with parameter transformations

#### Volatility Smile Features

- **Vol-of-vol** ($\sigma_v$): Controls smile curvature (ATM convexity)
- **Correlation** ($\rho$): Creates skew (negative $\rho$ → left skew for puts)
- **Mean-reversion** ($\kappa$): Flattens long-dated smile (mean-reversion dominates)

**Stylized Fact**: Equity markets exhibit $\rho \approx -0.7$ (leverage effect: stock drops → volatility rises)

#### Implementation

**File**: `src/neutryx/models/heston.py`

**Key Functions**:
- `characteristic_function(u, S0, v0, r, T, kappa, theta, sigma_v, rho)`: CF for FFT pricing
- `price_european_fft(S0, K, T, r, v0, kappa, theta, sigma_v, rho)`: FFT pricing (Carr-Madan)
- `simulate(S0, v0, T, r, kappa, theta, sigma_v, rho, n_steps, scheme='qe')`: Path simulation
- `calibrate(market_data, initial_guess)`: Calibration routine

**Calibration File**: `src/neutryx/calibration/heston.py`

**Validation**:
- Characteristic function verified against [Heston, 1993] Appendix A
- FFT prices match reference implementations (QuantLib)
- QE scheme preserves moments (tested against exact solution for simpler cases)
- Calibration tested on SPX surface

---

### SABR Model

#### Overview

The SABR (Stochastic Alpha Beta Rho) model, introduced by Hagan et al. (2002), is the industry standard for interest rate smile modeling.

**Key References**:
- [Hagan et al., 2002] "Managing Smile Risk"
- [Gatheral, 2006] *The Volatility Surface*, Chapter 4
- See [Theory: SABR Model](../../theory/pricing_models.md#sabr-model)

#### Model Specification

$$
\begin{align}
dF_t &= \alpha_t F_t^\beta \, dW_t^F \\
d\alpha_t &= \nu \alpha_t \, dW_t^\alpha \\
\langle dW_t^F, dW_t^\alpha \rangle &= \rho \, dt
\end{align}
$$

**Parameters**:

| Parameter | Symbol | Range | Description |
|-----------|--------|-------|-------------|
| Initial volatility | $\alpha_0$ | $> 0$ | ATM volatility level |
| CEV exponent | $\beta$ | $[0, 1]$ | 0=normal, 0.5=shifted lognormal, 1=lognormal |
| Vol-of-vol | $\nu$ | $> 0$ | Volatility of volatility |
| Correlation | $\rho$ | $[-1, 1]$ | Forward-vol correlation |

**Note**: $\beta$ is typically **fixed** (0.5 is common for rates), leaving 3 parameters to calibrate.

#### Hagan's Implied Volatility Approximation

**At-the-money** ($F = K$):

$$
\sigma_{\text{ATM}} \approx \frac{\alpha}{F^{1-\beta}} \left[1 + \left(\frac{(1-\beta)^2}{24} \frac{\alpha^2}{F^{2(1-\beta)}} + \frac{\rho\beta\nu\alpha}{4F^{1-\beta}} + \frac{2-3\rho^2}{24}\nu^2\right)T\right]
$$

**Away from ATM**: Full asymptotic expansion involves $z = \frac{\nu}{\alpha} (FK)^{(1-\beta)/2} \ln(F/K)$

**Application**: Direct mapping from parameters to implied volatility (no numerical pricing required)

#### Calibration

**Target**: Implied volatility smile at single maturity (or term structure)

**Loss Function**:

$$
\mathcal{L}(\alpha, \rho, \nu) = \sum_i \left(\sigma_i^{\text{market}} - \sigma_i^{\text{SABR}}(\alpha, \beta, \rho, \nu)\right)^2
$$

**Typical Parameter Ranges** (interest rates):
- $\alpha \in [0.001, 0.1]$ (1%-10% ATM vol)
- $\beta = 0.5$ (fixed, common choice)
- $\rho \in [-0.5, 0.5]$ (often negative for rates)
- $\nu \in [0.1, 1]$ (vol-of-vol)

#### Implementation

**File**: `src/neutryx/models/sabr.py`

**Key Functions**:
- `implied_volatility(F, K, T, alpha, beta, rho, nu)`: Hagan's approximation
- `calibrate(market_strikes, market_vols, F, T, beta_fixed)`: Calibration to smile

**Calibration File**: `src/neutryx/calibration/sabr.py`

**Validation**:
- Hagan's formula implementation tested against reference (QuantLib, original paper)
- Asymptotic behavior verified: $\sigma \to \alpha/F^{1-\beta}$ as $K \to F$
- Calibration tested on swaption market data

---

### Rough Bergomi

#### Overview

The Rough Bergomi model, introduced by Bayer, Friz, and Gatheral (2016), uses fractional Brownian motion to capture "rough" volatility dynamics with Hurst exponent $H \approx 0.1$.

**Key References**:
- [Bayer et al., 2016] "Pricing under rough volatility"
- [Gatheral et al., 2018] "Volatility is Rough"
- See [Theory: Rough Bergomi Model](../../theory/pricing_models.md#rough-bergomi-model)

#### Model Specification

$$
\begin{align}
dS_t &= \sqrt{v_t} S_t \, dW_t^S \\
v_t &= \xi_0(t) \exp\left(\eta W_t^H - \frac{\eta^2}{2} t^{2H}\right)
\end{align}
$$

**Parameters**:

| Parameter | Symbol | Range | Description |
|-----------|--------|-------|-------------|
| Forward variance curve | $\xi_0(t)$ | $> 0$ | Deterministic term structure |
| Vol-of-vol | $\eta$ | $> 0$ | Volatility of volatility |
| Hurst parameter | $H$ | $(0, 1)$ | Roughness (typically $H \approx 0.1$) |
| Correlation | $\rho$ | $[-1, 1]$ | Price-vol correlation |

**Fractional Brownian Motion**: $W_t^H$ is not a semimartingale for $H \neq 0.5$ (no standard Itô calculus).

#### Empirical Evidence

**Stylized Fact** [Gatheral et al., 2018]: Realized volatility exhibits roughness with $H \approx 0.1$ (anti-persistent, rougher than Brownian motion).

**Advantages over Heston**:
- Better fit to short-dated skew (steeper)
- Flatter vol-of-vol term structure
- Fewer parameters

#### Simulation

**Method**: Cholesky decomposition of fBm covariance matrix

1. Discretize time: $t_0, t_1, \ldots, t_N$
2. Compute covariance: $\Sigma_{ij} = \frac{1}{2}(t_i^{2H} + t_j^{2H} - |t_i - t_j|^{2H})$
3. Cholesky: $\Sigma = L L^T$
4. Sample: $\mathbf{W}^H = L \mathbf{Z}$, where $\mathbf{Z} \sim \mathcal{N}(0, I)$

**Complexity**: $O(N^3)$ for Cholesky (can be reduced to $O(N \log N)$ with hybrid schemes)

#### Implementation

**File**: `src/neutryx/models/rough_vol.py`

**Key Functions**:
- `simulate_fractional_brownian_motion(T, n_steps, hurst, n_paths)`: fBm generation
- `simulate_rough_bergomi(S0, xi0, eta, hurst, rho, T, n_steps, n_paths)`: Full path simulation
- `price_european(S0, K, T, r, xi0, eta, hurst, rho, n_paths)`: MC pricing

**Validation**:
- fBm covariance structure verified
- Reduces to standard Heston when $H = 0.5$
- Smile behavior matches [Bayer et al., 2016]

---

### Local Volatility (Dupire)

#### Overview

Dupire (1994) showed that any arbitrage-free implied volatility surface can be realized by a deterministic local volatility function $\sigma_{\text{loc}}(S, t)$.

**Key References**:
- [Dupire, 1994] "Pricing with a Smile"
- [Gatheral, 2006] *The Volatility Surface*, Chapter 5
- See [Theory: Local Volatility Models](../../theory/pricing_models.md#local-volatility-models)

#### Model Specification

$$
dS_t = r S_t \, dt + \sigma_{\text{loc}}(S_t, t) S_t \, dW_t
$$

**Key Property**: Local volatility is a **function** of current stock price and time (deterministic, not stochastic).

#### Dupire's Formula

Given market option prices $C(K, T)$ for all strikes and maturities:

$$
\sigma_{\text{loc}}^2(K, T) = \frac{\frac{\partial C}{\partial T} + rK \frac{\partial C}{\partial K}}{\frac{1}{2} K^2 \frac{\partial^2 C}{\partial K^2}}
$$

**Derivation**: Forward Kolmogorov equation for the probability density.

**In terms of implied volatility**: More complex formula involving $\frac{\partial \sigma_{\text{imp}}}{\partial K}$, $\frac{\partial \sigma_{\text{imp}}}{\partial T}$, etc.

#### Calibration Procedure

1. **Interpolate** implied volatility surface $\sigma_{\text{imp}}(K, T)$ (e.g., bicubic spline, SVI parameterization)
2. **Compute derivatives**: Numerical differentiation (with smoothing/regularization)
3. **Apply Dupire's formula** to obtain $\sigma_{\text{loc}}(K, T)$
4. **Enforce arbitrage-free conditions**:
   - $\frac{\partial C}{\partial T} \geq 0$ (calendar spread)
   - $\frac{\partial^2 C}{\partial K^2} \geq 0$ (butterfly, non-negative density)

**Challenges**:
- Noisy derivatives (market data is discrete and noisy)
- Extrapolation in illiquid regions
- No forward smile (volatility collapses to flat for future dates)

#### Implementation

**File**: `src/neutryx/models/local_vol.py`, `src/neutryx/models/dupire.py`

**Key Functions**:
- `dupire_local_vol(K, T, implied_vol_surface, S0, r)`: Dupire formula application
- `simulate_local_vol(S0, T, n_steps, local_vol_func)`: Monte Carlo with local vol
- `price_european_pde(S0, K, T, r, local_vol_surface)`: PDE pricing

**Calibration File**: `src/neutryx/calibration/local_vol.py`

**Validation**:
- Perfect fit to market by construction (within interpolation accuracy)
- Arbitrage-free constraints checked
- Comparison with stochastic volatility models

---

### Jump-Diffusion Models

#### Merton Jump-Diffusion

**Overview**: Merton (1976) added a compound Poisson jump process to GBM.

**Model**:

$$
dS_t = (\mu - \lambda \bar{J}) S_t \, dt + \sigma S_t \, dW_t + S_{t-} \, dJ_t
$$

where $J_t = \sum_{i=1}^{N_t} (Y_i - 1)$, $N_t \sim \text{Poisson}(\lambda)$, $Y_i \sim \text{LogNormal}(\mu_J, \sigma_J^2)$.

**Parameters**:
- $\sigma$: Diffusion volatility
- $\lambda$: Jump intensity (jumps per year)
- $\mu_J$: Mean of log-jump size
- $\sigma_J$: Volatility of log-jump size

**Pricing**: Analytical series expansion (weighted sum of Black-Scholes prices) or FFT (Carr-Madan).

**Key References**:
- [Merton, 1976] "Option Pricing when Underlying Stock Returns are Discontinuous"
- [Cont & Tankov, 2004] *Financial Modelling with Jump Processes*, Chapter 9
- See [Theory: Merton Jump-Diffusion](../../theory/pricing_models.md#merton-jump-diffusion)

**Implementation**: `src/neutryx/models/jump_diffusion.py`

---

#### Kou Double Exponential

**Overview**: Kou (2002) used asymmetric exponential jump distributions.

**Jump Distribution**:

$$
p_J(y) = p \cdot \eta_u e^{-\eta_u y} \mathbf{1}_{y > 0} + (1-p) \cdot \eta_d e^{\eta_d y} \mathbf{1}_{y < 0}
$$

**Parameters**:
- $p$: Probability of upward jump
- $\eta_u > 1$: Decay rate of upward jumps
- $\eta_d > 0$: Decay rate of downward jumps

**Advantage**: Captures asymmetry (negative jumps larger), analytically tractable for barriers/lookbacks.

**Key References**:
- [Kou, 2002] "A Jump-Diffusion Model for Option Pricing"
- See [Theory: Kou Double Exponential](../../theory/pricing_models.md#kou-double-exponential)

**Implementation**: `src/neutryx/models/kou.py`

---

#### Variance Gamma

**Overview**: Variance Gamma (Madan et al., 1998) is a pure jump Lévy process (infinite activity, no Brownian component).

**Model**: $S_t = S_0 \exp((r + \omega)t + X_t)$ where $X_t = \theta G_t + \sigma W_{G_t}$, $G_t \sim \text{Gamma}(\nu t, \nu)$.

**Parameters**:
- $\theta$: Drift of Brownian motion
- $\sigma$: Volatility of Brownian motion
- $\nu$: Variance rate of gamma time change

**Characteristic Function**:

$$
\phi_T(u) = \left(1 - i u \theta \nu + \frac{\sigma^2 \nu u^2}{2}\right)^{-T/\nu}
$$

**Advantages**: Fits smile with fewer parameters than stochastic vol, captures excess kurtosis (fat tails).

**Key References**:
- [Madan et al., 1998] "The Variance Gamma Process and Option Pricing"
- [Cont & Tankov, 2004], Section 8.4
- See [Theory: Variance Gamma](../../theory/pricing_models.md#variance-gamma)

**Implementation**: `src/neutryx/models/variance_gamma.py`

---

## Interest Rate Models

### Vasicek

#### Overview

Vasicek (1977) introduced the first mean-reverting short rate model.

**Key References**:
- [Vasicek, 1977] "An Equilibrium Characterization of the Term Structure"
- [Brigo & Mercurio, 2006] *Interest Rate Models*, Section 3.2
- See [Theory: Vasicek Model](../../theory/pricing_models.md#vasicek-model)

#### Model Specification

$$
dr_t = \kappa(\theta - r_t) \, dt + \sigma \, dW_t
$$

**Parameters**:

| Parameter | Symbol | Range | Description |
|-----------|--------|-------|-------------|
| Mean-reversion speed | $\kappa$ | $> 0$ | Speed of reversion (half-life = $\ln(2)/\kappa$) |
| Long-term mean | $\theta$ | $\mathbb{R}$ | Equilibrium rate level |
| Volatility | $\sigma$ | $> 0$ | Interest rate volatility |

**Properties**:
- Ornstein-Uhlenbeck process (Gaussian)
- Mean-reverting: $\mathbb{E}[r_t | r_0] = \theta + (r_0 - \theta) e^{-\kappa t}$
- **Limitation**: $r_t$ can be negative (though $\mathbb{P}(r_t < 0)$ can be small if parameters chosen carefully)

#### Zero-Coupon Bond Pricing

**Analytical formula**:

$$
P(t, T) = A(t, T) e^{-B(t, T) r_t}
$$

where:

$$
B(t, T) = \frac{1 - e^{-\kappa(T-t)}}{\kappa}
$$

$$
A(t, T) = \exp\left(\left(\theta - \frac{\sigma^2}{2\kappa^2}\right)(B(t, T) - (T-t)) - \frac{\sigma^2 B(t, T)^2}{4\kappa}\right)
$$

**Yield curve**: $y(t, T) = -\frac{\ln P(t, T)}{T - t}$

#### Implementation

**File**: `src/neutryx/models/vasicek.py`

**Key Functions**:
- `bond_price(r, t, T, kappa, theta, sigma)`: Zero-coupon bond pricing
- `yield_curve(r, t, maturities, kappa, theta, sigma)`: Yield curve construction
- `simulate(r0, T, n_steps, kappa, theta, sigma, scheme='exact')`: Path simulation (exact or Euler)
- `bond_option_price(...)`: European option on zero-coupon bond (Jamshidian formula)

**Validation**:
- Bond pricing formula verified against [Brigo & Mercurio, 2006]
- Exact simulation matches analytical distribution (Gaussian)
- Convergence to long-term mean verified

---

### Cox-Ingersoll-Ross (CIR)

#### Overview

CIR (1985) ensured non-negative interest rates via square-root diffusion.

**Key References**:
- [Cox et al., 1985] "A Theory of the Term Structure of Interest Rates"
- [Brigo & Mercurio, 2006] *Interest Rate Models*, Section 3.1
- See [Theory: CIR Model](../../theory/pricing_models.md#cox-ingersoll-ross-cir)

#### Model Specification

$$
dr_t = \kappa(\theta - r_t) \, dt + \sigma \sqrt{r_t} \, dW_t
$$

**Parameters**: Same as Vasicek, but volatility now $\propto \sqrt{r_t}$.

**Feller Condition**: To ensure $r_t > 0$ always:

$$
2\kappa\theta > \sigma^2
$$

If satisfied, $r_t$ never reaches zero. If violated, $r_t$ can reach zero (absorbing boundary).

#### Zero-Coupon Bond Pricing

**Analytical formula**:

$$
P(t, T) = A(t, T) e^{-B(t, T) r_t}
$$

where:

$$
B(t, T) = \frac{2(e^{\gamma(T-t)} - 1)}{(\gamma + \kappa)(e^{\gamma(T-t)} - 1) + 2\gamma}
$$

$$
A(t, T) = \left(\frac{2\gamma e^{(\kappa+\gamma)(T-t)/2}}{(\gamma + \kappa)(e^{\gamma(T-t)} - 1) + 2\gamma}\right)^{2\kappa\theta/\sigma^2}
$$

where $\gamma = \sqrt{\kappa^2 + 2\sigma^2}$.

#### Simulation

**Exact**: $r_t | r_0$ follows a scaled non-central chi-squared distribution.

**Euler**: Simple but may produce negative rates (use $\max(r_t, 0)$ or absorbing).

**Milstein**: Higher-order accuracy.

#### Implementation

**File**: `src/neutryx/models/cir.py`

**Key Functions**:
- `bond_price(r, t, T, kappa, theta, sigma)`: Zero-coupon bond pricing
- `simulate(r0, T, n_steps, kappa, theta, sigma, scheme='exact')`: Exact or Euler simulation
- `check_feller_condition(kappa, theta, sigma)`: Feller condition verification

**Validation**:
- Bond pricing verified against [Brigo & Mercurio, 2006]
- Exact simulation uses non-central chi-squared (scipy)
- Non-negativity enforced and tested

---

### Hull-White

#### Overview

Hull-White (1990) extended Vasicek to fit the initial term structure $P^{\text{market}}(0, T)$.

**Key References**:
- [Hull & White, 1990] "Pricing Interest-Rate-Derivative Securities"
- [Brigo & Mercurio, 2006] *Interest Rate Models*, Section 3.4
- See [Theory: Hull-White Model](../../theory/pricing_models.md#hull-white-model)

#### Model Specification

$$
dr_t = (\theta(t) - \kappa r_t) \, dt + \sigma \, dW_t
$$

where $\theta(t)$ is time-dependent, chosen to fit the initial yield curve.

**Calibration Formula**:

$$
\theta(t) = \frac{\partial f^{\text{market}}(0, t)}{\partial t} + \kappa f^{\text{market}}(0, t) + \frac{\sigma^2}{2\kappa} (1 - e^{-2\kappa t})
$$

where $f(0, t)$ is the instantaneous forward rate at time 0 for maturity $t$.

#### Bond Pricing

**Analytical formula** (same affine structure as Vasicek):

$$
P(t, T) = A(t, T) e^{-B(t, T) r_t}
$$

where $B(t, T) = \frac{1 - e^{-\kappa(T-t)}}{\kappa}$ and:

$$
\ln A(t, T) = \ln\frac{P^{\text{market}}(0, T)}{P^{\text{market}}(0, t)} - B(t, T) f^{\text{market}}(0, t) - \frac{\sigma^2}{4\kappa} B(t, T)^2 (1 - e^{-2\kappa t})
$$

#### Caplet and Cap Pricing

**Caplet** (closed-form):

$$
\text{Caplet}(t, T, K) = P(t, T) \left[(L_t + \text{spread}) \Phi(d_1) - K \Phi(d_2)\right]
$$

(Black-like formula with adjusted parameters)

**Application**: Calibrate $\kappa, \sigma$ to cap/floor market.

#### Implementation

**File**: `src/neutryx/models/hull_white.py`

**Key Functions**:
- `calibrate_theta(forward_curve, kappa, sigma)`: Fit $\theta(t)$ to market
- `bond_price(r, t, T, kappa, sigma, theta_func, forward_curve)`: Bond pricing
- `simulate(r0, T, n_steps, kappa, sigma, theta_func)`: Path simulation
- `caplet_price(...)`: Caplet pricing (closed-form)

**Validation**:
- Perfect fit to initial term structure by construction
- Caplet formula verified against [Brigo & Mercurio, 2006]
- Calibration tested on yield curve data

---

## References

**For complete bibliography and detailed derivations, see**:

- [Mathematical Foundations](../../theory/mathematical_foundations.md)
- [Pricing Models Theory](../../theory/pricing_models.md)
- [Numerical Methods Theory](../../theory/numerical_methods.md)
- [Calibration Theory](../../theory/calibration_theory.md)
- [Bibliography and References](../../references.md)

**Implementation files**: `src/neutryx/models/` and `src/neutryx/calibration/`
