# Pricing Models: Theoretical Background

This document provides rigorous mathematical derivations and theoretical foundations for all pricing models implemented in Neutryx.

## Table of Contents

1. [Black-Scholes-Merton Model](#black-scholes-merton-model)
2. [Stochastic Volatility Models](#stochastic-volatility-models)
   - [Heston Model](#heston-model)
   - [SABR Model](#sabr-model)
   - [Rough Bergomi Model](#rough-bergomi-model)
3. [Jump-Diffusion Models](#jump-diffusion-models)
   - [Merton Jump-Diffusion](#merton-jump-diffusion)
   - [Kou Double Exponential](#kou-double-exponential)
   - [Variance Gamma](#variance-gamma)
4. [Local Volatility Models](#local-volatility-models)
5. [Interest Rate Models](#interest-rate-models)
   - [Vasicek Model](#vasicek-model)
   - [Cox-Ingersoll-Ross (CIR)](#cox-ingersoll-ross-cir)
   - [Hull-White Model](#hull-white-model)

---

## Black-Scholes-Merton Model

### Historical Context

The **Black-Scholes-Merton** model [Black & Scholes, 1973; Merton, 1973] revolutionized finance by providing the first closed-form solution for European option pricing. It earned Scholes and Merton the 1997 Nobel Prize in Economics.

**Reference**: [Black & Scholes, 1973]; [Merton, 1973]; [Hull, 2022], Chapter 15

### Model Specification

Under the risk-neutral measure $\mathbb{Q}$, the stock price $S_t$ follows geometric Brownian motion:

$$
dS_t = r S_t \, dt + \sigma S_t \, dW_t^{\mathbb{Q}}
$$

where:
- $r$: risk-free interest rate (constant)
- $\sigma$: volatility (constant)
- $W_t^{\mathbb{Q}}$: Brownian motion under $\mathbb{Q}$

**Solution**: By Itô's lemma (see [Mathematical Foundations](mathematical_foundations.md)):

$$
S_t = S_0 \exp\left(\left(r - \frac{\sigma^2}{2}\right)t + \sigma W_t^{\mathbb{Q}}\right)
$$

### Black-Scholes PDE Derivation

Consider a European option with payoff $g(S_T)$ and value $V(S, t)$. Construct a **self-financing hedged portfolio**:

$$
\Pi_t = V(S_t, t) - \Delta_t S_t
$$

where $\Delta_t = \frac{\partial V}{\partial S}$ (delta hedge).

**Step 1**: Apply Itô's lemma to $V(S_t, t)$:

$$
dV = \left(\frac{\partial V}{\partial t} + r S \frac{\partial V}{\partial S} + \frac{\sigma^2 S^2}{2} \frac{\partial^2 V}{\partial S^2}\right) dt + \sigma S \frac{\partial V}{\partial S} \, dW_t
$$

**Step 2**: Compute portfolio change:

$$
d\Pi_t = dV - \Delta_t \, dS_t = \left(\frac{\partial V}{\partial t} + \frac{\sigma^2 S^2}{2} \frac{\partial^2 V}{\partial S^2}\right) dt
$$

(stochastic terms cancel by construction)

**Step 3**: No-arbitrage condition: $d\Pi_t = r \Pi_t \, dt$

$$
\frac{\partial V}{\partial t} + \frac{\sigma^2 S^2}{2} \frac{\partial^2 V}{\partial S^2} = r(V - S\Delta_t) = r\left(V - S \frac{\partial V}{\partial S}\right)
$$

**Black-Scholes PDE**:

$$
\boxed{\frac{\partial V}{\partial t} + r S \frac{\partial V}{\partial S} + \frac{\sigma^2 S^2}{2} \frac{\partial^2 V}{\partial S^2} - r V = 0}
$$

with terminal condition $V(S, T) = g(S)$.

**Reference**: [Hull, 2022], Chapter 15; [Wilmott et al., 1995]

### Black-Scholes Formula

For a **European call option** with strike $K$ and maturity $T$:

$$
\boxed{C(S_0, K, T) = S_0 \Phi(d_1) - K e^{-rT} \Phi(d_2)}
$$

where:

$$
d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}
$$

and $\Phi(\cdot)$ is the standard normal cumulative distribution function.

**Put-Call Parity**:

$$
C - P = S_0 - K e^{-rT}
$$

**Derivation**: Using risk-neutral pricing:

$$
C = e^{-rT} \mathbb{E}^{\mathbb{Q}}[\max(S_T - K, 0)]
$$

Substituting the lognormal distribution of $S_T$ and completing the square yields the formula.

**Reference**: [Black & Scholes, 1973]; [Hull, 2022], Chapter 15

### Greeks

**Delta** ($\Delta$): First derivative with respect to underlying

$$
\Delta_{\text{call}} = \frac{\partial C}{\partial S} = \Phi(d_1), \quad \Delta_{\text{put}} = -\Phi(-d_1)
$$

**Gamma** ($\Gamma$): Second derivative (curvature)

$$
\Gamma = \frac{\partial^2 V}{\partial S^2} = \frac{\phi(d_1)}{S_0 \sigma \sqrt{T}}
$$

where $\phi(\cdot)$ is the standard normal PDF.

**Vega** ($\mathcal{V}$): Sensitivity to volatility

$$
\mathcal{V} = \frac{\partial V}{\partial \sigma} = S_0 \phi(d_1) \sqrt{T}
$$

**Theta** ($\Theta$): Time decay

$$
\Theta_{\text{call}} = -\frac{S_0 \phi(d_1) \sigma}{2\sqrt{T}} - r K e^{-rT} \Phi(d_2)
$$

**Rho** ($\rho$): Interest rate sensitivity

$$
\rho_{\text{call}} = K T e^{-rT} \Phi(d_2)
$$

**Reference**: [Hull, 2022], Chapter 19; [Haug, 2007]

**Implementation**: `src/neutryx/models/bs.py` (analytical formulas), `src/neutryx/valuations/greeks/greeks.py` (JAX autodiff)

### Model Limitations

1. **Constant volatility**: Violated by volatility smile/skew
2. **Log-normal distribution**: Underestimates tail risk (fat tails in reality)
3. **Continuous trading**: Transaction costs ignored
4. **No jumps**: Cannot capture gap risk

**Empirical violations**: Implied volatility varies with strike and maturity (volatility surface).

**Extensions**: Stochastic volatility (Heston, SABR), jump models (Merton), local volatility (Dupire)

---

## Stochastic Volatility Models

### Motivation

Market-observed **implied volatility surfaces** exhibit:
- **Smile/Skew**: Implied vol varies with strike
- **Term structure**: Implied vol varies with maturity

Black-Scholes cannot capture these. **Stochastic volatility models** allow volatility to evolve randomly.

### Heston Model

#### Model Specification

**Heston (1993)** introduced the most popular stochastic volatility model with semi-analytical solutions.

$$
\begin{align}
dS_t &= r S_t \, dt + \sqrt{v_t} S_t \, dW_t^S \\
dv_t &= \kappa(\theta - v_t) \, dt + \sigma_v \sqrt{v_t} \, dW_t^v \\
\langle dW_t^S, dW_t^v \rangle &= \rho \, dt
\end{align}
$$

**Parameters**:
- $v_t$: instantaneous variance (stochastic)
- $\kappa > 0$: mean-reversion speed
- $\theta > 0$: long-term variance level
- $\sigma_v > 0$: volatility of volatility (vol-of-vol)
- $\rho \in [-1, 1]$: correlation between price and volatility

**Feller Condition**: To ensure $v_t > 0$, require:

$$
2\kappa\theta > \sigma_v^2
$$

If violated, $v_t$ can reach zero (boundary attainable).

**Reference**: [Heston, 1993]; [Gatheral, 2006], Chapter 3

**Implementation**: `src/neutryx/models/heston.py`

#### Characteristic Function

The key to Heston's tractability is the **characteristic function** of $\log S_T$:

$$
\phi_T(u) = \mathbb{E}^{\mathbb{Q}}[e^{i u \log S_T} | \mathcal{F}_t]
$$

**Heston characteristic function**:

$$
\phi_T(u) = \exp\left(C(T-t, u) + D(T-t, u) v_t + i u \log S_t\right)
$$

where $C(\tau, u)$ and $D(\tau, u)$ satisfy complex-valued Riccati ODEs:

$$
\begin{align}
\frac{dD}{d\tau} &= -\frac{\sigma_v^2}{2} D^2 + (i\rho\sigma_v u - \kappa) D + \frac{u^2 + iu}{2} \\
\frac{dC}{d\tau} &= \kappa\theta D + r i u
\end{align}
$$

with $D(0, u) = C(0, u) = 0$.

**Analytical solution** (using matrix exponentials):

$$
D(\tau, u) = \frac{(i\rho\sigma_v u - \kappa) - d}{\sigma_v^2} \cdot \frac{1 - e^{-d\tau}}{1 - g e^{-d\tau}}
$$

where:

$$
d = \sqrt{(i\rho\sigma_v u - \kappa)^2 + \sigma_v^2(u^2 + iu)}, \quad g = \frac{(i\rho\sigma_v u - \kappa) - d}{(i\rho\sigma_v u - \kappa) + d}
$$

**Reference**: [Heston, 1993], Appendix A; [Gatheral, 2006], Section 3.2

#### Option Pricing via FFT

Given the characteristic function, option prices can be computed using **Carr-Madan FFT** [Carr & Madan, 1999]:

$$
C(K) = \frac{e^{-\alpha k}}{\pi} \int_0^\infty e^{-iuk} \psi_T(u) \, du
$$

where $k = \log(K/S_0)$, $\alpha > 0$ is a damping parameter, and:

$$
\psi_T(u) = \frac{e^{-rT} \phi_T(u - (\alpha+1)i)}{(\alpha + iu)(\alpha + 1 + iu)}
$$

The FFT computes option prices across many strikes simultaneously in $O(N \log N)$ time.

**Reference**: [Carr & Madan, 1999]; [Gatheral, 2006], Chapter 2

**Implementation**: `src/neutryx/models/fft_pricing.py`, `src/neutryx/engines/fourier.py`

#### Simulation Schemes

**Euler Scheme**: Simple but can produce negative variance.

$$
v_{t+\Delta t} = v_t + \kappa(\theta - v_t) \Delta t + \sigma_v \sqrt{v_t} \sqrt{\Delta t} Z^v
$$

**Milstein Scheme**: Higher-order accuracy.

$$
v_{t+\Delta t} = v_t + \kappa(\theta - v_t) \Delta t + \sigma_v \sqrt{v_t} \sqrt{\Delta t} Z^v + \frac{\sigma_v^2}{4} \Delta t ((Z^v)^2 - 1)
$$

**Quadratic-Exponential (QE) Scheme** [Andersen, 2008]: Ensures non-negativity and preserves moments.

Split based on $\Psi_t = \frac{\sigma_v^2 e^{-\kappa\Delta t}}{4\kappa} \left(1 - e^{-\kappa\Delta t}\right)$:

- If $\Psi_t \leq \Psi_c$ (critical threshold): quadratic form
- If $\Psi_t > \Psi_c$: exponential form (handles tail)

**Reference**: [Andersen, 2008]; [Glasserman, 2004], Section 3.4

**Implementation**: `src/neutryx/models/heston.py:simulate()` (supports all three schemes)

#### Volatility Smile

Heston generates volatility smile through:

1. **Vol-of-vol** ($\sigma_v$): Increases curvature (smile)
2. **Correlation** ($\rho < 0$): Creates skew (OTM puts more expensive)
3. **Mean-reversion** ($\kappa$): Controls term structure

**Stylized fact**: Equity options have $\rho \approx -0.7$ (leverage effect).

**Reference**: [Gatheral, 2006], Chapter 3

### SABR Model

#### Model Specification

The **SABR** (Stochastic Alpha Beta Rho) model [Hagan et al., 2002] is an industry-standard for modeling the volatility smile, particularly in interest rate markets.

$$
\begin{align}
dF_t &= \alpha_t F_t^\beta \, dW_t^F \\
d\alpha_t &= \nu \alpha_t \, dW_t^\alpha \\
\langle dW_t^F, dW_t^\alpha \rangle &= \rho \, dt
\end{align}
$$

**Parameters**:
- $F_t$: forward price (or forward rate)
- $\alpha_t$: stochastic volatility
- $\beta \in [0, 1]$: CEV exponent (0 = normal, 1 = lognormal, 0.5 = shifted lognormal)
- $\nu > 0$: vol-of-vol
- $\rho \in [-1, 1]$: correlation

**No drift**: Model is directly for forward prices (already under forward measure).

**Reference**: [Hagan et al., 2002]; [Gatheral, 2006], Chapter 4

**Implementation**: `src/neutryx/models/sabr.py`

#### Hagan's Approximation

Hagan derived an **asymptotic expansion** for implied volatility as a function of strike:

$$
\sigma_{\text{imp}}(K, T) \approx \frac{\alpha}{(FK)^{(1-\beta)/2} \left[1 + \frac{(1-\beta)^2}{24} \ln^2(F/K) + \cdots\right]} \times \left[\frac{z}{\chi(z)}\right] \times \left[1 + \left(\frac{(1-\beta)^2}{24} \frac{\alpha^2}{(FK)^{1-\beta}} + \frac{\rho\beta\nu\alpha}{4(FK)^{(1-\beta)/2}} + \frac{2-3\rho^2}{24}\nu^2\right)T + \cdots\right]
$$

where:

$$
z = \frac{\nu}{\alpha} (FK)^{(1-\beta)/2} \ln(F/K), \quad \chi(z) = \ln\left(\frac{\sqrt{1 - 2\rho z + z^2} + z - \rho}{1 - \rho}\right)
$$

**At-the-money** ($F = K$):

$$
\sigma_{\text{ATM}} \approx \frac{\alpha}{F^{1-\beta}} \left[1 + \left(\frac{(1-\beta)^2}{24} \frac{\alpha^2}{F^{2(1-\beta)}} + \frac{\rho\beta\nu\alpha}{4F^{1-\beta}} + \frac{2-3\rho^2}{24}\nu^2\right)T\right]
$$

**Calibration**: Fit $\{\alpha, \beta, \rho, \nu\}$ to market implied volatilities (often fix $\beta$).

**Reference**: [Hagan et al., 2002], Section 3; [Gatheral, 2006], Section 4.2

**Implementation**: `src/neutryx/models/sabr.py:implied_volatility()`, `src/neutryx/calibration/sabr.py`

#### Density and Risk Management

SABR also provides the **probability density** via:

$$
p(K) = \frac{\partial^2 C}{\partial K^2}
$$

computed numerically from the implied volatility surface.

**Application**: Risk-neutral density, arbitrage detection (density must be non-negative).

**Reference**: [Hagan et al., 2002], Section 5

### Rough Bergomi Model

#### Rough Volatility

Recent empirical studies [Gatheral et al., 2018] show that realized volatility exhibits **roughness** with Hurst exponent $H \approx 0.1$ (anti-persistent, rougher than Brownian motion).

**Traditional SV models**: Assume $H = 0.5$ (Brownian volatility)

**Rough volatility**: $H < 0.5$ (fractional Brownian motion)

#### Model Specification

The **Rough Bergomi** model [Bayer et al., 2016] uses fractional Brownian motion:

$$
\begin{align}
dS_t &= \sqrt{v_t} S_t \, dW_t^S \\
v_t &= \xi_0(t) \exp\left(\eta W_t^H - \frac{\eta^2}{2} t^{2H}\right)
\end{align}
$$

where:
- $W_t^H$: fractional Brownian motion with Hurst parameter $H \in (0, 1)$
- $\xi_0(t)$: forward variance curve (deterministic)
- $\eta > 0$: vol-of-vol
- $\rho = \langle dW_t^S, dW_t^H \rangle / dt$: correlation (generalized)

**Properties of fBm**:
- $W_t^H \sim \mathcal{N}(0, t^{2H})$
- Covariance: $\mathbb{E}[W_s^H W_t^H] = \frac{1}{2}(s^{2H} + t^{2H} - |t-s|^{2H})$
- **Not a semimartingale** for $H \neq 1/2$ (no Itô calculus)

**Reference**: [Bayer et al., 2016]; [Gatheral et al., 2018]

**Implementation**: `src/neutryx/models/rough_vol.py`

#### Simulation via Cholesky

Since fBm is Gaussian, simulation uses **Cholesky decomposition** of the covariance matrix:

1. Discretize: $W_{t_i}^H$ for $i = 0, \ldots, N$
2. Compute covariance matrix: $\Sigma_{ij} = \mathbb{E}[W_{t_i}^H W_{t_j}^H]$
3. Cholesky: $\Sigma = L L^T$
4. Generate: $\mathbf{W}^H = L \mathbf{Z}$, where $\mathbf{Z} \sim \mathcal{N}(0, I)$

**Complexity**: $O(N^3)$ for Cholesky, $O(N^2)$ for sampling

**Alternative**: Hybrid scheme [McCrickerd & Pakkanen, 2018] reduces to $O(N \log N)$ using FFT.

**Reference**: [Bayer et al., 2016], Section 3; [Gatheral et al., 2018]

#### Empirical Fit

Rough Bergomi with $H \approx 0.1$ fits:
- **Short-dated skew**: Steeper than traditional SV models
- **Term structure**: Flatter vol-of-vol structure

**Advantage**: Fewer parameters, better empirical fit

**Disadvantage**: Computationally intensive, no characteristic function

**Reference**: [Gatheral et al., 2018]; [Fukasawa, 2011]

---

## Jump-Diffusion Models

### Motivation

**Empirical evidence** for jumps:
- Discontinuous price movements (earnings announcements, geopolitical events)
- Implied volatility smile not fully captured by diffusion models
- Fat tails in return distributions (excess kurtosis)

### Merton Jump-Diffusion

#### Model Specification

**Merton (1976)** added a **compound Poisson jump process** to GBM:

$$
dS_t = (\mu - \lambda \bar{J}) S_t \, dt + \sigma S_t \, dW_t + S_{t-} \, dJ_t
$$

where:
- $J_t = \sum_{i=1}^{N_t} (Y_i - 1)$: compound Poisson process
- $N_t$: Poisson process with intensity $\lambda$
- $Y_i \sim \text{LogNormal}(\mu_J, \sigma_J^2)$: jump sizes (independent, i.i.d.)
- $\bar{J} = \mathbb{E}[Y - 1] = e^{\mu_J + \sigma_J^2/2} - 1$: mean jump size

**Interpretation**: Between jumps, $S_t$ follows GBM; at jump times, $S$ multiplies by $Y_i$.

**Reference**: [Merton, 1976]; [Cont & Tankov, 2004], Chapter 9

**Implementation**: `src/neutryx/models/jump_diffusion.py`

#### Characteristic Function

Under risk-neutral measure $\mathbb{Q}$:

$$
\phi_T(u) = \mathbb{E}^{\mathbb{Q}}[e^{i u \log S_T}] = \exp\left(i u \log S_0 + i u (r - \lambda \bar{J})T - \frac{\sigma^2 u^2}{2} T + \lambda T (\mathbb{E}[Y^{iu}] - 1)\right)
$$

where:

$$
\mathbb{E}[Y^{iu}] = \exp\left(i u \mu_J - \frac{\sigma_J^2 u^2}{2}\right)
$$

**Application**: FFT pricing via Carr-Madan or COS method.

**Reference**: [Merton, 1976]; [Carr & Madan, 1999]

**Implementation**: `src/neutryx/engines/fourier.py`

#### Analytical Pricing (Series Expansion)

European call price is a **weighted sum** of Black-Scholes prices:

$$
C_{\text{Merton}}(S_0, K, T) = \sum_{n=0}^\infty \frac{e^{-\lambda' T} (\lambda' T)^n}{n!} C_{\text{BS}}(S_0, K, T, \sigma_n, r_n)
$$

where:

$$
\lambda' = \lambda (1 + \bar{J}), \quad \sigma_n^2 = \sigma^2 + \frac{n\sigma_J^2}{T}, \quad r_n = r - \lambda \bar{J} + \frac{n\mu_J}{T}
$$

**Interpretation**: Condition on number of jumps $n$, then use Black-Scholes with adjusted parameters.

**Convergence**: Truncate series when terms become negligible (typically $n < 20$).

**Reference**: [Merton, 1976]; [Haug, 2007], Section 5.6

**Implementation**: `src/neutryx/models/jump_diffusion.py:price_european()`

### Kou Double Exponential

#### Model Specification

**Kou (2002)** introduced **asymmetric exponential jumps**:

$$
dS_t = (\mu - \lambda \bar{J}) S_t \, dt + \sigma S_t \, dW_t + S_{t-} \, dJ_t
$$

where jumps have **double exponential distribution**:

$$
p_J(y) = p \cdot \eta_u e^{-\eta_u y} \mathbf{1}_{y > 0} + (1-p) \cdot \eta_d e^{\eta_d y} \mathbf{1}_{y < 0}
$$

**Parameters**:
- $p \in (0, 1)$: probability of upward jump
- $\eta_u > 1$: decay rate of upward jumps
- $\eta_d > 0$: decay rate of downward jumps

**Motivation**: Captures **asymmetry** (negative jumps are larger, matching empirical data).

**Reference**: [Kou, 2002]; [Cont & Tankov, 2004], Section 9.2

**Implementation**: `src/neutryx/models/kou.py`

#### Characteristic Function

$$
\phi(u) = \mathbb{E}[e^{iuY}] = \frac{p \eta_u}{\eta_u - iu} + \frac{(1-p) \eta_d}{\eta_d + iu}, \quad u \in \mathbb{R}
$$

**Asset price CF**:

$$
\phi_T(u) = \exp\left(i u \log S_0 + i u (r - \lambda \bar{J})T - \frac{\sigma^2 u^2}{2} T + \lambda T (\phi(u) - 1)\right)
$$

**Reference**: [Kou, 2002]

#### Analytical Tractability

Kou model admits **closed-form solutions** for:
- European options (via Wiener-Hopf factorization)
- Barrier options
- Lookback options
- Perpetual American options

**Advantage**: More tractable than Merton for path-dependent options.

**Reference**: [Kou, 2002], Sections 3-5

### Variance Gamma

#### Model Specification

The **Variance Gamma** (VG) model [Madan et al., 1998] is a **pure jump Lévy process** (infinite activity, no Brownian component).

**Time change representation**:

$$
S_t = S_0 \exp\left((r + \omega)t + X_t\right)
$$

where $X_t = \theta G_t + \sigma W_{G_t}$ and:
- $G_t \sim \text{Gamma}(\nu t, \nu)$: gamma time change with rate $\nu$
- $W_t$: Brownian motion (independent of $G_t$)
- $\omega$: drift correction for martingale property

**Parameters**:
- $\theta$: drift of Brownian motion
- $\sigma > 0$: volatility of Brownian motion
- $\nu > 0$: variance rate of gamma process

**Interpretation**: VG is a Brownian motion subordinated by gamma time (random time change).

**Reference**: [Madan et al., 1998]; [Cont & Tankov, 2004], Section 8.4

**Implementation**: `src/neutryx/models/variance_gamma.py`

#### Lévy Density

VG has **infinite activity** (infinite number of jumps in any interval) with Lévy density:

$$
\nu(x) = \frac{1}{|x|} \exp\left(-\frac{|x|}{\nu} \sqrt{\frac{2}{\sigma^2} + \frac{\theta^2}{\sigma^4}}\right), \quad x \neq 0
$$

Small jumps dominate (consistent with high-frequency price changes).

**Reference**: [Cont & Tankov, 2004], Section 8.4

#### Characteristic Function

$$
\phi_T(u) = \left(1 - i u \theta \nu + \frac{\sigma^2 \nu u^2}{2}\right)^{-T/\nu}
$$

**Moments**:
- Mean: $\mathbb{E}[X_t] = \theta t$
- Variance: $\text{Var}(X_t) = (\sigma^2 + \theta^2 \nu) t$
- Skewness: Nonzero (controlled by $\theta$)
- Kurtosis: Excess kurtosis $3\nu / T$ (heavy tails)

**Reference**: [Madan et al., 1998]

#### Option Pricing

VG pricing uses:
- **FFT methods**: Carr-Madan, COS
- **Numerical integration**: Direct integration of CF
- **Monte Carlo**: Simulate $G_t$ then $W_{G_t}$

**Advantage**: Fits smile better than Black-Scholes with fewer parameters than stochastic volatility.

**Reference**: [Madan et al., 1998]; [Carr & Madan, 1999]

---

## Local Volatility Models

### Dupire's Local Volatility

#### Model Specification

**Dupire (1994)** showed that any arbitrage-free implied volatility surface can be realized by a **deterministic local volatility function** $\sigma_{\text{loc}}(S, t)$:

$$
dS_t = r S_t \, dt + \sigma_{\text{loc}}(S_t, t) S_t \, dW_t
$$

**Key insight**: Local volatility is a *function* of current stock price and time, not constant.

**Reference**: [Dupire, 1994]; [Gatheral, 2006], Chapter 5

**Implementation**: `src/neutryx/models/local_vol.py`, `src/neutryx/models/dupire.py`

#### Dupire's Formula

Given the market prices $C(K, T)$ for all strikes $K$ and maturities $T$:

$$
\boxed{\sigma_{\text{loc}}^2(K, T) = \frac{\frac{\partial C}{\partial T} + rK \frac{\partial C}{\partial K}}{\frac{1}{2} K^2 \frac{\partial^2 C}{\partial K^2}}}
$$

**Derivation**: From the **forward Kolmogorov equation** (Fokker-Planck) for the probability density $p(S, t)$:

$$
\frac{\partial p}{\partial t} = -\frac{\partial}{\partial S}[rSp] + \frac{1}{2}\frac{\partial^2}{\partial S^2}[\sigma_{\text{loc}}^2(S, t) S^2 p]
$$

Taking derivatives of the call price $C(K, T) = \int_K^\infty (S - K) p(S, T) dS$ yields Dupire's formula.

**Alternative form** (in terms of implied volatility $\sigma_{\text{imp}}(K, T)$):

$$
\sigma_{\text{loc}}^2(K, T) = \frac{\sigma_{\text{imp}}^2 + 2\sigma_{\text{imp}} T \frac{\partial \sigma_{\text{imp}}}{\partial T} + 2rK T \sigma_{\text{imp}} \frac{\partial \sigma_{\text{imp}}}{\partial K}}{\left(1 + K d_1 \sqrt{T} \frac{\partial \sigma_{\text{imp}}}{\partial K}\right)^2 + K^2 T \sigma_{\text{imp}} \left(\frac{\partial^2 \sigma_{\text{imp}}}{\partial K^2} - d_1 \sqrt{T} \left(\frac{\partial \sigma_{\text{imp}}}{\partial K}\right)^2\right)}
$$

**Reference**: [Dupire, 1994]; [Gatheral, 2006], Section 5.1

#### Calibration

**Steps**:
1. Interpolate market implied volatility surface $\sigma_{\text{imp}}(K, T)$
2. Compute derivatives: $\frac{\partial \sigma_{\text{imp}}}{\partial K}$, $\frac{\partial \sigma_{\text{imp}}}{\partial T}$, $\frac{\partial^2 \sigma_{\text{imp}}}{\partial K^2}$
3. Apply Dupire's formula to obtain $\sigma_{\text{loc}}(K, T)$

**Challenges**:
- Noisy derivatives (need smoothing/regularization)
- Arbitrage-free constraints: $\frac{\partial C}{\partial T} \geq 0$, $\frac{\partial^2 C}{\partial K^2} \geq 0$ (no arbitrage)
- Extrapolation in low-liquidity regions

**Reference**: [Gatheral, 2006], Section 5.2; [Guyon & Henry-Labordère, 2014]

**Implementation**: `src/neutryx/calibration/local_vol.py`

#### Properties

**Advantages**:
- Perfectly fits market prices (by construction)
- Deterministic (no randomness in volatility path)
- Arbitrage-free

**Disadvantages**:
- No forward smile (future implied volatilities collapse to flat)
- Underestimates vega risk
- Requires entire surface (data-intensive)

**Forward smile problem**: Local vol predicts $\sigma_{\text{imp}}^{\text{forward}}(K, t, T) \to$ constant as $t \to T$, contradicting market observations.

**Extension**: Stochastic-Local Volatility (SLV) combines stochastic and local volatility.

**Reference**: [Gatheral, 2006], Section 5.3

---

## Interest Rate Models

### Vasicek Model

#### Model Specification

**Vasicek (1977)** introduced the first **mean-reverting** short rate model:

$$
dr_t = \kappa(\theta - r_t) \, dt + \sigma \, dW_t
$$

**Parameters**:
- $r_t$: instantaneous short rate
- $\kappa > 0$: mean-reversion speed
- $\theta$: long-term mean level
- $\sigma > 0$: volatility

**Properties**:
- **Ornstein-Uhlenbeck process**: Gaussian, mean-reverting
- $\mathbb{E}[r_t | r_0] = \theta + (r_0 - \theta) e^{-\kappa t}$
- $\text{Var}(r_t | r_0) = \frac{\sigma^2}{2\kappa} (1 - e^{-2\kappa t})$
- **Stationary distribution**: $r_\infty \sim \mathcal{N}\left(\theta, \frac{\sigma^2}{2\kappa}\right)$

**Limitation**: $r_t$ can become negative (unrealistic, though $\mathbb{P}(r_t < 0)$ can be small).

**Reference**: [Vasicek, 1977]; [Brigo & Mercurio, 2006], Section 3.2

**Implementation**: `src/neutryx/models/vasicek.py`

#### Zero-Coupon Bond Pricing

A **zero-coupon bond** pays $1$ at maturity $T$:

$$
P(t, T) = \mathbb{E}^{\mathbb{Q}}\left[\exp\left(-\int_t^T r_s \, ds\right) \, \Big| \, \mathcal{F}_t\right]
$$

**Vasicek bond price** (analytical formula):

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

**Derivation**: Solve the PDE via separation of variables (affine term structure).

**Reference**: [Brigo & Mercurio, 2006], Section 3.2; [Björk, 2009], Chapter 21

#### Bond Option Pricing

European options on zero-coupon bonds have **closed-form solutions** (Jamshidian, 1989):

$$
\text{Call}(t, T_{\text{bond}}, T_{\text{option}}, K) = P(t, T_{\text{bond}}) \Phi(d_1) - K P(t, T_{\text{option}}) \Phi(d_2)
$$

where $d_1, d_2$ are analogous to Black-Scholes (with bond volatility $\sigma_P$).

**Reference**: [Brigo & Mercurio, 2006], Section 3.3

### Cox-Ingersoll-Ross (CIR)

#### Model Specification

**Cox, Ingersoll, and Ross (1985)** ensured **non-negative rates** with:

$$
dr_t = \kappa(\theta - r_t) \, dt + \sigma \sqrt{r_t} \, dW_t
$$

**Key difference**: Volatility proportional to $\sqrt{r_t}$ (square-root diffusion).

**Feller Condition**: To ensure $r_t > 0$ always:

$$
2\kappa\theta > \sigma^2
$$

If satisfied, $r_t > 0$ for all $t$ almost surely. If violated, $r_t$ can reach zero (absorbing boundary).

**Reference**: [Cox et al., 1985]; [Brigo & Mercurio, 2006], Section 3.1

**Implementation**: `src/neutryx/models/cir.py`

#### Bond Pricing

CIR also admits **closed-form bond prices**:

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

**Reference**: [Cox et al., 1985]; [Brigo & Mercurio, 2006], Section 3.1

#### Simulation

**Exact simulation**: $r_t | r_0$ follows a **scaled non-central chi-squared distribution**:

$$
r_t = \frac{\sigma^2(1 - e^{-\kappa t})}{4\kappa} \chi'^2\left(\frac{4\kappa e^{-\kappa t}}{\sigma^2(1 - e^{-\kappa t})} r_0\right)
$$

where $\chi'^2(\delta)$ is a non-central chi-squared distribution with $\nu = \frac{4\kappa\theta}{\sigma^2}$ degrees of freedom and non-centrality parameter $\delta$.

**Euler/Milstein**: Simpler but requires care near zero (use $\max(r_t, 0)$ or absorbing).

**Reference**: [Glasserman, 2004], Section 3.3

### Hull-White Model

#### Model Specification

**Hull and White (1990)** extended Vasicek to fit the **initial term structure**:

$$
dr_t = (\theta(t) - \kappa r_t) \, dt + \sigma \, dW_t
$$

where $\theta(t)$ is a **time-dependent** function chosen such that:

$$
P(0, T) = P^{\text{market}}(0, T)
$$

(i.e., the model reproduces the observed yield curve at $t=0$).

**Calibration formula**:

$$
\theta(t) = \frac{\partial f^{\text{market}}(0, t)}{\partial t} + \kappa f^{\text{market}}(0, t) + \frac{\sigma^2}{2\kappa} (1 - e^{-2\kappa t})
$$

where $f(0, t)$ is the instantaneous forward rate.

**Reference**: [Hull & White, 1990]; [Brigo & Mercurio, 2006], Section 3.4

**Implementation**: `src/neutryx/models/hull_white.py`

#### Bond Pricing

Hull-White retains the **affine structure**:

$$
P(t, T) = A(t, T) e^{-B(t, T) r_t}
$$

with $B(t, T) = \frac{1 - e^{-\kappa(T-t)}}{\kappa}$ (same as Vasicek) and:

$$
\ln A(t, T) = \ln\frac{P^{\text{market}}(0, T)}{P^{\text{market}}(0, t)} - B(t, T) f^{\text{market}}(0, t) - \frac{\sigma^2}{4\kappa} B(t, T)^2 (1 - e^{-2\kappa t})
$$

**Reference**: [Brigo & Mercurio, 2006], Section 3.4

#### Caps and Floors

**Caplet formula** (closed-form):

$$
\text{Caplet}(t, T, K) = P(t, T) \left[(L_t + \text{spread}) \Phi(d_1) - K \Phi(d_2)\right]
$$

where $L_t$ is the LIBOR rate and $d_1, d_2$ are Black-like terms with volatility derived from Hull-White parameters.

**Application**: Calibrate to cap/floor market (swaption volatilities).

**Reference**: [Brigo & Mercurio, 2006], Section 3.5

---

## Summary

This document provided rigorous mathematical foundations for:

1. **Black-Scholes-Merton**: Foundational model with closed-form solutions
2. **Stochastic Volatility**: Heston, SABR, Rough Bergomi for smile modeling
3. **Jump-Diffusion**: Merton, Kou, Variance Gamma for tail risk
4. **Local Volatility**: Dupire's formula for perfect calibration
5. **Interest Rate Models**: Vasicek, CIR, Hull-White for rates derivatives

All models are implemented in Neutryx with:
- Analytical formulas where available
- Numerical methods (MC, PDE, FFT) for general cases
- Calibration routines to market data
- Validation against academic benchmarks

**Next**: [Numerical Methods Theory](numerical_methods.md) | [Calibration Theory](calibration_theory.md)

---

**References**: See [Bibliography](../references.md) for complete citations.
