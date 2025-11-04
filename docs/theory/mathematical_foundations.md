# Mathematical Foundations

This document establishes the mathematical framework underlying quantitative finance and the Neutryx implementation.

## Table of Contents

1. [Probability Theory Foundations](#probability-theory-foundations)
2. [Stochastic Processes](#stochastic-processes)
3. [Stochastic Calculus](#stochastic-calculus)
4. [Risk-Neutral Pricing](#risk-neutral-pricing)
5. [Martingale Theory](#martingale-theory)
6. [Change of Measure](#change-of-measure)

---

## Probability Theory Foundations

### Probability Space

All financial models are defined on a **filtered probability space** $(\Omega, \mathcal{F}, \{\mathcal{F}_t\}_{t \geq 0}, \mathbb{P})$, where:

- $\Omega$ is the sample space (set of all possible outcomes)
- $\mathcal{F}$ is a $\sigma$-algebra of events
- $\{\mathcal{F}_t\}_{t \geq 0}$ is a filtration (increasing family of $\sigma$-algebras representing information flow)
- $\mathbb{P}$ is a probability measure

**Reference**: [Shreve, 2004], Chapter 1

### Conditional Expectation

For a random variable $X$ and $\sigma$-algebra $\mathcal{G} \subseteq \mathcal{F}$, the **conditional expectation** $\mathbb{E}[X | \mathcal{G}]$ is characterized by:

1. $\mathbb{E}[X | \mathcal{G}]$ is $\mathcal{G}$-measurable
2. For all $A \in \mathcal{G}$: $\int_A \mathbb{E}[X | \mathcal{G}] \, d\mathbb{P} = \int_A X \, d\mathbb{P}$

**Properties**:
- Tower property: $\mathbb{E}[\mathbb{E}[X | \mathcal{G}]] = \mathbb{E}[X]$
- Taking out what is known: $\mathbb{E}[Y X | \mathcal{G}] = Y \mathbb{E}[X | \mathcal{G}]$ if $Y$ is $\mathcal{G}$-measurable

---

## Stochastic Processes

### Brownian Motion

A **Brownian motion** (or Wiener process) $\{W_t\}_{t \geq 0}$ is a continuous-time stochastic process with:

1. $W_0 = 0$ almost surely
2. **Independent increments**: For $0 \leq s < t$, $W_t - W_s$ is independent of $\mathcal{F}_s$
3. **Stationary increments**: $W_t - W_s \sim \mathcal{N}(0, t-s)$
4. **Continuous paths**: $t \mapsto W_t$ is continuous almost surely

**Properties**:
- $\mathbb{E}[W_t] = 0$, $\text{Var}(W_t) = t$
- $\text{Cov}(W_s, W_t) = \min(s, t)$
- **Quadratic variation**: $[W]_t = t$ (a.s.)
- **Non-differentiable**: Paths are continuous but nowhere differentiable

**Reference**: [Shreve, 2004], Chapter 3; [Björk, 2009], Chapter 4

**Implementation**: All diffusion models in `src/neutryx/models/` use Brownian motion as the fundamental noise source.

### Geometric Brownian Motion

The **geometric Brownian motion** (GBM) is the classical model for stock prices:

$$
dS_t = \mu S_t \, dt + \sigma S_t \, dW_t
$$

**Solution** (via Itô's lemma):

$$
S_t = S_0 \exp\left(\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma W_t\right)
$$

**Properties**:
- $S_t > 0$ always (lognormal distribution)
- $\mathbb{E}[S_t] = S_0 e^{\mu t}$
- $\text{Var}(S_t) = S_0^2 e^{2\mu t}(e^{\sigma^2 t} - 1)$
- **Log-returns** are normally distributed: $\log(S_t/S_0) \sim \mathcal{N}((\mu - \sigma^2/2)t, \sigma^2 t)$

**Reference**: [Shreve, 2004], Chapter 4; [Hull, 2022], Chapter 15

**Implementation**: Black-Scholes model in `src/neutryx/models/bs.py`

### Poisson Process

A **Poisson process** $\{N_t\}_{t \geq 0}$ with intensity $\lambda > 0$ satisfies:

1. $N_0 = 0$
2. Independent increments
3. $N_t - N_s \sim \text{Poisson}(\lambda(t-s))$ for $s < t$

**Properties**:
- $\mathbb{E}[N_t] = \text{Var}(N_t) = \lambda t$
- Inter-arrival times are exponentially distributed: $\text{Exp}(\lambda)$
- **Compensated Poisson process**: $\tilde{N}_t = N_t - \lambda t$ is a martingale

**Application**: Jump models (Merton, Kou, Variance Gamma)

**Reference**: [Cont & Tankov, 2004], Chapter 2

**Implementation**: Jump-diffusion models in `src/neutryx/models/jump_diffusion.py`, `kou.py`, `variance_gamma.py`

### Lévy Processes

A **Lévy process** $\{X_t\}_{t \geq 0}$ is a càdlàg process with:

1. $X_0 = 0$
2. Independent increments
3. Stationary increments
4. Stochastic continuity

**Lévy-Khintchine representation**: The characteristic function has the form:

$$
\mathbb{E}[e^{i u X_t}] = \exp(t \psi(u))
$$

where $\psi(u)$ is the **characteristic exponent**:

$$
\psi(u) = i \gamma u - \frac{\sigma^2}{2} u^2 + \int_{\mathbb{R}} \left(e^{iux} - 1 - iux\mathbf{1}_{|x|<1}\right) \nu(dx)
$$

- $\gamma \in \mathbb{R}$: drift
- $\sigma \geq 0$: Gaussian component
- $\nu$: Lévy measure (jump intensity)

**Examples**:
- Brownian motion: $\nu = 0$
- Poisson process: $\nu = \lambda \delta_1$ (point mass at 1)
- Variance Gamma: $\nu(dx) = \frac{C}{|x|} e^{-M|x|} dx$ (infinite activity)

**Reference**: [Cont & Tankov, 2004], Chapters 3-4

---

## Stochastic Calculus

### Itô Integral

For a progressively measurable process $\{X_t\}$ and Brownian motion $\{W_t\}$, the **Itô integral** is:

$$
\int_0^t X_s \, dW_s
$$

**Properties**:
- **Martingale property**: $\mathbb{E}\left[\int_0^t X_s \, dW_s\right] = 0$
- **Isometry**: $\mathbb{E}\left[\left(\int_0^t X_s \, dW_s\right)^2\right] = \mathbb{E}\left[\int_0^t X_s^2 \, ds\right]$
- **Not path-by-path**: Defined as a limit in $L^2(\mathbb{P})$

**Reference**: [Shreve, 2004], Chapter 4; [Björk, 2009], Chapter 5

### Itô's Lemma

**Itô's Lemma** is the chain rule for stochastic calculus. If $X_t$ satisfies:

$$
dX_t = \mu(X_t, t) \, dt + \sigma(X_t, t) \, dW_t
$$

and $f(x, t) \in C^{2,1}$, then:

$$
df(X_t, t) = \left(\frac{\partial f}{\partial t} + \mu \frac{\partial f}{\partial x} + \frac{\sigma^2}{2} \frac{\partial^2 f}{\partial x^2}\right) dt + \sigma \frac{\partial f}{\partial x} \, dW_t
$$

**Shorthand**: $dW_t^2 = dt$, $dt \cdot dW_t = 0$, $dt^2 = 0$

**Example**: Deriving GBM solution

Given $dS_t = \mu S_t dt + \sigma S_t dW_t$, let $f(S, t) = \log S$:

$$
\begin{align}
d\log S_t &= \frac{1}{S_t} dS_t - \frac{1}{2S_t^2} (dS_t)^2 \\
&= \frac{1}{S_t}(\mu S_t dt + \sigma S_t dW_t) - \frac{1}{2S_t^2} \sigma^2 S_t^2 dt \\
&= \left(\mu - \frac{\sigma^2}{2}\right) dt + \sigma dW_t
\end{align}
$$

Integrating: $\log S_t = \log S_0 + (\mu - \sigma^2/2)t + \sigma W_t$

**Reference**: [Shreve, 2004], Chapter 4; [Björk, 2009], Chapter 4

**Implementation**: Itô's lemma is used throughout for model derivations and is leveraged via JAX automatic differentiation.

### Multidimensional Itô's Lemma

For $\mathbf{X}_t \in \mathbb{R}^n$ satisfying:

$$
d\mathbf{X}_t = \boldsymbol{\mu}(\mathbf{X}_t, t) \, dt + \boldsymbol{\Sigma}(\mathbf{X}_t, t) \, d\mathbf{W}_t
$$

where $\mathbf{W}_t$ is a $d$-dimensional Brownian motion, and $f(\mathbf{x}, t) \in C^{2,1}$:

$$
df(\mathbf{X}_t, t) = \frac{\partial f}{\partial t} dt + \sum_{i=1}^n \frac{\partial f}{\partial x_i} dX_t^i + \frac{1}{2} \sum_{i,j=1}^n \frac{\partial^2 f}{\partial x_i \partial x_j} dX_t^i dX_t^j
$$

where $dX_t^i dX_t^j = (\boldsymbol{\Sigma} \boldsymbol{\Sigma}^T)_{ij} dt$

**Application**: Multi-asset derivatives, Heston model (2D system)

**Reference**: [Shreve, 2004], Chapter 4

---

## Risk-Neutral Pricing

### Fundamental Theorem of Asset Pricing

The **First Fundamental Theorem** states:

> A market model is **arbitrage-free** if and only if there exists an **equivalent martingale measure** $\mathbb{Q}$ (risk-neutral measure).

Under $\mathbb{Q}$, discounted asset prices are martingales:

$$
\frac{S_t}{B_t} = \mathbb{E}^{\mathbb{Q}}\left[\frac{S_T}{B_T} \, \Big| \, \mathcal{F}_t\right]
$$

where $B_t = e^{\int_0^t r_s ds}$ is the money market account.

**Second Fundamental Theorem**: The market is **complete** if and only if the risk-neutral measure $\mathbb{Q}$ is unique.

**Reference**: [Shreve, 2004], Chapter 5; [Björk, 2009], Chapter 10

### Risk-Neutral Pricing Formula

The **arbitrage-free price** of a contingent claim $V_T = g(S_T)$ at time $t$ is:

$$
V_t = e^{-r(T-t)} \mathbb{E}^{\mathbb{Q}}[g(S_T) | \mathcal{F}_t]
$$

**Interpretation**:
1. Compute expected payoff under $\mathbb{Q}$ (not real-world $\mathbb{P}$)
2. Discount at risk-free rate $r$

**Example**: European call option

$$
C(S_0, K, T) = e^{-rT} \mathbb{E}^{\mathbb{Q}}[\max(S_T - K, 0)]
$$

**Reference**: [Hull, 2022], Chapter 13; [Shreve, 2004], Chapter 5

**Implementation**: All pricing engines in `src/neutryx/engines/` implement risk-neutral pricing via Monte Carlo, PDE, or Fourier methods.

### Girsanov Theorem

**Girsanov's Theorem** allows us to change the drift of a Brownian motion by changing the probability measure.

Let $\{W_t\}$ be a Brownian motion under $\mathbb{P}$. Define the Radon-Nikodym derivative:

$$
\frac{d\mathbb{Q}}{d\mathbb{P}}\Big|_{\mathcal{F}_T} = Z_T = \exp\left(-\int_0^T \theta_s \, dW_s - \frac{1}{2}\int_0^T \theta_s^2 \, ds\right)
$$

Then under $\mathbb{Q}$:

$$
\tilde{W}_t = W_t + \int_0^t \theta_s \, ds
$$

is a Brownian motion.

**Application**: Deriving the risk-neutral measure from physical measure

Under $\mathbb{P}$: $dS_t = \mu S_t dt + \sigma S_t dW_t$

Choose $\theta_t = \frac{\mu - r}{\sigma}$ (market price of risk). Then under $\mathbb{Q}$:

$$
dS_t = r S_t dt + \sigma S_t d\tilde{W}_t
$$

**Reference**: [Shreve, 2004], Chapter 5; [Björk, 2009], Chapter 11

---

## Martingale Theory

### Martingales

A process $\{M_t\}$ adapted to $\{\mathcal{F}_t\}$ is a **martingale** under $\mathbb{P}$ if:

1. $\mathbb{E}[|M_t|] < \infty$ for all $t$
2. $\mathbb{E}[M_t | \mathcal{F}_s] = M_s$ for all $s \leq t$

**Interpretation**: Fair game - the expected future value equals the current value given current information.

**Examples**:
- Brownian motion $W_t$ is a martingale
- $W_t^2 - t$ is a martingale (compensated quadratic variation)
- Discounted stock price under $\mathbb{Q}$: $e^{-rt} S_t$

**Reference**: [Shreve, 2004], Chapter 3; [Björk, 2009], Chapter 6

### Doob's Optional Stopping Theorem

For a martingale $\{M_t\}$ and stopping time $\tau$ with $\mathbb{E}[\tau] < \infty$:

$$
\mathbb{E}[M_\tau] = \mathbb{E}[M_0]
$$

**Application**: Pricing American options (optimal stopping problems)

**Reference**: [Shreve, 2004], Chapter 3

---

## Change of Measure

### Change of Numeraire

The **change of numeraire** technique allows pricing in different units.

Let $N_t$ be a strictly positive traded asset (numeraire). Under the **$N$-forward measure** $\mathbb{Q}^N$:

$$
\frac{S_t}{N_t} = \mathbb{E}^{\mathbb{Q}^N}\left[\frac{S_T}{N_T} \, \Big| \, \mathcal{F}_t\right]
$$

**Common numeraires**:
- Money market account: $N_t = B_t = e^{rt}$ → standard risk-neutral measure $\mathbb{Q}$
- Zero-coupon bond: $N_t = P(t, T)$ → $T$-forward measure $\mathbb{Q}^T$
- Stock price: $N_t = S_t$ → stock measure (useful for volatility derivatives)

**Example**: Black's formula for bond options

Using the $T$-forward measure simplifies pricing: forward rates are martingales.

**Reference**: [Shreve, 2004], Chapter 6; [Brigo & Mercurio, 2006], Chapter 2

**Implementation**: Used implicitly in interest rate models (`src/neutryx/models/vasicek.py`, `hull_white.py`)

### Cameron-Martin-Girsanov Formula

The general form of measure change (for continuous semimartingales):

$$
\frac{d\mathbb{Q}}{d\mathbb{P}}\Big|_{\mathcal{F}_T} = \exp\left(\int_0^T \theta_s \, dX_s - \frac{1}{2}\int_0^T \theta_s^2 \, d[X]_s\right)
$$

where $[X]_s$ is the quadratic variation of $X$.

**Reference**: [Shreve, 2004], Chapter 5

---

## Feynman-Kac Formula

The **Feynman-Kac theorem** connects PDEs and expectations, providing the foundation for both PDE and Monte Carlo pricing.

**Theorem**: Suppose $X_t$ satisfies:

$$
dX_t = \mu(X_t, t) \, dt + \sigma(X_t, t) \, dW_t^{\mathbb{Q}}
$$

and define:

$$
u(x, t) = \mathbb{E}^{\mathbb{Q}}\left[e^{-\int_t^T r(X_s, s) ds} g(X_T) \, \Big| \, X_t = x\right]
$$

Then $u(x, t)$ satisfies the PDE:

$$
\frac{\partial u}{\partial t} + \mu(x, t) \frac{\partial u}{\partial x} + \frac{\sigma^2(x, t)}{2} \frac{\partial^2 u}{\partial x^2} - r(x, t) u = 0
$$

with terminal condition $u(x, T) = g(x)$.

**Black-Scholes PDE**: For GBM under $\mathbb{Q}$ ($\mu = r$):

$$
\frac{\partial V}{\partial t} + r S \frac{\partial V}{\partial S} + \frac{\sigma^2 S^2}{2} \frac{\partial^2 V}{\partial S^2} - r V = 0
$$

**Reference**: [Shreve, 2004], Chapter 5; [Björk, 2009], Chapter 7

**Implementation**:
- Monte Carlo: Directly samples the expectation (`src/neutryx/engines/mc.py`)
- PDE: Discretizes and solves the PDE (`src/neutryx/models/pde.py`)

---

## Numerical Implementation Notes

### JAX and Automatic Differentiation

Neutryx leverages **JAX** for automatic differentiation, which computes derivatives via:

- **Forward mode**: Efficient for $\mathbb{R}^n \to \mathbb{R}^m$ with $n \ll m$
- **Reverse mode** (backpropagation): Efficient for $\mathbb{R}^n \to \mathbb{R}^m$ with $m \ll n$

**Greeks computation**: $\frac{\partial V}{\partial S}$, $\frac{\partial^2 V}{\partial S^2}$ computed exactly via autodiff

**Reference**: [Bradbury et al., 2018]

**Implementation**: All models support automatic Greeks via JAX: `src/neutryx/valuations/greeks/`

### Precision and Stability

- **Float32 vs Float64**: Configurable precision (GPU prefers float32, CPU can use float64)
- **Log-space computations**: Avoid overflow in characteristic functions
- **Variance reduction**: Reduces simulation noise, improving convergence

**Implementation**: Precision set globally; variance reduction in `src/neutryx/engines/variance_reduction.py`

---

## Summary

This mathematical foundation provides:

1. **Probability framework**: Filtered probability spaces, conditional expectation
2. **Stochastic processes**: Brownian motion, Lévy processes, jump processes
3. **Stochastic calculus**: Itô's lemma, quadratic variation
4. **Risk-neutral pricing**: Fundamental theorems, martingale measures
5. **Measure changes**: Girsanov theorem, change of numeraire
6. **Feynman-Kac**: PDE-expectation duality

All Neutryx models and numerical methods are built on these rigorous mathematical foundations, as detailed in [References](../references.md).

---

**Next**: [Pricing Models Theory](pricing_models.md) | [Numerical Methods Theory](numerical_methods.md)
