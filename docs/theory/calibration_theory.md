# Calibration Theory

This document provides the theoretical foundation for model calibration in quantitative finance.

## Table of Contents

1. [Calibration Problem Formulation](#calibration-problem-formulation)
2. [Optimization Methods](#optimization-methods)
3. [Loss Functions](#loss-functions)
4. [Parameter Constraints](#parameter-constraints)
5. [Regularization](#regularization)
6. [Calibration Diagnostics](#calibration-diagnostics)
7. [Model-Specific Calibration](#model-specific-calibration)

---

## Calibration Problem Formulation

### Inverse Problem

Model calibration is an **inverse problem**: Given observed market prices $\{P_i^{\text{market}}\}_{i=1}^N$, find model parameters $\boldsymbol{\theta}$ such that model prices $P_i^{\text{model}}(\boldsymbol{\theta})$ match the market.

**Objective**:

$$
\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta} \in \Theta} \mathcal{L}(\boldsymbol{\theta})
$$

where $\mathcal{L}(\boldsymbol{\theta})$ is a **loss function** measuring model-market discrepancy.

**Reference**: [Cont & Tankov, 2009]; [Guyon & Henry-Labordère, 2014]

### Ill-Posedness

Calibration problems are often **ill-posed** in the sense of Hadamard:

1. **Existence**: Solution may not exist (model cannot fit all prices)
2. **Uniqueness**: Multiple parameter sets may fit equally well
3. **Stability**: Small changes in market data → large changes in $\boldsymbol{\theta}$

**Consequences**:
- Need **regularization** for stability
- **Overfitting** risk (fitting noise rather than signal)
- **Prior information** helps (parameter bounds, market conventions)

**Reference**: [Cont & Tankov, 2009]; [Engl et al., 1996]

### Well-Posed Calibration

To ensure well-posedness:

1. **Parameter bounds**: $\boldsymbol{\theta} \in [\boldsymbol{\theta}_{\min}, \boldsymbol{\theta}_{\max}]$
2. **Regularization**: Add penalty term $R(\boldsymbol{\theta})$
3. **Prior selection**: Choose liquid instruments with tight spreads
4. **Model selection**: Use parsimonious models (fewer parameters)

**Reference**: [Guyon & Henry-Labordère, 2014], Chapter 4

---

## Optimization Methods

### Gradient-Based Methods

#### Gradient Descent

**Update rule**:

$$
\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \alpha_k \nabla \mathcal{L}(\boldsymbol{\theta}_k)
$$

where $\alpha_k > 0$ is the learning rate.

**Convergence**: $O(1/k)$ for convex $\mathcal{L}$

**Variants**:
- **Momentum**: Adds inertia to escape local minima
- **Nesterov**: Accelerated gradient (look-ahead)

**Reference**: [Nocedal & Wright, 2006], Chapter 3

#### Adam Optimizer

**Adaptive Moment Estimation (Adam)** [Kingma & Ba, 2014] is the de facto standard in machine learning.

**Update**:

$$
\begin{align}
m_k &= \beta_1 m_{k-1} + (1 - \beta_1) g_k \quad \text{(first moment)} \\
v_k &= \beta_2 v_{k-1} + (1 - \beta_2) g_k^2 \quad \text{(second moment)} \\
\hat{m}_k &= m_k / (1 - \beta_1^k), \quad \hat{v}_k = v_k / (1 - \beta_2^k) \quad \text{(bias correction)} \\
\boldsymbol{\theta}_{k+1} &= \boldsymbol{\theta}_k - \alpha \frac{\hat{m}_k}{\sqrt{\hat{v}_k} + \varepsilon}
\end{align}
$$

**Hyperparameters**: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$ (defaults)

**Advantages**:
- Adaptive learning rates per parameter
- Robust to noisy gradients
- Little tuning required

**Reference**: [Kingma & Ba, 2014]

**Implementation**: `src/neutryx/calibration/` uses Adam via `optax` library

#### L-BFGS-B

**Limited-memory BFGS with Box constraints** is a quasi-Newton method.

**Idea**: Approximate Hessian $H \approx \nabla^2 \mathcal{L}$ using gradient history (last $m$ iterations).

**Update**:

$$
\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \alpha_k H_k^{-1} \nabla \mathcal{L}(\boldsymbol{\theta}_k)
$$

**Advantages**:
- Superlinear convergence (near minimum)
- Handles box constraints naturally
- Memory-efficient ($O(md)$ storage)

**Disadvantages**:
- Requires smooth $\mathcal{L}$ (not robust to noise)
- Can get stuck in local minima

**Reference**: [Nocedal & Wright, 2006], Chapter 7; [Byrd et al., 1995]

### Gradient-Free Methods

#### Differential Evolution

**Differential Evolution (DE)** [Storn & Price, 1997] is a global optimization algorithm.

**Procedure**:
1. Initialize population of $P$ candidates
2. For each candidate $\boldsymbol{\theta}_i$:
   - **Mutation**: $\boldsymbol{\theta}'_i = \boldsymbol{\theta}_a + F(\boldsymbol{\theta}_b - \boldsymbol{\theta}_c)$ (random triplet)
   - **Crossover**: Mix $\boldsymbol{\theta}_i$ and $\boldsymbol{\theta}'_i$ with probability $CR$
   - **Selection**: Keep better candidate

**Advantages**:
- Global search (avoids local minima)
- No gradient required
- Robust to noisy objectives

**Disadvantages**:
- Slow convergence (many function evaluations)
- Not suitable for high-dimensional problems ($d > 20$)

**Reference**: [Storn & Price, 1997]; [Price et al., 2005]

#### Nelder-Mead Simplex

**Nelder-Mead** [Nelder & Mead, 1965] uses a simplex ($(d+1)$ points in $d$ dimensions) that reflects, expands, and contracts.

**Operations**: Reflect, expand, contract, shrink

**Advantages**:
- Derivative-free
- Simple implementation

**Disadvantages**:
- Slow for $d > 10$
- Can stagnate without converging

**Reference**: [Nelder & Mead, 1965]

### Gradient Computation

**Finite Differences**:

$$
\frac{\partial \mathcal{L}}{\partial \theta_j} \approx \frac{\mathcal{L}(\boldsymbol{\theta} + h \mathbf{e}_j) - \mathcal{L}(\boldsymbol{\theta} - h \mathbf{e}_j)}{2h}
$$

Cost: $O(d)$ function evaluations

**Automatic Differentiation (JAX)**:

$$
\nabla \mathcal{L}(\boldsymbol{\theta}) = \texttt{jax.grad}(\mathcal{L})(\boldsymbol{\theta})
$$

Cost: $O(1)$ (same as function evaluation, up to small constant)

**Advantage**: Exact gradients, enabling efficient gradient-based optimization.

**Reference**: [Bradbury et al., 2018]; [Griewank & Walther, 2008]

**Implementation**: All calibration in `src/neutryx/calibration/` uses JAX autodiff

---

## Loss Functions

### Mean Squared Error (MSE)

**Definition**:

$$
\mathcal{L}_{\text{MSE}}(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^N \left(P_i^{\text{market}} - P_i^{\text{model}}(\boldsymbol{\theta})\right)^2
$$

**Properties**:
- **Quadratic**: Penalizes large errors heavily
- **Differentiable**: Suitable for gradient methods
- **Scale-dependent**: Sensitive to price magnitudes

**Use case**: General-purpose calibration

**Reference**: Standard

### Root Mean Squared Error (RMSE)

$$
\mathcal{L}_{\text{RMSE}}(\boldsymbol{\theta}) = \sqrt{\frac{1}{N} \sum_{i=1}^N \left(P_i^{\text{market}} - P_i^{\text{model}}(\boldsymbol{\theta})\right)^2}
$$

**Advantage**: Same units as prices (interpretable)

### Relative Error

$$
\mathcal{L}_{\text{rel}}(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^N \left(\frac{P_i^{\text{market}} - P_i^{\text{model}}(\boldsymbol{\theta})}{P_i^{\text{market}}}\right)^2
$$

**Advantage**: Scale-invariant (treats all prices equally in percentage terms)

**Use case**: When prices span multiple orders of magnitude

### Implied Volatility Error

$$
\mathcal{L}_{\text{IV}}(\boldsymbol{\theta}) = \sum_{i=1}^N w_i \left(\sigma_i^{\text{market}} - \sigma_i^{\text{model}}(\boldsymbol{\theta})\right)^2
$$

where $\sigma_i$ is the **implied volatility** (Black-Scholes).

**Advantages**:
- **Market convention**: Traders quote volatility, not price
- **Comparable**: Volatilities are dimensionless percentages
- **Vega-weighted**: Implicitly weights by vega (sensitivity)

**Disadvantage**: Requires Black-Scholes inversion (may not exist for all prices)

**Reference**: [Gatheral, 2006], Chapter 2; [Cont & Tankov, 2009]

**Use case**: Standard for volatility surface calibration (Heston, SABR)

**Implementation**: `src/neutryx/calibration/losses.py`

### Vega-Weighted Loss

$$
\mathcal{L}_{\text{vega}}(\boldsymbol{\theta}) = \sum_{i=1}^N \mathcal{V}_i \left(P_i^{\text{market}} - P_i^{\text{model}}(\boldsymbol{\theta})\right)^2
$$

where $\mathcal{V}_i = \frac{\partial P_i}{\partial \sigma}$ is the vega.

**Motivation**: ATM options have high vega (most sensitive to vol) → weight them more.

**Effect**: Prioritizes fitting liquid, vega-rich options.

**Reference**: [Cont & Tankov, 2009]

### Bid-Ask Spread Weighting

$$
\mathcal{L}_{\text{spread}}(\boldsymbol{\theta}) = \sum_{i=1}^N \frac{1}{s_i^2} \left(P_i^{\text{mid}} - P_i^{\text{model}}(\boldsymbol{\theta})\right)^2
$$

where $s_i = P_i^{\text{ask}} - P_i^{\text{bid}}$ is the bid-ask spread.

**Motivation**: Tight spreads → high liquidity → more reliable prices → higher weight.

**Effect**: Ignores illiquid options with wide spreads.

**Reference**: Market practice

---

## Parameter Constraints

### Box Constraints

**Simple bounds**:

$$
\theta_j^{\min} \leq \theta_j \leq \theta_j^{\max}, \quad j = 1, \ldots, d
$$

**Example (Heston)**:
- $\kappa > 0$ (mean-reversion)
- $\theta > 0$ (long-term variance)
- $\sigma_v > 0$ (vol-of-vol)
- $\rho \in [-1, 1]$ (correlation)
- $v_0 > 0$ (initial variance)

**Enforcement**:
- Clipping: $\theta_j \gets \max(\theta_j^{\min}, \min(\theta_j, \theta_j^{\max}))$
- Projected gradient descent
- Barrier methods (interior-point)

**Reference**: [Nocedal & Wright, 2006], Chapter 16

**Implementation**: `src/neutryx/calibration/constraints.py`

### Parameter Transformations

**Unconstrained optimization** is easier. Transform constrained $\theta \in [a, b]$ to unconstrained $\phi \in \mathbb{R}$:

**Log transform** (positive constraints $\theta > 0$):

$$
\phi = \log \theta, \quad \theta = e^\phi
$$

**Logit transform** (bounded $\theta \in [a, b]$):

$$
\phi = \log\left(\frac{\theta - a}{b - \theta}\right), \quad \theta = a + \frac{b - a}{1 + e^{-\phi}}
$$

**Tanh transform** (bounded $\theta \in [-1, 1]$, e.g., correlation):

$$
\phi = \text{arctanh}(\theta), \quad \theta = \tanh(\phi)
$$

**Advantage**: Optimize over $\phi$ without constraints (simpler).

**Disadvantage**: Jacobian adjustment required for gradients.

**Reference**: [Nocedal & Wright, 2006]; [Guyon & Henry-Labordère, 2014]

**Implementation**: `src/neutryx/calibration/transforms.py`

### Feller Condition

For **CIR** and **Heston** models, the **Feller condition** ensures positivity:

$$
2\kappa\theta > \sigma^2 \quad \text{(CIR: } \sigma_v^2 \text{ for Heston)}
$$

**Enforcement**:
- Add penalty: $\mathcal{L}_{\text{total}} = \mathcal{L} + \lambda \max(0, \sigma_v^2 - 2\kappa\theta)$
- Hard constraint: Project parameters after each update

**Reference**: [Heston, 1993]; [Cox et al., 1985]

### No-Arbitrage Constraints

For **local volatility** and **implied volatility surfaces**:

1. **Calendar spread arbitrage**: $\frac{\partial C}{\partial T} \geq 0$
2. **Butterfly arbitrage**: $\frac{\partial^2 C}{\partial K^2} \geq 0$ (density non-negative)

**Enforcement**:
- Regularization (penalize violations)
- Constrained optimization
- Post-processing (arbitrage removal)

**Reference**: [Gatheral, 2006], Section 5.2; [Fengler, 2009]

---

## Regularization

### Tikhonov Regularization

**Penalize large parameter values**:

$$
\mathcal{L}_{\text{reg}}(\boldsymbol{\theta}) = \mathcal{L}(\boldsymbol{\theta}) + \lambda \|\boldsymbol{\theta} - \boldsymbol{\theta}_0\|^2
$$

where:
- $\boldsymbol{\theta}_0$: prior/initial guess
- $\lambda > 0$: regularization strength

**Effect**: Shrinks $\boldsymbol{\theta}$ toward prior (prevents overfitting).

**Reference**: [Engl et al., 1996]; [Guyon & Henry-Labordère, 2014]

### Total Variation Regularization

For **local volatility surfaces** $\sigma_{\text{loc}}(K, T)$:

$$
\mathcal{L}_{\text{TV}}(\sigma) = \mathcal{L}(\sigma) + \lambda \int \|\nabla \sigma(K, T)\| \, dK \, dT
$$

**Effect**: Penalizes rapid changes (produces smooth surfaces).

**Reference**: [Fengler, 2009]; [Andersen & Brotherton-Ratcliffe, 1998]

### Early Stopping

In **iterative optimization**, stop before full convergence:

$$
k^* = \arg\min_k \mathcal{L}_{\text{validation}}(\boldsymbol{\theta}_k)
$$

**Effect**: Prevents overfitting to training data.

**Reference**: [Goodfellow et al., 2016], Section 7.8

---

## Calibration Diagnostics

### Residual Analysis

**Residuals**:

$$
r_i = P_i^{\text{market}} - P_i^{\text{model}}(\boldsymbol{\theta}^*)
$$

**Check**:
1. **Mean**: $\bar{r} \approx 0$ (unbiased)
2. **Pattern**: No systematic structure (e.g., smile across strikes)
3. **Outliers**: Flag instruments with $|r_i| > 3\hat{\sigma}_r$

**Visual**: Plot residuals vs. strike, maturity, moneyness.

**Reference**: Standard statistical practice

**Implementation**: `src/neutryx/calibration/diagnostics.py:residual_analysis()`

### Goodness of Fit

**R-squared**:

$$
R^2 = 1 - \frac{\sum_i r_i^2}{\sum_i (P_i^{\text{market}} - \bar{P}^{\text{market}})^2}
$$

**Interpretation**: Proportion of variance explained ($R^2 \approx 1$ is good).

**Mean Absolute Error (MAE)**:

$$
\text{MAE} = \frac{1}{N} \sum_{i=1}^N |r_i|
$$

**Reference**: Standard

### Parameter Uncertainty

**Covariance matrix** (Cramér-Rao bound):

$$
\text{Cov}(\boldsymbol{\theta}^*) \approx \left(\mathbf{J}^T \mathbf{J}\right)^{-1}
$$

where $\mathbf{J}$ is the **Jacobian**: $J_{ij} = \frac{\partial P_i}{\partial \theta_j}$.

**Standard errors**:

$$
\text{SE}(\theta_j) = \sqrt{[\text{Cov}(\boldsymbol{\theta}^*)]_{jj}}
$$

**Confidence intervals** (95%):

$$
\theta_j^* \pm 1.96 \cdot \text{SE}(\theta_j)
$$

**Reference**: [Nocedal & Wright, 2006], Section 10.4

**Implementation**: `src/neutryx/calibration/diagnostics.py:parameter_uncertainty()`

### Identifiability

**Correlation matrix** of parameters:

$$
\text{Corr}(\theta_i, \theta_j) = \frac{\text{Cov}(\theta_i, \theta_j)}{\text{SE}(\theta_i) \text{SE}(\theta_j)}
$$

**High correlation** (e.g., $|\text{Corr}| > 0.9$) indicates **weak identifiability**: Parameters compensate for each other.

**Example (Heston)**: $\kappa$ and $\theta$ often highly correlated (both control long-term behavior).

**Solution**: Fix one parameter or add prior information.

**Reference**: [Cont & Tankov, 2009]; [Guyon & Henry-Labordère, 2014]

**Implementation**: `src/neutryx/calibration/diagnostics.py:identifiability()`

---

## Model-Specific Calibration

### Heston Model

**Parameters**: $\boldsymbol{\theta} = \{v_0, \kappa, \theta, \sigma_v, \rho\}$

**Instruments**: European options across strikes and maturities.

**Loss**: Implied volatility error (vega-weighted).

$$
\mathcal{L}(\boldsymbol{\theta}) = \sum_{i,j} w_{ij} \left(\sigma_{ij}^{\text{market}} - \sigma_{ij}^{\text{Heston}}(\boldsymbol{\theta})\right)^2
$$

**Pricing**: FFT (Carr-Madan) or semi-analytical (Heston formula).

**Constraints**:
- $v_0, \kappa, \theta, \sigma_v > 0$
- $\rho \in [-1, 1]$
- Feller: $2\kappa\theta > \sigma_v^2$ (optional)

**Typical values** (equity):
- $v_0 \approx 0.04$ (20% vol)
- $\kappa \approx 2$ (mean-reversion)
- $\theta \approx 0.04$ (long-term vol)
- $\sigma_v \approx 0.4$ (vol-of-vol)
- $\rho \approx -0.7$ (negative correlation, leverage effect)

**Reference**: [Heston, 1993]; [Gatheral, 2006], Chapter 3

**Implementation**: `src/neutryx/calibration/heston.py`

### SABR Model

**Parameters**: $\boldsymbol{\theta} = \{\alpha, \beta, \rho, \nu\}$ (typically fix $\beta$).

**Instruments**: Swaption or cap volatilities (interest rate markets).

**Loss**: Implied volatility error.

**Pricing**: Hagan's approximation (fast, analytical).

**Constraints**:
- $\alpha, \nu > 0$
- $\beta \in [0, 1]$ (often fixed: $\beta = 0.5$ for shifted lognormal)
- $\rho \in [-1, 1]$

**Typical values** (interest rates):
- $\alpha \approx 0.02$ (ATM vol)
- $\beta = 0.5$ (fixed)
- $\rho \approx -0.3$ (negative correlation)
- $\nu \approx 0.3$ (vol-of-vol)

**Reference**: [Hagan et al., 2002]; [Gatheral, 2006], Chapter 4

**Implementation**: `src/neutryx/calibration/sabr.py`

### Jump-Diffusion (Merton)

**Parameters**: $\boldsymbol{\theta} = \{\sigma, \lambda, \mu_J, \sigma_J\}$

**Instruments**: Short-dated options (capture jump risk).

**Loss**: Price or implied volatility error.

**Pricing**: Analytical series or FFT.

**Constraints**:
- $\sigma, \lambda, \sigma_J > 0$
- $\mu_J \in \mathbb{R}$

**Typical values**:
- $\lambda \approx 0.5$ (1 jump every 2 years)
- $\mu_J \approx -0.1$ (10% downward jump on average)
- $\sigma_J \approx 0.2$ (20% jump volatility)

**Reference**: [Merton, 1976]; [Cont & Tankov, 2004], Chapter 9

**Implementation**: `src/neutryx/calibration/jump_diffusion.py`

### Local Volatility (Dupire)

**Parameters**: $\sigma_{\text{loc}}(K, T)$ (function, not finite-dimensional)

**Instruments**: Complete implied volatility surface.

**Method**: Dupire's formula (analytical, no optimization).

$$
\sigma_{\text{loc}}^2(K, T) = \frac{\frac{\partial C}{\partial T}}{\frac{1}{2} K^2 \frac{\partial^2 C}{\partial K^2}}
$$

**Challenges**:
- Noisy derivatives → regularization required
- Arbitrage-free interpolation
- Extrapolation beyond liquid strikes

**Reference**: [Dupire, 1994]; [Gatheral, 2006], Chapter 5

**Implementation**: `src/neutryx/calibration/local_vol.py`

---

## Practical Considerations

### Multi-Start Optimization

**Problem**: Non-convex loss landscapes have many local minima.

**Solution**: Run optimization from $M$ random initializations, select best result.

**Initialization strategies**:
1. Random sampling within bounds
2. Latin hypercube sampling (space-filling)
3. Prior-based (perturb market-calibrated parameters)

**Reference**: Market practice

### Incremental Calibration

**Daily recalibration**: Use previous day's parameters as initial guess.

**Advantage**: Warm start (faster convergence, parameter stability).

**Disadvantage**: Can get stuck in local minima if market regime shifts.

**Solution**: Periodic global search (e.g., weekly).

**Reference**: Market practice

### Cross-Validation

**Procedure**:
1. Split data: Training (80%), validation (20%)
2. Calibrate on training set
3. Evaluate loss on validation set
4. Select hyperparameters ($\lambda$, model complexity) minimizing validation loss

**Prevents**: Overfitting to calibration data.

**Reference**: [Goodfellow et al., 2016], Chapter 5

---

## Summary

This document covered:

1. **Problem formulation**: Inverse problem, ill-posedness
2. **Optimization**: Gradient-based (Adam, L-BFGS-B) and gradient-free (DE, Nelder-Mead)
3. **Loss functions**: MSE, RMSE, implied vol, vega-weighted, spread-weighted
4. **Constraints**: Box constraints, transformations, Feller condition, no-arbitrage
5. **Regularization**: Tikhonov, total variation, early stopping
6. **Diagnostics**: Residuals, R-squared, parameter uncertainty, identifiability
7. **Model-specific**: Heston, SABR, jump-diffusion, local volatility

All calibration methods are implemented in `src/neutryx/calibration/` with extensive diagnostics and validation.

**See also**: [Pricing Models Theory](pricing_models.md) | [Numerical Methods](numerical_methods.md)

---

**References**: See [Bibliography](../references.md) for complete citations.
