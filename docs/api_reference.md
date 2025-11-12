# API Reference

Comprehensive reference documentation for the Neutryx Core Python API.

## Overview

Neutryx Core provides a complete JAX-first quantitative finance platform with:

- **500+ tests** ensuring production-ready quality
- **Multi-asset class coverage**: Interest rates, FX, equity, credit, commodity
- **Enterprise features**: SSO/OAuth/MFA, Kubernetes orchestration, observability
- **Regulatory compliance**: FRTB SA/IMA, SA-CCR, SIMM, IFRS 9/13
- **Advanced analytics**: Backtesting, factor analysis, portfolio optimization

---

## Quick Navigation

| Module | Description |
|--------|-------------|
| [`core`](#core-modules) | Core pricing engines, Monte Carlo, PDE solvers |
| [`models`](#models) | Stochastic models (Heston, SABR, Hull-White, etc.) |
| [`products`](#products) | Multi-asset class derivatives products |
| [`risk`](#risk-management) | VaR, stress testing, Greeks, P&L attribution |
| [`valuations`](#valuations-xva) | XVA, margin, collateral, regulatory capital |
| [`calibration`](#calibration) | Model calibration, regularization, diagnostics |
| [`market`](#market-data) | Market data adapters, storage, validation |
| [`research`](#research-analytics) | Backtesting, factor analysis, performance |
| [`analytics`](#analytics) | Factor models, PCA, style attribution |
| [`infrastructure`](#infrastructure) | Observability, governance, authentication |
| [`regulatory`](#regulatory-compliance) | FRTB, SA-CCR, SIMM, Basel III/IV |
| [`api`](#api-services) | REST/gRPC endpoints, web services |

---

## Core Modules

### `neutryx.core`

Core computational infrastructure for pricing and risk calculations.

#### Key Components

**Engine Configuration**
```python
from neutryx.core.engine import MCConfig, PDEConfig, simulate_gbm, present_value

# Monte Carlo configuration
mc_config = MCConfig(
    steps=252,        # Time steps
    paths=100_000,    # Number of paths
    seed=42,          # Random seed
    antithetic=True   # Use antithetic variates
)

# Simulate GBM paths
paths = simulate_gbm(key, S0=100.0, mu=0.05, sigma=0.2, T=1.0, config=mc_config)

# Present value calculation
pv = present_value(payoff, maturity, discount_rate)
```

**PDE Grid Configuration**
```python
# PDE solver configuration
pde_config = PDEConfig(
    spatial_points=200,
    time_steps=100,
    theta=0.5  # Crank-Nicolson scheme
)
```

### `neutryx.core.rng`

Random number generation with reproducibility.

```python
from neutryx.core.rng import PRNGManager

# Initialize PRNG with seed
prng = PRNGManager(seed=42)
key = prng.get_key()

# Split for parallel operations
keys = prng.split(n=10)
```

### `neutryx.core.grid`

Grid generation for finite difference methods.

```python
from neutryx.core.grid import Grid1D, Grid2D

# 1D grid for vanilla options
grid = Grid1D(xmin=0.0, xmax=200.0, nx=200)

# 2D grid for basket options
grid2d = Grid2D(
    xmin=0.0, xmax=200.0, nx=100,
    ymin=0.0, ymax=200.0, ny=100
)
```

---

## Models

### Equity Models

#### Black-Scholes (`neutryx.models.bs`)

```python
from neutryx.models.bs import price, greeks, implied_volatility

# Vanilla option pricing
call_price = price(S=100, K=100, T=1.0, r=0.05, q=0.02, sigma=0.2, option_type="call")

# Greeks calculation
delta, gamma, vega, theta, rho = greeks(S=100, K=100, T=1.0, r=0.05, q=0.02, sigma=0.2)

# Implied volatility
iv = implied_volatility(price=10.5, S=100, K=100, T=1.0, r=0.05, q=0.02, option_type="call")
```

#### Heston Model (`neutryx.models.heston`)

Stochastic volatility model for equity and FX options.

```python
from neutryx.models.heston import (
    HestonParams,
    characteristic_function,
    price_european_call_put,
    calibrate_to_surface
)

# Heston parameters
params = HestonParams(
    v0=0.04,      # Initial variance
    theta=0.04,   # Long-term variance
    kappa=2.0,    # Mean reversion speed
    sigma_v=0.3,  # Vol of vol
    rho=-0.7      # Correlation
)

# Price European option
call_price = price_european_call_put(
    S=100, K=100, T=1.0, r=0.05, q=0.02, params=params, option_type="call"
)

# Calibrate to market surface
calibrated_params = calibrate_to_surface(strikes, maturities, market_prices, S=100, r=0.05, q=0.02)
```

#### Local Volatility - Dupire (`neutryx.models.dupire`)

```python
from neutryx.models.dupire import LocalVolModel, calibrate_local_vol

# Calibrate local vol surface from market
local_vol_model = calibrate_local_vol(
    spot=100.0,
    strikes=strikes_array,
    maturities=maturities_array,
    market_ivs=implied_vols_matrix
)

# Price exotic with local vol
price = local_vol_model.price_barrier(K=100, H=120, T=1.0, barrier_type="up-and-out")
```

#### Stochastic Local Volatility (`neutryx.models.equity_models`)

```python
from neutryx.models.equity_models import SLVModel

# Hybrid SLV model
slv = SLVModel(heston_params=heston_params, local_vol_surface=local_vol)
price = slv.price_european(S=100, K=100, T=1.0)
```

#### Jump Diffusion Models

```python
from neutryx.models.equity_models import (
    MertonJumpDiffusion,
    KouJumpDiffusion,
    VarianceGammaModel
)

# Merton jump-diffusion
merton = MertonJumpDiffusion(
    sigma=0.2,      # Diffusion volatility
    lambda_jump=0.5, # Jump intensity
    mu_jump=-0.05,  # Jump mean
    sigma_jump=0.1  # Jump volatility
)
price = merton.price_european(S=100, K=100, T=1.0, r=0.05)

# Variance Gamma
vg = VarianceGammaModel(sigma=0.2, nu=0.1, theta=-0.1)
```

### Interest Rate Models

#### Hull-White Model (`neutryx.models.hull_white`)

```python
from neutryx.models.hull_white import (
    HullWhite1F,
    HullWhite2F,
    calibrate_hw1f,
    price_zero_coupon_bond
)

# 1-factor Hull-White
hw1f = HullWhite1F(a=0.1, sigma=0.01)
zcb_price = price_zero_coupon_bond(r0=0.03, T=5.0, model=hw1f)

# 2-factor Hull-White for richer term structure dynamics
hw2f = HullWhite2F(a1=0.1, a2=0.2, sigma1=0.01, sigma2=0.015, rho=-0.3)

# Calibrate to cap/floor market
calibrated_hw = calibrate_hw1f(
    cap_strikes=strikes,
    cap_maturities=maturities,
    market_prices=prices,
    curve=discount_curve
)
```

#### Black-Karasinski (`neutryx.models.black_karasinski`)

```python
from neutryx.models.black_karasinski import BlackKarasinski, calibrate_bk

# Black-Karasinski (log-normal short rate)
bk = BlackKarasinski(a=0.1, sigma=0.15)
bond_price = bk.price_bond(T=10.0, r0=0.03)
```

#### CIR Model (`neutryx.models.cir`)

```python
from neutryx.models.cir import CIRModel, calibrate_cir

# Cox-Ingersoll-Ross (strictly positive rates)
cir = CIRModel(kappa=0.15, theta=0.04, sigma=0.05)
bond_price = cir.bond_price(r0=0.03, T=5.0)
```

#### LMM/BGM (`neutryx.models.lmm`)

LIBOR Market Model for interest rate derivatives.

```python
from neutryx.models.lmm import LMMModel, calibrate_lmm

# LMM with full volatility and correlation structure
lmm = LMMModel(
    tenors=[0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
    volatilities=vol_matrix,
    correlations=corr_matrix
)

# Price swaption with LMM
swaption_price = lmm.price_swaption(
    strike=0.05,
    option_maturity=2.0,
    swap_maturity=10.0
)
```

#### G2++ Model (`neutryx.models.g2pp`) 🆕

Two-factor Gaussian interest rate model.

```python
from neutryx.models.g2pp import G2PPModel

g2 = G2PPModel(
    a=0.1, b=0.2,           # Mean reversion speeds
    sigma=0.01, eta=0.015,  # Volatilities
    rho=-0.3                # Correlation
)
```

### FX Models

#### Garman-Kohlhagen (`neutryx.models.fx_models`)

```python
from neutryx.models.fx_models import (
    garman_kohlhagen_price,
    fx_heston_price,
    fx_sabr_smile
)

# FX vanilla option (Garman-Kohlhagen)
fx_call = garman_kohlhagen_price(
    S=1.20,        # FX spot
    K=1.25,        # Strike
    T=1.0,         # Maturity
    r_dom=0.02,    # Domestic rate
    r_for=0.01,    # Foreign rate
    sigma=0.10     # Volatility
)

# FX Heston model
fx_heston = fx_heston_price(S=1.20, K=1.25, T=1.0, r_dom=0.02, r_for=0.01, heston_params=params)

# FX SABR smile
smile = fx_sabr_smile(forward=1.20, strikes=strikes, T=1.0, alpha=0.1, beta=0.5, rho=-0.3, nu=0.3)
```

### Credit Models

#### Structural Models (`neutryx.models.credit_models`)

```python
from neutryx.models.credit_models import (
    MertonCreditModel,
    BlackCoxModel,
    JarrowTurnbullHazardModel
)

# Merton structural model
merton_credit = MertonCreditModel(
    asset_value=100,
    debt_face_value=80,
    asset_volatility=0.3,
    risk_free_rate=0.05,
    time_to_maturity=5.0
)
default_prob = merton_credit.default_probability()

# Jarrow-Turnbull hazard rate model
jt_model = JarrowTurnbullHazardModel(hazard_rate=0.02, recovery_rate=0.4)
cds_spread = jt_model.cds_spread(T=5.0)
```

#### Copula Models

```python
from neutryx.models.credit_models import (
    GaussianCopula,
    StudentTCopula,
    LargePortfolioApproximation
)

# Gaussian copula for CDO tranches
copula = GaussianCopula(correlation=0.3, n_names=125)
tranche_loss = copula.expected_tranche_loss(
    attachment=0.03,
    detachment=0.07,
    default_probs=default_probabilities
)

# Student-t copula for tail dependence
t_copula = StudentTCopula(correlation=0.3, degrees_of_freedom=5, n_names=125)
```

### Adaptive Mesh Refinement 🆕

#### AMR for PDEs (`neutryx.models.amr`)

Advanced adaptive mesh refinement for PDE solvers.

```python
from neutryx.models.amr import AMRSolver, AMRConfig

# Configure AMR
amr_config = AMRConfig(
    min_level=0,
    max_level=3,
    refine_threshold=0.01,
    coarsen_threshold=0.001
)

# Solve with adaptive refinement
solver = AMRSolver(config=amr_config)
solution = solver.solve_bs_pde(
    S_range=(0, 200),
    T=1.0,
    K=100,
    r=0.05,
    sigma=0.2
)
```

---

## Products

### Vanilla Options

#### European Options (`neutryx.products.base`)

```python
from neutryx.products.base import EuropeanOption

euro_call = EuropeanOption(
    strike=100.0,
    maturity=1.0,
    option_type="call"
)
price = euro_call.price(spot=100, vol=0.2, rate=0.05, div_yield=0.02)
delta = euro_call.delta(spot=100, vol=0.2, rate=0.05, div_yield=0.02)
```

#### American Options (`neutryx.products.american`)

Longstaff-Schwartz Monte Carlo for early exercise.

```python
from neutryx.products.american import AmericanOption, price_american_lsm

# American put option
american_put = AmericanOption(
    strike=100.0,
    maturity=1.0,
    option_type="put"
)

# Price with LSM
price = price_american_lsm(
    spot=100,
    strike=100,
    maturity=1.0,
    rate=0.05,
    div_yield=0.02,
    vol=0.2,
    option_type="put",
    n_paths=100_000,
    n_steps=50
)
```

### Path-Dependent Options

#### Asian Options (`neutryx.products.asian`)

```python
from neutryx.products.asian import AsianOption, price_asian_mc

asian = AsianOption(
    strike=100.0,
    maturity=1.0,
    option_type="call",
    averaging_type="arithmetic"  # or "geometric"
)
price = price_asian_mc(spot=100, strike=100, T=1.0, r=0.05, sigma=0.2, n_paths=100_000)
```

#### Barrier Options (`neutryx.products.barrier`)

```python
from neutryx.products.barrier import BarrierOption, price_barrier

barrier = BarrierOption(
    strike=100.0,
    barrier=120.0,
    maturity=1.0,
    barrier_type="up-and-out",
    option_type="call"
)
price = price_barrier(S=100, K=100, H=120, T=1.0, r=0.05, sigma=0.2, barrier_type="up-and-out")
```

### Interest Rate Products

#### Interest Rate Swaps (`neutryx.products.interest_rate`)

```python
from neutryx.products.interest_rate import (
    InterestRateSwap,
    OISSwap,
    CrossCurrencySwap,
    BasisSwap
)

# Vanilla IRS
irs = InterestRateSwap(
    notional=10_000_000,
    fixed_rate=0.03,
    tenor=5.0,
    payment_freq=0.5
)
pv = irs.present_value(curve=discount_curve, forward_curve=libor_curve)

# OIS (Overnight Index Swap)
ois = OISSwap(
    notional=10_000_000,
    fixed_rate=0.025,
    tenor=3.0,
    index="SOFR"  # or "ESTR", "SONIA"
)

# Cross-currency swap
ccs = CrossCurrencySwap(
    notional_domestic=10_000_000,
    notional_foreign=8_000_000,
    fixed_rate_dom=0.03,
    fixed_rate_for=0.02,
    tenor=5.0,
    fx_spot=1.25
)
```

#### Swaptions (`neutryx.products.swaptions`)

```python
from neutryx.products.swaptions import (
    EuropeanSwaption,
    AmericanSwaption,
    BermudanSwaption,
    european_swaption_black
)

# European swaption (Black formula)
payer_swaption = european_swaption_black(
    strike=0.05,
    option_maturity=1.0,
    swap_maturity=5.0,
    volatility=0.20,
    notional=1_000_000,
    is_payer=True
)

# Bermudan swaption (LSM)
bermudan = BermudanSwaption(
    strike=0.05,
    notional=1_000_000,
    exercise_dates=jnp.array([0.25, 0.5, 0.75, 1.0]),
    tenor=5.0,
    option_type='payer'
)
bermudan_price = bermudan.price_lsm(rate_paths, discount_factors)
```

#### Advanced Interest Rate Products (`neutryx.products.advanced_rates`)

```python
from neutryx.products.advanced_rates import (
    TargetRedemptionNote,
    RangeAccrual,
    SnowballNote,
    AutocallableNote
)

# TARN (Target Redemption Note)
tarn = TargetRedemptionNote(
    notional=1_000_000,
    target_coupon=100_000,
    coupon_rate=0.05,
    payment_freq=4,
    maturity=5.0
)

# Range accrual
range_accrual = RangeAccrual(
    notional=1_000_000,
    coupon_rate=0.06,
    lower_bound=0.02,
    upper_bound=0.04,
    tenor=3.0
)
```

#### CMS Products (`neutryx.products.interest_rate`)

```python
from neutryx.products.interest_rate import (
    cms_caplet_price,
    cms_floorlet_price,
    price_cms_spread_option
)

# CMS caplet with convexity adjustment
cms_cap = cms_caplet_price(
    cms_forward=0.04,
    strike=0.045,
    time_to_expiry=1.0,
    volatility=0.25,
    discount_factor=0.97,
    annuity=9.0,
    convexity_adjustment=0.002
)

# CMS spread option (10Y - 2Y)
spread_option = price_cms_spread_option(
    cms1_forward=0.045,
    cms2_forward=0.035,
    strike=0.01,
    time_to_expiry=1.0,
    spread_volatility=0.30,
    discount_factor=0.97,
    annuity=4.5,
    is_call=True
)
```

### Credit Derivatives

#### Credit Default Swaps (`neutryx.products.credit_derivatives`)

```python
from neutryx.products.credit_derivatives import (
    CDS,
    CDSIndex,
    CDOTranche,
    NthToDefault
)

# Single-name CDS
cds = CDS(
    notional=10_000_000,
    spread=100,  # bps
    maturity=5.0,
    recovery_rate=0.40
)
pv = cds.present_value(hazard_rate=0.02, discount_curve=curve)

# CDX/iTraxx index
cdx = CDSIndex(
    notional=10_000_000,
    index_spread=80,  # bps
    maturity=5.0,
    n_names=125
)

# CDO tranche
cdo_tranche = CDOTranche(
    notional=10_000_000,
    attachment=0.03,
    detachment=0.07,
    maturity=5.0
)
```

### Equity Derivatives

#### Equity Forwards and Swaps (`neutryx.products.equity`)

```python
from neutryx.products.equity import (
    equity_forward_price,
    variance_swap_value,
    dividend_swap_price,
    total_return_swap_value
)

# Equity forward
forward = equity_forward_price(
    spot=100.0,
    maturity=1.0,
    risk_free_rate=0.05,
    dividend_yield=0.02
)

# Variance swap
var_swap = variance_swap_value(
    realized_variance=0.04,
    strike_variance=0.0625,  # 25% vol squared
    notional=1_000_000,
    days_remaining=252
)
```

#### Autocallables and Structured Products

```python
from neutryx.products.equity_exotic import (
    PhoenixAutocallable,
    ReverseConvertible,
    WorstOfNote
)

# Phoenix autocallable
phoenix = PhoenixAutocallable(
    strike=100.0,
    barrier=70.0,
    coupon=0.10,
    autocall_barrier=100.0,
    observation_dates=[0.25, 0.5, 0.75, 1.0]
)
```

### FX Products

#### FX Exotics (`neutryx.products.fx`)

```python
from neutryx.products.fx import (
    FXForward,
    FXBarrier,
    TARF,
    FXAccumulator
)

# FX forward
fx_fwd = FXForward(
    notional=1_000_000,
    strike=1.25,
    maturity=0.25
)

# TARF (Target Redemption Forward)
tarf = TARF(
    notional=1_000_000,
    strike=1.25,
    target_profit=50_000,
    observation_dates=jnp.linspace(0, 1, 12)
)
```

### Commodity Derivatives

#### Commodity Products (`neutryx.products.commodity`)

```python
from neutryx.products.commodity import (
    commodity_forward_price,
    commodity_option_price,
    spread_option_price
)

# Commodity forward with convenience yield
forward = commodity_forward_price(
    spot=50.0,
    maturity=1.0,
    risk_free_rate=0.05,
    storage_cost=0.02,
    convenience_yield=0.03
)

# Commodity spread option (crack spread, calendar spread)
spread = spread_option_price(
    S1=50.0,
    S2=55.0,
    K=5.0,
    T=1.0,
    sigma1=0.30,
    sigma2=0.25,
    rho=0.6,
    is_call=True
)
```

---

## Risk Management

### VaR and Risk Metrics (`neutryx.risk.analytics`)

```python
from neutryx.risk.analytics import (
    historical_var,
    monte_carlo_var,
    parametric_var,
    expected_shortfall,
    incremental_var,
    component_var
)

# Historical VaR
var_95 = historical_var(returns, confidence=0.95)
var_99 = historical_var(returns, confidence=0.99)

# Expected Shortfall (CVaR)
es_95 = expected_shortfall(returns, confidence=0.95)

# Monte Carlo VaR
mc_var = monte_carlo_var(
    portfolio_value=10_000_000,
    returns_simulation=simulated_returns,
    confidence=0.99
)

# Component VaR (risk contribution by position)
comp_var = component_var(
    positions=position_weights,
    covariance=cov_matrix,
    portfolio_var=portfolio_var
)
```

### Stress Testing (`neutryx.risk.stress`)

```python
from neutryx.risk.stress import (
    apply_stress_scenario,
    historical_scenarios,
    hypothetical_scenarios,
    reverse_stress_test
)

# Historical stress scenarios
stress_results = historical_scenarios(
    portfolio=portfolio,
    scenarios=["2008_crisis", "covid_2020", "brexit", "lehman"]
)

# Hypothetical scenario
hypo_result = apply_stress_scenario(
    portfolio=portfolio,
    equity_shock=-0.30,      # -30% equity
    rate_shock=+0.02,        # +200bps rates
    fx_shock={"EURUSD": +0.10}  # +10% USD strength
)

# Reverse stress testing (find scenario causing target loss)
scenario = reverse_stress_test(
    portfolio=portfolio,
    target_loss=1_000_000,
    risk_factors=["equities", "rates", "credit_spreads"]
)
```

### Greeks and Sensitivities

```python
from neutryx.risk.greeks import (
    calculate_delta,
    calculate_gamma,
    calculate_vega,
    calculate_theta,
    calculate_rho,
    calculate_vanna,
    calculate_volga,
    calculate_charm
)

# First-order Greeks
delta = calculate_delta(portfolio, spot_shift=0.01)
vega = calculate_vega(portfolio, vol_shift=0.01)

# Second-order Greeks
gamma = calculate_gamma(portfolio, spot_shift=0.01)
volga = calculate_volga(portfolio, vol_shift=0.01)  # Vega convexity

# Cross Greeks
vanna = calculate_vanna(portfolio, spot_shift=0.01, vol_shift=0.01)  # dVega/dSpot
charm = calculate_charm(portfolio, spot_shift=0.01, time_shift=1/252)  # dDelta/dTime
```

### P&L Attribution (`neutryx.risk.pnl_attribution`)

```python
from neutryx.risk.pnl_attribution import (
    daily_pnl_explain,
    risk_factor_attribution,
    greeks_pnl_attribution
)

# Daily P&L decomposition
pnl_explain = daily_pnl_explain(
    portfolio_today=portfolio_t,
    portfolio_yesterday=portfolio_t1,
    market_data_today=market_t,
    market_data_yesterday=market_t1
)
# Returns: {carry, delta_pnl, gamma_pnl, vega_pnl, theta_pnl, unexplained}

# Risk factor attribution
factor_pnl = risk_factor_attribution(
    pnl=total_pnl,
    factor_changes={"equity_index": 0.02, "rates_10y": -0.0050, "fx_eurusd": 0.015}
)
```

---

## Valuations & XVA

### CVA/DVA/FVA (`neutryx.valuations.xva`)

```python
from neutryx.valuations.xva import (
    calculate_cva,
    calculate_dva,
    calculate_fva,
    calculate_mva,
    calculate_kva
)

# Credit Valuation Adjustment
cva = calculate_cva(
    exposure_profile=epe_profile,  # Expected Positive Exposure
    survival_probability=survival_curve,
    loss_given_default=0.60,
    discount_curve=curve
)

# Debit Valuation Adjustment (own default)
dva = calculate_dva(
    exposure_profile=ene_profile,  # Expected Negative Exposure
    survival_probability=own_survival_curve,
    loss_given_default=0.60,
    discount_curve=curve
)

# Funding Valuation Adjustment
fva = calculate_fva(
    exposure_profile=exposure,
    funding_spread=0.0050,  # 50bps funding spread
    discount_curve=curve
)

# Margin Valuation Adjustment
mva = calculate_mva(
    initial_margin_profile=im_profile,
    funding_spread=0.0050,
    discount_curve=curve
)

# Capital Valuation Adjustment (regulatory capital cost)
kva = calculate_kva(
    regulatory_capital=capital_profile,
    hurdle_rate=0.12,  # 12% ROE target
    discount_curve=curve
)
```

### Collateral Management (`neutryx.valuations.collateral`)

```python
from neutryx.valuations.collateral import (
    calculate_variation_margin,
    calculate_initial_margin_simm,
    margin_call_with_aging,
    collateral_optimization
)

# Variation margin
vm = calculate_variation_margin(
    mtm_today=1_500_000,
    mtm_yesterday=1_200_000,
    threshold=100_000,
    minimum_transfer_amount=50_000
)

# Initial margin (ISDA SIMM)
im = calculate_initial_margin_simm(
    portfolio=portfolio,
    sensitivities=delta_gamma_vega,
    simm_version="2.6"
)

# Margin call with aging
margin_call = margin_call_with_aging(
    required_collateral=2_000_000,
    posted_collateral=1_500_000,
    outstanding_calls=outstanding,
    valuation_date=date.today()
)
```

### Collateral Transformation 🆕

```python
from neutryx.valuations.collateral import (
    CollateralTransformationStrategy,
    optimize_collateral_basket
)

# Collateral transformation strategy
strategy = CollateralTransformationStrategy(
    available_collateral={"cash_usd": 1_000_000, "us_treasury": 5_000_000},
    required_collateral=3_000_000,
    haircuts={"cash_usd": 0.0, "us_treasury": 0.02},
    transformation_costs={"us_treasury_to_cash": 0.001}
)

optimal_allocation = strategy.optimize()
```

---

## Regulatory Compliance

### FRTB - Standardized Approach (`neutryx.regulatory.frtb`)

```python
from neutryx.valuations.regulatory.frtb import (
    calculate_delta_charge,
    calculate_vega_charge,
    calculate_curvature_charge,
    calculate_total_sa_capital
)

# Delta risk charge
delta_rc = calculate_delta_charge(
    sensitivities=delta_sensitivities,
    risk_class="IR",  # Interest Rate
    currency="USD"
)

# Vega risk charge
vega_rc = calculate_vega_charge(
    vega_sensitivities=vega_grid,
    risk_class="FX"
)

# Curvature risk charge
curv_rc = calculate_curvature_charge(
    curvature_sensitivities=curvature_grid,
    risk_class="Equity"
)

# Total SA capital
total_sa = calculate_total_sa_capital(
    delta_charge=delta_rc,
    vega_charge=vega_rc,
    curvature_charge=curv_rc
)
```

### FRTB - Internal Models Approach (IMA) 🆕

```python
from neutryx.regulatory.ima import (
    calculate_expected_shortfall,
    pla_test,
    backtest_traffic_light,
    identify_nmrf
)

# Expected Shortfall at 97.5%
es_97_5 = calculate_expected_shortfall(
    pnl_series=daily_pnl,
    confidence_level=0.975,
    liquidity_horizons={"equities": 10, "rates": 20, "credit": 60}
)

# P&L Attribution Test
pla_result = pla_test(
    risk_theoretical_pnl=rtpl,
    hypothetical_pnl=hpl,
    actual_pnl=apl
)

# Backtesting with traffic light approach
backtest_result = backtest_traffic_light(
    var_forecasts=var_series,
    actual_pnl=pnl_series,
    confidence_level=0.99
)
# Returns: {"breaches": 5, "zone": "yellow", "multiplier": 3.4}

# Non-modellable risk factors
nmrf = identify_nmrf(
    risk_factors=risk_factor_data,
    real_price_test_threshold=24,
    risk_factor_eligibility_test=rfet_criteria
)
```

### FRTB - Default Risk Charge (DRC) 🆕

```python
from neutryx.valuations.regulatory.frtb_drc import (
    calculate_drc_non_securitizations,
    calculate_drc_securitizations
)

# DRC for non-securitizations (bonds, CDS)
drc_nonsec = calculate_drc_non_securitizations(
    exposures=bond_exposures,
    credit_quality=rating_buckets,
    gross_jt m=gross_jump_to_default,
    hedging_sets=hedging_structure
)

# DRC for securitizations
drc_sec = calculate_drc_securitizations(
    exposures=securitization_exposures,
    attachment_points=attachment,
    detachment_points=detachment
)
```

### FRTB - Residual Risk Add-On (RRAO) 🆕

```python
from neutryx.valuations.regulatory.frtb_rrao import (
    calculate_rrao,
    classify_exotic_products
)

# Residual Risk Add-On for exotic payoffs
rrao = calculate_rrao(
    notional=10_000_000,
    product_type="barrier_option",
    underlying_type="FX"
)
# Returns RRAO capital charge based on product complexity
```

### SA-CCR (`neutryx.valuations.regulatory.sa_ccr`)

```python
from neutryx.valuations.regulatory.sa_ccr import (
    calculate_replacement_cost,
    calculate_pfe_addon,
    calculate_ead
)

# Replacement Cost (RC)
rc = calculate_replacement_cost(
    mtm=1_500_000,
    independent_collateral_amount=500_000,
    net_independent_collateral_amount=400_000
)

# Potential Future Exposure (PFE) Add-On
pfe = calculate_pfe_addon(
    trades=netting_set_trades,
    asset_class="IR",
    supervisory_delta=delta,
    maturity_factor=maturity_factor
)

# Exposure at Default
ead = calculate_ead(
    replacement_cost=rc,
    pfe_addon=pfe,
    alpha=1.4  # Supervisory alpha
)
```

### ISDA SIMM (`neutryx.valuations.simm`)

```python
from neutryx.valuations.simm import (
    calculate_simm_im,
    SIMMSensitivities,
    simm_risk_weights,
    simm_concentration_thresholds
)

# Calculate SIMM Initial Margin
sensitivities = SIMMSensitivities(
    delta=delta_grid,
    vega=vega_grid,
    curvature=curvature_grid
)

simm_im = calculate_simm_im(
    sensitivities=sensitivities,
    product_class="RatesFX",
    simm_version="2.6",
    calculation_currency="USD"
)
```

### Basel III Capital (`neutryx.regulatory.basel`)

```python
from neutryx.valuations.regulatory.basel import (
    calculate_credit_rwa,
    calculate_market_rwa,
    calculate_cva_capital,
    calculate_capital_ratios
)

# Credit Risk-Weighted Assets
credit_rwa = calculate_credit_rwa(
    exposures=credit_exposures,
    pd=probability_of_default,
    lgd=loss_given_default,
    approach="advanced_irb"
)

# CVA capital charge
cva_capital = calculate_cva_capital(
    cva_amount=cva,
    cva_volatility=cva_vol,
    approach="ba-cva"  # Basic Approach
)

# Capital ratios
ratios = calculate_capital_ratios(
    cet1_capital=10_000_000,
    tier1_capital=12_000_000,
    total_capital=15_000_000,
    rwa=100_000_000
)
# Returns: {cet1_ratio: 0.10, tier1_ratio: 0.12, total_ratio: 0.15}
```

### IFRS 9/13 Compliance

```python
from neutryx.accounting.ifrs import (
    classify_fair_value_level,
    calculate_ecl_staging,
    hedge_effectiveness_test
)

# Fair value hierarchy classification
fv_level = classify_fair_value_level(
    instrument=instrument,
    market_data_availability=availability,
    valuation_technique="model"
)
# Returns: "Level 1", "Level 2", or "Level 3"

# Expected Credit Loss (ECL) staging
ecl_stage = calculate_ecl_staging(
    current_credit_risk=current_risk,
    origination_credit_risk=origination_risk,
    significant_increase_threshold=0.10
)
# Returns: 1 (12-month ECL), 2 (lifetime ECL), or 3 (credit-impaired)

# Hedge effectiveness testing
effectiveness = hedge_effectiveness_test(
    hedge_pnl=hedge_pnl_series,
    hedged_item_pnl=item_pnl_series,
    test_type="prospective",  # or "retrospective"
    effectiveness_range=(0.80, 1.25)
)
```

---

## Calibration

### Model Calibration (`neutryx.calibration`)

```python
from neutryx.calibration import (
    calibrate_model,
    CalibrationConfig,
    LossFunction
)

# Calibration configuration
config = CalibrationConfig(
    optimizer="L-BFGS-B",
    max_iterations=1000,
    tolerance=1e-6,
    regularization="L2",
    regularization_strength=0.01
)

# Calibrate model to market
calibrated_params = calibrate_model(
    model=heston_model,
    market_data=market_prices,
    loss_fn=LossFunction.MSE,
    config=config
)
```

### Joint Calibration 🆕

```python
from neutryx.calibration.joint_calibration import (
    JointCalibrationFramework,
    calibrate_joint_instruments
)

# Joint calibration of caps and swaptions
joint_cal = calibrate_joint_instruments(
    instruments=["caps", "swaptions"],
    market_prices={"caps": cap_prices, "swaptions": swaption_prices},
    model=hull_white_2f,
    weights={"caps": 0.6, "swaptions": 0.4}  # Relative importance
)
```

### Bayesian Model Averaging 🆕

```python
from neutryx.calibration.bayesian_model_averaging import (
    BayesianModelAveraging,
    calculate_model_weights
)

# BMA across multiple models
bma = BayesianModelAveraging(
    models=[heston, sabr, local_vol],
    market_data=market_data
)

# Posterior model probabilities
weights = bma.calculate_weights(criterion="BIC")
# Returns: {heston: 0.45, sabr: 0.35, local_vol: 0.20}

# BMA price (weighted average)
bma_price = bma.price(strike=100, maturity=1.0)
```

### Regularization Techniques

```python
from neutryx.calibration.regularization import (
    TikhonovRegularization,
    L1Regularization,
    SmoothnessConstraint,
    ArbitrageFreeConstraint
)

# Tikhonov (L2) regularization
tikhonov = TikhonovRegularization(lambda_reg=0.01)

# Smoothness penalty for local vol
smoothness = SmoothnessConstraint(
    penalty_strike=0.1,
    penalty_maturity=0.1
)

# Enforce no-arbitrage
arbitrage_free = ArbitrageFreeConstraint(
    butterfly_tolerance=0.0,
    calendar_spread_tolerance=0.0
)
```

### Model Selection and Diagnostics

```python
from neutryx.calibration.model_selection import (
    calculate_aic,
    calculate_bic,
    k_fold_cross_validation,
    rolling_window_validation
)

# Information criteria
aic = calculate_aic(log_likelihood, n_params)
bic = calculate_bic(log_likelihood, n_params, n_observations)

# Cross-validation
cv_score = k_fold_cross_validation(
    model=model,
    data=market_data,
    n_folds=5
)

# Rolling window for time-series
rolling_scores = rolling_window_validation(
    model=model,
    time_series=historical_prices,
    window_size=252,  # 1 year
    forecast_horizon=20
)
```

---

## Market Data

### Bloomberg Integration (`neutryx.market.adapters.bloomberg`)

```python
from neutryx.market.adapters import BloombergAdapter, BloombergConfig

# Configure Bloomberg adapter
config = BloombergConfig(
    adapter_name="bloomberg",
    host="localhost",
    port=8194,
    timeout=30000
)

adapter = BloombergAdapter(config)

# Fetch real-time data
await adapter.connect()
quote = await adapter.get_quote("AAPL US Equity")

# Subscribe to real-time feed
await adapter.subscribe("equity", ["AAPL", "MSFT", "GOOGL"])
```

### Refinitiv Integration (`neutryx.market.adapters.refinitiv`)

```python
from neutryx.market.adapters import RefinitivAdapter, RefinitivConfig

config = RefinitivConfig(
    adapter_name="refinitiv",
    app_key="your_app_key",
    username="your_username"
)

adapter = RefinitivAdapter(config)
```

### TimescaleDB Storage (`neutryx.market.storage.timescale`)

```python
from neutryx.market.storage import TimescaleDBStorage, TimescaleDBConfig

# Configure TimescaleDB with compression
config = TimescaleDBConfig(
    host="localhost",
    port=5432,
    database="market_data",
    compression_enabled=True,
    compression_after_days=7,
    retention_policy_days=90
)

storage = TimescaleDBStorage(config)

# Store market data
await storage.store(
    asset_class="equity",
    symbol="AAPL",
    data=quote_data,
    timestamp=datetime.now()
)

# Query historical data
historical = await storage.query(
    asset_class="equity",
    symbols=["AAPL", "MSFT"],
    start_date=start,
    end_date=end
)
```

### Data Validation (`neutryx.market.validation`)

```python
from neutryx.market.validation import (
    ValidationPipeline,
    PriceRangeValidator,
    SpreadValidator,
    VolumeSpikeDetector
)

# Build validation pipeline
pipeline = ValidationPipeline()
pipeline.add_validator(PriceRangeValidator(max_jump_pct=0.20))
pipeline.add_validator(SpreadValidator(max_spread_bps=50))
pipeline.add_validator(VolumeSpikeDetector(threshold_sigma=5.0))

# Validate data
validation_result = pipeline.validate(market_data)
if not validation_result.is_valid:
    print(f"Validation failed: {validation_result.errors}")
```

---

## Research & Analytics

### Backtesting Framework 🆕

```python
from neutryx.research.backtest import (
    BacktestEngine,
    Strategy,
    TransactionCostModel
)

# Define trading strategy
class MyStrategy(Strategy):
    def generate_signals(self, data):
        # Your strategy logic
        return signals

    def size_positions(self, signals, capital):
        # Position sizing
        return positions

# Configure transaction costs
cost_model = TransactionCostModel(
    commission_pct=0.001,  # 10bps
    slippage_bps=5.0,
    market_impact_model="sqrt"
)

# Run backtest
engine = BacktestEngine(
    strategy=MyStrategy(),
    initial_capital=1_000_000,
    cost_model=cost_model
)

results = engine.run(
    data=historical_data,
    start_date="2020-01-01",
    end_date="2023-12-31"
)
```

### Walk-Forward Analysis 🆕

```python
from neutryx.research.walk_forward import (
    walk_forward_analysis,
    optimize_in_sample,
    validate_out_of_sample
)

# Walk-forward optimization
wf_results = walk_forward_analysis(
    strategy=strategy,
    data=historical_data,
    in_sample_period=252,   # 1 year
    out_of_sample_period=63,  # 3 months
    step_size=21  # 1 month
)
```

### Performance Attribution 🆕

```python
from neutryx.research.performance import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_max_drawdown,
    returns_attribution
)

# Performance metrics
sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
sortino = calculate_sortino_ratio(returns, mar=0.0)  # Minimum Acceptable Return
calmar = calculate_calmar_ratio(returns)
max_dd = calculate_max_drawdown(equity_curve)

# Returns attribution
attribution = returns_attribution(
    portfolio_returns=returns,
    factor_returns=factor_returns,
    factor_loadings=betas
)
```

---

## Analytics

### Factor Analysis 🆕

```python
from neutryx.analytics.factor_analysis import (
    FactorModel,
    PCA,
    BarraFactorModel,
    StyleAttribution
)

# Principal Component Analysis
pca = PCA(n_components=10)
factor_loadings = pca.fit(returns_matrix)
factor_returns = pca.transform(returns_matrix)

# Barra-style factor model
barra = BarraFactorModel(
    style_factors=["value", "growth", "momentum", "size"],
    industry_factors=["tech", "finance", "healthcare", "energy"]
)

factor_exposures = barra.calculate_exposures(stock_data)
factor_returns = barra.estimate_factor_returns(returns_data)

# Style attribution
attribution = StyleAttribution()
style_decomp = attribution.decompose(
    portfolio_returns=returns,
    style_exposures=exposures,
    style_factor_returns=factor_returns
)
# Returns: {value: 0.02, growth: 0.015, momentum: -0.005, ...}
```

### Portfolio Optimization

```python
from neutryx.portfolio.optimization import (
    MarkowitzOptimizer,
    RiskParityOptimizer,
    CVaROptimizer
)

# Mean-variance optimization (Markowitz)
markowitz = MarkowitzOptimizer(
    expected_returns=mu,
    covariance_matrix=sigma,
    risk_aversion=1.0
)
optimal_weights = markowitz.optimize()

# Risk parity
risk_parity = RiskParityOptimizer(covariance_matrix=sigma)
rp_weights = risk_parity.optimize()

# CVaR optimization
cvar_opt = CVaROptimizer(
    returns_scenarios=scenarios,
    confidence_level=0.95,
    target_return=0.10
)
cvar_weights = cvar_opt.optimize()
```

---

## Infrastructure

### Observability (`neutryx.infrastructure.observability`)

#### Prometheus Metrics

```python
from neutryx.infrastructure.observability.prometheus import (
    pricing_duration_metric,
    pricing_requests_total,
    active_sessions_gauge
)

# Record pricing duration
with pricing_duration_metric.time():
    price = price_option(...)

# Increment request counter
pricing_requests_total.labels(product="european_call", status="success").inc()
```

#### Distributed Tracing

```python
from neutryx.infrastructure.observability.tracing import (
    tracer,
    create_span
)

# Trace pricing workflow
with tracer.start_as_current_span("price_portfolio"):
    with create_span("load_market_data"):
        market_data = load_data()

    with create_span("calculate_sensitivities"):
        greeks = calculate_greeks(portfolio, market_data)
```

#### Performance Profiling

```python
from neutryx.infrastructure.observability.profiling import (
    profile_function,
    ProfilerConfig
)

@profile_function(threshold_ms=100)  # Profile if >100ms
def expensive_calculation():
    # Your code
    pass
```

### Authentication & Authorization 🆕

```python
from neutryx.infrastructure.auth import (
    OAuth2Provider,
    MFAProvider,
    LDAPConnector,
    RBACManager
)

# OAuth 2.0 / OpenID Connect
oauth = OAuth2Provider(
    client_id="your_client_id",
    client_secret="your_secret",
    redirect_uri="https://app.neutryx.tech/callback"
)

# Multi-Factor Authentication
mfa = MFAProvider(provider="totp")  # Time-based OTP
totp_secret = mfa.generate_secret()
is_valid = mfa.verify_token(user_token)

# LDAP Integration
ldap = LDAPConnector(
    server="ldap://ldap.company.com",
    base_dn="dc=company,dc=com"
)
user = ldap.authenticate(username, password)

# Role-Based Access Control
rbac = RBACManager()
rbac.assign_role(user_id="john.doe", role="trader")
has_permission = rbac.check_permission(user_id="john.doe", resource="portfolio", action="write")
```

### Kubernetes Orchestration 🆕

```python
from neutryx.infrastructure.k8s import (
    K8sDeployment,
    AutoScalingConfig,
    MultiRegionSetup
)

# Auto-scaling configuration
autoscaling = AutoScalingConfig(
    min_replicas=2,
    max_replicas=10,
    target_cpu_utilization=70,
    target_memory_utilization=80
)

# Multi-region deployment
multi_region = MultiRegionSetup(
    regions=["us-east-1", "eu-west-1", "ap-southeast-1"],
    replication_strategy="active-active",
    failover_policy="automatic"
)
```

---

## API Services

### REST API (`neutryx.api.rest`)

```python
from neutryx.api.rest import app
from fastapi import FastAPI

# Example endpoint usage
import httpx

# Price European option
response = httpx.post("http://api.neutryx.tech/v1/price/european", json={
    "spot": 100,
    "strike": 100,
    "maturity": 1.0,
    "rate": 0.05,
    "vol": 0.2,
    "option_type": "call"
})
price = response.json()["price"]

# Calculate portfolio VaR
response = httpx.post("http://api.neutryx.tech/v1/risk/var", json={
    "portfolio_id": "PORT123",
    "confidence": 0.99,
    "method": "historical"
})
var_99 = response.json()["var"]
```

### gRPC API (`neutryx.api.grpc`)

```python
import grpc
from neutryx.api.grpc import pricing_pb2, pricing_pb2_grpc

# Create gRPC channel
channel = grpc.insecure_channel('localhost:50051')
stub = pricing_pb2_grpc.PricingServiceStub(channel)

# Price option via gRPC
request = pricing_pb2.EuropeanOptionRequest(
    spot=100,
    strike=100,
    maturity=1.0,
    rate=0.05,
    vol=0.2,
    option_type="call"
)
response = stub.PriceEuropeanOption(request)
print(f"Price: {response.price}")
```

#### Exposure simulation inputs

The `/portfolio/xva` endpoint supports pathwise exposure simulation for the following
`ProductType` values. The market data payload must provide the listed keys to ensure
successful valuation.

| ProductType | Required market data |
| --- | --- |
| `EquityOption` | `equities.{underlying}.spot`, `equities.{underlying}.volatility`, optional `equities.{underlying}.dividend`, base currency rate under `rates.{currency}.rate` |
| `FxOption` | `fx.{pair}.spot`, `fx.{pair}.volatility`, domestic/foreign currencies (or inferrable from the 6-letter pair), domestic and foreign rates via `fx.{pair}` or `rates.{currency}.rate` |
| `InterestRateSwap` | `rates.{currency}.rate`, optional `rates.{currency}.volatility`, optional `rates.{currency}.discount_curve` |

Example market data payload:

```json
{
  "equities": {
    "AAPL": {"spot": 100.0, "volatility": 0.2, "dividend": 0.01}
  },
  "fx": {
    "EURUSD": {
      "spot": 1.10,
      "volatility": 0.15,
      "domestic_currency": "USD",
      "foreign_currency": "EUR"
    }
  },
  "rates": {
    "USD": {"rate": 0.03, "volatility": 0.01}
  }
}
```

Any missing spot, volatility or rate inputs trigger `400` validation errors to make the
required data explicit.

---

## Utilities and Helpers

### Date and Calendar Utilities

```python
from neutryx.utils.calendar import (
    BusinessDayCalendar,
    add_business_days,
    is_business_day
)

# Create calendar
calendar = BusinessDayCalendar(country="US")

# Add business days
settlement_date = add_business_days(trade_date, days=2, calendar=calendar)

# Check if business day
is_bday = is_business_day(date, calendar=calendar)
```

### Interpolation

```python
from neutryx.utils.interpolation import (
    LinearInterpolator,
    CubicSplineInterpolator,
    LogLinearInterpolator
)

# Linear interpolation
interp = LinearInterpolator(x=maturities, y=rates)
rate_at_2_5y = interp(2.5)

# Cubic spline (smooth curves)
spline = CubicSplineInterpolator(x=maturities, y=rates)
```

### Day Count Conventions

```python
from neutryx.utils.daycount import (
    DayCountConvention,
    year_fraction
)

# Common day count conventions
act_360 = DayCountConvention("ACT/360")
year_frac = year_fraction(start_date, end_date, convention=act_360)

# Other conventions: "30/360", "ACT/365", "ACT/ACT", "30E/360", etc.
```

---

## Configuration

### Config Management (`neutryx.config`)

```python
from neutryx.config import load_config, ConfigSchema

# Load configuration from YAML
config = load_config("config/production.yaml")

# Access config values
mc_config = config["pricing"]["monte_carlo"]
n_paths = mc_config["paths"]
```

---

## Testing

Neutryx Core includes 500+ comprehensive tests.

```bash
# Run all tests
pytest -v

# Run specific module tests
pytest src/neutryx/tests/products/ -v
pytest src/neutryx/tests/regulatory/ -v

# Run with coverage
pytest --cov=neutryx --cov-report=html

# Parallel execution
pytest -n auto  # Use all CPU cores
```

---

## Advanced Topics

### Custom Model Development

Create custom pricing models by extending base classes:

```python
from neutryx.models.base import StochasticModel
import jax.numpy as jnp

class MyCustomModel(StochasticModel):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def simulate_paths(self, key, S0, T, n_paths, n_steps):
        # Implement your simulation logic
        dt = T / n_steps
        # ... simulation code ...
        return paths

    def price(self, payoff_fn, S0, T):
        # Implement pricing
        paths = self.simulate_paths(...)
        payoff = payoff_fn(paths)
        return jnp.mean(payoff) * jnp.exp(-self.r * T)
```

### Custom Product Development

```python
from neutryx.products.base import Product

class CustomExotic(Product):
    def __init__(self, strike, barrier, maturity, **kwargs):
        super().__init__(**kwargs)
        self.strike = strike
        self.barrier = barrier
        self.maturity = maturity

    def payoff(self, paths):
        # Define custom payoff logic
        ST = paths[:, -1]
        # ... payoff calculation ...
        return payoff_values
```

---

## Migration Guides

### Upgrading from v0.1.x to v1.0.0

Key changes:
- SSO/OAuth authentication is now available
- Kubernetes orchestration support added
- FRTB IMA, DRC, RRAO modules added
- Backtesting and factor analysis modules added

For complete release history, refer to the git commit log or GitHub releases.

---

## Further Reading

- **[Getting Started](getting_started.md)** - Installation and quick start
- **[Tutorials](tutorials.md)** - Step-by-step guides
- **[Architecture Guide](architecture.md)** - Platform architecture
- **[Developer Guide](developer_guide.md)** - Contributing and development
- **[Performance Tuning](performance_tuning.md)** - Optimization tips

---

## Support

- **Documentation**: https://neutryx-lab.github.io/neutryx-core/
- **Issues**: https://github.com/neutryx-lab/neutryx-core/issues
- **Email**: dev@neutryx.tech

---

*Last updated: November 2025 | Neutryx Core v1.0.0+*
