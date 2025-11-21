"""Demonstration of advanced interest rate models in Neutryx.

This script showcases the implementation and usage of sophisticated interest rate
models including:
- Hull-White two-factor model
- Black-Karasinski (log-normal short rate)
- Cheyette model (extended HJM)
- Linear Gaussian Markov (LGM) models
- LIBOR Market Model (LMM/BGM)
- Heath-Jarrow-Morton (HJM) framework

Each example demonstrates:
1. Model setup and parameter specification
2. Path simulation
3. Derivative pricing
4. Key model features and applications
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from neutryx.models.hull_white_two_factor import (
    HullWhiteTwoFactorParams,
    zero_coupon_bond_price as hw2f_bond_price,
    simulate_paths as hw2f_simulate_paths,
    instantaneous_correlation,
)

from neutryx.models.black_karasinski import (
    BlackKarasinskiParams,
    simulate_paths as bk_simulate_paths,
    zero_coupon_bond_price_mc as bk_bond_price_mc,
)

from neutryx.models.cheyette import (
    CheyetteParams,
    zero_coupon_bond_price as chey_bond_price,
    simulate_paths as chey_simulate_paths,
)

from neutryx.models.lgm import (
    LGMParams,
    zero_coupon_bond_price as lgm_bond_price,
    simulate_paths as lgm_simulate_paths,
    caplet_price as lgm_caplet_price,
)

from neutryx.models.lmm import (
    LMMParams,
    simulate_paths as lmm_simulate_paths,
    zero_coupon_bond_price as lmm_bond_price,
    swap_rate,
    simple_volatility_structure,
    create_correlation_matrix,
)

from neutryx.models.hjm import (
    HJMParams,
    simulate_paths as hjm_simulate_paths,
    gaussian_hjm_volatility,
    exponential_hjm_volatility,
)


def demo_hull_white_two_factor():
    """Demonstrate Hull-White two-factor model.

    The two-factor model provides:
    - Richer dynamics than one-factor
    - Better fit to term structure of volatility
    - Realistic decorrelation between different maturities
    """
    print("\n" + "=" * 80)
    print("HULL-WHITE TWO-FACTOR MODEL")
    print("=" * 80)

    # Set up parameters
    params = HullWhiteTwoFactorParams(
        a=0.1,           # Mean reversion speed for short rate
        b=0.3,           # Mean reversion speed for second factor
        sigma1=0.01,     # Volatility of short rate factor
        sigma2=0.015,    # Volatility of second factor
        rho=0.3,         # Correlation between factors
        r0=0.03,         # Initial short rate (3%)
        u0=0.0,          # Initial second factor
    )

    print(f"\nModel Parameters:")
    print(f"  Mean reversion speeds: a = {params.a}, b = {params.b}")
    print(f"  Volatilities: σ₁ = {params.sigma1}, σ₂ = {params.sigma2}")
    print(f"  Correlation: ρ = {params.rho}")
    print(f"  Initial rate: r₀ = {params.r0}")

    # 1. Zero-coupon bond pricing
    print(f"\n1. Zero-Coupon Bond Prices:")
    maturities = jnp.array([1.0, 2.0, 5.0, 10.0, 30.0])
    for T in maturities:
        P = hw2f_bond_price(params, T, r_t=params.r0, u_t=params.u0)
        ytm = -jnp.log(P) / T
        print(f"  P(0, {T:4.0f}Y) = {P:.6f}  =>  Yield = {ytm*100:.2f}%")

    # 2. Simulate short rate paths
    print(f"\n2. Simulating Short Rate Paths:")
    key = jax.random.PRNGKey(42)
    T = 10.0
    n_steps = 200
    n_paths = 5

    r_paths, u_paths = hw2f_simulate_paths(params, T, n_steps, n_paths, key)
    print(f"  Simulated {n_paths} paths over {T} years")
    print(f"  Mean terminal rate: {jnp.mean(r_paths[:, -1])*100:.2f}%")
    print(f"  Std terminal rate: {jnp.std(r_paths[:, -1])*100:.2f}%")

    # 3. Instantaneous correlations
    print(f"\n3. Instantaneous Forward Rate Correlations:")
    print(f"  (Two-factor model provides realistic decorrelation)")
    T1 = 1.0
    for T2 in [1.0, 2.0, 5.0, 10.0]:
        corr = instantaneous_correlation(params, T1, T2)
        print(f"  Corr(f(0,{T1}Y), f(0,{T2}Y)) = {corr:.4f}")

    print(f"\n✓ Two-factor model allows for term structure of correlations")
    print(f"✓ Useful for pricing correlation-sensitive products like CMS spread options")


def demo_black_karasinski():
    """Demonstrate Black-Karasinski log-normal model.

    Key features:
    - Rates are always positive (log-normal specification)
    - Volatility proportional to rate level
    - Commonly used in practice for pricing caps/floors
    """
    print("\n" + "=" * 80)
    print("BLACK-KARASINSKI MODEL (Log-Normal Short Rate)")
    print("=" * 80)

    # Set up parameters
    params = BlackKarasinskiParams(
        a=0.2,          # Mean reversion speed
        sigma=0.15,     # Volatility of log(r)
        r0=0.03,        # Initial short rate (3%)
    )

    print(f"\nModel Parameters:")
    print(f"  Mean reversion: a = {params.a}")
    print(f"  Log-volatility: σ = {params.sigma}")
    print(f"  Initial rate: r₀ = {params.r0}")

    # 1. Simulate rate paths
    print(f"\n1. Simulating Short Rate Paths:")
    key = jax.random.PRNGKey(123)
    T = 5.0
    n_steps = 100
    n_paths = 1000

    r_paths = bk_simulate_paths(params, T, n_steps, n_paths, key)
    print(f"  Simulated {n_paths} paths over {T} years")
    print(f"  Mean terminal rate: {jnp.mean(r_paths[:, -1])*100:.2f}%")
    print(f"  Std terminal rate: {jnp.std(r_paths[:, -1])*100:.2f}%")
    print(f"  Min terminal rate: {jnp.min(r_paths[:, -1])*100:.2f}% (always positive!)")

    # 2. Bond pricing via Monte Carlo
    print(f"\n2. Zero-Coupon Bond Pricing (Monte Carlo):")
    for T_mat in [1.0, 3.0, 5.0]:
        P = bk_bond_price_mc(params, T=T_mat, n_paths=5000, n_steps=50, key=key)
        ytm = -jnp.log(P) / T_mat
        print(f"  P(0, {T_mat}Y) = {P:.6f}  =>  Yield = {ytm*100:.2f}%")

    print(f"\n✓ Log-normal rates ensure positivity - important for low rate environments")
    print(f"✓ Requires numerical methods (trees or Monte Carlo) for pricing")
    print(f"✓ Popular for pricing caps, floors, and callable bonds")


def demo_cheyette():
    """Demonstrate Cheyette model.

    Extended HJM framework with:
    - Markovian representation (finite state)
    - Flexible volatility term structure
    - Analytical bond option formulas
    """
    print("\n" + "=" * 80)
    print("CHEYETTE MODEL (Extended HJM)")
    print("=" * 80)

    # Single-factor Cheyette
    params = CheyetteParams(
        kappa=0.15,                          # Mean reversion speed
        sigma_fn=lambda t: 0.01 * jnp.exp(-0.1 * t),  # Time-decaying volatility
        forward_curve_fn=lambda t: 0.03 + 0.001 * t,  # Upward sloping curve
        r0=0.03,
        n_factors=1,
    )

    print(f"\nModel Parameters:")
    print(f"  Mean reversion: κ = {params.kappa}")
    print(f"  Volatility: σ(t) = 0.01 * exp(-0.1t) (decaying)")
    print(f"  Forward curve: f(0,t) = 3% + 0.1% * t (upward sloping)")

    # 1. Bond pricing
    print(f"\n1. Zero-Coupon Bond Prices:")
    maturities = jnp.array([1.0, 2.0, 5.0, 10.0])
    for T in maturities:
        P = chey_bond_price(params, T, x_t=0.0, y_t=0.0)
        ytm = -jnp.log(P) / T
        print(f"  P(0, {T:4.0f}Y) = {P:.6f}  =>  Yield = {ytm*100:.2f}%")

    # 2. Simulate paths
    print(f"\n2. Simulating State Variables:")
    key = jax.random.PRNGKey(42)
    r_paths, x_paths, y_paths = chey_simulate_paths(
        params, T=5.0, n_steps=100, n_paths=5, key=key
    )
    print(f"  Simulated 5 paths of (r, x, y) over 5 years")
    print(f"  Mean terminal rate: {jnp.mean(r_paths[:, -1])*100:.2f}%")
    print(f"  Mean terminal variance: {jnp.mean(y_paths[:, -1]):.6f}")

    print(f"\n✓ Markovian representation makes it efficient for simulation")
    print(f"✓ Supports analytical swaption pricing formulas")
    print(f"✓ Widely used for exotic interest rate derivatives")


def demo_lgm():
    """Demonstrate Linear Gaussian Markov (LGM) model.

    Modern formulation of Hull-White with:
    - Explicit calibration to market observables
    - Time-dependent mean reversion and volatility
    - Efficient implementation for exotic pricing
    """
    print("\n" + "=" * 80)
    print("LINEAR GAUSSIAN MARKOV (LGM) MODEL")
    print("=" * 80)

    # Set up LGM parameters
    params = LGMParams(
        alpha_fn=lambda t: 0.1 + 0.05 * t,          # Increasing mean reversion
        sigma_fn=lambda t: 0.01 * (1.0 + 0.5 * t),  # Increasing volatility
        forward_curve_fn=lambda t: 0.03 * (1.0 + 0.01 * t),  # Upward sloping
        r0=0.03,
        n_factors=1,
    )

    print(f"\nModel Parameters:")
    print(f"  Mean reversion: α(t) = 0.1 + 0.05t (time-dependent)")
    print(f"  Volatility: σ(t) = 0.01 * (1 + 0.5t) (time-dependent)")
    print(f"  Forward curve fitted to market")

    # 1. Bond pricing
    print(f"\n1. Zero-Coupon Bond Prices:")
    maturities = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0])
    for T in maturities:
        P = lgm_bond_price(params, T, x_t=0.0)
        ytm = -jnp.log(P) / T
        print(f"  P(0, {T:5.1f}Y) = {P:.6f}  =>  Yield = {ytm*100:.2f}%")

    # 2. Caplet pricing
    print(f"\n2. Caplet Prices (Analytical):")
    strikes = jnp.array([0.025, 0.030, 0.035, 0.040])
    caplet_mat = 1.0
    tenor = 0.25

    for K in strikes:
        caplet_val = lgm_caplet_price(params, strike=K, caplet_maturity=caplet_mat, tenor=tenor)
        print(f"  Caplet(K={K*100:.1f}%, T={caplet_mat}Y, τ={tenor}Y) = {caplet_val:.6f}")

    # 3. Simulate paths
    print(f"\n3. Simulating Short Rate Paths:")
    key = jax.random.PRNGKey(42)
    r_paths, x_paths = lgm_simulate_paths(params, T=5.0, n_steps=100, n_paths=1000, key=key)
    print(f"  Simulated 1000 paths over 5 years")
    print(f"  Mean terminal rate: {jnp.mean(r_paths[:, -1])*100:.2f}%")
    print(f"  Std terminal rate: {jnp.std(r_paths[:, -1])*100:.2f}%")

    print(f"\n✓ LGM is essentially Hull-White with better parameterization")
    print(f"✓ Easier calibration to market swaption volatilities")
    print(f"✓ Industry standard for Bermudan swaptions and exotic structures")


def demo_lmm():
    """Demonstrate LIBOR Market Model (LMM/BGM).

    Forward rate model that:
    - Directly models market-observable LIBOR rates
    - No-arbitrage evolution under different measures
    - Industry standard for complex derivatives
    """
    print("\n" + "=" * 80)
    print("LIBOR MARKET MODEL (LMM/BGM)")
    print("=" * 80)

    # Set up LIBOR tenor structure
    n_rates = 8
    tenor_structure = jnp.linspace(0, 2.0, n_rates + 1)  # 0, 0.25, 0.5, ..., 2.0
    forward_rates = jnp.array([0.03, 0.032, 0.034, 0.036, 0.038, 0.039, 0.040, 0.041])

    # Volatility structure (declining with maturity)
    vol_fn = simple_volatility_structure(initial_vol=0.20, decay_rate=0.1, n_rates=n_rates)

    # Correlation structure
    corr_matrix = create_correlation_matrix(n_rates, beta=0.15, rho_infty=0.5)

    params = LMMParams(
        forward_rates=forward_rates,
        tenor_structure=tenor_structure,
        volatility_fn=vol_fn,
        correlation_matrix=corr_matrix,
        terminal_measure=True,
    )

    print(f"\nModel Setup:")
    print(f"  Number of forward rates: {n_rates}")
    print(f"  Tenor structure: quarterly (3M LIBOR)")
    print(f"  Initial forward rates: {forward_rates[0]*100:.2f}% to {forward_rates[-1]*100:.2f}%")
    print(f"  Volatility: 20% with 10% decay")
    print(f"  Correlation: exponential with β=0.15")

    # 1. Display term structure
    print(f"\n1. Initial Term Structure:")
    for i in range(n_rates):
        T_start = tenor_structure[i]
        T_end = tenor_structure[i + 1]
        L_i = forward_rates[i]
        print(f"  L({T_start:.2f}, {T_end:.2f}) = {L_i*100:.2f}%")

    # 2. Simulate forward rate paths
    print(f"\n2. Simulating Forward LIBOR Rates:")
    key = jax.random.PRNGKey(42)
    L_paths = lmm_simulate_paths(params, T=1.0, n_steps=50, n_paths=1000, key=key)
    print(f"  Simulated 1000 paths over 1 year")
    print(f"  Shape: {L_paths.shape} (paths × time_steps × rates)")

    # Terminal rates
    print(f"\n  Terminal Forward Rates (mean across paths):")
    for i in range(n_rates):
        mean_L = jnp.mean(L_paths[:, -1, i])
        std_L = jnp.std(L_paths[:, -1, i])
        print(f"    L_{i}: {mean_L*100:.2f}% ± {std_L*100:.2f}%")

    # 3. Swap rate calculation
    print(f"\n3. Forward Swap Rates:")
    for i in range(0, n_rates, 2):
        T_start = tenor_structure[i]
        T_end = tenor_structure[min(i + 4, n_rates)]
        S = swap_rate(params, forward_rates, T_start, T_end)
        print(f"  Swap({T_start:.2f}Y, {T_end:.2f}Y) = {S*100:.3f}%")

    # 4. Bond pricing
    print(f"\n4. Zero-Coupon Bond Prices from LIBOR rates:")
    for i in range(0, n_rates, 2):
        T_end = tenor_structure[i + 1]
        P = lmm_bond_price(params, forward_rates, T_start=0.0, T_end=T_end)
        print(f"  P(0, {T_end:.2f}Y) = {P:.6f}")

    print(f"\n✓ LMM directly models market-traded LIBOR rates")
    print(f"✓ Ensures positive rates (log-normal dynamics)")
    print(f"✓ Industry standard for Bermudan swaptions, range accruals, TARNs")
    print(f"✓ Can be calibrated to full cap/swaption surface")


def demo_hjm():
    """Demonstrate Heath-Jarrow-Morton (HJM) framework.

    Most general continuous-time framework:
    - Models entire forward rate curve
    - No-arbitrage by construction
    - Many models are special cases of HJM
    """
    print("\n" + "=" * 80)
    print("HEATH-JARROW-MORTON (HJM) FRAMEWORK")
    print("=" * 80)

    # Single-factor HJM with Gaussian volatility (reduces to Hull-White)
    forward_curve = lambda T: 0.03 + 0.002 * T - 0.0001 * T**2
    vol_fn = gaussian_hjm_volatility(sigma=0.01, kappa=0.15)

    params = HJMParams(
        forward_curve_fn=forward_curve,
        volatility_fns=[vol_fn],
        r0=0.03,
        n_factors=1,
        max_maturity=20.0,
        n_maturities=40,
    )

    print(f"\nModel Setup:")
    print(f"  Forward curve: f(0,T) = 3% + 0.2%*T - 0.01%*T²")
    print(f"  Volatility: σ(t,T) = 1% * exp(-0.15*(T-t))  [Gaussian/Hull-White]")
    print(f"  Number of factors: 1")
    print(f"  Maturity discretization: {params.n_maturities} points up to {params.max_maturity}Y")

    # 1. Display initial forward curve
    print(f"\n1. Initial Forward Rate Curve:")
    sample_maturities = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    for T in sample_maturities:
        f = forward_curve(T)
        print(f"  f(0, {T:4.0f}Y) = {f*100:.3f}%")

    # 2. Simulate short rate
    print(f"\n2. Simulating Short Rate Path:")
    key = jax.random.PRNGKey(42)
    r_paths = hjm_simulate_paths(
        params, T=10.0, n_steps=200, n_paths=5, key=key, output_type="short_rate"
    )
    print(f"  Simulated 5 short rate paths over 10 years")
    print(f"  Shape: {r_paths.shape}")
    print(f"  Mean terminal rate: {jnp.mean(r_paths[:, -1])*100:.2f}%")

    # 3. Simulate full forward curve evolution
    print(f"\n3. Simulating Forward Curve Evolution:")
    curve_paths = hjm_simulate_paths(
        params, T=5.0, n_steps=100, n_paths=2, key=key, output_type="forward_curve"
    )
    print(f"  Simulated 2 forward curve paths over 5 years")
    print(f"  Shape: {curve_paths.shape} (paths × time_steps × maturities)")
    print(f"  This captures evolution of entire yield curve!")

    # 4. Multi-factor example
    print(f"\n4. Two-Factor HJM:")
    vol_fn1 = gaussian_hjm_volatility(sigma=0.01, kappa=0.15)
    vol_fn2 = gaussian_hjm_volatility(sigma=0.008, kappa=0.30)
    rho = jnp.array([[1.0, 0.5], [0.5, 1.0]])

    params_2f = HJMParams(
        forward_curve_fn=forward_curve,
        volatility_fns=[vol_fn1, vol_fn2],
        r0=0.03,
        n_factors=2,
        rho=rho,
        max_maturity=20.0,
        n_maturities=40,
    )
    print(f"  Two correlated factors (ρ=0.5)")
    print(f"  σ₁(t,T) = 1.0% * exp(-0.15*(T-t))")
    print(f"  σ₂(t,T) = 0.8% * exp(-0.30*(T-t))")

    print(f"\n✓ HJM is the most general interest rate framework")
    print(f"✓ No-arbitrage condition ensures consistent dynamics")
    print(f"✓ Vasicek, Hull-White, LMM are all special cases of HJM")
    print(f"✓ Provides theoretical foundation for term structure modeling")


def comparison_summary():
    """Provide a comparison summary of all models."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)

    comparison = """
╔══════════════════╦═══════════╦════════════╦═══════════╦═════════════════════════╗
║ Model            ║ Factors   ║ Rates      ║ Pricing   ║ Best Use Cases          ║
╠══════════════════╬═══════════╬════════════╬═══════════╬═════════════════════════╣
║ Hull-White 2F    ║ 2         ║ Can be -ve ║ Analytical║ Complex swaptions,      ║
║                  ║           ║ (Gaussian) ║ /Tree     ║ correlation products    ║
╠══════════════════╬═══════════╬════════════╬═══════════╬═════════════════════════╣
║ Black-Karasinski ║ 1         ║ Always +ve ║ Tree/MC   ║ Caps/floors, callable   ║
║                  ║           ║ (Log-norm) ║           ║ bonds, low rate envs    ║
╠══════════════════╬═══════════╬════════════╬═══════════╬═════════════════════════╣
║ Cheyette         ║ 1+ (multi)║ Can be -ve ║ Analytical║ Exotic swaptions, term  ║
║                  ║           ║ (Gaussian) ║ /MC       ║ structure products      ║
╠══════════════════╬═══════════╬════════════╬═══════════╬═════════════════════════╣
║ LGM              ║ 1+ (multi)║ Can be -ve ║ Analytical║ Bermudan swaptions,     ║
║                  ║           ║ (Gaussian) ║ /MC       ║ calibration to market   ║
╠══════════════════╬═══════════╬════════════╬═══════════╬═════════════════════════╣
║ LMM (BGM)        ║ N (many)  ║ Always +ve ║ MC        ║ Range accruals, TARNs,  ║
║                  ║           ║ (Log-norm) ║           ║ Bermudan swaptions      ║
╠══════════════════╬═══════════╬════════════╬═══════════╬═════════════════════════╣
║ HJM              ║ Any       ║ Depends on ║ MC/PDE    ║ Theoretical analysis,   ║
║                  ║           ║ volatility ║           ║ model development       ║
╚══════════════════╩═══════════╩════════════╩═══════════╩═════════════════════════╝

Key Considerations:

1. **Gaussian vs Log-Normal:**
   - Gaussian (HW, Cheyette, LGM): Can have negative rates, analytical formulas
   - Log-Normal (BK, LMM): Always positive rates, requires numerical methods

2. **Calibration:**
   - LGM/LMM: Designed for calibration to market cap/swaption volatilities
   - HW/Cheyette: Can fit to term structure but harder for full surface

3. **Computational Efficiency:**
   - HW 2F, Cheyette, LGM: Analytical bond options => fast for trees
   - LMM: Many factors => requires Monte Carlo (expensive but accurate)
   - HJM: Most general but also most computationally intensive

4. **Industry Usage:**
   - **Vanilla swaps/bonds:** Simple Hull-White often sufficient
   - **Bermudan swaptions:** LGM or LMM (depending on complexity)
   - **Exotic structures (TARNs, range accruals):** LMM
   - **Low rate environment:** Black-Karasinski (ensures positivity)
   - **Correlation products:** Hull-White 2F or multi-factor Cheyette

5. **Relationships:**
   - Hull-White is special case of HJM with specific volatility structure
   - LGM is reformulation of Hull-White with better parameterization
   - Cheyette is Markovian reduction of HJM
   - LMM is discrete-tenor version of HJM
"""

    print(comparison)


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print(" " * 20 + "ADVANCED INTEREST RATE MODELS")
    print(" " * 25 + "Neutryx Demonstration")
    print("=" * 80)

    # Run each demonstration
    demo_hull_white_two_factor()
    demo_black_karasinski()
    demo_cheyette()
    demo_lgm()
    demo_lmm()
    demo_hjm()

    # Show comparison
    comparison_summary()

    print("\n" + "=" * 80)
    print("All demonstrations completed successfully!")
    print("\nThese models provide a comprehensive toolkit for:")
    print("  • Pricing complex interest rate derivatives")
    print("  • Risk management and hedging")
    print("  • Model calibration to market data")
    print("  • Regulatory capital calculations")
    print("  • Research and model development")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
