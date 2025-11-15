"""Comprehensive demonstration of IR volatility products.

This demo showcases:
1. Swaption volatility cubes (3D: expiry x tenor x strike)
2. Caplet/floorlet volatility surfaces (2D: expiry x strike)
3. IR volatility trading products
   - Swaption straddles/strangles
   - IR variance swaps
   - Correlation swaps
   - Dispersion swaps
"""

import jax.numpy as jnp
import jax.random as jrand

from neutryx.market.ir_vol_surface import (
    CapletVolSurface,
    SwaptionVolCube,
    construct_caplet_surface_from_sabr,
    construct_swaption_cube_from_sabr,
)
from neutryx.products.ir_volatility import (
    SwaptionStraddle,
    SwaptionStrangle,
    IRVarianceSwap,
    ForwardIRVarianceSwap,
    RateCorrelationSwap,
    VolatilityDispersionSwap,
    compute_forward_variance_strike,
)


def demo_caplet_vol_surface():
    """Demonstrate caplet volatility surface construction and usage."""
    print("\n" + "=" * 80)
    print("1. CAPLET/FLOORLET VOLATILITY SURFACE")
    print("=" * 80)

    # Market expiries and forward rates
    expiries = jnp.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    forward_rates = jnp.array([0.030, 0.032, 0.035, 0.038, 0.040, 0.042, 0.043, 0.044])

    # SABR parameters with declining alpha term structure
    alphas = jnp.array([0.28, 0.26, 0.23, 0.20, 0.18, 0.16, 0.15, 0.14])
    rhos = jnp.full_like(alphas, -0.30)
    nus = jnp.full_like(alphas, 0.40)

    print("\nConstructing caplet vol surface with SABR parameterization...")
    print(f"  Expiries: {[f'{t:.2f}Y' for t in expiries]}")
    print(f"  Forward rates: {[f'{r:.2%}' for r in forward_rates]}")
    print(f"  Alpha (ATM vol): {[f'{a:.2%}' for a in alphas]}")
    print(f"  Beta: 0.50 (constant)")
    print(f"  Rho: -0.30 (constant)")
    print(f"  Nu: 0.40 (constant)")

    surface = construct_caplet_surface_from_sabr(
        expiries=expiries,
        forward_rates=forward_rates,
        alpha=alphas,
        beta=0.5,
        rho=rhos,
        nu=nus,
    )

    # Query volatilities
    print("\n  Querying volatilities from surface:")
    print("\n  Expiry   Strike   IV")
    print("  " + "-" * 30)

    test_points = [
        (1.0, 0.025),  # OTM put
        (1.0, 0.035),  # ATM
        (1.0, 0.045),  # OTM call
        (2.0, 0.030),  # OTM put
        (2.0, 0.038),  # ATM
        (2.0, 0.050),  # OTM call
    ]

    for expiry, strike in test_points:
        vol = surface.get_vol(strike, expiry)
        print(f"  {expiry:.1f}Y    {strike:.3f}    {vol:.2%}")

    # Get ATM vol term structure
    print("\n  ATM Volatility Term Structure:")
    print("  Expiry   ATM Vol")
    print("  " + "-" * 25)
    for expiry in [0.5, 1.0, 2.0, 5.0, 10.0]:
        atm_vol = surface.get_atm_vol(expiry)
        print(f"  {expiry:.1f}Y    {atm_vol:.2%}")

    # Get vol smile at 1Y
    print("\n  1Y Volatility Smile:")
    strikes_1y = jnp.linspace(0.02, 0.06, 9)
    vols_1y = surface.get_vol_slice(1.0, strikes_1y)

    print("  Strike   IV")
    print("  " + "-" * 20)
    for strike, vol in zip(strikes_1y, vols_1y):
        print(f"  {strike:.3f}    {vol:.2%}")

    return surface


def demo_swaption_vol_cube():
    """Demonstrate swaption volatility cube construction and usage."""
    print("\n" + "=" * 80)
    print("2. SWAPTION VOLATILITY CUBE")
    print("=" * 80)

    # Market standard points
    option_expiries = jnp.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
    swap_tenors = jnp.array([1.0, 2.0, 5.0, 10.0, 30.0])

    print(f"\nCube dimensions:")
    print(f"  Option expiries: {[f'{t:.2f}Y' for t in option_expiries]}")
    print(f"  Swap tenors: {[f'{t:.0f}Y' for t in swap_tenors]}")

    # Forward swap rates matrix [expiries x tenors]
    # Upward sloping curve
    forward_swap_rates = jnp.array([
        [0.030, 0.034, 0.039, 0.041, 0.042],  # 3M
        [0.032, 0.035, 0.040, 0.042, 0.043],  # 6M
        [0.034, 0.037, 0.041, 0.043, 0.044],  # 1Y
        [0.036, 0.039, 0.042, 0.044, 0.045],  # 2Y
        [0.038, 0.041, 0.044, 0.045, 0.046],  # 5Y
        [0.040, 0.043, 0.045, 0.046, 0.047],  # 10Y
    ])

    # SABR parameters [expiries x tenors x 4]
    # Declining alpha with expiry, slight hump in tenor
    sabr_params = jnp.zeros((len(option_expiries), len(swap_tenors), 4))

    for i, exp in enumerate(option_expiries):
        for j, ten in enumerate(swap_tenors):
            # Alpha: declines with expiry, humps at 5Y tenor
            alpha_base = 0.25 - 0.01 * i
            alpha_tenor_adj = 0.02 if ten == 5.0 else 0.0
            alpha = alpha_base + alpha_tenor_adj

            # Rho: more negative for longer tenors
            rho = -0.25 - 0.01 * j

            # Nu: increases with expiry
            nu = 0.35 + 0.01 * i

            sabr_params = sabr_params.at[i, j].set(
                jnp.array([alpha, 0.5, rho, nu])
            )

    print("\nConstructing swaption vol cube...")
    cube = construct_swaption_cube_from_sabr(
        option_expiries=option_expiries,
        swap_tenors=swap_tenors,
        forward_swap_rates=forward_swap_rates,
        sabr_params=sabr_params,
    )

    # ATM volatility matrix
    print("\n  ATM Swaption Volatility Matrix:")
    print("  " + " " * 12 + "Swap Tenor")
    print("  Expiry    " + "    ".join(f"{t:.0f}Y" for t in swap_tenors))
    print("  " + "-" * 60)

    atm_matrix = cube.get_atm_matrix()
    for i, exp in enumerate(option_expiries):
        row = f"  {exp:.2f}Y    "
        row += "   ".join(f"{atm_matrix[i, j]:.2%}" for j in range(len(swap_tenors)))
        print(row)

    # Query specific points
    print("\n  Sample Quotes:")
    print("  Expiry   Tenor   Strike   IV")
    print("  " + "-" * 40)

    test_quotes = [
        (1.0, 5.0, 0.041),  # ATM 1Yx5Y
        (1.0, 5.0, 0.035),  # ITM receiver 1Yx5Y
        (1.0, 5.0, 0.047),  # ITM payer 1Yx5Y
        (2.0, 10.0, 0.044),  # ATM 2Yx10Y
        (5.0, 10.0, 0.045),  # ATM 5Yx10Y
    ]

    for expiry, tenor, strike in test_quotes:
        vol = cube.get_vol(expiry, tenor, strike)
        opt_type = "ATM" if abs(strike - forward_swap_rates[
            list(option_expiries).index(expiry),
            list(swap_tenors).index(tenor)
        ]) < 0.001 else "OTM"
        print(f"  {expiry:.1f}Y    {tenor:.0f}Y     {strike:.3f}    {vol:.2%}  ({opt_type})")

    # Volatility smile for 1Yx5Y
    print("\n  1Yx5Y Swaption Volatility Smile:")
    strikes_smile = jnp.linspace(0.03, 0.05, 11)
    vols_smile = cube.get_vol_smile(1.0, 5.0, strikes_smile)

    print("  Strike   IV      Spread vs ATM")
    print("  " + "-" * 35)
    atm_vol_1y5y = cube.get_atm_vol(1.0, 5.0)

    for strike, vol in zip(strikes_smile, vols_smile):
        spread = (vol - atm_vol_1y5y) * 10000  # bps
        print(f"  {strike:.3f}    {vol:.2%}    {spread:+.0f} bps")

    return cube


def demo_swaption_straddle_strangle():
    """Demonstrate swaption volatility trading strategies."""
    print("\n" + "=" * 80)
    print("3. SWAPTION VOLATILITY TRADING")
    print("=" * 80)

    # Swaption straddle (ATM payer + receiver)
    print("\nSwaption Straddle (Long Volatility):")
    straddle = SwaptionStraddle(
        T=1.0,
        K=0.04,  # ATM strike
        annuity=4.5,
        notional=10_000_000,
    )

    print(f"  Structure: ATM Payer + ATM Receiver")
    print(f"  Strike: {straddle.K:.2%}")
    print(f"  Notional: ${straddle.notional:,.0f}")
    print(f"  Annuity: {straddle.annuity:.2f}")

    print("\n  Payoff Profile:")
    print("  Swap Rate   Payoff")
    print("  " + "-" * 30)

    rates = jnp.array([0.03, 0.035, 0.04, 0.045, 0.05])
    for rate in rates:
        payoff = straddle.payoff_terminal(rate)
        print(f"  {rate:.2%}        ${payoff:,.0f}")

    # Swaption strangle
    print("\n\nSwaption Strangle (Cheaper Volatility Play):")
    strangle = SwaptionStrangle(
        T=1.0,
        K_low=0.035,  # OTM receiver
        K_high=0.045,  # OTM payer
        annuity=4.5,
        notional=10_000_000,
    )

    print(f"  Structure: OTM Receiver + OTM Payer")
    print(f"  Lower strike: {strangle.K_low:.2%}")
    print(f"  Upper strike: {strangle.K_high:.2%}")
    print(f"  Dead zone: [{strangle.K_low:.2%}, {strangle.K_high:.2%}]")

    print("\n  Payoff Profile:")
    print("  Swap Rate   Payoff")
    print("  " + "-" * 30)

    for rate in rates:
        payoff = strangle.payoff_terminal(rate)
        print(f"  {rate:.2%}        ${payoff:,.0f}")

    return straddle, strangle


def demo_ir_variance_swap():
    """Demonstrate IR variance swap."""
    print("\n" + "=" * 80)
    print("4. INTEREST RATE VARIANCE SWAP")
    print("=" * 80)

    # Spot variance swap
    var_swap = IRVarianceSwap(
        T=1.0,
        strike_variance=0.0400,  # 20% vol squared
        notional_per_variance_point=100_000,
        observation_frequency=252,
    )

    strike_vol = jnp.sqrt(var_swap.strike_variance)

    print(f"\nVariance Swap on 10Y Swap Rate:")
    print(f"  Maturity: {var_swap.T} years")
    print(f"  Strike variance: {var_swap.strike_variance:.4f}")
    print(f"  Strike vol: {strike_vol:.2%}")
    print(f"  Vega notional: ${var_swap.notional_per_variance_point:,.0f} per variance point")
    print(f"  Observation frequency: {var_swap.observation_frequency} (daily)")

    # Simulate rate path
    print("\n  Simulating rate path...")
    key = jrand.PRNGKey(42)
    n_steps = 252
    rate_vol = 0.22  # Realized vol higher than strike

    # GBM simulation
    dt = 1.0 / n_steps
    returns = rate_vol * jnp.sqrt(dt) * jrand.normal(key, (n_steps,))
    path = 0.04 * jnp.exp(jnp.cumsum(returns))

    print(f"  Initial rate: {path[0]:.4f}")
    print(f"  Final rate: {path[-1]:.4f}")
    print(f"  Min rate: {jnp.min(path):.4f}")
    print(f"  Max rate: {jnp.max(path):.4f}")

    # Calculate payoff
    payoff = var_swap.payoff_path(path)

    # Calculate realized variance
    log_returns = jnp.log(path[1:] / path[:-1])
    realized_var = (252 / len(log_returns)) * jnp.sum(log_returns**2)
    realized_vol = jnp.sqrt(realized_var)

    print(f"\n  Realized variance: {realized_var:.4f}")
    print(f"  Realized vol: {realized_vol:.2%}")
    print(f"  Variance P&L: ${payoff:,.0f}")

    if payoff > 0:
        print(f"  \u2713 Long variance position profitable (realized vol > strike)")
    else:
        print(f"  \u2717 Long variance position unprofitable (realized vol < strike)")

    return var_swap


def demo_forward_variance_swap():
    """Demonstrate forward variance swap."""
    print("\n" + "=" * 80)
    print("5. FORWARD VARIANCE SWAP")
    print("=" * 80)

    # Compute forward variance from spot vols
    spot_vol_1y = 0.20  # 20%
    spot_vol_2y = 0.22  # 22%

    fwd_var = compute_forward_variance_strike(
        spot_variance_t1=spot_vol_1y**2,
        spot_variance_t2=spot_vol_2y**2,
        t1=1.0,
        t2=2.0,
    )

    fwd_vol = jnp.sqrt(fwd_var)

    print(f"\nForward Variance Calculation:")
    print(f"  1Y spot vol: {spot_vol_1y:.2%}")
    print(f"  2Y spot vol: {spot_vol_2y:.2%}")
    print(f"  1Y-2Y forward variance: {fwd_var:.4f}")
    print(f"  1Y-2Y forward vol: {fwd_vol:.2%}")

    # Forward variance swap
    fwd_var_swap = ForwardIRVarianceSwap(
        T=2.0,
        T1=1.0,
        strike_variance=fwd_var,
        notional_per_variance_point=100_000,
    )

    print(f"\nForward Variance Swap:")
    print(f"  Start date: {fwd_var_swap.T1} years")
    print(f"  End date: {fwd_var_swap.T} years")
    print(f"  Strike variance: {fwd_var_swap.strike_variance:.4f}")
    print(f"  Strike vol: {jnp.sqrt(fwd_var_swap.strike_variance):.2%}")
    print(f"  Vega notional: ${fwd_var_swap.notional_per_variance_point:,.0f}")

    # Scenario analysis
    print("\n  Scenario Analysis:")
    print("  Realized Fwd Var   Realized Fwd Vol   P&L")
    print("  " + "-" * 50)

    scenarios = [0.0360, 0.0400, 0.0441, 0.0484, 0.0529]
    for scenario_var in scenarios:
        scenario_vol = jnp.sqrt(scenario_var)
        payoff = fwd_var_swap.payoff_terminal(scenario_var)
        print(f"  {scenario_var:.4f}           {scenario_vol:.2%}            ${payoff:,.0f}")

    return fwd_var_swap


def demo_rate_correlation_swap():
    """Demonstrate rate correlation swap."""
    print("\n" + "=" * 80)
    print("6. RATE CORRELATION SWAP")
    print("=" * 80)

    corr_swap = RateCorrelationSwap(
        T=1.0,
        strike_correlation=0.70,
        notional_per_correlation_point=1_000_000,
    )

    print(f"\nCorrelation Swap (2Y vs 10Y Swap Rates):")
    print(f"  Maturity: {corr_swap.T} years")
    print(f"  Strike correlation: {corr_swap.strike_correlation:.2%}")
    print(f"  Notional per 1% correlation: ${corr_swap.notional_per_correlation_point:,.0f}")

    # Simulate correlated rate paths
    print("\n  Simulating correlated rate paths...")
    key = jrand.PRNGKey(123)
    n_steps = 252
    target_corr = 0.85  # Higher than strike

    # Generate correlated Brownian motions
    key1, key2 = jrand.split(key)
    z1 = jrand.normal(key1, (n_steps,))
    z2_indep = jrand.normal(key2, (n_steps,))
    z2 = target_corr * z1 + jnp.sqrt(1 - target_corr**2) * z2_indep

    # Rate paths
    dt = 1.0 / n_steps
    path1 = 0.02 * jnp.exp(jnp.cumsum(0.15 * jnp.sqrt(dt) * z1))  # 2Y
    path2 = 0.04 * jnp.exp(jnp.cumsum(0.18 * jnp.sqrt(dt) * z2))  # 10Y

    paths = jnp.stack([path1, path2])

    print(f"  2Y rate: {path1[0]:.2%} -> {path1[-1]:.2%}")
    print(f"  10Y rate: {path2[0]:.2%} -> {path2[-1]:.2%}")

    # Calculate payoff
    payoff = corr_swap.payoff_path(paths)

    # Calculate realized correlation
    returns1 = jnp.log(path1[1:] / path1[:-1])
    returns2 = jnp.log(path2[1:] / path2[:-1])
    realized_corr = jnp.corrcoef(returns1, returns2)[0, 1]

    print(f"\n  Realized correlation: {realized_corr:.2%}")
    print(f"  Strike correlation: {corr_swap.strike_correlation:.2%}")
    print(f"  Correlation difference: {(realized_corr - corr_swap.strike_correlation):.2%}")
    print(f"  P&L: ${payoff:,.0f}")

    return corr_swap


def demo_volatility_dispersion_swap():
    """Demonstrate volatility dispersion swap."""
    print("\n" + "=" * 80)
    print("7. VOLATILITY DISPERSION SWAP")
    print("=" * 80)

    disp_swap = VolatilityDispersionSwap(
        T=1.0,
        strike_dispersion=0.05,
        notional_per_dispersion_point=100_000,
        n_rates=4,
    )

    print(f"\nDispersion Swap (Curve: 2Y, 5Y, 10Y, 30Y):")
    print(f"  Maturity: {disp_swap.T} years")
    print(f"  Strike dispersion: {disp_swap.strike_dispersion:.2%}")
    print(f"  Notional per dispersion point: ${disp_swap.notional_per_dispersion_point:,.0f}")
    print(f"  Number of rates: {disp_swap.n_rates}")

    # Simulate paths with varying correlations
    print("\n  Simulating rate paths with varying correlations...")
    key = jrand.PRNGKey(456)
    n_steps = 252
    dt = 1.0 / n_steps

    # Different volatilities for each tenor
    vols = jnp.array([0.18, 0.20, 0.22, 0.16])  # 2Y, 5Y, 10Y, 30Y
    initial_rates = jnp.array([0.025, 0.035, 0.042, 0.044])

    # Generate paths
    paths = []
    for i in range(4):
        key, subkey = jrand.split(key)
        returns = vols[i] * jnp.sqrt(dt) * jrand.normal(subkey, (n_steps,))
        path = initial_rates[i] * jnp.exp(jnp.cumsum(returns))
        paths.append(path)

        print(f"  {['2Y', '5Y', '10Y', '30Y'][i]} rate: {path[0]:.2%} -> {path[-1]:.2%}, vol: {vols[i]:.2%}")

    paths_array = jnp.stack(paths)

    # Calculate payoff
    payoff = disp_swap.payoff_path(paths_array)

    # Calculate individual vols and index vol
    individual_vols = []
    for i in range(4):
        returns = jnp.log(paths[i][1:] / paths[i][:-1])
        vol = jnp.std(returns) * jnp.sqrt(252)
        individual_vols.append(vol)

    avg_individual_vol = jnp.mean(jnp.array(individual_vols))

    index = jnp.mean(paths_array, axis=0)
    index_returns = jnp.log(index[1:] / index[:-1])
    index_vol = jnp.std(index_returns) * jnp.sqrt(252)

    realized_dispersion = avg_individual_vol - index_vol

    print(f"\n  Average individual vol: {avg_individual_vol:.2%}")
    print(f"  Index vol: {index_vol:.2%}")
    print(f"  Realized dispersion: {realized_dispersion:.2%}")
    print(f"  Strike dispersion: {disp_swap.strike_dispersion:.2%}")
    print(f"  P&L: ${payoff:,.0f}")

    return disp_swap


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("IR VOLATILITY PRODUCTS - COMPREHENSIVE DEMO")
    print("=" * 80)
    print("\nThis demo showcases production-ready implementations of:")
    print("  \u2713 Caplet/floorlet volatility surfaces with SABR")
    print("  \u2713 Swaption volatility cubes (3D)")
    print("  \u2713 Swaption straddles and strangles")
    print("  \u2713 IR variance swaps")
    print("  \u2713 Forward variance swaps")
    print("  \u2713 Rate correlation swaps")
    print("  \u2713 Volatility dispersion swaps")

    try:
        # Run all demos
        demo_caplet_vol_surface()
        demo_swaption_vol_cube()
        demo_swaption_straddle_strangle()
        demo_ir_variance_swap()
        demo_forward_variance_swap()
        demo_rate_correlation_swap()
        demo_volatility_dispersion_swap()

        print("\n" + "=" * 80)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nKey Features:")
        print("  • Production-ready implementations")
        print("  • JAX-native for high performance")
        print("  • SABR parameterization for IR markets")
        print("  • 3D interpolation for swaption cubes")
        print("  • Full suite of volatility trading products")
        print("  • Variance space interpolation")
        print("  • Correlation and dispersion trading")

    except Exception as e:
        print(f"\nError in demonstration: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
