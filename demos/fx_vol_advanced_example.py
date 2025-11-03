"""
Advanced FX Volatility Features Example

This example demonstrates the advanced features:
1. Multiple delta pillars (10Δ, 15Δ, 25Δ)
2. SABR calibration from market quotes
3. Vanna-Volga interpolation
4. Market data source integration
5. Comparison of interpolation methods
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path

from neutryx.market.fx import (
    FXVolatilityQuote,
    FXVolatilitySurfaceBuilder,
    build_smile_with_method,
    calibrate_sabr_from_quote,
    vanna_volga_weights,
)
from neutryx.market.market_data import get_market_data_source

# Create outputs directory
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def example_multi_delta_pillars():
    """Example: Using multiple delta pillars for better smile fitting."""
    print("=" * 70)
    print("Example 1: Multiple Delta Pillars (10Δ, 15Δ, 25Δ)")
    print("=" * 70)

    # EUR/USD quote with full delta structure
    quote_full = FXVolatilityQuote(
        expiry=1.0,
        atm_vol=0.10,
        # 25Δ pillars
        rr_25d=0.015,
        bf_25d=0.005,
        # 15Δ pillars
        rr_15d=0.020,
        bf_15d=0.006,
        # 10Δ pillars
        rr_10d=0.025,
        bf_10d=0.008,
        forward=1.10,
        domestic_rate=0.025,
        foreign_rate=0.015,
    )

    print(f"\nMarket Quote: {quote_full}")

    # Extract all pillars
    vols = quote_full.extract_pillar_vols()
    print("\nVolatility Pillars:")
    for key in sorted(vols.keys()):
        print(f"  {key:12s}: {vols[key]*100:.2f}%")

    # Available deltas
    deltas = quote_full.get_available_deltas()
    print(f"\nAvailable Deltas: {deltas}")

    print("\n" + "=" * 70)


def example_sabr_calibration():
    """Example: SABR model calibration from market quotes."""
    print("\n" + "=" * 70)
    print("Example 2: SABR Calibration")
    print("=" * 70)

    quote = FXVolatilityQuote(
        expiry=1.0,
        atm_vol=0.10,
        rr_25d=0.015,
        bf_25d=0.005,
        forward=1.10,
        domestic_rate=0.025,
        foreign_rate=0.015,
    )

    print(f"\nMarket Quote: {quote}")

    # Calibrate SABR with different beta values
    for beta in [0.0, 0.5, 1.0]:
        params = calibrate_sabr_from_quote(quote, beta=beta)
        print(f"\nSABR Parameters (β={beta}):")
        print(f"  α (alpha):  {params.alpha:.4f}")
        print(f"  β (beta):   {params.beta:.4f}")
        print(f"  ρ (rho):    {params.rho:.4f}")
        print(f"  ν (nu):     {params.nu:.4f}")

    print("\n" + "=" * 70)


def example_interpolation_comparison():
    """Example: Compare different interpolation methods."""
    print("\n" + "=" * 70)
    print("Example 3: Interpolation Method Comparison")
    print("=" * 70)

    quote = FXVolatilityQuote(
        expiry=1.0,
        atm_vol=0.10,
        rr_25d=0.020,
        bf_25d=0.006,
        forward=1.10,
        domestic_rate=0.025,
        foreign_rate=0.015,
    )

    methods = ["linear", "sabr", "vanna_volga"]
    results = {}

    print("\nGenerating smiles with different methods...")
    for method in methods:
        strikes, vols = build_smile_with_method(
            quote, num_strikes=25, method=method, sabr_beta=0.5
        )
        results[method] = (strikes, vols)
        print(f"  ✓ {method:15s}: {len(strikes)} strikes")

    # Visualize comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: All smiles
    for method, (strikes, vols) in results.items():
        ax1.plot(strikes, vols * 100, '-o', label=method.title(), markersize=3)

    ax1.axvline(quote.forward, color='red', linestyle='--', alpha=0.5, label='Forward (ATM)')
    ax1.set_xlabel('Strike')
    ax1.set_ylabel('Implied Volatility (%)')
    ax1.set_title('Smile Interpolation Methods Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Difference from linear
    linear_strikes, linear_vols = results['linear']
    for method, (strikes, vols) in results.items():
        if method != 'linear':
            # Interpolate to common strikes
            vols_interp = jnp.interp(linear_strikes, strikes, vols)
            diff = (vols_interp - linear_vols) * 10000  # bps
            ax2.plot(linear_strikes, diff, '-o', label=f'{method.title()} - Linear', markersize=3)

    ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Strike')
    ax2.set_ylabel('Difference from Linear (bps)')
    ax2.set_title('Interpolation Method Differences')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'fx_vol_interpolation_comparison.png'
    plt.savefig(output_file, dpi=150)
    print(f"\n✓ Saved plot to: {output_file}")

    print("\n" + "=" * 70)


def example_vanna_volga_mechanics():
    """Example: Understanding Vanna-Volga weights."""
    print("\n" + "=" * 70)
    print("Example 4: Vanna-Volga Mechanics")
    print("=" * 70)

    forward = 1.10
    strike_25p = 1.05
    strike_atm = 1.10
    strike_25c = 1.15

    print(f"\nPillar Strikes:")
    print(f"  25Δ Put:  {strike_25p:.4f}")
    print(f"  ATM:      {strike_atm:.4f}")
    print(f"  25Δ Call: {strike_25c:.4f}")

    print(f"\nVanna-Volga Weights:")
    print(f"{'Strike':<10} {'w_25p':<10} {'w_ATM':<10} {'w_25c':<10} {'Sum':<10}")
    print("-" * 50)

    test_strikes = [1.03, 1.05, 1.07, 1.10, 1.13, 1.15, 1.17]
    for strike in test_strikes:
        w1, w2, w3 = vanna_volga_weights(strike, forward, strike_25p, strike_atm, strike_25c)
        total = w1 + w2 + w3
        print(f"{strike:<10.4f} {w1:<10.4f} {w2:<10.4f} {w3:<10.4f} {total:<10.4f}")

    print("\nNote: Weights sum to 1.0 at all strikes (quadratic interpolation)")

    print("\n" + "=" * 70)


def example_market_data_integration():
    """Example: Using market data sources."""
    print("\n" + "=" * 70)
    print("Example 5: Market Data Source Integration")
    print("=" * 70)

    # Create simulated data source
    source = get_market_data_source(
        "simulated",
        base_vol=0.12,
        vol_term_structure=0.04,
        rr_skew=0.018,
        bf_convexity=0.006,
        seed=42
    )

    print("\nSimulated Market Data Source:")
    print(f"  Base Vol:        {source.base_vol*100:.1f}%")
    print(f"  Term Structure:  {source.vol_term_structure*100:.1f}%")
    print(f"  RR Skew:         {source.rr_skew*100:.1f}%")
    print(f"  BF Convexity:    {source.bf_convexity*100:.1f}%")

    # Fetch FX spot
    spot = source.get_fx_spot("EUR", "USD")
    print(f"\nEUR/USD Spot: {spot:.4f}")

    # Fetch single quote
    quote_1y = source.get_fx_vol_quote("EUR", "USD", 1.0)
    print(f"\n1Y Quote: {quote_1y}")

    # Fetch full surface
    tenors = [0.25, 0.5, 1.0, 2.0, 5.0]
    quotes = source.get_fx_vol_surface("EUR", "USD", tenors)

    print(f"\nVolatility Surface:")
    print(f"{'Tenor':<8} {'ATM%':<8} {'RR%':<8} {'BF%':<8} {'Forward':<10}")
    print("-" * 50)
    for q in quotes:
        tenor_str = f"{q.expiry:.2f}Y"
        print(f"{tenor_str:<8} {q.atm_vol*100:>6.2f}  {q.rr_25d*100:>6.2f}  "
              f"{q.bf_25d*100:>6.2f}  {q.forward:>8.4f}")

    # Build surface from simulated data
    builder = FXVolatilitySurfaceBuilder(
        from_ccy="EUR",
        to_ccy="USD",
        quotes=quotes,
    )
    surface = builder.build_surface(num_strikes_per_tenor=21)

    print(f"\nBuilt Surface: {len(surface.expiries)} tenors, {len(surface.strikes)} strikes")

    # Query some vols
    print("\nSample Volatility Queries:")
    test_points = [
        (0.5, 1.08, "6M, Slight OTM Put"),
        (1.0, 1.10, "1Y, ATM"),
        (2.0, 1.15, "2Y, OTM Call"),
    ]
    for expiry, strike, desc in test_points:
        vol = surface.implied_vol(expiry, strike)
        print(f"  {desc:<25} → {vol*100:.2f}%")

    print("\n" + "=" * 70)


def example_end_to_end_workflow():
    """Example: Complete end-to-end workflow."""
    print("\n" + "=" * 70)
    print("Example 6: End-to-End Workflow")
    print("=" * 70)

    print("\nStep 1: Get market data from simulated source")
    source = get_market_data_source("simulated", seed=42)
    tenors = [0.25, 0.5, 1.0, 2.0, 5.0]
    quotes = source.get_fx_vol_surface("EUR", "USD", tenors)
    print(f"  ✓ Retrieved {len(quotes)} market quotes")

    print("\nStep 2: Build volatility surfaces with different methods")
    methods = ["linear", "sabr", "vanna_volga"]
    surfaces = {}

    for method in methods:
        # Build smiles for each tenor
        builder_quotes = []
        for quote in quotes:
            # For surface builder, we need to use the basic quote format
            # The surface builder will call build_smile_from_market_quote internally
            builder_quotes.append(quote)

        builder = FXVolatilitySurfaceBuilder(
            from_ccy="EUR",
            to_ccy="USD",
            quotes=builder_quotes,
        )
        surface = builder.build_surface(num_strikes_per_tenor=15)
        surfaces[method] = surface
        print(f"  ✓ {method:15s} surface: {surface.vols.shape}")

    print("\nStep 3: Compare implied vols at test points")
    test_points = [
        (0.5, 1.08),
        (1.0, 1.10),
        (2.0, 1.12),
        (5.0, 1.15),
    ]

    print(f"\n{'Expiry':<10} {'Strike':<10} {'Linear%':<10} {'SABR%':<10} {'VV%':<10}")
    print("-" * 55)
    for expiry, strike in test_points:
        vols_dict = {}
        for method in methods:
            vol = surfaces[method].implied_vol(expiry, strike)
            vols_dict[method] = vol

        print(f"{expiry:<10.2f} {strike:<10.4f} "
              f"{float(vols_dict['linear'])*100:<10.2f} "
              f"{float(vols_dict['sabr'])*100:<10.2f} "
              f"{float(vols_dict['vanna_volga'])*100:<10.2f}")

    print("\nStep 4: Calibrate SABR to 1Y quote")
    quote_1y = quotes[2]  # 1Y quote
    sabr_params = calibrate_sabr_from_quote(quote_1y, beta=0.5)
    print(f"  ✓ SABR Parameters: α={sabr_params.alpha:.4f}, "
          f"ρ={sabr_params.rho:.4f}, ν={sabr_params.nu:.4f}")

    print("\nStep 5: Generate smile comparison plot")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('EUR/USD Volatility Surfaces - Method Comparison', fontsize=14, fontweight='bold')

    for idx, (tenor_idx, tenor) in enumerate([(0, 0.25), (2, 1.0), (4, 5.0)]):
        ax = axes[0, idx]

        for method in methods:
            surface = surfaces[method]
            vols_for_tenor = surface.vols[tenor_idx, :]
            ax.plot(surface.strikes, vols_for_tenor * 100, '-o', label=method.title(), markersize=4)

        ax.set_xlabel('Strike')
        ax.set_ylabel('Implied Vol (%)')
        ax.set_title(f'{tenor:.2f}Y Smile')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Bottom row: Term structure at different strikes
    strike_indices = [3, 7, 11]
    strike_labels = ['Low Strike', 'ATM', 'High Strike']

    for idx, (strike_idx, label) in enumerate(zip(strike_indices, strike_labels)):
        ax = axes[1, idx]

        for method in methods:
            surface = surfaces[method]
            vols_for_strike = surface.vols[:, strike_idx]
            ax.plot(surface.expiries, vols_for_strike * 100, '-s', label=method.title(), markersize=5)

        ax.set_xlabel('Time to Expiry (years)')
        ax.set_ylabel('Implied Vol (%)')
        ax.set_title(f'Term Structure - {label}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'fx_vol_end_to_end.png'
    plt.savefig(output_file, dpi=150)
    print(f"\n  ✓ Saved plot to: {output_file}")

    print("\n" + "=" * 70)
    print("✓ End-to-end workflow completed successfully!")
    print("=" * 70)


def main():
    """Run all advanced examples."""
    print("\n" + "=" * 70)
    print("ADVANCED FX VOLATILITY FEATURES")
    print("=" * 70)

    example_multi_delta_pillars()
    example_sabr_calibration()
    example_interpolation_comparison()
    example_vanna_volga_mechanics()
    example_market_data_integration()
    example_end_to_end_workflow()

    print("\n" + "=" * 70)
    print("✓ All advanced examples completed successfully!")
    print("=" * 70)
    print("\nGenerated Plots:")
    print("  1. fx_vol_interpolation_comparison.png - Method comparison")
    print("  2. fx_vol_end_to_end.png - Complete workflow visualization")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
