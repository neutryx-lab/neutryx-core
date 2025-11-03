"""
Example: FX Volatility Surface Construction from Market Quotes

This example demonstrates how to build an FX volatility surface from
market conventions (ATM, 25Δ Butterfly, 25Δ Risk Reversal).

Market quotes are the standard way FX options are quoted:
- ATM Vol: At-the-money volatility
- 25Δ RR (Risk Reversal): vol_25d_call - vol_25d_put
- 25Δ BF (Butterfly): (vol_25d_call + vol_25d_put)/2 - vol_ATM
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from neutryx.market.fx import (
    FXVolatilityQuote,
    FXVolatilitySurfaceBuilder,
    build_smile_from_market_quote,
)


def example_single_tenor():
    """Example: Build volatility smile from single tenor quote."""
    print("=" * 70)
    print("Example 1: Single Tenor Volatility Smile")
    print("=" * 70)

    # EUR/USD 1Y market quote
    quote = FXVolatilityQuote(
        expiry=1.0,
        atm_vol=0.10,      # 10% ATM vol
        rr_25d=0.015,      # 1.5% risk reversal (calls more expensive than puts)
        bf_25d=0.005,      # 0.5% butterfly (wings more expensive than ATM)
        forward=1.10,
        domestic_rate=0.025,  # USD rate
        foreign_rate=0.015,   # EUR rate
    )

    print(f"\nMarket Quote: {quote}")

    # Extract theoretical vols at pillars
    vols = quote.extract_pillar_vols()
    print("\nTheoretical Volatilities at Market Pillars:")
    print(f"  25Δ Put Vol:  {vols['25d_put']*100:.2f}%")
    print(f"  ATM Vol:      {vols['atm']*100:.2f}%")
    print(f"  25Δ Call Vol: {vols['25d_call']*100:.2f}%")

    # Build complete smile
    strikes, smile_vols = build_smile_from_market_quote(quote, num_strikes=15)

    print(f"\nGenerated Smile with {len(strikes)} strikes")
    print(f"  Strike range: {strikes[0]:.4f} - {strikes[-1]:.4f}")
    print(f"  Vol range:    {jnp.min(smile_vols)*100:.2f}% - {jnp.max(smile_vols)*100:.2f}%")

    # Visualize
    plt.figure(figsize=(10, 6))
    plt.plot(strikes, smile_vols * 100, 'b-', linewidth=2, label='Volatility Smile')
    plt.axvline(quote.forward, color='red', linestyle='--', alpha=0.7, label='Forward (ATM)')
    plt.xlabel('Strike')
    plt.ylabel('Implied Volatility (%)')
    plt.title('EUR/USD 1Y Volatility Smile')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fx_vol_smile_single.png', dpi=150)
    print("\n✓ Saved plot to: fx_vol_smile_single.png")


def example_multi_tenor_surface():
    """Example: Build complete volatility surface from multiple tenors."""
    print("\n" + "=" * 70)
    print("Example 2: Multi-Tenor Volatility Surface")
    print("=" * 70)

    # EUR/USD market quotes across multiple tenors
    quotes = [
        FXVolatilityQuote(
            expiry=0.25,       # 3M
            atm_vol=0.095,
            rr_25d=0.010,
            bf_25d=0.003,
            forward=1.100,
            domestic_rate=0.025,
            foreign_rate=0.015,
        ),
        FXVolatilityQuote(
            expiry=0.5,        # 6M
            atm_vol=0.100,
            rr_25d=0.012,
            bf_25d=0.004,
            forward=1.105,
            domestic_rate=0.025,
            foreign_rate=0.015,
        ),
        FXVolatilityQuote(
            expiry=1.0,        # 1Y
            atm_vol=0.105,
            rr_25d=0.015,
            bf_25d=0.005,
            forward=1.110,
            domestic_rate=0.025,
            foreign_rate=0.015,
        ),
        FXVolatilityQuote(
            expiry=2.0,        # 2Y
            atm_vol=0.110,
            rr_25d=0.018,
            bf_25d=0.006,
            forward=1.120,
            domestic_rate=0.025,
            foreign_rate=0.015,
        ),
        FXVolatilityQuote(
            expiry=5.0,        # 5Y
            atm_vol=0.115,
            rr_25d=0.020,
            bf_25d=0.007,
            forward=1.145,
            domestic_rate=0.025,
            foreign_rate=0.015,
        ),
    ]

    print(f"\nMarket Quotes for EUR/USD:")
    print(f"{'Tenor':<8} {'ATM%':<8} {'RR%':<8} {'BF%':<8} {'Forward':<10}")
    print("-" * 50)
    for q in quotes:
        tenor_str = f"{q.expiry:.2f}Y"
        print(f"{tenor_str:<8} {q.atm_vol*100:>6.2f}  {q.rr_25d*100:>6.2f}  "
              f"{q.bf_25d*100:>6.2f}  {q.forward:>8.4f}")

    # Build surface
    builder = FXVolatilitySurfaceBuilder(
        from_ccy="EUR",
        to_ccy="USD",
        quotes=quotes,
    )

    print(f"\n{builder}")
    surface = builder.build_surface(num_strikes_per_tenor=21)

    print(f"\nGenerated Surface:")
    print(f"  Tenors:  {len(surface.expiries)}")
    print(f"  Strikes: {len(surface.strikes)}")
    print(f"  Shape:   {surface.vols.shape}")

    # Query vol at specific points
    test_points = [
        (0.5, 1.10, "6M ATM"),
        (1.0, 1.15, "1Y OTM Call"),
        (2.0, 1.05, "2Y OTM Put"),
    ]

    print("\nSample Volatility Queries:")
    for expiry, strike, desc in test_points:
        vol = surface.implied_vol(expiry, strike)
        print(f"  {desc:<15} (T={expiry:.2f}y, K={strike:.2f}): {vol*100:.2f}%")

    # Visualize surface
    fig = plt.figure(figsize=(14, 5))

    # Left panel: Volatility smiles by tenor
    ax1 = fig.add_subplot(121)
    for i, expiry in enumerate(surface.expiries):
        vols_for_tenor = surface.vols[i, :]
        label = f"{expiry:.2f}Y"
        ax1.plot(surface.strikes, vols_for_tenor * 100, marker='o', label=label)
    ax1.set_xlabel('Strike')
    ax1.set_ylabel('Implied Volatility (%)')
    ax1.set_title('Volatility Smiles by Tenor')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right panel: Term structure at selected strikes
    ax2 = fig.add_subplot(122)
    strike_indices = [0, len(surface.strikes)//2, len(surface.strikes)-1]
    strike_labels = ['Low Strike (Put)', 'ATM', 'High Strike (Call)']
    for idx, label in zip(strike_indices, strike_labels):
        vols_for_strike = surface.vols[:, idx]
        ax2.plot(surface.expiries, vols_for_strike * 100, marker='s', label=label)
    ax2.set_xlabel('Time to Expiry (years)')
    ax2.set_ylabel('Implied Volatility (%)')
    ax2.set_title('Volatility Term Structure')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fx_vol_surface.png', dpi=150)
    print("\n✓ Saved plot to: fx_vol_surface.png")

    # 3D surface plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create mesh for 3D plot
    T, K = jnp.meshgrid(surface.expiries, surface.strikes)
    V = surface.vols.T * 100  # Transpose to match meshgrid shape

    surf = ax.plot_surface(K, T, V, cmap='viridis', alpha=0.9)
    ax.set_xlabel('Strike')
    ax.set_ylabel('Time to Expiry (years)')
    ax.set_zlabel('Implied Volatility (%)')
    ax.set_title('EUR/USD FX Volatility Surface')
    fig.colorbar(surf, shrink=0.5)

    plt.tight_layout()
    plt.savefig('fx_vol_surface_3d.png', dpi=150)
    print("✓ Saved plot to: fx_vol_surface_3d.png")


def example_market_conventions():
    """Example: Demonstrate market convention conversions."""
    print("\n" + "=" * 70)
    print("Example 3: Understanding Market Conventions")
    print("=" * 70)

    # Show how BF/RR relate to individual vols
    atm = 0.10
    rr = 0.02
    bf = 0.005

    vol_25d_call = atm + bf + rr / 2
    vol_25d_put = atm + bf - rr / 2

    print("\nMarket Quotes:")
    print(f"  ATM Vol:      {atm*100:.2f}%")
    print(f"  25Δ RR:       {rr*100:.2f}%")
    print(f"  25Δ BF:       {bf*100:.2f}%")

    print("\nDerived Theoretical Vols:")
    print(f"  25Δ Call Vol: {vol_25d_call*100:.2f}%  (= ATM + BF + RR/2)")
    print(f"  25Δ Put Vol:  {vol_25d_put*100:.2f}%  (= ATM + BF - RR/2)")

    print("\nVerification:")
    computed_rr = vol_25d_call - vol_25d_put
    computed_bf = (vol_25d_call + vol_25d_put) / 2 - atm
    print(f"  RR from vols: {computed_rr*100:.2f}% ✓")
    print(f"  BF from vols: {computed_bf*100:.2f}% ✓")

    print("\nMarket Interpretation:")
    if rr > 0:
        print(f"  • Positive RR ({rr*100:.2f}%) → Calls more expensive than puts")
        print("    → Market expects upside skew")
    elif rr < 0:
        print(f"  • Negative RR ({rr*100:.2f}%) → Puts more expensive than calls")
        print("    → Market expects downside risk")
    else:
        print("  • Zero RR → Symmetric smile")

    if bf > 0:
        print(f"  • Positive BF ({bf*100:.2f}%) → Wings more expensive than ATM")
        print("    → Market prices tail risk (fat tails)")
    else:
        print(f"  • Negative BF → ATM more expensive than wings")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("FX VOLATILITY MARKET CONVENTIONS (BF/RR)")
    print("=" * 70)

    example_single_tenor()
    example_multi_tenor_surface()
    example_market_conventions()

    print("\n" + "=" * 70)
    print("✓ All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
