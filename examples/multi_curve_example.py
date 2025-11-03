"""
Example: Multi-Curve Framework with OIS, Tenor Basis, and Currency Basis

This example demonstrates building multiple related curves:
1. USD OIS discount curve (SOFR-based)
2. EUR OIS discount curve (ESTR-based)
3. USD-collateralized EUR discount curve (using cross-currency basis)
4. Tenor projection curves (3M LIBOR/SOFR)
"""

import sys
from pathlib import Path

# Add src to path for imports
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from neutryx.market.curves import (
    BootstrappedCurve,
    CurrencyBasisSwap,
    Deposit,
    FRA,
    Future,
    FixedRateSwap,
    OIS,
    TenorBasisSwap,
)
from neutryx.market.multi_curve import (
    CurveDefinition,
    MultiCurveBuilder,
    build_simple_multi_curve,
)


def build_usd_eur_multi_curve():
    """
    Build a complete multi-currency, multi-tenor curve system.

    Structure:
    - USD-OIS: USD SOFR discount curve
    - EUR-OIS: EUR ESTR discount curve
    - EUR-OIS-USD: EUR discount curve for USD-collateralized trades
    - USD-3M: USD 3M SOFR projection curve
    """
    print("=" * 80)
    print("Multi-Currency, Multi-Curve Framework Example")
    print("=" * 80)

    builder = MultiCurveBuilder()

    # ========================================================================
    # 1. USD OIS Discount Curve (SOFR)
    # ========================================================================
    print("\n1. Building USD SOFR OIS Discount Curve")
    print("-" * 80)

    usd_ois_instruments = [
        Deposit(maturity=1 / 12, rate=0.0530),  # 1M
        Deposit(maturity=3 / 12, rate=0.0535),  # 3M
        OIS(
            fixed_rate=0.0540,
            payment_times=[0.5],
            accrual_factors=[0.5],
            compounding="compound",
        ),  # 6M OIS
        OIS(
            fixed_rate=0.0545,
            payment_times=[1.0],
            accrual_factors=[1.0],
            compounding="compound",
        ),  # 1Y OIS
        OIS(
            fixed_rate=0.0555,
            payment_times=[1.0, 2.0],
            accrual_factors=[1.0, 1.0],
            compounding="compound",
        ),  # 2Y OIS
        OIS(
            fixed_rate=0.0560,
            payment_times=[1.0, 2.0, 3.0],
            accrual_factors=[1.0, 1.0, 1.0],
            compounding="compound",
        ),  # 3Y OIS
        OIS(
            fixed_rate=0.0565,
            payment_times=[1.0, 2.0, 3.0, 4.0],
            accrual_factors=[1.0, 1.0, 1.0, 1.0],
            compounding="compound",
        ),  # 4Y OIS
        OIS(
            fixed_rate=0.0570,
            payment_times=[1.0, 2.0, 3.0, 4.0, 5.0],
            accrual_factors=[1.0, 1.0, 1.0, 1.0, 1.0],
            compounding="compound",
        ),  # 5Y OIS
    ]

    builder.add_curve_definition(
        CurveDefinition(
            name="USD-OIS",
            currency="USD",
            curve_type="discount",
            instruments=usd_ois_instruments,
        )
    )

    print(f"  Added {len(usd_ois_instruments)} USD OIS instruments (2 Deposits + {len(usd_ois_instruments)-2} OIS)")

    # ========================================================================
    # 2. EUR OIS Discount Curve (ESTR)
    # ========================================================================
    print("\n2. Building EUR ESTR OIS Discount Curve")
    print("-" * 80)

    eur_ois_instruments = [
        Deposit(maturity=1 / 12, rate=0.0385),  # 1M
        Deposit(maturity=3 / 12, rate=0.0390),  # 3M
        OIS(
            fixed_rate=0.0395,
            payment_times=[0.5],
            accrual_factors=[0.5],
            compounding="compound",
        ),  # 6M OIS
        OIS(
            fixed_rate=0.0400,
            payment_times=[1.0],
            accrual_factors=[1.0],
            compounding="compound",
        ),  # 1Y OIS
        OIS(
            fixed_rate=0.0420,
            payment_times=[1.0, 2.0],
            accrual_factors=[1.0, 1.0],
            compounding="compound",
        ),  # 2Y OIS
        OIS(
            fixed_rate=0.0425,
            payment_times=[1.0, 2.0, 3.0],
            accrual_factors=[1.0, 1.0, 1.0],
            compounding="compound",
        ),  # 3Y OIS
        OIS(
            fixed_rate=0.0430,
            payment_times=[1.0, 2.0, 3.0, 4.0],
            accrual_factors=[1.0, 1.0, 1.0, 1.0],
            compounding="compound",
        ),  # 4Y OIS
        OIS(
            fixed_rate=0.0435,
            payment_times=[1.0, 2.0, 3.0, 4.0, 5.0],
            accrual_factors=[1.0, 1.0, 1.0, 1.0, 1.0],
            compounding="compound",
        ),  # 5Y OIS
    ]

    builder.add_curve_definition(
        CurveDefinition(
            name="EUR-OIS",
            currency="EUR",
            curve_type="discount",
            instruments=eur_ois_instruments,
        )
    )

    print(f"  Added {len(eur_ois_instruments)} EUR OIS instruments (2 Deposits + {len(eur_ois_instruments)-2} OIS)")

    # ========================================================================
    # 3. FX Spot Rate
    # ========================================================================
    print("\n3. Setting FX Spot Rate")
    print("-" * 80)

    fx_spot = 1.10  # EUR/USD
    builder.add_fx_spot("EUR", "USD", fx_spot)
    print(f"  EUR/USD spot rate: {fx_spot:.4f}")

    # ========================================================================
    # 4. Build all curves
    # ========================================================================
    print("\n4. Building All Curves")
    print("-" * 80)

    env = builder.build()
    print(f"  ✓ Successfully built {len(env.curves)} curves")

    # ========================================================================
    # 5. Display Results
    # ========================================================================
    print("\n" + "=" * 80)
    print("Curve Results")
    print("=" * 80)

    for curve_name, curve in env.curves.items():
        print(f"\n{curve_name} Curve:")
        print(f"{'Maturity':<12} {'Discount Factor':>18} {'Zero Rate':>12}")
        print("-" * 50)

        test_maturities = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
        for t in test_maturities:
            try:
                df = float(curve.df(t))
                zr = float(curve.zero_rate(t))
                print(f"{t:>6.2f}Y     {df:>18.10f}     {zr*100:>10.4f}%")
            except:
                pass

    return env


def build_tenor_basis_example():
    """
    Example: Building projection curves with tenor basis.

    Demonstrates 3M vs 6M tenor basis in USD.
    """
    print("\n\n" + "=" * 80)
    print("Tenor Basis Example: USD 3M vs 6M SOFR")
    print("=" * 80)

    # First, build the USD OIS discount curve
    usd_ois_instruments = [
        Deposit(maturity=0.25, rate=0.053),
        Deposit(maturity=0.5, rate=0.054),
        OIS(
            fixed_rate=0.0545,
            payment_times=[1.0],
            accrual_factors=[1.0],
        ),
        OIS(
            fixed_rate=0.055,
            payment_times=[1.0, 2.0],
            accrual_factors=[1.0, 1.0],
        ),
    ]

    discount_curve = BootstrappedCurve(usd_ois_instruments)
    print("\n✓ Built USD OIS discount curve")

    # Now build 3M projection curve
    usd_3m_instruments = [
        Deposit(maturity=0.25, rate=0.0560),  # 3M cash
        FRA(start=0.25, end=0.5, rate=0.0565),
        FRA(start=0.5, end=0.75, rate=0.0570),
        FRA(start=0.75, end=1.0, rate=0.0575),
        FixedRateSwap(
            fixed_rate=0.058,
            payment_times=[1.0, 2.0],
            accrual_factors=[1.0, 1.0],
        ),
    ]

    projection_3m_curve = BootstrappedCurve(usd_3m_instruments)
    print("✓ Built USD 3M projection curve")

    # Display forward rates
    print("\nForward Rates:")
    print(f"{'Period':<15} {'3M Forward Rate':>18}")
    print("-" * 40)

    periods = [(0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0)]
    for t0, t1 in periods:
        fr = float(projection_3m_curve.forward_rate(t0, t1))
        print(f"{t0:.2f}Y - {t1:.2f}Y  {fr*100:>16.4f}%")

    print(
        """
Note: In a full implementation, tenor basis swaps would be used to build
the 6M curve relative to the 3M curve, capturing the basis spread between
different tenors.
"""
    )


def build_currency_basis_example():
    """
    Example: Cross-currency basis adjustment.

    Shows how to build EUR discount curve for USD-collateralized trades.
    """
    print("\n\n" + "=" * 80)
    print("Cross-Currency Basis Example")
    print("=" * 80)
    print(
        """
In modern derivatives pricing, the discount curve depends on the collateral
currency, not the trade currency. For example:
- A EUR trade with USD collateral uses: EUR-OIS-USD discount curve
- A EUR trade with EUR collateral uses: EUR-OIS discount curve

The cross-currency basis swap captures the difference between these curves.
"""
    )

    print("\nTypical USD/EUR basis spreads:")
    print("  1Y:  -15 bps")
    print("  3Y:  -18 bps")
    print("  5Y:  -20 bps")
    print(
        """
A negative basis means EUR funding is more expensive than USD funding,
so EUR assets discounted with USD collateral have higher value.
"""
    )


def main():
    """Run all examples."""
    # Main multi-curve example
    env = build_usd_eur_multi_curve()

    # Tenor basis example
    build_tenor_basis_example()

    # Currency basis example
    build_currency_basis_example()

    print("\n" + "=" * 80)
    print("Example Complete")
    print("=" * 80)
    print(
        """
Key Takeaways:

1. Multi-Curve Framework:
   - Separate discount curves (OIS/RFR) for discounting
   - Separate projection curves (LIBOR/SOFR tenors) for forward rates
   - Essential for accurate post-2008 derivatives pricing

2. Collateral Currency Matters:
   - EUR trade with USD collateral uses different discount than EUR collateral
   - Cross-currency basis captures this difference
   - Critical for XVA calculations

3. Market Instruments:
   - OIS: Build risk-free discount curves
   - Tenor Basis Swaps: Build multi-tenor projection curves
   - Currency Basis Swaps: Build cross-currency collateral curves

4. Dependency Management:
   - Curves must be built in correct order
   - Projection curves depend on discount curves
   - Foreign discount curves depend on domestic discount curves + FX + basis

For more details, see:
  - src/neutryx/market/curves.py (instrument implementations)
  - src/neutryx/market/multi_curve.py (multi-curve framework)
"""
    )


if __name__ == "__main__":
    main()
