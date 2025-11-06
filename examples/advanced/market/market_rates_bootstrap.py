"""
Example: Building Yield Curves from Market Rates

This example demonstrates how to construct yield curves from various market rate
instruments including Cash deposits, FRAs, Futures, and Swaps.
"""

import sys
from pathlib import Path

# Add src to path for imports
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import jax.numpy as jnp
from neutryx.market.curves import BootstrappedCurve, Deposit, FRA, Future, FixedRateSwap


def build_usd_sofr_curve():
    """
    Build a USD SOFR curve using market instruments.

    Market data structure (typical for USD):
    - Cash deposits for short end (O/N to 1M)
    - FRAs for 1M-1Y period
    - Futures for 1Y-3Y period
    - Swaps for 3Y+ period
    """
    print("=" * 70)
    print("Building USD SOFR Curve from Market Rates")
    print("=" * 70)

    # 1. Cash Deposits (short-term money market rates)
    # Format: maturity in years, rate as decimal
    cash_deposits = [
        Deposit(maturity=1 / 12, rate=0.0530),  # 1M @ 5.30%
        Deposit(maturity=3 / 12, rate=0.0535),  # 3M @ 5.35%
        Deposit(maturity=6 / 12, rate=0.0540),  # 6M @ 5.40%
    ]
    print("\n1. Cash Deposits (Money Market):")
    for dep in cash_deposits:
        print(f"   {dep.maturity*12:>3.0f}M: {dep.rate*100:>6.3f}%")

    # 2. FRAs (Forward Rate Agreements)
    # Format: start time, end time, rate
    fras = [
        FRA(start=6 / 12, end=9 / 12, rate=0.0545),  # 6x9 FRA @ 5.45%
        FRA(start=9 / 12, end=12 / 12, rate=0.0550),  # 9x12 FRA @ 5.50%
    ]
    print("\n2. FRAs (Forward Rate Agreements):")
    for fra in fras:
        print(
            f"   {fra.start*12:.0f}x{fra.end*12:.0f}: {fra.rate*100:>6.3f}%"
        )

    # 3. Futures (Interest Rate Futures, e.g., SOFR Futures)
    # Format: start time, end time, price (100 - implied rate)
    # Note: Convexity adjustment converts futures rate to forward rate
    futures = [
        Future(start=1.0, end=1.5, price=94.40, convexity_adjustment=0.0002),
        Future(start=1.5, end=2.0, price=94.30, convexity_adjustment=0.0003),
    ]
    print("\n3. Futures (SOFR Futures):")
    for fut in futures:
        implied_rate = (100.0 - fut.price) / 100.0
        forward_rate = implied_rate - fut.convexity_adjustment
        print(
            f"   {fut.start:.2f}Y-{fut.end:.2f}Y: Price={fut.price:>6.2f} "
            f"(Implied: {implied_rate*100:.2f}%, Forward: {forward_rate*100:.2f}%)"
        )

    # 4. Swaps (Interest Rate Swaps - long end of the curve)
    # Format: fixed rate, payment times, accrual factors
    # Note: Each swap bootstraps only its final maturity point
    # For simplicity, we use annual payments that align with existing nodes
    swap_maturities = [3, 4, 5, 6, 7, 8, 9, 10]
    swap_rates = [0.0565, 0.05675, 0.0570, 0.0573, 0.0574, 0.05745, 0.05747, 0.0575]

    swaps = []
    for maturity, rate in zip(swap_maturities, swap_rates):
        payment_times = list(range(1, maturity + 1))
        accrual_factors = [1.0] * maturity
        swaps.append(
            FixedRateSwap(
                fixed_rate=rate,
                payment_times=[float(t) for t in payment_times],
                accrual_factors=accrual_factors,
            )
        )

    print("\n4. Swaps (Interest Rate Swaps):")
    swap_tenors = [f"{m}Y" for m in swap_maturities]
    for tenor, swap in zip(swap_tenors, swaps):
        print(f"   {tenor:>3s}: {swap.fixed_rate*100:>6.3f}%")

    # Bootstrap the curve
    print("\n" + "=" * 70)
    print("Bootstrapping Curve...")
    print("=" * 70)

    all_instruments = [*cash_deposits, *fras, *futures, *swaps]
    curve = BootstrappedCurve(all_instruments)

    print(f"✓ Successfully bootstrapped {len(all_instruments)} instruments")

    # Display curve results
    print("\n" + "=" * 70)
    print("Curve Results")
    print("=" * 70)
    print(f"{'Maturity':<12} {'Discount Factor':>18} {'Zero Rate':>12}")
    print("-" * 70)

    test_maturities = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    for t in test_maturities:
        df = float(curve.df(t))
        zr = float(curve.zero_rate(t))
        print(f"{t:>6.2f}Y     {df:>18.10f}     {zr*100:>10.4f}%")

    # Calculate forward rates
    print("\n" + "=" * 70)
    print("Forward Rates")
    print("=" * 70)
    print(f"{'Period':<12} {'Forward Rate':>15}")
    print("-" * 70)

    forward_periods = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 10.0)]
    for t0, t1 in forward_periods:
        fr = float(curve.forward_rate(t0, t1))
        print(f"{t0:.1f}Y - {t1:.1f}Y  {fr*100:>13.4f}%")

    return curve


def build_eur_curve():
    """
    Build a simplified EUR curve (ESTR-based).
    """
    print("\n\n" + "=" * 70)
    print("Building EUR ESTR Curve from Market Rates")
    print("=" * 70)

    instruments = [
        # Cash deposits
        Deposit(maturity=1 / 12, rate=0.0385),  # 1M @ 3.85%
        Deposit(maturity=3 / 12, rate=0.0390),  # 3M @ 3.90%
        Deposit(maturity=6 / 12, rate=0.0395),  # 6M @ 3.95%
        # FRAs
        FRA(start=0.5, end=0.75, rate=0.0400),  # 6x9 FRA @ 4.00%
        FRA(start=0.75, end=1.0, rate=0.0405),  # 9x12 FRA @ 4.05%
        # Swaps with annual payments
        FixedRateSwap(
            fixed_rate=0.0420,
            payment_times=[1.0, 2.0],
            accrual_factors=[1.0, 1.0],
        ),  # 2Y @ 4.20%
        FixedRateSwap(
            fixed_rate=0.0430,
            payment_times=[1.0, 2.0, 3.0],
            accrual_factors=[1.0, 1.0, 1.0],
        ),  # 3Y @ 4.30%
        FixedRateSwap(
            fixed_rate=0.0433,
            payment_times=[1.0, 2.0, 3.0, 4.0],
            accrual_factors=[1.0, 1.0, 1.0, 1.0],
        ),  # 4Y @ 4.33%
        FixedRateSwap(
            fixed_rate=0.0435,
            payment_times=[1.0, 2.0, 3.0, 4.0, 5.0],
            accrual_factors=[1.0, 1.0, 1.0, 1.0, 1.0],
        ),  # 5Y @ 4.35%
    ]

    curve = BootstrappedCurve(instruments)

    print(f"\n✓ Successfully bootstrapped {len(instruments)} instruments")
    print("\nCurve snapshot:")
    print(f"{'Maturity':<12} {'Discount Factor':>18} {'Zero Rate':>12}")
    print("-" * 70)

    for t in [0.25, 0.5, 1.0, 2.0, 5.0]:
        df = float(curve.df(t))
        zr = float(curve.zero_rate(t))
        print(f"{t:>6.2f}Y     {df:>18.10f}     {zr*100:>10.4f}%")

    return curve


def main():
    """Run all examples."""
    usd_curve = build_usd_sofr_curve()
    eur_curve = build_eur_curve()

    print("\n" + "=" * 70)
    print("Example Complete")
    print("=" * 70)
    print(
        """
The bootstrapped curves can now be used for:
- Pricing interest rate derivatives (swaps, swaptions, caps/floors)
- Calculating present values of cashflows
- Risk management (DV01, key rate durations)
- XVA calculations (CVA, DVA, FVA, MVA)

Key market instruments:
  1. Cash/Deposits: Short-term money market rates
  2. FRA: Forward Rate Agreements for medium-term forwards
  3. Futures: Exchange-traded interest rate futures (with convexity adjustment)
  4. Swaps: Interest rate swaps for long-term rates

For more details, see:
  - src/neutryx/market/curves.py (implementation)
  - src/neutryx/tests/market/test_curves.py (tests)
"""
    )


if __name__ == "__main__":
    main()
