"""
Example: Advanced Features - IMM Dates and Numerical Solvers

This example demonstrates:
1. IMM date calculation for futures contracts
2. Newton-Raphson solver for implied rate calculations
3. Building curves with futures using IMM dates
4. Finding implied forward rates from futures prices
"""

import sys
from datetime import date
from pathlib import Path

# Add src to path for imports
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from neutryx.market.conventions import (
    get_imm_date,
    get_imm_code,
    get_imm_dates_between,
    get_next_imm_date,
    is_imm_month,
    parse_imm_code,
)
from neutryx.market.curves import BootstrappedCurve, Deposit, Future, OIS
from neutryx.core.math.solvers import brent, newton_raphson


def demonstrate_imm_dates():
    """Show IMM date calculation for futures contracts."""
    print("=" * 80)
    print("IMM Dates for Futures Contracts")
    print("=" * 80)
    print(
        """
IMM dates are standardized settlement dates for futures contracts:
- 3rd Wednesday of March, June, September, December
- Used by CME, LIFFE, and other exchanges
- Provides liquidity concentration on standard dates
"""
    )

    # Example 1: Get IMM dates for 2025
    print("\n1. IMM Dates for 2025:")
    print("-" * 80)
    for month in [3, 6, 9, 12]:
        imm_date = get_imm_date(2025, month)
        imm_code = get_imm_code(imm_date)
        print(f"  {imm_code}: {imm_date.strftime('%Y-%m-%d (%A)')}")

    # Example 2: Check if dates are IMM months
    print("\n2. Checking IMM Months:")
    print("-" * 80)
    for month in [1, 3, 6, 7, 12]:
        is_imm = is_imm_month(month)
        print(f"  Month {month:2d}: {'✓ IMM month' if is_imm else '✗ Not IMM month'}")

    # Example 3: Get next IMM date from today
    print("\n3. Next IMM Date from Reference Dates:")
    print("-" * 80)
    reference_dates = [
        date(2025, 1, 15),
        date(2025, 3, 19),  # Day of March IMM
        date(2025, 6, 20),  # After June IMM
    ]
    for ref_date in reference_dates:
        next_imm = get_next_imm_date(ref_date)
        print(f"  From {ref_date}: Next IMM is {next_imm} ({get_imm_code(next_imm)})")

    # Example 4: Get all IMM dates in a range (for building futures strip)
    print("\n4. Futures Strip Schedule (Next 12 Months):")
    print("-" * 80)
    start_date = date(2025, 1, 1)
    end_date = date(2025, 12, 31)
    imm_dates = get_imm_dates_between(start_date, end_date)
    print(f"  From {start_date} to {end_date}:")
    for imm_date in imm_dates:
        code = get_imm_code(imm_date)
        print(f"    {code}: {imm_date.strftime('%Y-%m-%d')}")

    # Example 5: Parse IMM codes
    print("\n5. Parsing IMM Codes:")
    print("-" * 80)
    codes = ["H5", "M5", "U5", "Z5", "H6"]  # 2025-2026 contracts
    for code in codes:
        parsed_date = parse_imm_code(code, ref_year=2025)
        print(f"  {code} → {parsed_date.strftime('%Y-%m-%d')}")


def demonstrate_newton_raphson():
    """Show Newton-Raphson solver for implied calculations."""
    print("\n\n" + "=" * 80)
    print("Newton-Raphson Solver for Implied Rates")
    print("=" * 80)
    print(
        """
Newton-Raphson is used for finding implied values where direct calculation
is not possible. Common applications:
- Implied forward rates from futures prices
- Implied volatility from option prices
- Bootstrapping complex instruments
"""
    )

    # Example 1: Simple root finding
    print("\n1. Finding Square Root (x² - 2 = 0):")
    print("-" * 80)

    def f_square(x):
        return x**2 - 2.0

    root = newton_raphson(f_square, x0=1.0, tol=1e-10)
    print(f"  Starting guess: x₀ = 1.0")
    print(f"  Solution: x = {root:.12f}")
    print(f"  √2 = {2**0.5:.12f}")
    print(f"  Error: {abs(root - 2**0.5):.2e}")

    # Example 2: Implied forward rate from futures price
    print("\n2. Implied Forward Rate from Futures Price:")
    print("-" * 80)
    futures_price = 95.25  # 3-month futures
    convexity_adj = 0.0002  # 2 bps convexity adjustment

    def futures_price_error(rate):
        """Price error as function of implied rate."""
        return 100.0 - rate * 100.0 - futures_price

    implied_rate = newton_raphson(futures_price_error, x0=0.05, tol=1e-10)
    forward_rate = implied_rate - convexity_adj

    print(f"  Futures Price: {futures_price:.2f}")
    print(f"  Implied Rate: {implied_rate*100:.4f}%")
    print(f"  Convexity Adjustment: {convexity_adj*10000:.1f} bps")
    print(f"  Forward Rate: {forward_rate*100:.4f}%")

    # Example 3: Discount factor from par swap rate
    print("\n3. Implied Discount Factor from Par Swap Rate:")
    print("-" * 80)
    par_rate = 0.0550  # 5.50% par swap rate
    payment_times = [1.0, 2.0, 3.0]
    known_dfs = [0.95, 0.92, 0.89]  # Known discount factors

    def swap_pv(df_final):
        """PV of swap as function of final discount factor."""
        fixed_leg = par_rate * sum(known_dfs) + par_rate * df_final
        float_leg = 1.0 - df_final
        return fixed_leg - float_leg

    df_final = newton_raphson(swap_pv, x0=0.85, tol=1e-10)
    print(f"  Par Swap Rate: {par_rate*100:.2f}%")
    print(f"  Known DFs: {known_dfs}")
    print(f"  Implied Final DF: {df_final:.10f}")
    print(f"  PV Check: {swap_pv(df_final):.2e}")

    # Example 4: Comparison with Brent method
    print("\n4. Solver Comparison (finding zero of x³ - 2x - 5 = 0):")
    print("-" * 80)

    def cubic(x):
        return x**3 - 2 * x - 5

    # Newton-Raphson
    try:
        nr_root = newton_raphson(cubic, x0=2.0, tol=1e-10)
        print(f"  Newton-Raphson: x = {nr_root:.10f}, f(x) = {cubic(nr_root):.2e}")
    except Exception as e:
        print(f"  Newton-Raphson: Failed ({e})")

    # Brent method
    try:
        brent_root = brent(cubic, a=1.0, b=3.0, tol=1e-10)
        print(f"  Brent Method:   x = {brent_root:.10f}, f(x) = {cubic(brent_root):.2e}")
    except Exception as e:
        print(f"  Brent Method: Failed ({e})")


def demonstrate_futures_curve_with_imm():
    """Build a curve using futures contracts scheduled on IMM dates."""
    print("\n\n" + "=" * 80)
    print("Building Curve with Futures on IMM Dates")
    print("=" * 80)
    print(
        """
Practical example: Building USD SOFR curve using:
- Cash deposits for short end (0-3M)
- SOFR Futures on IMM dates (3M-2Y)
- OIS swaps for long end (2Y-5Y)
"""
    )

    # Step 1: Define IMM dates for futures
    print("\n1. Futures Strip Schedule:")
    print("-" * 80)
    start_date = date(2025, 1, 1)
    imm_dates = get_imm_dates_between(start_date, date(2026, 12, 31))[:8]  # 2 years

    print(f"  Reference Date: {start_date}")
    print(f"  {'IMM Code':<10} {'Settlement Date':<15} {'Days':<8} {'Years':<10}")
    print("  " + "-" * 50)

    futures_schedule = []
    for imm_date in imm_dates:
        days = (imm_date - start_date).days
        years = days / 365.25
        code = get_imm_code(imm_date)
        futures_schedule.append((code, imm_date, years))
        print(f"  {code:<10} {imm_date.strftime('%Y-%m-%d'):<15} {days:<8} {years:<10.4f}")

    # Step 2: Define market data
    print("\n2. Market Data:")
    print("-" * 80)

    # Cash deposits
    print("  Cash Deposits:")
    deposits = [
        Deposit(maturity=1 / 12, rate=0.0530),  # 1M
        Deposit(maturity=3 / 12, rate=0.0535),  # 3M
    ]
    for i, dep in enumerate(deposits):
        months = int(dep.maturity * 12)
        print(f"    {months}M: {dep.rate*100:.2f}%")

    # SOFR Futures (prices → implied rates)
    print("\n  SOFR Futures:")
    futures_prices = [94.70, 94.60, 94.50, 94.40, 94.35, 94.30, 94.25, 94.20]
    futures = []

    # Skip first IMM date if it's before 3M deposit end
    start_idx = 0
    deposit_3m_end = 0.25
    while start_idx < len(futures_schedule) and futures_schedule[start_idx][2] < deposit_3m_end:
        start_idx += 1

    for i in range(start_idx, len(futures_schedule)):
        code, imm_date, t_end = futures_schedule[i]
        price = futures_prices[i]

        # Start from previous IMM date or 3M deposit end
        if i > start_idx:
            t_start = futures_schedule[i - 1][2]
        else:
            t_start = deposit_3m_end

        convexity = 0.0001 * (i + 1)  # Convexity increases with maturity

        futures.append(
            Future(start=t_start, end=t_end, price=price, convexity_adjustment=convexity)
        )

        implied_rate = (100.0 - price) / 100.0
        forward_rate = implied_rate - convexity
        print(
            f"    {code}: Price={price:.2f}, Implied={implied_rate*100:.2f}%, "
            f"Convexity={convexity*10000:.1f}bps, Forward={forward_rate*100:.2f}%"
        )

    # OIS swaps for intermediate and long end
    print("\n  OIS Swaps:")
    ois_swaps = [
        OIS(
            fixed_rate=0.0555,
            payment_times=[1.0],
            accrual_factors=[1.0],
            compounding="compound",
        ),  # 1Y
        OIS(
            fixed_rate=0.0557,
            payment_times=[1.0, 2.0],
            accrual_factors=[1.0, 1.0],
            compounding="compound",
        ),  # 2Y
        OIS(
            fixed_rate=0.0560,
            payment_times=[1.0, 2.0, 3.0],
            accrual_factors=[1.0, 1.0, 1.0],
            compounding="compound",
        ),  # 3Y
        OIS(
            fixed_rate=0.0565,
            payment_times=[1.0, 2.0, 3.0, 4.0],
            accrual_factors=[1.0, 1.0, 1.0, 1.0],
            compounding="compound",
        ),  # 4Y
        OIS(
            fixed_rate=0.0570,
            payment_times=[1.0, 2.0, 3.0, 4.0, 5.0],
            accrual_factors=[1.0, 1.0, 1.0, 1.0, 1.0],
            compounding="compound",
        ),  # 5Y
    ]
    for ois in ois_swaps:
        maturity = int(max(ois.payment_times))
        print(f"    {maturity}Y: {ois.fixed_rate*100:.2f}%")

    # Step 3: Bootstrap the curve
    print("\n3. Bootstrapping Curve:")
    print("-" * 80)
    instruments = deposits + futures + ois_swaps
    curve = BootstrappedCurve(instruments)
    print(f"  ✓ Successfully bootstrapped {len(instruments)} instruments")
    print(f"    - {len(deposits)} Deposits")
    print(f"    - {len(futures)} Futures")
    print(f"    - {len(ois_swaps)} OIS Swaps")

    # Step 4: Display results
    print("\n4. Curve Results:")
    print("-" * 80)
    print(f"  {'Maturity':<12} {'Discount Factor':>18} {'Zero Rate':>12}")
    print("  " + "-" * 50)

    test_maturities = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    for t in test_maturities:
        df = float(curve.df(t))
        zr = float(curve.zero_rate(t))
        print(f"  {t:>6.2f}Y     {df:>18.10f}     {zr*100:>10.4f}%")

    # Step 5: Forward rates
    print("\n5. Forward Rates:")
    print("-" * 80)
    print(f"  {'Period':<20} {'Forward Rate':>15}")
    print("  " + "-" * 40)

    forward_periods = [
        (0.0, 0.25),
        (0.25, 0.5),
        (0.5, 1.0),
        (1.0, 2.0),
        (2.0, 3.0),
        (3.0, 5.0),
    ]
    for t0, t1 in forward_periods:
        fr = float(curve.forward_rate(t0, t1))
        print(f"  {t0:.2f}Y - {t1:.2f}Y     {fr*100:>13.4f}%")


def main():
    """Run all examples."""
    # IMM dates demonstration
    demonstrate_imm_dates()

    # Newton-Raphson solver demonstration
    demonstrate_newton_raphson()

    # Practical integration: futures curve with IMM dates
    demonstrate_futures_curve_with_imm()

    print("\n\n" + "=" * 80)
    print("Example Complete")
    print("=" * 80)
    print(
        """
Key Takeaways:

1. IMM Dates:
   - Standardized settlement dates (3rd Wednesday of Mar/Jun/Sep/Dec)
   - Essential for futures contract scheduling
   - Provides market liquidity concentration
   - Easy conversion between dates and contract codes (H/M/U/Z)

2. Newton-Raphson Solver:
   - Fast convergence for smooth functions
   - Requires good initial guess
   - JAX automatic differentiation eliminates need for manual derivatives
   - Critical for implied calculations (rates, volatilities, etc.)

3. Practical Integration:
   - Futures naturally align with IMM dates
   - Convexity adjustment increases with maturity
   - Smooth curve from cash → futures → swaps
   - Essential for building accurate forward rate curves

For more details, see:
  - src/neutryx/market/conventions.py (IMM date utilities)
  - src/neutryx/math/solvers.py (numerical solvers)
  - src/neutryx/market/curves.py (curve bootstrapping)
"""
    )


if __name__ == "__main__":
    main()
