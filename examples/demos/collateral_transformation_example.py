"""Example: Collateral Transformation Strategies for Margin Management

This example demonstrates how to use collateral transformation strategies to:
1. Optimize collateral selection from available inventory
2. Minimize transformation costs (haircuts, FX, funding)
3. Satisfy margin requirements with constraints
4. Handle multi-currency portfolios
5. Respect concentration limits

Scenario:
---------
A derivatives desk needs to post $5M in initial margin to a CCP. They have
a diverse collateral portfolio including cash (USD, EUR, GBP), government bonds,
and corporate bonds. The goal is to select the optimal mix that minimizes costs
while satisfying all eligibility and concentration constraints.

Use Cases:
----------
- Daily margin optimization for cleared derivatives
- Collateral allocation across multiple CCPs
- Cost-benefit analysis of collateral upgrades
- Stress testing collateral availability
"""

from neutryx.contracts.csa import CollateralType as CSACollateralType
from neutryx.contracts.csa import EligibleCollateral
from neutryx.valuations.margin.collateral_transformation import (
    CollateralHolding,
    CollateralPortfolio,
    GreedyLowestCostStrategy,
    OptimalMixStrategy,
    ConcentrationOptimizedStrategy,
)


# ==============================================================================
# Example 1: Basic Collateral Selection
# ==============================================================================


def example_basic_selection():
    """Example: Basic collateral selection with greedy strategy."""
    print("=" * 80)
    print("Example 1: Basic Collateral Selection")
    print("=" * 80)

    # Step 1: Create collateral portfolio
    portfolio = CollateralPortfolio(base_currency="USD")

    # Add USD cash (best option - no haircut, no FX)
    portfolio.add_holding(
        CollateralHolding(
            collateral_type=CSACollateralType.CASH,
            currency="USD",
            market_value=2_000_000.0,
            available=True,
        )
    )

    # Add US Treasury bonds (small haircut)
    portfolio.add_holding(
        CollateralHolding(
            collateral_type=CSACollateralType.GOVERNMENT_BOND,
            currency="USD",
            market_value=3_000_000.0,
            rating="AAA",
            maturity_years=5.0,
            issuer="US Treasury",
            available=True,
        )
    )

    # Add corporate bonds (higher haircut)
    portfolio.add_holding(
        CollateralHolding(
            collateral_type=CSACollateralType.CORPORATE_BOND,
            currency="USD",
            market_value=1_500_000.0,
            rating="A",
            maturity_years=7.0,
            issuer="Corporate Inc",
            available=True,
        )
    )

    print(f"\nPortfolio Summary:")
    print(f"Total Holdings: {len(portfolio.holdings)}")
    print(f"Total Market Value: ${portfolio.get_total_value():,.2f}")

    # Step 2: Define eligible collateral with haircuts
    eligible_collateral = [
        EligibleCollateral(
            collateral_type=CSACollateralType.CASH,
            haircut=0.0,  # No haircut for cash
        ),
        EligibleCollateral(
            collateral_type=CSACollateralType.GOVERNMENT_BOND,
            haircut=0.01,  # 1% haircut for govt bonds
            maturity_max_years=10.0,
        ),
        EligibleCollateral(
            collateral_type=CSACollateralType.CORPORATE_BOND,
            haircut=0.05,  # 5% haircut for corp bonds
            rating_threshold="BBB-",
            concentration_limit=0.30,  # Max 30% of total
        ),
    ]

    # Step 3: Set margin requirement
    margin_requirement = 4_000_000.0
    print(f"\nMargin Requirement: ${margin_requirement:,.2f}")

    # Step 4: Use greedy strategy to select collateral
    strategy = GreedyLowestCostStrategy(
        fx_spread=0.001,  # 10 bps FX spread
        funding_rate=0.05,  # 5% annual funding cost
        operational_cost_per_asset=100.0,  # $100 per asset
    )

    result = strategy.select_collateral(
        portfolio=portfolio,
        margin_requirement=margin_requirement,
        eligible_collateral=eligible_collateral,
    )

    # Step 5: Display results
    print(f"\n{result.summary()}")

    # Analysis
    print(f"\nAnalysis:")
    print(f"  - Strategy: Greedy Lowest Cost")
    print(f"  - Assets Selected: {len(result.selected_collateral)}")
    print(f"  - Requirement Satisfied: {result.satisfied}")
    print(f"  - Total Cost: ${result.total_cost.total_cost:,.2f}")
    print(f"  - Cost as % of Requirement: {result.total_cost.total_cost / margin_requirement * 100:.3f}%")


# ==============================================================================
# Example 2: Multi-Currency Optimization
# ==============================================================================


def example_multi_currency():
    """Example: Optimizing collateral across multiple currencies."""
    print("\n\n" + "=" * 80)
    print("Example 2: Multi-Currency Collateral Optimization")
    print("=" * 80)

    # Create portfolio with multi-currency holdings
    portfolio = CollateralPortfolio(
        base_currency="USD",
        fx_rates={
            "USD": 1.0,
            "EUR": 1.08,  # EUR/USD = 1.08
            "GBP": 1.27,  # GBP/USD = 1.27
            "JPY": 0.0067,  # USD/JPY = 150
        },
    )

    # Add cash in multiple currencies
    portfolio.add_holding(
        CollateralHolding(
            collateral_type=CSACollateralType.CASH,
            currency="USD",
            market_value=1_000_000.0,
        )
    )

    portfolio.add_holding(
        CollateralHolding(
            collateral_type=CSACollateralType.CASH,
            currency="EUR",
            market_value=2_000_000.0,  # ~$2.16M equivalent
        )
    )

    portfolio.add_holding(
        CollateralHolding(
            collateral_type=CSACollateralType.CASH,
            currency="GBP",
            market_value=1_500_000.0,  # ~$1.91M equivalent
        )
    )

    portfolio.add_holding(
        CollateralHolding(
            collateral_type=CSACollateralType.CASH,
            currency="JPY",
            market_value=500_000_000.0,  # ~$3.35M equivalent
        )
    )

    print(f"\nMulti-Currency Portfolio:")
    print(f"Total Value (USD): ${portfolio.get_total_value():,.2f}")

    # Eligible: Cash in any currency
    eligible_collateral = [
        EligibleCollateral(collateral_type=CSACollateralType.CASH, haircut=0.0)
    ]

    margin_requirement = 3_000_000.0

    # Compare strategies
    print(f"\nMargin Requirement: ${margin_requirement:,.2f}")
    print(f"\nComparing Strategies:")
    print("-" * 80)

    # Greedy strategy
    greedy = GreedyLowestCostStrategy(fx_spread=0.0015)  # 15 bps FX spread
    greedy_result = greedy.select_collateral(
        portfolio, margin_requirement, eligible_collateral
    )

    print(f"\n1. Greedy Strategy:")
    print(f"   Total Cost: ${greedy_result.total_cost.total_cost:,.2f}")
    print(f"   FX Cost: ${greedy_result.total_cost.fx_conversion_cost:,.2f}")
    print(f"   Currencies Used:")
    for selection in greedy_result.selected_collateral:
        print(f"      {selection.holding.currency}: ${selection.amount:,.2f}")

    # Optimal strategy
    try:
        import scipy

        optimal = OptimalMixStrategy(fx_spread=0.0015)
        optimal_result = optimal.select_collateral(
            portfolio, margin_requirement, eligible_collateral
        )

        print(f"\n2. Optimal Mix Strategy:")
        print(f"   Total Cost: ${optimal_result.total_cost.total_cost:,.2f}")
        print(f"   FX Cost: ${optimal_result.total_cost.fx_conversion_cost:,.2f}")
        print(f"   Currencies Used:")
        for selection in optimal_result.selected_collateral:
            print(f"      {selection.holding.currency}: ${selection.amount:,.2f}")

        savings = greedy_result.total_cost.total_cost - optimal_result.total_cost.total_cost
        print(f"\n   Optimization Savings: ${savings:,.2f}")

    except ImportError:
        print("\n2. Optimal Mix Strategy: (scipy required)")


# ==============================================================================
# Example 3: Concentration Limits
# ==============================================================================


def example_concentration_limits():
    """Example: Respecting concentration limits."""
    print("\n\n" + "=" * 80)
    print("Example 3: Concentration Limits")
    print("=" * 80)

    portfolio = CollateralPortfolio(base_currency="USD")

    # Add lots of corporate bonds (attractive but limited by concentration)
    for i in range(10):
        portfolio.add_holding(
            CollateralHolding(
                collateral_type=CSACollateralType.CORPORATE_BOND,
                currency="USD",
                market_value=500_000.0,
                rating="A",
                maturity_years=5.0,
                issuer=f"Corporation {i}",
            )
        )

    # Add government bonds
    for i in range(5):
        portfolio.add_holding(
            CollateralHolding(
                collateral_type=CSACollateralType.GOVERNMENT_BOND,
                currency="USD",
                market_value=1_000_000.0,
                maturity_years=5.0,
            )
        )

    # Add some cash
    portfolio.add_holding(
        CollateralHolding(
            collateral_type=CSACollateralType.CASH,
            currency="USD",
            market_value=2_000_000.0,
        )
    )

    print(f"\nPortfolio Composition:")
    print(f"  Corporate Bonds: ${5_000_000.0:,.2f} (10 assets)")
    print(f"  Government Bonds: ${5_000_000.0:,.2f} (5 assets)")
    print(f"  Cash: ${2_000_000.0:,.2f}")

    # Eligibility with concentration limits
    eligible_collateral = [
        EligibleCollateral(
            collateral_type=CSACollateralType.CORPORATE_BOND,
            haircut=0.04,
            concentration_limit=0.25,  # Max 25% corporate bonds
        ),
        EligibleCollateral(
            collateral_type=CSACollateralType.GOVERNMENT_BOND,
            haircut=0.01,
        ),
        EligibleCollateral(
            collateral_type=CSACollateralType.CASH,
            haircut=0.0,
        ),
    ]

    margin_requirement = 6_000_000.0

    print(f"\nMargin Requirement: ${margin_requirement:,.2f}")
    print(f"Concentration Limit: Corporate bonds max 25%")

    try:
        import scipy

        strategy = ConcentrationOptimizedStrategy()
        result = strategy.select_collateral(
            portfolio, margin_requirement, eligible_collateral
        )

        print(f"\n{result.summary()}")

        # Verify concentration
        concentration = result.get_concentration_ratios()
        print(f"\nConcentration Verification:")
        for ctype, ratio in concentration.items():
            status = "✓" if ratio <= 0.26 else "✗"  # Small tolerance
            print(f"  {status} {ctype.value}: {ratio*100:.2f}%")

    except ImportError:
        print("\nConcentration optimization requires scipy")


# ==============================================================================
# Example 4: Cost Breakdown Analysis
# ==============================================================================


def example_cost_analysis():
    """Example: Detailed cost breakdown and sensitivity."""
    print("\n\n" + "=" * 80)
    print("Example 4: Cost Breakdown and Sensitivity Analysis")
    print("=" * 80)

    portfolio = CollateralPortfolio(
        base_currency="USD",
        fx_rates={"USD": 1.0, "EUR": 1.10},
    )

    # Diverse portfolio
    portfolio.add_holding(
        CollateralHolding(
            collateral_type=CSACollateralType.CASH,
            currency="USD",
            market_value=2_000_000.0,
        )
    )

    portfolio.add_holding(
        CollateralHolding(
            collateral_type=CSACollateralType.CASH,
            currency="EUR",
            market_value=2_000_000.0,
        )
    )

    portfolio.add_holding(
        CollateralHolding(
            collateral_type=CSACollateralType.GOVERNMENT_BOND,
            currency="USD",
            market_value=3_000_000.0,
            maturity_years=7.0,
        )
    )

    portfolio.add_holding(
        CollateralHolding(
            collateral_type=CSACollateralType.CORPORATE_BOND,
            currency="USD",
            market_value=2_000_000.0,
            rating="BBB+",
            maturity_years=10.0,
        )
    )

    eligible_collateral = [
        EligibleCollateral(collateral_type=CSACollateralType.CASH, haircut=0.0),
        EligibleCollateral(
            collateral_type=CSACollateralType.GOVERNMENT_BOND, haircut=0.02
        ),
        EligibleCollateral(
            collateral_type=CSACollateralType.CORPORATE_BOND, haircut=0.08
        ),
    ]

    margin_requirement = 5_000_000.0

    # Test different funding rate scenarios
    funding_rates = [0.03, 0.05, 0.07]  # 3%, 5%, 7%

    print(f"\nCost Sensitivity to Funding Rates:")
    print(f"Margin Requirement: ${margin_requirement:,.2f}")
    print("\n" + f"{'Funding Rate':<15} {'Total Cost':<15} {'Haircut':<15} {'FX':<15} {'Funding':<15}")
    print("-" * 80)

    for rate in funding_rates:
        strategy = GreedyLowestCostStrategy(
            fx_spread=0.001,
            funding_rate=rate,
            operational_cost_per_asset=100.0,
        )

        result = strategy.select_collateral(
            portfolio, margin_requirement, eligible_collateral
        )

        print(
            f"{rate*100:>6.1f}%         "
            f"${result.total_cost.total_cost:>12,.2f}  "
            f"${result.total_cost.haircut_cost:>12,.2f}  "
            f"${result.total_cost.fx_conversion_cost:>12,.2f}  "
            f"${result.total_cost.funding_cost:>12,.2f}"
        )

    # Cost components analysis
    print(f"\n\nCost Component Analysis (5% funding rate):")
    strategy = GreedyLowestCostStrategy(fx_spread=0.001, funding_rate=0.05)
    result = strategy.select_collateral(
        portfolio, margin_requirement, eligible_collateral
    )

    total = result.total_cost.total_cost
    print(f"\nTotal Cost: ${total:,.2f}")
    print(f"\nBreakdown:")
    print(f"  Haircut Cost:      ${result.total_cost.haircut_cost:>12,.2f}  ({result.total_cost.haircut_cost/total*100:>5.1f}%)")
    print(f"  FX Conversion:     ${result.total_cost.fx_conversion_cost:>12,.2f}  ({result.total_cost.fx_conversion_cost/total*100:>5.1f}%)")
    print(f"  Funding Cost:      ${result.total_cost.funding_cost:>12,.2f}  ({result.total_cost.funding_cost/total*100:>5.1f}%)")
    print(f"  Operational Cost:  ${result.total_cost.operational_cost:>12,.2f}  ({result.total_cost.operational_cost/total*100:>5.1f}%)")


# ==============================================================================
# Example 5: Collateral Upgrade/Downgrade
# ==============================================================================


def example_collateral_upgrade():
    """Example: Analyzing collateral upgrade opportunities."""
    print("\n\n" + "=" * 80)
    print("Example 5: Collateral Upgrade Analysis")
    print("=" * 80)

    print("\nScenario: Currently posting corporate bonds, considering upgrade to cash")

    # Current: Posted corporate bonds
    current_posted = CollateralHolding(
        collateral_type=CSACollateralType.CORPORATE_BOND,
        currency="USD",
        market_value=6_000_000.0,  # Need $6M due to 5% haircut to cover $5.7M requirement
        rating="A",
    )

    # Alternative: Post cash
    alternative_cash = CollateralHolding(
        collateral_type=CSACollateralType.CASH,
        currency="USD",
        market_value=5_700_000.0,  # No haircut
    )

    # Cost comparison
    haircut_cost_current = 6_000_000.0 * 0.05
    haircut_cost_alternative = 0.0

    funding_rate = 0.05
    funding_cost_current = 6_000_000.0 * funding_rate / 365  # Daily
    funding_cost_alternative = 5_700_000.0 * funding_rate / 365

    print(f"\nCurrent (Corporate Bonds):")
    print(f"  Market Value Posted: ${6_000_000.0:,.2f}")
    print(f"  Haircut Cost: ${haircut_cost_current:,.2f}")
    print(f"  Daily Funding Cost: ${funding_cost_current:,.2f}")
    print(f"  Total Daily Cost: ${haircut_cost_current + funding_cost_current:,.2f}")

    print(f"\nAlternative (Cash):")
    print(f"  Market Value Posted: ${5_700_000.0:,.2f}")
    print(f"  Haircut Cost: ${haircut_cost_alternative:,.2f}")
    print(f"  Daily Funding Cost: ${funding_cost_alternative:,.2f}")
    print(f"  Total Daily Cost: ${haircut_cost_alternative + funding_cost_alternative:,.2f}")

    daily_savings = (haircut_cost_current + funding_cost_current) - (
        haircut_cost_alternative + funding_cost_alternative
    )
    annual_savings = daily_savings * 250  # Trading days

    print(f"\nSavings Analysis:")
    print(f"  Daily Savings: ${daily_savings:,.2f}")
    print(f"  Annual Savings (250 days): ${annual_savings:,.2f}")
    print(f"  Break-even Days: {300_000 / daily_savings if daily_savings > 0 else 'N/A':.0f}")

    if daily_savings > 0:
        print(f"\n✓ Recommendation: Upgrade to cash collateral")
        print(f"  - Immediate haircut savings of ${haircut_cost_current:,.2f}")
        print(f"  - Lower funding costs save ${funding_cost_current - funding_cost_alternative:,.2f}/day")
    else:
        print(f"\n→ Current collateral is cost-effective")


# ==============================================================================
# Run All Examples
# ==============================================================================


if __name__ == "__main__":
    print("\n")
    print("=" * 80)
    print(" COLLATERAL TRANSFORMATION STRATEGIES - COMPREHENSIVE EXAMPLES")
    print("=" * 80)

    example_basic_selection()
    example_multi_currency()
    example_concentration_limits()
    example_cost_analysis()
    example_collateral_upgrade()

    print("\n\n" + "=" * 80)
    print("Examples Completed!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. Greedy strategy provides fast, reasonable solutions")
    print("  2. Optimal mix minimizes cost via linear programming")
    print("  3. Multi-currency optimization reduces FX costs")
    print("  4. Concentration limits ensure diversification")
    print("  5. Cost analysis informs collateral upgrade decisions")
    print("\nNext Steps:")
    print("  - Integrate with margin calculation workflows")
    print("  - Connect to CCP submission systems")
    print("  - Implement real-time collateral monitoring")
    print("  - Add stress testing scenarios")
