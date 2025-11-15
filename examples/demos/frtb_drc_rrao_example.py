"""Example: FRTB Default Risk Charge (DRC) and Residual Risk Add-On (RRAO)

This example demonstrates:
1. DRC calculation for credit portfolios
2. DRC for securitized products
3. RRAO for exotic underlyings
4. RRAO for exotic payoffs
5. Combined capital charge calculation

Scenario:
---------
A trading desk manages a portfolio containing:
- Corporate bonds and CDS (credit risk → DRC)
- Securitized products (RMBS, CDO → DRC)
- Exotic derivatives (longevity swaps, weather → RRAO)
- Structured products with complex payoffs → RRAO

The goal is to calculate regulatory capital under FRTB's DRC and RRAO frameworks.
"""

from neutryx.valuations.regulatory.frtb_drc import (
    CreditRating,
    DefaultExposure,
    FRTBDefaultRiskCharge,
    SecuritizedExposure,
    Sector,
    SecuritizationType,
    Seniority,
)
from neutryx.valuations.regulatory.frtb_rrao import (
    ExoticUnderlying,
    FRTBResidualRiskAddOn,
    LiquidityClass,
    PayoffComplexity,
    RRAOExposure,
    classify_payoff_complexity,
    estimate_hedge_effectiveness,
)


# ==============================================================================
# Example 1: Basic DRC for Corporate Bonds
# ==============================================================================


def example_basic_drc():
    """Example: DRC for a simple corporate bond portfolio."""
    print("=" * 80)
    print("Example 1: Default Risk Charge for Corporate Bonds")
    print("=" * 80)

    # Create DRC calculator
    drc_calculator = FRTBDefaultRiskCharge()

    # Build portfolio of corporate bonds
    exposures = [
        # Investment grade bonds
        DefaultExposure(
            issuer_id="CORP_A",
            instrument_type="corporate_bond",
            notional=10_000_000.0,
            credit_rating=CreditRating.AA,
            seniority=Seniority.SENIOR_UNSECURED,
            sector=Sector.FINANCIAL,
            maturity_years=5.0,
            long_short="long",
        ),
        DefaultExposure(
            issuer_id="CORP_B",
            instrument_type="corporate_bond",
            notional=8_000_000.0,
            credit_rating=CreditRating.A,
            seniority=Seniority.SENIOR_UNSECURED,
            sector=Sector.INDUSTRIAL,
            maturity_years=7.0,
            long_short="long",
        ),
        # High yield bond
        DefaultExposure(
            issuer_id="CORP_C",
            instrument_type="corporate_bond",
            notional=5_000_000.0,
            credit_rating=CreditRating.BB,
            seniority=Seniority.SUBORDINATED,
            sector=Sector.ENERGY,
            maturity_years=10.0,
            long_short="long",
        ),
    ]

    print(f"\nPortfolio Summary:")
    print(f"  Number of Issuers: {len(set(exp.issuer_id for exp in exposures))}")
    print(f"  Total Notional: ${sum(exp.notional for exp in exposures):,.2f}")

    # Calculate DRC
    result = drc_calculator.calculate(non_securitized=exposures)

    print(f"\nDRC Results:")
    print(f"  Total DRC: ${result.total_drc:,.2f}")
    print(f"  Non-Securitized DRC: ${result.non_securitized_drc:,.2f}")
    print(f"  Net Long JTD: ${result.net_long_jtd:,.2f}")
    print(f"  Net Short JTD: ${result.net_short_jtd:,.2f}")

    print(f"\nDRC by Issuer:")
    for issuer, drc in sorted(result.drc_by_issuer.items()):
        print(f"  {issuer}: ${drc:,.2f}")

    print(f"\nDRC by Sector:")
    for sector, drc in sorted(result.drc_by_sector.items(), key=lambda x: x[1], reverse=True):
        print(f"  {sector.value}: ${drc:,.2f}")


# ==============================================================================
# Example 2: DRC with Hedging (Long/Short)
# ==============================================================================


def example_drc_hedging():
    """Example: DRC with CDS hedges."""
    print("\n\n" + "=" * 80)
    print("Example 2: DRC with CDS Hedging")
    print("=" * 80)

    drc_calculator = FRTBDefaultRiskCharge()

    # Portfolio: Long bonds hedged with short CDS
    exposures = [
        # Long bond position
        DefaultExposure(
            issuer_id="CORP_D",
            instrument_type="corporate_bond",
            notional=20_000_000.0,
            credit_rating=CreditRating.BBB,
            seniority=Seniority.SENIOR_UNSECURED,
            sector=Sector.TECHNOLOGY,
            maturity_years=5.0,
            long_short="long",
        ),
        # CDS hedge (short credit)
        DefaultExposure(
            issuer_id="CORP_D",
            instrument_type="CDS",
            notional=15_000_000.0,  # 75% hedge
            credit_rating=CreditRating.BBB,
            seniority=Seniority.SENIOR_UNSECURED,
            sector=Sector.TECHNOLOGY,
            maturity_years=5.0,
            long_short="short",
        ),
    ]

    print(f"\nHedging Strategy:")
    print(f"  Long Bond: ${20_000_000.0:,.2f}")
    print(f"  Short CDS: ${15_000_000.0:,.2f}")
    print(f"  Hedge Ratio: 75%")

    result = drc_calculator.calculate(non_securitized=exposures)

    print(f"\nDRC with Hedging:")
    print(f"  Total DRC: ${result.total_drc:,.2f}")
    print(f"  Net Long JTD: ${result.net_long_jtd:,.2f}")
    print(f"  Net Short JTD: ${result.net_short_jtd:,.2f}")

    # Compare to unhedged
    unhedged_exp = [exposures[0]]  # Only bond
    unhedged_result = drc_calculator.calculate(non_securitized=unhedged_exp)

    print(f"\nHedging Benefit:")
    print(f"  Unhedged DRC: ${unhedged_result.total_drc:,.2f}")
    print(f"  Hedged DRC: ${result.total_drc:,.2f}")
    print(f"  Capital Savings: ${unhedged_result.total_drc - result.total_drc:,.2f}")
    print(
        f"  Reduction: {(1 - result.total_drc / unhedged_result.total_drc) * 100:.1f}%"
    )


# ==============================================================================
# Example 3: DRC for Securitized Products
# ==============================================================================


def example_securitized_drc():
    """Example: DRC for securitized products."""
    print("\n\n" + "=" * 80)
    print("Example 3: DRC for Securitized Products")
    print("=" * 80)

    drc_calculator = FRTBDefaultRiskCharge()

    # Portfolio of securitizations
    securitized = [
        # RMBS senior tranche
        SecuritizedExposure(
            instrument_id="RMBS_2024_A",
            securitization_type=SecuritizationType.RMBS,
            notional=50_000_000.0,
            tranche_attachment=0.10,
            tranche_detachment=0.20,
            credit_rating=CreditRating.AAA,
            underlying_pool_rating=CreditRating.BBB,
            long_short="long",
        ),
        # CLO mezzanine tranche
        SecuritizedExposure(
            instrument_id="CLO_2024_B",
            securitization_type=SecuritizationType.CLO,
            notional=30_000_000.0,
            tranche_attachment=0.05,
            tranche_detachment=0.15,
            credit_rating=CreditRating.AA,
            underlying_pool_rating=CreditRating.BB,
            long_short="long",
        ),
        # ABS equity tranche (high risk)
        SecuritizedExposure(
            instrument_id="ABS_2024_C",
            securitization_type=SecuritizationType.ABS,
            notional=10_000_000.0,
            tranche_attachment=0.00,
            tranche_detachment=0.05,
            credit_rating=CreditRating.B,
            underlying_pool_rating=CreditRating.BBB,
            long_short="long",
        ),
    ]

    print(f"\nSecuritized Portfolio:")
    for sec in securitized:
        print(
            f"  {sec.instrument_id}: ${sec.notional:,.2f} "
            f"({sec.tranche_attachment*100:.0f}%-{sec.tranche_detachment*100:.0f}% tranche)"
        )

    result = drc_calculator.calculate(non_securitized=[], securitized=securitized)

    print(f"\nSecuritized DRC:")
    print(f"  Total DRC: ${result.total_drc:,.2f}")
    print(f"  Securitized Component: ${result.securitized_drc:,.2f}")
    print(f"  DRC as % of Notional: {result.total_drc / 90_000_000.0 * 100:.2f}%")


# ==============================================================================
# Example 4: Basic RRAO for Exotic Underlyings
# ==============================================================================


def example_basic_rrao():
    """Example: RRAO for exotic underlyings."""
    print("\n\n" + "=" * 80)
    print("Example 4: Residual Risk Add-On for Exotic Underlyings")
    print("=" * 80)

    rrao_calculator = FRTBResidualRiskAddOn()

    # Portfolio of exotic derivatives
    exposures = [
        # Longevity swap
        RRAOExposure(
            instrument_id="LONGEVITY_2024_A",
            instrument_type="longevity_swap",
            underlying_type=ExoticUnderlying.LONGEVITY,
            payoff_complexity=PayoffComplexity.EXOTIC_HIGH,
            liquidity_class=LiquidityClass.VERY_ILLIQUID,
            notional=100_000_000.0,
            tenor_years=20.0,
            is_hedged=False,
        ),
        # Weather derivative
        RRAOExposure(
            instrument_id="WEATHER_2024_B",
            instrument_type="temperature_derivative",
            underlying_type=ExoticUnderlying.WEATHER,
            payoff_complexity=PayoffComplexity.EXOTIC_MEDIUM,
            liquidity_class=LiquidityClass.ILLIQUID,
            notional=25_000_000.0,
            tenor_years=5.0,
            is_hedged=False,
        ),
        # Catastrophe bond
        RRAOExposure(
            instrument_id="CAT_2024_C",
            instrument_type="catastrophe_bond",
            underlying_type=ExoticUnderlying.NATURAL_CATASTROPHE,
            payoff_complexity=PayoffComplexity.EXOTIC_LOW,
            liquidity_class=LiquidityClass.VERY_ILLIQUID,
            notional=50_000_000.0,
            tenor_years=3.0,
            is_hedged=False,
        ),
    ]

    print(f"\nExotic Portfolio:")
    for exp in exposures:
        print(
            f"  {exp.instrument_id}: ${exp.notional:,.2f} "
            f"({exp.underlying_type.value}, {exp.tenor_years}Y)"
        )

    result = rrao_calculator.calculate(exposures)

    print(f"\nRRAO Results:")
    print(f"  Total RRAO: ${result.total_rrao:,.2f}")
    print(f"  Gross Notional: ${result.gross_notional:,.2f}")
    print(f"  Net Notional: ${result.net_notional:,.2f}")
    print(f"  RRAO as % of Notional: {result.total_rrao / result.gross_notional * 100:.2f}%")

    print(f"\nRRAO by Underlying Type:")
    for underlying, rrao in sorted(
        result.rrao_by_underlying.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {underlying.value}: ${rrao:,.2f}")

    print(f"\nRRAO by Complexity:")
    for complexity, rrao in sorted(
        result.rrao_by_complexity.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {complexity.value}: ${rrao:,.2f}")


# ==============================================================================
# Example 5: RRAO with Hedging
# ==============================================================================


def example_rrao_hedging():
    """Example: RRAO reduction through hedging."""
    print("\n\n" + "=" * 80)
    print("Example 5: RRAO with Hedging")
    print("=" * 80)

    rrao_calculator = FRTBResidualRiskAddOn()

    # Unhedged position
    unhedged = RRAOExposure(
        instrument_id="FREIGHT_2024_A",
        instrument_type="freight_derivative",
        underlying_type=ExoticUnderlying.FREIGHT,
        payoff_complexity=PayoffComplexity.EXOTIC_MEDIUM,
        liquidity_class=LiquidityClass.MODERATELY_LIQUID,
        notional=30_000_000.0,
        tenor_years=3.0,
        is_hedged=False,
    )

    # Hedged position (same parameters, different hedge effectiveness levels)
    hedge_scenarios = [
        ("No Hedge", 0.0),
        ("Partial Hedge (40%)", 0.40),
        ("Good Hedge (70%)", 0.70),
        ("Excellent Hedge (90%)", 0.90),
    ]

    print(f"\nPosition: {unhedged.instrument_id}")
    print(f"  Notional: ${unhedged.notional:,.2f}")
    print(f"  Underlying: {unhedged.underlying_type.value}")
    print(f"  Complexity: {unhedged.payoff_complexity.value}")

    print(f"\nRRAO by Hedge Effectiveness:")
    print(f"{'Scenario':<25} {'RRAO':<15} {'Reduction':<15}")
    print("-" * 80)

    base_rrao = None
    for scenario_name, effectiveness in hedge_scenarios:
        hedged = RRAOExposure(
            instrument_id=unhedged.instrument_id,
            instrument_type=unhedged.instrument_type,
            underlying_type=unhedged.underlying_type,
            payoff_complexity=unhedged.payoff_complexity,
            liquidity_class=unhedged.liquidity_class,
            notional=unhedged.notional,
            tenor_years=unhedged.tenor_years,
            is_hedged=effectiveness > 0,
            hedge_effectiveness=effectiveness,
        )

        result = rrao_calculator.calculate([hedged])

        if base_rrao is None:
            base_rrao = result.total_rrao

        reduction = (1 - result.total_rrao / base_rrao) * 100 if base_rrao > 0 else 0

        print(f"{scenario_name:<25} ${result.total_rrao:>12,.2f}  {reduction:>12.1f}%")


# ==============================================================================
# Example 6: Combined DRC and RRAO Capital
# ==============================================================================


def example_combined_capital():
    """Example: Total capital charge combining DRC and RRAO."""
    print("\n\n" + "=" * 80)
    print("Example 6: Combined DRC and RRAO Capital Charge")
    print("=" * 80)

    print("\nScenario: Trading book with credit risk and exotic underlyings")

    # DRC Calculator
    drc_calc = FRTBDefaultRiskCharge()

    # Credit portfolio
    credit_exposures = [
        DefaultExposure(
            issuer_id=f"ISSUER_{i}",
            instrument_type="bond",
            notional=10_000_000.0,
            credit_rating=CreditRating.A if i < 3 else CreditRating.BBB,
            seniority=Seniority.SENIOR_UNSECURED,
            sector=Sector.INDUSTRIAL if i % 2 == 0 else Sector.FINANCIAL,
            maturity_years=5.0,
        )
        for i in range(6)
    ]

    # RRAO Calculator
    rrao_calc = FRTBResidualRiskAddOn()

    # Exotic portfolio
    exotic_exposures = [
        RRAOExposure(
            instrument_id="LONGEVITY_MAIN",
            instrument_type="longevity_swap",
            underlying_type=ExoticUnderlying.LONGEVITY,
            payoff_complexity=PayoffComplexity.EXOTIC_VERY_HIGH,
            liquidity_class=LiquidityClass.VERY_ILLIQUID,
            notional=50_000_000.0,
            tenor_years=15.0,
        ),
        RRAOExposure(
            instrument_id="WEATHER_MAIN",
            instrument_type="weather_derivative",
            underlying_type=ExoticUnderlying.WEATHER,
            payoff_complexity=PayoffComplexity.EXOTIC_MEDIUM,
            liquidity_class=LiquidityClass.ILLIQUID,
            notional=20_000_000.0,
            tenor_years=5.0,
        ),
    ]

    # Calculate components
    drc_result = drc_calc.calculate(non_securitized=credit_exposures)
    rrao_result = rrao_calc.calculate(exotic_exposures)

    print(f"\nCredit Portfolio:")
    print(f"  Number of Issuers: {len(set(exp.issuer_id for exp in credit_exposures))}")
    print(f"  Total Notional: ${sum(exp.notional for exp in credit_exposures):,.2f}")
    print(f"  DRC: ${drc_result.total_drc:,.2f}")

    print(f"\nExotic Portfolio:")
    print(f"  Number of Instruments: {len(exotic_exposures)}")
    print(f"  Total Notional: ${sum(exp.notional for exp in exotic_exposures):,.2f}")
    print(f"  RRAO: ${rrao_result.total_rrao:,.2f}")

    # Total capital
    total_capital = drc_result.total_drc + rrao_result.total_rrao

    print(f"\n" + "=" * 80)
    print(f"TOTAL MARKET RISK CAPITAL (DRC + RRAO)")
    print(f"=" * 80)
    print(f"  DRC Component: ${drc_result.total_drc:>18,.2f}")
    print(f"  RRAO Component: ${rrao_result.total_rrao:>17,.2f}")
    print(f"  {'-' * 80}")
    print(f"  TOTAL CAPITAL: ${total_capital:>18,.2f}")
    print(f"\nNote: This is in addition to standard FRTB delta/vega/curvature charges")


# ==============================================================================
# Run All Examples
# ==============================================================================


if __name__ == "__main__":
    print("\n")
    print("=" * 80)
    print(" FRTB DEFAULT RISK CHARGE (DRC) AND RESIDUAL RISK ADD-ON (RRAO)")
    print(" COMPREHENSIVE EXAMPLES")
    print("=" * 80)

    example_basic_drc()
    example_drc_hedging()
    example_securitized_drc()
    example_basic_rrao()
    example_rrao_hedging()
    example_combined_capital()

    print("\n\n" + "=" * 80)
    print("Examples Completed!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. DRC captures jump-to-default risk for credit-sensitive instruments")
    print("  2. Correlation reduces capital through diversification")
    print("  3. Hedging (CDS) provides partial netting benefit")
    print("  4. Securitized products have separate DRC treatment")
    print("  5. RRAO applies to exotic underlyings and complex payoffs")
    print("  6. Hedge effectiveness reduces RRAO charge")
    print("  7. Combined DRC + RRAO adds to standard FRTB capital")
    print("\nRegulatory References:")
    print("  - BCBS d352: Minimum capital requirements for market risk")
    print("  - MAR21: Standardised approach - DRC and RRAO")
    print("  - MAR22: Internal models approach")
