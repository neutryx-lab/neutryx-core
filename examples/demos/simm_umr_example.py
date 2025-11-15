"""Comprehensive SIMM and UMR compliance example.

This example demonstrates the complete workflow for calculating initial margin
using ISDA SIMM and applying BCBS-IOSCO UMR compliance rules:

1. Define a portfolio of derivatives trades
2. Calculate risk factor sensitivities
3. Compute SIMM initial margin
4. Apply UMR thresholds and rules
5. Generate margin calls
6. Manage collateral and CSA terms
7. Produce regulatory reports

This integrates:
- SIMM calculator (neutryx.valuations.simm)
- UMR compliance (neutryx.valuations.margin.umr_compliance)
- Sensitivity analysis (neutryx.risk.sensitivity_analysis)
"""

from datetime import date, timedelta

import jax.numpy as jnp

# SIMM imports
from neutryx.valuations.simm.calculator import calculate_simm
from neutryx.valuations.simm.risk_weights import RiskClass
from neutryx.valuations.simm.sensitivities import (
    RiskFactorSensitivity,
    RiskFactorType,
    SensitivityType,
)

# UMR imports
from neutryx.valuations.margin import (
    AANACalculation,
    CollateralType,
    CSAManager,
    CSAPortfolio,
    CSATerms,
    CustodianAccount,
    CustodianInterface,
    MarginType,
    UMRComplianceChecker,
    UMRPhase,
    generate_margin_report,
)


# ==============================================================================
# Example 1: Basic SIMM Calculation for Interest Rate Swaps
# ==============================================================================


def example_1_ir_swap_simm():
    """Calculate SIMM for a portfolio of interest rate swaps."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: SIMM for Interest Rate Swap Portfolio")
    print("=" * 80)

    # Define IR delta sensitivities for a USD IRS portfolio
    # Portfolio: $100MM 5Y receiver, $50MM 10Y payer
    # Note: For IR, bucket = currency (e.g., "USD", "EUR")
    sensitivities = [
        # 5Y receiver swap - positive DV01s
        RiskFactorSensitivity(
            risk_factor_type=RiskFactorType.IR,
            sensitivity_type=SensitivityType.DELTA,
            bucket="USD",  # Currency for IR
            risk_factor="USD-OIS",
            sensitivity=50000,  # $50k DV01 at 1Y
            tenor="1Y",
        ),
        RiskFactorSensitivity(
            risk_factor_type=RiskFactorType.IR,
            sensitivity_type=SensitivityType.DELTA,
            bucket="USD",
            risk_factor="USD-OIS",
            sensitivity=100000,  # $100k DV01 at 5Y
            tenor="5Y",
        ),
        # 10Y payer swap - negative DV01s
        RiskFactorSensitivity(
            risk_factor_type=RiskFactorType.IR,
            sensitivity_type=SensitivityType.DELTA,
            bucket="USD",
            risk_factor="USD-OIS",
            sensitivity=-30000,  # -$30k DV01 at 2Y
            tenor="2Y",
        ),
        RiskFactorSensitivity(
            risk_factor_type=RiskFactorType.IR,
            sensitivity_type=SensitivityType.DELTA,
            bucket="USD",
            risk_factor="USD-OIS",
            sensitivity=-50000,  # -$50k DV01 at 10Y
            tenor="10Y",
        ),
    ]

    # Calculate SIMM
    result = calculate_simm(sensitivities)

    print(f"\nSIMM Calculation Results:")
    print(f"  Total IM:        ${result.total_im:>15,.0f}")
    print(f"  Delta IM:        ${result.delta_im:>15,.0f}")
    print(f"  Vega IM:         ${result.vega_im:>15,.0f}")
    print(f"  Curvature IM:    ${result.curvature_im:>15,.0f}")
    print(f"\nIM by Risk Class:")
    for risk_class, im in result.im_by_risk_class.items():
        print(f"  {risk_class.value:20s} ${im:>15,.0f}")

    return result


# ==============================================================================
# Example 2: Multi-Asset Portfolio SIMM
# ==============================================================================


def example_2_multi_asset_simm():
    """Calculate SIMM for a multi-asset class portfolio."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Multi-Asset Class Portfolio SIMM")
    print("=" * 80)

    sensitivities = []

    # Interest Rate: USD swaps
    sensitivities.extend([
        RiskFactorSensitivity(
            risk_factor_type=RiskFactorType.IR,
            sensitivity_type=SensitivityType.DELTA,
            bucket="USD",
            risk_factor="USD-OIS",
            sensitivity=75000,
            tenor="5Y",
        ),
        RiskFactorSensitivity(
            risk_factor_type=RiskFactorType.IR,
            sensitivity_type=SensitivityType.VEGA,
            bucket="USD",
            risk_factor="USD-OIS",
            sensitivity=5000,  # Vega for swaptions
            tenor="5Y",
        ),
    ])

    # FX: EUR/USD options
    sensitivities.extend([
        RiskFactorSensitivity(
            risk_factor_type=RiskFactorType.FX,
            sensitivity_type=SensitivityType.DELTA,
            bucket="1",
            risk_factor="EURUSD",
            sensitivity=1_000_000,  # $1MM delta
            tenor=None,
        ),
        RiskFactorSensitivity(
            risk_factor_type=RiskFactorType.FX,
            sensitivity_type=SensitivityType.VEGA,
            bucket="1",
            risk_factor="EURUSD",
            sensitivity=50000,  # $50k vega
            tenor="1Y",
        ),
    ])

    # Credit: CDS on investment grade names
    sensitivities.extend([
        RiskFactorSensitivity(
            risk_factor_type=RiskFactorType.CREDIT_Q,
            sensitivity_type=SensitivityType.DELTA,
            bucket="1",  # IG Sovereigns
            risk_factor="CORP-A-RATED",
            sensitivity=25000,  # $25k CS01
            tenor="5Y",
        ),
    ])

    # Equity: Index options
    sensitivities.extend([
        RiskFactorSensitivity(
            risk_factor_type=RiskFactorType.EQUITY,
            sensitivity_type=SensitivityType.DELTA,
            bucket="1",  # Large cap developed
            risk_factor="SPX",
            sensitivity=500_000,  # $500k delta
            tenor=None,
        ),
        RiskFactorSensitivity(
            risk_factor_type=RiskFactorType.EQUITY,
            sensitivity_type=SensitivityType.VEGA,
            bucket="1",
            risk_factor="SPX",
            sensitivity=75000,  # $75k vega
            tenor="1Y",
        ),
    ])

    # Calculate SIMM
    result = calculate_simm(sensitivities, product_class_multiplier=1.0)

    print(f"\nMulti-Asset Portfolio:")
    print(f"  Total IM:        ${result.total_im:>15,.0f}")
    print(f"  Delta IM:        ${result.delta_im:>15,.0f}")
    print(f"  Vega IM:         ${result.vega_im:>15,.0f}")
    print(f"\nIM by Risk Class:")
    for risk_class, im in sorted(result.im_by_risk_class.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * im / result.total_im if result.total_im > 0 else 0
        print(f"  {risk_class.value:20s} ${im:>15,.0f}  ({pct:5.1f}%)")

    # Calculate diversification benefit
    total_undiversified = sum(result.im_by_risk_class.values())
    diversification_benefit = total_undiversified - result.total_im
    if total_undiversified > 0:
        div_pct = 100 * diversification_benefit / total_undiversified
        print(f"\nDiversification Benefit:")
        print(f"  Undiversified IM: ${total_undiversified:>15,.0f}")
        print(f"  Diversified IM:   ${result.total_im:>15,.0f}")
        print(f"  Benefit:          ${diversification_benefit:>15,.0f}  ({div_pct:5.1f}%)")

    return result


# ==============================================================================
# Example 3: UMR Compliance and AANA Calculation
# ==============================================================================


def example_3_umr_aana():
    """Calculate AANA and determine UMR phase-in applicability."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: UMR AANA Calculation and Phase-In Determination")
    print("=" * 80)

    checker = UMRComplianceChecker()

    # Example: Mid-sized derivatives dealer
    aana = checker.calculate_aana(
        march_notional=95_000_000_000,  # $95B
        april_notional=102_000_000_000,  # $102B
        may_notional=98_000_000_000,  # $98B
        calculation_year=2024,
    )

    print(f"\nAANA Calculation:")
    print(f"  March 2024:      ${aana.march_notional / 1e9:>10.1f}B")
    print(f"  April 2024:      ${aana.april_notional / 1e9:>10.1f}B")
    print(f"  May 2024:        ${aana.may_notional / 1e9:>10.1f}B")
    print(f"  AANA:            ${aana.aana / 1e9:>10.1f}B")
    print(f"  Calculation Date: {aana.calculation_date}")

    if aana.is_subject_to_umr():
        print(f"\n  Status: SUBJECT TO UMR")
        print(f"  Applicable Phase: {aana.applicable_phase.name}")
        print(f"  Phase Number: {aana.applicable_phase.value}")
    else:
        print(f"\n  Status: NOT SUBJECT TO UMR")
        print(f"  AANA below $8 billion threshold")

    # Show threshold table
    print(f"\n  UMR Phase-In Schedule:")
    print(f"    Phase I   (Sep 2016): AANA > $3.00 trillion")
    print(f"    Phase II  (Sep 2017): AANA > $2.25 trillion")
    print(f"    Phase III (Sep 2018): AANA > $1.50 trillion")
    print(f"    Phase IV  (Sep 2019): AANA > $0.75 trillion")
    print(f"    Phase V   (Sep 2020): AANA > $0.05 trillion")
    print(f"    Phase VI  (Sep 2022): AANA > $0.008 trillion")

    return aana


# ==============================================================================
# Example 4: UMR Margin Calls with CSA Terms
# ==============================================================================


def example_4_umr_margin_calls():
    """Generate margin calls with UMR compliance and CSA terms."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: UMR Margin Calls with CSA Terms")
    print("=" * 80)

    # First calculate SIMM for the portfolio
    sensitivities = [
        RiskFactorSensitivity(
            risk_factor_type=RiskFactorType.IR,
            sensitivity_type=SensitivityType.DELTA,
            bucket="USD",
            risk_factor="USD-OIS",
            sensitivity=100000,
            tenor="5Y",
        ),
        RiskFactorSensitivity(
            risk_factor_type=RiskFactorType.FX,
            sensitivity_type=SensitivityType.DELTA,
            bucket="1",
            risk_factor="EURUSD",
            sensitivity=1_500_000,
            tenor=None,
        ),
    ]

    simm_result = calculate_simm(sensitivities)
    simm_im = simm_result.total_im

    print(f"\nPortfolio SIMM IM: ${simm_im:,.0f}")

    # Set up CSA terms with standard UMR parameters
    csa_terms = CSATerms(
        csa_type="bilateral",
        im_threshold=50_000_000,  # $50MM
        vm_threshold=0.0,  # No VM threshold under UMR
        mta=500_000,  # $500k MTA
        rounding=100_000,  # Round to $100k
        dispute_threshold=250_000,  # $250k
        currency="USD",
    )

    # Create CSA portfolio
    portfolio = CSAPortfolio(
        counterparty="DEALER-A",
        portfolio_id="DERIV-001",
        csa_terms=csa_terms,
        outstanding_im=0.0,  # New portfolio
        outstanding_vm=0.0,
    )

    # Create CSA manager and register portfolio
    csa_manager = CSAManager(current_date=date(2024, 11, 5))
    csa_manager.register_portfolio(portfolio)

    # Calculate margin calls
    # Scenario: SIMM IM = result, MTM gain of $2MM (we owe VM)
    mtm_change = -2_000_000  # Negative = loss = we post VM

    im_call, vm_call = csa_manager.calculate_margin_calls(
        counterparty="DEALER-A",
        portfolio_id="DERIV-001",
        simm_im=simm_im,
        mtm_change=mtm_change,
    )

    print(f"\nMargin Call Results:")
    print(f"\n1. INITIAL MARGIN:")
    if im_call:
        print(f"   Gross SIMM IM:       ${simm_im:>15,.0f}")
        print(f"   IM Threshold:        ${csa_terms.im_threshold:>15,.0f}")
        print(f"   Net IM Required:     ${im_call.amount:>15,.0f}")
        print(f"   Outstanding IM:      ${im_call.outstanding_amount:>15,.0f}")
        print(f"   IM Call:             ${im_call.net_call:>15,.0f}")
        print(f"   Action:              {'POST' if im_call.is_posting else 'COLLECT'}")
        print(f"   Due Date:            {im_call.due_date}")
    else:
        print(f"   No IM call (below threshold + MTA)")

    print(f"\n2. VARIATION MARGIN:")
    if vm_call:
        print(f"   MTM Change:          ${mtm_change:>15,.0f}")
        print(f"   VM Required:         ${vm_call.amount:>15,.0f}")
        print(f"   Outstanding VM:      ${vm_call.outstanding_amount:>15,.0f}")
        print(f"   VM Call:             ${vm_call.net_call:>15,.0f}")
        print(f"   Action:              {'POST' if vm_call.is_posting else 'COLLECT'}")
        print(f"   Due Date:            {vm_call.due_date}")
    else:
        print(f"   No VM call (below MTA)")

    # Settle margin calls
    if im_call:
        csa_manager.settle_margin_call(
            im_call, CollateralType.GOVERNMENT_BONDS, im_call.amount
        )
        print(f"\n   IM settled with government bonds")

    if vm_call:
        csa_manager.settle_margin_call(vm_call, CollateralType.CASH, vm_call.amount)
        print(f"   VM settled with cash")

    # Get portfolio summary
    summary = csa_manager.get_portfolio_summary("DEALER-A", "DERIV-001")
    print(f"\nPortfolio Summary After Settlement:")
    print(f"   Outstanding IM:      ${summary['outstanding_im']:>15,.0f}")
    print(f"   Outstanding VM:      ${summary['outstanding_vm']:>15,.0f}")
    print(f"   Collateral Posted:   ${summary['collateral_posted_value']:>15,.0f}")

    return im_call, vm_call


# ==============================================================================
# Example 5: Multi-Counterparty Portfolio with Margin Reports
# ==============================================================================


def example_5_multi_counterparty():
    """Manage multiple counterparty portfolios with comprehensive reporting."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Multi-Counterparty Portfolio Management")
    print("=" * 80)

    csa_manager = CSAManager(current_date=date(2024, 11, 5))
    margin_calls = []

    # Define three counterparties with different exposures
    counterparties = [
        {
            "name": "BANK-A",
            "portfolio": "RATES-001",
            "simm_im": 75_000_000,  # $75MM
            "mtm_change": -1_500_000,  # Loss
        },
        {
            "name": "BANK-B",
            "portfolio": "FX-002",
            "simm_im": 45_000_000,  # $45MM (below threshold)
            "mtm_change": 800_000,  # Gain
        },
        {
            "name": "DEALER-C",
            "portfolio": "CREDIT-003",
            "simm_im": 120_000_000,  # $120MM
            "mtm_change": -2_000_000,  # Loss
        },
    ]

    print(f"\nProcessing {len(counterparties)} counterparty portfolios...")

    for cp_data in counterparties:
        # Create CSA portfolio
        csa_terms = CSATerms(
            im_threshold=50_000_000,
            mta=500_000,
            rounding=100_000,
        )

        portfolio = CSAPortfolio(
            counterparty=cp_data["name"],
            portfolio_id=cp_data["portfolio"],
            csa_terms=csa_terms,
        )

        csa_manager.register_portfolio(portfolio)

        # Calculate margin calls
        im_call, vm_call = csa_manager.calculate_margin_calls(
            counterparty=cp_data["name"],
            portfolio_id=cp_data["portfolio"],
            simm_im=cp_data["simm_im"],
            mtm_change=cp_data["mtm_change"],
        )

        if im_call:
            margin_calls.append(im_call)
        if vm_call:
            margin_calls.append(vm_call)

    # Generate comprehensive margin report
    report = generate_margin_report(margin_calls)
    print(report)

    return margin_calls


# ==============================================================================
# Example 6: Custodian Integration and Collateral Management
# ==============================================================================


def example_6_custodian_integration():
    """Demonstrate custodian integration for collateral management."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Custodian Integration and Collateral Management")
    print("=" * 80)

    # Set up custodian interface
    custodian = CustodianInterface(custodian_name="Euroclear")

    # Register our segregated account
    our_account = CustodianAccount(
        custodian_name="Euroclear",
        account_number="EC-123456",
        account_type="segregated",
        currency="USD",
        balance=50_000_000,  # $50MM cash
        securities={
            "US912828XG90": 100_000_000,  # $100MM US Treasuries
            "US912828XH73": 50_000_000,  # $50MM US Treasuries
        },
    )

    # Register counterparty account
    cp_account = CustodianAccount(
        custodian_name="Euroclear",
        account_number="EC-789012",
        account_type="segregated",
        currency="USD",
        balance=25_000_000,
        securities={},
    )

    custodian.register_account(our_account)
    custodian.register_account(cp_account)

    print(f"\nCustodian: {custodian.custodian_name}")
    print(f"\nOur Account ({our_account.account_number}):")
    print(f"  Cash Balance:    ${our_account.balance:>15,.0f}")
    print(f"  Securities:      {len(our_account.securities)} holdings")

    # Create a margin call scenario
    from neutryx.valuations.margin.umr_compliance import MarginCall, MarginType

    margin_call = MarginCall(
        margin_type=MarginType.INITIAL_MARGIN,
        amount=30_000_000,  # $30MM IM call
        currency="USD",
        due_date=date(2024, 11, 6),
        counterparty="DEALER-A",
        portfolio="RATES-001",
        calculation_date=date(2024, 11, 5),
        outstanding_amount=0.0,
    )

    print(f"\nMargin Call:")
    print(f"  Type:            {margin_call.margin_type.value}")
    print(f"  Amount:          ${margin_call.amount:,.0f}")
    print(f"  Counterparty:    {margin_call.counterparty}")
    print(f"  Due Date:        {margin_call.due_date}")

    # Initiate collateral movement
    movement = custodian.initiate_collateral_movement(
        margin_call=margin_call,
        from_account_number=our_account.account_number,
        to_account_number=cp_account.account_number,
        collateral_type=CollateralType.GOVERNMENT_BONDS,
    )

    print(f"\nCollateral Movement Initiated:")
    print(f"  Movement ID:     {movement.movement_id}")
    print(f"  From:            {movement.from_account.account_number}")
    print(f"  To:              {movement.to_account.account_number}")
    print(f"  Type:            {movement.collateral_type.value}")
    print(f"  Amount:          ${movement.amount:,.0f}")
    print(f"  Value Date:      {movement.value_date}")
    print(f"  Status:          {movement.status}")

    # Apply collateral haircut
    checker = UMRComplianceChecker()
    collateral_value = movement.amount
    adjusted_value = checker.apply_collateral_haircut(
        collateral_value, movement.collateral_type
    )
    haircut_pct = checker.thresholds.haircuts[movement.collateral_type] * 100

    print(f"\nCollateral Valuation:")
    print(f"  Market Value:    ${collateral_value:>15,.0f}")
    print(f"  Haircut:         {haircut_pct:>15.1f}%")
    print(f"  Adjusted Value:  ${adjusted_value:>15,.0f}")

    # Check eligibility
    is_eligible, reason = checker.check_collateral_eligibility(movement.collateral_type)
    print(f"\nCollateral Eligibility:")
    print(f"  Eligible:        {is_eligible}")
    print(f"  Reason:          {reason}")

    return movement


# ==============================================================================
# Main Execution
# ==============================================================================


def main():
    """Run all SIMM and UMR examples."""
    print("\n" + "=" * 80)
    print("NEUTRYX: SIMM AND UMR COMPLIANCE EXAMPLES")
    print("=" * 80)
    print("\nThis demo shows the complete workflow for SIMM initial margin")
    print("calculation and UMR compliance for bilateral derivatives portfolios.")

    # Run all examples
    example_1_ir_swap_simm()
    example_2_multi_asset_simm()
    example_3_umr_aana()
    example_4_umr_margin_calls()
    example_5_multi_counterparty()
    example_6_custodian_integration()

    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETED")
    print("=" * 80)
    print("\nKey Features Demonstrated:")
    print("  ✓ SIMM calculation for single and multi-asset portfolios")
    print("  ✓ AANA calculation and UMR phase-in determination")
    print("  ✓ IM and VM margin calls with CSA terms")
    print("  ✓ Threshold, MTA, and rounding application")
    print("  ✓ Multi-counterparty portfolio management")
    print("  ✓ Custodian integration and collateral management")
    print("  ✓ Comprehensive margin reporting")
    print("\nFor production use:")
    print("  - Integrate with real-time sensitivity calculations")
    print("  - Connect to custodian APIs for automated settlement")
    print("  - Add regulatory reporting and audit trails")
    print("  - Implement collateral optimization algorithms")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
