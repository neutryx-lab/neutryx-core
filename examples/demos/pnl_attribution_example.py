"""
P&L Attribution Framework Demo
=================================

This script demonstrates the comprehensive P&L attribution capabilities:

1. Daily P&L Explain by Greeks
2. Risk Factor Attribution
3. Model Risk Quantification
4. Basel P&L Attribution Test

Example Use Cases:
- Trading desk P&L validation
- Risk management reporting
- Regulatory compliance (FRTB)
- Model validation
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd

from neutryx.analytics.pnl_attribution import (
    GreekPLCalculator,
    ModelRiskCalculator,
    RiskFactorPLCalculator,
)


def demo_1_daily_pnl_explain():
    """Demo 1: Daily P&L explain for an options portfolio."""
    print("\n" + "=" * 70)
    print("Demo 1: Daily P&L Explain for Options Portfolio")
    print("=" * 70)

    calc = GreekPLCalculator()

    # Scenario: Long call option position
    # Portfolio Greeks at T0
    portfolio_greeks = {
        "delta": 1500.0,  # Long 1500 shares equivalent
        "gamma": 80.0,  # Positive convexity
        "vega": 3000.0,  # Long volatility
        "theta": -100.0,  # Daily time decay
        "rho": 800.0,  # Interest rate sensitivity
    }

    # Market moves during the day
    market_moves = {
        "price_change": 3.5,  # Stock up $3.50
        "vol_change": 0.015,  # Vol up 1.5%
        "rate_change": 0.0010,  # Rates up 10bp
    }

    # Actual P&L from marking positions
    actual_pnl = 6200.0

    # Carry P&L (funding, accruals)
    carry_pnl = 150.0

    # Compute P&L explain
    explain = calc.compute_daily_pnl_explain(
        date="2024-01-15",
        total_pnl=actual_pnl,
        delta=portfolio_greeks["delta"],
        gamma=portfolio_greeks["gamma"],
        vega=portfolio_greeks["vega"],
        theta=portfolio_greeks["theta"],
        rho=portfolio_greeks["rho"],
        price_change=market_moves["price_change"],
        vol_change=market_moves["vol_change"],
        rate_change=market_moves["rate_change"],
        carry_pnl=carry_pnl,
    )

    print(f"\nDate: {explain.date}")
    print(f"Total P&L: ${explain.total_pnl:,.2f}")
    print(f"\nP&L Attribution:")
    print(f"  Carry P&L:       ${explain.carry_pnl:>10,.2f}")
    print(f"  Delta P&L:       ${explain.delta_pnl:>10,.2f}")
    print(f"  Gamma P&L:       ${explain.gamma_pnl:>10,.2f}")
    print(f"  Vega P&L:        ${explain.vega_pnl:>10,.2f}")
    print(f"  Theta P&L:       ${explain.theta_pnl:>10,.2f}")
    print(f"  Rho P&L:         ${explain.rho_pnl:>10,.2f}")
    print(f"  {'─' * 35}")
    print(f"  Explained P&L:   ${explain.explained_pnl:>10,.2f}")
    print(f"  Unexplained:     ${explain.unexplained_pnl:>10,.2f}")

    print(f"\nExplanation Ratio: {explain.explanation_ratio:.2%}")
    print(f"Passes Basel Test (90%): {explain.passes_attribution_test(0.9)}")

    # Visualize breakdown
    visualize_pnl_breakdown(explain)

    return explain


def demo_2_risk_factor_attribution():
    """Demo 2: Risk factor attribution across asset classes."""
    print("\n" + "=" * 70)
    print("Demo 2: Risk Factor Attribution")
    print("=" * 70)

    calc = RiskFactorPLCalculator()

    # ===== Interest Rate Risk =====
    print("\n--- Interest Rate Risk Attribution ---")

    rate_sensitivities = {
        "USD.3M": 5000.0,
        "USD.1Y": 15000.0,
        "USD.5Y": 25000.0,
        "USD.10Y": 20000.0,
        "USD.30Y": 10000.0,
    }

    rate_changes = {
        "USD.3M": 0.0005,  # 5bp
        "USD.1Y": 0.0008,  # 8bp
        "USD.5Y": 0.0012,  # 12bp
        "USD.10Y": 0.0015,  # 15bp
        "USD.30Y": 0.0010,  # 10bp
    }

    ir_attributions = calc.compute_ir_attribution(rate_sensitivities, rate_changes)

    print(f"{'Curve Point':<15} {'DV01':>12} {'Rate Δ (bp)':>15} {'P&L':>12}")
    print("─" * 60)
    for attr in ir_attributions:
        curve_point = attr.risk_factor
        sensitivity = rate_sensitivities[curve_point]
        change_bp = rate_changes[curve_point] * 10000
        print(
            f"{curve_point:<15} ${sensitivity:>10,.0f} {change_bp:>14,.1f} ${attr.delta_pnl:>10,.2f}"
        )

    total_ir = sum(a.delta_pnl for a in ir_attributions)
    print(f"{'─' * 60}")
    print(f"{'Total IR P&L':<15} {' ' * 27} ${total_ir:>10,.2f}")

    # ===== Credit Spread Risk =====
    print("\n--- Credit Spread Attribution ---")

    credit_sensitivities = {
        "CORP.AAA.5Y": 8000.0,
        "CORP.AA.5Y": 12000.0,
        "CORP.A.5Y": 15000.0,
        "CORP.BBB.5Y": 10000.0,
    }

    spread_changes = {
        "CORP.AAA.5Y": 0.0002,  # 2bp widening
        "CORP.AA.5Y": 0.0003,  # 3bp widening
        "CORP.A.5Y": -0.0005,  # 5bp tightening
        "CORP.BBB.5Y": 0.0008,  # 8bp widening
    }

    credit_attributions = calc.compute_credit_attribution(
        credit_sensitivities, spread_changes
    )

    print(f"{'Credit Curve':<20} {'CS01':>12} {'Spread Δ (bp)':>18} {'P&L':>12}")
    print("─" * 70)
    for attr in credit_attributions:
        curve = attr.risk_factor
        sensitivity = credit_sensitivities[curve]
        change_bp = spread_changes[curve] * 10000
        print(
            f"{curve:<20} ${sensitivity:>10,.0f} {change_bp:>17,.1f} ${attr.delta_pnl:>10,.2f}"
        )

    total_credit = sum(a.delta_pnl for a in credit_attributions)
    print(f"{'─' * 70}")
    print(f"{'Total Credit P&L':<20} {' ' * 32} ${total_credit:>10,.2f}")

    # ===== FX Risk =====
    print("\n--- FX Risk Attribution ---")

    fx_deltas = {
        "EURUSD": 50000.0,
        "GBPUSD": 30000.0,
        "USDJPY": -40000.0,
        "AUDUSD": 20000.0,
    }

    fx_changes = {
        "EURUSD": 0.0120,  # 1.2% EUR appreciation
        "GBPUSD": -0.0080,  # 0.8% GBP depreciation
        "USDJPY": 0.0150,  # 1.5% JPY depreciation
        "AUDUSD": 0.0050,  # 0.5% AUD appreciation
    }

    fx_attributions = calc.compute_fx_attribution(fx_deltas, fx_changes)

    print(f"{'FX Pair':<15} {'Delta':>12} {'Spot Δ (%)':>15} {'P&L':>12}")
    print("─" * 60)
    for attr in fx_attributions:
        pair = attr.risk_factor
        delta = fx_deltas[pair]
        change_pct = fx_changes[pair] * 100
        print(
            f"{pair:<15} ${delta:>10,.0f} {change_pct:>14,.2f}% ${attr.delta_pnl:>10,.2f}"
        )

    total_fx = sum(a.delta_pnl for a in fx_attributions)
    print(f"{'─' * 60}")
    print(f"{'Total FX P&L':<15} {' ' * 27} ${total_fx:>10,.2f}")

    # ===== Equity Risk =====
    print("\n--- Equity Risk Attribution ---")

    equity_deltas = {
        "AAPL": 2000.0,
        "MSFT": 1500.0,
        "GOOGL": 1000.0,
        "AMZN": 800.0,
    }

    equity_gammas = {
        "AAPL": 100.0,
        "MSFT": 80.0,
        "GOOGL": 60.0,
        "AMZN": 50.0,
    }

    equity_changes = {
        "AAPL": 4.25,
        "MSFT": -2.80,
        "GOOGL": 6.50,
        "AMZN": 3.20,
    }

    equity_attributions = calc.compute_equity_attribution(
        equity_deltas, equity_changes, equity_gammas
    )

    print(f"{'Stock':<10} {'Delta':>10} {'Gamma':>10} {'Price Δ':>12} {'P&L':>12}")
    print("─" * 60)
    for attr in equity_attributions:
        stock = attr.risk_factor
        delta = equity_deltas[stock]
        gamma = equity_gammas[stock]
        price_change = equity_changes[stock]
        print(
            f"{stock:<10} {delta:>10,.0f} {gamma:>10,.0f} ${price_change:>10,.2f} ${attr.total_pnl:>10,.2f}"
        )

    total_equity = sum(a.total_pnl for a in equity_attributions)
    print(f"{'─' * 60}")
    print(f"{'Total Equity P&L':<10} {' ' * 32} ${total_equity:>10,.2f}")

    return {
        "ir": ir_attributions,
        "credit": credit_attributions,
        "fx": fx_attributions,
        "equity": equity_attributions,
    }


def demo_3_model_risk():
    """Demo 3: Model risk quantification."""
    print("\n" + "=" * 70)
    print("Demo 3: Model Risk Quantification")
    print("=" * 70)

    calc = ModelRiskCalculator()

    # Scenario: Exotic option with model risk

    # Mark-to-Model Reserve
    print("\n--- Mark-to-Model Reserve ---")
    model_price = 105.50
    market_price = 103.00  # Observable market price
    bid_ask_spread = 1.00

    mtm_reserve = calc.compute_mark_to_model_reserve(
        model_price, market_price, bid_ask_spread
    )

    print(f"Model Price:       ${model_price:,.2f}")
    print(f"Market Price:      ${market_price:,.2f}")
    print(f"Difference:        ${model_price - market_price:,.2f}")
    print(f"Bid-Ask Spread:    ${bid_ask_spread:,.2f}")
    print(f"MTM Reserve:       ${mtm_reserve:,.2f}")

    # Parameter Uncertainty
    print("\n--- Parameter Uncertainty ---")

    # Two key parameters: volatility and correlation
    parameter_std = jnp.array([0.02, 0.05])  # 2% vol std, 5% corr std
    sensitivity_to_param = jnp.array([80000.0, 30000.0])  # Sensitivities

    param_uncertainty = calc.compute_parameter_uncertainty(
        parameter_std, sensitivity_to_param
    )

    print(f"Parameter 1 (Vol):   σ = 2.0%, Sensitivity = $80,000")
    print(f"Parameter 2 (Corr):  σ = 5.0%, Sensitivity = $30,000")
    print(f"P&L Uncertainty (1σ): ${param_uncertainty:,.2f}")
    print(f"P&L Uncertainty (2σ): ${2 * param_uncertainty:,.2f} (95% conf)")

    # Model Replacement Impact
    print("\n--- Model Replacement Impact ---")

    current_model_value = 105.50
    alternative_model_value = 108.20  # Alternative model gives higher value

    replacement_impact = calc.compute_model_replacement_impact(
        current_model_value, alternative_model_value
    )

    print(f"Current Model:         ${current_model_value:,.2f}")
    print(f"Alternative Model:     ${alternative_model_value:,.2f}")
    print(f"Replacement Impact:    ${replacement_impact:,.2f}")

    # Unexplained P&L Analysis
    print("\n--- Unexplained P&L Analysis ---")

    # Historical unexplained P&L (last 20 days)
    unexplained_history = jnp.array(
        [
            120.0,
            -80.0,
            150.0,
            -40.0,
            200.0,
            -100.0,
            90.0,
            -60.0,
            180.0,
            -120.0,
            110.0,
            -70.0,
            160.0,
            -90.0,
            130.0,
            -50.0,
            140.0,
            -110.0,
            170.0,
            -85.0,
        ]
    )

    mean, std = calc.analyze_unexplained_pnl(unexplained_history)

    print(f"Sample Size:           {len(unexplained_history)} days")
    print(f"Mean Unexplained:      ${mean:,.2f}")
    print(f"Std Dev:               ${std:,.2f}")
    print(f"Max Absolute:          ${float(jnp.max(jnp.abs(unexplained_history))):,.2f}")

    # Comprehensive Model Risk Metrics
    print("\n--- Total Model Risk Summary ---")

    metrics = calc.compute_model_risk_metrics(
        model_price=model_price,
        market_price=market_price,
        bid_ask_spread=bid_ask_spread,
        parameter_std=parameter_std,
        sensitivity_to_param=sensitivity_to_param,
        alternative_model_value=alternative_model_value,
        unexplained_pnl_history=unexplained_history,
    )

    print(f"MTM Reserve:              ${metrics.mark_to_model_reserve:>10,.2f}")
    print(f"Parameter Uncertainty:    ${metrics.parameter_uncertainty:>10,.2f}")
    print(f"Model Replacement Impact: ${metrics.model_replacement_impact:>10,.2f}")
    print(f"Unexplained P&L (mean):   ${metrics.unexplained_pnl_avg:>10,.2f}")
    print(f"Unexplained P&L (std):    ${metrics.unexplained_pnl_std:>10,.2f}")
    print(f"{'─' * 50}")
    print(f"Total Model Risk:         ${metrics.total_model_risk:>10,.2f}")

    return metrics


def demo_4_basel_pnl_test():
    """Demo 4: Basel P&L attribution test for regulatory compliance."""
    print("\n" + "=" * 70)
    print("Demo 4: Basel P&L Attribution Test (Regulatory Compliance)")
    print("=" * 70)

    calc = GreekPLCalculator()

    # Simulate 10 days of P&L attribution
    dates = [f"2024-01-{day:02d}" for day in range(10, 20)]

    results = []

    print("\nDaily P&L Attribution Results:")
    print(f"{'Date':<15} {'Total P&L':>12} {'Explained':>12} {'Ratio':>10} {'Pass?':>8}")
    print("─" * 65)

    for i, date in enumerate(dates):
        # Simulate different market scenarios
        total_pnl = 5000 + 1000 * (i - 5)  # Varying P&L
        delta = 800 + 50 * i
        gamma = 40 + 5 * i
        vega = 1500 + 100 * i
        theta = -40 - 2 * i
        rho = 300 + 20 * i

        price_change = 2.0 + 0.5 * (i % 3)
        vol_change = 0.01 + 0.005 * (i % 2)
        rate_change = 0.0005 + 0.0003 * (i % 2)
        carry_pnl = 100.0

        explain = calc.compute_daily_pnl_explain(
            date=date,
            total_pnl=total_pnl,
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho=rho,
            price_change=price_change,
            vol_change=vol_change,
            rate_change=rate_change,
            carry_pnl=carry_pnl,
        )

        passes = explain.passes_attribution_test(0.9)
        results.append(
            {
                "date": date,
                "total_pnl": total_pnl,
                "explained_pnl": explain.explained_pnl,
                "ratio": explain.explanation_ratio,
                "passes": passes,
            }
        )

        print(
            f"{date:<15} ${total_pnl:>10,.0f} ${explain.explained_pnl:>10,.0f} "
            f"{explain.explanation_ratio:>9.1%} {'✓' if passes else '✗':>8}"
        )

    # Summary statistics
    df = pd.DataFrame(results)
    pass_rate = df["passes"].sum() / len(df) * 100
    avg_ratio = df["ratio"].mean()

    print(f"\n{'═' * 65}")
    print(f"BASEL P&L ATTRIBUTION TEST SUMMARY")
    print(f"{'═' * 65}")
    print(f"Total Days:           {len(df)}")
    print(f"Days Passed (≥90%):   {df['passes'].sum()}")
    print(f"Pass Rate:            {pass_rate:.1f}%")
    print(f"Average Ratio:        {avg_ratio:.2%}")
    print(f"\nRegulatory Threshold: 90% explanation ratio required")
    print(f"Overall Assessment:   {'PASS ✓' if pass_rate >= 90 else 'FAIL ✗'}")

    visualize_attribution_test(df)

    return df


def visualize_pnl_breakdown(explain):
    """Visualize P&L breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of components
    components = ["Carry", "Delta", "Gamma", "Vega", "Theta", "Rho", "Unexplained"]
    values = [
        explain.carry_pnl,
        explain.delta_pnl,
        explain.gamma_pnl,
        explain.vega_pnl,
        explain.theta_pnl,
        explain.rho_pnl,
        explain.unexplained_pnl,
    ]

    colors = ["green" if v > 0 else "red" for v in values]
    ax1.bar(components, values, color=colors, alpha=0.7)
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax1.set_ylabel("P&L ($)")
    ax1.set_title(f"P&L Breakdown - {explain.date}")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(axis="y", alpha=0.3)

    # Waterfall effect (cumulative)
    cumulative = [0]
    for v in values[:-1]:  # Exclude unexplained
        cumulative.append(cumulative[-1] + v)

    ax2.bar(range(len(components)), values, color=colors, alpha=0.7)
    ax2.set_xticks(range(len(components)))
    ax2.set_xticklabels(components, rotation=45)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2.axhline(y=explain.total_pnl, color="blue", linestyle="--", label="Total P&L")
    ax2.set_ylabel("P&L ($)")
    ax2.set_title(f"Explanation Ratio: {explain.explanation_ratio:.1%}")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("pnl_breakdown.png", dpi=300, bbox_inches="tight")
    print("\nVisualization saved as 'pnl_breakdown.png'")
    plt.close()


def visualize_attribution_test(df):
    """Visualize Basel P&L attribution test results."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot explanation ratios
    x = range(len(df))
    ax.plot(x, df["ratio"] * 100, marker="o", linewidth=2, markersize=8, label="Explanation Ratio")

    # Highlight pass/fail
    passed = df[df["passes"]]
    failed = df[~df["passes"]]

    ax.scatter(
        passed.index,
        passed["ratio"] * 100,
        color="green",
        s=150,
        marker="o",
        zorder=5,
        label="Pass (≥90%)",
    )
    ax.scatter(
        failed.index,
        failed["ratio"] * 100,
        color="red",
        s=150,
        marker="x",
        zorder=5,
        label="Fail (<90%)",
    )

    # 90% threshold
    ax.axhline(y=90, color="red", linestyle="--", linewidth=2, label="Regulatory Threshold")

    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Explanation Ratio (%)")
    ax.set_title("Basel P&L Attribution Test - Daily Results")
    ax.set_xticks(x)
    ax.set_xticklabels(df["date"], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(70, 105)

    plt.tight_layout()
    plt.savefig("basel_pnl_test.png", dpi=300, bbox_inches="tight")
    print("\nVisualization saved as 'basel_pnl_test.png'")
    plt.close()


def main():
    """Run all P&L attribution demos."""
    print("\n" + "=" * 70)
    print("P&L ATTRIBUTION FRAMEWORK - COMPREHENSIVE DEMO")
    print("=" * 70)
    print("\nThis demo showcases:")
    print("  • Daily P&L explain by Greeks")
    print("  • Risk factor attribution across asset classes")
    print("  • Model risk quantification")
    print("  • Basel P&L attribution test")

    # Run demos
    explain = demo_1_daily_pnl_explain()
    risk_factors = demo_2_risk_factor_attribution()
    model_risk = demo_3_model_risk()
    basel_test = demo_4_basel_pnl_test()

    print("\n" + "=" * 70)
    print("ALL DEMOS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nGenerated files:")
    print("  • pnl_breakdown.png")
    print("  • basel_pnl_test.png")


if __name__ == "__main__":
    main()
