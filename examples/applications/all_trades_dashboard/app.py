"""Fictional Bank All Trades Dashboard.

This interactive dashboard provides real-time monitoring of all trades in the
Fictional Bank portfolio, with comprehensive PV, risk analytics, and XVA calculations.

Features:
- Real-time trade overview with PV and Greeks
- Risk heatmap by counterparty and product type
- Stress testing scenarios
- XVA analytics (CVA, DVA, FVA, MVA)
- Real-time alerts for risk limit breaches
- CSV export functionality
"""
from __future__ import annotations

import csv
import io
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import jax.numpy as jnp
import pandas as pd
import yaml

from neutryx.engines.fourier import BlackScholesCharacteristicModel, cos_method
from neutryx.portfolio.portfolio import Portfolio
from neutryx.tests.fixtures.fictional_portfolio import create_fictional_portfolio


@dataclass
class DashboardConfig:
    """Configuration for the dashboard."""

    refresh_interval: int = 5
    max_net_delta: float = 1_000_000
    max_counterparty_exposure: float = 5_000_000
    max_gamma: float = 50_000
    concentration_threshold: float = 0.3
    default_lgd: float = 0.6
    funding_spread_bps: int = 50


# Global portfolio instance (loaded once at startup)
PORTFOLIO: Portfolio | None = None
BOOK_HIERARCHY = None
CONFIG: DashboardConfig | None = None


def load_config() -> DashboardConfig:
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f)
            risk_limits = data.get("risk_limits", {})
            xva_params = data.get("xva_params", {})
            return DashboardConfig(
                refresh_interval=data.get("dashboard", {}).get("refresh_interval_seconds", 5),
                max_net_delta=risk_limits.get("max_net_delta", 1_000_000),
                max_counterparty_exposure=risk_limits.get("max_counterparty_exposure", 5_000_000),
                max_gamma=risk_limits.get("max_gamma", 50_000),
                concentration_threshold=risk_limits.get("concentration_threshold", 0.3),
                default_lgd=xva_params.get("default_lgd", 0.6),
                funding_spread_bps=xva_params.get("funding_spread_bps", 50),
            )
    return DashboardConfig()


def initialize_portfolio() -> tuple[Portfolio, Any]:
    """Initialize the fictional portfolio."""
    return create_fictional_portfolio()


def calculate_simple_pv(trade: Any) -> float:
    """Calculate a simplified PV for a trade using its MTM.

    In a production system, this would use proper pricing models.
    For this demo, we use MTM as a proxy for PV with some randomization
    to simulate real-time market movements.
    """
    import random

    # Use MTM as base, add small random variation to simulate real-time changes
    base_pv = trade.mtm if hasattr(trade, "mtm") and trade.mtm is not None else 0.0
    # Add ±2% random variation
    variation = random.uniform(-0.02, 0.02)
    return base_pv * (1 + variation)


def calculate_greeks(trade: Any) -> dict[str, float]:
    """Calculate simplified Greeks for a trade.

    In production, this would use proper derivative pricing models.
    For this demo, we generate realistic-looking values based on trade characteristics.
    """
    import random

    notional = trade.notional if hasattr(trade, "notional") else 1_000_000

    # Generate realistic Greeks based on notional
    greeks = {
        "delta": random.uniform(-0.5, 0.5) * notional / 1_000_000,
        "gamma": random.uniform(0, 0.1) * notional / 10_000_000,
        "vega": random.uniform(0, 0.3) * notional / 1_000_000,
        "theta": random.uniform(-0.02, 0) * notional / 1_000_000,
        "rho": random.uniform(-0.1, 0.1) * notional / 1_000_000,
    }
    return greeks


def get_all_trades_data(
    counterparty_filter: str = "All",
    product_filter: str = "All",
    desk_filter: str = "All",
) -> pd.DataFrame:
    """Get all trades with PV and risk metrics."""
    if PORTFOLIO is None:
        return pd.DataFrame()

    trades_data = []
    for trade in PORTFOLIO.trades.values():
        # Apply filters
        if counterparty_filter != "All" and trade.counterparty_id != counterparty_filter:
            continue

        product_type = trade.product_type.value if hasattr(trade.product_type, "value") else str(trade.product_type)
        if product_filter != "All" and product_type != product_filter:
            continue

        # Calculate PV and Greeks
        pv = calculate_simple_pv(trade)
        greeks = calculate_greeks(trade)

        trade_data = {
            "Trade ID": trade.id,
            "Counterparty": trade.counterparty_id,
            "Product": product_type,
            "Notional": f"{trade.notional:,.0f}" if hasattr(trade, "notional") else "N/A",
            "Currency": trade.currency if hasattr(trade, "currency") else "USD",
            "MTM": f"{trade.mtm:,.2f}" if hasattr(trade, "mtm") and trade.mtm is not None else "0.00",
            "PV": f"{pv:,.2f}",
            "Delta": f"{greeks['delta']:,.2f}",
            "Gamma": f"{greeks['gamma']:,.4f}",
            "Vega": f"{greeks['vega']:,.2f}",
        }
        trades_data.append(trade_data)

    return pd.DataFrame(trades_data)


def calculate_dashboard_metrics() -> dict[str, Any]:
    """Calculate key dashboard metrics."""
    if PORTFOLIO is None:
        return {}

    total_trades = len(PORTFOLIO.trades)
    total_notional = sum(
        trade.notional for trade in PORTFOLIO.trades.values() if hasattr(trade, "notional")
    )
    total_mtm = sum(
        trade.mtm for trade in PORTFOLIO.trades.values() if hasattr(trade, "mtm") and trade.mtm is not None
    )

    # Calculate total PV and Greeks
    total_pv = 0.0
    net_delta = 0.0
    total_gamma = 0.0
    total_vega = 0.0

    for trade in PORTFOLIO.trades.values():
        pv = calculate_simple_pv(trade)
        greeks = calculate_greeks(trade)
        total_pv += pv
        net_delta += greeks["delta"]
        total_gamma += greeks["gamma"]
        total_vega += greeks["vega"]

    return {
        "Total Trades": total_trades,
        "Total Notional": f"${total_notional/1_000_000:,.1f}M",
        "Total MTM": f"${total_mtm:,.0f}",
        "Total PV": f"${total_pv:,.0f}",
        "Net Delta": f"{net_delta:,.0f}",
        "Total Gamma": f"{total_gamma:,.4f}",
        "Total Vega": f"{total_vega:,.0f}",
        "Counterparties": len(PORTFOLIO.counterparties),
    }


def format_metrics_display(metrics: dict[str, Any]) -> str:
    """Format metrics as markdown for display."""
    return f"""
### Key Metrics

| Metric | Value |
|--------|-------|
| Total Trades | {metrics.get('Total Trades', 0)} |
| Total Notional | {metrics.get('Total Notional', '$0')} |
| Total MTM | {metrics.get('Total MTM', '$0')} |
| Total PV | {metrics.get('Total PV', '$0')} |
| Net Delta | {metrics.get('Net Delta', '0')} |
| Total Gamma | {metrics.get('Total Gamma', '0')} |
| Total Vega | {metrics.get('Total Vega', '0')} |
| Counterparties | {metrics.get('Counterparties', 0)} |
"""


def generate_risk_heatmap() -> pd.DataFrame:
    """Generate risk heatmap data by counterparty and product type."""
    if PORTFOLIO is None:
        return pd.DataFrame()

    # Calculate exposure by counterparty and product
    heatmap_data = defaultdict(lambda: defaultdict(float))

    for trade in PORTFOLIO.trades.values():
        cp = trade.counterparty_id
        product = trade.product_type.value if hasattr(trade.product_type, "value") else str(trade.product_type)
        pv = calculate_simple_pv(trade)
        heatmap_data[cp][product] += abs(pv)

    # Convert to DataFrame
    if not heatmap_data:
        return pd.DataFrame()

    # Get all unique products
    all_products = set()
    for products in heatmap_data.values():
        all_products.update(products.keys())

    rows = []
    for cp, products in heatmap_data.items():
        row = {"Counterparty": cp}
        for product in sorted(all_products):
            row[product] = f"${products.get(product, 0):,.0f}"
        rows.append(row)

    return pd.DataFrame(rows)


def run_stress_tests() -> pd.DataFrame:
    """Run stress test scenarios."""
    if PORTFOLIO is None:
        return pd.DataFrame()

    # Base case
    base_metrics = calculate_dashboard_metrics()
    base_pv = float(base_metrics["Total PV"].replace("$", "").replace(",", ""))
    base_delta = float(base_metrics["Net Delta"].replace(",", ""))

    scenarios = [
        {"Scenario": "Base Case", "PV Change": "$0", "Delta Change": "0", "Total PV": f"${base_pv:,.0f}"},
        {
            "Scenario": "Rates +100bps",
            "PV Change": f"${base_pv * -0.05:,.0f}",
            "Delta Change": f"{base_delta * 0.1:,.0f}",
            "Total PV": f"${base_pv * 0.95:,.0f}",
        },
        {
            "Scenario": "Rates -100bps",
            "PV Change": f"${base_pv * 0.05:,.0f}",
            "Delta Change": f"{base_delta * -0.1:,.0f}",
            "Total PV": f"${base_pv * 1.05:,.0f}",
        },
        {
            "Scenario": "Vol +20%",
            "PV Change": f"${base_pv * 0.08:,.0f}",
            "Delta Change": f"{base_delta * 0.05:,.0f}",
            "Total PV": f"${base_pv * 1.08:,.0f}",
        },
        {
            "Scenario": "Vol -20%",
            "PV Change": f"${base_pv * -0.08:,.0f}",
            "Delta Change": f"{base_delta * -0.05:,.0f}",
            "Total PV": f"${base_pv * 0.92:,.0f}",
        },
        {
            "Scenario": "FX +10%",
            "PV Change": f"${base_pv * 0.03:,.0f}",
            "Delta Change": f"{base_delta * 0.12:,.0f}",
            "Total PV": f"${base_pv * 1.03:,.0f}",
        },
        {
            "Scenario": "Equity -20%",
            "PV Change": f"${base_pv * -0.12:,.0f}",
            "Delta Change": f"{base_delta * -0.15:,.0f}",
            "Total PV": f"${base_pv * 0.88:,.0f}",
        },
    ]

    return pd.DataFrame(scenarios)


def calculate_xva_by_counterparty() -> pd.DataFrame:
    """Calculate XVA metrics by counterparty."""
    if PORTFOLIO is None or CONFIG is None:
        return pd.DataFrame()

    xva_data = []

    # Group trades by counterparty
    cp_trades = defaultdict(list)
    for trade in PORTFOLIO.trades.values():
        cp_trades[trade.counterparty_id].append(trade)

    for cp_id, trades in cp_trades.items():
        # Calculate positive and negative exposure
        positive_exposure = 0.0
        negative_exposure = 0.0
        total_notional = 0.0

        for trade in trades:
            pv = calculate_simple_pv(trade)
            if pv > 0:
                positive_exposure += pv
            else:
                negative_exposure += abs(pv)

            if hasattr(trade, "notional"):
                total_notional += trade.notional

        # Simplified XVA calculations
        # CVA = LGD * Positive Exposure * PD (simplified, assuming PD = 1%)
        cva = CONFIG.default_lgd * positive_exposure * 0.01

        # DVA = LGD * Negative Exposure * Own PD (simplified, assuming Own PD = 0.5%)
        dva = CONFIG.default_lgd * negative_exposure * 0.005

        # FVA = Funding spread * Net exposure
        fva = (CONFIG.funding_spread_bps / 10000) * (positive_exposure - negative_exposure)

        # MVA = simplified margin cost
        mva = 0.001 * total_notional

        total_xva = cva - dva + fva + mva

        xva_data.append(
            {
                "Counterparty": cp_id,
                "Positive Exposure": f"${positive_exposure:,.0f}",
                "CVA": f"${cva:,.0f}",
                "DVA": f"${dva:,.0f}",
                "FVA": f"${fva:,.0f}",
                "MVA": f"${mva:,.0f}",
                "Total XVA": f"${total_xva:,.0f}",
            }
        )

    return pd.DataFrame(xva_data)


def check_alerts() -> str:
    """Check for risk limit breaches and generate alerts."""
    if PORTFOLIO is None or CONFIG is None:
        return "No alerts"

    alerts = []
    metrics = calculate_dashboard_metrics()

    # Check net delta
    net_delta = float(metrics["Net Delta"].replace(",", ""))
    if abs(net_delta) > CONFIG.max_net_delta:
        alerts.append(f"⚠️ Net Delta Alert: {net_delta:,.0f} exceeds limit of {CONFIG.max_net_delta:,.0f}")

    # Check gamma
    total_gamma = float(metrics["Total Gamma"].replace(",", ""))
    if abs(total_gamma) > CONFIG.max_gamma:
        alerts.append(f"⚠️ Gamma Alert: {total_gamma:,.4f} exceeds limit of {CONFIG.max_gamma:,.0f}")

    # Check counterparty concentration
    total_pv = float(metrics["Total PV"].replace("$", "").replace(",", ""))
    for trade in PORTFOLIO.trades.values():
        cp_pv = sum(
            calculate_simple_pv(t) for t in PORTFOLIO.trades.values() if t.counterparty_id == trade.counterparty_id
        )
        concentration = abs(cp_pv) / total_pv if total_pv > 0 else 0
        if concentration > CONFIG.concentration_threshold:
            alerts.append(
                f"⚠️ Concentration Alert: {trade.counterparty_id} represents {concentration*100:.1f}% of total exposure"
            )
            break  # Only show once per counterparty

    if not alerts:
        return "✅ All risk limits within acceptable ranges"

    return "\n".join(alerts)


def export_to_csv() -> str:
    """Export current trades data to CSV format."""
    df = get_all_trades_data()
    if df.empty:
        return "No data to export"

    # Create CSV in memory
    output = io.StringIO()
    df.to_csv(output, index=False)
    csv_content = output.getvalue()

    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"fictional_bank_trades_{timestamp}.csv"

    return csv_content


def refresh_dashboard(counterparty_filter: str, product_filter: str, desk_filter: str) -> tuple:
    """Refresh all dashboard components."""
    # Get trades data
    trades_df = get_all_trades_data(counterparty_filter, product_filter, desk_filter)

    # Calculate metrics
    metrics = calculate_dashboard_metrics()
    metrics_md = format_metrics_display(metrics)

    # Generate heatmap
    heatmap_df = generate_risk_heatmap()

    # Run stress tests
    stress_df = run_stress_tests()

    # Calculate XVA
    xva_df = calculate_xva_by_counterparty()

    # Check alerts
    alerts = check_alerts()

    return trades_df, metrics_md, heatmap_df, stress_df, xva_df, alerts


def build_dashboard() -> gr.Blocks:
    """Build the Gradio dashboard interface."""
    global PORTFOLIO, BOOK_HIERARCHY, CONFIG

    # Initialize
    PORTFOLIO, BOOK_HIERARCHY = initialize_portfolio()
    CONFIG = load_config()

    # Get filter options
    counterparties = ["All"] + sorted([cp.id for cp in PORTFOLIO.counterparties.values()])
    products = ["All"] + sorted(
        list(set(trade.product_type.value if hasattr(trade.product_type, "value") else str(trade.product_type) for trade in PORTFOLIO.trades.values()))
    )

    with gr.Blocks(title="Fictional Bank - All Trades Dashboard", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🏦 Fictional Bank - All Trades Dashboard

            Real-time monitoring of all trades with comprehensive PV, risk analytics, and XVA calculations.
            Dashboard auto-refreshes every 5 seconds to simulate real-time market movements.
            """
        )

        # Alerts section at the top
        with gr.Row():
            alerts_display = gr.Markdown(value="Loading alerts...", label="Alerts")

        # Metrics KPI cards
        with gr.Row():
            metrics_display = gr.Markdown(value="Loading metrics...")

        # Main content
        with gr.Tabs():
            # Tab 1: All Trades
            with gr.Tab("📊 All Trades"):
                with gr.Row():
                    counterparty_filter = gr.Dropdown(
                        choices=counterparties, value="All", label="Filter by Counterparty"
                    )
                    product_filter = gr.Dropdown(choices=products, value="All", label="Filter by Product")
                    desk_filter = gr.Dropdown(
                        choices=["All", "Rates", "FX", "Equity"], value="All", label="Filter by Desk"
                    )
                    refresh_btn = gr.Button("🔄 Refresh Now", variant="primary")

                trades_table = gr.Dataframe(
                    label="All Trades - Real-time PV and Greeks",
                    wrap=True,
                    interactive=False,
                )

                export_btn = gr.Button("📥 Export to CSV")
                csv_output = gr.Textbox(label="CSV Export", visible=False)

            # Tab 2: Risk Heatmap
            with gr.Tab("🔥 Risk Heatmap"):
                gr.Markdown("### Exposure by Counterparty and Product Type")
                heatmap_table = gr.Dataframe(label="Risk Concentration Heatmap", wrap=True)

            # Tab 3: Stress Tests
            with gr.Tab("⚡ Stress Tests"):
                gr.Markdown("### Scenario Analysis")
                gr.Markdown(
                    "Impact of market stress scenarios on portfolio PV and Delta. "
                    "Scenarios include rate shifts, volatility changes, and equity/FX shocks."
                )
                stress_table = gr.Dataframe(label="Stress Test Results", wrap=True)

            # Tab 4: XVA Analysis
            with gr.Tab("💰 XVA Analysis"):
                gr.Markdown("### Credit and Funding Valuation Adjustments by Counterparty")
                gr.Markdown(
                    "CVA (Credit Valuation Adjustment), DVA (Debit Valuation Adjustment), "
                    "FVA (Funding Valuation Adjustment), and MVA (Margin Valuation Adjustment)"
                )
                xva_table = gr.Dataframe(label="XVA by Counterparty", wrap=True)

        # Wire up refresh functionality
        refresh_inputs = [counterparty_filter, product_filter, desk_filter]
        refresh_outputs = [trades_table, metrics_display, heatmap_table, stress_table, xva_table, alerts_display]

        # Manual refresh button
        refresh_btn.click(fn=refresh_dashboard, inputs=refresh_inputs, outputs=refresh_outputs)

        # Filter changes trigger refresh
        counterparty_filter.change(fn=refresh_dashboard, inputs=refresh_inputs, outputs=refresh_outputs)
        product_filter.change(fn=refresh_dashboard, inputs=refresh_inputs, outputs=refresh_outputs)
        desk_filter.change(fn=refresh_dashboard, inputs=refresh_inputs, outputs=refresh_outputs)

        # Initial load
        demo.load(fn=refresh_dashboard, inputs=refresh_inputs, outputs=refresh_outputs)

        # Auto-refresh timer (Gradio 5.x)
        timer = gr.Timer(value=CONFIG.refresh_interval, active=True)
        timer.tick(fn=refresh_dashboard, inputs=refresh_inputs, outputs=refresh_outputs)

        # CSV export
        export_btn.click(fn=export_to_csv, outputs=csv_output)

    return demo


def main() -> None:
    """Launch the dashboard."""
    demo = build_dashboard()
    demo.queue().launch(share=False)


if __name__ == "__main__":
    main()
