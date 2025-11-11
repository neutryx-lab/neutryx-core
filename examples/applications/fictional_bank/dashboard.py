#!/usr/bin/env python3
"""Fictional Bank Portfolio - Interactive Dashboard.

A comprehensive Gradio-based dashboard that provides a complete overview of the
Fictional Bank portfolio with interactive visualizations and analysis tools.

Features:
- Portfolio Overview with key statistics
- Counterparty Analysis with credit ratings and CSA coverage
- Trade Details with filtering and search
- Risk Metrics and exposure analysis
- Interactive visualizations
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from neutryx.tests.fixtures.fictional_portfolio import (
    create_fictional_portfolio,
    get_portfolio_summary,
)


# Global state for portfolio data
PORTFOLIO_DATA = None
BOOK_HIERARCHY = None
SUMMARY_DATA = None


def load_portfolio_data():
    """Load portfolio data on startup."""
    global PORTFOLIO_DATA, BOOK_HIERARCHY, SUMMARY_DATA

    if PORTFOLIO_DATA is None:
        portfolio, book_hierarchy = create_fictional_portfolio()
        summary = get_portfolio_summary(portfolio, book_hierarchy)

        PORTFOLIO_DATA = portfolio
        BOOK_HIERARCHY = book_hierarchy
        SUMMARY_DATA = summary

    return PORTFOLIO_DATA, BOOK_HIERARCHY, SUMMARY_DATA


def format_currency(value: float) -> str:
    """Format value as USD currency."""
    return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """Format value as percentage."""
    return f"{value:.1f}%"


# ============================================================================
# Tab 1: Portfolio Overview
# ============================================================================

def get_portfolio_overview() -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Generate portfolio overview with key statistics."""
    portfolio, book_hierarchy, summary = load_portfolio_data()

    # Key statistics
    stats = summary["statistics"]
    overview_text = f"""
# Portfolio Overview

**Portfolio Name:** {summary['portfolio_name']}
**Base Currency:** {summary['base_currency']}
**As of Date:** {date.today().isoformat()}

## Key Statistics

| Metric | Value |
|--------|-------|
| Total Counterparties | {stats['counterparties']} |
| Netting Sets | {stats['netting_sets']} |
| Total Trades | {stats['trades']} |
| Active Trades | {stats['active_trades']} |
| **Total MTM** | **{format_currency(summary['total_mtm'])}** |
| **Gross Notional** | **{format_currency(summary['gross_notional'])}** |
"""

    # Counterparty summary table
    cp_data = []
    for cp_id, cp_info in summary["counterparties"].items():
        cp_data.append({
            "Counterparty": cp_info["name"],
            "Rating": cp_info["rating"],
            "Entity Type": cp_info["entity_type"],
            "Trades": cp_info["num_trades"],
            "Net MTM": format_currency(cp_info["net_mtm"]),
            "CSA": "Yes" if cp_info["has_csa"] else "No",
        })

    cp_df = pd.DataFrame(cp_data)

    # Book summary table
    book_data = []
    for book_id, book_info in summary["books"].items():
        book_data.append({
            "Book": book_info.get("name", book_id),
            "Desk": book_info.get("desk", "N/A"),
            "Trader": book_info.get("trader", "N/A"),
            "Trades": book_info.get("num_trades", 0),
            "Active": book_info.get("active_trades", 0),
            "MTM": format_currency(book_info.get("total_mtm", 0.0)),
        })

    book_df = pd.DataFrame(book_data)

    return overview_text, cp_df, book_df


# ============================================================================
# Tab 2: Counterparty Analysis
# ============================================================================

def get_counterparty_analysis() -> tuple[Any, Any, Any]:
    """Generate counterparty analysis with charts."""
    portfolio, book_hierarchy, summary = load_portfolio_data()

    # Credit rating distribution
    rating_counts = {}
    for cp_info in summary["counterparties"].values():
        rating = cp_info["rating"]
        rating_counts[rating] = rating_counts.get(rating, 0) + 1

    fig_ratings = px.pie(
        values=list(rating_counts.values()),
        names=list(rating_counts.keys()),
        title="Counterparty Credit Rating Distribution",
        hole=0.3,
    )

    # CSA coverage
    csa_counts = {"With CSA": 0, "Without CSA": 0}
    for cp_info in summary["counterparties"].values():
        if cp_info["has_csa"]:
            csa_counts["With CSA"] += 1
        else:
            csa_counts["Without CSA"] += 1

    fig_csa = px.bar(
        x=list(csa_counts.keys()),
        y=list(csa_counts.values()),
        title="CSA Coverage",
        labels={"x": "CSA Status", "y": "Number of Counterparties"},
        color=list(csa_counts.keys()),
        color_discrete_map={"With CSA": "#2ecc71", "Without CSA": "#e74c3c"},
    )

    # Net MTM by counterparty
    cp_names = []
    cp_mtm = []
    for cp_info in summary["counterparties"].values():
        cp_names.append(cp_info["name"][:30])  # Truncate long names
        cp_mtm.append(cp_info["net_mtm"])

    fig_mtm = px.bar(
        x=cp_names,
        y=cp_mtm,
        title="Net MTM by Counterparty",
        labels={"x": "Counterparty", "y": "Net MTM (USD)"},
        color=cp_mtm,
        color_continuous_scale="RdYlGn",
    )
    fig_mtm.update_xaxes(tickangle=-45)

    return fig_ratings, fig_csa, fig_mtm


# ============================================================================
# Tab 3: Trade Details
# ============================================================================

def get_trade_details(filter_product: str = "All") -> pd.DataFrame:
    """Generate detailed trade listing with filtering."""
    portfolio, book_hierarchy, summary = load_portfolio_data()

    trade_data = []
    for trade in portfolio.trades.values():
        # Basic filtering
        if filter_product != "All" and str(trade.product_type.value) != filter_product:
            continue

        # Get counterparty name
        cp = portfolio.counterparties.get(trade.counterparty_id)
        cp_name = cp.name if cp else trade.counterparty_id

        # Get netting set info
        ns = portfolio.netting_sets.get(trade.netting_set_id)
        has_csa = "Yes" if (ns and ns.csa_id) else "No"

        trade_data.append({
            "Trade ID": trade.id,
            "Counterparty": cp_name[:30],
            "Product": trade.product_type.value,
            "Notional": format_currency(trade.notional),
            "MTM": format_currency(trade.mtm) if trade.mtm else "N/A",
            "Trade Date": trade.trade_date.isoformat() if trade.trade_date else "N/A",
            "Maturity": trade.maturity_date.isoformat() if trade.maturity_date else "N/A",
            "Status": trade.status.value,
            "CSA": has_csa,
        })

    return pd.DataFrame(trade_data)


def get_product_types() -> list[str]:
    """Get list of unique product types for filtering."""
    portfolio, _, _ = load_portfolio_data()

    product_types = set()
    for trade in portfolio.trades.values():
        product_types.add(trade.product_type.value)

    return ["All"] + sorted(list(product_types))


# ============================================================================
# Tab 4: Risk Metrics & Visualizations
# ============================================================================

def get_risk_metrics() -> tuple[str, Any, Any]:
    """Generate risk metrics and exposure analysis."""
    portfolio, book_hierarchy, summary = load_portfolio_data()

    # Risk metrics summary
    stats = summary["statistics"]
    total_mtm = summary["total_mtm"]
    gross_notional = summary["gross_notional"]

    # Calculate some basic risk metrics
    num_counterparties = stats["counterparties"]
    avg_mtm_per_cp = total_mtm / num_counterparties if num_counterparties > 0 else 0

    # CSA coverage ratio
    csa_count = sum(1 for cp in summary["counterparties"].values() if cp["has_csa"])
    csa_ratio = (csa_count / num_counterparties * 100) if num_counterparties > 0 else 0

    metrics_text = f"""
# Risk Metrics

## Portfolio-Level Metrics

| Metric | Value |
|--------|-------|
| Total Gross Notional | {format_currency(gross_notional)} |
| Total Net MTM | {format_currency(total_mtm)} |
| Number of Counterparties | {num_counterparties} |
| Average MTM per Counterparty | {format_currency(avg_mtm_per_cp)} |
| CSA Coverage Ratio | {format_percentage(csa_ratio)} |
| Number of Netting Sets | {stats['netting_sets']} |

## Concentration Risk

- Largest counterparty exposure: {_get_largest_exposure(summary)}
- Top 3 counterparties represent: {_get_top_concentration(summary)}

## Credit Risk Distribution

- Investment Grade (AAA-BBB): {_count_by_rating_class(summary, 'IG')} counterparties
- Non-Investment Grade: {_count_by_rating_class(summary, 'HY')} counterparties
"""

    # Notional by product type
    product_notional = {}
    for trade in portfolio.trades.values():
        product = trade.product_type.value
        product_notional[product] = product_notional.get(product, 0) + trade.notional

    fig_products = px.pie(
        values=list(product_notional.values()),
        names=list(product_notional.keys()),
        title="Notional Breakdown by Product Type",
        hole=0.4,
    )

    # MTM by desk
    desk_mtm = {}
    for book_id, book_info in summary["books"].items():
        desk = book_info.get("desk", "Unknown")
        desk_mtm[desk] = desk_mtm.get(desk, 0) + book_info.get("total_mtm", 0.0)

    fig_desks = px.bar(
        x=list(desk_mtm.keys()),
        y=list(desk_mtm.values()),
        title="MTM by Trading Desk",
        labels={"x": "Desk", "y": "MTM (USD)"},
        color=list(desk_mtm.values()),
        color_continuous_scale="Viridis",
    )

    return metrics_text, fig_products, fig_desks


def _get_largest_exposure(summary: dict) -> str:
    """Get largest counterparty exposure."""
    max_mtm = 0
    max_cp = ""
    for cp_info in summary["counterparties"].values():
        if abs(cp_info["net_mtm"]) > abs(max_mtm):
            max_mtm = cp_info["net_mtm"]
            max_cp = cp_info["name"]
    return f"{max_cp} ({format_currency(max_mtm)})"


def _get_top_concentration(summary: dict) -> str:
    """Calculate top 3 counterparty concentration."""
    mtm_values = sorted(
        [abs(cp["net_mtm"]) for cp in summary["counterparties"].values()],
        reverse=True
    )
    top_3 = sum(mtm_values[:3])
    total = sum(mtm_values)
    pct = (top_3 / total * 100) if total > 0 else 0
    return f"{format_percentage(pct)} of total exposure"


def _count_by_rating_class(summary: dict, rating_class: str) -> int:
    """Count counterparties by rating class."""
    ig_ratings = {"AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-"}
    count = 0

    for cp_info in summary["counterparties"].values():
        rating = cp_info["rating"]
        is_ig = rating in ig_ratings

        if rating_class == "IG" and is_ig:
            count += 1
        elif rating_class == "HY" and not is_ig:
            count += 1

    return count


# ============================================================================
# Tab 5: Reports & Export
# ============================================================================

def generate_json_report() -> str:
    """Generate JSON report of portfolio summary."""
    import json
    portfolio, book_hierarchy, summary = load_portfolio_data()

    # Create simplified summary for export
    export_data = {
        "portfolio_name": summary["portfolio_name"],
        "base_currency": summary["base_currency"],
        "generated_date": date.today().isoformat(),
        "statistics": summary["statistics"],
        "total_mtm": summary["total_mtm"],
        "gross_notional": summary["gross_notional"],
        "counterparties": [
            {
                "name": cp["name"],
                "rating": cp["rating"],
                "entity_type": cp["entity_type"],
                "num_trades": cp["num_trades"],
                "net_mtm": cp["net_mtm"],
                "has_csa": cp["has_csa"],
            }
            for cp in summary["counterparties"].values()
        ],
        "books": [
            {
                "name": book.get("name", "Unknown"),
                "desk": book.get("desk", "N/A"),
                "trader": book.get("trader", "N/A"),
                "num_trades": book.get("num_trades", 0),
                "total_mtm": book.get("total_mtm", 0.0),
            }
            for book in summary["books"].values()
        ],
    }

    return json.dumps(export_data, indent=2)


def generate_csv_report() -> str:
    """Generate CSV report of trades."""
    portfolio, book_hierarchy, summary = load_portfolio_data()

    df = get_trade_details("All")
    return df.to_csv(index=False)


# ============================================================================
# Main Dashboard Interface
# ============================================================================

def build_dashboard() -> gr.Blocks:
    """Build the complete dashboard interface."""

    with gr.Blocks(
        title="Fictional Bank Portfolio Dashboard",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown("""
        # Fictional Bank Portfolio Dashboard

        **Comprehensive Analytics for Global Trading Portfolio**

        This dashboard provides complete visibility into the Fictional Bank's trading
        portfolio, including counterparty analysis, trade details, risk metrics, and
        interactive visualizations.
        """)

        # Refresh button
        refresh_btn = gr.Button("🔄 Refresh All Data", variant="secondary", size="sm")

        with gr.Tabs():

            # Tab 1: Portfolio Overview
            with gr.Tab("📊 Portfolio Overview"):
                gr.Markdown("### Portfolio Summary and Key Statistics")

                overview_text = gr.Markdown()

                gr.Markdown("### Counterparty Summary")
                cp_table = gr.Dataframe(
                    interactive=False,
                    wrap=True,
                )

                gr.Markdown("### Book Summary")
                book_table = gr.Dataframe(
                    interactive=False,
                    wrap=True,
                )

                load_overview_btn = gr.Button("Load Overview", variant="primary")
                load_overview_btn.click(
                    fn=get_portfolio_overview,
                    outputs=[overview_text, cp_table, book_table],
                )

            # Tab 2: Counterparty Analysis
            with gr.Tab("🏢 Counterparty Analysis"):
                gr.Markdown("### Interactive Counterparty Analytics")

                with gr.Row():
                    plot_ratings = gr.Plot(label="Credit Rating Distribution")
                    plot_csa = gr.Plot(label="CSA Coverage")

                plot_mtm = gr.Plot(label="Net MTM by Counterparty")

                load_cp_btn = gr.Button("Load Analysis", variant="primary")
                load_cp_btn.click(
                    fn=get_counterparty_analysis,
                    outputs=[plot_ratings, plot_csa, plot_mtm],
                )

            # Tab 3: Trade Details
            with gr.Tab("📋 Trade Details"):
                gr.Markdown("### Trade Listing and Filtering")

                with gr.Row():
                    product_filter = gr.Dropdown(
                        choices=["All"],
                        value="All",
                        label="Filter by Product Type",
                        interactive=True,
                    )
                    load_trades_btn = gr.Button("Load Trades", variant="primary")

                trade_table = gr.Dataframe(
                    interactive=False,
                    wrap=True,
                    max_height=600,
                )

                # Load product types
                def update_filters():
                    return gr.Dropdown(choices=get_product_types())

                demo.load(fn=update_filters, outputs=product_filter)

                load_trades_btn.click(
                    fn=get_trade_details,
                    inputs=product_filter,
                    outputs=trade_table,
                )

            # Tab 4: Risk Metrics
            with gr.Tab("⚠️ Risk Metrics"):
                gr.Markdown("### Portfolio Risk Analysis")

                metrics_text = gr.Markdown()

                with gr.Row():
                    plot_products = gr.Plot(label="Notional by Product")
                    plot_desks = gr.Plot(label="MTM by Desk")

                load_risk_btn = gr.Button("Load Risk Metrics", variant="primary")
                load_risk_btn.click(
                    fn=get_risk_metrics,
                    outputs=[metrics_text, plot_products, plot_desks],
                )

            # Tab 5: Reports & Export
            with gr.Tab("📄 Reports & Export"):
                gr.Markdown("### Generate and Export Reports")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### JSON Report")
                        json_output = gr.Code(language="json", label="JSON Report")
                        gen_json_btn = gr.Button("Generate JSON", variant="primary")
                        gen_json_btn.click(
                            fn=generate_json_report,
                            outputs=json_output,
                        )

                    with gr.Column():
                        gr.Markdown("#### CSV Trade Report")
                        csv_output = gr.Textbox(
                            label="CSV Report",
                            lines=20,
                            max_lines=30,
                            show_copy_button=True,
                        )
                        gen_csv_btn = gr.Button("Generate CSV", variant="primary")
                        gen_csv_btn.click(
                            fn=generate_csv_report,
                            outputs=csv_output,
                        )

                gr.Markdown("""
                ### Report Notes

                - **JSON Report**: Complete portfolio summary with counterparty and book details
                - **CSV Report**: Detailed trade-level data for spreadsheet analysis
                - Use the browser's download feature to save reports locally
                """)

        # Auto-load overview on startup
        demo.load(
            fn=get_portfolio_overview,
            outputs=[overview_text, cp_table, book_table],
        )

        # Refresh functionality
        def refresh_all():
            global PORTFOLIO_DATA, BOOK_HIERARCHY, SUMMARY_DATA
            PORTFOLIO_DATA = None
            BOOK_HIERARCHY = None
            SUMMARY_DATA = None
            return get_portfolio_overview()

        refresh_btn.click(
            fn=refresh_all,
            outputs=[overview_text, cp_table, book_table],
        )

    return demo


def main() -> None:
    """Launch the dashboard."""
    print("=" * 80)
    print("Fictional Bank Portfolio Dashboard")
    print("=" * 80)
    print()
    print("Loading portfolio data...")

    # Pre-load data
    load_portfolio_data()
    print("✓ Portfolio data loaded")
    print()

    # Build and launch dashboard
    demo = build_dashboard()
    print("Starting dashboard server...")
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7862,
        show_error=True,
    )


if __name__ == "__main__":
    main()
