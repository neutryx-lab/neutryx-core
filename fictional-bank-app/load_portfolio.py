#!/usr/bin/env python3
"""Load the fictional bank portfolio into Neutryx.

This script loads the comprehensive fictional portfolio and can optionally
register it with the Neutryx API for XVA calculations.
"""
import json
import sys
import io
from pathlib import Path

# Fix encoding for Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from neutryx.tests.fixtures.fictional_portfolio import (
    create_fictional_portfolio,
    get_portfolio_summary,
)


def main():
    print("=" * 80)
    print("Loading Fictional Bank Portfolio")
    print("=" * 80)
    print()

    # Create the portfolio
    print("Creating portfolio with book hierarchy...")
    portfolio, book_hierarchy = create_fictional_portfolio()
    print(f"✓ Created portfolio: {portfolio.name}")
    print()

    # Display summary
    print("Portfolio Summary")
    print("-" * 80)
    summary = get_portfolio_summary(portfolio, book_hierarchy)

    print(f"Portfolio Name: {summary['portfolio_name']}")
    print(f"Base Currency: {summary['base_currency']}")
    print()

    stats = summary["statistics"]
    print(f"Counterparties: {stats['counterparties']}")
    print(f"Netting Sets: {stats['netting_sets']}")
    print(f"Total Trades: {stats['trades']}")
    print(f"Active Trades: {stats['active_trades']}")
    print()

    print(f"Total MTM: ${summary['total_mtm']:,.2f}")
    print(f"Gross Notional: ${summary['gross_notional']:,.2f}")
    print()

    # Counterparty breakdown
    print("Counterparty Breakdown")
    print("-" * 80)
    print(f"{'Counterparty':<40} {'Rating':<8} {'Trades':<8} {'Net MTM':<15} {'CSA':<5}")
    print("-" * 80)

    for cp_id, cp_info in summary["counterparties"].items():
        name = cp_info["name"][:38]
        rating = cp_info["rating"]
        num_trades = cp_info["num_trades"]
        net_mtm = cp_info["net_mtm"]
        has_csa = "Yes" if cp_info["has_csa"] else "No"

        print(f"{name:<40} {rating:<8} {num_trades:<8} ${net_mtm:>13,.2f} {has_csa:<5}")
    print()

    # Book breakdown
    print("Book Breakdown")
    print("-" * 80)
    print(f"{'Book Name':<40} {'Trades':<8} {'Active':<8} {'MTM':<15}")
    print("-" * 80)

    for book_id, book_info in summary["books"].items():
        name = book_info["name"][:38]
        num_trades = book_info["num_trades"]
        active_trades = book_info["active_trades"]
        mtm = book_info["total_mtm"]

        print(f"{name:<40} {num_trades:<8} {active_trades:<8} ${mtm:>13,.2f}")
    print()

    # Desk breakdown
    print("Desk Breakdown")
    print("-" * 80)
    print(f"{'Desk Name':<40} {'Books':<8} {'Trades':<8} {'MTM':<15}")
    print("-" * 80)

    for desk_id, desk_info in summary["desks"].items():
        name = desk_info["name"][:38]
        num_books = desk_info["num_books"]
        num_trades = desk_info["num_trades"]
        mtm = desk_info["total_mtm"]

        print(f"{name:<40} {num_books:<8} {num_trades:<8} ${mtm:>13,.2f}")
    print()

    # Save portfolio snapshot
    output_dir = Path(__file__).parent / "snapshots"
    output_dir.mkdir(exist_ok=True)

    snapshot_file = output_dir / "portfolio_snapshot.json"
    print(f"Saving portfolio snapshot to {snapshot_file}...")

    # Export portfolio to JSON
    portfolio_data = portfolio.model_dump(mode="json")

    with open(snapshot_file, "w") as f:
        json.dump(portfolio_data, f, indent=2, default=str)

    print(f"✓ Portfolio snapshot saved ({snapshot_file.stat().st_size:,} bytes)")
    print()

    # Display sample trades
    print("Sample Trades")
    print("-" * 80)

    sample_trades = list(portfolio.trades.values())[:5]
    for trade in sample_trades:
        print(f"Trade ID: {trade.id}")
        print(f"  Product: {trade.product_type.value}")
        print(f"  Counterparty: {portfolio.get_counterparty(trade.counterparty_id).name}")
        print(f"  Book: {trade.book_id}")
        print(f"  Notional: ${trade.notional:,.2f} {trade.currency}")
        print(f"  MTM: ${trade.mtm:,.2f}")
        print(f"  Maturity: {trade.maturity_date}")
        print()

    print("=" * 80)
    print("Portfolio loaded successfully!")
    print()
    print("Next steps:")
    print("1. Start the Neutryx API: uvicorn neutryx.api.rest:create_app --factory")
    print("2. Run compute_xva.py to calculate XVA metrics")
    print("3. Run portfolio_report.py to generate detailed reports")
    print("=" * 80)

    return portfolio, book_hierarchy


if __name__ == "__main__":
    main()
