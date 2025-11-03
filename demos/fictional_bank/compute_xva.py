#!/usr/bin/env python3
"""Compute XVA metrics for the fictional bank portfolio via API.

This script demonstrates how to:
1. Load and register a portfolio with the Neutryx API
2. Compute XVA metrics at portfolio and netting set levels
3. Generate XVA reports and analysis
"""
import json
import sys
from pathlib import Path

import requests

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from neutryx.tests.fixtures.fictional_portfolio import create_fictional_portfolio

# API configuration
API_BASE_URL = "http://localhost:8000"


def check_api_health() -> bool:
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/docs", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def register_portfolio(portfolio) -> dict:
    """Register portfolio with the API."""
    print("Registering portfolio with API...")

    portfolio_data = portfolio.model_dump(mode="json")

    response = requests.post(
        f"{API_BASE_URL}/portfolio/register",
        json=portfolio_data,
        headers={"Content-Type": "application/json"},
    )

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        sys.exit(1)

    result = response.json()
    print(f"✓ Portfolio registered: {result['portfolio_id']}")
    print(f"  Summary: {result['summary']}")
    print()
    return result


def get_portfolio_summary(portfolio_id: str) -> dict:
    """Get portfolio summary from API."""
    print(f"Fetching portfolio summary for '{portfolio_id}'...")

    response = requests.get(f"{API_BASE_URL}/portfolio/{portfolio_id}/summary")

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        sys.exit(1)

    return response.json()


def get_netting_sets(portfolio_id: str) -> dict:
    """Get netting sets from API."""
    print(f"Fetching netting sets for '{portfolio_id}'...")

    response = requests.get(f"{API_BASE_URL}/portfolio/{portfolio_id}/netting-sets")

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        sys.exit(1)

    return response.json()


def compute_portfolio_xva(
    portfolio_id: str,
    netting_set_id: str | None = None,
    compute_all: bool = True,
) -> dict:
    """Compute XVA for portfolio or netting set."""
    scope = f"netting set '{netting_set_id}'" if netting_set_id else "entire portfolio"
    print(f"Computing XVA for {scope}...")

    request_data = {
        "portfolio_id": portfolio_id,
        "valuation_date": "2024-01-15",
        "compute_cva": compute_all,
        "compute_dva": compute_all,
        "compute_fva": compute_all,
        "compute_mva": compute_all,
        "lgd": 0.6,
        "funding_spread_bps": 50.0,
    }

    if netting_set_id:
        request_data["netting_set_id"] = netting_set_id

    response = requests.post(
        f"{API_BASE_URL}/portfolio/xva",
        json=request_data,
        headers={"Content-Type": "application/json"},
    )

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        sys.exit(1)

    return response.json()


def main():
    print("=" * 80)
    print("Fictional Bank Portfolio - XVA Calculation")
    print("=" * 80)
    print()

    # Check if API is running
    if not check_api_health():
        print("Error: Neutryx API is not running!")
        print()
        print("Please start the API with:")
        print("  uvicorn neutryx.api.rest:create_app --factory --reload")
        print()
        sys.exit(1)

    print("✓ API is running")
    print()

    # Create and load portfolio
    print("Loading portfolio...")
    portfolio, book_hierarchy = create_fictional_portfolio()
    print(f"✓ Loaded portfolio: {portfolio.name}")
    print()

    # Register portfolio
    register_result = register_portfolio(portfolio)
    portfolio_id = register_result["portfolio_id"]

    # Get portfolio summary
    summary = get_portfolio_summary(portfolio_id)
    print()
    print("Portfolio Statistics")
    print("-" * 80)
    print(f"Counterparties: {summary['counterparties']}")
    print(f"Netting Sets: {summary['netting_sets']}")
    print(f"Total Trades: {summary['trades']}")
    print(f"Total MTM: ${summary['total_mtm']:,.2f}")
    print(f"Gross Notional: ${summary['gross_notional']:,.2f}")
    print()

    # Compute portfolio-level XVA
    print("=" * 80)
    print("Portfolio-Level XVA")
    print("=" * 80)
    print()

    portfolio_xva = compute_portfolio_xva(portfolio_id)

    print(f"Scope: {portfolio_xva['scope']}")
    print(f"Number of Trades: {portfolio_xva['num_trades']}")
    print(f"Net MTM: ${portfolio_xva['net_mtm']:,.2f}")
    print(f"Positive Exposure: ${portfolio_xva['positive_exposure']:,.2f}")
    print(f"Negative Exposure: ${portfolio_xva['negative_exposure']:,.2f}")
    print()

    print("XVA Components:")
    print(f"  CVA: ${portfolio_xva['cva']:,.2f}")
    print(f"  DVA: ${portfolio_xva['dva']:,.2f}")
    print(f"  FVA: ${portfolio_xva['fva']:,.2f}")
    print(f"  MVA: ${portfolio_xva['mva']:,.2f}")
    print()
    print(f"Total XVA: ${portfolio_xva['total_xva']:,.2f}")
    print()

    # Get netting sets
    netting_sets = get_netting_sets(portfolio_id)
    print()

    # Compute XVA for each netting set
    print("=" * 80)
    print("Netting Set XVA Analysis")
    print("=" * 80)
    print()

    xva_results = []

    for ns_info in netting_sets["netting_sets"]:
        ns_id = ns_info["netting_set_id"]
        cp_name = ns_info["counterparty_name"]
        has_csa = ns_info["has_csa"]

        print(f"Netting Set: {ns_id}")
        print(f"  Counterparty: {cp_name}")
        print(f"  CSA: {'Yes' if has_csa else 'No'}")
        print(f"  Trades: {ns_info['num_trades']}")
        print(f"  Net MTM: ${ns_info['net_mtm']:,.2f}")
        print()

        # Compute XVA
        ns_xva = compute_portfolio_xva(portfolio_id, netting_set_id=ns_id)

        print(f"  XVA Results:")
        print(f"    CVA: ${ns_xva['cva']:,.2f}")
        print(f"    DVA: ${ns_xva['dva']:,.2f}")
        print(f"    FVA: ${ns_xva['fva']:,.2f}")
        print(f"    MVA: ${ns_xva['mva']:,.2f}")
        print(f"    Total: ${ns_xva['total_xva']:,.2f}")
        print()

        xva_results.append(
            {
                "netting_set_id": ns_id,
                "counterparty": cp_name,
                "has_csa": has_csa,
                **ns_xva,
            }
        )

    # Summary table
    print("=" * 80)
    print("XVA Summary by Netting Set")
    print("=" * 80)
    print()
    print(
        f"{'Counterparty':<30} {'CSA':<5} {'CVA':<12} {'DVA':<12} {'FVA':<12} {'Total XVA':<12}"
    )
    print("-" * 80)

    for result in xva_results:
        cp_name = result["counterparty"][:28]
        csa = "Yes" if result["has_csa"] else "No"
        cva = result["cva"]
        dva = result["dva"]
        fva = result["fva"]
        total = result["total_xva"]

        print(
            f"{cp_name:<30} {csa:<5} ${cva:>10,.0f} ${dva:>10,.0f} ${fva:>10,.0f} ${total:>10,.0f}"
        )
    print()

    # Save results
    output_dir = Path(__file__).parent / "reports"
    output_dir.mkdir(exist_ok=True)

    results_file = output_dir / "xva_results.json"
    print(f"Saving XVA results to {results_file}...")

    results_data = {
        "portfolio_id": portfolio_id,
        "valuation_date": portfolio_xva["valuation_date"],
        "portfolio_xva": portfolio_xva,
        "netting_set_xva": xva_results,
    }

    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2, default=str)

    print(f"✓ Results saved")
    print()

    # Key insights
    print("=" * 80)
    print("Key Insights")
    print("=" * 80)
    print()

    total_cva = sum(r["cva"] for r in xva_results)
    total_dva = sum(r["dva"] for r in xva_results)
    total_fva = sum(r["fva"] for r in xva_results)

    print(f"1. Total CVA across all netting sets: ${total_cva:,.2f}")
    print(f"2. Total DVA across all netting sets: ${total_dva:,.2f}")
    print(f"3. Total FVA across all netting sets: ${total_fva:,.2f}")
    print()

    # CSA impact
    csa_results = [r for r in xva_results if r["has_csa"]]
    non_csa_results = [r for r in xva_results if not r["has_csa"]]

    avg_csa_xva = sum(r["total_xva"] for r in csa_results) / len(csa_results) if csa_results else 0
    avg_non_csa_xva = (
        sum(r["total_xva"] for r in non_csa_results) / len(non_csa_results)
        if non_csa_results
        else 0
    )

    print(f"4. Average XVA for CSA netting sets: ${avg_csa_xva:,.2f}")
    print(f"5. Average XVA for non-CSA netting sets: ${avg_non_csa_xva:,.2f}")
    print()

    if avg_non_csa_xva > 0:
        benefit = (avg_non_csa_xva - avg_csa_xva) / avg_non_csa_xva * 100
        print(f"6. CSA reduces XVA by approximately {benefit:.1f}%")
    print()

    print("=" * 80)
    print("XVA Calculation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
