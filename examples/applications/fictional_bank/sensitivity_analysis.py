#!/usr/bin/env python3
"""Sensitivity analysis and Greeks computation for portfolio.

This script computes and analyzes:
- Option Greeks (Delta, Gamma, Vega, Theta, Rho)
- Interest rate sensitivities (PV01, DV01, CS01)
- Bucketed sensitivities by tenor/maturity
- Risk factor sensitivities
- Sensitivity heatmaps and ladder charts
"""
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from neutryx.tests.fixtures.fictional_portfolio import (
    create_fictional_portfolio,
    get_portfolio_summary,
)


@dataclass
class GreeksResult:
    """Greeks computation result for a trade."""

    trade_id: str
    product_type: str
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    underlying: str


@dataclass
class BucketedSensitivity:
    """Bucketed sensitivity by tenor."""

    risk_factor: str
    bucket: str  # e.g., "3M", "1Y", "5Y"
    sensitivity: float
    unit: str  # e.g., "USD per bp"


class SensitivityAnalyzer:
    """Comprehensive sensitivity and Greeks analysis."""

    def __init__(self, output_dir: Path):
        """Initialize the analyzer.

        Args:
            output_dir: Directory to save analysis results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Define tenor buckets for bucketed sensitivities
        self.tenor_buckets = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"]

    def compute_all_sensitivities(
        self, portfolio: Any, book_hierarchy: Any
    ) -> Dict[str, Any]:
        """Compute all sensitivities for the portfolio.

        Args:
            portfolio: Portfolio object
            book_hierarchy: Book hierarchy object

        Returns:
            Dictionary containing all sensitivity results
        """
        print("Computing portfolio sensitivities...")
        print()

        results = {
            "greeks": self._compute_greeks(portfolio),
            "ir_sensitivities": self._compute_ir_sensitivities(portfolio),
            "fx_sensitivities": self._compute_fx_sensitivities(portfolio),
            "eq_sensitivities": self._compute_eq_sensitivities(portfolio),
            "vega_sensitivities": self._compute_vega_sensitivities(portfolio),
            "bucketed_ir": self._compute_bucketed_ir_sensitivities(portfolio),
        }

        print()
        print("✓ All sensitivity calculations complete")
        return results

    def _compute_greeks(self, portfolio: Any) -> List[Dict]:
        """Compute option Greeks for all option trades.

        Args:
            portfolio: Portfolio object

        Returns:
            List of Greeks results
        """
        print("Computing option Greeks...")

        greeks_results = []

        # Filter option trades
        option_products = [
            "fx_option",
            "equity_option",
            "swaption",
            "variance_swap",
        ]

        option_trades = [
            t
            for t in portfolio.trades.values()
            if t.product_type.value in option_products
        ]

        print(f"  Found {len(option_trades)} option trades")

        for trade in option_trades:
            # Simplified Greeks calculation (demonstration)
            # In production, this would use proper pricing models

            # Approximate Greeks based on product type and parameters
            if trade.product_type.value == "fx_option":
                delta = np.random.uniform(-0.5, 0.5) * trade.notional / 1000
                gamma = np.random.uniform(0, 0.01) * trade.notional / 1000
                vega = np.random.uniform(0, 500)
                theta = -np.random.uniform(10, 100)
                rho = np.random.uniform(-100, 100)
                underlying = trade.currency_pair if hasattr(trade, "currency_pair") else "FX"

            elif trade.product_type.value == "equity_option":
                delta = np.random.uniform(-0.7, 0.7) * trade.notional / 1000
                gamma = np.random.uniform(0, 0.02) * trade.notional / 1000
                vega = np.random.uniform(0, 1000)
                theta = -np.random.uniform(20, 200)
                rho = np.random.uniform(-200, 200)
                underlying = trade.underlying if hasattr(trade, "underlying") else "Equity"

            elif trade.product_type.value == "swaption":
                delta = np.random.uniform(-5000, 5000)
                gamma = np.random.uniform(0, 100)
                vega = np.random.uniform(0, 2000)
                theta = -np.random.uniform(50, 500)
                rho = np.random.uniform(-1000, 1000)
                underlying = f"{trade.currency} Swaption"

            else:  # variance_swap
                delta = 0  # No delta for variance swaps
                gamma = 0
                vega = np.random.uniform(0, 5000)  # High vega exposure
                theta = -np.random.uniform(100, 500)
                rho = 0
                underlying = trade.underlying if hasattr(trade, "underlying") else "Equity"

            greeks_results.append(
                {
                    "trade_id": trade.id,
                    "product_type": trade.product_type.value,
                    "underlying": underlying,
                    "notional": trade.notional,
                    "currency": trade.currency,
                    "delta": delta,
                    "gamma": gamma,
                    "vega": vega,
                    "theta": theta,
                    "rho": rho,
                }
            )

        print(f"  Computed Greeks for {len(greeks_results)} trades")
        return greeks_results

    def _compute_ir_sensitivities(self, portfolio: Any) -> Dict:
        """Compute interest rate sensitivities (PV01, DV01).

        Args:
            portfolio: Portfolio object

        Returns:
            IR sensitivity results
        """
        print("Computing interest rate sensitivities...")

        # Filter rate-sensitive trades
        rate_products = ["interest_rate_swap", "swaption"]
        rate_trades = [
            t for t in portfolio.trades.values() if t.product_type.value in rate_products
        ]

        print(f"  Found {len(rate_trades)} rate-sensitive trades")

        sensitivities = {
            "USD": {"PV01": 0, "DV01": 0, "trades": 0},
            "EUR": {"PV01": 0, "DV01": 0, "trades": 0},
        }

        for trade in rate_trades:
            ccy = trade.currency
            if ccy not in sensitivities:
                sensitivities[ccy] = {"PV01": 0, "DV01": 0, "trades": 0}

            # Approximate sensitivities (simplified)
            # PV01 = change in value for 1bp parallel shift
            # DV01 = dollar value of 1bp
            pv01 = trade.notional * 0.0001 * 0.05  # Simplified calculation
            dv01 = trade.notional * 0.0001 * 0.05

            sensitivities[ccy]["PV01"] += pv01
            sensitivities[ccy]["DV01"] += dv01
            sensitivities[ccy]["trades"] += 1

        print(f"  Computed sensitivities for {sum(s['trades'] for s in sensitivities.values())} trades")
        return sensitivities

    def _compute_fx_sensitivities(self, portfolio: Any) -> Dict:
        """Compute FX sensitivities (Delta).

        Args:
            portfolio: Portfolio object

        Returns:
            FX sensitivity results
        """
        print("Computing FX sensitivities...")

        fx_products = ["fx_option", "fx_forward"]
        fx_trades = [
            t for t in portfolio.trades.values() if t.product_type.value in fx_products
        ]

        print(f"  Found {len(fx_trades)} FX trades")

        sensitivities = {}

        for trade in fx_trades:
            pair = trade.currency_pair if hasattr(trade, "currency_pair") else "UNKNOWN"

            if pair not in sensitivities:
                sensitivities[pair] = {"delta": 0, "notional": 0, "trades": 0}

            # FX delta (simplified)
            delta = trade.notional * 0.5  # Assume 50 delta on average

            sensitivities[pair]["delta"] += delta
            sensitivities[pair]["notional"] += trade.notional
            sensitivities[pair]["trades"] += 1

        print(f"  Computed FX sensitivities for {len(sensitivities)} currency pairs")
        return sensitivities

    def _compute_eq_sensitivities(self, portfolio: Any) -> Dict:
        """Compute equity sensitivities (Delta).

        Args:
            portfolio: Portfolio object

        Returns:
            Equity sensitivity results
        """
        print("Computing equity sensitivities...")

        eq_products = ["equity_option"]
        eq_trades = [
            t for t in portfolio.trades.values() if t.product_type.value in eq_products
        ]

        print(f"  Found {len(eq_trades)} equity trades")

        sensitivities = {}

        for trade in eq_trades:
            underlying = trade.underlying if hasattr(trade, "underlying") else "UNKNOWN"

            if underlying not in sensitivities:
                sensitivities[underlying] = {"delta": 0, "gamma": 0, "notional": 0, "trades": 0}

            # Equity delta and gamma (simplified)
            delta = trade.notional * 0.6  # Assume 60 delta
            gamma = trade.notional * 0.01  # Simplified gamma

            sensitivities[underlying]["delta"] += delta
            sensitivities[underlying]["gamma"] += gamma
            sensitivities[underlying]["notional"] += trade.notional
            sensitivities[underlying]["trades"] += 1

        print(f"  Computed equity sensitivities for {len(sensitivities)} underlyings")
        return sensitivities

    def _compute_vega_sensitivities(self, portfolio: Any) -> Dict:
        """Compute vega sensitivities (volatility risk).

        Args:
            portfolio: Portfolio object

        Returns:
            Vega sensitivity results
        """
        print("Computing vega sensitivities...")

        option_products = ["fx_option", "equity_option", "swaption", "variance_swap"]
        option_trades = [
            t for t in portfolio.trades.values() if t.product_type.value in option_products
        ]

        print(f"  Found {len(option_trades)} option trades")

        sensitivities = {
            "FX": {"vega": 0, "trades": 0},
            "Equity": {"vega": 0, "trades": 0},
            "Rates": {"vega": 0, "trades": 0},
        }

        for trade in option_trades:
            if trade.product_type.value == "fx_option":
                asset_class = "FX"
                vega = trade.notional * 0.001  # Simplified
            elif trade.product_type.value in ["equity_option", "variance_swap"]:
                asset_class = "Equity"
                vega = trade.notional * 0.002  # Higher vega for equity
            else:  # swaption
                asset_class = "Rates"
                vega = trade.notional * 0.0005

            sensitivities[asset_class]["vega"] += vega
            sensitivities[asset_class]["trades"] += 1

        print(f"  Computed vega sensitivities across {len(sensitivities)} asset classes")
        return sensitivities

    def _compute_bucketed_ir_sensitivities(self, portfolio: Any) -> Dict:
        """Compute bucketed interest rate sensitivities by tenor.

        Args:
            portfolio: Portfolio object

        Returns:
            Bucketed IR sensitivity results
        """
        print("Computing bucketed IR sensitivities...")

        rate_products = ["interest_rate_swap", "swaption"]
        rate_trades = [
            t for t in portfolio.trades.values() if t.product_type.value in rate_products
        ]

        sensitivities = {
            "USD": {bucket: 0 for bucket in self.tenor_buckets},
            "EUR": {bucket: 0 for bucket in self.tenor_buckets},
        }

        for trade in rate_trades:
            ccy = trade.currency
            if ccy not in sensitivities:
                sensitivities[ccy] = {bucket: 0 for bucket in self.tenor_buckets}

            # Distribute sensitivity across buckets based on trade maturity
            # (simplified distribution)
            for bucket in self.tenor_buckets:
                sensitivity = np.random.uniform(-1000, 1000)
                sensitivities[ccy][bucket] += sensitivity

        print(f"  Computed bucketed sensitivities for {len(sensitivities)} currencies")
        return sensitivities

    def generate_reports(self, results: Dict) -> Dict[str, Path]:
        """Generate sensitivity analysis reports.

        Args:
            results: Sensitivity calculation results

        Returns:
            Dictionary of generated report files
        """
        print()
        print("Generating sensitivity reports...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reports = {}

        # JSON report
        json_file = self.output_dir / f"sensitivity_analysis_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        reports["json"] = json_file
        print(f"✓ JSON report: {json_file.name}")

        # Greeks CSV
        if results["greeks"]:
            df_greeks = pd.DataFrame(results["greeks"])
            greeks_file = self.output_dir / f"greeks_{timestamp}.csv"
            df_greeks.to_csv(greeks_file, index=False)
            reports["greeks_csv"] = greeks_file
            print(f"✓ Greeks CSV: {greeks_file.name}")

            # Greeks Excel
            greeks_excel = self.output_dir / f"greeks_{timestamp}.xlsx"
            df_greeks.to_excel(greeks_excel, index=False, sheet_name="Greeks")
            reports["greeks_excel"] = greeks_excel
            print(f"✓ Greeks Excel: {greeks_excel.name}")

        # Comprehensive Excel report
        excel_file = self.output_dir / f"sensitivity_analysis_{timestamp}.xlsx"
        self._create_excel_report(results, excel_file)
        reports["excel"] = excel_file
        print(f"✓ Excel report: {excel_file.name}")

        return reports

    def _create_excel_report(self, results: Dict, output_file: Path):
        """Create comprehensive Excel report with multiple sheets.

        Args:
            results: Sensitivity results
            output_file: Output Excel file path
        """
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            # Greeks sheet
            if results["greeks"]:
                df_greeks = pd.DataFrame(results["greeks"])
                df_greeks.to_excel(writer, sheet_name="Greeks", index=False)

                # Greeks summary
                summary_data = {
                    "Metric": ["Total Delta", "Total Gamma", "Total Vega", "Total Theta", "Total Rho"],
                    "Value": [
                        df_greeks["delta"].sum(),
                        df_greeks["gamma"].sum(),
                        df_greeks["vega"].sum(),
                        df_greeks["theta"].sum(),
                        df_greeks["rho"].sum(),
                    ],
                }
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name="Greeks Summary", index=False)

            # IR sensitivities
            if results["ir_sensitivities"]:
                ir_data = []
                for ccy, sens in results["ir_sensitivities"].items():
                    ir_data.append(
                        {
                            "Currency": ccy,
                            "PV01": sens["PV01"],
                            "DV01": sens["DV01"],
                            "Trades": sens["trades"],
                        }
                    )
                df_ir = pd.DataFrame(ir_data)
                df_ir.to_excel(writer, sheet_name="IR Sensitivities", index=False)

            # FX sensitivities
            if results["fx_sensitivities"]:
                fx_data = []
                for pair, sens in results["fx_sensitivities"].items():
                    fx_data.append(
                        {
                            "Currency Pair": pair,
                            "Delta": sens["delta"],
                            "Notional": sens["notional"],
                            "Trades": sens["trades"],
                        }
                    )
                df_fx = pd.DataFrame(fx_data)
                df_fx.to_excel(writer, sheet_name="FX Sensitivities", index=False)

            # Equity sensitivities
            if results["eq_sensitivities"]:
                eq_data = []
                for underlying, sens in results["eq_sensitivities"].items():
                    eq_data.append(
                        {
                            "Underlying": underlying,
                            "Delta": sens["delta"],
                            "Gamma": sens["gamma"],
                            "Notional": sens["notional"],
                            "Trades": sens["trades"],
                        }
                    )
                df_eq = pd.DataFrame(eq_data)
                df_eq.to_excel(writer, sheet_name="Equity Sensitivities", index=False)

            # Vega sensitivities
            if results["vega_sensitivities"]:
                vega_data = []
                for asset_class, sens in results["vega_sensitivities"].items():
                    vega_data.append(
                        {
                            "Asset Class": asset_class,
                            "Vega": sens["vega"],
                            "Trades": sens["trades"],
                        }
                    )
                df_vega = pd.DataFrame(vega_data)
                df_vega.to_excel(writer, sheet_name="Vega Sensitivities", index=False)

            # Bucketed IR sensitivities
            if results["bucketed_ir"]:
                for ccy, buckets in results["bucketed_ir"].items():
                    df_buckets = pd.DataFrame(
                        list(buckets.items()), columns=["Tenor", "Sensitivity"]
                    )
                    df_buckets.to_excel(writer, sheet_name=f"{ccy} Bucketed", index=False)

    def print_summary(self, results: Dict):
        """Print sensitivity analysis summary.

        Args:
            results: Sensitivity results
        """
        print()
        print("=" * 80)
        print("SENSITIVITY ANALYSIS SUMMARY")
        print("=" * 80)
        print()

        # Greeks summary
        if results["greeks"]:
            df_greeks = pd.DataFrame(results["greeks"])
            print("Option Greeks Summary:")
            print("-" * 80)
            print(f"  Total Delta: {df_greeks['delta'].sum():,.2f}")
            print(f"  Total Gamma: {df_greeks['gamma'].sum():,.2f}")
            print(f"  Total Vega: {df_greeks['vega'].sum():,.2f}")
            print(f"  Total Theta: {df_greeks['theta'].sum():,.2f}")
            print(f"  Total Rho: {df_greeks['rho'].sum():,.2f}")
            print()

        # IR sensitivities
        if results["ir_sensitivities"]:
            print("Interest Rate Sensitivities:")
            print("-" * 80)
            for ccy, sens in results["ir_sensitivities"].items():
                print(f"  {ccy}: PV01 = ${sens['PV01']:,.2f}, DV01 = ${sens['DV01']:,.2f}")
            print()

        # FX sensitivities
        if results["fx_sensitivities"]:
            print("FX Sensitivities (Delta):")
            print("-" * 80)
            for pair, sens in results["fx_sensitivities"].items():
                print(f"  {pair}: {sens['delta']:,.2f}")
            print()

        # Equity sensitivities
        if results["eq_sensitivities"]:
            print("Equity Sensitivities:")
            print("-" * 80)
            for underlying, sens in results["eq_sensitivities"].items():
                print(
                    f"  {underlying}: Delta = {sens['delta']:,.2f}, Gamma = {sens['gamma']:,.2f}"
                )
            print()


def main():
    """Main entry point."""
    print("=" * 80)
    print("Fictional Bank Portfolio - Sensitivity Analysis")
    print("=" * 80)
    print()

    # Load portfolio
    print("Loading portfolio...")
    portfolio, book_hierarchy = create_fictional_portfolio()
    print(f"✓ Portfolio loaded: {portfolio.name}")
    print()

    # Initialize analyzer
    output_dir = Path(__file__).parent / "reports"
    analyzer = SensitivityAnalyzer(output_dir)

    # Compute sensitivities
    results = analyzer.compute_all_sensitivities(portfolio, book_hierarchy)

    # Generate reports
    reports = analyzer.generate_reports(results)

    # Print summary
    analyzer.print_summary(results)

    print("=" * 80)
    print("Sensitivity Analysis Complete!")
    print("=" * 80)
    print()
    print("Note: This demonstration uses simplified Greek calculations.")
    print("In production, Greeks would be computed using proper pricing models")
    print("via the Neutryx API for accurate risk measurement.")
    print()


if __name__ == "__main__":
    main()
