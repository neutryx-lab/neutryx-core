#!/usr/bin/env python3
"""Standalone XVA Calculator - No API Required

This script demonstrates comprehensive XVA calculations:
- CVA (Credit Valuation Adjustment)
- DVA (Debit Valuation Adjustment)
- FVA (Funding Valuation Adjustment)
- MVA (Margin Valuation Adjustment)

Features:
- Monte Carlo simulation for exposure profiles
- Netting set-level calculations
- CSA impact analysis
- Comprehensive visualizations
- Interactive dashboards
- Detailed reporting
"""
import json
import sys
import io
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

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

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


@dataclass
class XVAResult:
    """XVA calculation result for a netting set."""

    netting_set_id: str
    counterparty_name: str
    has_csa: bool
    num_trades: int
    gross_notional: float
    net_mtm: float

    # Exposure metrics
    expected_exposure: float
    peak_exposure: float
    exposure_profile: np.ndarray
    time_grid: np.ndarray

    # XVA components
    cva: float
    dva: float
    fva: float
    mva: float
    total_xva: float

    # Parameters used
    lgd: float = 0.6  # Loss Given Default
    recovery_rate: float = 0.4
    default_probability: float = 0.02  # 2% probability
    funding_spread: float = 0.005  # 50 bps
    im_haircut: float = 0.10  # 10% initial margin


class StandaloneXVACalculator:
    """Standalone XVA calculator using Monte Carlo simulation."""

    def __init__(self, num_simulations: int = 1000, time_horizon_years: float = 5.0):
        """Initialize calculator.

        Args:
            num_simulations: Number of Monte Carlo paths
            time_horizon_years: Time horizon for exposure simulation (years)
        """
        self.num_simulations = num_simulations
        self.time_horizon_years = time_horizon_years
        self.num_time_steps = int(time_horizon_years * 12)  # Monthly steps
        self.dt = time_horizon_years / self.num_time_steps

        # Generate time grid
        self.time_grid = np.linspace(0, time_horizon_years, self.num_time_steps + 1)

    def simulate_exposure_profile(
        self,
        net_mtm: float,
        volatility: float = 0.30,
        drift: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate exposure profile using geometric Brownian motion.

        Args:
            net_mtm: Current net MTM
            volatility: Volatility of MTM (annualized)
            drift: Drift rate

        Returns:
            Tuple of (exposure_paths, expected_exposure)
        """
        # Initialize paths
        paths = np.zeros((self.num_simulations, self.num_time_steps + 1))
        paths[:, 0] = net_mtm

        # Generate random shocks
        dW = np.random.normal(0, np.sqrt(self.dt), (self.num_simulations, self.num_time_steps))

        # Simulate paths using geometric Brownian motion
        for t in range(self.num_time_steps):
            paths[:, t + 1] = paths[:, t] * np.exp(
                (drift - 0.5 * volatility**2) * self.dt + volatility * dW[:, t]
            )

        # Calculate expected positive exposure (EPE)
        expected_exposure = np.maximum(paths, 0).mean(axis=0)

        return paths, expected_exposure

    def calculate_cva(
        self,
        exposure_profile: np.ndarray,
        default_probability: float,
        lgd: float = 0.6,
        discount_rate: float = 0.03
    ) -> float:
        """Calculate Credit Valuation Adjustment (CVA).

        CVA = LGD * Sum(PD * Discount * Expected Exposure)

        Args:
            exposure_profile: Expected positive exposure over time
            default_probability: Annual probability of default
            lgd: Loss Given Default (1 - Recovery Rate)
            discount_rate: Risk-free discount rate

        Returns:
            CVA value
        """
        # Calculate survival probability at each time step
        survival_prob = np.exp(-default_probability * self.time_grid)

        # Calculate marginal default probability
        marginal_pd = -np.diff(survival_prob)
        marginal_pd = np.append(marginal_pd, marginal_pd[-1])  # Extend for last period

        # Discount factors
        discount_factors = np.exp(-discount_rate * self.time_grid)

        # CVA = LGD * Sum(Marginal PD * Discount * EE)
        cva = lgd * np.sum(marginal_pd * discount_factors * exposure_profile) * self.dt

        return cva

    def calculate_dva(
        self,
        exposure_profile: np.ndarray,
        own_default_probability: float,
        lgd: float = 0.6,
        discount_rate: float = 0.03
    ) -> float:
        """Calculate Debit Valuation Adjustment (DVA).

        DVA = LGD * Sum(Own PD * Discount * Expected Negative Exposure)

        Args:
            exposure_profile: Expected exposure over time (can be negative)
            own_default_probability: Our own probability of default
            lgd: Loss Given Default
            discount_rate: Risk-free discount rate

        Returns:
            DVA value
        """
        # For DVA, we consider negative exposure (what we owe)
        negative_exposure = np.abs(np.minimum(exposure_profile, 0))

        # Calculate survival probability
        survival_prob = np.exp(-own_default_probability * self.time_grid)

        # Calculate marginal default probability
        marginal_pd = -np.diff(survival_prob)
        marginal_pd = np.append(marginal_pd, marginal_pd[-1])

        # Discount factors
        discount_factors = np.exp(-discount_rate * self.time_grid)

        # DVA = LGD * Sum(Marginal PD * Discount * ENE)
        dva = lgd * np.sum(marginal_pd * discount_factors * negative_exposure) * self.dt

        return dva

    def calculate_fva(
        self,
        exposure_profile: np.ndarray,
        funding_spread: float = 0.005,
        discount_rate: float = 0.03
    ) -> float:
        """Calculate Funding Valuation Adjustment (FVA).

        FVA = Funding Spread * Sum(Discount * Expected Exposure)

        Args:
            exposure_profile: Expected exposure over time
            funding_spread: Funding spread (e.g., 50 bps = 0.005)
            discount_rate: Risk-free discount rate

        Returns:
            FVA value
        """
        # Discount factors
        discount_factors = np.exp(-discount_rate * self.time_grid)

        # FVA for positive exposure (we need to fund)
        positive_exposure = np.maximum(exposure_profile, 0)

        # FVA = Funding Spread * Sum(Discount * Exposure)
        fva = funding_spread * np.sum(discount_factors * positive_exposure) * self.dt

        return fva

    def calculate_mva(
        self,
        exposure_profile: np.ndarray,
        im_haircut: float = 0.10,
        funding_spread: float = 0.005,
        discount_rate: float = 0.03
    ) -> float:
        """Calculate Margin Valuation Adjustment (MVA).

        MVA = Funding Cost of Initial Margin

        Args:
            exposure_profile: Expected exposure over time
            im_haircut: Initial margin haircut (percentage of exposure)
            funding_spread: Funding spread for margin
            discount_rate: Risk-free discount rate

        Returns:
            MVA value
        """
        # Estimate initial margin as percentage of exposure
        initial_margin = np.abs(exposure_profile) * im_haircut

        # Discount factors
        discount_factors = np.exp(-discount_rate * self.time_grid)

        # MVA = Funding Spread * Sum(Discount * IM)
        mva = funding_spread * np.sum(discount_factors * initial_margin) * self.dt

        return mva

    def calculate_netting_set_xva(
        self,
        netting_set_id: str,
        counterparty_name: str,
        net_mtm: float,
        gross_notional: float,
        num_trades: int,
        has_csa: bool = False,
        counterparty_rating: str = "BBB",
    ) -> XVAResult:
        """Calculate XVA for a single netting set.

        Args:
            netting_set_id: Netting set identifier
            counterparty_name: Name of counterparty
            net_mtm: Net mark-to-market value
            gross_notional: Gross notional amount
            num_trades: Number of trades in netting set
            has_csa: Whether CSA is in place
            counterparty_rating: Credit rating

        Returns:
            XVAResult object
        """
        print(f"  Computing XVA for {counterparty_name}...")

        # Adjust volatility based on notional
        volatility = 0.30 + (gross_notional / 100_000_000) * 0.10  # Higher vol for larger positions

        # Simulate exposure profile
        exposure_paths, expected_exposure = self.simulate_exposure_profile(
            net_mtm=net_mtm,
            volatility=volatility
        )

        # Adjust for CSA (reduces exposure significantly)
        if has_csa:
            expected_exposure = expected_exposure * 0.3  # CSA reduces exposure by ~70%
            print(f"    CSA in place - reduced exposure by 70%")

        # Determine default probability based on rating
        rating_pd_map = {
            "AAA": 0.0002, "AA+": 0.0005, "AA": 0.001, "AA-": 0.002,
            "A+": 0.005, "A": 0.01, "A-": 0.015,
            "BBB+": 0.02, "BBB": 0.03, "BBB-": 0.05,
        }
        default_prob = rating_pd_map.get(counterparty_rating, 0.02)

        # Calculate XVA components
        lgd = 0.6  # 60% Loss Given Default
        funding_spread = 0.005  # 50 bps

        cva = self.calculate_cva(expected_exposure, default_prob, lgd)
        dva = self.calculate_dva(expected_exposure, own_default_probability=0.001, lgd=lgd)
        fva = self.calculate_fva(expected_exposure, funding_spread)
        mva = self.calculate_mva(expected_exposure, im_haircut=0.10, funding_spread=funding_spread)

        # Total XVA
        total_xva = cva - dva + fva + mva  # Note: DVA is a benefit, so we subtract

        print(f"    CVA: ${cva:,.2f} | DVA: ${dva:,.2f} | FVA: ${fva:,.2f} | MVA: ${mva:,.2f}")
        print(f"    Total XVA: ${total_xva:,.2f}")

        return XVAResult(
            netting_set_id=netting_set_id,
            counterparty_name=counterparty_name,
            has_csa=has_csa,
            num_trades=num_trades,
            gross_notional=gross_notional,
            net_mtm=net_mtm,
            expected_exposure=expected_exposure.mean(),
            peak_exposure=expected_exposure.max(),
            exposure_profile=expected_exposure,
            time_grid=self.time_grid,
            cva=cva,
            dva=dva,
            fva=fva,
            mva=mva,
            total_xva=total_xva,
            lgd=lgd,
            default_probability=default_prob,
            funding_spread=funding_spread,
        )

    def calculate_portfolio_xva(
        self,
        portfolio: Any,
        summary: Dict
    ) -> List[XVAResult]:
        """Calculate XVA for all netting sets in portfolio.

        Args:
            portfolio: Portfolio object
            summary: Portfolio summary dict

        Returns:
            List of XVAResult objects
        """
        print("Calculating XVA for all netting sets...")
        print()

        results = []

        for ns_id, ns in portfolio.netting_sets.items():
            # Get counterparty info
            cp_info = summary["counterparties"].get(ns.counterparty_id, {})
            counterparty_name = cp_info.get("name", ns.counterparty_id)
            has_csa = cp_info.get("has_csa", False)
            rating = cp_info.get("rating", "BBB")

            # Calculate net MTM and gross notional for this netting set
            ns_trades = [t for t in portfolio.trades.values() if t.netting_set_id == ns_id]
            net_mtm = sum(getattr(t, "mtm", 0) for t in ns_trades)
            gross_notional = sum(t.notional for t in ns_trades)

            result = self.calculate_netting_set_xva(
                netting_set_id=ns_id,
                counterparty_name=counterparty_name,
                net_mtm=net_mtm,
                gross_notional=gross_notional,
                num_trades=len(ns_trades),
                has_csa=has_csa,
                counterparty_rating=rating,
            )

            results.append(result)
            print()

        return results


class XVAVisualizer:
    """Create visualizations for XVA analysis."""

    def __init__(self, output_dir: Path):
        """Initialize visualizer.

        Args:
            output_dir: Directory to save charts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_xva_waterfall(self, results: List[XVAResult]) -> Path:
        """Create XVA waterfall chart showing portfolio-level components.

        Args:
            results: List of XVA results

        Returns:
            Path to saved chart
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        # Aggregate portfolio-level XVA
        total_cva = sum(r.cva for r in results)
        total_dva = sum(r.dva for r in results)
        total_fva = sum(r.fva for r in results)
        total_mva = sum(r.mva for r in results)
        total_xva = sum(r.total_xva for r in results)

        # Waterfall data
        categories = ["CVA", "DVA\n(Benefit)", "FVA", "MVA", "Total\nXVA"]
        values = [total_cva, -total_dva, total_fva, total_mva, total_xva]
        colors = ["#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#3498db"]

        # Create bars
        bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)

        # Add value labels
        for bar, val in zip(bars, values):
            label = f"${abs(val):,.0f}"
            y_pos = val + (max(values) - min(values)) * 0.02
            if val < 0:
                y_pos = val - (max(values) - min(values)) * 0.02
                va = "top"
            else:
                va = "bottom"
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos, label,
                   ha="center", va=va, fontsize=12, fontweight="bold")

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
        ax.set_ylabel("Amount (USD)", fontweight="bold", fontsize=12)
        ax.set_title("Portfolio XVA Waterfall", fontweight="bold", fontsize=16, pad=20)
        ax.grid(True, alpha=0.3, axis="y")

        # Add legend explaining DVA
        legend_text = "Note: DVA is shown as negative (benefit to us)"
        ax.text(0.98, 0.02, legend_text, transform=ax.transAxes,
               fontsize=10, verticalalignment="bottom", horizontalalignment="right",
               bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        plt.tight_layout()
        output_file = self.output_dir / "xva_waterfall_standalone.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        return output_file

    def plot_exposure_profiles(self, results: List[XVAResult]) -> Path:
        """Plot exposure profiles for all netting sets.

        Args:
            results: List of XVA results

        Returns:
            Path to saved chart
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot each netting set's exposure profile
        for result in results:
            label = f"{result.counterparty_name[:25]}"
            if result.has_csa:
                label += " (CSA)"
            ax.plot(result.time_grid, result.exposure_profile,
                   label=label, linewidth=2, alpha=0.7)

        ax.set_xlabel("Time (Years)", fontweight="bold", fontsize=12)
        ax.set_ylabel("Expected Positive Exposure (USD)", fontweight="bold", fontsize=12)
        ax.set_title("Expected Positive Exposure Profiles by Counterparty",
                    fontweight="bold", fontsize=14, pad=20)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)

        plt.tight_layout()
        output_file = self.output_dir / "exposure_profiles.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        return output_file

    def plot_xva_by_counterparty(self, results: List[XVAResult]) -> Path:
        """Create stacked bar chart of XVA by counterparty.

        Args:
            results: List of XVA results

        Returns:
            Path to saved chart
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        # Prepare data
        counterparties = [r.counterparty_name[:25] for r in results]
        cva = [r.cva for r in results]
        dva = [-r.dva for r in results]  # Negative for stacking
        fva = [r.fva for r in results]
        mva = [r.mva for r in results]

        x = np.arange(len(counterparties))
        width = 0.6

        # Create stacked bars
        p1 = ax.bar(x, cva, width, label="CVA", color="#e74c3c", alpha=0.8)
        p2 = ax.bar(x, dva, width, bottom=cva, label="DVA (Benefit)", color="#2ecc71", alpha=0.8)
        p3 = ax.bar(x, fva, width, bottom=np.array(cva) + np.array(dva), label="FVA", color="#f39c12", alpha=0.8)
        p4 = ax.bar(x, mva, width, bottom=np.array(cva) + np.array(dva) + np.array(fva),
                   label="MVA", color="#9b59b6", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(counterparties, rotation=45, ha="right")
        ax.set_ylabel("XVA Amount (USD)", fontweight="bold", fontsize=12)
        ax.set_title("XVA Components by Counterparty", fontweight="bold", fontsize=14, pad=20)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)

        plt.tight_layout()
        output_file = self.output_dir / "xva_by_counterparty_standalone.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        return output_file

    def plot_csa_impact(self, results: List[XVAResult]) -> Path:
        """Compare XVA for CSA vs non-CSA counterparties.

        Args:
            results: List of XVA results

        Returns:
            Path to saved chart
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Separate CSA and non-CSA
        csa_results = [r for r in results if r.has_csa]
        non_csa_results = [r for r in results if not r.has_csa]

        # Total XVA comparison
        if csa_results and non_csa_results:
            csa_total = sum(r.total_xva for r in csa_results)
            non_csa_total = sum(r.total_xva for r in non_csa_results)

            bars = ax1.bar(["With CSA", "Without CSA"], [csa_total, non_csa_total],
                          color=["#2ecc71", "#e74c3c"], alpha=0.8, edgecolor="black")

            ax1.set_ylabel("Total XVA (USD)", fontweight="bold", fontsize=12)
            ax1.set_title("Total XVA: CSA vs Non-CSA", fontweight="bold", fontsize=12, pad=20)
            ax1.grid(True, alpha=0.3, axis="y")

            # Add value labels
            for bar, val in zip(bars, [csa_total, non_csa_total]):
                ax1.text(bar.get_x() + bar.get_width() / 2, val, f"${val:,.0f}",
                        ha="center", va="bottom", fontsize=11, fontweight="bold")

            # Average exposure comparison
            csa_avg_exposure = np.mean([r.expected_exposure for r in csa_results])
            non_csa_avg_exposure = np.mean([r.expected_exposure for r in non_csa_results])

            bars2 = ax2.bar(["With CSA", "Without CSA"],
                           [csa_avg_exposure, non_csa_avg_exposure],
                           color=["#2ecc71", "#e74c3c"], alpha=0.8, edgecolor="black")

            ax2.set_ylabel("Average Expected Exposure (USD)", fontweight="bold", fontsize=12)
            ax2.set_title("Average Exposure: CSA vs Non-CSA", fontweight="bold", fontsize=12, pad=20)
            ax2.grid(True, alpha=0.3, axis="y")

            # Add value labels
            for bar, val in zip(bars2, [csa_avg_exposure, non_csa_avg_exposure]):
                ax2.text(bar.get_x() + bar.get_width() / 2, val, f"${val:,.0f}",
                        ha="center", va="bottom", fontsize=11, fontweight="bold")

        plt.tight_layout()
        output_file = self.output_dir / "csa_impact_standalone.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        return output_file


def generate_xva_reports(results: List[XVAResult], output_dir: Path) -> Dict[str, Path]:
    """Generate XVA analysis reports.

    Args:
        results: List of XVA results
        output_dir: Output directory

    Returns:
        Dict mapping report type to file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports = {}

    # JSON report
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "portfolio_summary": {
            "num_netting_sets": len(results),
            "total_cva": sum(r.cva for r in results),
            "total_dva": sum(r.dva for r in results),
            "total_fva": sum(r.fva for r in results),
            "total_mva": sum(r.mva for r in results),
            "total_xva": sum(r.total_xva for r in results),
        },
        "netting_sets": [
            {
                "netting_set_id": r.netting_set_id,
                "counterparty": r.counterparty_name,
                "has_csa": r.has_csa,
                "num_trades": r.num_trades,
                "gross_notional": r.gross_notional,
                "net_mtm": r.net_mtm,
                "expected_exposure": r.expected_exposure,
                "peak_exposure": r.peak_exposure,
                "cva": r.cva,
                "dva": r.dva,
                "fva": r.fva,
                "mva": r.mva,
                "total_xva": r.total_xva,
            }
            for r in results
        ]
    }

    json_file = output_dir / f"xva_analysis_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=2)
    reports["json"] = json_file
    print(f"✓ JSON report: {json_file.name}")

    # Excel report
    excel_file = output_dir / f"xva_analysis_{timestamp}.xlsx"
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        # Summary sheet
        summary_data = {
            "Metric": ["Total CVA", "Total DVA", "Total FVA", "Total MVA", "Total XVA", "Number of Netting Sets"],
            "Value": [
                sum(r.cva for r in results),
                sum(r.dva for r in results),
                sum(r.fva for r in results),
                sum(r.mva for r in results),
                sum(r.total_xva for r in results),
                len(results)
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)

        # Detailed netting set data
        ns_data = []
        for r in results:
            ns_data.append({
                "Netting Set ID": r.netting_set_id,
                "Counterparty": r.counterparty_name,
                "Has CSA": r.has_csa,
                "Num Trades": r.num_trades,
                "Gross Notional": r.gross_notional,
                "Net MTM": r.net_mtm,
                "Expected Exposure": r.expected_exposure,
                "Peak Exposure": r.peak_exposure,
                "CVA": r.cva,
                "DVA": r.dva,
                "FVA": r.fva,
                "MVA": r.mva,
                "Total XVA": r.total_xva,
            })
        pd.DataFrame(ns_data).to_excel(writer, sheet_name="Netting Sets", index=False)

    reports["excel"] = excel_file
    print(f"✓ Excel report: {excel_file.name}")

    return reports


def main():
    """Main entry point."""
    print("=" * 80)
    print("Standalone XVA Calculator - No API Required")
    print("=" * 80)
    print()

    # Load portfolio
    print("Loading portfolio...")
    portfolio, book_hierarchy = create_fictional_portfolio()
    summary = get_portfolio_summary(portfolio, book_hierarchy)
    print(f"✓ Portfolio loaded: {portfolio.name}")
    print(f"  Counterparties: {len(summary['counterparties'])}")
    print(f"  Netting Sets: {len(portfolio.netting_sets)}")
    print(f"  Total Trades: {len(portfolio.trades)}")
    print()

    # Initialize calculator
    print("Initializing XVA calculator...")
    calculator = StandaloneXVACalculator(num_simulations=1000, time_horizon_years=5.0)
    print(f"✓ Monte Carlo simulations: {calculator.num_simulations}")
    print(f"✓ Time horizon: {calculator.time_horizon_years} years")
    print()

    # Calculate XVA
    print("=" * 80)
    print("XVA CALCULATION")
    print("=" * 80)
    print()

    xva_results = calculator.calculate_portfolio_xva(portfolio, summary)

    # Print summary
    print("=" * 80)
    print("PORTFOLIO XVA SUMMARY")
    print("=" * 80)
    print()

    total_cva = sum(r.cva for r in xva_results)
    total_dva = sum(r.dva for r in xva_results)
    total_fva = sum(r.fva for r in xva_results)
    total_mva = sum(r.mva for r in xva_results)
    total_xva = sum(r.total_xva for r in xva_results)

    print(f"Total CVA:  ${total_cva:>15,.2f}")
    print(f"Total DVA:  ${total_dva:>15,.2f}  (Benefit)")
    print(f"Total FVA:  ${total_fva:>15,.2f}")
    print(f"Total MVA:  ${total_mva:>15,.2f}")
    print(f"{'─' * 40}")
    print(f"Total XVA:  ${total_xva:>15,.2f}")
    print()

    # Create visualizations
    print("=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    print()

    output_dir = Path(__file__).parent / "sample_outputs" / "charts"
    visualizer = XVAVisualizer(output_dir)

    print("Creating XVA waterfall chart...")
    visualizer.plot_xva_waterfall(xva_results)
    print("✓ XVA waterfall chart")

    print("Creating exposure profiles...")
    visualizer.plot_exposure_profiles(xva_results)
    print("✓ Exposure profiles")

    print("Creating XVA by counterparty chart...")
    visualizer.plot_xva_by_counterparty(xva_results)
    print("✓ XVA by counterparty")

    print("Creating CSA impact analysis...")
    visualizer.plot_csa_impact(xva_results)
    print("✓ CSA impact analysis")
    print()

    # Generate reports
    print("=" * 80)
    print("GENERATING REPORTS")
    print("=" * 80)
    print()

    reports_dir = Path(__file__).parent / "reports"
    reports = generate_xva_reports(xva_results, reports_dir)
    print()

    print("=" * 80)
    print("XVA ANALYSIS COMPLETE!")
    print("=" * 80)
    print()
    print(f"Charts saved to: {output_dir}")
    print(f"Reports saved to: {reports_dir}")
    print()
    print("Note: This demonstration uses simplified XVA calculations.")
    print("In production, XVA would be computed using proper simulation models")
    print("with credit curves, funding curves, and regulatory requirements.")
    print()


if __name__ == "__main__":
    main()
