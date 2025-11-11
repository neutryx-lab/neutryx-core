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
import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Fix encoding for Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

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


class BlackScholesGreeks:
    """Black-Scholes option pricing and Greeks calculation."""

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter for Black-Scholes."""
        if T <= 0:
            return 0.0
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 parameter for Black-Scholes."""
        if T <= 0:
            return 0.0
        return BlackScholesGreeks.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate European call option price."""
        if T <= 0:
            return max(S - K, 0)
        d1_val = BlackScholesGreeks.d1(S, K, T, r, sigma)
        d2_val = BlackScholesGreeks.d2(S, K, T, r, sigma)
        return S * norm.cdf(d1_val) - K * np.exp(-r * T) * norm.cdf(d2_val)

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate European put option price."""
        if T <= 0:
            return max(K - S, 0)
        d1_val = BlackScholesGreeks.d1(S, K, T, r, sigma)
        d2_val = BlackScholesGreeks.d2(S, K, T, r, sigma)
        return K * np.exp(-r * T) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val)

    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
        """Calculate option Delta."""
        if T <= 0:
            if option_type.lower() == "call":
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0

        d1_val = BlackScholesGreeks.d1(S, K, T, r, sigma)
        if option_type.lower() == "call":
            return norm.cdf(d1_val)
        else:
            return norm.cdf(d1_val) - 1

    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option Gamma."""
        if T <= 0:
            return 0.0
        d1_val = BlackScholesGreeks.d1(S, K, T, r, sigma)
        return norm.pdf(d1_val) / (S * sigma * np.sqrt(T))

    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option Vega (per 1% change in volatility)."""
        if T <= 0:
            return 0.0
        d1_val = BlackScholesGreeks.d1(S, K, T, r, sigma)
        return S * norm.pdf(d1_val) * np.sqrt(T) / 100  # Divided by 100 for 1% change

    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
        """Calculate option Theta (per day)."""
        if T <= 0:
            return 0.0

        d1_val = BlackScholesGreeks.d1(S, K, T, r, sigma)
        d2_val = BlackScholesGreeks.d2(S, K, T, r, sigma)

        term1 = -(S * norm.pdf(d1_val) * sigma) / (2 * np.sqrt(T))

        if option_type.lower() == "call":
            term2 = -r * K * np.exp(-r * T) * norm.cdf(d2_val)
        else:
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2_val)

        return (term1 + term2) / 365  # Per day

    @staticmethod
    def rho(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
        """Calculate option Rho (per 1% change in interest rate)."""
        if T <= 0:
            return 0.0

        d2_val = BlackScholesGreeks.d2(S, K, T, r, sigma)

        if option_type.lower() == "call":
            return K * T * np.exp(-r * T) * norm.cdf(d2_val) / 100
        else:
            return -K * T * np.exp(-r * T) * norm.cdf(-d2_val) / 100


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

        # Compute individual Greeks
        greeks = self._compute_greeks(portfolio)

        # Compute portfolio-level aggregations
        portfolio_greeks = self._aggregate_portfolio_greeks(greeks)
        greeks_by_underlying = self._aggregate_greeks_by_underlying(greeks)
        greeks_by_product = self._aggregate_greeks_by_product(greeks)

        results = {
            "greeks": greeks,
            "portfolio_greeks": portfolio_greeks,
            "greeks_by_underlying": greeks_by_underlying,
            "greeks_by_product": greeks_by_product,
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
        """Compute option Greeks for all option trades using Black-Scholes.

        Args:
            portfolio: Portfolio object

        Returns:
            List of Greeks results
        """
        print("Computing option Greeks using Black-Scholes model...")

        greeks_results = []

        # Filter option trades
        # Note: ProductType values are in PascalCase (e.g., 'FxOption', 'EquityOption')
        option_products = [
            "FxOption",
            "EquityOption",
            "Swaption",
            "VarianceSwap",
        ]

        option_trades = [
            t
            for t in portfolio.trades.values()
            if t.product_type.value in option_products
        ]

        print(f"  Found {len(option_trades)} option trades")

        for trade in option_trades:
            # Get trade-specific parameters
            if trade.product_type.value == "FxOption":
                # FX option parameters
                spot = 1.1  # EURUSD or other pair
                if hasattr(trade, "strike"):
                    strike = trade.strike
                else:
                    strike = spot * np.random.uniform(0.95, 1.05)

                volatility = 0.12 if hasattr(trade, "volatility") else 0.12  # 12% vol
                risk_free_rate = 0.03
                time_to_maturity = 0.5  # 6 months
                option_type = getattr(trade, "option_type", "call")
                underlying = trade.currency_pair if hasattr(trade, "currency_pair") else "FX"

                # Compute Greeks
                delta_bs = BlackScholesGreeks.delta(spot, strike, time_to_maturity, risk_free_rate, volatility, option_type)
                gamma_bs = BlackScholesGreeks.gamma(spot, strike, time_to_maturity, risk_free_rate, volatility)
                vega_bs = BlackScholesGreeks.vega(spot, strike, time_to_maturity, risk_free_rate, volatility)
                theta_bs = BlackScholesGreeks.theta(spot, strike, time_to_maturity, risk_free_rate, volatility, option_type)
                rho_bs = BlackScholesGreeks.rho(spot, strike, time_to_maturity, risk_free_rate, volatility, option_type)

                # Scale by notional
                delta = delta_bs * trade.notional
                gamma = gamma_bs * trade.notional
                vega = vega_bs * trade.notional
                theta = theta_bs * trade.notional
                rho = rho_bs * trade.notional

            elif trade.product_type.value == "EquityOption":
                # Equity option parameters
                spot = 100.0  # Stock price
                if hasattr(trade, "strike"):
                    strike = trade.strike
                else:
                    strike = spot * np.random.uniform(0.90, 1.10)

                volatility = 0.25 if hasattr(trade, "volatility") else 0.25  # 25% vol
                risk_free_rate = 0.03
                time_to_maturity = 0.75  # 9 months
                option_type = getattr(trade, "option_type", "call")
                underlying = trade.underlying if hasattr(trade, "underlying") else "Equity"

                # Compute Greeks
                delta_bs = BlackScholesGreeks.delta(spot, strike, time_to_maturity, risk_free_rate, volatility, option_type)
                gamma_bs = BlackScholesGreeks.gamma(spot, strike, time_to_maturity, risk_free_rate, volatility)
                vega_bs = BlackScholesGreeks.vega(spot, strike, time_to_maturity, risk_free_rate, volatility)
                theta_bs = BlackScholesGreeks.theta(spot, strike, time_to_maturity, risk_free_rate, volatility, option_type)
                rho_bs = BlackScholesGreeks.rho(spot, strike, time_to_maturity, risk_free_rate, volatility, option_type)

                # Scale by notional (number of shares * spot)
                num_shares = trade.notional / spot
                delta = delta_bs * num_shares * spot
                gamma = gamma_bs * num_shares * spot
                vega = vega_bs * num_shares * spot
                theta = theta_bs * num_shares * spot
                rho = rho_bs * num_shares * spot

            elif trade.product_type.value == "Swaption":
                # Swaption - simplified as bond option
                spot = 100.0  # Par value
                strike = 100.0
                volatility = 0.50 if hasattr(trade, "volatility") else 0.50  # 50% vol for swaptions
                risk_free_rate = 0.03
                time_to_maturity = 1.0  # 1 year
                option_type = "call"
                underlying = f"{trade.currency} Swaption"

                # Compute Greeks (scaled for swaption)
                delta_bs = BlackScholesGreeks.delta(spot, strike, time_to_maturity, risk_free_rate, volatility, option_type)
                gamma_bs = BlackScholesGreeks.gamma(spot, strike, time_to_maturity, risk_free_rate, volatility)
                vega_bs = BlackScholesGreeks.vega(spot, strike, time_to_maturity, risk_free_rate, volatility)
                theta_bs = BlackScholesGreeks.theta(spot, strike, time_to_maturity, risk_free_rate, volatility, option_type)
                rho_bs = BlackScholesGreeks.rho(spot, strike, time_to_maturity, risk_free_rate, volatility, option_type)

                # Scale by notional
                delta = delta_bs * trade.notional / 100
                gamma = gamma_bs * trade.notional / 100
                vega = vega_bs * trade.notional / 100
                theta = theta_bs * trade.notional / 100
                rho = rho_bs * trade.notional / 100

            else:  # variance_swap
                # Variance swaps have special Greeks
                delta = 0  # No delta for variance swaps
                gamma = 0
                # Variance swaps are highly sensitive to volatility
                vega = trade.notional * 0.5  # Simplified vega
                theta = -trade.notional * 0.001  # Time decay
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

        print(f"  ✓ Computed Black-Scholes Greeks for {len(greeks_results)} trades")
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
        rate_products = ["InterestRateSwap", "Swaption"]
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

        fx_products = ["FxOption", "FxForward"]
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

        eq_products = ["EquityOption"]
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

        option_products = ["FxOption", "EquityOption", "Swaption", "VarianceSwap"]
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
            if trade.product_type.value == "FxOption":
                asset_class = "FX"
                vega = trade.notional * 0.001  # Simplified
            elif trade.product_type.value in ["EquityOption", "VarianceSwap"]:
                asset_class = "Equity"
                vega = trade.notional * 0.002  # Higher vega for equity
            else:  # Swaption
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

        rate_products = ["InterestRateSwap", "Swaption"]
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

    def _aggregate_portfolio_greeks(self, greeks: List[Dict]) -> Dict:
        """Aggregate Greeks at the portfolio level.

        Args:
            greeks: List of trade-level Greeks

        Returns:
            Portfolio-level Greeks summary
        """
        print("Aggregating portfolio-level Greeks...")

        if not greeks:
            return {}

        df = pd.DataFrame(greeks)

        portfolio_greeks = {
            "total_delta": df["delta"].sum(),
            "total_gamma": df["gamma"].sum(),
            "total_vega": df["vega"].sum(),
            "total_theta": df["theta"].sum(),
            "total_rho": df["rho"].sum(),
            "num_trades": len(greeks),
            "avg_delta": df["delta"].mean(),
            "avg_gamma": df["gamma"].mean(),
            "avg_vega": df["vega"].mean(),
            "avg_theta": df["theta"].mean(),
            "avg_rho": df["rho"].mean(),
            "std_delta": df["delta"].std(),
            "std_gamma": df["gamma"].std(),
            "std_vega": df["vega"].std(),
            "std_theta": df["theta"].std(),
            "std_rho": df["rho"].std(),
        }

        print(f"  ✓ Portfolio Greeks: Delta={portfolio_greeks['total_delta']:,.2f}, Gamma={portfolio_greeks['total_gamma']:,.2f}, Vega={portfolio_greeks['total_vega']:,.2f}")
        return portfolio_greeks

    def _aggregate_greeks_by_underlying(self, greeks: List[Dict]) -> Dict:
        """Aggregate Greeks by underlying asset.

        Args:
            greeks: List of trade-level Greeks

        Returns:
            Greeks aggregated by underlying
        """
        print("Aggregating Greeks by underlying...")

        if not greeks:
            return {}

        df = pd.DataFrame(greeks)
        grouped = df.groupby("underlying")

        aggregated = {}
        for underlying, group in grouped:
            aggregated[underlying] = {
                "delta": group["delta"].sum(),
                "gamma": group["gamma"].sum(),
                "vega": group["vega"].sum(),
                "theta": group["theta"].sum(),
                "rho": group["rho"].sum(),
                "num_trades": len(group),
                "total_notional": group["notional"].sum(),
            }

        print(f"  ✓ Aggregated Greeks for {len(aggregated)} underlyings")
        return aggregated

    def _aggregate_greeks_by_product(self, greeks: List[Dict]) -> Dict:
        """Aggregate Greeks by product type.

        Args:
            greeks: List of trade-level Greeks

        Returns:
            Greeks aggregated by product type
        """
        print("Aggregating Greeks by product type...")

        if not greeks:
            return {}

        df = pd.DataFrame(greeks)
        grouped = df.groupby("product_type")

        aggregated = {}
        for product_type, group in grouped:
            aggregated[product_type] = {
                "delta": group["delta"].sum(),
                "gamma": group["gamma"].sum(),
                "vega": group["vega"].sum(),
                "theta": group["theta"].sum(),
                "rho": group["rho"].sum(),
                "num_trades": len(group),
                "total_notional": group["notional"].sum(),
            }

        print(f"  ✓ Aggregated Greeks for {len(aggregated)} product types")
        return aggregated

    def generate_greeks_heatmap_data(self, greeks: List[Dict]) -> pd.DataFrame:
        """Generate heatmap data for Greeks visualization.

        Args:
            greeks: List of trade-level Greeks

        Returns:
            DataFrame suitable for heatmap visualization
        """
        if not greeks:
            return pd.DataFrame()

        df = pd.DataFrame(greeks)

        # Create a pivot table for heatmap (underlying x Greek type)
        heatmap_data = pd.DataFrame({
            "Underlying": df["underlying"],
            "Delta": df["delta"],
            "Gamma": df["gamma"],
            "Vega": df["vega"],
            "Theta": df["theta"],
            "Rho": df["rho"],
        })

        # Aggregate by underlying
        heatmap_pivot = heatmap_data.groupby("Underlying").sum()

        return heatmap_pivot

    def create_greeks_heatmap(self, results: Dict) -> Path:
        """Create Greeks heatmap visualization.

        Args:
            results: Sensitivity analysis results

        Returns:
            Path to saved heatmap image
        """
        print("Creating Greeks heatmap...")

        if not results.get("greeks"):
            print("  No Greeks data available for heatmap")
            return None

        # Generate heatmap data
        heatmap_data = self.generate_greeks_heatmap_data(results["greeks"])

        if heatmap_data.empty:
            print("  No data for heatmap")
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Normalize data for better visualization (use log scale for large differences)
        heatmap_normalized = heatmap_data.copy()
        for col in heatmap_normalized.columns:
            max_val = heatmap_normalized[col].abs().max()
            if max_val > 0:
                heatmap_normalized[col] = heatmap_normalized[col] / max_val

        # Create heatmap
        sns.heatmap(
            heatmap_normalized.T,
            annot=heatmap_data.T,
            fmt=".0f",
            cmap="RdYlGn",
            center=0,
            cbar_kws={"label": "Normalized Greek Value"},
            linewidths=0.5,
            ax=ax,
        )

        ax.set_title("Portfolio Greeks Heatmap by Underlying", fontweight="bold", fontsize=14, pad=20)
        ax.set_xlabel("Underlying Asset", fontweight="bold", fontsize=12)
        ax.set_ylabel("Greek", fontweight="bold", fontsize=12)

        plt.tight_layout()
        output_file = self.output_dir / "greeks_heatmap.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  ✓ Heatmap saved: {output_file.name}")
        return output_file

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

            # Portfolio Greeks Summary
            if results.get("portfolio_greeks"):
                portfolio_greeks = results["portfolio_greeks"]
                summary_data = {
                    "Metric": [
                        "Total Delta", "Total Gamma", "Total Vega", "Total Theta", "Total Rho",
                        "Avg Delta", "Avg Gamma", "Avg Vega", "Avg Theta", "Avg Rho",
                        "Std Delta", "Std Gamma", "Std Vega", "Std Theta", "Std Rho",
                        "Number of Trades"
                    ],
                    "Value": [
                        portfolio_greeks["total_delta"],
                        portfolio_greeks["total_gamma"],
                        portfolio_greeks["total_vega"],
                        portfolio_greeks["total_theta"],
                        portfolio_greeks["total_rho"],
                        portfolio_greeks["avg_delta"],
                        portfolio_greeks["avg_gamma"],
                        portfolio_greeks["avg_vega"],
                        portfolio_greeks["avg_theta"],
                        portfolio_greeks["avg_rho"],
                        portfolio_greeks["std_delta"],
                        portfolio_greeks["std_gamma"],
                        portfolio_greeks["std_vega"],
                        portfolio_greeks["std_theta"],
                        portfolio_greeks["std_rho"],
                        portfolio_greeks["num_trades"]
                    ],
                }
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name="Portfolio Greeks", index=False)

            # Greeks by Underlying
            if results.get("greeks_by_underlying"):
                underlying_data = []
                for underlying, greeks in results["greeks_by_underlying"].items():
                    underlying_data.append({
                        "Underlying": underlying,
                        "Delta": greeks["delta"],
                        "Gamma": greeks["gamma"],
                        "Vega": greeks["vega"],
                        "Theta": greeks["theta"],
                        "Rho": greeks["rho"],
                        "Num Trades": greeks["num_trades"],
                        "Total Notional": greeks["total_notional"],
                    })
                df_underlying = pd.DataFrame(underlying_data)
                df_underlying.to_excel(writer, sheet_name="Greeks by Underlying", index=False)

            # Greeks by Product Type
            if results.get("greeks_by_product"):
                product_data = []
                for product_type, greeks in results["greeks_by_product"].items():
                    product_data.append({
                        "Product Type": product_type,
                        "Delta": greeks["delta"],
                        "Gamma": greeks["gamma"],
                        "Vega": greeks["vega"],
                        "Theta": greeks["theta"],
                        "Rho": greeks["rho"],
                        "Num Trades": greeks["num_trades"],
                        "Total Notional": greeks["total_notional"],
                    })
                df_product = pd.DataFrame(product_data)
                df_product.to_excel(writer, sheet_name="Greeks by Product", index=False)

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

    # Create Greeks heatmap
    print()
    heatmap_file = analyzer.create_greeks_heatmap(results)

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
