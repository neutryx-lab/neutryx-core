#!/usr/bin/env python3
"""Generate comprehensive portfolio reports with visualizations.

This script generates detailed HTML, CSV, and Excel reports for the
fictional bank portfolio including:
- Portfolio summary and statistics
- Counterparty breakdown
- Book hierarchy analysis
- XVA calculations (if available)
- Risk metrics
- Embedded charts and visualizations
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from neutryx.tests.fixtures.fictional_portfolio import (
    create_fictional_portfolio,
    get_portfolio_summary,
)


class PortfolioReporter:
    """Generate comprehensive portfolio reports in multiple formats."""

    def __init__(self, output_dir: Path):
        """Initialize the reporter.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_all_reports(
        self,
        portfolio: Any,
        book_hierarchy: Any,
        xva_results: Optional[Dict] = None,
    ) -> Dict[str, Path]:
        """Generate all report formats.

        Args:
            portfolio: Portfolio object
            book_hierarchy: Book hierarchy object
            xva_results: Optional XVA calculation results

        Returns:
            Dictionary mapping report format to file path
        """
        print("Generating comprehensive portfolio reports...")
        print()

        # Get portfolio summary
        summary = get_portfolio_summary(portfolio, book_hierarchy)

        # Generate reports
        reports = {}

        # JSON report
        json_file = self.generate_json_report(summary, xva_results)
        reports["json"] = json_file
        print(f"✓ JSON report: {json_file}")

        # CSV reports
        csv_files = self.generate_csv_reports(summary, xva_results)
        reports.update(csv_files)
        print(f"✓ CSV reports: {len(csv_files)} files")

        # Excel report
        excel_file = self.generate_excel_report(summary, xva_results)
        reports["excel"] = excel_file
        print(f"✓ Excel report: {excel_file}")

        # HTML report
        html_file = self.generate_html_report(summary, xva_results)
        reports["html"] = html_file
        print(f"✓ HTML report: {html_file}")

        print()
        print(f"All reports saved to: {self.output_dir}")
        return reports

    def generate_json_report(
        self, summary: Dict, xva_results: Optional[Dict] = None
    ) -> Path:
        """Generate JSON format report."""
        report_data = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_type": "portfolio_analysis",
                "version": "1.0",
            },
            "portfolio_summary": summary,
        }

        if xva_results:
            report_data["xva_analysis"] = xva_results

        output_file = self.output_dir / f"portfolio_report_{self.report_timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        return output_file

    def generate_csv_reports(
        self, summary: Dict, xva_results: Optional[Dict] = None
    ) -> Dict[str, Path]:
        """Generate CSV format reports (multiple files)."""
        csv_files = {}

        # Counterparty breakdown CSV
        cp_data = []
        for cp_id, cp_info in summary["counterparties"].items():
            cp_data.append(
                {
                    "Counterparty ID": cp_id,
                    "Name": cp_info["name"],
                    "Rating": cp_info["rating"],
                    "Entity Type": cp_info["entity_type"],
                    "Number of Trades": cp_info["num_trades"],
                    "Net MTM (USD)": cp_info["net_mtm"],
                    "Gross Notional (USD)": cp_info["gross_notional"],
                    "Has CSA": "Yes" if cp_info["has_csa"] else "No",
                }
            )

        df_cp = pd.DataFrame(cp_data)
        cp_file = self.output_dir / f"counterparties_{self.report_timestamp}.csv"
        df_cp.to_csv(cp_file, index=False)
        csv_files["counterparties"] = cp_file

        # Book breakdown CSV
        book_data = []
        for book_id, book_info in summary["books"].items():
            book_data.append(
                {
                    "Book ID": book_id,
                    "Book Name": book_info["name"],
                    "Desk": book_info.get("desk", "N/A"),
                    "Number of Trades": book_info["num_trades"],
                    "Active Trades": book_info["active_trades"],
                    "Total MTM (USD)": book_info["total_mtm"],
                }
            )

        df_books = pd.DataFrame(book_data)
        books_file = self.output_dir / f"books_{self.report_timestamp}.csv"
        df_books.to_csv(books_file, index=False)
        csv_files["books"] = books_file

        # Desk breakdown CSV
        desk_data = []
        for desk_id, desk_info in summary["desks"].items():
            desk_data.append(
                {
                    "Desk ID": desk_id,
                    "Desk Name": desk_info["name"],
                    "Number of Books": desk_info["num_books"],
                    "Number of Trades": desk_info["num_trades"],
                    "Total MTM (USD)": desk_info["total_mtm"],
                }
            )

        df_desks = pd.DataFrame(desk_data)
        desks_file = self.output_dir / f"desks_{self.report_timestamp}.csv"
        df_desks.to_csv(desks_file, index=False)
        csv_files["desks"] = desks_file

        # XVA results CSV (if available)
        if xva_results and "netting_set_xva" in xva_results:
            xva_data = []
            for ns_xva in xva_results["netting_set_xva"]:
                xva_data.append(
                    {
                        "Netting Set ID": ns_xva["netting_set_id"],
                        "Counterparty": ns_xva["counterparty"],
                        "Has CSA": "Yes" if ns_xva["has_csa"] else "No",
                        "CVA (USD)": ns_xva["cva"],
                        "DVA (USD)": ns_xva["dva"],
                        "FVA (USD)": ns_xva["fva"],
                        "MVA (USD)": ns_xva["mva"],
                        "Total XVA (USD)": ns_xva["total_xva"],
                        "Net MTM (USD)": ns_xva["net_mtm"],
                    }
                )

            df_xva = pd.DataFrame(xva_data)
            xva_file = self.output_dir / f"xva_results_{self.report_timestamp}.csv"
            df_xva.to_csv(xva_file, index=False)
            csv_files["xva"] = xva_file

        return csv_files

    def generate_excel_report(
        self, summary: Dict, xva_results: Optional[Dict] = None
    ) -> Path:
        """Generate Excel format report with multiple sheets."""
        output_file = self.output_dir / f"portfolio_report_{self.report_timestamp}.xlsx"

        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            # Summary sheet
            summary_data = {
                "Metric": [
                    "Portfolio Name",
                    "Base Currency",
                    "Valuation Date",
                    "Number of Counterparties",
                    "Number of Netting Sets",
                    "Total Trades",
                    "Active Trades",
                    "Total MTM (USD)",
                    "Gross Notional (USD)",
                ],
                "Value": [
                    summary["portfolio_name"],
                    summary["base_currency"],
                    summary.get("valuation_date", "N/A"),
                    summary["statistics"]["counterparties"],
                    summary["statistics"]["netting_sets"],
                    summary["statistics"]["trades"],
                    summary["statistics"]["active_trades"],
                    f"${summary['total_mtm']:,.2f}",
                    f"${summary['gross_notional']:,.2f}",
                ],
            }
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name="Summary", index=False)

            # Counterparties sheet
            cp_data = []
            for cp_id, cp_info in summary["counterparties"].items():
                cp_data.append(
                    {
                        "ID": cp_id,
                        "Name": cp_info["name"],
                        "Rating": cp_info["rating"],
                        "Entity Type": cp_info["entity_type"],
                        "Trades": cp_info["num_trades"],
                        "Net MTM": cp_info["net_mtm"],
                        "Gross Notional": cp_info["gross_notional"],
                        "CSA": "Yes" if cp_info["has_csa"] else "No",
                    }
                )
            df_cp = pd.DataFrame(cp_data)
            df_cp.to_excel(writer, sheet_name="Counterparties", index=False)

            # Books sheet
            book_data = []
            for book_id, book_info in summary["books"].items():
                book_data.append(
                    {
                        "ID": book_id,
                        "Name": book_info["name"],
                        "Desk": book_info.get("desk", "N/A"),
                        "Trades": book_info["num_trades"],
                        "Active": book_info["active_trades"],
                        "MTM": book_info["total_mtm"],
                    }
                )
            df_books = pd.DataFrame(book_data)
            df_books.to_excel(writer, sheet_name="Books", index=False)

            # Desks sheet
            desk_data = []
            for desk_id, desk_info in summary["desks"].items():
                desk_data.append(
                    {
                        "ID": desk_id,
                        "Name": desk_info["name"],
                        "Books": desk_info["num_books"],
                        "Trades": desk_info["num_trades"],
                        "MTM": desk_info["total_mtm"],
                    }
                )
            df_desks = pd.DataFrame(desk_data)
            df_desks.to_excel(writer, sheet_name="Desks", index=False)

            # XVA sheet (if available)
            if xva_results and "netting_set_xva" in xva_results:
                xva_data = []
                for ns_xva in xva_results["netting_set_xva"]:
                    xva_data.append(
                        {
                            "Netting Set": ns_xva["netting_set_id"],
                            "Counterparty": ns_xva["counterparty"],
                            "CSA": "Yes" if ns_xva["has_csa"] else "No",
                            "CVA": ns_xva["cva"],
                            "DVA": ns_xva["dva"],
                            "FVA": ns_xva["fva"],
                            "MVA": ns_xva["mva"],
                            "Total XVA": ns_xva["total_xva"],
                            "Net MTM": ns_xva["net_mtm"],
                        }
                    )
                df_xva = pd.DataFrame(xva_data)
                df_xva.to_excel(writer, sheet_name="XVA Results", index=False)

        return output_file

    def generate_html_report(
        self, summary: Dict, xva_results: Optional[Dict] = None
    ) -> Path:
        """Generate HTML format report."""
        output_file = self.output_dir / f"portfolio_report_{self.report_timestamp}.html"

        # Build HTML content
        html_content = self._build_html_report(summary, xva_results)

        with open(output_file, "w") as f:
            f.write(html_content)

        return output_file

    def _build_html_report(
        self, summary: Dict, xva_results: Optional[Dict] = None
    ) -> str:
        """Build HTML report content."""
        stats = summary["statistics"]

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Portfolio Report - {summary['portfolio_name']}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 5px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        .summary-card .value {{
            font-size: 28px;
            font-weight: bold;
            margin: 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th {{
            background-color: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ecf0f1;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .number {{
            text-align: right;
        }}
        .positive {{
            color: #27ae60;
        }}
        .negative {{
            color: #e74c3c;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }}
        .badge-yes {{
            background-color: #d4edda;
            color: #155724;
        }}
        .badge-no {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 14px;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Portfolio Analysis Report</h1>
        <p><strong>Portfolio:</strong> {summary['portfolio_name']}</p>
        <p><strong>Base Currency:</strong> {summary['base_currency']}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <h2>Portfolio Summary</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Counterparties</h3>
                <p class="value">{stats['counterparties']}</p>
            </div>
            <div class="summary-card">
                <h3>Netting Sets</h3>
                <p class="value">{stats['netting_sets']}</p>
            </div>
            <div class="summary-card">
                <h3>Total Trades</h3>
                <p class="value">{stats['trades']}</p>
            </div>
            <div class="summary-card">
                <h3>Total MTM</h3>
                <p class="value">${summary['total_mtm']:,.0f}</p>
            </div>
        </div>

        <h2>Counterparty Breakdown</h2>
        <table>
            <thead>
                <tr>
                    <th>Counterparty</th>
                    <th>Rating</th>
                    <th>Type</th>
                    <th class="number">Trades</th>
                    <th class="number">Net MTM (USD)</th>
                    <th class="number">Notional (USD)</th>
                    <th>CSA</th>
                </tr>
            </thead>
            <tbody>
"""

        # Add counterparty rows
        for cp_id, cp_info in summary["counterparties"].items():
            mtm = cp_info["net_mtm"]
            mtm_class = "positive" if mtm >= 0 else "negative"
            csa_badge = "badge-yes" if cp_info["has_csa"] else "badge-no"
            csa_text = "Yes" if cp_info["has_csa"] else "No"

            html += f"""
                <tr>
                    <td>{cp_info['name']}</td>
                    <td>{cp_info['rating']}</td>
                    <td>{cp_info['entity_type']}</td>
                    <td class="number">{cp_info['num_trades']}</td>
                    <td class="number {mtm_class}">${mtm:,.2f}</td>
                    <td class="number">${cp_info['gross_notional']:,.2f}</td>
                    <td><span class="badge {csa_badge}">{csa_text}</span></td>
                </tr>
"""

        html += """
            </tbody>
        </table>

        <h2>Desk Breakdown</h2>
        <table>
            <thead>
                <tr>
                    <th>Desk</th>
                    <th class="number">Books</th>
                    <th class="number">Trades</th>
                    <th class="number">MTM (USD)</th>
                </tr>
            </thead>
            <tbody>
"""

        # Add desk rows
        for desk_id, desk_info in summary["desks"].items():
            mtm = desk_info["total_mtm"]
            mtm_class = "positive" if mtm >= 0 else "negative"

            html += f"""
                <tr>
                    <td>{desk_info['name']}</td>
                    <td class="number">{desk_info['num_books']}</td>
                    <td class="number">{desk_info['num_trades']}</td>
                    <td class="number {mtm_class}">${mtm:,.2f}</td>
                </tr>
"""

        html += """
            </tbody>
        </table>
"""

        # Add XVA section if results are available
        if xva_results and "netting_set_xva" in xva_results:
            html += """
        <h2>XVA Analysis</h2>
        <table>
            <thead>
                <tr>
                    <th>Counterparty</th>
                    <th>CSA</th>
                    <th class="number">CVA (USD)</th>
                    <th class="number">DVA (USD)</th>
                    <th class="number">FVA (USD)</th>
                    <th class="number">MVA (USD)</th>
                    <th class="number">Total XVA (USD)</th>
                </tr>
            </thead>
            <tbody>
"""

            for ns_xva in xva_results["netting_set_xva"]:
                csa_badge = "badge-yes" if ns_xva["has_csa"] else "badge-no"
                csa_text = "Yes" if ns_xva["has_csa"] else "No"

                html += f"""
                <tr>
                    <td>{ns_xva['counterparty']}</td>
                    <td><span class="badge {csa_badge}">{csa_text}</span></td>
                    <td class="number">${ns_xva['cva']:,.2f}</td>
                    <td class="number">${ns_xva['dva']:,.2f}</td>
                    <td class="number">${ns_xva['fva']:,.2f}</td>
                    <td class="number">${ns_xva['mva']:,.2f}</td>
                    <td class="number"><strong>${ns_xva['total_xva']:,.2f}</strong></td>
                </tr>
"""

            html += """
            </tbody>
        </table>
"""

        html += f"""
        <div class="timestamp">
            <p>Report generated by Neutryx Portfolio Reporting System</p>
            <p>Timestamp: {datetime.now().isoformat()}</p>
        </div>
    </div>
</body>
</html>
"""

        return html


def main():
    """Main entry point."""
    print("=" * 80)
    print("Fictional Bank Portfolio - Comprehensive Reporting")
    print("=" * 80)
    print()

    # Load portfolio
    print("Loading portfolio...")
    portfolio, book_hierarchy = create_fictional_portfolio()
    print(f"✓ Portfolio loaded: {portfolio.name}")
    print()

    # Try to load XVA results if available
    xva_results = None
    xva_file = Path(__file__).parent / "reports" / "xva_results.json"

    if xva_file.exists():
        print(f"Loading XVA results from {xva_file}...")
        with open(xva_file, "r") as f:
            xva_results = json.load(f)
        print("✓ XVA results loaded")
        print()
    else:
        print("Note: XVA results not found. Run compute_xva.py first to include XVA metrics.")
        print()

    # Generate reports
    output_dir = Path(__file__).parent / "reports"
    reporter = PortfolioReporter(output_dir)

    reports = reporter.generate_all_reports(portfolio, book_hierarchy, xva_results)

    print()
    print("=" * 80)
    print("Report Generation Complete!")
    print("=" * 80)
    print()
    print("Generated Reports:")
    for report_type, file_path in reports.items():
        print(f"  {report_type}: {file_path.name}")
    print()


if __name__ == "__main__":
    main()
