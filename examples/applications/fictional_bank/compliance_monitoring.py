#!/usr/bin/env python3
"""Compliance Monitoring Framework

This module provides comprehensive compliance monitoring and limit checking:
- Trading limits by counterparty, desk, product type
- Risk metric limits (Delta, Gamma, Vega, notional, concentration)
- Real-time limit breach detection
- Alert generation with severity levels
- Regulatory reporting

Key Features:
- Configurable limit structures (YAML-based)
- Multi-level limit hierarchies
- Color-coded alert system (GREEN/YELLOW/RED)
- Comprehensive breach reporting
- Utilization tracking
"""
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from neutryx.tests.fixtures.fictional_portfolio import get_portfolio_summary


class AlertLevel(Enum):
    """Alert severity levels."""
    GREEN = "green"      # < 70% utilization
    YELLOW = "yellow"    # 70-90% utilization
    RED = "red"          # > 90% utilization
    BREACH = "breach"    # Limit exceeded


@dataclass
class Limit:
    """Definition of a trading or risk limit."""
    name: str
    limit_type: str  # 'notional', 'mtm', 'delta', 'gamma', 'vega', 'concentration', etc.
    value: float
    currency: str = "USD"
    scope: str = "portfolio"  # 'portfolio', 'counterparty', 'desk', 'product'
    entity_id: Optional[str] = None  # Specific counterparty/desk/product ID
    soft_limit: Optional[float] = None  # Warning threshold (default 70%)
    hard_limit: Optional[float] = None  # Critical threshold (default 90%)
    description: str = ""


@dataclass
class LimitBreach:
    """Record of a limit breach or warning."""
    limit_name: str
    limit_type: str
    limit_value: float
    current_value: float
    utilization_pct: float
    alert_level: AlertLevel
    scope: str
    entity_id: Optional[str]
    entity_name: Optional[str]
    timestamp: datetime
    message: str
    breach_amount: Optional[float] = None


@dataclass
class ComplianceReport:
    """Comprehensive compliance status report."""
    report_date: datetime
    portfolio_name: str
    total_limits: int
    limits_ok: int
    limits_warning: int
    limits_critical: int
    limits_breached: int
    breaches: List[LimitBreach] = field(default_factory=list)
    summary_stats: Dict[str, Any] = field(default_factory=dict)


class ComplianceMonitor:
    """Monitor portfolio compliance with trading and risk limits."""

    def __init__(self, config_file: Optional[Path] = None):
        """Initialize the compliance monitor.

        Args:
            config_file: Path to YAML configuration file with limit definitions
        """
        self.config_file = config_file
        self.limits: List[Limit] = []
        self.breaches: List[LimitBreach] = []

        # Default thresholds
        self.soft_threshold = 0.70  # 70% = Yellow warning
        self.hard_threshold = 0.90  # 90% = Red critical

        if config_file and config_file.exists():
            self.load_limits(config_file)

    def load_limits(self, config_file: Path):
        """Load limit definitions from YAML configuration.

        Args:
            config_file: Path to YAML file
        """
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Load thresholds
        if 'thresholds' in config:
            self.soft_threshold = config['thresholds'].get('soft', 0.70)
            self.hard_threshold = config['thresholds'].get('hard', 0.90)

        # Load limits
        for limit_def in config.get('limits', []):
            limit = Limit(
                name=limit_def['name'],
                limit_type=limit_def['type'],
                value=limit_def['value'],
                currency=limit_def.get('currency', 'USD'),
                scope=limit_def.get('scope', 'portfolio'),
                entity_id=limit_def.get('entity_id'),
                soft_limit=limit_def.get('soft_limit'),
                hard_limit=limit_def.get('hard_limit'),
                description=limit_def.get('description', ''),
            )
            self.limits.append(limit)

        print(f"✓ Loaded {len(self.limits)} limit(s) from {config_file.name}")

    def add_limit(self, limit: Limit):
        """Add a limit to the monitoring framework.

        Args:
            limit: Limit object to add
        """
        self.limits.append(limit)

    def check_all_limits(
        self,
        portfolio: Any,
        book_hierarchy: Any,
        additional_metrics: Optional[Dict[str, float]] = None
    ) -> ComplianceReport:
        """Check all limits and generate compliance report.

        Args:
            portfolio: Portfolio object
            book_hierarchy: Book hierarchy
            additional_metrics: Optional dictionary of additional metrics (Greeks, etc.)

        Returns:
            ComplianceReport with all breaches and warnings
        """
        print("Checking compliance limits...")
        print()

        # Get portfolio summary
        summary = get_portfolio_summary(portfolio, book_hierarchy)

        # Merge additional metrics
        metrics = {
            'total_mtm': summary['total_mtm'],
            'total_notional': summary['total_notional'],
            'num_counterparties': len(summary['counterparties']),
            'num_trades': summary['num_trades'],
        }

        if additional_metrics:
            metrics.update(additional_metrics)

        # Check each limit
        self.breaches = []

        for limit in self.limits:
            breach = self._check_limit(limit, summary, metrics)
            if breach:
                self.breaches.append(breach)

                # Print breach/warning
                color = "🔴" if breach.alert_level in [AlertLevel.RED, AlertLevel.BREACH] else "🟡"
                if breach.alert_level == AlertLevel.GREEN:
                    color = "🟢"

                if breach.alert_level != AlertLevel.GREEN:
                    print(f"{color} {breach.message}")

        print()

        # Generate report
        report = self._generate_report(summary['portfolio_name'])

        return report

    def _check_limit(
        self,
        limit: Limit,
        summary: Dict,
        metrics: Dict[str, float]
    ) -> Optional[LimitBreach]:
        """Check a single limit.

        Args:
            limit: Limit to check
            summary: Portfolio summary
            metrics: Current metric values

        Returns:
            LimitBreach if limit is breached or warning threshold exceeded, None otherwise
        """
        # Get current value based on limit type and scope
        current_value, entity_name = self._get_current_value(limit, summary, metrics)

        if current_value is None:
            return None

        # Calculate utilization
        utilization_pct = (current_value / limit.value * 100) if limit.value != 0 else 0

        # Determine alert level
        soft_limit = limit.soft_limit if limit.soft_limit else limit.value * self.soft_threshold
        hard_limit = limit.hard_limit if limit.hard_limit else limit.value * self.hard_threshold

        if current_value >= limit.value:
            alert_level = AlertLevel.BREACH
            message = f"BREACH: {limit.name} - {current_value:,.0f} / {limit.value:,.0f} ({utilization_pct:.1f}%)"
            breach_amount = current_value - limit.value
        elif current_value >= hard_limit:
            alert_level = AlertLevel.RED
            message = f"CRITICAL: {limit.name} - {current_value:,.0f} / {limit.value:,.0f} ({utilization_pct:.1f}%)"
            breach_amount = None
        elif current_value >= soft_limit:
            alert_level = AlertLevel.YELLOW
            message = f"WARNING: {limit.name} - {current_value:,.0f} / {limit.value:,.0f} ({utilization_pct:.1f}%)"
            breach_amount = None
        else:
            alert_level = AlertLevel.GREEN
            message = f"OK: {limit.name} - {current_value:,.0f} / {limit.value:,.0f} ({utilization_pct:.1f}%)"
            breach_amount = None

        return LimitBreach(
            limit_name=limit.name,
            limit_type=limit.limit_type,
            limit_value=limit.value,
            current_value=current_value,
            utilization_pct=utilization_pct,
            alert_level=alert_level,
            scope=limit.scope,
            entity_id=limit.entity_id,
            entity_name=entity_name,
            timestamp=datetime.now(),
            message=message,
            breach_amount=breach_amount,
        )

    def _get_current_value(
        self,
        limit: Limit,
        summary: Dict,
        metrics: Dict[str, float]
    ) -> Tuple[Optional[float], Optional[str]]:
        """Get current value for a limit.

        Args:
            limit: Limit definition
            summary: Portfolio summary
            metrics: Current metrics

        Returns:
            Tuple of (current_value, entity_name)
        """
        entity_name = None

        if limit.scope == 'portfolio':
            # Portfolio-level limit
            if limit.limit_type == 'notional':
                return abs(summary['total_notional']), None
            elif limit.limit_type == 'mtm':
                return abs(summary['total_mtm']), None
            elif limit.limit_type == 'num_counterparties':
                return len(summary['counterparties']), None
            elif limit.limit_type == 'num_trades':
                return summary['num_trades'], None
            elif limit.limit_type in metrics:
                return abs(metrics[limit.limit_type]), None

        elif limit.scope == 'counterparty':
            # Counterparty-specific limit
            if limit.entity_id and limit.entity_id in summary['counterparties']:
                cp = summary['counterparties'][limit.entity_id]
                entity_name = cp['name']

                if limit.limit_type == 'notional':
                    return abs(cp['gross_notional']), entity_name
                elif limit.limit_type == 'mtm':
                    return abs(cp['net_mtm']), entity_name
                elif limit.limit_type == 'num_trades':
                    return cp['num_trades'], entity_name

        elif limit.scope == 'desk':
            # Desk-specific limit
            if limit.entity_id and limit.entity_id in summary['desks']:
                desk = summary['desks'][limit.entity_id]
                entity_name = desk['name']

                if limit.limit_type == 'notional':
                    return abs(desk['gross_notional']), entity_name
                elif limit.limit_type == 'mtm':
                    return abs(desk['total_mtm']), entity_name
                elif limit.limit_type == 'num_trades':
                    return desk['num_trades'], entity_name

        elif limit.scope == 'concentration':
            # Concentration limits (e.g., max exposure to single counterparty as % of total)
            if limit.entity_id and limit.entity_id in summary['counterparties']:
                cp = summary['counterparties'][limit.entity_id]
                entity_name = cp['name']

                if limit.limit_type == 'concentration_pct':
                    total = abs(summary['total_notional'])
                    cp_notional = abs(cp['gross_notional'])
                    concentration_pct = (cp_notional / total * 100) if total > 0 else 0
                    return concentration_pct, entity_name

        return None, None

    def _generate_report(self, portfolio_name: str) -> ComplianceReport:
        """Generate compliance report from breach records.

        Args:
            portfolio_name: Name of the portfolio

        Returns:
            ComplianceReport object
        """
        limits_ok = sum(1 for b in self.breaches if b.alert_level == AlertLevel.GREEN)
        limits_warning = sum(1 for b in self.breaches if b.alert_level == AlertLevel.YELLOW)
        limits_critical = sum(1 for b in self.breaches if b.alert_level == AlertLevel.RED)
        limits_breached = sum(1 for b in self.breaches if b.alert_level == AlertLevel.BREACH)

        # Calculate summary statistics
        summary_stats = {
            'health_score': (limits_ok / len(self.breaches) * 100) if self.breaches else 100.0,
            'avg_utilization': sum(b.utilization_pct for b in self.breaches) / len(self.breaches) if self.breaches else 0,
            'max_utilization': max((b.utilization_pct for b in self.breaches), default=0),
            'total_breach_amount': sum(b.breach_amount for b in self.breaches if b.breach_amount),
        }

        return ComplianceReport(
            report_date=datetime.now(),
            portfolio_name=portfolio_name,
            total_limits=len(self.breaches),
            limits_ok=limits_ok,
            limits_warning=limits_warning,
            limits_critical=limits_critical,
            limits_breached=limits_breached,
            breaches=self.breaches,
            summary_stats=summary_stats,
        )

    def generate_breach_report(self, report: ComplianceReport, output_dir: Path) -> Path:
        """Generate detailed breach report files.

        Args:
            report: ComplianceReport object
            output_dir: Directory to save reports

        Returns:
            Path to JSON report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Prepare data for export
        breach_data = []
        for breach in report.breaches:
            breach_data.append({
                'limit_name': breach.limit_name,
                'limit_type': breach.limit_type,
                'scope': breach.scope,
                'entity': breach.entity_name or breach.entity_id or 'Portfolio',
                'limit_value': breach.limit_value,
                'current_value': breach.current_value,
                'utilization_pct': breach.utilization_pct,
                'alert_level': breach.alert_level.value,
                'breach_amount': breach.breach_amount,
                'timestamp': breach.timestamp.isoformat(),
                'message': breach.message,
            })

        df = pd.DataFrame(breach_data)

        # JSON report
        json_file = output_dir / f"compliance_report_{timestamp}.json"
        report_dict = {
            'report_metadata': {
                'generated_at': report.report_date.isoformat(),
                'portfolio': report.portfolio_name,
            },
            'summary': {
                'total_limits': report.total_limits,
                'limits_ok': report.limits_ok,
                'limits_warning': report.limits_warning,
                'limits_critical': report.limits_critical,
                'limits_breached': report.limits_breached,
                'health_score': report.summary_stats['health_score'],
                'avg_utilization': report.summary_stats['avg_utilization'],
                'max_utilization': report.summary_stats['max_utilization'],
            },
            'breaches': breach_data,
        }

        with open(json_file, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)

        # CSV report
        csv_file = output_dir / f"compliance_report_{timestamp}.csv"
        df.to_csv(csv_file, index=False)

        # Excel report with multiple sheets
        excel_file = output_dir / f"compliance_report_{timestamp}.xlsx"
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Summary sheet
            summary_df = pd.DataFrame([{
                'Portfolio': report.portfolio_name,
                'Report Date': report.report_date,
                'Total Limits': report.total_limits,
                'OK (Green)': report.limits_ok,
                'Warning (Yellow)': report.limits_warning,
                'Critical (Red)': report.limits_critical,
                'Breached': report.limits_breached,
                'Health Score': f"{report.summary_stats['health_score']:.1f}%",
                'Avg Utilization': f"{report.summary_stats['avg_utilization']:.1f}%",
                'Max Utilization': f"{report.summary_stats['max_utilization']:.1f}%",
            }])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # All breaches
            df.to_excel(writer, sheet_name='All Limits', index=False)

            # By alert level
            for level in [AlertLevel.BREACH, AlertLevel.RED, AlertLevel.YELLOW]:
                level_df = df[df['alert_level'] == level.value]
                if not level_df.empty:
                    level_df.to_excel(writer, sheet_name=level.value.upper(), index=False)

        print(f"✓ Compliance reports generated:")
        print(f"  JSON: {json_file.name}")
        print(f"  CSV: {csv_file.name}")
        print(f"  Excel: {excel_file.name}")

        return json_file

    def print_summary(self, report: ComplianceReport):
        """Print compliance report summary to console.

        Args:
            report: ComplianceReport object
        """
        print("=" * 80)
        print("COMPLIANCE MONITORING SUMMARY")
        print("=" * 80)
        print()
        print(f"Portfolio: {report.portfolio_name}")
        print(f"Report Date: {report.report_date.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Overall status
        print("Overall Status:")
        print("-" * 80)
        print(f"  Total Limits Checked: {report.total_limits}")
        print(f"  🟢 OK (Green):        {report.limits_ok:3d}  ({report.limits_ok/report.total_limits*100:5.1f}%)")
        print(f"  🟡 Warning (Yellow):  {report.limits_warning:3d}  ({report.limits_warning/report.total_limits*100:5.1f}%)")
        print(f"  🔴 Critical (Red):    {report.limits_critical:3d}  ({report.limits_critical/report.total_limits*100:5.1f}%)")
        print(f"  🚨 Breached:          {report.limits_breached:3d}  ({report.limits_breached/report.total_limits*100:5.1f}%)")
        print()

        # Health score
        health_score = report.summary_stats['health_score']
        health_color = "🟢" if health_score >= 80 else "🟡" if health_score >= 60 else "🔴"
        print(f"  {health_color} Health Score: {health_score:.1f}%")
        print(f"     Average Utilization: {report.summary_stats['avg_utilization']:.1f}%")
        print(f"     Maximum Utilization: {report.summary_stats['max_utilization']:.1f}%")
        print()

        # Breaches and critical warnings
        critical_breaches = [b for b in report.breaches if b.alert_level in [AlertLevel.BREACH, AlertLevel.RED]]

        if critical_breaches:
            print("Critical Issues:")
            print("-" * 80)
            for breach in sorted(critical_breaches, key=lambda x: x.utilization_pct, reverse=True):
                icon = "🚨" if breach.alert_level == AlertLevel.BREACH else "🔴"
                entity_str = f" [{breach.entity_name or breach.entity_id}]" if breach.entity_name or breach.entity_id else ""
                print(f"  {icon} {breach.limit_name}{entity_str}")
                print(f"      Current: {breach.current_value:,.0f}  |  Limit: {breach.limit_value:,.0f}  |  Utilization: {breach.utilization_pct:.1f}%")
                if breach.breach_amount:
                    print(f"      Breach Amount: {breach.breach_amount:,.0f}")
            print()

        # Warnings
        warnings = [b for b in report.breaches if b.alert_level == AlertLevel.YELLOW]

        if warnings:
            print(f"Warnings ({len(warnings)}):")
            print("-" * 80)
            for breach in sorted(warnings, key=lambda x: x.utilization_pct, reverse=True)[:5]:
                entity_str = f" [{breach.entity_name or breach.entity_id}]" if breach.entity_name or breach.entity_id else ""
                print(f"  🟡 {breach.limit_name}{entity_str}: {breach.utilization_pct:.1f}% utilized")

            if len(warnings) > 5:
                print(f"     ... and {len(warnings) - 5} more")
            print()


def main():
    """Demo of compliance monitoring."""
    print("=" * 80)
    print("Fictional Bank - Compliance Monitoring Demo")
    print("=" * 80)
    print()

    # Load portfolio
    from neutryx.tests.fixtures.fictional_portfolio import create_fictional_portfolio

    print("Loading portfolio...")
    portfolio, book_hierarchy = create_fictional_portfolio()
    print(f"✓ Portfolio loaded: {portfolio.name}")
    print()

    # Initialize compliance monitor
    # Note: In real usage, load from config file
    # monitor = ComplianceMonitor(config_file=Path("compliance_limits.yaml"))
    monitor = ComplianceMonitor()

    # Add some example limits programmatically
    monitor.add_limit(Limit(
        name="Portfolio Total Notional",
        limit_type="notional",
        value=200_000_000,  # $200M
        scope="portfolio",
        description="Maximum portfolio notional exposure"
    ))

    monitor.add_limit(Limit(
        name="Portfolio Total MTM",
        limit_type="mtm",
        value=50_000_000,  # $50M
        scope="portfolio",
        description="Maximum absolute MTM"
    ))

    # Check compliance
    report = monitor.check_all_limits(portfolio, book_hierarchy)

    # Print summary
    monitor.print_summary(report)

    # Generate reports
    output_dir = Path(__file__).parent / "reports"
    monitor.generate_breach_report(report, output_dir)
    print()

    print("=" * 80)
    print("Compliance Monitoring Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
