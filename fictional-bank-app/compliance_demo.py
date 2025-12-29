#!/usr/bin/env python3
"""Comprehensive Compliance Monitoring Demo

This script demonstrates the full compliance monitoring framework:
1. Loading portfolio and limit configurations
2. Checking all compliance limits
3. Generating breach reports and alerts
4. Creating interactive dashboards
5. Regulatory reporting

Run this to see the complete compliance monitoring system in action!
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from neutryx.tests.fixtures.fictional_portfolio import create_fictional_portfolio
from compliance_monitoring import ComplianceMonitor, Limit
from compliance_dashboard import ComplianceDashboard


def main():
    """Run comprehensive compliance monitoring demo."""
    print("=" * 80)
    print("Compliance Monitoring Demo - Fictional Bank Portfolio")
    print("=" * 80)
    print()

    # -------------------------------------------------------------------------
    # 1. Setup
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("STEP 1: Portfolio and Configuration Setup")
    print("=" * 80)
    print()

    # Load portfolio
    print("Loading portfolio...")
    portfolio, book_hierarchy = create_fictional_portfolio()
    print(f"✓ Portfolio loaded: {portfolio.name}")
    print()

    # Initialize compliance monitor with configuration
    config_file = Path(__file__).parent / "data" / "compliance" / "compliance_limits.yaml"

    if config_file.exists():
        print(f"Loading compliance limits from {config_file.name}...")
        monitor = ComplianceMonitor(config_file)
        print()
    else:
        print(f"Configuration file not found: {config_file}")
        print("Using programmatically defined limits...")
        monitor = ComplianceMonitor()

        # Add example limits programmatically
        monitor.add_limit(Limit(
            name="Portfolio Total Notional",
            limit_type="notional",
            value=200_000_000,
            scope="portfolio",
            description="Maximum portfolio notional exposure"
        ))

        monitor.add_limit(Limit(
            name="Portfolio Total MTM",
            limit_type="mtm",
            value=50_000_000,
            scope="portfolio",
            description="Maximum absolute MTM"
        ))

        monitor.add_limit(Limit(
            name="AAA Global Bank - Notional Limit",
            limit_type="notional",
            scope="counterparty",
            entity_id="CP001",
            value=75_000_000,
            description="Maximum notional with AAA rated bank"
        ))

        monitor.add_limit(Limit(
            name="IR Desk - Total Notional",
            limit_type="notional",
            scope="desk",
            entity_id="DESK001",
            value=100_000_000,
            description="Maximum notional for Interest Rates desk"
        ))

        print(f"✓ Added {len(monitor.limits)} limit(s)")
        print()

    print(f"Total limits configured: {len(monitor.limits)}")
    print()

    # -------------------------------------------------------------------------
    # 2. Run Compliance Checks
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("STEP 2: Running Compliance Checks")
    print("=" * 80)
    print()

    # Optional: Add Greeks/sensitivity metrics for risk limit checks
    # In a real system, these would come from sensitivity_analysis.py
    additional_metrics = {
        'total_delta': 25000,   # Example Delta exposure
        'total_gamma': 5000,    # Example Gamma exposure
        'total_vega': 15000,    # Example Vega exposure
        'total_theta': -2000,   # Example Theta exposure
        'total_rho': 8000,      # Example Rho exposure
    }

    # Check all limits
    report = monitor.check_all_limits(
        portfolio,
        book_hierarchy,
        additional_metrics=additional_metrics
    )

    # -------------------------------------------------------------------------
    # 3. Display Results
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("STEP 3: Compliance Report Summary")
    print("=" * 80)
    print()

    # Print detailed summary
    monitor.print_summary(report)
    print()

    # -------------------------------------------------------------------------
    # 4. Generate Reports
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("STEP 4: Generating Compliance Reports")
    print("=" * 80)
    print()

    output_dir = Path(__file__).parent / "reports"
    report_file = monitor.generate_breach_report(report, output_dir)
    print()

    # -------------------------------------------------------------------------
    # 5. Create Dashboards
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("STEP 5: Creating Interactive Dashboards")
    print("=" * 80)
    print()

    dashboard_dir = Path(__file__).parent / "sample_outputs" / "charts"
    dashboard = ComplianceDashboard(dashboard_dir)

    dashboards = dashboard.create_all_dashboards(report, create_interactive=True)
    print()

    # -------------------------------------------------------------------------
    # 6. Generate Alerts
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("STEP 6: Alert Generation")
    print("=" * 80)
    print()

    # Get critical alerts
    from compliance_monitoring import AlertLevel

    breaches = [b for b in report.breaches if b.alert_level == AlertLevel.BREACH]
    critical = [b for b in report.breaches if b.alert_level == AlertLevel.RED]
    warnings = [b for b in report.breaches if b.alert_level == AlertLevel.YELLOW]

    print("Alert Summary:")
    print(f"  🚨 BREACHES (Immediate Action Required): {len(breaches)}")
    if breaches:
        for breach in breaches[:3]:
            print(f"      - {breach.limit_name}: {breach.utilization_pct:.1f}% (over limit by ${breach.breach_amount:,.0f})")
        if len(breaches) > 3:
            print(f"      ... and {len(breaches) - 3} more")

    print()
    print(f"  🔴 CRITICAL (Action Needed Soon): {len(critical)}")
    if critical:
        for crit in critical[:3]:
            print(f"      - {crit.limit_name}: {crit.utilization_pct:.1f}%")
        if len(critical) > 3:
            print(f"      ... and {len(critical) - 3} more")

    print()
    print(f"  🟡 WARNINGS (Monitor Closely): {len(warnings)}")
    if warnings:
        for warn in warnings[:3]:
            print(f"      - {warn.limit_name}: {warn.utilization_pct:.1f}%")
        if len(warnings) > 3:
            print(f"      ... and {len(warnings) - 3} more")

    print()

    # -------------------------------------------------------------------------
    # 7. Remediation Suggestions
    # -------------------------------------------------------------------------
    if breaches or critical:
        print("=" * 80)
        print("STEP 7: Remediation Suggestions")
        print("=" * 80)
        print()

        print("Suggested Actions:")
        print()

        if breaches:
            print("🚨 IMMEDIATE ACTIONS (Limit Breaches):")
            for i, breach in enumerate(breaches[:5], 1):
                print(f"{i}. {breach.limit_name}")
                print(f"   Current: ${breach.current_value:,.0f} | Limit: ${breach.limit_value:,.0f}")
                print(f"   Breach: ${breach.breach_amount:,.0f}")
                print(f"   → REQUIRED: Reduce exposure immediately")
                print()

        if critical:
            print("🔴 NEAR-TERM ACTIONS (Critical Warnings):")
            for i, crit in enumerate(critical[:5], 1):
                print(f"{i}. {crit.limit_name}")
                print(f"   Current: ${crit.current_value:,.0f} | Limit: ${crit.limit_value:,.0f}")
                print(f"   Utilization: {crit.utilization_pct:.1f}%")
                print(f"   → RECOMMENDED: Review and consider reducing exposure")
                print()

    # -------------------------------------------------------------------------
    # 8. Summary
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("COMPLIANCE MONITORING SUMMARY")
    print("=" * 80)
    print()

    print("Checks Performed:")
    print(f"  Total limits checked: {report.total_limits}")
    print(f"  Portfolio: {report.portfolio_name}")
    print(f"  Timestamp: {report.report_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Status Overview:")
    print(f"  🟢 OK: {report.limits_ok} ({report.limits_ok/report.total_limits*100:.1f}%)")
    print(f"  🟡 Warning: {report.limits_warning} ({report.limits_warning/report.total_limits*100:.1f}%)")
    print(f"  🔴 Critical: {report.limits_critical} ({report.limits_critical/report.total_limits*100:.1f}%)")
    print(f"  🚨 Breached: {report.limits_breached} ({report.limits_breached/report.total_limits*100:.1f}%)")
    print()

    health_score = report.summary_stats['health_score']
    health_icon = "🟢" if health_score >= 80 else "🟡" if health_score >= 60 else "🔴"
    print(f"  {health_icon} Health Score: {health_score:.1f}%")
    print(f"     (Percentage of limits in OK status)")
    print()

    print("Generated Outputs:")
    print(f"  Reports: {output_dir}")
    print(f"    - JSON: {report_file.name}")
    print(f"    - CSV: {report_file.stem.replace('report', 'report')}.csv")
    print(f"    - Excel: {report_file.stem}.xlsx")
    print()

    print(f"  Dashboards ({len(dashboards)}): {dashboard_dir}")
    for name, path in sorted(dashboards.items()):
        icon = "🌐" if path.suffix == ".html" else "📊"
        print(f"    {icon} {name}: {path.name}")
    print()

    # -------------------------------------------------------------------------
    # 9. Next Steps
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()

    print("1. Review Compliance Reports:")
    print(f"   - Open: {output_dir / report_file.name}")
    print(f"   - Or Excel: {output_dir / report_file.stem}.xlsx")
    print()

    print("2. View Interactive Dashboards:")
    if "interactive_dashboard" in dashboards:
        print(f"   - Main Dashboard: {dashboards['interactive_dashboard']}")
    if "gauge_dashboard" in dashboards:
        print(f"   - Gauge Dashboard: {dashboards['gauge_dashboard']}")
    print()

    print("3. Address Critical Issues:")
    if breaches:
        print(f"   - {len(breaches)} limit breach(es) require immediate action")
    if critical:
        print(f"   - {len(critical)} critical warning(s) need attention")
    if not breaches and not critical:
        print("   - ✓ No critical issues detected")
    print()

    print("4. Customize Limits:")
    if config_file.exists():
        print(f"   - Edit: {config_file}")
    print("   - Add custom limits for your risk tolerance")
    print("   - Adjust thresholds (soft/hard)")
    print()

    print("5. Automate Monitoring:")
    print("   - Schedule regular compliance checks")
    print("   - Set up email/SMS alerts for breaches")
    print("   - Integrate with risk management systems")
    print()

    print("=" * 80)
    print("Compliance Monitoring Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
