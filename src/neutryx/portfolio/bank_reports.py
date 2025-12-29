"""Comprehensive reporting and analytics for fictional bank.

This module provides various report generators:
- Daily P&L reports
- Risk reports (exposure, concentration)
- Desk performance reports
- Counterparty credit reports
- Regulatory reports
- Executive dashboards
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional
import json

from neutryx.portfolio.fictional_bank import FictionalBank
from neutryx.portfolio.contracts.trade import ProductType, TradeStatus


class BankReportGenerator:
    """Generate comprehensive reports for fictional bank."""

    def __init__(self, bank: FictionalBank):
        """Initialize report generator.

        Parameters
        ----------
        bank : FictionalBank
            Bank to generate reports for
        """
        self.bank = bank

    async def generate_daily_pnl_report(
        self, report_date: Optional[date] = None
    ) -> Dict:
        """Generate daily P&L report.

        Parameters
        ----------
        report_date : date, optional
            Report date (defaults to today)

        Returns
        -------
        dict
            Daily P&L report
        """
        as_of = report_date or date.today()

        report = {
            "report_type": "Daily P&L",
            "bank_name": self.bank.name,
            "report_date": as_of.isoformat(),
            "total_mtm": self.bank.portfolio.calculate_total_mtm(),
            "gross_notional": self.bank.portfolio.calculate_gross_notional(),
            "by_desk": {},
            "by_product": {},
            "by_counterparty": {},
        }

        # Desk breakdown
        desk_ids = set(
            t.desk_id for t in self.bank.portfolio.trades.values() if t.desk_id
        )
        for desk_id in desk_ids:
            desk = self.bank.book_hierarchy.desks.get(desk_id)
            desk_mtm = self.bank.portfolio.calculate_mtm_by_desk(desk_id)
            desk_notional = self.bank.portfolio.calculate_notional_by_desk(desk_id)
            desk_trades = self.bank.portfolio.get_trades_by_desk(desk_id)

            report["by_desk"][desk_id] = {
                "name": desk.name if desk else desk_id,
                "mtm": desk_mtm,
                "notional": desk_notional,
                "num_trades": len(desk_trades),
            }

        # Product breakdown
        for product_type in ProductType:
            trades = self.bank.portfolio.get_trades_by_product_type(product_type)
            if trades:
                mtm = sum(t.get_mtm(0.0) for t in trades)
                notional = sum(t.notional or 0.0 for t in trades)

                report["by_product"][product_type.value] = {
                    "mtm": mtm,
                    "notional": notional,
                    "num_trades": len(trades),
                }

        # Counterparty breakdown
        for cp_id, counterparty in self.bank.portfolio.counterparties.items():
            cp_mtm = self.bank.portfolio.calculate_net_mtm_by_counterparty(cp_id)
            cp_trades = self.bank.portfolio.get_trades_by_counterparty(cp_id)

            report["by_counterparty"][cp_id] = {
                "name": counterparty.name,
                "mtm": cp_mtm,
                "num_trades": len(cp_trades),
            }

        return report

    async def generate_risk_report(self, report_date: Optional[date] = None) -> Dict:
        """Generate risk report with exposure and concentration metrics.

        Parameters
        ----------
        report_date : date, optional
            Report date (defaults to today)

        Returns
        -------
        dict
            Risk report
        """
        as_of = report_date or date.today()

        report = {
            "report_type": "Risk Report",
            "bank_name": self.bank.name,
            "report_date": as_of.isoformat(),
            "counterparty_exposures": {},
            "concentration_metrics": {},
            "credit_metrics": {},
        }

        total_exposure = 0.0
        exposures_by_rating = {}

        # Counterparty exposures
        for cp_id, counterparty in self.bank.portfolio.counterparties.items():
            exposure_data = await self.bank.get_counterparty_exposure(cp_id, as_of)

            exposure = exposure_data["total_mtm"]
            total_exposure += abs(exposure)

            rating = (
                counterparty.credit.rating.value
                if counterparty.credit and counterparty.credit.rating
                else "NR"
            )

            report["counterparty_exposures"][cp_id] = {
                "name": counterparty.name,
                "exposure": exposure,
                "rating": rating,
                "num_trades": exposure_data["trade_count"],
                "has_csa": len(exposure_data["csas"]) > 0,
                "lgd": counterparty.credit.lgd if counterparty.credit else 0.6,
                "credit_spread_bps": (
                    counterparty.credit.credit_spread_bps
                    if counterparty.credit
                    else None
                ),
            }

            # Aggregate by rating
            if rating not in exposures_by_rating:
                exposures_by_rating[rating] = {"count": 0, "total_exposure": 0.0}

            exposures_by_rating[rating]["count"] += 1
            exposures_by_rating[rating]["total_exposure"] += abs(exposure)

        # Concentration metrics
        report["concentration_metrics"] = {
            "total_exposure": total_exposure,
            "by_rating": exposures_by_rating,
            "top_exposures": self._get_top_exposures(
                report["counterparty_exposures"], n=5
            ),
        }

        # Credit metrics
        collateralized_exposure = sum(
            abs(exp["exposure"])
            for exp in report["counterparty_exposures"].values()
            if exp["has_csa"]
        )

        report["credit_metrics"] = {
            "collateralized_percentage": (
                collateralized_exposure / total_exposure * 100
                if total_exposure > 0
                else 0.0
            ),
            "uncollateralized_exposure": total_exposure - collateralized_exposure,
        }

        return report

    def _get_top_exposures(
        self, exposures: Dict, n: int = 5
    ) -> List[Dict]:
        """Get top N exposures.

        Parameters
        ----------
        exposures : dict
            Counterparty exposures
        n : int
            Number of top exposures to return

        Returns
        -------
        list
            Top N exposures
        """
        sorted_exposures = sorted(
            exposures.items(), key=lambda x: abs(x[1]["exposure"]), reverse=True
        )

        return [
            {
                "counterparty_id": cp_id,
                "name": data["name"],
                "exposure": data["exposure"],
                "rating": data["rating"],
            }
            for cp_id, data in sorted_exposures[:n]
        ]

    async def generate_desk_performance_report(
        self, report_date: Optional[date] = None
    ) -> Dict:
        """Generate desk performance report.

        Parameters
        ----------
        report_date : date, optional
            Report date (defaults to today)

        Returns
        -------
        dict
            Desk performance report
        """
        as_of = report_date or date.today()

        report = {
            "report_type": "Desk Performance",
            "bank_name": self.bank.name,
            "report_date": as_of.isoformat(),
            "desks": {},
        }

        desk_ids = set(
            t.desk_id for t in self.bank.portfolio.trades.values() if t.desk_id
        )

        for desk_id in desk_ids:
            desk = self.bank.book_hierarchy.desks.get(desk_id)
            if not desk:
                continue

            desk_summary = await self.bank.get_desk_summary(desk_id)

            # Get books for this desk
            books = [
                b
                for b in self.bank.book_hierarchy.books.values()
                if b.desk_id == desk_id
            ]

            # Get traders for this desk
            traders = [
                t
                for t in self.bank.book_hierarchy.traders.values()
                if t.desk_id == desk_id
            ]

            report["desks"][desk_id] = {
                "name": desk.name,
                "type": desk.desk_type,
                "summary": desk_summary,
                "books": [{"id": b.id, "name": b.name, "type": b.book_type} for b in books],
                "traders": [
                    {"id": t.id, "name": t.name, "email": t.email} for t in traders
                ],
            }

        return report

    async def generate_counterparty_credit_report(
        self, report_date: Optional[date] = None
    ) -> Dict:
        """Generate counterparty credit report.

        Parameters
        ----------
        report_date : date, optional
            Report date (defaults to today)

        Returns
        -------
        dict
            Counterparty credit report
        """
        as_of = report_date or date.today()

        report = {
            "report_type": "Counterparty Credit",
            "bank_name": self.bank.name,
            "report_date": as_of.isoformat(),
            "counterparties": {},
            "summary": {
                "total_counterparties": len(self.bank.portfolio.counterparties),
                "with_csa": 0,
                "without_csa": 0,
                "banks": 0,
                "corporates": 0,
                "funds": 0,
            },
        }

        for cp_id, counterparty in self.bank.portfolio.counterparties.items():
            exposure_data = await self.bank.get_counterparty_exposure(cp_id, as_of)

            csas = exposure_data["csas"]
            has_csa = len(csas) > 0

            if has_csa:
                report["summary"]["with_csa"] += 1
            else:
                report["summary"]["without_csa"] += 1

            if counterparty.is_bank:
                report["summary"]["banks"] += 1
            elif counterparty.entity_type.value == "Corporate":
                report["summary"]["corporates"] += 1
            elif counterparty.entity_type.value == "Fund":
                report["summary"]["funds"] += 1

            report["counterparties"][cp_id] = {
                "name": counterparty.name,
                "entity_type": counterparty.entity_type.value,
                "lei": counterparty.lei,
                "jurisdiction": counterparty.jurisdiction,
                "is_bank": counterparty.is_bank,
                "credit": {
                    "rating": (
                        counterparty.credit.rating.value
                        if counterparty.credit and counterparty.credit.rating
                        else "NR"
                    ),
                    "lgd": counterparty.credit.lgd if counterparty.credit else None,
                    "credit_spread_bps": (
                        counterparty.credit.credit_spread_bps
                        if counterparty.credit
                        else None
                    ),
                },
                "has_csa": has_csa,
                "csa_details": (
                    [
                        {
                            "id": csa.id,
                            "threshold_party_b": csa.threshold_terms.threshold_party_b,
                            "mta_party_b": csa.threshold_terms.mta_party_b,
                            "base_currency": csa.collateral_terms.base_currency,
                        }
                        for csa in csas
                    ]
                    if has_csa
                    else []
                ),
                "exposure": exposure_data["total_mtm"],
                "active_trades": exposure_data["trade_count"],
            }

        return report

    async def generate_executive_dashboard(
        self, report_date: Optional[date] = None
    ) -> Dict:
        """Generate executive dashboard with key metrics.

        Parameters
        ----------
        report_date : date, optional
            Report date (defaults to today)

        Returns
        -------
        dict
            Executive dashboard
        """
        as_of = report_date or date.today()

        # Get all reports
        pnl_report = await self.generate_daily_pnl_report(as_of)
        risk_report = await self.generate_risk_report(as_of)

        # Count active vs inactive trades
        active_trades = [
            t for t in self.bank.portfolio.trades.values() if t.status == TradeStatus.ACTIVE
        ]
        pending_trades = [
            t for t in self.bank.portfolio.trades.values() if t.status == TradeStatus.PENDING
        ]

        dashboard = {
            "report_type": "Executive Dashboard",
            "bank_name": self.bank.name,
            "report_date": as_of.isoformat(),
            "key_metrics": {
                "total_mtm": pnl_report["total_mtm"],
                "gross_notional": pnl_report["gross_notional"],
                "num_counterparties": len(self.bank.portfolio.counterparties),
                "num_active_trades": len(active_trades),
                "num_pending_trades": len(pending_trades),
                "num_desks": len(
                    set(t.desk_id for t in self.bank.portfolio.trades.values() if t.desk_id)
                ),
                "collateralized_percentage": risk_report["credit_metrics"][
                    "collateralized_percentage"
                ],
            },
            "desk_performance": pnl_report["by_desk"],
            "top_exposures": risk_report["concentration_metrics"]["top_exposures"],
            "rating_distribution": risk_report["concentration_metrics"]["by_rating"],
        }

        return dashboard

    def export_report_to_json(self, report: Dict, filename: str) -> None:
        """Export report to JSON file.

        Parameters
        ----------
        report : dict
            Report to export
        filename : str
            Output filename
        """
        with open(filename, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"✓ Report exported to {filename}")

    def print_executive_dashboard(self, dashboard: Dict) -> None:
        """Print executive dashboard in formatted text.

        Parameters
        ----------
        dashboard : dict
            Executive dashboard to print
        """
        print(f"\n{'='*80}")
        print(f"EXECUTIVE DASHBOARD - {dashboard['bank_name']}")
        print(f"Report Date: {dashboard['report_date']}")
        print(f"{'='*80}\n")

        metrics = dashboard["key_metrics"]
        print("KEY METRICS:")
        print(f"  Total MTM:              ${metrics['total_mtm']:>15,.2f}")
        print(f"  Gross Notional:         ${metrics['gross_notional']:>15,.2f}")
        print(f"  Active Trades:          {metrics['num_active_trades']:>17,}")
        print(f"  Pending Trades:         {metrics['num_pending_trades']:>17,}")
        print(f"  Counterparties:         {metrics['num_counterparties']:>17,}")
        print(f"  Trading Desks:          {metrics['num_desks']:>17,}")
        print(
            f"  Collateralized:         {metrics['collateralized_percentage']:>16.1f}%"
        )

        print(f"\n{'─'*80}")
        print("DESK PERFORMANCE:")
        for desk_id, desk_data in dashboard["desk_performance"].items():
            print(f"\n  {desk_data['name']}:")
            print(f"    MTM:          ${desk_data['mtm']:>12,.2f}")
            print(f"    Notional:     ${desk_data['notional']:>12,.2f}")
            print(f"    Trades:       {desk_data['num_trades']:>14,}")

        print(f"\n{'─'*80}")
        print("TOP 5 EXPOSURES:")
        for i, exp in enumerate(dashboard["top_exposures"][:5], 1):
            print(
                f"  {i}. {exp['name']:<30} ${exp['exposure']:>12,.2f}  [{exp['rating']}]"
            )

        print(f"\n{'─'*80}")
        print("RATING DISTRIBUTION:")
        for rating, data in sorted(dashboard["rating_distribution"].items()):
            print(
                f"  {rating:<10} {data['count']:>3} counterparties  ${data['total_exposure']:>12,.2f}"
            )

        print(f"\n{'='*80}\n")


__all__ = ["BankReportGenerator"]
