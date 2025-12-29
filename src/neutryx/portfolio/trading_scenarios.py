"""Trading scenarios and simulations for fictional bank.

This module provides realistic trading scenarios that demonstrate:
- Daily trading operations
- Multi-desk coordination
- Trade lifecycle management
- Risk events and mitigation
- Settlement workflows
- Market stress scenarios
"""
from __future__ import annotations

import asyncio
import random
from datetime import date, timedelta
from typing import Dict, List

from neutryx.portfolio.fictional_bank import FictionalBank
from neutryx.portfolio.contracts.trade import ProductType, TradeStatus
from neutryx.portfolio.trade_execution_service import ExecutionResult


class TradingScenario:
    """Base class for trading scenarios."""

    def __init__(self, name: str, description: str):
        """Initialize scenario.

        Parameters
        ----------
        name : str
            Scenario name
        description : str
            Scenario description
        """
        self.name = name
        self.description = description
        self.results: List[ExecutionResult] = []

    async def execute(self, bank: FictionalBank) -> Dict:
        """Execute the scenario.

        Parameters
        ----------
        bank : FictionalBank
            Bank to execute scenario on

        Returns
        -------
        dict
            Scenario results
        """
        raise NotImplementedError

    def print_results(self) -> None:
        """Print scenario results."""
        print(f"\n{'='*80}")
        print(f"Scenario: {self.name}")
        print(f"Description: {self.description}")
        print(f"{'='*80}")

        success_count = sum(1 for r in self.results if r.is_success())
        failed_count = sum(1 for r in self.results if r.is_failed())

        print(f"\nResults:")
        print(f"  Total trades: {len(self.results)}")
        print(f"  Successful: {success_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Success rate: {success_count/len(self.results)*100:.1f}%")


class DailyTradingScenario(TradingScenario):
    """Simulate a typical trading day across all desks."""

    def __init__(self):
        super().__init__(
            name="Daily Trading Operations",
            description="Simulate normal trading day with multiple trades across desks",
        )

    async def execute(self, bank: FictionalBank) -> Dict:
        """Execute daily trading scenario."""
        print(f"\n--- Starting {self.name} ---\n")

        trade_date = date.today()
        self.results = []

        # Rates desk: Book 3 IRS trades
        print("📊 Rates Desk:")
        for i in range(3):
            counterparty_id = random.choice(["CP_BANK_AAA", "CP_CORP_A", "CP_INSURANCE"])
            notional = random.randint(5, 50) * 1_000_000
            maturity_years = random.randint(2, 10)

            result = await bank.book_trade(
                counterparty_id=counterparty_id,
                product_type=ProductType.INTEREST_RATE_SWAP,
                trade_date=trade_date,
                notional=float(notional),
                currency="USD",
                maturity_date=trade_date + timedelta(days=365 * maturity_years),
                book_id="BOOK_IRS_USD",
                desk_id="DESK_RATES",
                trader_id="TRADER_001",
                product_details={
                    "fixed_rate": 0.03 + random.random() * 0.03,
                    "floating_index": "USD-SOFR",
                    "payment_frequency": "quarterly",
                },
                validate_counterparty=True,
                auto_confirm=True,
            )
            self.results.append(result)

        # FX desk: Book 2 FX options
        print("\n💱 FX Desk:")
        for i in range(2):
            counterparty_id = random.choice(["CP_CORP_A", "CP_CORP_BBB", "CP_HEDGE_FUND"])
            notional = random.randint(3, 10) * 1_000_000

            result = await bank.book_trade(
                counterparty_id=counterparty_id,
                product_type=ProductType.FX_OPTION,
                trade_date=trade_date,
                notional=float(notional),
                currency="EUR",
                maturity_date=trade_date + timedelta(days=90),
                book_id="BOOK_FX_MAJORS",
                desk_id="DESK_FX",
                trader_id="TRADER_003",
                product_details={
                    "currency_pair": "EUR/USD",
                    "strike": 1.05 + random.random() * 0.15,
                    "option_type": "call",
                },
                validate_counterparty=True,
                auto_confirm=True,
            )
            self.results.append(result)

        # Equity desk: Book 1 equity option
        print("\n📈 Equity Desk:")
        counterparty_id = random.choice(["CP_SOVEREIGN", "CP_HEDGE_FUND"])
        notional = random.randint(10, 30) * 1_000_000

        result = await bank.book_trade(
            counterparty_id=counterparty_id,
            product_type=ProductType.EQUITY_OPTION,
            trade_date=trade_date,
            notional=float(notional),
            currency="USD",
            maturity_date=trade_date + timedelta(days=180),
            book_id="BOOK_EQ_VANILLA",
            desk_id="DESK_EQUITY",
            trader_id="TRADER_005",
            product_details={
                "underlyer": "SPX",
                "strike": 4500.0 + random.random() * 500,
                "option_type": "call",
            },
            validate_counterparty=True,
            auto_confirm=True,
        )
        self.results.append(result)

        self.print_results()

        return {
            "scenario": self.name,
            "trades_booked": len(self.results),
            "success_rate": sum(1 for r in self.results if r.is_success()) / len(self.results),
        }


class CounterpartyOnboardingScenario(TradingScenario):
    """Simulate onboarding a new counterparty and executing first trades."""

    def __init__(self):
        super().__init__(
            name="Counterparty Onboarding",
            description="Onboard new counterparty with CSA and execute initial trades",
        )

    async def execute(self, bank: FictionalBank) -> Dict:
        """Execute onboarding scenario."""
        print(f"\n--- Starting {self.name} ---\n")

        from neutryx.portfolio.contracts.counterparty import (
            Counterparty,
            CounterpartyCredit,
            CreditRating,
            EntityType,
        )
        from neutryx.portfolio.contracts.csa import (
            CSA,
            CollateralTerms,
            ThresholdTerms,
            EligibleCollateral,
            CollateralType,
            ValuationFrequency,
        )

        # Step 1: Create new counterparty
        print("📝 Step 1: Creating new counterparty")
        new_cp = Counterparty(
            id="CP_NEW_CLIENT",
            name="New Technology Corp",
            entity_type=EntityType.CORPORATE,
            lei="NEW123456789TECHCORP",
            jurisdiction="US",
            credit=CounterpartyCredit(
                rating=CreditRating.BBB_PLUS,
                lgd=0.55,
                credit_spread_bps=110.0,
            ),
        )

        await bank.add_counterparty(new_cp, persist=True)

        # Step 2: Create CSA
        print("\n📋 Step 2: Creating CSA agreement")
        new_csa = CSA(
            id="CSA_CP_NEW_CLIENT",
            party_a_id=bank.bank_id,
            party_b_id="CP_NEW_CLIENT",
            effective_date=date.today().isoformat(),
            threshold_terms=ThresholdTerms(
                threshold_party_a=500_000.0,
                threshold_party_b=2_000_000.0,
                mta_party_a=100_000.0,
                mta_party_b=100_000.0,
            ),
            collateral_terms=CollateralTerms(
                base_currency="USD",
                valuation_frequency=ValuationFrequency.DAILY,
                eligible_collateral=[
                    EligibleCollateral(
                        collateral_type=CollateralType.CASH,
                        currency="USD",
                        haircut=0.0,
                    ),
                ],
            ),
        )

        await bank.add_csa(new_csa, persist=True)

        # Step 3: Execute first trades
        print("\n💼 Step 3: Executing initial trades")
        self.results = []

        # First trade: IRS
        result1 = await bank.book_trade(
            counterparty_id="CP_NEW_CLIENT",
            product_type=ProductType.INTEREST_RATE_SWAP,
            trade_date=date.today(),
            notional=15_000_000.0,
            currency="USD",
            maturity_date=date.today() + timedelta(days=365 * 5),
            book_id="BOOK_IRS_USD",
            product_details={
                "fixed_rate": 0.042,
                "floating_index": "USD-SOFR",
            },
            validate_counterparty=True,
            validate_csa=True,
            auto_confirm=True,
        )
        self.results.append(result1)

        # Second trade: FX Option
        result2 = await bank.book_trade(
            counterparty_id="CP_NEW_CLIENT",
            product_type=ProductType.FX_OPTION,
            trade_date=date.today(),
            notional=5_000_000.0,
            currency="EUR",
            maturity_date=date.today() + timedelta(days=90),
            book_id="BOOK_FX_MAJORS",
            product_details={
                "currency_pair": "EUR/USD",
                "strike": 1.10,
                "option_type": "put",
            },
            validate_counterparty=True,
            validate_csa=True,
            auto_confirm=True,
        )
        self.results.append(result2)

        self.print_results()

        return {
            "scenario": self.name,
            "counterparty_created": True,
            "csa_created": True,
            "initial_trades": len(self.results),
        }


class PortfolioRebalancingScenario(TradingScenario):
    """Simulate portfolio rebalancing with terminations and new trades."""

    def __init__(self):
        super().__init__(
            name="Portfolio Rebalancing",
            description="Terminate old positions and establish new ones",
        )

    async def execute(self, bank: FictionalBank) -> Dict:
        """Execute rebalancing scenario."""
        print(f"\n--- Starting {self.name} ---\n")

        self.results = []
        terminated_count = 0

        # Step 1: Identify trades to terminate (oldest trades in portfolio)
        print("🔄 Step 1: Terminating old positions")
        all_trades = list(bank.portfolio.trades.values())
        trades_to_terminate = sorted(all_trades, key=lambda t: t.trade_date)[:3]

        for trade in trades_to_terminate:
            if trade.status == TradeStatus.ACTIVE:
                result = await bank.execution_service.terminate_trade(
                    trade.id, date.today()
                )
                if result.is_success():
                    terminated_count += 1
                    print(f"  ✓ Terminated: {trade.id}")

        # Step 2: Book replacement trades
        print(f"\n💼 Step 2: Booking replacement positions")

        for i in range(3):
            counterparty_id = random.choice(["CP_BANK_AAA", "CP_CORP_A", "CP_SOVEREIGN"])
            product_types = [
                ProductType.INTEREST_RATE_SWAP,
                ProductType.FX_OPTION,
                ProductType.EQUITY_OPTION,
            ]
            product_type = random.choice(product_types)

            result = await bank.book_trade(
                counterparty_id=counterparty_id,
                product_type=product_type,
                trade_date=date.today(),
                notional=float(random.randint(10, 40) * 1_000_000),
                currency="USD",
                maturity_date=date.today() + timedelta(days=365 * random.randint(1, 7)),
                validate_counterparty=True,
                auto_confirm=True,
            )
            self.results.append(result)

        self.print_results()

        return {
            "scenario": self.name,
            "terminated": terminated_count,
            "new_trades": len(self.results),
        }


class StressTestScenario(TradingScenario):
    """Simulate high-volume trading under stress."""

    def __init__(self, num_trades: int = 20):
        super().__init__(
            name=f"Stress Test ({num_trades} trades)",
            description="High-volume trading to test system performance",
        )
        self.num_trades = num_trades

    async def execute(self, bank: FictionalBank) -> Dict:
        """Execute stress test scenario."""
        print(f"\n--- Starting {self.name} ---\n")

        import time

        start_time = time.time()
        self.results = []

        counterparties = list(bank.portfolio.counterparties.keys())
        product_types = [
            ProductType.INTEREST_RATE_SWAP,
            ProductType.FX_OPTION,
            ProductType.EQUITY_OPTION,
            ProductType.SWAPTION,
        ]
        books = ["BOOK_IRS_USD", "BOOK_FX_MAJORS", "BOOK_EQ_VANILLA"]

        print(f"📊 Executing {self.num_trades} trades...")

        for i in range(self.num_trades):
            result = await bank.book_trade(
                counterparty_id=random.choice(counterparties),
                product_type=random.choice(product_types),
                trade_date=date.today(),
                notional=float(random.randint(5, 50) * 1_000_000),
                currency=random.choice(["USD", "EUR", "GBP"]),
                maturity_date=date.today()
                + timedelta(days=random.randint(90, 3650)),
                book_id=random.choice(books),
                validate_counterparty=True,
                auto_confirm=True,
            )
            self.results.append(result)

            if (i + 1) % 5 == 0:
                print(f"  Progress: {i+1}/{self.num_trades} trades executed")

        elapsed_time = time.time() - start_time

        print(f"\n⏱️  Execution time: {elapsed_time:.2f} seconds")
        print(f"  Trades per second: {self.num_trades/elapsed_time:.2f}")

        self.print_results()

        return {
            "scenario": self.name,
            "total_trades": self.num_trades,
            "elapsed_seconds": elapsed_time,
            "trades_per_second": self.num_trades / elapsed_time,
            "success_rate": sum(1 for r in self.results if r.is_success())
            / len(self.results),
        }


class ExposureMonitoringScenario(TradingScenario):
    """Monitor and report on counterparty exposures."""

    def __init__(self):
        super().__init__(
            name="Exposure Monitoring",
            description="Calculate and monitor counterparty exposures",
        )

    async def execute(self, bank: FictionalBank) -> Dict:
        """Execute exposure monitoring scenario."""
        print(f"\n--- Starting {self.name} ---\n")

        exposures = {}

        print("📊 Counterparty Exposures:\n")

        for cp_id, counterparty in bank.portfolio.counterparties.items():
            exposure = await bank.get_counterparty_exposure(cp_id)

            exposures[cp_id] = {
                "name": counterparty.name,
                "rating": (
                    counterparty.credit.rating.value
                    if counterparty.credit and counterparty.credit.rating
                    else "NR"
                ),
                "active_trades": exposure["trade_count"],
                "total_mtm": exposure["total_mtm"],
                "has_csa": len(exposure["csas"]) > 0,
            }

            print(f"  {counterparty.name} ({cp_id}):")
            print(f"    Rating: {exposures[cp_id]['rating']}")
            print(f"    Active trades: {exposures[cp_id]['active_trades']}")
            print(f"    Total MTM: ${exposures[cp_id]['total_mtm']:,.2f}")
            print(f"    CSA: {'Yes' if exposures[cp_id]['has_csa'] else 'No'}")
            print()

        # Calculate aggregate statistics
        total_exposure = sum(exp["total_mtm"] for exp in exposures.values())
        collateralized_exposure = sum(
            exp["total_mtm"] for exp in exposures.values() if exp["has_csa"]
        )

        print(f"📈 Summary:")
        print(f"  Total exposure: ${total_exposure:,.2f}")
        print(
            f"  Collateralized: ${collateralized_exposure:,.2f} ({collateralized_exposure/total_exposure*100:.1f}%)"
        )

        return {
            "scenario": self.name,
            "counterparties": exposures,
            "total_exposure": total_exposure,
            "collateralized_exposure": collateralized_exposure,
        }


async def run_all_scenarios(bank: FictionalBank) -> Dict:
    """Run all trading scenarios.

    Parameters
    ----------
    bank : FictionalBank
        Bank to run scenarios on

    Returns
    -------
    dict
        Results from all scenarios
    """
    results = {}

    scenarios = [
        DailyTradingScenario(),
        CounterpartyOnboardingScenario(),
        PortfolioRebalancingScenario(),
        ExposureMonitoringScenario(),
        StressTestScenario(num_trades=10),
    ]

    for scenario in scenarios:
        result = await scenario.execute(bank)
        results[scenario.name] = result

    return results


__all__ = [
    "TradingScenario",
    "DailyTradingScenario",
    "CounterpartyOnboardingScenario",
    "PortfolioRebalancingScenario",
    "StressTestScenario",
    "ExposureMonitoringScenario",
    "run_all_scenarios",
]
