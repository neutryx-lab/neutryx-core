"""Fictional portfolio with complex book hierarchy and multiple counterparties.

This module provides a comprehensive test portfolio with:
- Multiple desks: Rates, FX, Equity
- Multiple traders and books per desk
- Diverse counterparty types with credit ratings
- Various trade types (FpML-compatible)
- Complete netting set structure
"""
from datetime import date, timedelta
from typing import Dict, List

from neutryx.portfolio.contracts.counterparty import (
    Counterparty,
    CounterpartyCredit,
    CreditRating,
    EntityType,
)
from neutryx.portfolio.contracts.csa import (
    CSA,
    CollateralTerms,
    CollateralType,
    EligibleCollateral,
    ThresholdTerms,
)
from neutryx.portfolio.contracts.master_agreement import AgreementType, MasterAgreement
from neutryx.portfolio.contracts.trade import ProductType, SettlementType, Trade, TradeStatus
from neutryx.portfolio.books import (
    Book,
    BookHierarchy,
    BusinessUnit,
    Desk,
    EntityStatus,
    LegalEntity,
    Trader,
)
from neutryx.portfolio.netting_set import NettingSet
from neutryx.portfolio.portfolio import Portfolio


def create_fictional_portfolio() -> tuple[Portfolio, BookHierarchy]:
    """Create a comprehensive fictional portfolio for testing.

    Returns
    -------
    tuple[Portfolio, BookHierarchy]
        Complete portfolio with all hierarchies and book structure
    """
    portfolio = Portfolio(name="Global Trading Portfolio", base_currency="USD")
    book_hierarchy = BookHierarchy()

    # Create organizational hierarchy
    _setup_book_hierarchy(book_hierarchy)

    # Create counterparties with diverse characteristics
    counterparties = _create_counterparties()
    for cp in counterparties.values():
        portfolio.add_counterparty(cp)

    # Create master agreements and CSAs
    agreements = _create_master_agreements(counterparties)
    for agreement in agreements.values():
        portfolio.add_master_agreement(agreement)

    csas = _create_csas(counterparties)
    for csa in csas.values():
        portfolio.add_csa(csa)

    # Create netting sets
    netting_sets = _create_netting_sets(counterparties, agreements, csas)
    for ns in netting_sets.values():
        portfolio.add_netting_set(ns)

    # Create trades across all desks and books
    trades = _create_trades(counterparties, netting_sets, book_hierarchy)
    for trade in trades:
        portfolio.add_trade(trade)

    return portfolio, book_hierarchy


def _setup_book_hierarchy(hierarchy: BookHierarchy) -> None:
    """Setup complete book hierarchy structure."""
    # Legal Entity
    legal_entity = LegalEntity(
        id="LE001",
        name="Global Investment Bank Ltd",
        lei="529900T8BM49AURSDO55",
        jurisdiction="GB",
        status=EntityStatus.ACTIVE,
    )
    hierarchy.add_legal_entity(legal_entity)

    # Business Unit
    bu = BusinessUnit(
        id="BU_TRADING",
        name="Global Trading",
        legal_entity_id="LE001",
        status=EntityStatus.ACTIVE,
    )
    hierarchy.add_business_unit(bu)

    # Desks
    desks = [
        Desk(
            id="DESK_RATES",
            name="Interest Rates Desk",
            business_unit_id="BU_TRADING",
            desk_type="rates",
            status=EntityStatus.ACTIVE,
        ),
        Desk(
            id="DESK_FX",
            name="Foreign Exchange Desk",
            business_unit_id="BU_TRADING",
            desk_type="fx",
            status=EntityStatus.ACTIVE,
        ),
        Desk(
            id="DESK_EQUITY",
            name="Equity Derivatives Desk",
            business_unit_id="BU_TRADING",
            desk_type="equity",
            status=EntityStatus.ACTIVE,
        ),
    ]
    for desk in desks:
        hierarchy.add_desk(desk)

    # Traders
    traders = [
        Trader(
            id="TRADER_001",
            name="Alice Chen",
            email="alice.chen@example.com",
            desk_id="DESK_RATES",
            status=EntityStatus.ACTIVE,
            hire_date=date(2018, 3, 15),
        ),
        Trader(
            id="TRADER_002",
            name="Bob Martinez",
            email="bob.martinez@example.com",
            desk_id="DESK_RATES",
            status=EntityStatus.ACTIVE,
            hire_date=date(2019, 7, 1),
        ),
        Trader(
            id="TRADER_003",
            name="Carol Zhang",
            email="carol.zhang@example.com",
            desk_id="DESK_FX",
            status=EntityStatus.ACTIVE,
            hire_date=date(2017, 1, 10),
        ),
        Trader(
            id="TRADER_004",
            name="David Kim",
            email="david.kim@example.com",
            desk_id="DESK_FX",
            status=EntityStatus.ACTIVE,
            hire_date=date(2020, 5, 20),
        ),
        Trader(
            id="TRADER_005",
            name="Emma Wilson",
            email="emma.wilson@example.com",
            desk_id="DESK_EQUITY",
            status=EntityStatus.ACTIVE,
            hire_date=date(2016, 9, 12),
        ),
        Trader(
            id="TRADER_006",
            name="Frank Johnson",
            email="frank.johnson@example.com",
            desk_id="DESK_EQUITY",
            status=EntityStatus.ACTIVE,
            hire_date=date(2021, 2, 1),
        ),
    ]
    for trader in traders:
        hierarchy.add_trader(trader)

    # Books
    books = [
        # Rates desk books
        Book(
            id="BOOK_IRS_USD",
            name="USD Interest Rate Swaps",
            desk_id="DESK_RATES",
            book_type="flow",
            status=EntityStatus.ACTIVE,
            primary_trader_id="TRADER_001",
            created_date=date(2020, 1, 1),
        ),
        Book(
            id="BOOK_IRS_EUR",
            name="EUR Interest Rate Swaps",
            desk_id="DESK_RATES",
            book_type="flow",
            status=EntityStatus.ACTIVE,
            primary_trader_id="TRADER_002",
            created_date=date(2020, 1, 1),
        ),
        Book(
            id="BOOK_SWAPTIONS",
            name="Swaptions Book",
            desk_id="DESK_RATES",
            book_type="proprietary",
            status=EntityStatus.ACTIVE,
            primary_trader_id="TRADER_001",
            created_date=date(2021, 6, 15),
        ),
        # FX desk books
        Book(
            id="BOOK_FX_MAJORS",
            name="FX Majors Options",
            desk_id="DESK_FX",
            book_type="flow",
            status=EntityStatus.ACTIVE,
            primary_trader_id="TRADER_003",
            created_date=date(2020, 1, 1),
        ),
        Book(
            id="BOOK_FX_EMERGING",
            name="FX Emerging Markets",
            desk_id="DESK_FX",
            book_type="proprietary",
            status=EntityStatus.ACTIVE,
            primary_trader_id="TRADER_004",
            created_date=date(2020, 6, 1),
        ),
        # Equity desk books
        Book(
            id="BOOK_EQ_VANILLA",
            name="Equity Vanilla Options",
            desk_id="DESK_EQUITY",
            book_type="flow",
            status=EntityStatus.ACTIVE,
            primary_trader_id="TRADER_005",
            created_date=date(2020, 1, 1),
        ),
        Book(
            id="BOOK_EQ_EXOTIC",
            name="Equity Exotic Options",
            desk_id="DESK_EQUITY",
            book_type="proprietary",
            status=EntityStatus.ACTIVE,
            primary_trader_id="TRADER_006",
            created_date=date(2021, 3, 1),
        ),
    ]
    for book in books:
        hierarchy.add_book(book)


def _create_counterparties() -> Dict[str, Counterparty]:
    """Create diverse set of counterparties."""
    counterparties = {
        "CP_BANK_AAA": Counterparty(
            id="CP_BANK_AAA",
            name="AAA Global Bank",
            entity_type=EntityType.FINANCIAL,
            lei="213800WAVVOPS85N2205",
            jurisdiction="US",
            is_bank=True,
            credit=CounterpartyCredit(
                rating=CreditRating.AAA,
                lgd=0.4,
                credit_spread_bps=15.0,
            ),
        ),
        "CP_CORP_A": Counterparty(
            id="CP_CORP_A",
            name="Tech Corporation A",
            entity_type=EntityType.CORPORATE,
            lei="549300VZKC1YSLMVXU38",
            jurisdiction="US",
            credit=CounterpartyCredit(
                rating=CreditRating.A,
                lgd=0.6,
                credit_spread_bps=75.0,
            ),
        ),
        "CP_CORP_BBB": Counterparty(
            id="CP_CORP_BBB",
            name="Industrial Group BBB",
            entity_type=EntityType.CORPORATE,
            lei="549300LLCY4K6URF6Z41",
            jurisdiction="DE",
            credit=CounterpartyCredit(
                rating=CreditRating.BBB,
                lgd=0.65,
                credit_spread_bps=125.0,
            ),
        ),
        "CP_HEDGE_FUND": Counterparty(
            id="CP_HEDGE_FUND",
            name="Alpha Strategies Fund",
            entity_type=EntityType.FUND,
            lei="5493001KJTIIGC8Y1R12",
            jurisdiction="KY",
            credit=CounterpartyCredit(
                rating=CreditRating.A_MINUS,
                lgd=0.7,
                credit_spread_bps=150.0,
            ),
        ),
        "CP_SOVEREIGN": Counterparty(
            id="CP_SOVEREIGN",
            name="Republic Investment Authority",
            entity_type=EntityType.SOVEREIGN,
            jurisdiction="SG",
            credit=CounterpartyCredit(
                rating=CreditRating.AA_PLUS,
                lgd=0.5,
                credit_spread_bps=50.0,
            ),
        ),
        "CP_INSURANCE": Counterparty(
            id="CP_INSURANCE",
            name="Global Insurance Group",
            entity_type=EntityType.FINANCIAL,
            lei="549300QYCBDK3RUHVD89",
            jurisdiction="GB",
            credit=CounterpartyCredit(
                rating=CreditRating.AA,
                lgd=0.45,
                credit_spread_bps=45.0,
            ),
        ),
    }
    return counterparties


def _create_master_agreements(
    counterparties: Dict[str, Counterparty]
) -> Dict[str, MasterAgreement]:
    """Create master agreements with counterparties."""
    from neutryx.portfolio.contracts.master_agreement import GoverningLaw

    base_date = date(2020, 1, 1)
    agreements = {}

    for i, (cp_id, cp) in enumerate(counterparties.items(), 1):
        # Determine governing law based on jurisdiction
        if cp.jurisdiction in ["GB", "UK"]:
            gov_law = GoverningLaw.ENGLISH
        elif cp.jurisdiction in ["US", "USA"]:
            gov_law = GoverningLaw.NEW_YORK
        elif cp.jurisdiction == "DE":
            gov_law = GoverningLaw.GERMAN
        else:
            gov_law = GoverningLaw.ENGLISH

        agreement = MasterAgreement(
            id=f"MA_{cp_id}",
            agreement_type=AgreementType.ISDA_2002,
            party_a_id="OUR_INSTITUTION",  # Our institution
            party_b_id=cp_id,  # Counterparty
            effective_date=base_date,
            governing_law=gov_law,
        )
        agreements[agreement.id] = agreement

    return agreements


def _create_csas(counterparties: Dict[str, Counterparty]) -> Dict[str, CSA]:
    """Create CSA agreements for some counterparties."""
    base_date = date(2020, 1, 1)
    csas = {}

    # Only create CSAs for high-quality counterparties
    # Note: In this simplified model, we use "party_a" as our institution
    # and "party_b" as the counterparty
    csa_counterparties = ["CP_BANK_AAA", "CP_CORP_A", "CP_SOVEREIGN", "CP_INSURANCE"]

    for cp_id in csa_counterparties:
        # Define eligible collateral
        eligible_collateral = [
            EligibleCollateral(
                collateral_type=CollateralType.CASH,
                currency="USD",
                haircut=0.0,
            ),
            EligibleCollateral(
                collateral_type=CollateralType.GOVERNMENT_BOND,
                haircut=0.02,  # 2% haircut
                rating_threshold="AA-",
                maturity_max_years=10.0,
            ),
        ]

        # Define threshold terms (bilateral)
        threshold_terms = ThresholdTerms(
            threshold_party_a=1_000_000.0,
            threshold_party_b=1_000_000.0,
            mta_party_a=100_000.0,
            mta_party_b=100_000.0,
            independent_amount_party_a=0.0,
            independent_amount_party_b=0.0,
            rounding=100_000.0,
        )

        # Define collateral terms
        collateral_terms = CollateralTerms(
            base_currency="USD",
            eligible_collateral=eligible_collateral,
        )

        csa = CSA(
            id=f"CSA_{cp_id}",
            party_a_id="OUR_INSTITUTION",  # Our institution
            party_b_id=cp_id,  # Counterparty
            effective_date=base_date.isoformat(),
            threshold_terms=threshold_terms,
            collateral_terms=collateral_terms,
            initial_margin_required=False,
            variation_margin_required=True,
        )
        csas[csa.id] = csa

    return csas


def _create_netting_sets(
    counterparties: Dict[str, Counterparty],
    agreements: Dict[str, MasterAgreement],
    csas: Dict[str, CSA],
) -> Dict[str, NettingSet]:
    """Create netting sets linking agreements and CSAs."""
    netting_sets = {}

    for cp_id in counterparties.keys():
        agreement_id = f"MA_{cp_id}"
        csa_id = f"CSA_{cp_id}"

        # Determine if this counterparty has CSA
        has_csa = csa_id in csas

        netting_set = NettingSet(
            id=f"NS_{cp_id}",
            counterparty_id=cp_id,
            master_agreement_id=agreement_id,
            csa_id=csa_id if has_csa else None,
        )
        netting_sets[netting_set.id] = netting_set

    return netting_sets


def _create_trades(
    counterparties: Dict[str, Counterparty],
    netting_sets: Dict[str, NettingSet],
    book_hierarchy: BookHierarchy,
) -> List[Trade]:
    """Create diverse trades across all books."""
    trades = []
    trade_date = date(2024, 1, 15)
    trade_counter = 1

    # Helper to create trade IDs
    def trade_id():
        nonlocal trade_counter
        tid = f"TRD_{trade_counter:05d}"
        trade_counter += 1
        return tid

    # -------------------------------------------------------------------------
    # Rates Desk Trades
    # -------------------------------------------------------------------------

    # USD IRS trades with CP_BANK_AAA
    trades.extend(
        [
            Trade(
                id=trade_id(),
                trade_number="IRS-USD-001",
                counterparty_id="CP_BANK_AAA",
                netting_set_id="NS_CP_BANK_AAA",
                book_id="BOOK_IRS_USD",
                desk_id="DESK_RATES",
                trader_id="TRADER_001",
                product_type=ProductType.INTEREST_RATE_SWAP,
                trade_date=trade_date,
                effective_date=trade_date + timedelta(days=2),
                maturity_date=trade_date + timedelta(days=365 * 5),
                status=TradeStatus.ACTIVE,
                notional=10_000_000.0,
                currency="USD",
                settlement_type=SettlementType.CASH,
                mtm=125_000.0,
                last_valuation_date=trade_date,
                product_details={
                    "fixed_rate": 0.045,
                    "floating_rate": "USD-LIBOR-3M",
                    "payment_frequency": "quarterly",
                    "direction": "receive_fixed",
                },
            ),
            Trade(
                id=trade_id(),
                trade_number="IRS-USD-002",
                counterparty_id="CP_BANK_AAA",
                netting_set_id="NS_CP_BANK_AAA",
                book_id="BOOK_IRS_USD",
                desk_id="DESK_RATES",
                trader_id="TRADER_001",
                product_type=ProductType.INTEREST_RATE_SWAP,
                trade_date=trade_date - timedelta(days=30),
                effective_date=trade_date - timedelta(days=28),
                maturity_date=trade_date + timedelta(days=365 * 10),
                status=TradeStatus.ACTIVE,
                notional=25_000_000.0,
                currency="USD",
                settlement_type=SettlementType.CASH,
                mtm=-85_000.0,
                last_valuation_date=trade_date,
                product_details={
                    "fixed_rate": 0.038,
                    "floating_rate": "USD-SOFR",
                    "payment_frequency": "semiannual",
                    "direction": "pay_fixed",
                },
            ),
        ]
    )

    # EUR IRS trades with CP_CORP_BBB
    trades.extend(
        [
            Trade(
                id=trade_id(),
                trade_number="IRS-EUR-001",
                counterparty_id="CP_CORP_BBB",
                netting_set_id="NS_CP_CORP_BBB",
                book_id="BOOK_IRS_EUR",
                desk_id="DESK_RATES",
                trader_id="TRADER_002",
                product_type=ProductType.INTEREST_RATE_SWAP,
                trade_date=trade_date - timedelta(days=60),
                effective_date=trade_date - timedelta(days=58),
                maturity_date=trade_date + timedelta(days=365 * 7),
                status=TradeStatus.ACTIVE,
                notional=15_000_000.0,
                currency="EUR",
                settlement_type=SettlementType.CASH,
                mtm=235_000.0,
                last_valuation_date=trade_date,
                product_details={
                    "fixed_rate": 0.025,
                    "floating_rate": "EUR-EURIBOR-6M",
                    "payment_frequency": "semiannual",
                    "direction": "receive_fixed",
                },
            ),
        ]
    )

    # Swaptions with CP_INSURANCE
    trades.extend(
        [
            Trade(
                id=trade_id(),
                trade_number="SWPN-001",
                counterparty_id="CP_INSURANCE",
                netting_set_id="NS_CP_INSURANCE",
                book_id="BOOK_SWAPTIONS",
                desk_id="DESK_RATES",
                trader_id="TRADER_001",
                product_type=ProductType.SWAPTION,
                trade_date=trade_date - timedelta(days=15),
                effective_date=trade_date - timedelta(days=13),
                maturity_date=trade_date + timedelta(days=180),
                status=TradeStatus.ACTIVE,
                notional=50_000_000.0,
                currency="USD",
                settlement_type=SettlementType.CASH,
                mtm=450_000.0,
                last_valuation_date=trade_date,
                product_details={
                    "option_type": "payer",
                    "strike_rate": 0.04,
                    "underlying_tenor": "10Y",
                    "exercise_style": "european",
                },
            ),
        ]
    )

    # -------------------------------------------------------------------------
    # FX Desk Trades
    # -------------------------------------------------------------------------

    # FX Options with CP_CORP_A
    trades.extend(
        [
            Trade(
                id=trade_id(),
                trade_number="FXO-EURUSD-001",
                counterparty_id="CP_CORP_A",
                netting_set_id="NS_CP_CORP_A",
                book_id="BOOK_FX_MAJORS",
                desk_id="DESK_FX",
                trader_id="TRADER_003",
                product_type=ProductType.FX_OPTION,
                trade_date=trade_date,
                effective_date=trade_date,
                maturity_date=trade_date + timedelta(days=90),
                status=TradeStatus.ACTIVE,
                notional=5_000_000.0,
                currency="EUR",
                settlement_type=SettlementType.CASH,
                mtm=75_000.0,
                last_valuation_date=trade_date,
                product_details={
                    "currency_pair": "EUR/USD",
                    "call_currency": "EUR",
                    "put_currency": "USD",
                    "strike": 1.085,
                    "option_type": "call",
                    "exercise_style": "european",
                },
            ),
            Trade(
                id=trade_id(),
                trade_number="FXO-USDJPY-001",
                counterparty_id="CP_CORP_A",
                netting_set_id="NS_CP_CORP_A",
                book_id="BOOK_FX_MAJORS",
                desk_id="DESK_FX",
                trader_id="TRADER_003",
                product_type=ProductType.FX_OPTION,
                trade_date=trade_date - timedelta(days=45),
                effective_date=trade_date - timedelta(days=45),
                maturity_date=trade_date + timedelta(days=180),
                status=TradeStatus.ACTIVE,
                notional=8_000_000.0,
                currency="USD",
                settlement_type=SettlementType.CASH,
                mtm=-125_000.0,
                last_valuation_date=trade_date,
                product_details={
                    "currency_pair": "USD/JPY",
                    "call_currency": "USD",
                    "put_currency": "JPY",
                    "strike": 148.5,
                    "option_type": "put",
                    "exercise_style": "american",
                },
            ),
        ]
    )

    # FX Emerging with CP_HEDGE_FUND
    trades.extend(
        [
            Trade(
                id=trade_id(),
                trade_number="FXO-USDBRL-001",
                counterparty_id="CP_HEDGE_FUND",
                netting_set_id="NS_CP_HEDGE_FUND",
                book_id="BOOK_FX_EMERGING",
                desk_id="DESK_FX",
                trader_id="TRADER_004",
                product_type=ProductType.FX_OPTION,
                trade_date=trade_date - timedelta(days=20),
                effective_date=trade_date - timedelta(days=20),
                maturity_date=trade_date + timedelta(days=120),
                status=TradeStatus.ACTIVE,
                notional=3_000_000.0,
                currency="USD",
                settlement_type=SettlementType.CASH,
                mtm=180_000.0,
                last_valuation_date=trade_date,
                product_details={
                    "currency_pair": "USD/BRL",
                    "call_currency": "USD",
                    "put_currency": "BRL",
                    "strike": 5.15,
                    "option_type": "call",
                    "exercise_style": "european",
                },
            ),
        ]
    )

    # -------------------------------------------------------------------------
    # Equity Desk Trades
    # -------------------------------------------------------------------------

    # Vanilla equity options with CP_SOVEREIGN
    trades.extend(
        [
            Trade(
                id=trade_id(),
                trade_number="EQO-SPX-001",
                counterparty_id="CP_SOVEREIGN",
                netting_set_id="NS_CP_SOVEREIGN",
                book_id="BOOK_EQ_VANILLA",
                desk_id="DESK_EQUITY",
                trader_id="TRADER_005",
                product_type=ProductType.EQUITY_OPTION,
                trade_date=trade_date,
                effective_date=trade_date,
                maturity_date=trade_date + timedelta(days=365),
                status=TradeStatus.ACTIVE,
                notional=20_000_000.0,
                currency="USD",
                settlement_type=SettlementType.CASH,
                mtm=950_000.0,
                last_valuation_date=trade_date,
                product_details={
                    "underlyer": "SPX",
                    "underlyer_name": "S&P 500 Index",
                    "option_type": "call",
                    "strike": 4500.0,
                    "exercise_style": "european",
                    "number_of_options": 100,
                },
            ),
            Trade(
                id=trade_id(),
                trade_number="EQO-AAPL-001",
                counterparty_id="CP_SOVEREIGN",
                netting_set_id="NS_CP_SOVEREIGN",
                book_id="BOOK_EQ_VANILLA",
                desk_id="DESK_EQUITY",
                trader_id="TRADER_005",
                product_type=ProductType.EQUITY_OPTION,
                trade_date=trade_date - timedelta(days=10),
                effective_date=trade_date - timedelta(days=10),
                maturity_date=trade_date + timedelta(days=180),
                status=TradeStatus.ACTIVE,
                notional=5_000_000.0,
                currency="USD",
                settlement_type=SettlementType.PHYSICAL,
                mtm=250_000.0,
                last_valuation_date=trade_date,
                product_details={
                    "underlyer": "AAPL",
                    "underlyer_name": "Apple Inc.",
                    "option_type": "call",
                    "strike": 180.0,
                    "exercise_style": "american",
                    "number_of_options": 10000,
                },
            ),
        ]
    )

    # Exotic equity options with CP_HEDGE_FUND
    trades.extend(
        [
            Trade(
                id=trade_id(),
                trade_number="EQO-EXOTIC-001",
                counterparty_id="CP_HEDGE_FUND",
                netting_set_id="NS_CP_HEDGE_FUND",
                book_id="BOOK_EQ_EXOTIC",
                desk_id="DESK_EQUITY",
                trader_id="TRADER_006",
                product_type=ProductType.EQUITY_OPTION,
                trade_date=trade_date - timedelta(days=30),
                effective_date=trade_date - timedelta(days=30),
                maturity_date=trade_date + timedelta(days=270),
                status=TradeStatus.ACTIVE,
                notional=10_000_000.0,
                currency="USD",
                settlement_type=SettlementType.CASH,
                mtm=625_000.0,
                last_valuation_date=trade_date,
                product_details={
                    "underlyer": "TSLA",
                    "underlyer_name": "Tesla Inc.",
                    "product_subtype": "barrier_option",
                    "option_type": "call",
                    "strike": 250.0,
                    "barrier_type": "knock_out",
                    "barrier_level": 300.0,
                    "exercise_style": "european",
                },
            ),
        ]
    )

    # Add some variance swaps with CP_INSURANCE
    trades.extend(
        [
            Trade(
                id=trade_id(),
                trade_number="VARSWAP-001",
                counterparty_id="CP_INSURANCE",
                netting_set_id="NS_CP_INSURANCE",
                book_id="BOOK_EQ_EXOTIC",
                desk_id="DESK_EQUITY",
                trader_id="TRADER_006",
                product_type=ProductType.VARIANCE_SWAP,
                trade_date=trade_date - timedelta(days=90),
                effective_date=trade_date - timedelta(days=90),
                maturity_date=trade_date + timedelta(days=275),
                status=TradeStatus.ACTIVE,
                notional=1_000_000.0,
                currency="USD",
                settlement_type=SettlementType.CASH,
                mtm=-150_000.0,
                last_valuation_date=trade_date,
                product_details={
                    "underlyer": "SPX",
                    "strike_variance": 0.0400,
                    "vega_notional": 1_000_000.0,
                    "observation_frequency": "daily",
                },
            ),
        ]
    )

    return trades


def get_portfolio_summary(
    portfolio: Portfolio, book_hierarchy: BookHierarchy
) -> Dict[str, any]:
    """Get comprehensive portfolio summary.

    Returns
    -------
    dict
        Summary statistics for the entire portfolio
    """
    summary = {
        "portfolio_name": portfolio.name,
        "base_currency": portfolio.base_currency,
        "statistics": portfolio.summary(),
        "total_mtm": portfolio.calculate_total_mtm(),
        "gross_notional": portfolio.calculate_gross_notional(),
        "counterparties": {},
        "books": {},
        "desks": {},
    }

    # Counterparty breakdown
    for cp_id, cp in portfolio.counterparties.items():
        cp_trades = portfolio.get_trades_by_counterparty(cp_id)
        summary["counterparties"][cp_id] = {
            "name": cp.name,
            "entity_type": cp.entity_type.value,
            "rating": cp.credit.rating.value if cp.credit and cp.credit.rating else "NR",
            "num_trades": len(cp_trades),
            "net_mtm": portfolio.calculate_net_mtm_by_counterparty(cp_id),
            "has_csa": any(ns.has_csa() for ns in portfolio.get_netting_sets_by_counterparty(cp_id)),
        }

    # Book breakdown
    book_ids = set(t.book_id for t in portfolio.trades.values() if t.book_id)
    for book_id in book_ids:
        book = book_hierarchy.books.get(book_id)
        if book:
            summary["books"][book_id] = {
                "name": book.name,
                "desk": book.desk_id,
                **portfolio.get_book_summary(book_id),
            }

    # Desk breakdown
    desk_ids = set(t.desk_id for t in portfolio.trades.values() if t.desk_id)
    for desk_id in desk_ids:
        desk = book_hierarchy.desks.get(desk_id)
        if desk:
            summary["desks"][desk_id] = {
                "name": desk.name,
                **portfolio.get_desk_summary(desk_id),
            }

    return summary


__all__ = [
    "create_fictional_portfolio",
    "get_portfolio_summary",
]
