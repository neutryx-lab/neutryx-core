"""Tests for UMR compliance module."""

from datetime import date, timedelta

import pytest

from neutryx.valuations.margin.umr_compliance import (
    AANACalculation,
    CollateralMovement,
    CollateralType,
    CSAManager,
    CSAPortfolio,
    CSATerms,
    CustodianAccount,
    CustodianInterface,
    MarginCall,
    MarginType,
    UMRComplianceChecker,
    UMRPhase,
    UMRThresholds,
    generate_margin_report,
)


class TestUMRPhase:
    """Tests for UMR phase-in schedule."""

    def test_phase_enum_values(self):
        """Test UMR phase enum values."""
        assert UMRPhase.PHASE_I.value == 1
        assert UMRPhase.PHASE_II.value == 2
        assert UMRPhase.PHASE_VI.value == 6

    def test_phase_names(self):
        """Test UMR phase names."""
        assert UMRPhase.PHASE_I.name == "PHASE_I"
        assert UMRPhase.PHASE_V.name == "PHASE_V"


class TestCollateralType:
    """Tests for collateral type enum."""

    def test_collateral_types(self):
        """Test collateral type values."""
        assert CollateralType.CASH.value == "cash"
        assert CollateralType.GOVERNMENT_BONDS.value == "govt_bonds"
        assert CollateralType.CORPORATE_BONDS.value == "corp_bonds"
        assert CollateralType.EQUITY.value == "equity"
        assert CollateralType.GOLD.value == "gold"


class TestAANACalculation:
    """Tests for AANA calculation."""

    def test_aana_creation(self):
        """Test AANA calculation creation."""
        aana = AANACalculation(
            march_notional=100e9,
            april_notional=105e9,
            may_notional=95e9,
            aana=100e9,
            calculation_date=date(2024, 6, 1),
            applicable_phase=UMRPhase.PHASE_V,
        )

        assert aana.aana == 100e9
        assert aana.applicable_phase == UMRPhase.PHASE_V
        assert aana.is_subject_to_umr()

    def test_aana_not_subject_to_umr(self):
        """Test AANA below threshold."""
        aana = AANACalculation(
            march_notional=5e9,
            april_notional=6e9,
            may_notional=4e9,
            aana=5e9,
            calculation_date=date(2024, 6, 1),
            applicable_phase=None,
        )

        assert not aana.is_subject_to_umr()


class TestMarginCall:
    """Tests for margin call dataclass."""

    def test_margin_call_creation(self):
        """Test margin call creation."""
        call = MarginCall(
            margin_type=MarginType.INITIAL_MARGIN,
            amount=10_000_000,
            currency="USD",
            due_date=date(2024, 11, 6),
            counterparty="DEALER-A",
            portfolio="PORT-001",
            calculation_date=date(2024, 11, 5),
            outstanding_amount=0.0,
        )

        assert call.amount == 10_000_000
        assert call.margin_type == MarginType.INITIAL_MARGIN
        assert call.net_call == 10_000_000

    def test_margin_call_is_posting(self):
        """Test posting vs collection detection."""
        posting_call = MarginCall(
            margin_type=MarginType.INITIAL_MARGIN,
            amount=10_000_000,
            currency="USD",
            due_date=date(2024, 11, 6),
            counterparty="DEALER-A",
            portfolio="PORT-001",
            calculation_date=date(2024, 11, 5),
            outstanding_amount=5_000_000,
        )

        assert posting_call.is_posting
        assert not posting_call.is_collection
        assert posting_call.net_call == 5_000_000

    def test_margin_call_is_collection(self):
        """Test collection scenario."""
        collection_call = MarginCall(
            margin_type=MarginType.VARIATION_MARGIN,
            amount=5_000_000,
            currency="USD",
            due_date=date(2024, 11, 5),
            counterparty="DEALER-A",
            portfolio="PORT-001",
            calculation_date=date(2024, 11, 5),
            outstanding_amount=10_000_000,
        )

        assert collection_call.is_collection
        assert not collection_call.is_posting
        assert collection_call.net_call == -5_000_000


class TestUMRThresholds:
    """Tests for UMR threshold configuration."""

    def test_default_thresholds(self):
        """Test default UMR thresholds."""
        thresholds = UMRThresholds()

        assert thresholds.im_threshold == 50_000_000
        assert thresholds.mta == 500_000
        assert thresholds.independent_amount == 0.0
        assert CollateralType.CASH in thresholds.eligible_collateral
        assert thresholds.haircuts[CollateralType.CASH] == 0.0
        assert thresholds.haircuts[CollateralType.GOVERNMENT_BONDS] == 0.01
        assert thresholds.haircuts[CollateralType.CORPORATE_BONDS] == 0.04

    def test_custom_thresholds(self):
        """Test custom threshold configuration."""
        custom_haircuts = {
            CollateralType.CASH: 0.0,
            CollateralType.GOVERNMENT_BONDS: 0.02,  # Higher haircut
        }

        thresholds = UMRThresholds(
            im_threshold=100_000_000,
            mta=1_000_000,
            haircuts=custom_haircuts,
        )

        assert thresholds.im_threshold == 100_000_000
        assert thresholds.mta == 1_000_000
        assert thresholds.haircuts[CollateralType.GOVERNMENT_BONDS] == 0.02


class TestUMRComplianceChecker:
    """Tests for UMR compliance checker."""

    def test_aana_calculation_phase_v(self):
        """Test AANA calculation for Phase V."""
        checker = UMRComplianceChecker()
        aana = checker.calculate_aana(
            march_notional=95e9,
            april_notional=100e9,
            may_notional=105e9,
        )

        assert abs(aana.aana - 100e9) < 1e6  # Allow small numerical error
        assert aana.applicable_phase == UMRPhase.PHASE_V
        assert aana.is_subject_to_umr()

    def test_aana_calculation_phase_iv(self):
        """Test AANA calculation for Phase IV."""
        checker = UMRComplianceChecker()
        aana = checker.calculate_aana(
            march_notional=800e9,
            april_notional=850e9,
            may_notional=750e9,
        )

        assert abs(aana.aana - 800e9) < 1e6
        assert aana.applicable_phase == UMRPhase.PHASE_IV

    def test_aana_calculation_below_threshold(self):
        """Test AANA below minimum threshold."""
        checker = UMRComplianceChecker()
        aana = checker.calculate_aana(
            march_notional=5e9,
            april_notional=6e9,
            may_notional=7e9,
        )

        assert abs(aana.aana - 6e9) < 1e6
        assert aana.applicable_phase is None
        assert not aana.is_subject_to_umr()

    def test_im_requirement_above_threshold(self):
        """Test IM calculation above threshold."""
        checker = UMRComplianceChecker(current_date=date(2024, 11, 5))
        call = checker.calculate_im_requirement(
            simm_im=75_000_000,  # Above $50MM threshold
            counterparty="DEALER-A",
            portfolio="PORT-001",
        )

        # IM = (75MM - 50MM threshold) = 25MM
        assert call.amount == 25_000_000
        assert call.margin_type == MarginType.INITIAL_MARGIN
        assert call.due_date == date(2024, 11, 6)  # T+1

    def test_im_requirement_below_threshold(self):
        """Test IM calculation below threshold."""
        checker = UMRComplianceChecker(current_date=date(2024, 11, 5))
        call = checker.calculate_im_requirement(
            simm_im=45_000_000,  # Below $50MM threshold
            counterparty="DEALER-A",
            portfolio="PORT-001",
        )

        # No IM posted if below threshold
        assert call.amount == 0.0

    def test_im_requirement_with_mta(self):
        """Test IM with MTA application."""
        checker = UMRComplianceChecker(current_date=date(2024, 11, 5))
        call = checker.calculate_im_requirement(
            simm_im=50_400_000,  # 50MM + 400k (below 500k MTA)
            counterparty="DEALER-A",
            portfolio="PORT-001",
        )

        # Amount after threshold is 400k, below 500k MTA, so no posting
        assert call.amount == 0.0

    def test_vm_requirement_loss(self):
        """Test VM calculation for loss."""
        checker = UMRComplianceChecker(current_date=date(2024, 11, 5))
        call = checker.calculate_vm_requirement(
            mtm_change=-2_000_000,  # Loss = we post VM
            counterparty="DEALER-A",
            portfolio="PORT-001",
        )

        # VM = -(-2MM) = 2MM to post
        assert call.amount == 2_000_000
        assert call.margin_type == MarginType.VARIATION_MARGIN
        assert call.due_date == date(2024, 11, 5)  # Same day

    def test_vm_requirement_gain(self):
        """Test VM calculation for gain."""
        checker = UMRComplianceChecker(current_date=date(2024, 11, 5))
        call = checker.calculate_vm_requirement(
            mtm_change=1_500_000,  # Gain = we collect VM
            counterparty="DEALER-A",
            portfolio="PORT-001",
        )

        # VM = -(1.5MM) = -1.5MM (negative = collection)
        assert call.amount == -1_500_000

    def test_vm_requirement_with_mta(self):
        """Test VM with MTA."""
        checker = UMRComplianceChecker(current_date=date(2024, 11, 5))

        # Small change below MTA
        call = checker.calculate_vm_requirement(
            mtm_change=-300_000,  # Below 500k MTA
            counterparty="DEALER-A",
            portfolio="PORT-001",
            outstanding_vm=0.0,
        )

        # Below MTA, so no new VM posted
        assert call.amount == 0.0

    def test_apply_collateral_haircut(self):
        """Test collateral haircut application."""
        checker = UMRComplianceChecker()

        # Cash: no haircut
        cash_value = checker.apply_collateral_haircut(10_000_000, CollateralType.CASH)
        assert cash_value == 10_000_000

        # Government bonds: 1% haircut
        bond_value = checker.apply_collateral_haircut(
            10_000_000, CollateralType.GOVERNMENT_BONDS
        )
        assert bond_value == 9_900_000  # 10MM * (1 - 0.01)

        # Corporate bonds: 4% haircut
        corp_value = checker.apply_collateral_haircut(
            10_000_000, CollateralType.CORPORATE_BONDS
        )
        assert corp_value == 9_600_000  # 10MM * (1 - 0.04)

        # Equity: 15% haircut
        equity_value = checker.apply_collateral_haircut(
            10_000_000, CollateralType.EQUITY
        )
        assert equity_value == 8_500_000  # 10MM * (1 - 0.15)

    def test_check_collateral_eligibility(self):
        """Test collateral eligibility checks."""
        checker = UMRComplianceChecker()

        # Eligible collateral
        is_eligible, reason = checker.check_collateral_eligibility(CollateralType.CASH)
        assert is_eligible
        assert "Eligible" in reason

        # Non-eligible collateral (equity not in default list)
        thresholds = UMRThresholds(
            eligible_collateral=[CollateralType.CASH, CollateralType.GOVERNMENT_BONDS]
        )
        checker_custom = UMRComplianceChecker(thresholds=thresholds)
        is_eligible, reason = checker_custom.check_collateral_eligibility(
            CollateralType.EQUITY
        )
        assert not is_eligible


class TestCSATerms:
    """Tests for CSA terms."""

    def test_default_csa_terms(self):
        """Test default CSA terms."""
        terms = CSATerms()

        assert terms.csa_type == "bilateral"
        assert terms.im_threshold == 50_000_000
        assert terms.vm_threshold == 0.0
        assert terms.mta == 500_000
        assert terms.rounding == 100_000
        assert terms.dispute_threshold == 250_000

    def test_custom_csa_terms(self):
        """Test custom CSA terms."""
        terms = CSATerms(
            im_threshold=100_000_000,
            mta=1_000_000,
            rounding=500_000,
            currency="EUR",
        )

        assert terms.im_threshold == 100_000_000
        assert terms.mta == 1_000_000
        assert terms.currency == "EUR"


class TestCSAPortfolio:
    """Tests for CSA portfolio."""

    def test_portfolio_creation(self):
        """Test CSA portfolio creation."""
        terms = CSATerms()
        portfolio = CSAPortfolio(
            counterparty="DEALER-A",
            portfolio_id="PORT-001",
            csa_terms=terms,
        )

        assert portfolio.counterparty == "DEALER-A"
        assert portfolio.outstanding_im == 0.0
        assert portfolio.outstanding_vm == 0.0

    def test_portfolio_collateral_value(self):
        """Test collateral value calculation."""
        terms = CSATerms()
        portfolio = CSAPortfolio(
            counterparty="DEALER-A",
            portfolio_id="PORT-001",
            csa_terms=terms,
            collateral_posted={
                CollateralType.CASH: 5_000_000,
                CollateralType.GOVERNMENT_BONDS: 10_000_000,
            },
        )

        # Without haircuts
        total_no_haircut = portfolio.total_collateral_posted_value(include_haircuts=False)
        assert total_no_haircut == 15_000_000

        # With haircuts: 5MM cash (no haircut) + 10MM bonds * 0.99
        total_with_haircut = portfolio.total_collateral_posted_value(include_haircuts=True)
        assert total_with_haircut == 5_000_000 + 9_900_000


class TestCSAManager:
    """Tests for CSA manager."""

    def test_manager_creation(self):
        """Test CSA manager creation."""
        manager = CSAManager(current_date=date(2024, 11, 5))
        assert manager.current_date == date(2024, 11, 5)
        assert len(manager.portfolios) == 0

    def test_register_portfolio(self):
        """Test portfolio registration."""
        manager = CSAManager()
        terms = CSATerms()
        portfolio = CSAPortfolio(
            counterparty="DEALER-A",
            portfolio_id="PORT-001",
            csa_terms=terms,
        )

        manager.register_portfolio(portfolio)
        assert len(manager.portfolios) == 1
        assert "DEALER-A_PORT-001" in manager.portfolios

    def test_calculate_margin_calls(self):
        """Test margin call calculation."""
        manager = CSAManager(current_date=date(2024, 11, 5))
        terms = CSATerms()
        portfolio = CSAPortfolio(
            counterparty="DEALER-A",
            portfolio_id="PORT-001",
            csa_terms=terms,
        )
        manager.register_portfolio(portfolio)

        # Calculate margin calls
        im_call, vm_call = manager.calculate_margin_calls(
            counterparty="DEALER-A",
            portfolio_id="PORT-001",
            simm_im=75_000_000,  # Above threshold
            mtm_change=-2_000_000,  # Loss
        )

        # IM: 75MM - 50MM threshold = 25MM
        assert im_call is not None
        assert im_call.amount == 25_000_000

        # VM: 2MM loss
        assert vm_call is not None
        assert vm_call.amount == 2_000_000

    def test_settle_margin_call(self):
        """Test margin call settlement."""
        manager = CSAManager(current_date=date(2024, 11, 5))
        terms = CSATerms()
        portfolio = CSAPortfolio(
            counterparty="DEALER-A",
            portfolio_id="PORT-001",
            csa_terms=terms,
        )
        manager.register_portfolio(portfolio)

        # Create a margin call
        margin_call = MarginCall(
            margin_type=MarginType.INITIAL_MARGIN,
            amount=25_000_000,
            currency="USD",
            due_date=date(2024, 11, 6),
            counterparty="DEALER-A",
            portfolio="PORT-001",
            calculation_date=date(2024, 11, 5),
        )

        # Settle with government bonds
        manager.settle_margin_call(
            margin_call, CollateralType.GOVERNMENT_BONDS, 25_000_000
        )

        # Check portfolio updated
        portfolio = manager.portfolios["DEALER-A_PORT-001"]
        assert portfolio.outstanding_im == 25_000_000
        assert portfolio.collateral_posted[CollateralType.GOVERNMENT_BONDS] == 25_000_000

    def test_get_portfolio_summary(self):
        """Test portfolio summary."""
        manager = CSAManager()
        terms = CSATerms()
        portfolio = CSAPortfolio(
            counterparty="DEALER-A",
            portfolio_id="PORT-001",
            csa_terms=terms,
            outstanding_im=10_000_000,
            outstanding_vm=2_000_000,
            collateral_posted={CollateralType.CASH: 12_000_000},
        )
        manager.register_portfolio(portfolio)

        summary = manager.get_portfolio_summary("DEALER-A", "PORT-001")

        assert summary["outstanding_im"] == 10_000_000
        assert summary["outstanding_vm"] == 2_000_000
        assert summary["collateral_posted_value"] == 12_000_000


class TestCustodianInterface:
    """Tests for custodian interface."""

    def test_custodian_creation(self):
        """Test custodian interface creation."""
        custodian = CustodianInterface(custodian_name="Euroclear")
        assert custodian.custodian_name == "Euroclear"
        assert len(custodian.accounts) == 0

    def test_register_account(self):
        """Test account registration."""
        custodian = CustodianInterface(custodian_name="Euroclear")
        account = CustodianAccount(
            custodian_name="Euroclear",
            account_number="EC-123456",
            account_type="segregated",
            currency="USD",
            balance=50_000_000,
        )

        custodian.register_account(account)
        assert len(custodian.accounts) == 1
        assert "EC-123456" in custodian.accounts

    def test_initiate_collateral_movement(self):
        """Test collateral movement initiation."""
        custodian = CustodianInterface(custodian_name="Euroclear")

        # Register accounts
        from_account = CustodianAccount(
            custodian_name="Euroclear",
            account_number="EC-123456",
            account_type="segregated",
            currency="USD",
            balance=50_000_000,
        )
        to_account = CustodianAccount(
            custodian_name="Euroclear",
            account_number="EC-789012",
            account_type="segregated",
            currency="USD",
            balance=0.0,
        )

        custodian.register_account(from_account)
        custodian.register_account(to_account)

        # Create margin call
        margin_call = MarginCall(
            margin_type=MarginType.INITIAL_MARGIN,
            amount=30_000_000,
            currency="USD",
            due_date=date(2024, 11, 6),
            counterparty="DEALER-A",
            portfolio="PORT-001",
            calculation_date=date(2024, 11, 5),
        )

        # Initiate movement
        movement = custodian.initiate_collateral_movement(
            margin_call=margin_call,
            from_account_number="EC-123456",
            to_account_number="EC-789012",
            collateral_type=CollateralType.CASH,
        )

        assert movement.amount == 30_000_000
        assert movement.collateral_type == CollateralType.CASH
        assert movement.status == "pending"

    def test_get_available_collateral(self):
        """Test available collateral query."""
        custodian = CustodianInterface(custodian_name="Euroclear")
        account = CustodianAccount(
            custodian_name="Euroclear",
            account_number="EC-123456",
            account_type="segregated",
            currency="USD",
            balance=50_000_000,
        )
        custodian.register_account(account)

        # Check cash availability
        available = custodian.get_available_collateral("EC-123456", CollateralType.CASH)
        assert available == 50_000_000


class TestMarginReport:
    """Tests for margin report generation."""

    def test_generate_margin_report(self):
        """Test margin report generation."""
        margin_calls = [
            MarginCall(
                margin_type=MarginType.INITIAL_MARGIN,
                amount=25_000_000,
                currency="USD",
                due_date=date(2024, 11, 6),
                counterparty="DEALER-A",
                portfolio="PORT-001",
                calculation_date=date(2024, 11, 5),
            ),
            MarginCall(
                margin_type=MarginType.VARIATION_MARGIN,
                amount=2_000_000,
                currency="USD",
                due_date=date(2024, 11, 5),
                counterparty="DEALER-A",
                portfolio="PORT-001",
                calculation_date=date(2024, 11, 5),
            ),
        ]

        report = generate_margin_report(margin_calls)

        assert "MARGIN CALL REPORT" in report
        assert "DEALER-A" in report
        assert "25,000,000" in report
        assert "2,000,000" in report
        assert "POST" in report
