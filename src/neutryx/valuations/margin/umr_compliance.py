"""UMR (Uncleared Margin Rules) compliance framework.

This module implements regulatory requirements for bilateral margin exchange
under BCBS-IOSCO uncleared margin rules, including:
- Phase-in schedule (Phase I-VI)
- AANA (Average Aggregate Notional Amount) calculations
- IM and VM posting/collection requirements
- Threshold monitoring and compliance checks
- Documentation and audit trails

References:
    - BCBS-IOSCO (2015): Margin requirements for non-centrally cleared derivatives
    - BCBS-IOSCO (2020): Final revisions to the framework
    - Local implementations: US (Prudential/CFTC), EU (EMIR), UK (PRA)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
from jax import Array


class UMRPhase(Enum):
    """UMR phase-in schedule."""

    PHASE_I = 1    # Sep 2016: AANA > $3 trillion
    PHASE_II = 2   # Sep 2017: AANA > $2.25 trillion
    PHASE_III = 3  # Sep 2018: AANA > $1.5 trillion
    PHASE_IV = 4   # Sep 2019: AANA > $750 billion
    PHASE_V = 5    # Sep 2020: AANA > $50 billion
    PHASE_VI = 6   # Sep 2022: AANA > $8 billion


class MarginType(Enum):
    """Type of margin."""

    INITIAL_MARGIN = "IM"
    VARIATION_MARGIN = "VM"


class CollateralType(Enum):
    """Eligible collateral types."""

    CASH = "cash"
    GOVERNMENT_BONDS = "govt_bonds"
    CORPORATE_BONDS = "corp_bonds"
    EQUITY = "equity"
    GOLD = "gold"


@dataclass
class AANACalculation:
    """AANA (Average Aggregate Notional Amount) calculation result.

    AANA is calculated as the average of month-end notional amounts
    over March, April, and May for the current year.

    Attributes:
        march_notional: Notional amount at end of March
        april_notional: Notional amount at end of April
        may_notional: Notional amount at end of May
        aana: Average aggregate notional amount
        calculation_date: Date of AANA calculation
        applicable_phase: UMR phase based on AANA threshold
    """

    march_notional: float
    april_notional: float
    may_notional: float
    aana: float
    calculation_date: date
    applicable_phase: Optional[UMRPhase]

    def is_subject_to_umr(self) -> bool:
        """Check if counterparty is subject to UMR."""
        return self.applicable_phase is not None


@dataclass
class MarginCall:
    """Margin call details.

    Attributes:
        margin_type: IM or VM
        amount: Margin amount to be posted/collected
        currency: Currency of margin call
        due_date: Date margin is due
        counterparty: Counterparty identifier
        portfolio: Portfolio or netting set identifier
        calculation_date: Date margin was calculated
        outstanding_amount: Previously posted amount
        net_call: Net margin call (positive = post, negative = collect)
    """

    margin_type: MarginType
    amount: float
    currency: str
    due_date: date
    counterparty: str
    portfolio: str
    calculation_date: date
    outstanding_amount: float = 0.0

    @property
    def net_call(self) -> float:
        """Net margin call amount."""
        return self.amount - self.outstanding_amount

    @property
    def is_posting(self) -> bool:
        """True if this is a posting (we owe margin)."""
        return self.net_call > 0

    @property
    def is_collection(self) -> bool:
        """True if this is a collection (we receive margin)."""
        return self.net_call < 0


@dataclass
class UMRThresholds:
    """UMR threshold configuration.

    Attributes:
        im_threshold: IM threshold (typically $50MM per counterparty)
        mta: Minimum Transfer Amount (typically $500k)
        independent_amount: Independent amount if any
        eligible_collateral: List of eligible collateral types
        haircuts: Haircut percentages by collateral type
    """

    im_threshold: float = 50_000_000  # $50MM standard threshold
    mta: float = 500_000  # $500k standard MTA
    independent_amount: float = 0.0
    eligible_collateral: List[CollateralType] = None
    haircuts: Dict[CollateralType, float] = None

    def __post_init__(self):
        if self.eligible_collateral is None:
            # Default eligible collateral
            self.eligible_collateral = [
                CollateralType.CASH,
                CollateralType.GOVERNMENT_BONDS,
                CollateralType.CORPORATE_BONDS,
            ]

        if self.haircuts is None:
            # Default haircuts per BCBS-IOSCO
            self.haircuts = {
                CollateralType.CASH: 0.0,
                CollateralType.GOVERNMENT_BONDS: 0.01,  # 1% for high-quality govt bonds
                CollateralType.CORPORATE_BONDS: 0.04,   # 4% for investment grade
                CollateralType.EQUITY: 0.15,             # 15% for main index equities
                CollateralType.GOLD: 0.15,               # 15% for gold
            }


class UMRComplianceChecker:
    """UMR compliance checker and margin calculator.

    This class implements the BCBS-IOSCO uncleared margin rules for
    bilateral derivatives, including:
    - AANA calculation and phase-in determination
    - IM and VM threshold monitoring
    - Margin call generation
    - Collateral eligibility and haircut application
    """

    def __init__(
        self,
        thresholds: Optional[UMRThresholds] = None,
        current_date: Optional[date] = None
    ):
        """Initialize UMR compliance checker.

        Args:
            thresholds: UMR threshold configuration
            current_date: Current date for compliance checks (default: today)
        """
        self.thresholds = thresholds or UMRThresholds()
        self.current_date = current_date or date.today()

    def calculate_aana(
        self,
        march_notional: float,
        april_notional: float,
        may_notional: float,
        calculation_year: Optional[int] = None
    ) -> AANACalculation:
        """Calculate AANA and determine applicable UMR phase.

        Args:
            march_notional: Month-end notional for March
            april_notional: Month-end notional for April
            may_notional: Month-end notional for May
            calculation_year: Year of calculation (default: current year)

        Returns:
            AANACalculation with phase determination

        Example:
            >>> checker = UMRComplianceChecker()
            >>> aana = checker.calculate_aana(
            ...     march_notional=100e9,
            ...     april_notional=110e9,
            ...     may_notional=105e9
            ... )
            >>> print(f"AANA: ${aana.aana/1e9:.1f}B")
            >>> print(f"Phase: {aana.applicable_phase}")
        """
        # Calculate average
        aana = (march_notional + april_notional + may_notional) / 3.0

        calc_year = calculation_year or self.current_date.year
        calc_date = date(calc_year, 6, 1)  # AANA calculated as of June 1

        # Determine phase based on AANA thresholds
        phase = None
        if aana > 3_000_000_000_000:  # $3 trillion
            phase = UMRPhase.PHASE_I
        elif aana > 2_250_000_000_000:  # $2.25 trillion
            phase = UMRPhase.PHASE_II
        elif aana > 1_500_000_000_000:  # $1.5 trillion
            phase = UMRPhase.PHASE_III
        elif aana > 750_000_000_000:  # $750 billion
            phase = UMRPhase.PHASE_IV
        elif aana > 50_000_000_000:  # $50 billion
            phase = UMRPhase.PHASE_V
        elif aana > 8_000_000_000:  # $8 billion
            phase = UMRPhase.PHASE_VI

        return AANACalculation(
            march_notional=march_notional,
            april_notional=april_notional,
            may_notional=may_notional,
            aana=aana,
            calculation_date=calc_date,
            applicable_phase=phase
        )

    def calculate_im_requirement(
        self,
        simm_im: float,
        counterparty: str,
        portfolio: str,
        currency: str = "USD"
    ) -> MarginCall:
        """Calculate IM requirement with threshold application.

        Args:
            simm_im: SIMM initial margin amount
            counterparty: Counterparty identifier
            portfolio: Portfolio/netting set identifier
            currency: Currency of margin

        Returns:
            MarginCall with IM requirement

        Note:
            - IM threshold is typically $50MM per counterparty
            - Only excess over threshold is posted
            - Subject to Minimum Transfer Amount (MTA)
        """
        # Apply IM threshold (first $50MM not posted)
        im_after_threshold = max(0, simm_im - self.thresholds.im_threshold)

        # Apply MTA - only post if above MTA
        im_to_post = im_after_threshold if im_after_threshold >= self.thresholds.mta else 0.0

        # IM due T+1 business day
        from datetime import timedelta
        due_date = self.current_date + timedelta(days=1)

        return MarginCall(
            margin_type=MarginType.INITIAL_MARGIN,
            amount=im_to_post,
            currency=currency,
            due_date=due_date,
            counterparty=counterparty,
            portfolio=portfolio,
            calculation_date=self.current_date
        )

    def calculate_vm_requirement(
        self,
        mtm_change: float,
        counterparty: str,
        portfolio: str,
        currency: str = "USD",
        outstanding_vm: float = 0.0
    ) -> MarginCall:
        """Calculate VM (Variation Margin) requirement.

        Args:
            mtm_change: Mark-to-market change since last VM exchange
            counterparty: Counterparty identifier
            portfolio: Portfolio/netting set identifier
            currency: Currency of margin
            outstanding_vm: Previously posted VM amount

        Returns:
            MarginCall with VM requirement

        Note:
            - VM has no threshold under UMR
            - VM exchanges daily
            - Subject to MTA for net change
        """
        # VM is the negative of MTM (if we lost money, post VM)
        vm_amount = -mtm_change

        # Check MTA on net change
        net_change = vm_amount - outstanding_vm
        vm_to_post = vm_amount if abs(net_change) >= self.thresholds.mta else outstanding_vm

        # VM due same day (T+0)
        due_date = self.current_date

        return MarginCall(
            margin_type=MarginType.VARIATION_MARGIN,
            amount=vm_to_post,
            currency=currency,
            due_date=due_date,
            counterparty=counterparty,
            portfolio=portfolio,
            calculation_date=self.current_date,
            outstanding_amount=outstanding_vm
        )

    def apply_collateral_haircut(
        self,
        collateral_value: float,
        collateral_type: CollateralType
    ) -> float:
        """Apply regulatory haircut to collateral value.

        Args:
            collateral_value: Market value of collateral
            collateral_type: Type of collateral

        Returns:
            Adjusted collateral value after haircut

        Example:
            >>> checker = UMRComplianceChecker()
            >>> bond_value = 10_000_000
            >>> adjusted = checker.apply_collateral_haircut(
            ...     bond_value, CollateralType.GOVERNMENT_BONDS
            ... )
            >>> print(f"After 1% haircut: ${adjusted:,.0f}")
        """
        haircut = self.thresholds.haircuts.get(collateral_type, 0.0)
        return collateral_value * (1.0 - haircut)

    def check_collateral_eligibility(
        self,
        collateral_type: CollateralType
    ) -> Tuple[bool, str]:
        """Check if collateral type is eligible.

        Args:
            collateral_type: Type of collateral to check

        Returns:
            Tuple of (is_eligible, reason)
        """
        if collateral_type in self.thresholds.eligible_collateral:
            return True, "Eligible under CSA"
        else:
            return False, f"{collateral_type.value} not in eligible collateral list"


def generate_margin_report(
    margin_calls: List[MarginCall]
) -> str:
    """Generate formatted margin report.

    Args:
        margin_calls: List of margin calls for reporting period

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("MARGIN CALL REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Summary by margin type
    im_calls = [c for c in margin_calls if c.margin_type == MarginType.INITIAL_MARGIN]
    vm_calls = [c for c in margin_calls if c.margin_type == MarginType.VARIATION_MARGIN]

    lines.append("SUMMARY")
    lines.append("-" * 80)
    lines.append(f"Total Initial Margin Calls:    {len(im_calls)}")
    lines.append(f"Total Variation Margin Calls:  {len(vm_calls)}")
    lines.append("")

    if im_calls:
        total_im = sum(c.net_call for c in im_calls)
        lines.append(f"Total IM Amount:  ${total_im:>20,.0f}")

    if vm_calls:
        total_vm = sum(c.net_call for c in vm_calls)
        lines.append(f"Total VM Amount:  ${total_vm:>20,.0f}")

    lines.append("")
    lines.append("MARGIN CALLS BY COUNTERPARTY")
    lines.append("-" * 80)

    # Group by counterparty
    by_cpty = {}
    for call in margin_calls:
        if call.counterparty not in by_cpty:
            by_cpty[call.counterparty] = []
        by_cpty[call.counterparty].append(call)

    for cpty, calls in sorted(by_cpty.items()):
        lines.append(f"\n{cpty}")
        lines.append("  " + "-" * 76)

        for call in calls:
            action = "POST" if call.is_posting else "COLLECT"
            lines.append(
                f"  {call.margin_type.value}  "
                f"{action:8s}  "
                f"${abs(call.net_call):>15,.0f}  "
                f"Due: {call.due_date}  "
                f"Portfolio: {call.portfolio}"
            )

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


# ============================================================================
# Custodian Integration Framework
# ============================================================================


@dataclass
class CustodianAccount:
    """Custodian account information.

    Attributes:
        custodian_name: Name of custodian (e.g., Euroclear, BNY Mellon)
        account_number: Account number
        account_type: Account type (segregated, omnibus, etc.)
        currency: Base currency
        balance: Current cash balance
        securities: Holdings of securities
    """

    custodian_name: str
    account_number: str
    account_type: str
    currency: str
    balance: float
    securities: Dict[str, float] = None  # ISIN -> quantity


@dataclass
class CollateralMovement:
    """Collateral movement instruction.

    Attributes:
        movement_id: Unique movement identifier
        from_account: Source custodian account
        to_account: Destination custodian account
        collateral_type: Type of collateral
        amount: Amount to transfer
        value_date: Settlement date
        reference: Reference (e.g., margin call ID)
        status: Movement status
    """

    movement_id: str
    from_account: CustodianAccount
    to_account: CustodianAccount
    collateral_type: CollateralType
    amount: float
    value_date: date
    reference: str
    status: str = "pending"


class CustodianInterface:
    """Interface for custodian integration.

    This class provides a framework for integrating with collateral
    custodians (tri-party agents) for margin settlement.
    """

    def __init__(self, custodian_name: str):
        """Initialize custodian interface.

        Args:
            custodian_name: Name of custodian
        """
        self.custodian_name = custodian_name
        self.accounts: Dict[str, CustodianAccount] = {}

    def register_account(self, account: CustodianAccount):
        """Register a custodian account.

        Args:
            account: Custodian account to register
        """
        self.accounts[account.account_number] = account

    def initiate_collateral_movement(
        self,
        margin_call: MarginCall,
        from_account_number: str,
        to_account_number: str,
        collateral_type: CollateralType
    ) -> CollateralMovement:
        """Initiate collateral movement for margin call.

        Args:
            margin_call: Margin call requiring collateral movement
            from_account_number: Source account
            to_account_number: Destination account
            collateral_type: Type of collateral to move

        Returns:
            CollateralMovement instruction
        """
        from_account = self.accounts[from_account_number]
        to_account = self.accounts[to_account_number]

        movement = CollateralMovement(
            movement_id=f"MOV-{margin_call.calculation_date}-{margin_call.counterparty}",
            from_account=from_account,
            to_account=to_account,
            collateral_type=collateral_type,
            amount=abs(margin_call.net_call),
            value_date=margin_call.due_date,
            reference=f"{margin_call.margin_type.value}-{margin_call.counterparty}"
        )

        return movement

    def get_available_collateral(
        self,
        account_number: str,
        collateral_type: CollateralType
    ) -> float:
        """Get available collateral in account.

        Args:
            account_number: Account to check
            collateral_type: Type of collateral

        Returns:
            Available amount
        """
        account = self.accounts.get(account_number)
        if not account:
            return 0.0

        if collateral_type == CollateralType.CASH:
            return account.balance
        else:
            # For securities, would need pricing
            return 0.0  # Placeholder


# ============================================================================
# CSA Integration Framework
# ============================================================================


@dataclass
class CSATerms:
    """Credit Support Annex (CSA) terms.

    Attributes:
        csa_type: Type of CSA (bilateral, one-way)
        im_threshold: Initial margin threshold
        vm_threshold: Variation margin threshold (typically 0 for UMR)
        mta: Minimum Transfer Amount
        independent_amount: Independent amount if any
        eligible_collateral: List of eligible collateral types
        rounding: Rounding amount for margin calls
        dispute_threshold: Threshold for dispute resolution
        haircuts: Collateral haircuts by type
        currency: Base currency
    """

    csa_type: str = "bilateral"
    im_threshold: float = 50_000_000  # $50MM standard
    vm_threshold: float = 0.0  # No threshold for VM under UMR
    mta: float = 500_000  # $500k standard
    independent_amount: float = 0.0
    eligible_collateral: List[CollateralType] = None
    rounding: float = 100_000  # Round to nearest $100k
    dispute_threshold: float = 250_000  # $250k
    haircuts: Dict[CollateralType, float] = None
    currency: str = "USD"

    def __post_init__(self):
        if self.eligible_collateral is None:
            self.eligible_collateral = [
                CollateralType.CASH,
                CollateralType.GOVERNMENT_BONDS,
                CollateralType.CORPORATE_BONDS,
            ]

        if self.haircuts is None:
            # Default BCBS-IOSCO haircuts
            self.haircuts = {
                CollateralType.CASH: 0.0,
                CollateralType.GOVERNMENT_BONDS: 0.01,
                CollateralType.CORPORATE_BONDS: 0.04,
                CollateralType.EQUITY: 0.15,
                CollateralType.GOLD: 0.15,
            }


@dataclass
class CSAPortfolio:
    """CSA portfolio with counterparty and collateral information.

    Attributes:
        counterparty: Counterparty name/identifier
        portfolio_id: Portfolio or netting set ID
        csa_terms: CSA terms
        outstanding_im: Outstanding initial margin posted
        outstanding_vm: Outstanding variation margin posted
        collateral_posted: Collateral posted by us
        collateral_received: Collateral received from counterparty
    """

    counterparty: str
    portfolio_id: str
    csa_terms: CSATerms
    outstanding_im: float = 0.0
    outstanding_vm: float = 0.0
    collateral_posted: Dict[CollateralType, float] = None
    collateral_received: Dict[CollateralType, float] = None

    def __post_init__(self):
        if self.collateral_posted is None:
            self.collateral_posted = {}
        if self.collateral_received is None:
            self.collateral_received = {}

    def total_collateral_posted_value(self, include_haircuts: bool = True) -> float:
        """Calculate total value of collateral posted.

        Args:
            include_haircuts: Whether to apply haircuts

        Returns:
            Total collateral value
        """
        total = 0.0
        for coll_type, amount in self.collateral_posted.items():
            if include_haircuts:
                haircut = self.csa_terms.haircuts.get(coll_type, 0.0)
                total += amount * (1.0 - haircut)
            else:
                total += amount
        return total

    def total_collateral_received_value(self, include_haircuts: bool = True) -> float:
        """Calculate total value of collateral received.

        Args:
            include_haircuts: Whether to apply haircuts

        Returns:
            Total collateral value
        """
        total = 0.0
        for coll_type, amount in self.collateral_received.items():
            if include_haircuts:
                haircut = self.csa_terms.haircuts.get(coll_type, 0.0)
                total += amount * (1.0 - haircut)
            else:
                total += amount
        return total


class CSAManager:
    """Manager for CSA portfolios and margin calculations.

    This class integrates SIMM calculations with UMR compliance
    and CSA terms to generate margin calls.
    """

    def __init__(self, current_date: Optional[date] = None):
        """Initialize CSA manager.

        Args:
            current_date: Current date (default: today)
        """
        self.current_date = current_date or date.today()
        self.portfolios: Dict[str, CSAPortfolio] = {}

    def register_portfolio(self, portfolio: CSAPortfolio):
        """Register a CSA portfolio.

        Args:
            portfolio: CSA portfolio to register
        """
        key = f"{portfolio.counterparty}_{portfolio.portfolio_id}"
        self.portfolios[key] = portfolio

    def calculate_margin_calls(
        self,
        counterparty: str,
        portfolio_id: str,
        simm_im: float,
        mtm_change: float,
    ) -> Tuple[Optional[MarginCall], Optional[MarginCall]]:
        """Calculate IM and VM margin calls for a portfolio.

        Args:
            counterparty: Counterparty identifier
            portfolio_id: Portfolio identifier
            simm_im: SIMM initial margin amount
            mtm_change: Mark-to-market change for VM

        Returns:
            Tuple of (im_call, vm_call), either can be None if no call needed
        """
        key = f"{counterparty}_{portfolio_id}"
        portfolio = self.portfolios.get(key)

        if not portfolio:
            raise ValueError(f"Portfolio {key} not registered")

        # Create compliance checker with CSA terms
        checker = UMRComplianceChecker(
            thresholds=UMRThresholds(
                im_threshold=portfolio.csa_terms.im_threshold,
                mta=portfolio.csa_terms.mta,
                independent_amount=portfolio.csa_terms.independent_amount,
                eligible_collateral=portfolio.csa_terms.eligible_collateral,
                haircuts=portfolio.csa_terms.haircuts,
            ),
            current_date=self.current_date,
        )

        # Calculate IM call
        im_call = checker.calculate_im_requirement(
            simm_im=simm_im,
            counterparty=counterparty,
            portfolio=portfolio_id,
            currency=portfolio.csa_terms.currency,
        )
        im_call.outstanding_amount = portfolio.outstanding_im

        # Calculate VM call
        vm_call = checker.calculate_vm_requirement(
            mtm_change=mtm_change,
            counterparty=counterparty,
            portfolio=portfolio_id,
            currency=portfolio.csa_terms.currency,
            outstanding_vm=portfolio.outstanding_vm,
        )

        # Apply rounding
        if portfolio.csa_terms.rounding > 0:
            im_call.amount = self._round_to_nearest(
                im_call.amount, portfolio.csa_terms.rounding
            )
            vm_call.amount = self._round_to_nearest(
                vm_call.amount, portfolio.csa_terms.rounding
            )

        # Return calls only if net amount exceeds dispute threshold
        im_result = im_call if abs(im_call.net_call) >= portfolio.csa_terms.dispute_threshold else None
        vm_result = vm_call if abs(vm_call.net_call) >= portfolio.csa_terms.dispute_threshold else None

        return im_result, vm_result

    def settle_margin_call(
        self,
        margin_call: MarginCall,
        collateral_type: CollateralType,
        amount: float,
    ):
        """Settle a margin call by updating portfolio collateral.

        Args:
            margin_call: Margin call being settled
            collateral_type: Type of collateral used for settlement
            amount: Amount of collateral
        """
        key = f"{margin_call.counterparty}_{margin_call.portfolio}"
        portfolio = self.portfolios.get(key)

        if not portfolio:
            raise ValueError(f"Portfolio {key} not found")

        # Update outstanding margin amounts
        if margin_call.margin_type == MarginType.INITIAL_MARGIN:
            portfolio.outstanding_im = margin_call.amount
        else:
            portfolio.outstanding_vm = margin_call.amount

        # Update collateral positions
        if margin_call.is_posting:
            # We are posting collateral
            if collateral_type not in portfolio.collateral_posted:
                portfolio.collateral_posted[collateral_type] = 0.0
            portfolio.collateral_posted[collateral_type] += amount
        else:
            # We are receiving collateral
            if collateral_type not in portfolio.collateral_received:
                portfolio.collateral_received[collateral_type] = 0.0
            portfolio.collateral_received[collateral_type] += amount

    def _round_to_nearest(self, value: float, rounding: float) -> float:
        """Round value to nearest rounding amount.

        Args:
            value: Value to round
            rounding: Rounding increment

        Returns:
            Rounded value
        """
        if rounding <= 0:
            return value
        return round(value / rounding) * rounding

    def get_portfolio_summary(self, counterparty: str, portfolio_id: str) -> Dict:
        """Get summary of portfolio collateral and margin.

        Args:
            counterparty: Counterparty identifier
            portfolio_id: Portfolio identifier

        Returns:
            Dictionary with portfolio summary
        """
        key = f"{counterparty}_{portfolio_id}"
        portfolio = self.portfolios.get(key)

        if not portfolio:
            return {}

        return {
            "counterparty": counterparty,
            "portfolio_id": portfolio_id,
            "outstanding_im": portfolio.outstanding_im,
            "outstanding_vm": portfolio.outstanding_vm,
            "collateral_posted_value": portfolio.total_collateral_posted_value(
                include_haircuts=True
            ),
            "collateral_received_value": portfolio.total_collateral_received_value(
                include_haircuts=True
            ),
            "collateral_posted_breakdown": dict(portfolio.collateral_posted),
            "collateral_received_breakdown": dict(portfolio.collateral_received),
            "csa_terms": {
                "im_threshold": portfolio.csa_terms.im_threshold,
                "vm_threshold": portfolio.csa_terms.vm_threshold,
                "mta": portfolio.csa_terms.mta,
                "rounding": portfolio.csa_terms.rounding,
            },
        }
