"""CSA (Credit Support Annex) contract terms for collateral management.

This module provides data models for representing CSA agreements, which govern
collateral posting between counterparties for OTC derivatives.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, Field


class CollateralType(str, Enum):
    """Types of eligible collateral."""

    CASH = "Cash"
    GOVERNMENT_BOND = "GovernmentBond"
    CORPORATE_BOND = "CorporateBond"
    EQUITY = "Equity"
    LETTER_OF_CREDIT = "LetterOfCredit"
    GOLD = "Gold"
    OTHER = "Other"


class ValuationFrequency(str, Enum):
    """Frequency of collateral valuation and margin calls."""

    DAILY = "Daily"
    WEEKLY = "Weekly"
    MONTHLY = "Monthly"


class DisputeResolution(str, Enum):
    """Method for resolving valuation disputes."""

    MARKET_QUOTATION = "MarketQuotation"
    LOSS = "Loss"
    DEALER_POLL = "DealerPoll"


class ThresholdTerms(BaseModel):
    """Bilateral threshold and minimum transfer terms.

    Attributes
    ----------
    threshold_party_a : float
        Threshold for Party A (uncollateralized exposure allowed), default 0
    threshold_party_b : float
        Threshold for Party B (uncollateralized exposure allowed), default 0
    mta_party_a : float
        Minimum Transfer Amount for Party A, default 0
    mta_party_b : float
        Minimum Transfer Amount for Party B, default 0
    independent_amount_party_a : float
        Independent Amount (IA) posted by Party A, default 0
    independent_amount_party_b : float
        Independent Amount (IA) posted by Party B, default 0
    rounding : float
        Rounding increment for collateral amounts, default 100,000
    """

    threshold_party_a: float = Field(default=0.0, ge=0.0)
    threshold_party_b: float = Field(default=0.0, ge=0.0)
    mta_party_a: float = Field(default=0.0, ge=0.0)
    mta_party_b: float = Field(default=0.0, ge=0.0)
    independent_amount_party_a: float = Field(default=0.0, ge=0.0)
    independent_amount_party_b: float = Field(default=0.0, ge=0.0)
    rounding: float = Field(default=100_000.0, gt=0.0)

    def get_threshold(self, party: str) -> float:
        """Get threshold for a specific party.

        Parameters
        ----------
        party : str
            "A" or "B"

        Returns
        -------
        float
            Threshold amount
        """
        if party.upper() == "A":
            return self.threshold_party_a
        elif party.upper() == "B":
            return self.threshold_party_b
        else:
            raise ValueError(f"Party must be 'A' or 'B', got '{party}'")

    def get_mta(self, party: str) -> float:
        """Get MTA for a specific party."""
        if party.upper() == "A":
            return self.mta_party_a
        elif party.upper() == "B":
            return self.mta_party_b
        else:
            raise ValueError(f"Party must be 'A' or 'B', got '{party}'")

    def get_independent_amount(self, party: str) -> float:
        """Get Independent Amount for a specific party."""
        if party.upper() == "A":
            return self.independent_amount_party_a
        elif party.upper() == "B":
            return self.independent_amount_party_b
        else:
            raise ValueError(f"Party must be 'A' or 'B', got '{party}'")


class EligibleCollateral(BaseModel):
    """Specification of eligible collateral with haircuts.

    Attributes
    ----------
    collateral_type : CollateralType
        Type of collateral asset
    currency : str, optional
        Currency code for cash collateral (ISO 4217)
    haircut : float
        Haircut percentage (0-1), e.g., 0.02 = 2% haircut
    concentration_limit : float, optional
        Maximum percentage of total collateral allowed (0-1)
    rating_threshold : str, optional
        Minimum credit rating required (e.g., "BBB-")
    maturity_max_years : float, optional
        Maximum maturity for bonds (in years)
    """

    collateral_type: CollateralType
    currency: Optional[str] = Field(default=None, min_length=3, max_length=3)
    haircut: float = Field(default=0.0, ge=0.0, le=1.0)
    concentration_limit: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    rating_threshold: Optional[str] = None
    maturity_max_years: Optional[float] = Field(default=None, gt=0.0)

    def apply_haircut(self, market_value: float) -> float:
        """Apply haircut to market value to get collateral value.

        Parameters
        ----------
        market_value : float
            Market value of the collateral asset

        Returns
        -------
        float
            Collateral value after haircut: market_value * (1 - haircut)
        """
        return market_value * (1.0 - self.haircut)


class CollateralTerms(BaseModel):
    """Operational terms for collateral management.

    Attributes
    ----------
    base_currency : str
        Base currency for collateral calculations (ISO 4217)
    valuation_frequency : ValuationFrequency
        How often collateral is valued and margin calls made
    valuation_time : str, optional
        Time of day for valuation (e.g., "10:00:00 EST")
    dispute_threshold : float
        Threshold for raising valuation disputes, default 0
    dispute_resolution : DisputeResolution
        Method for resolving disputes
    eligible_collateral : list[EligibleCollateral]
        List of acceptable collateral types with haircuts
    """

    base_currency: str = Field(min_length=3, max_length=3)
    valuation_frequency: ValuationFrequency = ValuationFrequency.DAILY
    valuation_time: Optional[str] = None
    dispute_threshold: float = Field(default=0.0, ge=0.0)
    dispute_resolution: DisputeResolution = DisputeResolution.MARKET_QUOTATION
    eligible_collateral: list[EligibleCollateral] = Field(default_factory=list)

    def get_collateral_spec(self, collateral_type: CollateralType) -> Optional[EligibleCollateral]:
        """Get specification for a specific collateral type.

        Parameters
        ----------
        collateral_type : CollateralType
            Type of collateral to look up

        Returns
        -------
        EligibleCollateral or None
            Collateral specification if eligible, None otherwise
        """
        for spec in self.eligible_collateral:
            if spec.collateral_type == collateral_type:
                return spec
        return None

    def is_eligible(self, collateral_type: CollateralType) -> bool:
        """Check if a collateral type is eligible."""
        return self.get_collateral_spec(collateral_type) is not None


class CSA(BaseModel):
    """Credit Support Annex (CSA) agreement.

    A CSA is an annex to an ISDA Master Agreement that governs the posting
    of collateral for OTC derivatives transactions.

    Attributes
    ----------
    id : str
        Unique identifier for the CSA agreement
    party_a_id : str
        Counterparty ID for Party A
    party_b_id : str
        Counterparty ID for Party B
    effective_date : str
        Effective date of the CSA (ISO 8601 format)
    threshold_terms : ThresholdTerms
        Bilateral threshold, MTA, and IA terms
    collateral_terms : CollateralTerms
        Operational collateral management terms
    initial_margin_required : bool
        Whether initial margin is required (e.g., for uncleared swaps), default False
    variation_margin_required : bool
        Whether variation margin is required, default True
    rehypothecation_allowed : bool
        Whether posted collateral can be rehypothecated, default False
    segregation_required : bool
        Whether collateral must be segregated, default False
    """

    id: str
    party_a_id: str
    party_b_id: str
    effective_date: str
    threshold_terms: ThresholdTerms = Field(default_factory=ThresholdTerms)
    collateral_terms: CollateralTerms
    initial_margin_required: bool = False
    variation_margin_required: bool = True
    rehypothecation_allowed: bool = False
    segregation_required: bool = False

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CSA(id='{self.id}', "
            f"parties=['{self.party_a_id}', '{self.party_b_id}'], "
            f"IM={self.initial_margin_required}, VM={self.variation_margin_required})"
        )

    def is_party(self, counterparty_id: str) -> bool:
        """Check if a counterparty is a party to this CSA."""
        return counterparty_id in (self.party_a_id, self.party_b_id)

    def get_other_party(self, counterparty_id: str) -> str:
        """Get the other party's ID given one party's ID.

        Parameters
        ----------
        counterparty_id : str
            ID of one party

        Returns
        -------
        str
            ID of the other party

        Raises
        ------
        ValueError
            If counterparty_id is not a party to this CSA
        """
        if counterparty_id == self.party_a_id:
            return self.party_b_id
        elif counterparty_id == self.party_b_id:
            return self.party_a_id
        else:
            raise ValueError(f"Counterparty '{counterparty_id}' is not a party to this CSA")

    def get_party_label(self, counterparty_id: str) -> str:
        """Get party label ('A' or 'B') for a counterparty.

        Parameters
        ----------
        counterparty_id : str
            Counterparty ID

        Returns
        -------
        str
            "A" or "B"

        Raises
        ------
        ValueError
            If counterparty_id is not a party to this CSA
        """
        if counterparty_id == self.party_a_id:
            return "A"
        elif counterparty_id == self.party_b_id:
            return "B"
        else:
            raise ValueError(f"Counterparty '{counterparty_id}' is not a party to this CSA")

    def calculate_collateral_requirement(
        self,
        exposure: float,
        posted_by: str,
        include_independent_amount: bool = True,
    ) -> float:
        """Calculate collateral requirement for a given exposure.

        Parameters
        ----------
        exposure : float
            Mark-to-market exposure (positive = owed to posted_by party)
        posted_by : str
            ID of the party posting collateral
        include_independent_amount : bool
            Whether to include Independent Amount in calculation

        Returns
        -------
        float
            Required collateral amount (before rounding and MTA)

        Notes
        -----
        Formula: max(0, exposure - threshold) + independent_amount
        """
        party_label = self.get_party_label(posted_by)
        threshold = self.threshold_terms.get_threshold(party_label)
        ia = (
            self.threshold_terms.get_independent_amount(party_label)
            if include_independent_amount
            else 0.0
        )

        # Collateral required when exposure exceeds threshold
        requirement = max(0.0, exposure - threshold) + ia
        return requirement

    def apply_rounding(self, amount: float) -> float:
        """Apply rounding convention to collateral amount.

        Parameters
        ----------
        amount : float
            Collateral amount before rounding

        Returns
        -------
        float
            Rounded collateral amount
        """
        rounding = self.threshold_terms.rounding
        return round(amount / rounding) * rounding

    def apply_mta(self, current_collateral: float, required_collateral: float, posted_by: str) -> float:
        """Apply Minimum Transfer Amount logic.

        Parameters
        ----------
        current_collateral : float
            Currently posted collateral
        required_collateral : float
            Required collateral (after rounding)
        posted_by : str
            ID of party posting collateral

        Returns
        -------
        float
            Transfer amount (0 if below MTA)
        """
        party_label = self.get_party_label(posted_by)
        mta = self.threshold_terms.get_mta(party_label)

        transfer = required_collateral - current_collateral
        if abs(transfer) < mta:
            return 0.0
        return transfer
