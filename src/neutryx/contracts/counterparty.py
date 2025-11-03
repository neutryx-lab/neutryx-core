"""Counterparty data models for credit risk and XVA calculations.

This module provides models for representing counterparties and their credit
attributes, supporting CVA, DVA, and bilateral XVA calculations.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from neutryx.market.credit.hazard import HazardRateCurve


class CreditRating(str, Enum):
    """Standard credit rating categories (S&P-style)."""

    AAA = "AAA"
    AA_PLUS = "AA+"
    AA = "AA"
    AA_MINUS = "AA-"
    A_PLUS = "A+"
    A = "A"
    A_MINUS = "A-"
    BBB_PLUS = "BBB+"
    BBB = "BBB"
    BBB_MINUS = "BBB-"
    BB_PLUS = "BB+"
    BB = "BB"
    BB_MINUS = "BB-"
    B_PLUS = "B+"
    B = "B"
    B_MINUS = "B-"
    CCC_PLUS = "CCC+"
    CCC = "CCC"
    CCC_MINUS = "CCC-"
    CC = "CC"
    C = "C"
    D = "D"
    NR = "NR"  # Not Rated


class EntityType(str, Enum):
    """Legal entity type classification."""

    CORPORATE = "Corporate"
    FINANCIAL = "Financial"
    SOVEREIGN = "Sovereign"
    MUNICIPAL = "Municipal"
    FUND = "Fund"
    SPV = "SPV"  # Special Purpose Vehicle
    OTHER = "Other"


class CounterpartyCredit(BaseModel):
    """Credit attributes for a counterparty.

    Attributes
    ----------
    rating : CreditRating, optional
        External credit rating (e.g., S&P, Moody's equivalent)
    internal_rating : str, optional
        Internal credit rating or score
    lgd : float
        Loss Given Default (0-1), default 0.6 (60% loss, 40% recovery)
    recovery_rate : float, optional
        Recovery rate (0-1), alternative to LGD (recovery_rate = 1 - lgd)
    hazard_curve : HazardRateCurve, optional
        Term structure of default intensities
    credit_spread_bps : float, optional
        Credit spread in basis points over risk-free rate
    """

    rating: Optional[CreditRating] = None
    internal_rating: Optional[str] = None
    lgd: float = Field(default=0.6, ge=0.0, le=1.0)
    recovery_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    hazard_curve: Optional[HazardRateCurve] = None
    credit_spread_bps: Optional[float] = Field(default=None, ge=0.0)

    class Config:
        arbitrary_types_allowed = True  # Allow HazardRateCurve

    def get_lgd(self) -> float:
        """Get Loss Given Default, computing from recovery_rate if needed."""
        if self.recovery_rate is not None:
            return 1.0 - self.recovery_rate
        return self.lgd

    def get_recovery_rate(self) -> float:
        """Get recovery rate, computing from LGD if needed."""
        if self.recovery_rate is not None:
            return self.recovery_rate
        return 1.0 - self.lgd


class Counterparty(BaseModel):
    """Counterparty representation for risk management and XVA.

    Attributes
    ----------
    id : str
        Unique identifier for the counterparty
    name : str
        Legal name of the counterparty
    entity_type : EntityType
        Classification of the legal entity
    lei : str, optional
        Legal Entity Identifier (ISO 17442)
    jurisdiction : str, optional
        Legal jurisdiction (ISO 3166 country code)
    credit : CounterpartyCredit, optional
        Credit risk attributes
    parent_id : str, optional
        ID of parent entity for corporate hierarchies
    is_bank : bool
        Whether the counterparty is a financial institution, default False
    is_clearinghouse : bool
        Whether the counterparty is a CCP (central counterparty), default False
    """

    id: str
    name: str
    entity_type: EntityType
    lei: Optional[str] = Field(
        default=None,
        min_length=20,
        max_length=20,
        description="Legal Entity Identifier (20 characters)",
    )
    jurisdiction: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=3,
        description="ISO 3166 country code",
    )
    credit: Optional[CounterpartyCredit] = None
    parent_id: Optional[str] = None
    is_bank: bool = False
    is_clearinghouse: bool = False

    def __repr__(self) -> str:
        """String representation."""
        rating_str = (
            f", rating={self.credit.rating.value}" if self.credit and self.credit.rating else ""
        )
        return f"Counterparty(id='{self.id}', name='{self.name}'{rating_str})"

    def has_credit_curve(self) -> bool:
        """Check if counterparty has a hazard rate curve for default modeling."""
        return self.credit is not None and self.credit.hazard_curve is not None

    def get_lgd(self) -> float:
        """Get Loss Given Default for CVA calculations.

        Returns
        -------
        float
            LGD value (0-1), defaults to 0.6 if no credit info available
        """
        if self.credit is None:
            return 0.6  # Market standard default
        return self.credit.get_lgd()
