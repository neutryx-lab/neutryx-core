"""Master netting agreement representations (ISDA Master, etc.).

This module provides data models for master agreements that govern the legal
relationship between counterparties for OTC derivatives.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AgreementType(str, Enum):
    """Type of master agreement."""

    ISDA_1992 = "ISDA1992"
    ISDA_2002 = "ISDA2002"
    ISDA_2006 = "ISDA2006"
    ISDA_2021 = "ISDA2021"
    GMRA = "GMRA"  # Global Master Repurchase Agreement
    GMSLA = "GMSLA"  # Global Master Securities Lending Agreement
    CUSTOM = "Custom"
    OTHER = "Other"


class GoverningLaw(str, Enum):
    """Governing law jurisdiction."""

    ENGLISH = "English"
    NEW_YORK = "NewYork"
    JAPANESE = "Japanese"
    GERMAN = "German"
    FRENCH = "French"
    OTHER = "Other"


class TerminationCurrency(str, Enum):
    """Currency for termination payments."""

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    BASE = "Base"  # Base currency of the CSA


class MasterAgreement(BaseModel):
    """Master netting agreement (e.g., ISDA Master Agreement).

    The master agreement establishes the legal framework for bilateral netting
    of derivatives transactions between two counterparties.

    Attributes
    ----------
    id : str
        Unique identifier for the master agreement
    agreement_type : AgreementType
        Type of master agreement (ISDA 1992, 2002, etc.)
    party_a_id : str
        Counterparty ID for Party A
    party_b_id : str
        Counterparty ID for Party B
    effective_date : date
        Date the agreement became effective
    governing_law : GoverningLaw
        Legal jurisdiction governing the agreement
    termination_currency : TerminationCurrency
        Currency for close-out termination payments
    csa_id : str, optional
        ID of the associated Credit Support Annex (if any)
    payment_netting : bool
        Whether payment netting is enabled, default True
    close_out_netting : bool
        Whether close-out netting is enabled, default True
    automatic_early_termination : bool
        Whether automatic early termination applies, default False
    walkaway_clause : bool
        Whether a walkaway clause exists (non-defaulting party can walk away), default False
    additional_termination_events : list[str], optional
        List of additional termination event descriptions
    credit_event_upon_merger : bool
        Whether credit event upon merger applies, default True
    cross_default_applicable : bool
        Whether cross-default provisions apply, default True
    """

    id: str
    agreement_type: AgreementType
    party_a_id: str
    party_b_id: str
    effective_date: date
    governing_law: GoverningLaw = GoverningLaw.ENGLISH
    termination_currency: TerminationCurrency = TerminationCurrency.USD
    csa_id: Optional[str] = None
    payment_netting: bool = True
    close_out_netting: bool = True
    automatic_early_termination: bool = False
    walkaway_clause: bool = False
    additional_termination_events: Optional[list[str]] = Field(default_factory=list)
    credit_event_upon_merger: bool = True
    cross_default_applicable: bool = True

    def __repr__(self) -> str:
        """String representation."""
        csa_str = f", CSA={self.csa_id}" if self.csa_id else ""
        return (
            f"MasterAgreement(id='{self.id}', "
            f"type={self.agreement_type.value}, "
            f"parties=['{self.party_a_id}', '{self.party_b_id}']"
            f"{csa_str})"
        )

    def is_party(self, counterparty_id: str) -> bool:
        """Check if a counterparty is a party to this agreement."""
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
            If counterparty_id is not a party to this agreement
        """
        if counterparty_id == self.party_a_id:
            return self.party_b_id
        elif counterparty_id == self.party_b_id:
            return self.party_a_id
        else:
            raise ValueError(
                f"Counterparty '{counterparty_id}' is not a party to this master agreement"
            )

    def has_csa(self) -> bool:
        """Check if this master agreement has an associated CSA."""
        return self.csa_id is not None

    def allows_netting(self) -> bool:
        """Check if netting is allowed under this agreement.

        Returns
        -------
        bool
            True if either payment netting or close-out netting is enabled
        """
        return self.payment_netting or self.close_out_netting

    def is_isda(self) -> bool:
        """Check if this is an ISDA agreement."""
        return self.agreement_type in (
            AgreementType.ISDA_1992,
            AgreementType.ISDA_2002,
            AgreementType.ISDA_2006,
            AgreementType.ISDA_2021,
        )

    def get_isda_version(self) -> Optional[int]:
        """Get ISDA version year if applicable.

        Returns
        -------
        int or None
            Year of ISDA version (1992, 2002, etc.) or None if not ISDA
        """
        if self.agreement_type == AgreementType.ISDA_1992:
            return 1992
        elif self.agreement_type == AgreementType.ISDA_2002:
            return 2002
        elif self.agreement_type == AgreementType.ISDA_2006:
            return 2006
        elif self.agreement_type == AgreementType.ISDA_2021:
            return 2021
        return None

    def supports_bilateral_netting(self) -> bool:
        """Check if bilateral close-out netting is supported.

        Returns
        -------
        bool
            True if close-out netting is enabled and no walkaway clause exists
        """
        return self.close_out_netting and not self.walkaway_clause
