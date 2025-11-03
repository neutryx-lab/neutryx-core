"""
Contract and counterparty management for XVA and SIMM calculations.

This module provides data models and utilities for managing:
- Counterparty information and credit attributes
- CSA (Credit Support Annex) contract terms
- Trade and position abstractions
- ISDA Master Agreement representations
"""

from neutryx.contracts.counterparty import Counterparty, CounterpartyCredit
from neutryx.contracts.csa import (
    CSA,
    CollateralTerms,
    EligibleCollateral,
    ThresholdTerms,
)
from neutryx.contracts.master_agreement import MasterAgreement, AgreementType
from neutryx.contracts.trade import Trade, TradeStatus

__all__ = [
    "Counterparty",
    "CounterpartyCredit",
    "CSA",
    "CollateralTerms",
    "EligibleCollateral",
    "ThresholdTerms",
    "MasterAgreement",
    "AgreementType",
    "Trade",
    "TradeStatus",
]
