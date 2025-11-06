"""Regulatory capital and risk management frameworks.

This module implements regulatory requirements including:
- Basel III/IV market risk (IMA)
- FRTB (Fundamental Review of the Trading Book)
- SA-CCR (Standardized Approach for Counterparty Credit Risk)
- CVA capital requirements
- Accounting standards (IFRS 9/13)
- Trade reporting (EMIR, MiFID II, Basel)
"""

from . import accounting, ima, reporting

__all__ = ["accounting", "ima", "reporting"]
