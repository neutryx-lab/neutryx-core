"""Regulatory capital and risk management frameworks.

This module implements regulatory requirements including:
- Basel III/IV market risk (IMA)
- FRTB (Fundamental Review of the Trading Book)
- SA-CCR (Standardized Approach for Counterparty Credit Risk)
- CVA capital requirements
"""

from . import ima

__all__ = ["ima"]
