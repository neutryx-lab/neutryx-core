"""CLS (Continuous Linked Settlement) integration.

CLS is a specialized settlement system for FX transactions, providing
payment-versus-payment (PvP) settlement to eliminate settlement risk.
"""

from .connector import CLSConnector
from .messages import CLSSettlementInstruction, CLSConfirmation, CLSStatus
from .settlement import CLSSettlementService

__all__ = [
    "CLSConnector",
    "CLSSettlementInstruction",
    "CLSConfirmation",
    "CLSStatus",
    "CLSSettlementService",
]
