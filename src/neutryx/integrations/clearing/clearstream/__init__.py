"""Clearstream integration for securities settlement.

Clearstream (part of Deutsche Börse Group) is one of the world's
leading international central securities depositories (ICSDs), providing
settlement and custody services for international securities transactions.

Note: Clearstream integration closely mirrors Euroclear as both are
ICSDs with similar functionality. In practice, the main differences are
in connectivity details and some messaging specifics.
"""

from ..euroclear.connector import EuroclearConnector as ClearstreamConnectorBase
from ..euroclear.messages import (
    EuroclearSettlementInstruction as ClearstreamSettlementInstruction,
    EuroclearConfirmation as ClearstreamConfirmation,
    EuroclearStatus as ClearstreamStatus,
    SettlementType,
)
from ..euroclear.settlement import EuroclearSettlementService as ClearstreamSettlementServiceBase


class ClearstreamConnector(ClearstreamConnectorBase):
    """Connector for Clearstream integration.

    Inherits from Euroclear connector as the functionality is nearly identical.
    Main differences are in endpoint configuration and BIC codes.
    """

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ClearstreamConnector("
            f"member={self.config.member_id}, "
            f"connected={self._connected})"
        )


class ClearstreamSettlementService(ClearstreamSettlementServiceBase):
    """High-level service for Clearstream settlement operations.

    Inherits from Euroclear settlement service as the business logic
    is nearly identical between the two ICSDs.
    """

    pass


__all__ = [
    "ClearstreamConnector",
    "ClearstreamSettlementInstruction",
    "ClearstreamConfirmation",
    "ClearstreamStatus",
    "SettlementType",
    "ClearstreamSettlementService",
]
