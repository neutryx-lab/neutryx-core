"""Netting set representation for bilateral close-out netting.

This module provides the NettingSet class, which groups trades eligible for
bilateral netting under a master agreement.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator


class NettingSet(BaseModel):
    """Collection of trades eligible for bilateral netting.

    A netting set groups all trades between two counterparties that fall under
    a single master agreement and are subject to close-out netting provisions.

    Attributes
    ----------
    id : str
        Unique identifier for the netting set
    name : str, optional
        Human-readable name for the netting set
    master_agreement_id : str
        ID of the master agreement governing this netting set
    counterparty_id : str
        ID of the counterparty (all trades must share this counterparty)
    csa_id : str, optional
        ID of the CSA agreement (if collateralized)
    trade_ids : list[str]
        List of trade IDs belonging to this netting set
    is_cleared : bool
        Whether trades are centrally cleared, default False
    clearinghouse_id : str, optional
        ID of the clearinghouse (if is_cleared=True)
    """

    id: str
    name: Optional[str] = None
    master_agreement_id: str
    counterparty_id: str
    csa_id: Optional[str] = None
    trade_ids: list[str] = Field(default_factory=list)
    is_cleared: bool = False
    clearinghouse_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_clearinghouse(self) -> "NettingSet":
        """Validate that clearinghouse_id is set if is_cleared is True."""
        if self.is_cleared and self.clearinghouse_id is None:
            raise ValueError("clearinghouse_id must be set when is_cleared=True")
        return self

    def __repr__(self) -> str:
        """String representation."""
        csa_str = f", CSA={self.csa_id}" if self.csa_id else ""
        return (
            f"NettingSet(id='{self.id}', "
            f"counterparty='{self.counterparty_id}', "
            f"trades={len(self.trade_ids)}{csa_str})"
        )

    def add_trade(self, trade_id: str) -> None:
        """Add a trade to the netting set.

        Parameters
        ----------
        trade_id : str
            ID of the trade to add

        Notes
        -----
        This does not validate that the trade belongs to the correct counterparty.
        Validation should be done by the caller.
        """
        if trade_id not in self.trade_ids:
            self.trade_ids.append(trade_id)

    def remove_trade(self, trade_id: str) -> bool:
        """Remove a trade from the netting set.

        Parameters
        ----------
        trade_id : str
            ID of the trade to remove

        Returns
        -------
        bool
            True if trade was removed, False if not found
        """
        if trade_id in self.trade_ids:
            self.trade_ids.remove(trade_id)
            return True
        return False

    def contains_trade(self, trade_id: str) -> bool:
        """Check if a trade is in this netting set.

        Parameters
        ----------
        trade_id : str
            Trade ID to check

        Returns
        -------
        bool
            True if trade is in the netting set
        """
        return trade_id in self.trade_ids

    def num_trades(self) -> int:
        """Get number of trades in the netting set.

        Returns
        -------
        int
            Number of trades
        """
        return len(self.trade_ids)

    def is_empty(self) -> bool:
        """Check if netting set has no trades.

        Returns
        -------
        bool
            True if no trades in the netting set
        """
        return len(self.trade_ids) == 0

    def has_csa(self) -> bool:
        """Check if this netting set has an associated CSA.

        Returns
        -------
        bool
            True if csa_id is set
        """
        return self.csa_id is not None

    def is_bilaterally_cleared(self) -> bool:
        """Check if this is a bilateral (non-cleared) netting set.

        Returns
        -------
        bool
            True if not centrally cleared
        """
        return not self.is_cleared

    def get_display_name(self) -> str:
        """Get display name for the netting set.

        Returns
        -------
        str
            Name if set, otherwise ID
        """
        return self.name if self.name else self.id
