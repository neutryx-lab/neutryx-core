"""Portfolio hierarchy and management for XVA and risk analytics.

This module provides the Portfolio class, which manages the complete hierarchy
of counterparties, master agreements, CSAs, netting sets, and trades.
"""
from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from neutryx.contracts.counterparty import Counterparty
from neutryx.contracts.csa import CSA
from neutryx.contracts.master_agreement import MasterAgreement
from neutryx.contracts.trade import Trade, ProductType, TradeStatus
from neutryx.portfolio.netting_set import NettingSet


class Portfolio(BaseModel):
    """Portfolio container managing the full trade hierarchy.

    The Portfolio class manages:
    - Counterparties
    - Master Agreements
    - CSA Agreements
    - Netting Sets
    - Individual Trades

    And provides methods for:
    - Portfolio aggregation
    - Counterparty-level exposure calculation
    - Netting set filtering and analysis
    - Trade lifecycle management

    Attributes
    ----------
    name : str
        Portfolio name
    counterparties : dict[str, Counterparty]
        Map of counterparty_id -> Counterparty
    master_agreements : dict[str, MasterAgreement]
        Map of agreement_id -> MasterAgreement
    csas : dict[str, CSA]
        Map of csa_id -> CSA
    netting_sets : dict[str, NettingSet]
        Map of netting_set_id -> NettingSet
    trades : dict[str, Trade]
        Map of trade_id -> Trade
    base_currency : str
        Base currency for portfolio-level aggregation (ISO 4217)
    """

    name: str
    counterparties: Dict[str, Counterparty] = Field(default_factory=dict)
    master_agreements: Dict[str, MasterAgreement] = Field(default_factory=dict)
    csas: Dict[str, CSA] = Field(default_factory=dict)
    netting_sets: Dict[str, NettingSet] = Field(default_factory=dict)
    trades: Dict[str, Trade] = Field(default_factory=dict)
    base_currency: str = Field(default="USD", min_length=3, max_length=3)

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Portfolio(name='{self.name}', "
            f"counterparties={len(self.counterparties)}, "
            f"netting_sets={len(self.netting_sets)}, "
            f"trades={len(self.trades)})"
        )

    # -------------------------------------------------------------------------
    # Entity Management
    # -------------------------------------------------------------------------

    def add_counterparty(self, counterparty: Counterparty) -> None:
        """Add a counterparty to the portfolio."""
        self.counterparties[counterparty.id] = counterparty

    def add_master_agreement(self, agreement: MasterAgreement) -> None:
        """Add a master agreement to the portfolio."""
        self.master_agreements[agreement.id] = agreement

    def add_csa(self, csa: CSA) -> None:
        """Add a CSA to the portfolio."""
        self.csas[csa.id] = csa

    def add_netting_set(self, netting_set: NettingSet) -> None:
        """Add a netting set to the portfolio."""
        self.netting_sets[netting_set.id] = netting_set

    def add_trade(self, trade: Trade) -> None:
        """Add a trade to the portfolio and update its netting set.

        Parameters
        ----------
        trade : Trade
            Trade to add

        Notes
        -----
        If trade has a netting_set_id, it will be automatically added to that
        netting set. The netting set must already exist in the portfolio.
        """
        self.trades[trade.id] = trade
        if trade.netting_set_id:
            netting_set = self.netting_sets.get(trade.netting_set_id)
            if netting_set:
                netting_set.add_trade(trade.id)

    def remove_trade(self, trade_id: str) -> bool:
        """Remove a trade from the portfolio.

        Parameters
        ----------
        trade_id : str
            ID of trade to remove

        Returns
        -------
        bool
            True if trade was removed, False if not found
        """
        trade = self.trades.get(trade_id)
        if trade is None:
            return False

        # Remove from netting set
        if trade.netting_set_id:
            netting_set = self.netting_sets.get(trade.netting_set_id)
            if netting_set:
                netting_set.remove_trade(trade_id)

        # Remove from portfolio
        del self.trades[trade_id]
        return True

    # -------------------------------------------------------------------------
    # Portfolio Queries
    # -------------------------------------------------------------------------

    def get_counterparty(self, counterparty_id: str) -> Optional[Counterparty]:
        """Get a counterparty by ID."""
        return self.counterparties.get(counterparty_id)

    def get_trade(self, trade_id: str) -> Optional[Trade]:
        """Get a trade by ID."""
        return self.trades.get(trade_id)

    def get_netting_set(self, netting_set_id: str) -> Optional[NettingSet]:
        """Get a netting set by ID."""
        return self.netting_sets.get(netting_set_id)

    def get_csa(self, csa_id: str) -> Optional[CSA]:
        """Get a CSA by ID."""
        return self.csas.get(csa_id)

    def get_master_agreement(self, agreement_id: str) -> Optional[MasterAgreement]:
        """Get a master agreement by ID."""
        return self.master_agreements.get(agreement_id)

    # -------------------------------------------------------------------------
    # Trade Filtering
    # -------------------------------------------------------------------------

    def get_trades_by_counterparty(self, counterparty_id: str) -> List[Trade]:
        """Get all trades with a specific counterparty.

        Parameters
        ----------
        counterparty_id : str
            Counterparty ID

        Returns
        -------
        list[Trade]
            List of trades with the counterparty
        """
        return [t for t in self.trades.values() if t.counterparty_id == counterparty_id]

    def get_trades_by_netting_set(self, netting_set_id: str) -> List[Trade]:
        """Get all trades in a specific netting set.

        Parameters
        ----------
        netting_set_id : str
            Netting set ID

        Returns
        -------
        list[Trade]
            List of trades in the netting set
        """
        netting_set = self.netting_sets.get(netting_set_id)
        if netting_set is None:
            return []
        return [self.trades[tid] for tid in netting_set.trade_ids if tid in self.trades]

    def get_trades_by_product_type(self, product_type: ProductType) -> List[Trade]:
        """Get all trades of a specific product type.

        Parameters
        ----------
        product_type : ProductType
            Product type to filter by

        Returns
        -------
        list[Trade]
            List of trades with the specified product type
        """
        return [t for t in self.trades.values() if t.product_type == product_type]

    def get_active_trades(self) -> List[Trade]:
        """Get all active trades.

        Returns
        -------
        list[Trade]
            List of active trades
        """
        return [t for t in self.trades.values() if t.status == TradeStatus.ACTIVE]

    # -------------------------------------------------------------------------
    # Netting Set Queries
    # -------------------------------------------------------------------------

    def get_netting_sets_by_counterparty(self, counterparty_id: str) -> List[NettingSet]:
        """Get all netting sets for a specific counterparty.

        Parameters
        ----------
        counterparty_id : str
            Counterparty ID

        Returns
        -------
        list[NettingSet]
            List of netting sets with the counterparty
        """
        return [ns for ns in self.netting_sets.values() if ns.counterparty_id == counterparty_id]

    def get_collateralized_netting_sets(self) -> List[NettingSet]:
        """Get all netting sets with CSA agreements.

        Returns
        -------
        list[NettingSet]
            List of netting sets with CSA
        """
        return [ns for ns in self.netting_sets.values() if ns.has_csa()]

    def get_uncollateralized_netting_sets(self) -> List[NettingSet]:
        """Get all netting sets without CSA agreements.

        Returns
        -------
        list[NettingSet]
            List of netting sets without CSA
        """
        return [ns for ns in self.netting_sets.values() if not ns.has_csa()]

    # -------------------------------------------------------------------------
    # Exposure Aggregation
    # -------------------------------------------------------------------------

    def calculate_net_mtm_by_netting_set(self, netting_set_id: str) -> float:
        """Calculate net MTM for a netting set.

        Parameters
        ----------
        netting_set_id : str
            Netting set ID

        Returns
        -------
        float
            Sum of MTM across all trades in the netting set
        """
        trades = self.get_trades_by_netting_set(netting_set_id)
        return sum(t.get_mtm(default=0.0) for t in trades)

    def calculate_net_mtm_by_counterparty(self, counterparty_id: str) -> float:
        """Calculate net MTM across all netting sets with a counterparty.

        Parameters
        ----------
        counterparty_id : str
            Counterparty ID

        Returns
        -------
        float
            Sum of MTM across all trades with the counterparty
        """
        trades = self.get_trades_by_counterparty(counterparty_id)
        return sum(t.get_mtm(default=0.0) for t in trades)

    def calculate_gross_notional(self) -> float:
        """Calculate total gross notional across all trades.

        Returns
        -------
        float
            Sum of absolute notional amounts

        Notes
        -----
        Only includes trades with notional set. No currency conversion applied.
        """
        return sum(abs(t.notional) for t in self.trades.values() if t.notional is not None)

    def calculate_total_mtm(self) -> float:
        """Calculate total portfolio MTM.

        Returns
        -------
        float
            Sum of MTM across all trades

        Notes
        -----
        No currency conversion applied. Assumes all MTM in base currency.
        """
        return sum(t.get_mtm(default=0.0) for t in self.trades.values())

    # -------------------------------------------------------------------------
    # Portfolio Statistics
    # -------------------------------------------------------------------------

    def num_counterparties(self) -> int:
        """Get number of counterparties."""
        return len(self.counterparties)

    def num_netting_sets(self) -> int:
        """Get number of netting sets."""
        return len(self.netting_sets)

    def num_trades(self) -> int:
        """Get total number of trades."""
        return len(self.trades)

    def num_active_trades(self) -> int:
        """Get number of active trades."""
        return len(self.get_active_trades())

    def summary(self) -> Dict[str, int]:
        """Get portfolio summary statistics.

        Returns
        -------
        dict
            Summary with keys: counterparties, netting_sets, trades, active_trades
        """
        return {
            "counterparties": self.num_counterparties(),
            "netting_sets": self.num_netting_sets(),
            "trades": self.num_trades(),
            "active_trades": self.num_active_trades(),
        }

    # -------------------------------------------------------------------------
    # Maturity Analysis
    # -------------------------------------------------------------------------

    def get_trades_maturing_before(self, cutoff_date: date) -> List[Trade]:
        """Get trades maturing before a cutoff date.

        Parameters
        ----------
        cutoff_date : date
            Cutoff date

        Returns
        -------
        list[Trade]
            Trades with maturity_date < cutoff_date
        """
        return [
            t
            for t in self.trades.values()
            if t.maturity_date is not None and t.maturity_date < cutoff_date
        ]

    def get_trades_maturing_in_range(self, start_date: date, end_date: date) -> List[Trade]:
        """Get trades maturing within a date range.

        Parameters
        ----------
        start_date : date
            Range start (inclusive)
        end_date : date
            Range end (inclusive)

        Returns
        -------
        list[Trade]
            Trades with maturity in [start_date, end_date]
        """
        return [
            t
            for t in self.trades.values()
            if t.maturity_date is not None
            and start_date <= t.maturity_date <= end_date
        ]
