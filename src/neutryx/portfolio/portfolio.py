"""Portfolio hierarchy and management for XVA and risk analytics.

This module provides the Portfolio class, which manages the complete hierarchy
of counterparties, master agreements, CSAs, netting sets, and trades.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Dict, List, Optional, Sequence

import jax.numpy as jnp
from pydantic import BaseModel, ConfigDict, Field

from neutryx.portfolio.contracts.counterparty import Counterparty
from neutryx.portfolio.contracts.csa import CSA
from neutryx.portfolio.contracts.master_agreement import MasterAgreement
from neutryx.portfolio.contracts.trade import Trade, ProductType, TradeStatus
from neutryx.portfolio.netting_set import NettingSet
from neutryx.portfolio.positions import PortfolioPosition


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
    positions: Dict[str, PortfolioPosition] = Field(default_factory=dict)
    cash_balances: Dict[str, Decimal] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

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

    # ------------------------------------------------------------------
    # Position and cash management
    # ------------------------------------------------------------------

    def get_position(self, security_id: str) -> Optional[PortfolioPosition]:
        """Return a position by security identifier."""

        return self.positions.get(security_id)

    def upsert_position(
        self,
        security_id: str,
        *,
        quantity: Decimal,
        cost_basis: Optional[Decimal] = None,
    ) -> PortfolioPosition:
        """Create or update a position in place."""

        position = self.positions.get(security_id)
        if position is None:
            position = PortfolioPosition(
                security_id=security_id,
                quantity=quantity,
                cost_basis=cost_basis or Decimal("0"),
            )
            self.positions[security_id] = position
        else:
            position.quantity = quantity
            if cost_basis is not None:
                position.cost_basis = cost_basis

        return position

    def adjust_position(
        self,
        security_id: str,
        quantity_delta: Decimal,
        *,
        cost_basis: Optional[Decimal] = None,
    ) -> PortfolioPosition:
        """Adjust an existing position quantity by a delta."""

        position = self.positions.get(security_id)
        if position is None:
            position = PortfolioPosition(
                security_id=security_id,
                quantity=Decimal("0"),
                cost_basis=cost_basis or Decimal("0"),
            )
            self.positions[security_id] = position

        position.adjust_quantity(quantity_delta)
        if cost_basis is not None:
            position.cost_basis = cost_basis

        if position.quantity == Decimal("0") and not position.metadata:
            del self.positions[security_id]

        return position

    def record_cash_flow(self, currency: str, amount: Decimal) -> None:
        """Record a cash movement impacting the portfolio."""

        current = self.cash_balances.get(currency, Decimal("0"))
        self.cash_balances[currency] = current + amount

    def get_cash_balance(self, currency: str) -> Decimal:
        """Retrieve the current cash balance for a currency."""

        return self.cash_balances.get(currency, Decimal("0"))

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

    # -------------------------------------------------------------------------
    # Credit profile helpers
    # -------------------------------------------------------------------------

    def get_counterparty_default_probabilities(
        self, counterparty_id: str, times: Sequence[float]
    ) -> jnp.ndarray | None:
        """Return cumulative default probabilities for the given counterparty."""

        counterparty = self.get_counterparty(counterparty_id)
        if counterparty and counterparty.has_credit_curve():
            curve = counterparty.credit.hazard_curve  # type: ignore[union-attr]
            return jnp.asarray(curve.default_probability(jnp.asarray(times)))
        return None

    def get_counterparty_lgd_curve(
        self, counterparty_id: str, times: Sequence[float]
    ) -> jnp.ndarray | None:
        """Return the LGD profile for a counterparty if available."""

        counterparty = self.get_counterparty(counterparty_id)
        if counterparty and counterparty.credit is not None:
            lgd = counterparty.credit.get_lgd()
            return jnp.full((len(times),), float(lgd), dtype=jnp.float32)
        return None

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

    # -------------------------------------------------------------------------
    # Book Hierarchy Queries
    # -------------------------------------------------------------------------

    def get_trades_by_book(self, book_id: str) -> List[Trade]:
        """Get all trades assigned to a specific book.

        Parameters
        ----------
        book_id : str
            Book ID

        Returns
        -------
        list[Trade]
            List of trades in the book
        """
        return [t for t in self.trades.values() if t.book_id == book_id]

    def get_trades_by_desk(self, desk_id: str) -> List[Trade]:
        """Get all trades assigned to a specific desk.

        Parameters
        ----------
        desk_id : str
            Desk ID

        Returns
        -------
        list[Trade]
            List of trades on the desk
        """
        return [t for t in self.trades.values() if t.desk_id == desk_id]

    def get_trades_by_trader(self, trader_id: str) -> List[Trade]:
        """Get all trades assigned to a specific trader.

        Parameters
        ----------
        trader_id : str
            Trader ID

        Returns
        -------
        list[Trade]
            List of trades for the trader
        """
        return [t for t in self.trades.values() if t.trader_id == trader_id]

    def calculate_mtm_by_book(self, book_id: str) -> float:
        """Calculate total MTM for a specific book.

        Parameters
        ----------
        book_id : str
            Book ID

        Returns
        -------
        float
            Sum of MTM for all trades in the book
        """
        trades = self.get_trades_by_book(book_id)
        return sum(t.get_mtm(default=0.0) for t in trades)

    def calculate_mtm_by_desk(self, desk_id: str) -> float:
        """Calculate total MTM for a specific desk.

        Parameters
        ----------
        desk_id : str
            Desk ID

        Returns
        -------
        float
            Sum of MTM for all trades on the desk
        """
        trades = self.get_trades_by_desk(desk_id)
        return sum(t.get_mtm(default=0.0) for t in trades)

    def calculate_mtm_by_trader(self, trader_id: str) -> float:
        """Calculate total MTM for a specific trader.

        Parameters
        ----------
        trader_id : str
            Trader ID

        Returns
        -------
        float
            Sum of MTM for all trades by the trader
        """
        trades = self.get_trades_by_trader(trader_id)
        return sum(t.get_mtm(default=0.0) for t in trades)

    def calculate_notional_by_book(self, book_id: str) -> float:
        """Calculate total notional for a specific book.

        Parameters
        ----------
        book_id : str
            Book ID

        Returns
        -------
        float
            Sum of absolute notional amounts in the book
        """
        trades = self.get_trades_by_book(book_id)
        return sum(abs(t.notional) for t in trades if t.notional is not None)

    def calculate_notional_by_desk(self, desk_id: str) -> float:
        """Calculate total notional for a specific desk.

        Parameters
        ----------
        desk_id : str
            Desk ID

        Returns
        -------
        float
            Sum of absolute notional amounts on the desk
        """
        trades = self.get_trades_by_desk(desk_id)
        return sum(abs(t.notional) for t in trades if t.notional is not None)

    def get_book_summary(self, book_id: str) -> Dict[str, any]:
        """Get summary statistics for a book.

        Parameters
        ----------
        book_id : str
            Book ID

        Returns
        -------
        dict
            Summary with keys: num_trades, total_mtm, total_notional, active_trades
        """
        trades = self.get_trades_by_book(book_id)
        active_trades = [t for t in trades if t.status == TradeStatus.ACTIVE]

        return {
            "book_id": book_id,
            "num_trades": len(trades),
            "active_trades": len(active_trades),
            "total_mtm": sum(t.get_mtm(default=0.0) for t in trades),
            "total_notional": sum(abs(t.notional) for t in trades if t.notional is not None),
        }

    def get_desk_summary(self, desk_id: str) -> Dict[str, any]:
        """Get summary statistics for a desk.

        Parameters
        ----------
        desk_id : str
            Desk ID

        Returns
        -------
        dict
            Summary with keys: num_trades, total_mtm, total_notional, active_trades, unique_books
        """
        trades = self.get_trades_by_desk(desk_id)
        active_trades = [t for t in trades if t.status == TradeStatus.ACTIVE]
        unique_books = set(t.book_id for t in trades if t.book_id)

        return {
            "desk_id": desk_id,
            "num_trades": len(trades),
            "active_trades": len(active_trades),
            "num_books": len(unique_books),
            "total_mtm": sum(t.get_mtm(default=0.0) for t in trades),
            "total_notional": sum(abs(t.notional) for t in trades if t.notional is not None),
        }

    def get_trader_summary(self, trader_id: str) -> Dict[str, any]:
        """Get summary statistics for a trader.

        Parameters
        ----------
        trader_id : str
            Trader ID

        Returns
        -------
        dict
            Summary with keys: num_trades, total_mtm, total_notional, active_trades, unique_books
        """
        trades = self.get_trades_by_trader(trader_id)
        active_trades = [t for t in trades if t.status == TradeStatus.ACTIVE]
        unique_books = set(t.book_id for t in trades if t.book_id)

        return {
            "trader_id": trader_id,
            "num_trades": len(trades),
            "active_trades": len(active_trades),
            "num_books": len(unique_books),
            "total_mtm": sum(t.get_mtm(default=0.0) for t in trades),
            "total_notional": sum(abs(t.notional) for t in trades if t.notional is not None),
        }

    def aggregate_by_book(self) -> Dict[str, Dict[str, any]]:
        """Aggregate portfolio metrics by book.

        Returns
        -------
        dict
            Map of book_id -> book summary
        """
        book_ids = set(t.book_id for t in self.trades.values() if t.book_id)
        return {book_id: self.get_book_summary(book_id) for book_id in book_ids}

    def aggregate_by_desk(self) -> Dict[str, Dict[str, any]]:
        """Aggregate portfolio metrics by desk.

        Returns
        -------
        dict
            Map of desk_id -> desk summary
        """
        desk_ids = set(t.desk_id for t in self.trades.values() if t.desk_id)
        return {desk_id: self.get_desk_summary(desk_id) for desk_id in desk_ids}

    def aggregate_by_trader(self) -> Dict[str, Dict[str, any]]:
        """Aggregate portfolio metrics by trader.

        Returns
        -------
        dict
            Map of trader_id -> trader summary
        """
        trader_ids = set(t.trader_id for t in self.trades.values() if t.trader_id)
        return {trader_id: self.get_trader_summary(trader_id) for trader_id in trader_ids}
