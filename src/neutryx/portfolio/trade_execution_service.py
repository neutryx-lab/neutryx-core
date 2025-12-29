"""Trade execution service for managing trade lifecycle and settlement.

This module provides a comprehensive trade execution service that:
- Validates trades before execution
- Manages trade lifecycle (booking, confirmation, settlement)
- Validates counterparty and CSA relationships
- Coordinates with repositories for persistence
- Provides trade booking and confirmation workflows
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from neutryx.portfolio.contracts.trade import Trade, TradeStatus, ProductType
from neutryx.portfolio.contracts.counterparty import Counterparty
from neutryx.portfolio.contracts.csa import CSA
from neutryx.portfolio.bank_connection_manager import BankConnectionManager
from neutryx.portfolio.id_generator import TradeIdGenerator


class ExecutionStatus(str, Enum):
    """Status of trade execution."""

    SUCCESS = "Success"
    FAILED = "Failed"
    PENDING = "Pending"
    REJECTED = "Rejected"


class ExecutionError(str, Enum):
    """Types of execution errors."""

    COUNTERPARTY_NOT_FOUND = "CounterpartyNotFound"
    NO_CSA_AGREEMENT = "NoCSAgreement"
    INVALID_TRADE_DATA = "InvalidTradeData"
    BOOK_NOT_FOUND = "BookNotFound"
    DUPLICATE_TRADE_ID = "DuplicateTradeID"
    DATABASE_ERROR = "DatabaseError"
    VALIDATION_ERROR = "ValidationError"


class ExecutionResult(BaseModel):
    """Result of trade execution.

    Attributes
    ----------
    status : ExecutionStatus
        Overall status of execution
    trade_id : str, optional
        ID of the executed trade
    error : ExecutionError, optional
        Error type if execution failed
    error_message : str, optional
        Detailed error message
    timestamp : datetime
        Timestamp of execution attempt
    """

    status: ExecutionStatus
    trade_id: Optional[str] = None
    error: Optional[ExecutionError] = None
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ExecutionStatus.SUCCESS

    def is_failed(self) -> bool:
        """Check if execution failed."""
        return self.status in (ExecutionStatus.FAILED, ExecutionStatus.REJECTED)


class TradeExecutionService:
    """Service for executing and managing trades.

    This service provides:
    - Trade validation and execution
    - Counterparty and CSA verification
    - Trade lifecycle management
    - Integration with repository layer

    Attributes
    ----------
    manager : BankConnectionManager
        Connection manager for repository access
    id_generator : TradeIdGenerator
        Generator for trade IDs
    """

    def __init__(self, manager: BankConnectionManager):
        """Initialize trade execution service.

        Parameters
        ----------
        manager : BankConnectionManager
            Connection manager with configured repositories
        """
        self.manager = manager
        self.id_generator = TradeIdGenerator()

    async def execute_trade(
        self,
        trade: Trade,
        validate_counterparty: bool = True,
        validate_csa: bool = False,
        auto_confirm: bool = False,
    ) -> ExecutionResult:
        """Execute a trade with validation and persistence.

        Parameters
        ----------
        trade : Trade
            Trade to execute
        validate_counterparty : bool
            Whether to validate counterparty exists (default: True)
        validate_csa : bool
            Whether to validate CSA agreement exists (default: False)
        auto_confirm : bool
            Whether to automatically confirm the trade (default: False)

        Returns
        -------
        ExecutionResult
            Result of trade execution
        """
        try:
            await self.manager.connect()

            # Validate counterparty
            if validate_counterparty:
                counterparty = await self.manager.counterparty_repo.find_by_id_async(
                    trade.counterparty_id
                )
                if not counterparty:
                    return ExecutionResult(
                        status=ExecutionStatus.REJECTED,
                        trade_id=trade.id,
                        error=ExecutionError.COUNTERPARTY_NOT_FOUND,
                        error_message=f"Counterparty '{trade.counterparty_id}' not found",
                    )

            # Validate CSA agreement if required
            if validate_csa:
                # Assuming our bank is the other party, we need to find CSA
                # This would need the bank's ID configured in the service
                csas = await self.manager.csa_repo.find_by_counterparty_async(
                    trade.counterparty_id
                )
                if not csas:
                    return ExecutionResult(
                        status=ExecutionStatus.REJECTED,
                        trade_id=trade.id,
                        error=ExecutionError.NO_CSA_AGREEMENT,
                        error_message=f"No CSA agreement found for counterparty '{trade.counterparty_id}'",
                    )

            # Check for duplicate trade ID
            existing_trade = await self.manager.trade_repo.find_by_id_async(trade.id)
            if existing_trade:
                return ExecutionResult(
                    status=ExecutionStatus.REJECTED,
                    trade_id=trade.id,
                    error=ExecutionError.DUPLICATE_TRADE_ID,
                    error_message=f"Trade with ID '{trade.id}' already exists",
                )

            # Auto-confirm if requested
            if auto_confirm and trade.status == TradeStatus.PENDING:
                trade.status = TradeStatus.ACTIVE

            # Save trade
            await self.manager.trade_repo.save_async(trade)

            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                trade_id=trade.id,
            )

        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                trade_id=trade.id,
                error=ExecutionError.DATABASE_ERROR,
                error_message=str(e),
            )

    async def confirm_trade(self, trade_id: str) -> ExecutionResult:
        """Confirm a pending trade.

        Parameters
        ----------
        trade_id : str
            ID of trade to confirm

        Returns
        -------
        ExecutionResult
            Result of confirmation
        """
        try:
            await self.manager.connect()

            trade = await self.manager.trade_repo.find_by_id_async(trade_id)
            if not trade:
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    trade_id=trade_id,
                    error=ExecutionError.INVALID_TRADE_DATA,
                    error_message=f"Trade '{trade_id}' not found",
                )

            if trade.status != TradeStatus.PENDING:
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    trade_id=trade_id,
                    error=ExecutionError.VALIDATION_ERROR,
                    error_message=f"Trade is not pending (status: {trade.status.value})",
                )

            # Update status to active
            trade.status = TradeStatus.ACTIVE
            await self.manager.trade_repo.save_async(trade)

            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                trade_id=trade_id,
            )

        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                trade_id=trade_id,
                error=ExecutionError.DATABASE_ERROR,
                error_message=str(e),
            )

    async def cancel_trade(self, trade_id: str) -> ExecutionResult:
        """Cancel a pending trade.

        Parameters
        ----------
        trade_id : str
            ID of trade to cancel

        Returns
        -------
        ExecutionResult
            Result of cancellation
        """
        try:
            await self.manager.connect()

            trade = await self.manager.trade_repo.find_by_id_async(trade_id)
            if not trade:
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    trade_id=trade_id,
                    error=ExecutionError.INVALID_TRADE_DATA,
                    error_message=f"Trade '{trade_id}' not found",
                )

            if trade.status not in (TradeStatus.PENDING, TradeStatus.ACTIVE):
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    trade_id=trade_id,
                    error=ExecutionError.VALIDATION_ERROR,
                    error_message=f"Trade cannot be cancelled (status: {trade.status.value})",
                )

            # Update status to cancelled
            trade.status = TradeStatus.CANCELLED
            await self.manager.trade_repo.save_async(trade)

            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                trade_id=trade_id,
            )

        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                trade_id=trade_id,
                error=ExecutionError.DATABASE_ERROR,
                error_message=str(e),
            )

    async def terminate_trade(self, trade_id: str, termination_date: date) -> ExecutionResult:
        """Terminate an active trade early.

        Parameters
        ----------
        trade_id : str
            ID of trade to terminate
        termination_date : date
            Date of termination

        Returns
        -------
        ExecutionResult
            Result of termination
        """
        try:
            await self.manager.connect()

            trade = await self.manager.trade_repo.find_by_id_async(trade_id)
            if not trade:
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    trade_id=trade_id,
                    error=ExecutionError.INVALID_TRADE_DATA,
                    error_message=f"Trade '{trade_id}' not found",
                )

            if trade.status != TradeStatus.ACTIVE:
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    trade_id=trade_id,
                    error=ExecutionError.VALIDATION_ERROR,
                    error_message=f"Trade is not active (status: {trade.status.value})",
                )

            # Update status to terminated
            trade.status = TradeStatus.TERMINATED
            # Optionally update maturity date to termination date
            trade.maturity_date = termination_date
            await self.manager.trade_repo.save_async(trade)

            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                trade_id=trade_id,
            )

        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                trade_id=trade_id,
                error=ExecutionError.DATABASE_ERROR,
                error_message=str(e),
            )

    async def book_trade(
        self,
        counterparty_id: str,
        product_type: ProductType,
        trade_date: date,
        notional: float,
        currency: str,
        maturity_date: Optional[date] = None,
        book_id: Optional[str] = None,
        product_details: Optional[Dict] = None,
        validate_counterparty: bool = True,
        validate_csa: bool = False,
    ) -> ExecutionResult:
        """Book a new trade with automatic ID generation.

        Parameters
        ----------
        counterparty_id : str
            Counterparty ID
        product_type : ProductType
            Type of product
        trade_date : date
            Trade execution date
        notional : float
            Notional amount
        currency : str
            Currency code
        maturity_date : date, optional
            Maturity date
        book_id : str, optional
            Book ID for trade assignment
        product_details : dict, optional
            Product-specific details
        validate_counterparty : bool
            Whether to validate counterparty exists
        validate_csa : bool
            Whether to validate CSA agreement exists

        Returns
        -------
        ExecutionResult
            Result of trade booking
        """
        # Generate trade ID
        trade_id = self.id_generator.generate()

        # Create trade
        trade = Trade(
            id=trade_id,
            counterparty_id=counterparty_id,
            product_type=product_type,
            trade_date=trade_date,
            notional=notional,
            currency=currency,
            maturity_date=maturity_date,
            book_id=book_id,
            status=TradeStatus.PENDING,
            product_details=product_details,
        )

        # Execute trade
        return await self.execute_trade(
            trade=trade,
            validate_counterparty=validate_counterparty,
            validate_csa=validate_csa,
            auto_confirm=False,
        )

    async def get_counterparty_exposure(
        self, counterparty_id: str, as_of_date: Optional[date] = None
    ) -> Dict:
        """Get exposure summary for a counterparty.

        Parameters
        ----------
        counterparty_id : str
            Counterparty ID
        as_of_date : date, optional
            Valuation date (defaults to today)

        Returns
        -------
        dict
            Exposure summary containing:
            - counterparty: Counterparty object
            - active_trades: List of active trades
            - total_mtm: Total mark-to-market
            - trade_count: Number of active trades
            - csas: List of CSA agreements
        """
        await self.manager.connect()

        as_of = as_of_date or date.today()

        # Get counterparty and trades
        data = await self.manager.get_counterparty_trades_with_csa(counterparty_id)

        # Filter active trades
        active_trades = [
            t for t in data["trades"]
            if t.status == TradeStatus.ACTIVE and not t.is_expired(as_of)
        ]

        # Calculate total MTM
        total_mtm = sum(t.get_mtm(0.0) for t in active_trades)

        return {
            "counterparty": data["counterparty"],
            "active_trades": active_trades,
            "total_mtm": total_mtm,
            "trade_count": len(active_trades),
            "csas": data["csas"],
        }

    async def batch_execute_trades(
        self,
        trades: List[Trade],
        validate_counterparty: bool = True,
        validate_csa: bool = False,
        stop_on_error: bool = False,
    ) -> List[ExecutionResult]:
        """Execute multiple trades in batch.

        Parameters
        ----------
        trades : list[Trade]
            List of trades to execute
        validate_counterparty : bool
            Whether to validate counterparties
        validate_csa : bool
            Whether to validate CSA agreements
        stop_on_error : bool
            Whether to stop on first error

        Returns
        -------
        list[ExecutionResult]
            List of execution results
        """
        results = []

        for trade in trades:
            result = await self.execute_trade(
                trade=trade,
                validate_counterparty=validate_counterparty,
                validate_csa=validate_csa,
            )
            results.append(result)

            if stop_on_error and result.is_failed():
                break

        return results

    async def get_pending_trades(self) -> List[Trade]:
        """Get all pending trades awaiting confirmation.

        Returns
        -------
        list[Trade]
            List of pending trades
        """
        await self.manager.connect()
        return await self.manager.trade_repo.find_by_status_async(TradeStatus.PENDING)

    async def get_active_trades_by_counterparty(self, counterparty_id: str) -> List[Trade]:
        """Get all active trades for a counterparty.

        Parameters
        ----------
        counterparty_id : str
            Counterparty ID

        Returns
        -------
        list[Trade]
            List of active trades
        """
        await self.manager.connect()

        all_trades = await self.manager.trade_repo.find_by_counterparty_async(counterparty_id)
        return [t for t in all_trades if t.status == TradeStatus.ACTIVE]


__all__ = [
    "TradeExecutionService",
    "ExecutionStatus",
    "ExecutionError",
    "ExecutionResult",
]
