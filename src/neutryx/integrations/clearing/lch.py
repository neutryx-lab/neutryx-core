"""LCH SwapClear integration.

This module provides connectivity to LCH (London Clearing House) SwapClear,
the world's leading clearinghouse for interest rate swaps.

Features:
- Trade submission and confirmation
- Real-time margin calculations
- Position reporting
- Compression services
- Risk management

API Documentation: https://www.lch.com/services/swapclear
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

from .base import (
    CCPAuthenticationError,
    CCPConfig,
    CCPConnectionError,
    CCPConnector,
    CCPError,
    CCPMetrics,
    CCPTimeoutError,
    CCPTradeRejectionError,
    MarginCall,
    PositionReport,
    Trade,
    TradeStatus,
    TradeSubmissionResponse,
)


class LCHSwapClearConfig(CCPConfig):
    """LCH SwapClear-specific configuration."""

    clearing_service: str = Field(default="SwapClear", description="LCH clearing service")
    compression_enabled: bool = Field(default=False, description="Enable trade compression")
    account_structure: str = Field(default="omnibus", description="Account structure type")

    # SwapClear specific
    settlement_currency: str = Field(default="USD", description="Settlement currency")
    margin_method: str = Field(default="PAI", description="Margin method: PAI/PAIRS")

    class Config:
        """Pydantic config."""
        extra = "allow"


class LCHTradeDetails(BaseModel):
    """LCH-specific trade details."""

    usi: str = Field(..., description="Unique Swap Identifier")
    collateralization: str = Field(default="fully_collateralized")
    swap_stream_buyer: Dict[str, Any] = Field(..., description="Buyer swap stream")
    swap_stream_seller: Dict[str, Any] = Field(..., description="Seller swap stream")
    confirmation_method: str = Field(default="electronic")


class LCHSwapClearConnector(CCPConnector):
    """LCH SwapClear connectivity implementation.

    Provides full integration with LCH SwapClear including:
    - Trade submission via SwapClear API
    - Real-time trade status updates
    - Margin requirement calculations
    - Position and risk reporting
    - Trade compression workflows
    """

    def __init__(self, config: LCHSwapClearConfig):
        """Initialize LCH SwapClear connector.

        Args:
            config: LCH SwapClear configuration
        """
        super().__init__(config)
        self.config: LCHSwapClearConfig = config
        self.metrics = CCPMetrics()
        self._client: Optional[httpx.AsyncClient] = None
        self._compression_eligible_trades: List[str] = []

    async def connect(self) -> bool:
        """Establish connection to LCH SwapClear.

        Returns:
            True if connection successful

        Raises:
            CCPConnectionError: If connection fails
            CCPAuthenticationError: If authentication fails
        """
        try:
            # Initialize HTTP client with proper headers
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-LCH-Member-ID": self.config.member_id,
            }

            if self.config.api_key:
                headers["X-API-Key"] = self.config.api_key

            timeout = httpx.Timeout(self.config.timeout, connect=10.0)

            self._client = httpx.AsyncClient(
                base_url=self.config.api_endpoint,
                headers=headers,
                timeout=timeout,
            )

            # Perform authentication handshake
            auth_payload = {
                "member_id": self.config.member_id,
                "service": self.config.clearing_service,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Sign request if secret provided
            if self.config.api_secret:
                signature = self._sign_request(auth_payload)
                auth_payload["signature"] = signature

            response = await self._client.post("/api/v1/auth", json=auth_payload)

            if response.status_code == 200:
                auth_data = response.json()
                self._session_id = auth_data.get("session_id")
                self._connected = True
                self.metrics.last_heartbeat = datetime.utcnow()
                return True
            elif response.status_code == 401:
                raise CCPAuthenticationError("Invalid credentials for LCH SwapClear")
            else:
                raise CCPConnectionError(
                    f"Connection failed with status {response.status_code}: {response.text}"
                )

        except CCPAuthenticationError:
            # Re-raise authentication errors as-is
            raise
        except httpx.ConnectError as e:
            raise CCPConnectionError(f"Failed to connect to LCH SwapClear: {e}")
        except Exception as e:
            raise CCPConnectionError(f"Unexpected error during connection: {e}")

    async def disconnect(self) -> bool:
        """Disconnect from LCH SwapClear.

        Returns:
            True if disconnection successful
        """
        if self._client:
            await self._client.aclose()
            self._client = None
        self._connected = False
        self._session_id = None
        return True

    async def submit_trade(self, trade: Trade) -> TradeSubmissionResponse:
        """Submit trade to LCH SwapClear for clearing.

        Args:
            trade: Trade to submit

        Returns:
            Submission response with status

        Raises:
            CCPTradeRejectionError: If trade is rejected
            CCPConnectionError: If connection fails
            CCPTimeoutError: If request times out
        """
        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to LCH SwapClear")

        start_time = datetime.utcnow()

        try:
            # Convert trade to LCH format
            lch_trade = self._format_lch_trade(trade)

            # Submit trade
            response = await self._client.post(
                "/api/v1/trades/submit",
                json=lch_trade,
                headers={"X-Session-ID": self._session_id or ""},
            )

            response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            if response.status_code == 201:
                result = response.json()
                submission_response = TradeSubmissionResponse(
                    submission_id=result["submission_id"],
                    trade_id=trade.trade_id,
                    status=TradeStatus.ACCEPTED,
                    ccp_trade_id=result.get("lch_trade_id"),
                    metadata={
                        "clearing_house": "LCH",
                        "service": "SwapClear",
                        "compression_eligible": result.get("compression_eligible", False),
                    },
                )

                # Track compression-eligible trades
                if result.get("compression_eligible"):
                    self._compression_eligible_trades.append(trade.trade_id)

                self.metrics.record_submission(True, response_time_ms)
                return submission_response

            elif response.status_code == 400:
                error_data = response.json()
                rejection_code = error_data.get("error_code", "UNKNOWN")
                rejection_reason = error_data.get("message", "Trade validation failed")

                submission_response = TradeSubmissionResponse(
                    submission_id=str(uuid.uuid4()),
                    trade_id=trade.trade_id,
                    status=TradeStatus.REJECTED,
                    rejection_reason=rejection_reason,
                    rejection_code=rejection_code,
                )

                self.metrics.record_submission(False, response_time_ms, rejected=True)

                raise CCPTradeRejectionError(
                    f"LCH rejected trade: {rejection_reason}",
                    rejection_code=rejection_code
                )

            else:
                self.metrics.record_submission(False, response_time_ms)
                raise CCPError(
                    f"Unexpected response from LCH: {response.status_code} - {response.text}"
                )

        except httpx.TimeoutException:
            raise CCPTimeoutError("Trade submission to LCH timed out")
        except CCPTradeRejectionError:
            raise
        except Exception as e:
            raise CCPError(f"Failed to submit trade to LCH: {e}")

    async def get_trade_status(self, trade_id: str) -> TradeStatus:
        """Get current status of a trade.

        Args:
            trade_id: Trade identifier

        Returns:
            Current trade status
        """
        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to LCH SwapClear")

        try:
            response = await self._client.get(
                f"/api/v1/trades/{trade_id}/status",
                headers={"X-Session-ID": self._session_id or ""},
            )

            if response.status_code == 200:
                status_data = response.json()
                return TradeStatus(status_data["status"])
            else:
                raise CCPError(f"Failed to get trade status: {response.text}")

        except Exception as e:
            raise CCPError(f"Error getting trade status: {e}")

    async def cancel_trade(self, trade_id: str) -> bool:
        """Cancel a pending trade.

        Args:
            trade_id: Trade identifier

        Returns:
            True if cancellation successful
        """
        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to LCH SwapClear")

        try:
            response = await self._client.delete(
                f"/api/v1/trades/{trade_id}",
                headers={"X-Session-ID": self._session_id or ""},
            )

            return response.status_code == 200

        except Exception as e:
            raise CCPError(f"Error cancelling trade: {e}")

    async def get_margin_requirements(
        self,
        member_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get current margin requirements from LCH.

        Args:
            member_id: Member ID (defaults to configured member)

        Returns:
            Dictionary with margin requirements including IM, VM, and add-ons
        """
        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to LCH SwapClear")

        member_id = member_id or self.config.member_id

        try:
            response = await self._client.get(
                f"/api/v1/margin/requirements",
                params={"member_id": member_id},
                headers={"X-Session-ID": self._session_id or ""},
            )

            if response.status_code == 200:
                return response.json()
            else:
                raise CCPError(f"Failed to get margin requirements: {response.text}")

        except Exception as e:
            raise CCPError(f"Error getting margin requirements: {e}")

    async def get_position_report(
        self,
        as_of_date: Optional[datetime] = None
    ) -> PositionReport:
        """Get position report from LCH.

        Args:
            as_of_date: Report date (defaults to today)

        Returns:
            Position report with portfolio details
        """
        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to LCH SwapClear")

        as_of_date = as_of_date or datetime.utcnow()

        try:
            response = await self._client.get(
                f"/api/v1/positions/report",
                params={
                    "member_id": self.config.member_id,
                    "as_of_date": as_of_date.strftime("%Y-%m-%d"),
                },
                headers={"X-Session-ID": self._session_id or ""},
            )

            if response.status_code == 200:
                data = response.json()
                return PositionReport(
                    report_id=data["report_id"],
                    member_id=self.config.member_id,
                    as_of_date=as_of_date,
                    positions=data["positions"],
                    total_exposure=Decimal(str(data.get("total_exposure", 0))),
                    initial_margin=Decimal(str(data.get("initial_margin", 0))),
                    variation_margin=Decimal(str(data.get("variation_margin", 0))),
                    metadata={
                        "clearing_house": "LCH",
                        "service": "SwapClear",
                        "compression_opportunities": len(self._compression_eligible_trades),
                    },
                )
            else:
                raise CCPError(f"Failed to get position report: {response.text}")

        except Exception as e:
            raise CCPError(f"Error getting position report: {e}")

    async def healthcheck(self) -> bool:
        """Check LCH SwapClear connectivity health.

        Returns:
            True if connection healthy
        """
        if not self._connected or not self._client:
            return False

        try:
            response = await self._client.get(
                "/api/v1/health",
                timeout=5.0,
            )
            if response.status_code == 200:
                self.metrics.last_heartbeat = datetime.utcnow()
                return True
            return False
        except Exception:
            return False

    async def request_compression(
        self,
        trade_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Request portfolio compression for eligible trades.

        Args:
            trade_ids: Specific trade IDs to compress (None for all eligible)

        Returns:
            Compression results
        """
        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to LCH SwapClear")

        trade_ids = trade_ids or self._compression_eligible_trades

        try:
            response = await self._client.post(
                "/api/v1/compression/request",
                json={
                    "member_id": self.config.member_id,
                    "trade_ids": trade_ids,
                    "method": "triOptima",
                },
                headers={"X-Session-ID": self._session_id or ""},
            )

            if response.status_code == 200:
                return response.json()
            else:
                raise CCPError(f"Compression request failed: {response.text}")

        except Exception as e:
            raise CCPError(f"Error requesting compression: {e}")

    def _format_lch_trade(self, trade: Trade) -> Dict[str, Any]:
        """Convert Neutryx trade to LCH format.

        Args:
            trade: Neutryx trade object

        Returns:
            LCH-formatted trade dictionary
        """
        return {
            "usi": trade.uti or str(uuid.uuid4()),
            "product_type": trade.product_type.value,
            "trade_date": trade.trade_date.isoformat(),
            "effective_date": trade.effective_date.isoformat(),
            "maturity_date": trade.maturity_date.isoformat(),
            "notional": {
                "amount": float(trade.economics.notional),
                "currency": trade.economics.currency,
            },
            "fixed_rate": float(trade.economics.fixed_rate) if trade.economics.fixed_rate else None,
            "buyer": {
                "party_id": trade.buyer.party_id,
                "lei": trade.buyer.lei,
                "member_id": trade.buyer.member_id,
            },
            "seller": {
                "party_id": trade.seller.party_id,
                "lei": trade.seller.lei,
                "member_id": trade.seller.member_id,
            },
            "clearing_broker": trade.clearing_broker,
            "settlement_currency": self.config.settlement_currency,
            "collateral_currency": trade.collateral_currency or "USD",
            "compression_eligible": self.config.compression_enabled,
            "margin_method": self.config.margin_method,
        }

    def _sign_request(self, payload: Dict[str, Any]) -> str:
        """Sign request with HMAC-SHA256.

        Args:
            payload: Request payload

        Returns:
            Base64-encoded signature
        """
        message = json.dumps(payload, sort_keys=True)
        signature = hmac.new(
            self.config.api_secret.encode() if self.config.api_secret else b"",
            message.encode(),
            hashlib.sha256,
        ).hexdigest()
        return signature


__all__ = [
    "LCHSwapClearConnector",
    "LCHSwapClearConfig",
    "LCHTradeDetails",
]
