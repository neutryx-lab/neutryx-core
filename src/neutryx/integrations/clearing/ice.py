"""ICE Clear integration (Credit and Europe).

This module provides connectivity to ICE Clear Credit and ICE Clear Europe,
specializing in credit derivatives, European interest rate derivatives, and energy.

Features:
- ICE Clear Credit: CDS index and single-name clearing
- ICE Clear Europe: Energy, FX, and European IRS
- Real-time margin calculations
- Portfolio compression
- Integration with ICE Trade Vault

API Documentation: https://www.theice.com/clear
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field

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


class ICEClearService(str, Enum):
    """ICE Clear services."""
    CREDIT = "ICE_CLEAR_CREDIT"
    EUROPE = "ICE_CLEAR_EUROPE"
    US = "ICE_CLEAR_US"
    SINGAPORE = "ICE_CLEAR_SINGAPORE"


class ICEClearConfig(CCPConfig):
    """ICE Clear-specific configuration."""

    service: ICEClearService = Field(..., description="ICE Clear service")
    clearing_member_code: str = Field(..., description="ICE clearing member code")
    trade_source: str = Field(default="API", description="Trade source identifier")

    # ICE specific
    compression_enabled: bool = Field(default=True, description="Enable compression services")
    trade_vault_enabled: bool = Field(default=True, description="Enable ICE Trade Vault")
    margin_model: str = Field(default="ICE_MARGIN", description="Margin model")

    model_config = ConfigDict(extra="allow")


class ICECreditProduct(str, Enum):
    """ICE Clear Credit product types."""
    CDS_INDEX = "cds_index"
    CDS_SINGLE_NAME = "cds_single_name"
    CDS_TRANCHE = "cds_tranche"
    ITRAXX = "itraxx"
    CDX = "cdx"


class ICEMarginBreakdown(BaseModel):
    """ICE margin calculation breakdown."""

    initial_margin: Decimal = Field(..., description="Initial margin requirement")
    concentration_margin: Decimal = Field(default=Decimal("0"), description="Concentration add-on")
    jump_to_default: Decimal = Field(default=Decimal("0"), description="Jump-to-default charge")
    liquidity_premium: Decimal = Field(default=Decimal("0"), description="Liquidity premium")
    total_margin: Decimal = Field(..., description="Total margin requirement")


class ICEClearConnector(CCPConnector):
    """ICE Clear connectivity implementation.

    Provides full integration with ICE Clear Credit and ICE Clear Europe:
    - Multi-service support (Credit, Europe, US, Singapore)
    - CDS and IRS clearing
    - Real-time margin calculations
    - Portfolio compression via triReduce
    - ICE Trade Vault integration for trade reporting
    """

    def __init__(self, config: ICEClearConfig):
        """Initialize ICE Clear connector.

        Args:
            config: ICE Clear configuration
        """
        super().__init__(config)
        self.config: ICEClearConfig = config
        self.metrics = CCPMetrics()
        self._client: Optional[httpx.AsyncClient] = None
        self._compression_cycles: List[str] = []

    async def connect(self) -> bool:
        """Establish connection to ICE Clear.

        Returns:
            True if connection successful

        Raises:
            CCPConnectionError: If connection fails
            CCPAuthenticationError: If authentication fails
        """
        try:
            # Initialize HTTP client
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-ICE-Service": self.config.service.value,
                "X-ICE-Member": self.config.clearing_member_code,
            }

            if self.config.api_key:
                headers["X-API-Key"] = self.config.api_key

            timeout = httpx.Timeout(self.config.timeout, connect=10.0)

            self._client = httpx.AsyncClient(
                base_url=self.config.api_endpoint,
                headers=headers,
                timeout=timeout,
            )

            # Authenticate
            auth_payload = {
                "service": self.config.service.value,
                "member_code": self.config.clearing_member_code,
                "member_id": self.config.member_id,
                "timestamp": datetime.utcnow().isoformat(),
            }

            response = await self._client.post("/api/v3/auth/login", json=auth_payload)

            if response.status_code == 200:
                auth_data = response.json()
                self._session_id = auth_data.get("access_token")
                self._connected = True
                self.metrics.last_heartbeat = datetime.utcnow()

                # Subscribe to compression cycles if enabled
                if self.config.compression_enabled:
                    await self._subscribe_compression_cycles()

                return True
            elif response.status_code == 401:
                raise CCPAuthenticationError("Invalid credentials for ICE Clear")
            else:
                raise CCPConnectionError(
                    f"Connection failed with status {response.status_code}: {response.text}"
                )

        except httpx.ConnectError as e:
            raise CCPConnectionError(f"Failed to connect to ICE Clear: {e}")
        except Exception as e:
            raise CCPConnectionError(f"Unexpected error during connection: {e}")

    async def disconnect(self) -> bool:
        """Disconnect from ICE Clear.

        Returns:
            True if disconnection successful
        """
        if self._client:
            # Logout
            try:
                await self._client.post(
                    "/api/v3/auth/logout",
                    headers={"Authorization": f"Bearer {self._session_id}"},
                )
            except Exception:
                pass
            await self._client.aclose()
            self._client = None

        self._connected = False
        self._session_id = None
        self._compression_cycles.clear()
        return True

    async def submit_trade(self, trade: Trade) -> TradeSubmissionResponse:
        """Submit trade to ICE Clear.

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
            raise CCPConnectionError("Not connected to ICE Clear")

        start_time = datetime.utcnow()

        try:
            # Convert trade to ICE format
            ice_trade = self._format_ice_trade(trade)

            # Submit trade
            response = await self._client.post(
                "/api/v3/trades/submit",
                json=ice_trade,
                headers={"Authorization": f"Bearer {self._session_id}"},
            )

            response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            if response.status_code == 201:
                result = response.json()

                # Calculate initial margin
                margin_breakdown = None
                if result.get("margin_calculated"):
                    margin_breakdown = ICEMarginBreakdown(
                        initial_margin=Decimal(str(result["margin"]["initial_margin"])),
                        concentration_margin=Decimal(str(result["margin"].get("concentration", 0))),
                        jump_to_default=Decimal(str(result["margin"].get("jump_to_default", 0))),
                        liquidity_premium=Decimal(str(result["margin"].get("liquidity_premium", 0))),
                        total_margin=Decimal(str(result["margin"]["total"])),
                    )

                submission_response = TradeSubmissionResponse(
                    submission_id=result["submission_id"],
                    trade_id=trade.trade_id,
                    status=TradeStatus.ACCEPTED,
                    ccp_trade_id=result.get("ice_trade_id"),
                    metadata={
                        "clearing_house": "ICE",
                        "service": self.config.service.value,
                        "clearing_member": self.config.clearing_member_code,
                        "margin_breakdown": margin_breakdown.dict() if margin_breakdown else None,
                        "trade_vault_registered": result.get("trade_vault_registered", False),
                        "compression_eligible": result.get("compression_eligible", False),
                    },
                )

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
                    f"ICE rejected trade: {rejection_reason}",
                    rejection_code=rejection_code
                )

            else:
                self.metrics.record_submission(False, response_time_ms)
                raise CCPError(
                    f"Unexpected response from ICE: {response.status_code} - {response.text}"
                )

        except httpx.TimeoutException:
            raise CCPTimeoutError("Trade submission to ICE timed out")
        except CCPTradeRejectionError:
            raise
        except Exception as e:
            raise CCPError(f"Failed to submit trade to ICE: {e}")

    async def get_trade_status(self, trade_id: str) -> TradeStatus:
        """Get current status of a trade.

        Args:
            trade_id: Trade identifier

        Returns:
            Current trade status
        """
        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to ICE Clear")

        try:
            response = await self._client.get(
                f"/api/v3/trades/{trade_id}/status",
                headers={"Authorization": f"Bearer {self._session_id}"},
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
            raise CCPConnectionError("Not connected to ICE Clear")

        try:
            response = await self._client.delete(
                f"/api/v3/trades/{trade_id}",
                headers={"Authorization": f"Bearer {self._session_id}"},
            )

            return response.status_code == 200

        except Exception as e:
            raise CCPError(f"Error cancelling trade: {e}")

    async def get_margin_requirements(
        self,
        member_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get current margin requirements from ICE.

        Args:
            member_id: Member ID (defaults to configured member)

        Returns:
            Dictionary with margin requirements and breakdown
        """
        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to ICE Clear")

        member_id = member_id or self.config.member_id

        try:
            response = await self._client.get(
                f"/api/v3/margin/requirements",
                params={
                    "member_code": self.config.clearing_member_code,
                    "member_id": member_id,
                },
                headers={"Authorization": f"Bearer {self._session_id}"},
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
        """Get position report from ICE.

        Args:
            as_of_date: Report date (defaults to today)

        Returns:
            Position report with portfolio details
        """
        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to ICE Clear")

        as_of_date = as_of_date or datetime.utcnow()

        try:
            response = await self._client.get(
                f"/api/v3/positions/report",
                params={
                    "member_code": self.config.clearing_member_code,
                    "as_of_date": as_of_date.strftime("%Y-%m-%d"),
                },
                headers={"Authorization": f"Bearer {self._session_id}"},
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
                        "clearing_house": "ICE",
                        "service": self.config.service.value,
                        "clearing_member": self.config.clearing_member_code,
                        "compression_cycles": len(self._compression_cycles),
                    },
                )
            else:
                raise CCPError(f"Failed to get position report: {response.text}")

        except Exception as e:
            raise CCPError(f"Error getting position report: {e}")

    async def healthcheck(self) -> bool:
        """Check ICE Clear connectivity health.

        Returns:
            True if connection healthy
        """
        if not self._connected or not self._client:
            return False

        try:
            response = await self._client.get(
                "/api/v3/health",
                timeout=5.0,
            )
            if response.status_code == 200:
                self.metrics.last_heartbeat = datetime.utcnow()
                return True
            return False
        except Exception:
            return False

    async def submit_to_compression(
        self,
        trade_ids: List[str],
        cycle_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Submit trades to ICE triReduce compression.

        Args:
            trade_ids: Trade IDs to include in compression
            cycle_id: Specific compression cycle (None for next available)

        Returns:
            Compression submission results
        """
        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to ICE Clear")

        try:
            payload = {
                "member_code": self.config.clearing_member_code,
                "trade_ids": trade_ids,
                "cycle_id": cycle_id or "AUTO",
                "compression_method": "triReduce",
            }

            response = await self._client.post(
                "/api/v3/compression/submit",
                json=payload,
                headers={"Authorization": f"Bearer {self._session_id}"},
            )

            if response.status_code == 200:
                return response.json()
            else:
                raise CCPError(f"Compression submission failed: {response.text}")

        except Exception as e:
            raise CCPError(f"Error submitting to compression: {e}")

    async def register_with_trade_vault(self, trade: Trade) -> bool:
        """Register trade with ICE Trade Vault for reporting.

        Args:
            trade: Trade to register

        Returns:
            True if registration successful
        """
        if not self.config.trade_vault_enabled:
            return False

        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to ICE Clear")

        try:
            vault_payload = {
                "trade_id": trade.trade_id,
                "uti": trade.uti,
                "trade_date": trade.trade_date.isoformat(),
                "product_type": trade.product_type.value,
                "counterparties": [
                    {"party_id": trade.buyer.party_id, "lei": trade.buyer.lei},
                    {"party_id": trade.seller.party_id, "lei": trade.seller.lei},
                ],
                "economics": trade.economics.to_dict(),
            }

            response = await self._client.post(
                "/api/v3/trade-vault/register",
                json=vault_payload,
                headers={"Authorization": f"Bearer {self._session_id}"},
            )

            return response.status_code == 200

        except Exception:
            return False

    async def _subscribe_compression_cycles(self) -> None:
        """Subscribe to compression cycle notifications."""
        if not self._connected or not self._client:
            return

        try:
            response = await self._client.get(
                "/api/v3/compression/cycles",
                headers={"Authorization": f"Bearer {self._session_id}"},
            )

            if response.status_code == 200:
                data = response.json()
                self._compression_cycles = [c["cycle_id"] for c in data.get("cycles", [])]

        except Exception:
            pass

    def _format_ice_trade(self, trade: Trade) -> Dict[str, Any]:
        """Convert Neutryx trade to ICE format.

        Args:
            trade: Neutryx trade object

        Returns:
            ICE-formatted trade dictionary
        """
        return {
            "trade_id": trade.trade_id,
            "uti": trade.uti or str(uuid.uuid4()),
            "product_type": trade.product_type.value,
            "trade_date": trade.trade_date.strftime("%Y-%m-%d"),
            "effective_date": trade.effective_date.strftime("%Y-%m-%d"),
            "maturity_date": trade.maturity_date.strftime("%Y-%m-%d"),
            "notional": {
                "amount": float(trade.economics.notional),
                "currency": trade.economics.currency,
            },
            "fixed_rate": float(trade.economics.fixed_rate) if trade.economics.fixed_rate else None,
            "spread": float(trade.economics.spread) if trade.economics.spread else None,
            "buyer": {
                "party_id": trade.buyer.party_id,
                "lei": trade.buyer.lei,
                "member_code": self.config.clearing_member_code,
            },
            "seller": {
                "party_id": trade.seller.party_id,
                "lei": trade.seller.lei,
                "member_code": self.config.clearing_member_code,
            },
            "trade_source": self.config.trade_source,
            "margin_model": self.config.margin_model,
            "compression_eligible": self.config.compression_enabled,
            "trade_vault_enabled": self.config.trade_vault_enabled,
        }


__all__ = [
    "ICEClearConnector",
    "ICEClearConfig",
    "ICEClearService",
    "ICECreditProduct",
    "ICEMarginBreakdown",
]
