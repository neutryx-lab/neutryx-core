"""Eurex Clearing integration.

This module provides connectivity to Eurex Clearing, Europe's leading clearing house
for derivatives and repo markets.

Features:
- Multi-asset clearing (equity, fixed income, derivatives, repos)
- Prisma margin methodology
- Cross-margining across asset classes
- Integration with Eurex T7 trading platform
- C7 clearing system connectivity

API Documentation: https://www.eurex.com/ex-en/clear
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


class EurexAssetClass(str, Enum):
    """Eurex clearing asset classes."""
    EQUITY_DERIVATIVES = "equity_derivatives"
    FIXED_INCOME_DERIVATIVES = "fixed_income_derivatives"
    REPO_GC_POOLING = "repo_gc_pooling"
    ETF = "etf"
    BONDS = "bonds"


class EurexClearingConfig(CCPConfig):
    """Eurex Clearing-specific configuration."""

    clearing_member_id: str = Field(..., description="Eurex clearing member ID")
    participant_code: str = Field(..., description="Participant code")
    asset_class: EurexAssetClass = Field(..., description="Primary asset class")

    # Eurex specific
    prisma_enabled: bool = Field(default=True, description="Enable Prisma margin calculation")
    cross_margining_enabled: bool = Field(default=True, description="Enable cross-margining")
    c7_connectivity: bool = Field(default=True, description="Use C7 clearing system")
    t7_integration: bool = Field(default=False, description="Integrate with T7 trading")

    model_config = ConfigDict(extra="allow")


class PrismaMarginBreakdown(BaseModel):
    """Prisma margin methodology breakdown."""

    core_margin: Decimal = Field(..., description="Core margin requirement")
    add_on_margin: Decimal = Field(default=Decimal("0"), description="Add-on margin")
    concentration_charge: Decimal = Field(default=Decimal("0"), description="Concentration charge")
    liquidity_add_on: Decimal = Field(default=Decimal("0"), description="Liquidity add-on")
    cross_margining_benefit: Decimal = Field(default=Decimal("0"), description="Cross-margining offset")
    total_margin: Decimal = Field(..., description="Total margin requirement")


class EurexClearingConnector(CCPConnector):
    """Eurex Clearing connectivity implementation.

    Provides full integration with Eurex Clearing including:
    - Multi-asset class clearing
    - Prisma margin methodology
    - C7 clearing system connectivity
    - Cross-margining benefits
    - Integration with T7 trading platform
    - GC Pooling for repo markets
    """

    def __init__(self, config: EurexClearingConfig):
        """Initialize Eurex Clearing connector.

        Args:
            config: Eurex Clearing configuration
        """
        super().__init__(config)
        self.config: EurexClearingConfig = config
        self.metrics = CCPMetrics()
        self._client: Optional[httpx.AsyncClient] = None
        self._prisma_parameters: Dict[str, Any] = {}
        self._cross_margin_groups: List[str] = []

    async def connect(self) -> bool:
        """Establish connection to Eurex Clearing.

        Returns:
            True if connection successful

        Raises:
            CCPConnectionError: If connection fails
            CCPAuthenticationError: If authentication fails
        """
        try:
            # Initialize HTTP client with Eurex-specific headers
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-Eurex-Member-ID": self.config.clearing_member_id,
                "X-Eurex-Participant": self.config.participant_code,
            }

            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            timeout = httpx.Timeout(self.config.timeout, connect=10.0)

            # Use C7 or REST endpoint based on configuration
            base_url = self.config.api_endpoint
            if self.config.c7_connectivity:
                base_url = base_url.replace("/api", "/c7/api")

            self._client = httpx.AsyncClient(
                base_url=base_url,
                headers=headers,
                timeout=timeout,
            )

            # Authenticate with Eurex
            auth_payload = {
                "member_id": self.config.clearing_member_id,
                "participant_code": self.config.participant_code,
                "asset_class": self.config.asset_class.value,
                "timestamp": datetime.utcnow().isoformat(),
            }

            response = await self._client.post("/api/v2/authentication", json=auth_payload)

            if response.status_code == 200:
                auth_data = response.json()
                self._session_id = auth_data.get("session_token")
                self._connected = True
                self.metrics.last_heartbeat = datetime.utcnow()

                # Load Prisma parameters if enabled
                if self.config.prisma_enabled:
                    await self._load_prisma_parameters()

                # Initialize cross-margining groups if enabled
                if self.config.cross_margining_enabled:
                    await self._load_cross_margin_groups()

                return True
            elif response.status_code == 401:
                raise CCPAuthenticationError("Invalid credentials for Eurex Clearing")
            else:
                raise CCPConnectionError(
                    f"Connection failed with status {response.status_code}: {response.text}"
                )

        except httpx.ConnectError as e:
            raise CCPConnectionError(f"Failed to connect to Eurex Clearing: {e}")
        except Exception as e:
            raise CCPConnectionError(f"Unexpected error during connection: {e}")

    async def disconnect(self) -> bool:
        """Disconnect from Eurex Clearing.

        Returns:
            True if disconnection successful
        """
        if self._client:
            # Logout from C7
            try:
                await self._client.post(
                    "/api/v2/logout",
                    headers={"Authorization": f"Bearer {self._session_id}"},
                )
            except Exception:
                pass
            await self._client.aclose()
            self._client = None

        self._connected = False
        self._session_id = None
        self._prisma_parameters.clear()
        self._cross_margin_groups.clear()
        return True

    async def submit_trade(self, trade: Trade) -> TradeSubmissionResponse:
        """Submit trade to Eurex Clearing.

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
            raise CCPConnectionError("Not connected to Eurex Clearing")

        start_time = datetime.utcnow()

        try:
            # Convert trade to Eurex format
            eurex_trade = self._format_eurex_trade(trade)

            # Submit trade via C7 or REST
            endpoint = "/api/v2/trades/submit"
            if self.config.c7_connectivity:
                endpoint = "/c7/api/v2/trades/submit"

            response = await self._client.post(
                endpoint,
                json=eurex_trade,
                headers={"Authorization": f"Bearer {self._session_id}"},
            )

            response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            if response.status_code == 201:
                result = response.json()

                # Calculate Prisma margin if enabled
                prisma_margin = None
                if self.config.prisma_enabled and result.get("margin_calculated"):
                    prisma_margin = await self._calculate_prisma_margin(trade)

                submission_response = TradeSubmissionResponse(
                    submission_id=result["submission_id"],
                    trade_id=trade.trade_id,
                    status=TradeStatus.ACCEPTED,
                    ccp_trade_id=result.get("eurex_trade_id"),
                    metadata={
                        "clearing_house": "EUREX",
                        "clearing_member": self.config.clearing_member_id,
                        "asset_class": self.config.asset_class.value,
                        "prisma_margin": prisma_margin.model_dump() if prisma_margin else None,
                        "cross_margin_group": result.get("cross_margin_group"),
                        "t7_matched": result.get("t7_matched", False),
                        "c7_reference": result.get("c7_reference"),
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
                    f"Eurex rejected trade: {rejection_reason}",
                    rejection_code=rejection_code
                )

            else:
                self.metrics.record_submission(False, response_time_ms)
                raise CCPError(
                    f"Unexpected response from Eurex: {response.status_code} - {response.text}"
                )

        except httpx.TimeoutException:
            raise CCPTimeoutError("Trade submission to Eurex timed out")
        except CCPTradeRejectionError:
            raise
        except Exception as e:
            raise CCPError(f"Failed to submit trade to Eurex: {e}")

    async def get_trade_status(self, trade_id: str) -> TradeStatus:
        """Get current status of a trade.

        Args:
            trade_id: Trade identifier

        Returns:
            Current trade status
        """
        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to Eurex Clearing")

        try:
            response = await self._client.get(
                f"/api/v2/trades/{trade_id}/status",
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
            raise CCPConnectionError("Not connected to Eurex Clearing")

        try:
            response = await self._client.delete(
                f"/api/v2/trades/{trade_id}",
                headers={"Authorization": f"Bearer {self._session_id}"},
            )

            return response.status_code == 200

        except Exception as e:
            raise CCPError(f"Error cancelling trade: {e}")

    async def get_margin_requirements(
        self,
        member_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get current margin requirements using Prisma.

        Args:
            member_id: Member ID (defaults to configured member)

        Returns:
            Dictionary with Prisma margin requirements
        """
        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to Eurex Clearing")

        member_id = member_id or self.config.member_id

        try:
            response = await self._client.get(
                f"/api/v2/margin/prisma",
                params={
                    "member_id": self.config.clearing_member_id,
                    "participant": self.config.participant_code,
                },
                headers={"Authorization": f"Bearer {self._session_id}"},
            )

            if response.status_code == 200:
                data = response.json()

                # Add cross-margining benefits if enabled
                if self.config.cross_margining_enabled:
                    cross_margin_benefits = await self._get_cross_margin_benefits()
                    data["cross_margining"] = cross_margin_benefits

                return data
            else:
                raise CCPError(f"Failed to get margin requirements: {response.text}")

        except Exception as e:
            raise CCPError(f"Error getting margin requirements: {e}")

    async def get_position_report(
        self,
        as_of_date: Optional[datetime] = None
    ) -> PositionReport:
        """Get position report from Eurex.

        Args:
            as_of_date: Report date (defaults to today)

        Returns:
            Position report with portfolio details
        """
        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to Eurex Clearing")

        as_of_date = as_of_date or datetime.utcnow()

        try:
            response = await self._client.get(
                f"/api/v2/positions/report",
                params={
                    "member_id": self.config.clearing_member_id,
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
                    initial_margin=Decimal(str(data.get("prisma_margin", 0))),
                    variation_margin=Decimal(str(data.get("variation_margin", 0))),
                    metadata={
                        "clearing_house": "EUREX",
                        "clearing_member": self.config.clearing_member_id,
                        "asset_class": self.config.asset_class.value,
                        "prisma_enabled": self.config.prisma_enabled,
                        "cross_margining_enabled": self.config.cross_margining_enabled,
                        "cross_margin_groups": len(self._cross_margin_groups),
                    },
                )
            else:
                raise CCPError(f"Failed to get position report: {response.text}")

        except Exception as e:
            raise CCPError(f"Error getting position report: {e}")

    async def healthcheck(self) -> bool:
        """Check Eurex Clearing connectivity health.

        Returns:
            True if connection healthy
        """
        if not self._connected or not self._client:
            return False

        try:
            response = await self._client.get(
                "/api/v2/health",
                timeout=5.0,
            )
            if response.status_code == 200:
                self.metrics.last_heartbeat = datetime.utcnow()
                return True
            return False
        except Exception:
            return False

    async def _calculate_prisma_margin(self, trade: Trade) -> PrismaMarginBreakdown:
        """Calculate Prisma margin for a trade.

        Args:
            trade: Trade to calculate margin for

        Returns:
            Prisma margin breakdown
        """
        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to Eurex Clearing")

        try:
            response = await self._client.post(
                "/api/v2/margin/prisma/calculate",
                json=self._format_eurex_trade(trade),
                headers={"Authorization": f"Bearer {self._session_id}"},
            )

            if response.status_code == 200:
                data = response.json()
                return PrismaMarginBreakdown(
                    core_margin=Decimal(str(data["core_margin"])),
                    add_on_margin=Decimal(str(data.get("add_on_margin", 0))),
                    concentration_charge=Decimal(str(data.get("concentration_charge", 0))),
                    liquidity_add_on=Decimal(str(data.get("liquidity_add_on", 0))),
                    cross_margining_benefit=Decimal(str(data.get("cross_margining_benefit", 0))),
                    total_margin=Decimal(str(data["total_margin"])),
                )
            else:
                raise CCPError(f"Prisma calculation failed: {response.text}")

        except Exception as e:
            raise CCPError(f"Error calculating Prisma margin: {e}")

    async def _get_cross_margin_benefits(self) -> Dict[str, Any]:
        """Get cross-margining benefits across asset classes.

        Returns:
            Cross-margining benefits dictionary
        """
        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to Eurex Clearing")

        try:
            response = await self._client.get(
                "/api/v2/margin/cross-margin-benefits",
                params={"member_id": self.config.clearing_member_id},
                headers={"Authorization": f"Bearer {self._session_id}"},
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {}

        except Exception:
            return {}

    async def _load_prisma_parameters(self) -> None:
        """Load Prisma margin parameters."""
        if not self._connected or not self._client:
            return

        try:
            response = await self._client.get(
                "/api/v2/margin/prisma/parameters",
                headers={"Authorization": f"Bearer {self._session_id}"},
            )

            if response.status_code == 200:
                self._prisma_parameters = response.json()

        except Exception:
            pass

    async def _load_cross_margin_groups(self) -> None:
        """Load cross-margining group configurations."""
        if not self._connected or not self._client:
            return

        try:
            response = await self._client.get(
                "/api/v2/cross-margin/groups",
                params={"member_id": self.config.clearing_member_id},
                headers={"Authorization": f"Bearer {self._session_id}"},
            )

            if response.status_code == 200:
                data = response.json()
                self._cross_margin_groups = [g["group_id"] for g in data.get("groups", [])]

        except Exception:
            pass

    def _format_eurex_trade(self, trade: Trade) -> Dict[str, Any]:
        """Convert Neutryx trade to Eurex format.

        Args:
            trade: Neutryx trade object

        Returns:
            Eurex-formatted trade dictionary
        """
        return {
            "trade_id": trade.trade_id,
            "uti": trade.uti or str(uuid.uuid4()),
            "product_type": trade.product_type.value,
            "asset_class": self.config.asset_class.value,
            "trade_date": trade.trade_date.strftime("%Y%m%d"),
            "effective_date": trade.effective_date.strftime("%Y%m%d"),
            "maturity_date": trade.maturity_date.strftime("%Y%m%d"),
            "notional": {
                "amount": float(trade.economics.notional),
                "currency": trade.economics.currency,
            },
            "price": float(trade.economics.price) if trade.economics.price else None,
            "fixed_rate": float(trade.economics.fixed_rate) if trade.economics.fixed_rate else None,
            "buyer": {
                "party_id": trade.buyer.party_id,
                "clearing_member": self.config.clearing_member_id,
                "participant_code": self.config.participant_code,
            },
            "seller": {
                "party_id": trade.seller.party_id,
                "clearing_member": self.config.clearing_member_id,
                "participant_code": self.config.participant_code,
            },
            "prisma_enabled": self.config.prisma_enabled,
            "cross_margining_enabled": self.config.cross_margining_enabled,
            "t7_integration": self.config.t7_integration,
        }


__all__ = [
    "EurexClearingConnector",
    "EurexClearingConfig",
    "EurexAssetClass",
    "PrismaMarginBreakdown",
]
