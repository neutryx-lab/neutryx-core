"""CME Clearing integration.

This module provides connectivity to CME (Chicago Mercantile Exchange) Clearing,
covering IRS, credit, FX, and energy derivatives.

Features:
- Multi-asset class clearing (IRS, FX, Commodities, Options)
- Real-time risk analytics via CORE (Clearing Operations Real-time Engine)
- SPAN margin calculations
- Integration with CME Globex for straight-through processing

API Documentation: https://www.cmegroup.com/clearing
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
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


class CMEClearingConfig(CCPConfig):
    """CME Clearing-specific configuration."""

    clearing_firm_id: str = Field(..., description="CME clearing firm ID")
    account_origin: str = Field(default="CUSTOMER", description="Account origin code")
    give_up_broker: Optional[str] = Field(None, description="Give-up broker ID")

    # CME specific
    span_enabled: bool = Field(default=True, description="Enable SPAN margining")
    core_enabled: bool = Field(default=True, description="Enable CORE analytics")
    product_code: Optional[str] = Field(None, description="CME product code")

    model_config = ConfigDict(extra="allow")


class CMESPANMargin(BaseModel):
    """SPAN margin calculation results."""

    total_margin: Decimal = Field(..., description="Total margin requirement")
    scanning_risk: Decimal = Field(..., description="Scanning risk charge")
    inter_commodity_spread: Decimal = Field(default=Decimal("0"), description="Inter-commodity spread")
    delivery_risk: Decimal = Field(default=Decimal("0"), description="Delivery risk")
    short_option_minimum: Decimal = Field(default=Decimal("0"), description="Short option minimum")


class CMECOREAnalytics(BaseModel):
    """CORE real-time analytics."""

    portfolio_var: Decimal = Field(..., description="Portfolio VaR")
    stress_loss: Decimal = Field(..., description="Maximum stress loss")
    concentration_charge: Decimal = Field(default=Decimal("0"), description="Concentration charge")
    wrong_way_risk: Decimal = Field(default=Decimal("0"), description="Wrong-way risk charge")
    liquidity_risk: Decimal = Field(default=Decimal("0"), description="Liquidity risk add-on")


class CMEClearingConnector(CCPConnector):
    """CME Clearing connectivity implementation.

    Provides full integration with CME Clearing including:
    - Multi-asset trade submission
    - SPAN-based margin calculations
    - CORE real-time risk analytics
    - Integration with CME Globex
    - Cross-margining with other CME products
    """

    def __init__(self, config: CMEClearingConfig):
        """Initialize CME Clearing connector.

        Args:
            config: CME Clearing configuration
        """
        super().__init__(config)
        self.config: CMEClearingConfig = config
        self.metrics = CCPMetrics()
        self._client: Optional[httpx.AsyncClient] = None
        self._span_files_cache: Dict[str, Any] = {}

    async def connect(self) -> bool:
        """Establish connection to CME Clearing.

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
                "X-CME-Firm-ID": self.config.clearing_firm_id,
            }

            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            timeout = httpx.Timeout(self.config.timeout, connect=10.0)

            self._client = httpx.AsyncClient(
                base_url=self.config.api_endpoint,
                headers=headers,
                timeout=timeout,
            )

            # Authenticate
            auth_payload = {
                "firm_id": self.config.clearing_firm_id,
                "member_id": self.config.member_id,
                "timestamp": datetime.utcnow().isoformat(),
            }

            response = await self._client.post("/api/v2/authenticate", json=auth_payload)

            if response.status_code == 200:
                auth_data = response.json()
                self._session_id = auth_data.get("session_token")
                self._connected = True
                self.metrics.last_heartbeat = datetime.utcnow()

                # Download SPAN files if enabled
                if self.config.span_enabled:
                    await self._download_span_files()

                return True
            elif response.status_code == 401:
                raise CCPAuthenticationError("Invalid credentials for CME Clearing")
            else:
                raise CCPConnectionError(
                    f"Connection failed with status {response.status_code}: {response.text}"
                )

        except httpx.ConnectError as e:
            raise CCPConnectionError(f"Failed to connect to CME Clearing: {e}")
        except Exception as e:
            raise CCPConnectionError(f"Unexpected error during connection: {e}")

    async def disconnect(self) -> bool:
        """Disconnect from CME Clearing.

        Returns:
            True if disconnection successful
        """
        if self._client:
            # Logout
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
        self._span_files_cache.clear()
        return True

    async def submit_trade(self, trade: Trade) -> TradeSubmissionResponse:
        """Submit trade to CME Clearing.

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
            raise CCPConnectionError("Not connected to CME Clearing")

        start_time = datetime.utcnow()

        try:
            # Convert trade to CME format
            cme_trade = self._format_cme_trade(trade)

            # Submit trade
            response = await self._client.post(
                "/api/v2/trades/submit",
                json=cme_trade,
                headers={"Authorization": f"Bearer {self._session_id}"},
            )

            response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            if response.status_code == 201:
                result = response.json()

                # Calculate initial SPAN margin if enabled
                span_margin = None
                if self.config.span_enabled:
                    span_margin = await self._calculate_span_margin(trade)

                submission_response = TradeSubmissionResponse(
                    submission_id=result["submission_id"],
                    trade_id=trade.trade_id,
                    status=TradeStatus.ACCEPTED,
                    ccp_trade_id=result.get("cme_trade_id"),
                    metadata={
                        "clearing_house": "CME",
                        "clearing_firm": self.config.clearing_firm_id,
                        "product_code": result.get("product_code"),
                        "span_margin": float(span_margin.total_margin) if span_margin else None,
                        "globex_matched": result.get("globex_matched", False),
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
                    f"CME rejected trade: {rejection_reason}",
                    rejection_code=rejection_code
                )

            else:
                self.metrics.record_submission(False, response_time_ms)
                raise CCPError(
                    f"Unexpected response from CME: {response.status_code} - {response.text}"
                )

        except httpx.TimeoutException:
            raise CCPTimeoutError("Trade submission to CME timed out")
        except CCPTradeRejectionError:
            raise
        except Exception as e:
            raise CCPError(f"Failed to submit trade to CME: {e}")

    async def get_trade_status(self, trade_id: str) -> TradeStatus:
        """Get current status of a trade.

        Args:
            trade_id: Trade identifier

        Returns:
            Current trade status
        """
        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to CME Clearing")

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
            raise CCPConnectionError("Not connected to CME Clearing")

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
        """Get current margin requirements using SPAN.

        Args:
            member_id: Member ID (defaults to configured member)

        Returns:
            Dictionary with SPAN margin requirements
        """
        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to CME Clearing")

        member_id = member_id or self.config.member_id

        try:
            response = await self._client.get(
                f"/api/v2/margin/span",
                params={"firm_id": self.config.clearing_firm_id, "member_id": member_id},
                headers={"Authorization": f"Bearer {self._session_id}"},
            )

            if response.status_code == 200:
                data = response.json()

                # Include CORE analytics if enabled
                if self.config.core_enabled:
                    core_analytics = await self._get_core_analytics()
                    data["core_analytics"] = core_analytics

                return data
            else:
                raise CCPError(f"Failed to get margin requirements: {response.text}")

        except Exception as e:
            raise CCPError(f"Error getting margin requirements: {e}")

    async def get_position_report(
        self,
        as_of_date: Optional[datetime] = None
    ) -> PositionReport:
        """Get position report from CME.

        Args:
            as_of_date: Report date (defaults to today)

        Returns:
            Position report with portfolio details
        """
        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to CME Clearing")

        as_of_date = as_of_date or datetime.utcnow()

        try:
            response = await self._client.get(
                f"/api/v2/positions/report",
                params={
                    "firm_id": self.config.clearing_firm_id,
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
                    initial_margin=Decimal(str(data.get("span_margin", 0))),
                    variation_margin=Decimal(str(data.get("variation_margin", 0))),
                    metadata={
                        "clearing_house": "CME",
                        "clearing_firm": self.config.clearing_firm_id,
                        "span_enabled": self.config.span_enabled,
                        "core_enabled": self.config.core_enabled,
                    },
                )
            else:
                raise CCPError(f"Failed to get position report: {response.text}")

        except Exception as e:
            raise CCPError(f"Error getting position report: {e}")

    async def healthcheck(self) -> bool:
        """Check CME Clearing connectivity health.

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

    async def _calculate_span_margin(self, trade: Trade) -> CMESPANMargin:
        """Calculate SPAN margin for a trade.

        Args:
            trade: Trade to calculate margin for

        Returns:
            SPAN margin breakdown
        """
        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to CME Clearing")

        try:
            response = await self._client.post(
                "/api/v2/margin/span/calculate",
                json=self._format_cme_trade(trade),
                headers={"Authorization": f"Bearer {self._session_id}"},
            )

            if response.status_code == 200:
                data = response.json()
                return CMESPANMargin(
                    total_margin=Decimal(str(data["total_margin"])),
                    scanning_risk=Decimal(str(data["scanning_risk"])),
                    inter_commodity_spread=Decimal(str(data.get("inter_commodity_spread", 0))),
                    delivery_risk=Decimal(str(data.get("delivery_risk", 0))),
                    short_option_minimum=Decimal(str(data.get("short_option_minimum", 0))),
                )
            else:
                raise CCPError(f"SPAN calculation failed: {response.text}")

        except Exception as e:
            raise CCPError(f"Error calculating SPAN margin: {e}")

    async def _get_core_analytics(self) -> Dict[str, Any]:
        """Get CORE real-time analytics.

        Returns:
            CORE analytics dictionary
        """
        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to CME Clearing")

        try:
            response = await self._client.get(
                "/api/v2/analytics/core",
                params={"firm_id": self.config.clearing_firm_id},
                headers={"Authorization": f"Bearer {self._session_id}"},
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {}

        except Exception:
            return {}

    async def _download_span_files(self) -> None:
        """Download SPAN parameter files."""
        if not self._connected or not self._client:
            return

        try:
            response = await self._client.get(
                "/api/v2/span/files",
                headers={"Authorization": f"Bearer {self._session_id}"},
            )

            if response.status_code == 200:
                self._span_files_cache = response.json()

        except Exception:
            pass

    def _format_cme_trade(self, trade: Trade) -> Dict[str, Any]:
        """Convert Neutryx trade to CME format.

        Args:
            trade: Neutryx trade object

        Returns:
            CME-formatted trade dictionary
        """
        return {
            "trade_id": trade.trade_id,
            "uti": trade.uti or str(uuid.uuid4()),
            "product_type": trade.product_type.value,
            "product_code": self.config.product_code,
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
                "clearing_firm": self.config.clearing_firm_id,
                "account_origin": self.config.account_origin,
            },
            "seller": {
                "party_id": trade.seller.party_id,
                "clearing_firm": self.config.clearing_firm_id,
                "account_origin": self.config.account_origin,
            },
            "give_up_broker": self.config.give_up_broker,
        }


__all__ = [
    "CMEClearingConnector",
    "CMEClearingConfig",
    "CMESPANMargin",
    "CMECOREAnalytics",
]
