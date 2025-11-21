"""CCP Routing Service for optimal clearing house selection.

This module provides intelligent routing of trades to the most suitable CCP
based on product eligibility, margin requirements, and operational efficiency.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from .base import (
    CCPConnector,
    CCPError,
    ProductType,
    Trade,
    TradeSubmissionResponse,
    TradeStatus,
)


class RoutingStrategy(str, Enum):
    """CCP routing strategy."""
    LOWEST_MARGIN = "lowest_margin"  # Minimize margin requirements
    LOWEST_FEES = "lowest_fees"  # Minimize clearing fees
    BEST_EXECUTION = "best_execution"  # Balance margin and fees
    NETTING_EFFICIENCY = "netting_efficiency"  # Maximize netting benefits
    MANUAL = "manual"  # Manual CCP selection


class CCPCapability(str, Enum):
    """CCP capabilities."""
    PORTFOLIO_MARGINING = "portfolio_margining"
    COMPRESSION = "compression"
    NETTING = "netting"
    CROSS_CURRENCY = "cross_currency"
    MULTI_ASSET = "multi_asset"
    REALTIME_MARGIN = "realtime_margin"


@dataclass
class CCPEligibilityRule:
    """Rules for CCP eligibility."""

    ccp_name: str
    clearable_products: Set[ProductType]
    supported_currencies: Set[str]
    min_notional: Optional[Decimal] = None
    max_notional: Optional[Decimal] = None
    min_maturity_days: Optional[int] = None
    max_maturity_days: Optional[int] = None
    capabilities: Set[CCPCapability] = field(default_factory=set)
    jurisdiction_restrictions: Set[str] = field(default_factory=set)

    def is_eligible(self, trade: Trade) -> tuple[bool, Optional[str]]:
        """Check if trade is eligible for this CCP.

        Args:
            trade: Trade to check

        Returns:
            Tuple of (is_eligible, rejection_reason)
        """
        # Check product type
        if trade.product_type not in self.clearable_products:
            return False, f"Product type {trade.product_type.value} not supported"

        # Check currency
        if trade.economics.currency not in self.supported_currencies:
            return False, f"Currency {trade.economics.currency} not supported"

        # Check notional limits
        if self.min_notional and trade.economics.notional < self.min_notional:
            return False, f"Notional below minimum {self.min_notional}"

        if self.max_notional and trade.economics.notional > self.max_notional:
            return False, f"Notional above maximum {self.max_notional}"

        # Check maturity
        if self.min_maturity_days or self.max_maturity_days:
            maturity_days = (trade.maturity_date - trade.trade_date).days
            if self.min_maturity_days and maturity_days < self.min_maturity_days:
                return False, f"Maturity below minimum {self.min_maturity_days} days"
            if self.max_maturity_days and maturity_days > self.max_maturity_days:
                return False, f"Maturity above maximum {self.max_maturity_days} days"

        return True, None


class MarginQuote(BaseModel):
    """Margin quote from a CCP."""

    ccp_name: str = Field(..., description="CCP name")
    quote_id: str = Field(..., description="Quote identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Margin components
    initial_margin: Decimal = Field(..., description="Initial margin requirement")
    variation_margin: Optional[Decimal] = Field(None, description="Expected variation margin")
    additional_margin: Optional[Decimal] = Field(None, description="Additional margin buffers")
    total_margin: Decimal = Field(..., description="Total margin requirement")

    # Fees
    clearing_fee: Optional[Decimal] = Field(None, description="Clearing fee")
    submission_fee: Optional[Decimal] = Field(None, description="Submission fee")
    total_fees: Optional[Decimal] = Field(None, description="Total fees")

    # Netting benefits
    netting_benefit: Optional[Decimal] = Field(None, description="Margin reduction from netting")
    portfolio_margin_benefit: Optional[Decimal] = Field(None, description="Portfolio margining benefit")

    # Quote validity
    valid_until: datetime = Field(..., description="Quote expiration")

    currency: str = Field(default="USD", description="Quote currency")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def total_cost(self) -> Decimal:
        """Calculate total cost (margin + fees)."""
        fees = self.total_fees or Decimal(0)
        return self.total_margin + fees


class RoutingDecision(BaseModel):
    """CCP routing decision."""

    trade_id: str = Field(..., description="Trade identifier")
    selected_ccp: str = Field(..., description="Selected CCP name")
    routing_strategy: RoutingStrategy = Field(..., description="Strategy used")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Evaluation details
    eligible_ccps: List[str] = Field(default_factory=list, description="All eligible CCPs")
    margin_quotes: Dict[str, MarginQuote] = Field(default_factory=dict, description="Margin quotes received")

    # Decision rationale
    selection_score: float = Field(..., description="Selection score")
    rationale: str = Field(..., description="Human-readable rationale")

    # Alternatives
    alternative_ccps: List[tuple[str, float]] = Field(
        default_factory=list,
        description="Alternative CCPs with their scores"
    )

    metadata: Dict[str, Any] = Field(default_factory=dict)


class CCPRouter:
    """Intelligent CCP routing service.

    This service determines the optimal CCP for clearing each trade based on:
    - Product eligibility
    - Margin requirements
    - Clearing fees
    - Netting benefits
    - Operational efficiency
    """

    def __init__(
        self,
        connectors: Dict[str, CCPConnector],
        eligibility_rules: Optional[List[CCPEligibilityRule]] = None,
        default_strategy: RoutingStrategy = RoutingStrategy.LOWEST_MARGIN,
    ):
        """Initialize CCP router.

        Args:
            connectors: Dictionary of CCP name to connector instances
            eligibility_rules: List of eligibility rules (auto-generated if None)
            default_strategy: Default routing strategy
        """
        self.connectors = connectors
        self.eligibility_rules = eligibility_rules or self._default_eligibility_rules()
        self.default_strategy = default_strategy
        self._routing_history: List[RoutingDecision] = []

    def _default_eligibility_rules(self) -> List[CCPEligibilityRule]:
        """Generate default eligibility rules for known CCPs."""
        rules = []

        # LCH SwapClear - Leading IRS clearing
        rules.append(CCPEligibilityRule(
            ccp_name="LCH SwapClear",
            clearable_products={ProductType.IRS, ProductType.SWAPTION},
            supported_currencies={"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "SEK", "NOK", "DKK"},
            min_notional=Decimal("1000000"),  # $1M minimum
            max_maturity_days=365 * 50,  # 50 years max
            capabilities={
                CCPCapability.PORTFOLIO_MARGINING,
                CCPCapability.COMPRESSION,
                CCPCapability.NETTING,
                CCPCapability.CROSS_CURRENCY,
            },
        ))

        # CME Clearing - Multi-asset clearing
        rules.append(CCPEligibilityRule(
            ccp_name="CME Clearing",
            clearable_products={ProductType.IRS, ProductType.FX_FORWARD, ProductType.FX_SWAP},
            supported_currencies={"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "MXN", "BRL"},
            min_notional=Decimal("500000"),  # $500K minimum
            max_maturity_days=365 * 30,  # 30 years max
            capabilities={
                CCPCapability.PORTFOLIO_MARGINING,
                CCPCapability.NETTING,
                CCPCapability.MULTI_ASSET,
                CCPCapability.REALTIME_MARGIN,
            },
        ))

        # ICE Clear - Credit and European derivatives
        rules.append(CCPEligibilityRule(
            ccp_name="ICE Clear",
            clearable_products={ProductType.CDS, ProductType.IRS, ProductType.SWAPTION},
            supported_currencies={"USD", "EUR", "GBP"},
            min_notional=Decimal("1000000"),  # $1M minimum
            max_maturity_days=365 * 30,  # 30 years max
            capabilities={
                CCPCapability.PORTFOLIO_MARGINING,
                CCPCapability.COMPRESSION,
                CCPCapability.NETTING,
            },
        ))

        # Eurex Clearing - European derivatives with Prisma margining
        rules.append(CCPEligibilityRule(
            ccp_name="Eurex Clearing",
            clearable_products={ProductType.IRS, ProductType.REPO, ProductType.SWAPTION},
            supported_currencies={"EUR", "USD", "GBP", "CHF"},
            min_notional=Decimal("500000"),  # €500K minimum
            max_maturity_days=365 * 30,  # 30 years max
            capabilities={
                CCPCapability.PORTFOLIO_MARGINING,
                CCPCapability.NETTING,
                CCPCapability.CROSS_CURRENCY,
            },
        ))

        return rules

    async def route_trade(
        self,
        trade: Trade,
        strategy: Optional[RoutingStrategy] = None,
        preferred_ccp: Optional[str] = None,
    ) -> RoutingDecision:
        """Route trade to optimal CCP.

        Args:
            trade: Trade to route
            strategy: Routing strategy (uses default if None)
            preferred_ccp: Preferred CCP name (for manual routing)

        Returns:
            Routing decision with selected CCP

        Raises:
            CCPError: If no eligible CCP found or routing fails
        """
        strategy = strategy or self.default_strategy

        # Step 1: Determine eligible CCPs
        eligible_ccps = self._get_eligible_ccps(trade)

        if not eligible_ccps:
            raise CCPError(
                f"No eligible CCP found for trade {trade.trade_id}. "
                f"Product: {trade.product_type.value}, Currency: {trade.economics.currency}"
            )

        # Step 2: Handle manual routing
        if strategy == RoutingStrategy.MANUAL:
            if not preferred_ccp:
                raise CCPError("Manual routing requires preferred_ccp parameter")
            if preferred_ccp not in eligible_ccps:
                raise CCPError(f"Preferred CCP {preferred_ccp} is not eligible for this trade")

            return RoutingDecision(
                trade_id=trade.trade_id,
                selected_ccp=preferred_ccp,
                routing_strategy=strategy,
                eligible_ccps=eligible_ccps,
                selection_score=1.0,
                rationale=f"Manual selection of {preferred_ccp}",
            )

        # Step 3: Get margin quotes from eligible CCPs
        margin_quotes = await self._get_margin_quotes(trade, eligible_ccps)

        if not margin_quotes:
            # Fallback: select first eligible CCP
            selected_ccp = eligible_ccps[0]
            return RoutingDecision(
                trade_id=trade.trade_id,
                selected_ccp=selected_ccp,
                routing_strategy=strategy,
                eligible_ccps=eligible_ccps,
                selection_score=0.5,
                rationale=f"Fallback selection (no margin quotes available): {selected_ccp}",
            )

        # Step 4: Apply routing strategy
        selected_ccp, score, alternatives = self._apply_routing_strategy(
            strategy, margin_quotes
        )

        # Step 5: Create routing decision
        decision = RoutingDecision(
            trade_id=trade.trade_id,
            selected_ccp=selected_ccp,
            routing_strategy=strategy,
            eligible_ccps=eligible_ccps,
            margin_quotes=margin_quotes,
            selection_score=score,
            rationale=self._generate_rationale(strategy, selected_ccp, margin_quotes),
            alternative_ccps=alternatives,
        )

        self._routing_history.append(decision)
        return decision

    def _get_eligible_ccps(self, trade: Trade) -> List[str]:
        """Get list of eligible CCPs for trade.

        Args:
            trade: Trade to check

        Returns:
            List of eligible CCP names
        """
        eligible = []

        for rule in self.eligibility_rules:
            is_eligible, reason = rule.is_eligible(trade)
            if is_eligible and rule.ccp_name in self.connectors:
                eligible.append(rule.ccp_name)

        return eligible

    async def _get_margin_quotes(
        self,
        trade: Trade,
        eligible_ccps: List[str],
    ) -> Dict[str, MarginQuote]:
        """Get margin quotes from eligible CCPs.

        Args:
            trade: Trade to quote
            eligible_ccps: List of eligible CCP names

        Returns:
            Dictionary of CCP name to margin quote
        """
        quote_tasks = []
        ccp_names = []

        for ccp_name in eligible_ccps:
            connector = self.connectors.get(ccp_name)
            if connector and connector.is_connected:
                quote_tasks.append(self._get_single_margin_quote(connector, trade))
                ccp_names.append(ccp_name)

        if not quote_tasks:
            return {}

        # Execute margin quote requests in parallel
        results = await asyncio.gather(*quote_tasks, return_exceptions=True)

        quotes = {}
        for ccp_name, result in zip(ccp_names, results):
            if isinstance(result, MarginQuote):
                quotes[ccp_name] = result

        return quotes

    async def _get_single_margin_quote(
        self,
        connector: CCPConnector,
        trade: Trade,
    ) -> MarginQuote:
        """Get margin quote from a single CCP.

        Args:
            connector: CCP connector
            trade: Trade to quote

        Returns:
            Margin quote
        """
        try:
            # Get margin requirements
            margin_data = await connector.get_margin_requirements()

            # Extract margin components
            initial_margin = Decimal(str(margin_data.get("initial_margin", 0)))
            variation_margin = margin_data.get("variation_margin")
            additional_margin = margin_data.get("additional_margin")

            # Estimate based on notional if specific margin not available
            if initial_margin == 0:
                # Rule of thumb: 2-5% of notional for IRS
                initial_margin = trade.economics.notional * Decimal("0.03")

            total_margin = initial_margin
            if variation_margin:
                total_margin += Decimal(str(variation_margin))
            if additional_margin:
                total_margin += Decimal(str(additional_margin))

            # Extract fees
            clearing_fee = margin_data.get("clearing_fee")
            submission_fee = margin_data.get("submission_fee")
            total_fees = None
            if clearing_fee or submission_fee:
                total_fees = Decimal(str(clearing_fee or 0)) + Decimal(str(submission_fee or 0))

            # Create quote
            return MarginQuote(
                ccp_name=connector.config.ccp_name,
                quote_id=f"Q-{trade.trade_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                initial_margin=initial_margin,
                variation_margin=Decimal(str(variation_margin)) if variation_margin else None,
                additional_margin=Decimal(str(additional_margin)) if additional_margin else None,
                total_margin=total_margin,
                clearing_fee=Decimal(str(clearing_fee)) if clearing_fee else None,
                submission_fee=Decimal(str(submission_fee)) if submission_fee else None,
                total_fees=total_fees,
                netting_benefit=Decimal(str(margin_data.get("netting_benefit", 0))),
                portfolio_margin_benefit=Decimal(str(margin_data.get("portfolio_benefit", 0))),
                valid_until=datetime.utcnow(),  # Should be set by CCP
                currency=trade.economics.currency,
                metadata=margin_data,
            )

        except Exception as e:
            # Return a fallback quote with estimated margin
            return MarginQuote(
                ccp_name=connector.config.ccp_name,
                quote_id=f"Q-{trade.trade_id}-ESTIMATED",
                initial_margin=trade.economics.notional * Decimal("0.03"),
                total_margin=trade.economics.notional * Decimal("0.03"),
                valid_until=datetime.utcnow(),
                currency=trade.economics.currency,
                metadata={"error": str(e), "estimated": True},
            )

    def _apply_routing_strategy(
        self,
        strategy: RoutingStrategy,
        margin_quotes: Dict[str, MarginQuote],
    ) -> tuple[str, float, List[tuple[str, float]]]:
        """Apply routing strategy to select CCP.

        Args:
            strategy: Routing strategy
            margin_quotes: Margin quotes from CCPs

        Returns:
            Tuple of (selected_ccp, score, alternatives)
        """
        if strategy == RoutingStrategy.LOWEST_MARGIN:
            return self._select_lowest_margin(margin_quotes)

        elif strategy == RoutingStrategy.LOWEST_FEES:
            return self._select_lowest_fees(margin_quotes)

        elif strategy == RoutingStrategy.BEST_EXECUTION:
            return self._select_best_execution(margin_quotes)

        elif strategy == RoutingStrategy.NETTING_EFFICIENCY:
            return self._select_netting_efficiency(margin_quotes)

        else:
            # Fallback to lowest margin
            return self._select_lowest_margin(margin_quotes)

    def _select_lowest_margin(
        self,
        margin_quotes: Dict[str, MarginQuote],
    ) -> tuple[str, float, List[tuple[str, float]]]:
        """Select CCP with lowest margin requirement."""
        scored_ccps = []

        for ccp_name, quote in margin_quotes.items():
            margin = float(quote.total_margin)
            scored_ccps.append((ccp_name, margin))

        # Sort by margin (lower is better)
        scored_ccps.sort(key=lambda x: x[1])

        selected_ccp = scored_ccps[0][0]
        min_margin = scored_ccps[0][1]
        max_margin = max(m for _, m in scored_ccps)

        # Normalize score (0-1, higher is better)
        if max_margin > 0:
            score = 1.0 - (min_margin / max_margin)
        else:
            score = 1.0

        alternatives = [(ccp, 1.0 - (m / max_margin)) for ccp, m in scored_ccps[1:]]

        return selected_ccp, score, alternatives

    def _select_lowest_fees(
        self,
        margin_quotes: Dict[str, MarginQuote],
    ) -> tuple[str, float, List[tuple[str, float]]]:
        """Select CCP with lowest fees."""
        scored_ccps = []

        for ccp_name, quote in margin_quotes.items():
            fees = float(quote.total_fees) if quote.total_fees else 0.0
            scored_ccps.append((ccp_name, fees))

        scored_ccps.sort(key=lambda x: x[1])

        selected_ccp = scored_ccps[0][0]
        min_fees = scored_ccps[0][1]
        max_fees = max(f for _, f in scored_ccps) if scored_ccps else 0

        if max_fees > 0:
            score = 1.0 - (min_fees / max_fees)
        else:
            score = 1.0

        alternatives = [(ccp, 1.0 - (f / max_fees)) for ccp, f in scored_ccps[1:]] if max_fees > 0 else []

        return selected_ccp, score, alternatives

    def _select_best_execution(
        self,
        margin_quotes: Dict[str, MarginQuote],
    ) -> tuple[str, float, List[tuple[str, float]]]:
        """Select CCP with best overall execution (margin + fees)."""
        scored_ccps = []

        for ccp_name, quote in margin_quotes.items():
            total_cost = float(quote.total_cost())
            scored_ccps.append((ccp_name, total_cost))

        scored_ccps.sort(key=lambda x: x[1])

        selected_ccp = scored_ccps[0][0]
        min_cost = scored_ccps[0][1]
        max_cost = max(c for _, c in scored_ccps)

        if max_cost > 0:
            score = 1.0 - (min_cost / max_cost)
        else:
            score = 1.0

        alternatives = [(ccp, 1.0 - (c / max_cost)) for ccp, c in scored_ccps[1:]]

        return selected_ccp, score, alternatives

    def _select_netting_efficiency(
        self,
        margin_quotes: Dict[str, MarginQuote],
    ) -> tuple[str, float, List[tuple[str, float]]]:
        """Select CCP with best netting benefits."""
        scored_ccps = []

        for ccp_name, quote in margin_quotes.items():
            netting_benefit = float(quote.netting_benefit or 0)
            portfolio_benefit = float(quote.portfolio_margin_benefit or 0)
            total_benefit = netting_benefit + portfolio_benefit
            scored_ccps.append((ccp_name, total_benefit))

        # Sort by benefit (higher is better)
        scored_ccps.sort(key=lambda x: x[1], reverse=True)

        selected_ccp = scored_ccps[0][0]
        max_benefit = scored_ccps[0][1]

        score = 1.0  # Best benefit gets score of 1.0
        alternatives = [(ccp, b / max_benefit if max_benefit > 0 else 0) for ccp, b in scored_ccps[1:]]

        return selected_ccp, score, alternatives

    def _generate_rationale(
        self,
        strategy: RoutingStrategy,
        selected_ccp: str,
        margin_quotes: Dict[str, MarginQuote],
    ) -> str:
        """Generate human-readable rationale for CCP selection."""
        quote = margin_quotes[selected_ccp]

        if strategy == RoutingStrategy.LOWEST_MARGIN:
            return (
                f"Selected {selected_ccp} for lowest margin requirement of "
                f"{quote.currency} {quote.total_margin:,.2f}"
            )

        elif strategy == RoutingStrategy.LOWEST_FEES:
            fees = quote.total_fees or Decimal(0)
            return (
                f"Selected {selected_ccp} for lowest fees of "
                f"{quote.currency} {fees:,.2f}"
            )

        elif strategy == RoutingStrategy.BEST_EXECUTION:
            return (
                f"Selected {selected_ccp} for best execution cost of "
                f"{quote.currency} {quote.total_cost():,.2f} "
                f"(margin: {quote.total_margin:,.2f}, fees: {quote.total_fees or 0:,.2f})"
            )

        elif strategy == RoutingStrategy.NETTING_EFFICIENCY:
            benefit = (quote.netting_benefit or 0) + (quote.portfolio_margin_benefit or 0)
            return (
                f"Selected {selected_ccp} for best netting efficiency with "
                f"{quote.currency} {benefit:,.2f} margin reduction"
            )

        else:
            return f"Selected {selected_ccp}"

    async def route_and_submit(
        self,
        trade: Trade,
        strategy: Optional[RoutingStrategy] = None,
    ) -> tuple[RoutingDecision, TradeSubmissionResponse]:
        """Route trade to optimal CCP and submit for clearing.

        Args:
            trade: Trade to route and submit
            strategy: Routing strategy

        Returns:
            Tuple of (routing_decision, submission_response)
        """
        # Route to optimal CCP
        decision = await self.route_trade(trade, strategy)

        # Submit to selected CCP
        connector = self.connectors[decision.selected_ccp]
        response = await connector.submit_trade(trade)

        # Update decision with submission result
        decision.metadata["submission_response"] = response.model_dump()

        return decision, response

    def get_routing_history(
        self,
        trade_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[RoutingDecision]:
        """Get routing history.

        Args:
            trade_id: Filter by trade ID
            limit: Maximum number of decisions to return

        Returns:
            List of routing decisions
        """
        history = self._routing_history

        if trade_id:
            history = [d for d in history if d.trade_id == trade_id]

        if limit:
            history = history[-limit:]

        return history

    def get_ccp_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get CCP routing statistics.

        Returns:
            Dictionary of CCP statistics
        """
        stats = {}

        for ccp_name in self.connectors.keys():
            ccp_decisions = [d for d in self._routing_history if d.selected_ccp == ccp_name]

            stats[ccp_name] = {
                "total_routed": len(ccp_decisions),
                "avg_selection_score": sum(d.selection_score for d in ccp_decisions) / len(ccp_decisions) if ccp_decisions else 0,
                "last_used": ccp_decisions[-1].timestamp if ccp_decisions else None,
            }

        return stats


__all__ = [
    "CCPRouter",
    "RoutingStrategy",
    "RoutingDecision",
    "MarginQuote",
    "CCPEligibilityRule",
    "CCPCapability",
]
