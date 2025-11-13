"""Portfolio position data structures."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict

from pydantic import BaseModel, Field


class PortfolioPosition(BaseModel):
    """Represents a security holding within a portfolio."""

    security_id: str
    quantity: Decimal
    cost_basis: Decimal = Decimal("0")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def adjust_quantity(self, delta: Decimal) -> None:
        """Increment the position quantity by ``delta``."""

        self.quantity += delta

    def update_cost_basis(self, new_cost_basis: Decimal) -> None:
        """Replace the cost basis for the position."""

        self.cost_basis = new_cost_basis


__all__ = ["PortfolioPosition"]
