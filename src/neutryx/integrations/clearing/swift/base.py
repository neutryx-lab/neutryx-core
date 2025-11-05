"""Base classes for SWIFT messaging."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class SwiftMessageType(str, Enum):
    """SWIFT message category types."""

    # Customer Payments and Cheques (MT 1xx)
    MT103 = "MT103"  # Single Customer Credit Transfer

    # Financial Institution Transfers (MT 2xx)
    MT202 = "MT202"  # General Financial Institution Transfer
    MT210 = "MT210"  # Notice to Receive

    # Treasury Markets - Foreign Exchange (MT 3xx)
    MT300 = "MT300"  # Foreign Exchange Confirmation
    MT320 = "MT320"  # Fixed Loan/Deposit Confirmation

    # Collections and Cash Letters (MT 4xx)
    MT400 = "MT400"  # Advice of Payment

    # Securities Markets (MT 5xx)
    MT540 = "MT540"  # Receive Free
    MT541 = "MT541"  # Receive Against Payment
    MT542 = "MT542"  # Deliver Free
    MT543 = "MT543"  # Deliver Against Payment
    MT544 = "MT544"  # Receive Free Confirmation
    MT545 = "MT545"  # Receive Against Payment Confirmation
    MT546 = "MT546"  # Deliver Free Confirmation
    MT547 = "MT547"  # Deliver Against Payment Confirmation

    # ISO 20022 Messages (MX)
    PACS_008 = "pacs.008"  # Financial Institution Credit Transfer
    PACS_009 = "pacs.009"  # Financial Institution Credit Transfer Response
    SETR_002 = "setr.002"  # Redemption Order
    SESE_023 = "sese.023"  # Settlement Instruction
    SESE_025 = "sese.025"  # Settlement Status


class SwiftError(Exception):
    """Base exception for SWIFT messaging errors."""
    pass


class SwiftValidationError(SwiftError):
    """SWIFT message validation error."""
    pass


class SwiftParseError(SwiftError):
    """SWIFT message parsing error."""
    pass


class SwiftMessage(BaseModel, ABC):
    """Abstract base class for SWIFT messages."""

    message_type: str = Field(..., description="SWIFT message type (e.g., MT540, pacs.008)")
    sender_bic: str = Field(..., description="Sender BIC code")
    receiver_bic: str = Field(..., description="Receiver BIC code")
    message_ref: str = Field(..., description="Unique message reference")
    creation_date: datetime = Field(default_factory=datetime.utcnow, description="Message creation timestamp")

    @abstractmethod
    def to_swift(self) -> str:
        """Convert message to SWIFT format (MT or MX).

        Returns:
            Formatted SWIFT message as string
        """
        pass

    @classmethod
    @abstractmethod
    def from_swift(cls, swift_text: str) -> SwiftMessage:
        """Parse SWIFT message from text.

        Args:
            swift_text: Raw SWIFT message text

        Returns:
            Parsed SwiftMessage instance

        Raises:
            SwiftParseError: If message cannot be parsed
        """
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate message format and content.

        Returns:
            True if valid

        Raises:
            SwiftValidationError: If validation fails
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return self.model_dump()


class SwiftConfig(BaseModel):
    """Configuration for SWIFT connectivity."""

    bic_code: str = Field(..., description="Institution BIC code")
    branch_code: Optional[str] = Field(None, description="Branch code (if applicable)")

    # Connection details
    swift_network: str = Field(default="SWIFTNET", description="SWIFT network type")
    alliance_endpoint: Optional[str] = Field(None, description="Alliance Lite2 endpoint")

    # Authentication
    certificate_path: Optional[str] = Field(None, description="PKI certificate path")
    private_key_path: Optional[str] = Field(None, description="Private key path")

    # Message settings
    message_priority: str = Field(default="NORM", description="Message priority: NORM/URGENT")
    delivery_monitoring: bool = Field(default=True, description="Enable delivery monitoring")

    # Environment
    environment: str = Field(default="production", description="Environment: production/test")
    use_simulator: bool = Field(default=False, description="Use SWIFT simulator")

    # Retry settings
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: int = Field(default=5, description="Delay between retries in seconds")
    timeout: int = Field(default=30, description="Request timeout in seconds")


__all__ = [
    "SwiftMessage",
    "SwiftMessageType",
    "SwiftError",
    "SwiftValidationError",
    "SwiftParseError",
    "SwiftConfig",
]
