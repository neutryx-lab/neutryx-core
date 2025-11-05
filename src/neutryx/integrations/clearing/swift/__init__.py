"""SWIFT messaging integration for settlement and clearing.

This module provides support for SWIFT MT (plain text) and MX (ISO 20022 XML)
message formats used in cross-border payment and settlement instructions.
"""

from .base import SwiftMessage, SwiftMessageType, SwiftError
from .mt import MT540, MT542, MT543, MT544, MTMessage
from .mx import MXMessage, PACS008, SETR002

__all__ = [
    "SwiftMessage",
    "SwiftMessageType",
    "SwiftError",
    "MTMessage",
    "MT540",
    "MT542",
    "MT543",
    "MT544",
    "MXMessage",
    "PACS008",
    "SETR002",
]
