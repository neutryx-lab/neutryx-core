"""Schema validation helpers for regulatory report XML outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
import xml.etree.ElementTree as ET

try:  # Optional dependency for XSD validation
    import xmlschema  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    xmlschema = None


@dataclass
class SchemaDefinition:
    """Definition of a report XML schema."""

    name: str
    required_paths: List[str] = field(default_factory=list)
    xsd: Optional[str] = None


class SchemaValidationError(Exception):
    """Raised when XML schema validation fails."""


class SchemaValidator:
    """Validator for report XML payloads using simple structure or XSD."""

    def __init__(self, schemas: Dict[Any, SchemaDefinition]) -> None:
        self.schemas = schemas
        self._compiled_xsd: Dict[Any, Any] = {}
        if xmlschema is not None:
            for report_type, schema in schemas.items():
                if schema.xsd:
                    try:
                        self._compiled_xsd[report_type] = xmlschema.XMLSchema(schema.xsd)
                    except Exception as exc:  # pragma: no cover - defensive
                        raise SchemaValidationError(
                            f"Failed to compile XSD for schema '{schema.name}': {exc}"
                        ) from exc

    def validate(self, report: "RegulatoryReport") -> List[str]:
        """Validate a report instance and return validation errors."""

        schema = self.schemas.get(report.report_type)
        if schema is None:
            return []

        try:
            root = ET.fromstring(report.to_xml())
        except ET.ParseError as exc:
            return [f"XML parsing error: {exc}"]

        errors: List[str] = []
        if schema.xsd and report.report_type in self._compiled_xsd:
            validator = self._compiled_xsd[report.report_type]
            if not validator.is_valid(root):
                for err in validator.iter_errors(root):  # pragma: no cover - requires xmlschema
                    errors.append(str(err))
        else:
            errors.extend(self._validate_structure(root, schema))

        return errors

    @staticmethod
    def _validate_structure(root: ET.Element, schema: SchemaDefinition) -> List[str]:
        errors: List[str] = []
        for path in schema.required_paths:
            if not _element_exists(root, path.split("/")):
                errors.append(f"Missing element '{path}' for schema '{schema.name}'")
        return errors

    def register_schema(self, key: Any, schema: SchemaDefinition) -> None:
        """Register a new schema definition."""

        self.schemas[key] = schema
        if xmlschema is not None and schema.xsd:
            self._compiled_xsd[key] = xmlschema.XMLSchema(schema.xsd)


def _element_exists(root: ET.Element, path: Iterable[str]) -> bool:
    segments = list(path)
    if not segments:
        return True

    def _strip(tag: str) -> str:
        return tag.split("}")[-1]

    if _strip(root.tag) != segments[0]:
        return False

    current = [root]
    for segment in segments[1:]:
        next_level: List[ET.Element] = []
        for element in current:
            for child in element:
                if _strip(child.tag) == segment:
                    next_level.append(child)
        if not next_level:
            return False
        current = next_level
    return True


def build_default_schema_definitions(report_type_enum: Any) -> Dict[Any, SchemaDefinition]:
    """Create default schema definitions for the standard report types."""

    return {
        report_type_enum.EMIR_TRADE: SchemaDefinition(
            name="EMIR R0001 Trade",
            required_paths=[
                "Document/ReportHeader/ReportId",
                "Document/TradeReport/ReportingCounterparty/LEI",
                "Document/TradeReport/TradeData/TradeId",
                "Document/TradeReport/TradeData/Product/UPI",
                "Document/TradeReport/TradeData/Notional/Amount",
            ],
        ),
        report_type_enum.EMIR_LIFECYCLE: SchemaDefinition(
            name="EMIR Lifecycle",
            required_paths=[
                "Document/ReportHeader/ReportId",
                "Document/LifecycleEvent/UTI",
                "Document/LifecycleEvent/EventType",
            ],
        ),
        report_type_enum.EMIR_VALUATION: SchemaDefinition(
            name="EMIR Valuation",
            required_paths=[
                "Document/ReportHeader/ReportId",
                "Document/ValuationReport/UTI",
                "Document/ValuationReport/ValuationAmount",
            ],
        ),
        report_type_enum.MIFID_TRANSACTION: SchemaDefinition(
            name="MiFID RTS22",
            required_paths=[
                "Document/Header/ReportId",
                "Document/Transaction/TransactionReference",
                "Document/Transaction/Instrument/ISIN",
                "Document/Transaction/Price/Amount",
            ],
        ),
        report_type_enum.MIFID_REFERENCE_DATA: SchemaDefinition(
            name="MiFID RTS23",
            required_paths=[
                "Document/Header/ReportId",
                "Document/Instrument/ISIN",
                "Document/Instrument/Name",
            ],
        ),
        report_type_enum.BASEL_CAPITAL: SchemaDefinition(
            name="Basel Pillar3",
            required_paths=[
                "Pillar3Report/Header/ReportId",
                "Pillar3Report/Header/BankLEI",
                "Pillar3Report/CapitalRatios/CET1Ratio",
            ],
        ),
        report_type_enum.BASEL_CVA: SchemaDefinition(
            name="Basel CVA",
            required_paths=[
                "CVAReport/Header/ReportId",
                "CVAReport/Header/BankLEI",
                "CVAReport/CVACharge/CapitalCharge",
            ],
        ),
        report_type_enum.BASEL_FRTB: SchemaDefinition(
            name="Basel FRTB",
            required_paths=[
                "FRTBReport/Header/ReportId",
                "FRTBReport/Header/BankLEI",
                "FRTBReport/CapitalRequirement/Total",
            ],
        ),
        report_type_enum.BASEL_LEVERAGE: SchemaDefinition(
            name="Basel Leverage",
            required_paths=[
                "LeverageReport/Header/ReportId",
                "LeverageReport/Header/BankLEI",
                "LeverageReport/LeverageRatio/Ratio",
            ],
        ),
    }


if False:  # pragma: no cover - typing aid
    from ..report_engine import RegulatoryReport
