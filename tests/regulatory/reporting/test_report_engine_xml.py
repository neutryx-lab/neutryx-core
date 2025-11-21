"""Tests for XML generation and schema validation in the regulatory report engine."""

from datetime import datetime
from textwrap import dedent
from uuid import UUID

import pytest

from neutryx.regulatory.reporting.report_engine import (
    BaselCapitalRegulatoryReport,
    BaselCVARegulatoryReport,
    BaselFRTBRegulatoryReport,
    BaselLeverageRegulatoryReport,
    EMIRLifecycleRegulatoryReport,
    EMIRTradeRegulatoryReport,
    EMIRValuationRegulatoryReport,
    MiFIDReferenceDataRegulatoryReport,
    MiFIDTransactionRegulatoryReport,
    RegulatoryReportEngine,
    ReportStatus,
    ReportType,
)


def _fixed_datetime() -> datetime:
    return datetime(2024, 1, 2, 0, 0, 0)


def _fixed_uuid() -> UUID:
    return UUID("00000000-0000-0000-0000-000000000001")


def _create_engine() -> RegulatoryReportEngine:
    return RegulatoryReportEngine(entity_id="ENTITY1", lei="5493001KJTIIGC8Y1R11")


@pytest.mark.parametrize(
    "report_type,report_class,data,expected_xml",
    [
        (
            ReportType.EMIR_TRADE,
            EMIRTradeRegulatoryReport,
            {
                "reporting_counterparty": {"lei": "5493001KJTIIGC8Y1R12"},
                "counterparty": {"lei": "5493001KJTIIGC8Y1R13"},
                "trade": {
                    "trade_id": "UTI-12345",
                    "product": {"upi": "UPI-IRS-USD", "asset_class": "InterestRate"},
                    "notional": {"amount": "1000000", "currency": "USD"},
                    "execution_timestamp": "2024-01-02T10:15:30Z",
                },
            },
            dedent(
                """
                <?xml version="1.0" encoding="UTF-8"?>
                <Document xmlns="urn:esma:emir:r0001">
                  <ReportHeader>
                    <ReportId>00000000-0000-0000-0000-000000000001</ReportId>
                    <CreationTimestamp>2024-01-02T00:00:00</CreationTimestamp>
                    <ReportingEntity>
                      <LEI>5493001KJTIIGC8Y1R12</LEI>
                    </ReportingEntity>
                  </ReportHeader>
                  <TradeReport>
                    <ReportingCounterparty>
                      <LEI>5493001KJTIIGC8Y1R12</LEI>
                    </ReportingCounterparty>
                    <Counterparty>
                      <LEI>5493001KJTIIGC8Y1R13</LEI>
                    </Counterparty>
                    <TradeData>
                      <TradeId>UTI-12345</TradeId>
                      <Product>
                        <UPI>UPI-IRS-USD</UPI>
                        <AssetClass>InterestRate</AssetClass>
                      </Product>
                      <Notional>
                        <Amount>1000000</Amount>
                        <Currency>USD</Currency>
                      </Notional>
                      <ExecutionTimestamp>2024-01-02T10:15:30Z</ExecutionTimestamp>
                    </TradeData>
                  </TradeReport>
                </Document>
                """
            ).strip(),
        ),
        (
            ReportType.EMIR_LIFECYCLE,
            EMIRLifecycleRegulatoryReport,
            {
                "event": {
                    "uti": "UTI-12345",
                    "event_type": "MODI",
                    "event_timestamp": "2024-01-02T11:30:00Z",
                    "changes": [
                        {"field": "Notional", "old_value": "1000000", "new_value": "1500000"},
                    ],
                }
            },
            dedent(
                """
                <?xml version="1.0" encoding="UTF-8"?>
                <Document xmlns="urn:esma:emir:r0001:lifecycle">
                  <ReportHeader>
                    <ReportId>00000000-0000-0000-0000-000000000001</ReportId>
                    <CreationTimestamp>2024-01-02T00:00:00</CreationTimestamp>
                  </ReportHeader>
                  <LifecycleEvent>
                    <UTI>UTI-12345</UTI>
                    <EventType>MODI</EventType>
                    <EventTimestamp>2024-01-02T11:30:00Z</EventTimestamp>
                    <Changes>
                      <Change>
                        <Field>Notional</Field>
                        <OldValue>1000000</OldValue>
                        <NewValue>1500000</NewValue>
                      </Change>
                    </Changes>
                  </LifecycleEvent>
                </Document>
                """
            ).strip(),
        ),
        (
            ReportType.EMIR_VALUATION,
            EMIRValuationRegulatoryReport,
            {
                "valuation": {
                    "uti": "UTI-12345",
                    "valuation_date": "2024-01-02",
                    "valuation_amount": "250000",
                    "valuation_currency": "USD",
                    "valuation_type": "MTM",
                    "valuation_timestamp": "2024-01-02T12:00:00Z",
                }
            },
            dedent(
                """
                <?xml version="1.0" encoding="UTF-8"?>
                <Document xmlns="urn:esma:emir:r0010">
                  <ReportHeader>
                    <ReportId>00000000-0000-0000-0000-000000000001</ReportId>
                    <CreationTimestamp>2024-01-02T00:00:00</CreationTimestamp>
                  </ReportHeader>
                  <ValuationReport>
                    <UTI>UTI-12345</UTI>
                    <ValuationDate>2024-01-02</ValuationDate>
                    <ValuationAmount currency="USD">250000</ValuationAmount>
                    <ValuationType>MTM</ValuationType>
                    <ValuationTimestamp>2024-01-02T12:00:00Z</ValuationTimestamp>
                  </ValuationReport>
                </Document>
                """
            ).strip(),
        ),
        (
            ReportType.MIFID_TRANSACTION,
            MiFIDTransactionRegulatoryReport,
            {
                "reporting_firm": {"lei": "5493001KJTIIGC8Y1R12"},
                "transaction": {
                    "transaction_reference": "TRX-0001",
                    "execution_timestamp": "2024-01-02T09:30:00Z",
                    "quantity": "1000",
                    "instrument": {"isin": "EU000A1G0AA2", "classification": "IRS"},
                    "buyer": {"lei": "5493001KJTIIGC8Y1R14"},
                    "seller": {"lei": "5493001KJTIIGC8Y1R15"},
                    "price": {"amount": "102.5", "currency": "EUR"},
                },
            },
            dedent(
                """
                <?xml version="1.0" encoding="UTF-8"?>
                <Document xmlns="urn:eu:esma:rr:mifid:rts22">
                  <Header>
                    <ReportId>00000000-0000-0000-0000-000000000001</ReportId>
                    <SubmissionDate>2024-01-02</SubmissionDate>
                    <ReportingFirm>
                      <LEI>5493001KJTIIGC8Y1R12</LEI>
                    </ReportingFirm>
                  </Header>
                  <Transaction>
                    <TransactionReference>TRX-0001</TransactionReference>
                    <ExecutionTimestamp>2024-01-02T09:30:00Z</ExecutionTimestamp>
                    <Quantity>1000</Quantity>
                    <Instrument>
                      <ISIN>EU000A1G0AA2</ISIN>
                      <Classification>IRS</Classification>
                    </Instrument>
                    <Buyer>
                      <LEI>5493001KJTIIGC8Y1R14</LEI>
                    </Buyer>
                    <Seller>
                      <LEI>5493001KJTIIGC8Y1R15</LEI>
                    </Seller>
                    <Price>
                      <Amount>102.5</Amount>
                      <Currency>EUR</Currency>
                    </Price>
                  </Transaction>
                </Document>
                """
            ).strip(),
        ),
        (
            ReportType.MIFID_REFERENCE_DATA,
            MiFIDReferenceDataRegulatoryReport,
            {
                "instrument": {
                    "isin": "EU000A1G0AA2",
                    "full_name": "Example Instrument",
                    "classification": "IRS",
                    "currency": "EUR",
                    "issuance_date": "2024-01-01",
                }
            },
            dedent(
                """
                <?xml version="1.0" encoding="UTF-8"?>
                <Document xmlns="urn:eu:esma:rr:mifid:rts23">
                  <Header>
                    <ReportId>00000000-0000-0000-0000-000000000001</ReportId>
                    <SubmissionDate>2024-01-02</SubmissionDate>
                  </Header>
                  <Instrument>
                    <ISIN>EU000A1G0AA2</ISIN>
                    <Name>Example Instrument</Name>
                    <Classification>IRS</Classification>
                    <Currency>EUR</Currency>
                    <IssuanceDate>2024-01-01</IssuanceDate>
                  </Instrument>
                </Document>
                """
            ).strip(),
        ),
        (
            ReportType.BASEL_CAPITAL,
            BaselCapitalRegulatoryReport,
            {
                "bank": {"lei": "5493001KJTIIGC8Y1R20", "name": "Example Bank"},
                "capital": {"cet1": "500000000", "tier1": "600000000", "tier2": "200000000"},
                "risk_weighted_assets": {"credit": "4000000000", "market": "1000000000", "operational": "800000000"},
                "ratios": {"cet1": "12.5", "tier1": "13.5", "total": "16.0"},
            },
            dedent(
                """
                <?xml version="1.0" encoding="UTF-8"?>
                <Pillar3Report xmlns="urn:bis:basel:capital">
                  <Header>
                    <ReportId>00000000-0000-0000-0000-000000000001</ReportId>
                    <ReferenceDate>2024-01-02</ReferenceDate>
                    <BankName>Example Bank</BankName>
                    <BankLEI>5493001KJTIIGC8Y1R20</BankLEI>
                  </Header>
                  <CapitalStructure>
                    <CET1>500000000</CET1>
                    <Tier1>600000000</Tier1>
                    <Tier2>200000000</Tier2>
                  </CapitalStructure>
                  <RiskWeightedAssets>
                    <Credit>4000000000</Credit>
                    <Market>1000000000</Market>
                    <Operational>800000000</Operational>
                  </RiskWeightedAssets>
                  <CapitalRatios>
                    <CET1Ratio>12.5</CET1Ratio>
                    <Tier1Ratio>13.5</Tier1Ratio>
                    <TotalCapitalRatio>16.0</TotalCapitalRatio>
                  </CapitalRatios>
                </Pillar3Report>
                """
            ).strip(),
        ),
        (
            ReportType.BASEL_CVA,
            BaselCVARegulatoryReport,
            {
                "bank": {"lei": "5493001KJTIIGC8Y1R20"},
                "cva": {
                    "capital_charge": "35000000",
                    "advanced_method": "true",
                    "hedges": {"eligible": "1000000", "non_eligible": "500000"},
                },
            },
            dedent(
                """
                <?xml version="1.0" encoding="UTF-8"?>
                <CVAReport xmlns="urn:bis:basel:cva">
                  <Header>
                    <ReportId>00000000-0000-0000-0000-000000000001</ReportId>
                    <ReferenceDate>2024-01-02</ReferenceDate>
                    <BankLEI>5493001KJTIIGC8Y1R20</BankLEI>
                  </Header>
                  <CVACharge>
                    <CapitalCharge>35000000</CapitalCharge>
                    <AdvancedMethod>true</AdvancedMethod>
                  </CVACharge>
                  <Hedges>
                    <Eligible>1000000</Eligible>
                    <NonEligible>500000</NonEligible>
                  </Hedges>
                </CVAReport>
                """
            ).strip(),
        ),
        (
            ReportType.BASEL_FRTB,
            BaselFRTBRegulatoryReport,
            {
                "bank": {"lei": "5493001KJTIIGC8Y1R20"},
                "requirement": {"standardised": "25000000", "internal_model": "15000000", "total": "40000000"},
            },
            dedent(
                """
                <?xml version="1.0" encoding="UTF-8"?>
                <FRTBReport xmlns="urn:bis:basel:frtb">
                  <Header>
                    <ReportId>00000000-0000-0000-0000-000000000001</ReportId>
                    <ReferenceDate>2024-01-02</ReferenceDate>
                    <BankLEI>5493001KJTIIGC8Y1R20</BankLEI>
                  </Header>
                  <CapitalRequirement>
                    <Standardised>25000000</Standardised>
                    <InternalModel>15000000</InternalModel>
                    <Total>40000000</Total>
                  </CapitalRequirement>
                </FRTBReport>
                """
            ).strip(),
        ),
        (
            ReportType.BASEL_LEVERAGE,
            BaselLeverageRegulatoryReport,
            {
                "bank": {"lei": "5493001KJTIIGC8Y1R20"},
                "leverage": {"tier1": "800000000", "exposure": "20000000000", "ratio": "4.0"},
            },
            dedent(
                """
                <?xml version="1.0" encoding="UTF-8"?>
                <LeverageReport xmlns="urn:bis:basel:leverage">
                  <Header>
                    <ReportId>00000000-0000-0000-0000-000000000001</ReportId>
                    <ReferenceDate>2024-01-02</ReferenceDate>
                    <BankLEI>5493001KJTIIGC8Y1R20</BankLEI>
                  </Header>
                  <LeverageRatio>
                    <Tier1Capital>800000000</Tier1Capital>
                    <ExposureMeasure>20000000000</ExposureMeasure>
                    <Ratio>4.0</Ratio>
                  </LeverageRatio>
                </LeverageReport>
                """
            ).strip(),
        ),
    ],
)
def test_report_xml_generation(report_type, report_class, data, expected_xml):
    engine = _create_engine()
    report = engine.create_report(
        report_type=report_type,
        data=data,
        report_id=_fixed_uuid(),
        reporting_date=_fixed_datetime(),
    )

    assert isinstance(report, report_class)
    xml_output = report.to_xml().strip()
    assert xml_output == expected_xml

    assert engine.validate_report(report) is True
    assert report.status == ReportStatus.VALIDATED


def test_schema_validation_missing_required_element():
    engine = _create_engine()
    report = engine.create_report(
        report_type=ReportType.EMIR_TRADE,
        data={
            "reporting_counterparty": {"lei": "5493001KJTIIGC8Y1R12"},
            "counterparty": {"lei": "5493001KJTIIGC8Y1R13"},
            "trade": {
                # Missing trade_id and notional to trigger schema errors
                "product": {"upi": "UPI-IRS-USD", "asset_class": "InterestRate"},
            },
        },
        report_id=_fixed_uuid(),
        reporting_date=_fixed_datetime(),
    )

    assert not engine.validate_report(report)
    assert any("TradeId" in err for err in report.errors)
    assert any("Notional/Amount" in err for err in report.errors)
