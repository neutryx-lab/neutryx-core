"""Risk factor sensitivities and bucketing for SIMM.

This module handles the collection and bucketing of risk sensitivities
(delta, vega) by risk factor for SIMM calculations.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class SensitivityType(str, Enum):
    """Type of risk sensitivity."""

    DELTA = "Delta"  # First-order sensitivity to underlying
    VEGA = "Vega"  # Sensitivity to implied volatility
    CURVATURE = "Curvature"  # Sensitivity to gamma (for options)


class RiskFactorType(str, Enum):
    """SIMM risk factor classification."""

    IR = "InterestRate"  # Interest rate risk
    FX = "ForeignExchange"  # FX risk
    EQUITY = "Equity"  # Equity risk
    COMMODITY = "Commodity"  # Commodity risk
    CREDIT_Q = "CreditQualifying"  # Investment grade credit
    CREDIT_NON_Q = "CreditNonQualifying"  # High yield credit


@dataclass
class RiskFactorSensitivity:
    """Single risk factor sensitivity.

    Attributes
    ----------
    risk_factor_type : RiskFactorType
        Type of risk factor (IR, FX, Equity, etc.)
    sensitivity_type : SensitivityType
        Type of sensitivity (Delta, Vega, Curvature)
    bucket : str
        SIMM bucket identifier (e.g., "USD" for IR, "1" for Equity)
    risk_factor : str
        Specific risk factor identifier (e.g., "USD-LIBOR-3M", "AAPL")
    sensitivity : float
        Sensitivity value in base currency
    tenor : Optional[str]
        Tenor for IR sensitivities (e.g., "3M", "5Y")
    """

    risk_factor_type: RiskFactorType
    sensitivity_type: SensitivityType
    bucket: str
    risk_factor: str
    sensitivity: float
    tenor: Optional[str] = None

    def __repr__(self) -> str:
        """String representation."""
        tenor_str = f", tenor={self.tenor}" if self.tenor else ""
        return (
            f"RiskFactorSensitivity("
            f"type={self.risk_factor_type.value}, "
            f"sens={self.sensitivity_type.value}, "
            f"bucket={self.bucket}, "
            f"factor={self.risk_factor}, "
            f"value={self.sensitivity:.2f}{tenor_str})"
        )


@dataclass
class BucketedSensitivities:
    """Sensitivities grouped by bucket.

    Attributes
    ----------
    risk_factor_type : RiskFactorType
        Risk factor type
    sensitivity_type : SensitivityType
        Sensitivity type
    bucket_sensitivities : Dict[str, List[RiskFactorSensitivity]]
        Map of bucket -> list of sensitivities in that bucket
    """

    risk_factor_type: RiskFactorType
    sensitivity_type: SensitivityType
    bucket_sensitivities: Dict[str, List[RiskFactorSensitivity]]

    def get_bucket_net_sensitivity(self, bucket: str) -> float:
        """Get net sensitivity for a bucket.

        Parameters
        ----------
        bucket : str
            Bucket identifier

        Returns
        -------
        float
            Net sensitivity (sum of all sensitivities in bucket)
        """
        sensitivities = self.bucket_sensitivities.get(bucket, [])
        return sum(s.sensitivity for s in sensitivities)

    def get_all_buckets(self) -> List[str]:
        """Get list of all buckets."""
        return list(self.bucket_sensitivities.keys())


def bucket_sensitivities(
    sensitivities: List[RiskFactorSensitivity],
) -> Dict[tuple[RiskFactorType, SensitivityType], BucketedSensitivities]:
    """Group sensitivities by risk factor type, sensitivity type, and bucket.

    Parameters
    ----------
    sensitivities : List[RiskFactorSensitivity]
        List of all risk factor sensitivities

    Returns
    -------
    Dict[tuple[RiskFactorType, SensitivityType], BucketedSensitivities]
        Bucketed sensitivities grouped by (risk_factor_type, sensitivity_type)
    """
    bucketed: Dict[
        tuple[RiskFactorType, SensitivityType],
        Dict[str, List[RiskFactorSensitivity]],
    ] = {}

    # Group sensitivities
    for sens in sensitivities:
        key = (sens.risk_factor_type, sens.sensitivity_type)
        if key not in bucketed:
            bucketed[key] = {}

        bucket = sens.bucket
        if bucket not in bucketed[key]:
            bucketed[key][bucket] = []

        bucketed[key][bucket].append(sens)

    # Convert to BucketedSensitivities objects
    result = {}
    for (rf_type, sens_type), bucket_dict in bucketed.items():
        result[(rf_type, sens_type)] = BucketedSensitivities(
            risk_factor_type=rf_type,
            sensitivity_type=sens_type,
            bucket_sensitivities=bucket_dict,
        )

    return result


def get_ir_bucket(currency: str) -> str:
    """Get SIMM bucket for interest rate risk.

    Parameters
    ----------
    currency : str
        Currency code (e.g., "USD", "EUR")

    Returns
    -------
    str
        SIMM bucket identifier (same as currency for IR)

    Notes
    -----
    For IR, bucket = currency (e.g., USD, EUR, JPY, GBP)
    """
    return currency


def get_fx_bucket(currency_pair: str) -> str:
    """Get SIMM bucket for FX risk.

    Parameters
    ----------
    currency_pair : str
        Currency pair (e.g., "EURUSD")

    Returns
    -------
    str
        SIMM bucket identifier

    Notes
    -----
    For FX, there's typically only one bucket, but we return
    the currency pair for tracking purposes.
    """
    return "1"  # FX has single bucket in SIMM


def get_equity_bucket(region: str, sector: str) -> str:
    """Get SIMM bucket for equity risk.

    Parameters
    ----------
    region : str
        Geographic region (e.g., "EmergingMarkets", "Developed")
    sector : str
        Sector classification (e.g., "Tech", "Financial")

    Returns
    -------
    str
        SIMM bucket identifier (1-12)

    Notes
    -----
    SIMM equity buckets are based on region and sector:
    - Buckets 1-4: Emerging markets (Consumer, Telecom, etc.)
    - Buckets 5-8: Developed markets (Consumer, Telecom, etc.)
    - Buckets 9-11: Indices
    - Bucket 12: Other
    """
    # Simplified bucketing (full implementation would use detailed mapping)
    if region == "EmergingMarkets":
        if sector in ["Consumer", "Utilities"]:
            return "1"
        elif sector in ["Telecom", "Industrial"]:
            return "2"
        elif sector in ["Energy", "Materials"]:
            return "3"
        else:
            return "4"
    else:  # Developed markets
        if sector in ["Consumer", "Utilities"]:
            return "5"
        elif sector in ["Telecom", "Industrial"]:
            return "6"
        elif sector in ["Energy", "Materials"]:
            return "7"
        else:
            return "8"


def get_credit_bucket(rating: str, sector: str) -> str:
    """Get SIMM bucket for credit risk.

    Parameters
    ----------
    rating : str
        Credit rating (e.g., "AAA", "BBB", "BB")
    sector : str
        Sector classification

    Returns
    -------
    str
        SIMM bucket identifier (1-12 for qualifying, 1-3 for non-qualifying)

    Notes
    -----
    Credit qualifying buckets (IG):
    - Buckets 1-6: By sector (Sovereigns, Financials, Basic Materials, etc.)
    - Buckets 7-12: Indices and structured products

    Credit non-qualifying buckets (HY):
    - Bucket 1: RMBS/CMBS
    - Bucket 2: Other
    """
    # Simplified (full implementation uses detailed sector mapping)
    investment_grade = rating in ["AAA", "AA", "A", "BBB"]

    if investment_grade:
        # Credit qualifying
        if sector == "Sovereign":
            return "1"
        elif sector == "Financial":
            return "2"
        elif sector == "BasicMaterials":
            return "3"
        else:
            return "4"
    else:
        # Credit non-qualifying
        if "MBS" in sector:
            return "1"
        else:
            return "2"


def aggregate_sensitivities_by_tenor(
    sensitivities: List[RiskFactorSensitivity],
    currency: str,
) -> Dict[str, float]:
    """Aggregate IR sensitivities by tenor for a currency.

    Parameters
    ----------
    sensitivities : List[RiskFactorSensitivity]
        List of IR sensitivities
    currency : str
        Currency to filter by

    Returns
    -------
    Dict[str, float]
        Map of tenor -> net sensitivity

    Notes
    -----
    SIMM requires IR sensitivities by tenor vertex:
    2W, 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 10Y, 15Y, 20Y, 30Y
    """
    tenor_sensitivities: Dict[str, float] = {}

    for sens in sensitivities:
        if sens.risk_factor_type != RiskFactorType.IR:
            continue
        if sens.bucket != currency:
            continue
        if sens.tenor is None:
            continue

        tenor = sens.tenor
        if tenor not in tenor_sensitivities:
            tenor_sensitivities[tenor] = 0.0

        tenor_sensitivities[tenor] += sens.sensitivity

    return tenor_sensitivities
