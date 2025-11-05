"""FRTB Default Risk Charge (DRC) calculations.

This module implements the Default Risk Charge under FRTB, which captures jump-to-default (JTD)
risk for credit-sensitive instruments including bonds, CDS, and securitized products.

The DRC framework uses:
- Credit ratings or PD estimates to determine risk weights
- Loss-given-default (LGD) assumptions by seniority
- Default correlation based on sector and rating
- Separate treatment for non-securitized and securitized products

References:
    - BCBS d352: Minimum capital requirements for market risk (2016/2019)
    - CRE22: Calculation of RWA for credit risk - standardised approach
    - MAR21: Boundary between the banking book and trading book
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
from jax import Array


# ==============================================================================
# Enumerations
# ==============================================================================


class CreditRating(str, Enum):
    """Credit quality steps (CQS) mapped to external ratings."""

    AAA = "AAA"
    AA = "AA"
    A = "A"
    BBB = "BBB"
    BB = "BB"
    B = "B"
    CCC_OR_BELOW = "CCC"
    UNRATED = "NR"


class Seniority(str, Enum):
    """Seniority class for debt instruments."""

    SENIOR_SECURED = "senior_secured"
    SENIOR_UNSECURED = "senior_unsecured"
    SUBORDINATED = "subordinated"
    EQUITY = "equity"


class Sector(str, Enum):
    """Industry sectors for correlation purposes."""

    GOVERNMENT = "government"
    FINANCIAL = "financial"
    INDUSTRIAL = "industrial"
    UTILITIES = "utilities"
    RETAIL = "retail"
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    ENERGY = "energy"
    OTHER = "other"


class SecuritizationType(str, Enum):
    """Type of securitized product."""

    RMBS = "rmbs"  # Residential mortgage-backed
    CMBS = "cmbs"  # Commercial mortgage-backed
    ABS = "abs"  # Asset-backed securities
    CDO = "cdo"  # Collateralized debt obligation
    CLO = "clo"  # Collateralized loan obligation
    SYNTHETIC = "synthetic"  # Synthetic securitization
    OTHER = "other"


# ==============================================================================
# Data Structures
# ==============================================================================


@dataclass(frozen=True)
class DefaultExposure:
    """Exposure to default risk for a single issuer/obligor.

    Attributes
    ----------
    issuer_id : str
        Unique identifier for the issuer/obligor
    instrument_type : str
        Type of instrument (bond, CDS, loan, etc.)
    notional : float
        Notional amount or market value
    credit_rating : CreditRating
        External or internal credit rating
    seniority : Seniority
        Seniority in capital structure
    sector : Sector
        Industry sector
    maturity_years : float
        Remaining maturity in years
    long_short : str
        "long" or "short" position
    lgd_override : Optional[float]
        Override LGD if not using supervisory values
    """

    issuer_id: str
    instrument_type: str
    notional: float
    credit_rating: CreditRating
    seniority: Seniority
    sector: Sector
    maturity_years: float
    long_short: str = "long"
    lgd_override: Optional[float] = None

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.long_short.lower() == "long"

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.long_short.lower() == "short"


@dataclass(frozen=True)
class SecuritizedExposure:
    """Exposure to securitized products.

    Attributes
    ----------
    instrument_id : str
        Unique identifier
    securitization_type : SecuritizationType
        Type of securitization
    notional : float
        Notional or market value
    tranche_attachment : float
        Attachment point (0-1)
    tranche_detachment : float
        Detachment point (0-1)
    credit_rating : CreditRating
        Rating of the tranche
    underlying_pool_rating : CreditRating
        Average rating of underlying pool
    long_short : str
        Position direction
    """

    instrument_id: str
    securitization_type: SecuritizationType
    notional: float
    tranche_attachment: float
    tranche_detachment: float
    credit_rating: CreditRating
    underlying_pool_rating: CreditRating
    long_short: str = "long"


@dataclass(frozen=True)
class DRCResult:
    """Result of DRC calculation.

    Attributes
    ----------
    total_drc : float
        Total default risk charge
    non_securitized_drc : float
        DRC for non-securitized exposures
    securitized_drc : float
        DRC for securitized products
    net_long_jt

d : float
        Net long jump-to-default risk
    net_short_jtd : float
        Net short jump-to-default risk
    drc_by_issuer : Dict[str, float]
        Per-issuer DRC breakdown
    drc_by_sector : Dict[Sector, float]
        Per-sector DRC breakdown
    """

    total_drc: float
    non_securitized_drc: float
    securitized_drc: float
    net_long_jtd: float
    net_short_jtd: float
    drc_by_issuer: Dict[str, float]
    drc_by_sector: Dict[Sector, float]


# ==============================================================================
# Risk Weight and LGD Calibrations
# ==============================================================================


# Supervisory risk weights by rating (FIRB approach, MAR21.6)
SUPERVISORY_RISK_WEIGHTS: Dict[CreditRating, float] = {
    CreditRating.AAA: 0.005,
    CreditRating.AA: 0.005,
    CreditRating.A: 0.005,
    CreditRating.BBB: 0.01,
    CreditRating.BB: 0.02,
    CreditRating.B: 0.05,
    CreditRating.CCC_OR_BELOW: 0.10,
    CreditRating.UNRATED: 0.02,  # Treat as BB
}

# Supervisory LGD by seniority (MAR21.7)
SUPERVISORY_LGD: Dict[Seniority, float] = {
    Seniority.SENIOR_SECURED: 0.40,
    Seniority.SENIOR_UNSECURED: 0.40,
    Seniority.SUBORDINATED: 0.60,
    Seniority.EQUITY: 1.00,  # Full loss on default
}

# Default correlation by sector (simplified)
DEFAULT_CORRELATION_WITHIN_SECTOR: Dict[Sector, float] = {
    Sector.GOVERNMENT: 0.50,
    Sector.FINANCIAL: 0.50,
    Sector.INDUSTRIAL: 0.40,
    Sector.UTILITIES: 0.40,
    Sector.RETAIL: 0.40,
    Sector.TECHNOLOGY: 0.40,
    Sector.HEALTHCARE: 0.40,
    Sector.ENERGY: 0.45,
    Sector.OTHER: 0.35,
}

DEFAULT_CORRELATION_CROSS_SECTOR = 0.25

# Securitization risk weights (FIRB approach)
SECURITIZATION_RISK_WEIGHTS: Dict[CreditRating, float] = {
    CreditRating.AAA: 0.007,
    CreditRating.AA: 0.008,
    CreditRating.A: 0.010,
    CreditRating.BBB: 0.025,
    CreditRating.BB: 0.070,
    CreditRating.B: 0.150,
    CreditRating.CCC_OR_BELOW: 0.300,
    CreditRating.UNRATED: 0.070,
}


# ==============================================================================
# DRC Calculator
# ==============================================================================


class FRTBDefaultRiskCharge:
    """Calculate Default Risk Charge under FRTB.

    The DRC captures jump-to-default risk for:
    1. Non-securitized credit instruments (bonds, CDS, loans)
    2. Securitized products (RMBS, CMBS, ABS, CDO, CLO)

    Calculation approach:
    1. Compute gross JTD (jump-to-default) for each position
    2. Aggregate using default correlation
    3. Separate treatment for long and short positions
    4. Apply netting with hedging recognition
    """

    def __init__(
        self,
        within_sector_correlation: Optional[Dict[Sector, float]] = None,
        cross_sector_correlation: float = DEFAULT_CORRELATION_CROSS_SECTOR,
    ):
        """Initialize DRC calculator.

        Parameters
        ----------
        within_sector_correlation : Optional[Dict[Sector, float]]
            Correlation within each sector
        cross_sector_correlation : float
            Correlation across sectors
        """
        self.within_sector_correlation = (
            within_sector_correlation or DEFAULT_CORRELATION_WITHIN_SECTOR
        )
        self.cross_sector_correlation = cross_sector_correlation

    def calculate(
        self,
        non_securitized: List[DefaultExposure],
        securitized: Optional[List[SecuritizedExposure]] = None,
    ) -> DRCResult:
        """Calculate total DRC.

        Parameters
        ----------
        non_securitized : List[DefaultExposure]
            Non-securitized credit exposures
        securitized : Optional[List[SecuritizedExposure]]
            Securitized product exposures

        Returns
        -------
        DRCResult
            Default risk charge components
        """
        # Calculate non-securitized DRC
        if non_securitized:
            non_sec_drc, net_long, net_short, by_issuer, by_sector = self._calculate_non_securitized(
                non_securitized
            )
        else:
            non_sec_drc = 0.0
            net_long = 0.0
            net_short = 0.0
            by_issuer = {}
            by_sector = {}

        # Calculate securitized DRC
        if securitized:
            sec_drc = self._calculate_securitized(securitized)
        else:
            sec_drc = 0.0

        total_drc = non_sec_drc + sec_drc

        return DRCResult(
            total_drc=total_drc,
            non_securitized_drc=non_sec_drc,
            securitized_drc=sec_drc,
            net_long_jtd=net_long,
            net_short_jtd=net_short,
            drc_by_issuer=by_issuer,
            drc_by_sector=by_sector,
        )

    def _calculate_non_securitized(
        self, exposures: List[DefaultExposure]
    ) -> Tuple[float, float, float, Dict[str, float], Dict[Sector, float]]:
        """Calculate DRC for non-securitized exposures.

        Returns
        -------
        Tuple containing:
            - Total DRC
            - Net long JTD
            - Net short JTD
            - DRC by issuer
            - DRC by sector
        """
        # Group by issuer
        issuer_exposures: Dict[str, List[DefaultExposure]] = {}
        for exp in exposures:
            if exp.issuer_id not in issuer_exposures:
                issuer_exposures[exp.issuer_id] = []
            issuer_exposures[exp.issuer_id].append(exp)

        # Calculate gross JTD for each issuer
        issuer_jtds = {}
        issuer_sectors = {}
        for issuer_id, exps in issuer_exposures.items():
            net_jtd = self._calculate_issuer_jtd(exps)
            issuer_jtds[issuer_id] = net_jtd
            issuer_sectors[issuer_id] = exps[0].sector  # Assume same sector per issuer

        # Separate long and short
        long_jtds = {k: v for k, v in issuer_jtds.items() if v > 0}
        short_jtds = {k: abs(v) for k, v in issuer_jtds.items() if v < 0}

        # Aggregate with correlation
        long_drc = self._aggregate_jtds(long_jtds, issuer_sectors)
        short_drc = self._aggregate_jtds(short_jtds, issuer_sectors)

        # Net long and short
        net_long = sum(long_jtds.values())
        net_short = sum(short_jtds.values())

        # Total DRC with partial netting
        total_drc = jnp.sqrt(long_drc**2 + short_drc**2)

        # By-issuer breakdown
        drc_by_issuer = {k: abs(v) for k, v in issuer_jtds.items()}

        # By-sector breakdown
        drc_by_sector: Dict[Sector, float] = {}
        for issuer_id, jtd in issuer_jtds.items():
            sector = issuer_sectors[issuer_id]
            drc_by_sector[sector] = drc_by_sector.get(sector, 0.0) + abs(jtd)

        return float(total_drc), net_long, net_short, drc_by_issuer, drc_by_sector

    def _calculate_issuer_jtd(self, exposures: List[DefaultExposure]) -> float:
        """Calculate net JTD for a single issuer across all instruments.

        JTD = Notional × LGD × Risk_Weight
        """
        net_jtd = 0.0
        for exp in exposures:
            # Get LGD
            lgd = exp.lgd_override if exp.lgd_override is not None else SUPERVISORY_LGD[exp.seniority]

            # Get risk weight
            rw = SUPERVISORY_RISK_WEIGHTS[exp.credit_rating]

            # Calculate JTD
            jtd = exp.notional * lgd * rw

            # Apply sign
            if exp.is_short:
                jtd = -jtd

            net_jtd += jtd

        return net_jtd

    def _aggregate_jtds(
        self,
        jtds: Dict[str, float],
        sectors: Dict[str, Sector],
    ) -> float:
        """Aggregate JTDs with default correlation.

        Uses simplified correlation approach:
        Capital = sqrt(Σ JTD_i² + ΣΣ ρ_ij × JTD_i × JTD_j)
        """
        if not jtds:
            return 0.0

        issuers = list(jtds.keys())
        n = len(issuers)

        # Build correlation matrix
        corr_matrix = jnp.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                issuer_i = issuers[i]
                issuer_j = issuers[j]
                sector_i = sectors[issuer_i]
                sector_j = sectors[issuer_j]

                if sector_i == sector_j:
                    # Same sector
                    corr = self.within_sector_correlation[sector_i]
                else:
                    # Cross sector
                    corr = self.cross_sector_correlation

                corr_matrix = corr_matrix.at[i, j].set(corr)
                corr_matrix = corr_matrix.at[j, i].set(corr)

        # JTD vector
        jtd_vector = jnp.array([jtds[issuer] for issuer in issuers])

        # Quadratic form: JTD' × Corr × JTD
        capital_squared = jtd_vector @ corr_matrix @ jtd_vector
        capital = jnp.sqrt(jnp.maximum(capital_squared, 0.0))

        return float(capital)

    def _calculate_securitized(self, exposures: List[SecuritizedExposure]) -> float:
        """Calculate DRC for securitized products.

        Securitizations use a simplified approach:
        - Gross notional weighted by supervisory risk weight
        - Limited netting between tranches
        """
        total_charge = 0.0

        for exp in exposures:
            # Risk weight based on tranche rating
            rw = SECURITIZATION_RISK_WEIGHTS[exp.credit_rating]

            # Tranche thickness
            thickness = exp.tranche_detachment - exp.tranche_attachment

            # Capital charge (simplified)
            charge = exp.notional * rw * thickness

            # Apply sign
            if exp.long_short.lower() == "short":
                charge = -charge

            total_charge += charge

        # Simple aggregation (no correlation benefit for securitizations)
        return abs(total_charge)


# ==============================================================================
# Utility Functions
# ==============================================================================


def map_external_rating_to_cqs(rating: str) -> CreditRating:
    """Map external rating string to CreditRating enum.

    Parameters
    ----------
    rating : str
        External rating (e.g., "Aaa", "BBB+", "B-")

    Returns
    -------
    CreditRating
        Mapped credit rating enum
    """
    rating_upper = rating.upper().strip()

    # Remove +/- modifiers
    if rating_upper.endswith("+") or rating_upper.endswith("-"):
        rating_upper = rating_upper[:-1]

    # Map to standard ratings
    if rating_upper in ["AAA", "AAA"]:
        return CreditRating.AAA
    elif rating_upper in ["AA", "AA1", "AA2", "AA3"]:
        return CreditRating.AA
    elif rating_upper in ["A", "A1", "A2", "A3"]:
        return CreditRating.A
    elif rating_upper in ["BBB", "BAA", "BAA1", "BAA2", "BAA3"]:
        return CreditRating.BBB
    elif rating_upper in ["BB", "BA", "BA1", "BA2", "BA3"]:
        return CreditRating.BB
    elif rating_upper in ["B", "B1", "B2", "B3"]:
        return CreditRating.B
    elif rating_upper in ["CCC", "CC", "C", "CAA", "CA"]:
        return CreditRating.CCC_OR_BELOW
    else:
        return CreditRating.UNRATED


def calculate_lgd_from_recovery(recovery_rate: float) -> float:
    """Calculate Loss-Given-Default from recovery rate.

    Parameters
    ----------
    recovery_rate : float
        Recovery rate (0-1)

    Returns
    -------
    float
        LGD = 1 - recovery_rate
    """
    return 1.0 - recovery_rate


__all__ = [
    # Enums
    "CreditRating",
    "Seniority",
    "Sector",
    "SecuritizationType",
    # Data Structures
    "DefaultExposure",
    "SecuritizedExposure",
    "DRCResult",
    # Calculator
    "FRTBDefaultRiskCharge",
    # Utility Functions
    "map_external_rating_to_cqs",
    "calculate_lgd_from_recovery",
    # Constants
    "SUPERVISORY_RISK_WEIGHTS",
    "SUPERVISORY_LGD",
    "SECURITIZATION_RISK_WEIGHTS",
]
