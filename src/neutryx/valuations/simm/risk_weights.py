"""ISDA SIMM risk weights and correlations.

This module provides risk weights, concentration thresholds, and correlation
parameters from the ISDA SIMM methodology (based on SIMM 2.6).

Note: This is a simplified implementation. Production use requires the full
ISDA SIMM specification with all buckets, sub-curves, and calibrations.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, Optional

import jax.numpy as jnp


class RiskClass(str, Enum):
    """SIMM risk class."""

    INTEREST_RATE = "InterestRate"
    CREDIT_QUALIFYING = "CreditQualifying"
    CREDIT_NON_QUALIFYING = "CreditNonQualifying"
    EQUITY = "Equity"
    COMMODITY = "Commodity"
    FX = "FX"


# ============================================================================
# DELTA RISK WEIGHTS
# ============================================================================

# Interest Rate Delta Risk Weights (by currency)
IR_DELTA_RISK_WEIGHTS: Dict[str, Dict[str, float]] = {
    "USD": {
        "2W": 109.0,
        "1M": 109.0,
        "3M": 99.0,
        "6M": 87.0,
        "1Y": 76.0,
        "2Y": 64.0,
        "3Y": 58.0,
        "5Y": 52.0,
        "10Y": 52.0,
        "15Y": 52.0,
        "20Y": 54.0,
        "30Y": 63.0,
    },
    "EUR": {
        "2W": 112.0,
        "1M": 112.0,
        "3M": 99.0,
        "6M": 85.0,
        "1Y": 71.0,
        "2Y": 60.0,
        "3Y": 54.0,
        "5Y": 50.0,
        "10Y": 50.0,
        "15Y": 50.0,
        "20Y": 52.0,
        "30Y": 60.0,
    },
    "GBP": {
        "2W": 112.0,
        "1M": 112.0,
        "3M": 98.0,
        "6M": 83.0,
        "1Y": 69.0,
        "2Y": 58.0,
        "3Y": 52.0,
        "5Y": 48.0,
        "10Y": 48.0,
        "15Y": 48.0,
        "20Y": 49.0,
        "30Y": 57.0,
    },
    # Add other currencies as needed
}

# FX Delta Risk Weight (single weight for all currency pairs)
FX_DELTA_RISK_WEIGHT = 10.0  # Simplified: actual SIMM has category-based weights

# Equity Delta Risk Weights (by bucket)
EQUITY_DELTA_RISK_WEIGHTS: Dict[str, float] = {
    "1": 26.0,  # Emerging Markets - Consumer, Utilities
    "2": 28.0,  # Emerging Markets - Telecom, Industrials
    "3": 34.0,  # Emerging Markets - Heavy Industrials
    "4": 29.0,  # Emerging Markets - Other
    "5": 24.0,  # Developed Markets - Consumer, Utilities
    "6": 26.0,  # Developed Markets - Telecom, Industrials
    "7": 30.0,  # Developed Markets - Heavy Industrials
    "8": 27.0,  # Developed Markets - Other
    "9": 18.0,  # Indices, Funds, ETFs (Emerging)
    "10": 16.0,  # Indices, Funds, ETFs (Developed)
    "11": 16.0,  # Indices, Funds, ETFs (Volatility)
    "12": 32.0,  # Other
}

# Credit Qualifying Delta Risk Weights (by bucket)
CREDIT_Q_DELTA_RISK_WEIGHTS: Dict[str, float] = {
    "1": 85.0,  # Sovereigns (IG)
    "2": 85.0,  # Sovereigns (HY)
    "3": 73.0,  # Financials (IG)
    "4": 73.0,  # Financials (HY)
    "5": 58.0,  # Basic Materials (IG)
    "6": 43.0,  # Consumer (IG)
    "7": 161.0,  # TMT (IG)
    "8": 238.0,  # Other (IG)
    "9": 151.0,  # Indices (IG)
    "10": 210.0,  # Indices (HY)
    "11": 141.0,  # Structured (IG)
    "12": 102.0,  # Structured (HY)
}

# ============================================================================
# VEGA RISK WEIGHTS
# ============================================================================

# Simplified vega risk weights (percentage of delta RW)
VEGA_RW_MULTIPLIER = {
    RiskClass.INTEREST_RATE: 0.18,  # ~18% of delta RW
    RiskClass.FX: 0.47,
    RiskClass.EQUITY: 0.78,
    RiskClass.CREDIT_QUALIFYING: 0.40,
    RiskClass.COMMODITY: 0.61,
}

# ============================================================================
# CONCENTRATION THRESHOLDS
# ============================================================================

CONCENTRATION_THRESHOLDS: Dict[RiskClass, Dict[str, float]] = {
    RiskClass.INTEREST_RATE: {
        "USD": 230_000_000.0,
        "EUR": 230_000_000.0,
        "GBP": 230_000_000.0,
    },
    RiskClass.FX: {
        "Category1": 8_400_000.0,  # High volatility pairs
        "Category2": 1_900_000.0,  # Significantly high volatility pairs
        "Category3": 560_000.0,  # Others
    },
    RiskClass.EQUITY: {
        "1": 9_600_000.0,
        "2": 9_600_000.0,
        "3": 9_600_000.0,
        "4": 9_600_000.0,
        "5": 20_000_000.0,
        "6": 20_000_000.0,
        "7": 20_000_000.0,
        "8": 20_000_000.0,
        "9": 20_000_000.0,
        "10": 70_000_000.0,
        "11": 20_000_000.0,
        "12": 9_600_000.0,
    },
}

# ============================================================================
# CORRELATION PARAMETERS
# ============================================================================

# Within-bucket correlations (simplified)
WITHIN_BUCKET_CORRELATION = {
    RiskClass.INTEREST_RATE: 0.99,  # Very high correlation within currency
    RiskClass.FX: 0.50,
    RiskClass.EQUITY: 0.15,  # Low correlation within equity bucket
    RiskClass.CREDIT_QUALIFYING: 0.35,
}

# Cross-bucket correlations
CROSS_BUCKET_CORRELATION = {
    RiskClass.INTEREST_RATE: 0.27,  # Between different currencies
    RiskClass.FX: 0.60,
    RiskClass.EQUITY: 0.15,  # Between equity buckets
    RiskClass.CREDIT_QUALIFYING: 0.21,
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_risk_weights(
    risk_class: RiskClass,
    bucket: Optional[str] = None,
    tenor: Optional[str] = None,
) -> float:
    """Get delta risk weight for a risk class.

    Parameters
    ----------
    risk_class : RiskClass
        Risk class
    bucket : str, optional
        Bucket identifier (required for some risk classes)
    tenor : str, optional
        Tenor (required for IR)

    Returns
    -------
    float
        Risk weight

    Raises
    ------
    ValueError
        If required parameters are missing
    """
    if risk_class == RiskClass.INTEREST_RATE:
        if bucket is None or tenor is None:
            raise ValueError("IR risk weight requires bucket (currency) and tenor")
        currency_weights = IR_DELTA_RISK_WEIGHTS.get(bucket)
        if currency_weights is None:
            # Fallback to USD weights
            currency_weights = IR_DELTA_RISK_WEIGHTS["USD"]
        return currency_weights.get(tenor, 100.0)

    elif risk_class == RiskClass.FX:
        return FX_DELTA_RISK_WEIGHT

    elif risk_class == RiskClass.EQUITY:
        if bucket is None:
            raise ValueError("Equity risk weight requires bucket")
        return EQUITY_DELTA_RISK_WEIGHTS.get(bucket, 30.0)

    elif risk_class == RiskClass.CREDIT_QUALIFYING:
        if bucket is None:
            raise ValueError("Credit risk weight requires bucket")
        return CREDIT_Q_DELTA_RISK_WEIGHTS.get(bucket, 100.0)

    else:
        return 100.0  # Default


def get_vega_risk_weight(risk_class: RiskClass, delta_rw: float) -> float:
    """Get vega risk weight.

    Parameters
    ----------
    risk_class : RiskClass
        Risk class
    delta_rw : float
        Delta risk weight

    Returns
    -------
    float
        Vega risk weight

    Notes
    -----
    Vega RW = multiplier * delta RW
    """
    multiplier = VEGA_RW_MULTIPLIER.get(risk_class, 0.50)
    return multiplier * delta_rw


def get_correlations(
    risk_class: RiskClass,
    within_bucket: bool = True,
) -> float:
    """Get correlation parameter for aggregation.

    Parameters
    ----------
    risk_class : RiskClass
        Risk class
    within_bucket : bool
        Whether correlation is within bucket (True) or cross-bucket (False)

    Returns
    -------
    float
        Correlation parameter
    """
    if within_bucket:
        return WITHIN_BUCKET_CORRELATION.get(risk_class, 0.50)
    else:
        return CROSS_BUCKET_CORRELATION.get(risk_class, 0.30)


def get_concentration_threshold(
    risk_class: RiskClass,
    bucket: str,
) -> float:
    """Get concentration threshold for a bucket.

    Parameters
    ----------
    risk_class : RiskClass
        Risk class
    bucket : str
        Bucket identifier

    Returns
    -------
    float
        Concentration threshold in base currency
    """
    thresholds = CONCENTRATION_THRESHOLDS.get(risk_class, {})
    return thresholds.get(bucket, 1_000_000_000.0)  # Large default


def apply_concentration_risk(
    weighted_sensitivity: float,
    concentration_threshold: float,
) -> float:
    """Apply concentration risk factor.

    Parameters
    ----------
    weighted_sensitivity : float
        Absolute weighted sensitivity
    concentration_threshold : float
        Concentration threshold

    Returns
    -------
    float
        Concentration risk factor (≥ 1.0)

    Notes
    -----
    CR = max(1, sqrt(abs(WS) / T))
    where WS = weighted sensitivity, T = threshold
    """
    if concentration_threshold <= 0:
        return 1.0

    ratio = abs(weighted_sensitivity) / concentration_threshold
    cr = max(1.0, jnp.sqrt(ratio))
    return float(cr)
