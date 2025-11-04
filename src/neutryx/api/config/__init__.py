"""Configuration constants for API calculations."""

# XVA Default Configuration
# =========================

# CVA (Credit Valuation Adjustment) Defaults
DEFAULT_COUNTERPARTY_PD_ANNUAL = 0.01  # 1% annual default probability
DEFAULT_LGD = 0.6  # Loss Given Default (60%)

# DVA (Debit Valuation Adjustment) Defaults
DEFAULT_OWN_PD_ANNUAL = 0.005  # 0.5% annual default probability (our own default)
DEFAULT_OWN_LGD = 0.6  # Our own Loss Given Default

# FVA (Funding Valuation Adjustment) Defaults
DEFAULT_FUNDING_SPREAD_BPS = 50.0  # 50 basis points

# MVA (Margin Valuation Adjustment) Defaults
DEFAULT_IM_RATIO = 0.06  # 6% of gross notional as initial margin estimate
DEFAULT_IM_SPREAD = 0.015  # 1.5% (150 bps) spread on initial margin

# Monte Carlo Defaults
# ====================
DEFAULT_MC_STEPS = 252  # One year of daily steps
DEFAULT_MC_PATHS = 100_000  # 100k paths
DEFAULT_SEED = 42  # Default random seed


__all__ = [
    "DEFAULT_COUNTERPARTY_PD_ANNUAL",
    "DEFAULT_LGD",
    "DEFAULT_OWN_PD_ANNUAL",
    "DEFAULT_OWN_LGD",
    "DEFAULT_FUNDING_SPREAD_BPS",
    "DEFAULT_IM_RATIO",
    "DEFAULT_IM_SPREAD",
    "DEFAULT_MC_STEPS",
    "DEFAULT_MC_PATHS",
    "DEFAULT_SEED",
]
