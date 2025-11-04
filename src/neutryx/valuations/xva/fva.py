"""Funding Valuation Adjustment (FVA) calculation.

FVA represents the cost of funding the expected positive exposure of a
derivative portfolio. It accounts for the difference between the risk-free
rate and the actual funding rate available to the institution.
"""


def fva(epe_t, funding_spread, df_t):
    """Calculate Funding Valuation Adjustment (FVA).

    FVA quantifies the funding costs associated with entering into uncollateralized
    derivatives. It is calculated as the present value of funding spreads applied
    to the expected positive exposure over the life of the trade.

    Parameters
    ----------
    epe_t : Array
        Expected Positive Exposure at each time step, shape [time_steps]
    funding_spread : float or Array
        Funding spread over risk-free rate (in basis points or decimal).
        Can be scalar or array of shape [time_steps]
    df_t : Array
        Discount factors at each time step, shape [time_steps]

    Returns
    -------
    float
        Funding Valuation Adjustment (present value of funding costs)

    Notes
    -----
    FVA = sum_t [ DF(t) * EPE(t) * FundingSpread(t) ]

    This is a simplified implementation. In practice, FVA calculations may
    distinguish between:
    - FBA (Funding Benefit Adjustment) for negative exposure
    - Different funding curves for borrowing vs. lending
    - Collateral effects and margin agreements
    """
    return (df_t * epe_t * funding_spread).sum()
