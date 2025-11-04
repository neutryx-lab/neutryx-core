"""Margin Valuation Adjustment (MVA) calculation.

MVA represents the cost of funding the initial margin requirements for
centrally cleared or bilaterally collateralized derivatives. It captures
the present value of the cost of posting margin over the life of a trade.
"""


def mva(initial_margin_profile, df_t, spread):
    """Calculate Margin Valuation Adjustment (MVA).

    MVA quantifies the funding cost of initial margin that must be posted
    for derivatives subject to margin requirements (e.g., centrally cleared
    derivatives or non-cleared margin rules under SIMM).

    Parameters
    ----------
    initial_margin_profile : Array
        Expected initial margin requirements at each time step, shape [time_steps]
    df_t : Array
        Discount factors at each time step, shape [time_steps]
    spread : float or Array
        Funding spread applied to margin (cost of funding posted collateral).
        Can be scalar or array of shape [time_steps]

    Returns
    -------
    float
        Margin Valuation Adjustment (present value of margin funding costs)

    Notes
    -----
    MVA = sum_t [ DF(t) * IM(t) * Spread(t) ]

    where:
    - IM(t) is the initial margin profile
    - DF(t) is the discount factor
    - Spread(t) is the funding cost spread

    MVA is particularly relevant for:
    - Centrally cleared derivatives (CCP margin requirements)
    - Non-cleared swaps under regulatory margin rules
    - SIMM (Standardized Initial Margin Model) calculations
    """
    return (df_t * initial_margin_profile * spread).sum()
