"""Variation Margin (VM) calculations for OTC derivatives.

Variation margin is the daily mark-to-market collateral posted between
counterparties to reflect changes in portfolio value.
"""
from __future__ import annotations

from typing import Optional

from neutryx.portfolio.contracts.csa import CSA


def calculate_variation_margin(
    portfolio_mtm: float,
    existing_collateral: float = 0.0,
    csa: Optional[CSA] = None,
    posted_by_party: Optional[str] = None,
) -> float:
    """Calculate variation margin requirement.

    Parameters
    ----------
    portfolio_mtm : float
        Net mark-to-market value of the portfolio (positive = owed to us)
    existing_collateral : float
        Currently posted collateral amount
    csa : CSA, optional
        CSA agreement with threshold and MTA terms
    posted_by_party : str, optional
        ID of party posting collateral (required if CSA is provided)

    Returns
    -------
    float
        Required variation margin amount

    Notes
    -----
    If no CSA is provided, full MTM collateralization is assumed.
    With CSA: VM = max(0, MTM - threshold) - existing_collateral
    """
    if csa is None:
        # No CSA: full collateralization
        return portfolio_mtm - existing_collateral

    if posted_by_party is None:
        raise ValueError("posted_by_party must be specified when CSA is provided")

    # Get threshold for the posting party
    party_label = csa.get_party_label(posted_by_party)
    threshold = csa.threshold_terms.get_threshold(party_label)

    # Calculate required collateral (VM only, no IA)
    required_vm = max(0.0, portfolio_mtm - threshold)

    # Net against existing collateral
    vm_requirement = required_vm - existing_collateral
    return vm_requirement


def calculate_vm_call(
    portfolio_mtm: float,
    existing_collateral: float = 0.0,
    csa: Optional[CSA] = None,
    posted_by_party: Optional[str] = None,
) -> float:
    """Calculate variation margin call amount (after rounding and MTA).

    Parameters
    ----------
    portfolio_mtm : float
        Net mark-to-market value of the portfolio
    existing_collateral : float
        Currently posted collateral amount
    csa : CSA, optional
        CSA agreement with threshold, rounding, and MTA terms
    posted_by_party : str, optional
        ID of party posting collateral (required if CSA is provided)

    Returns
    -------
    float
        Margin call amount (0 if below MTA)

    Notes
    -----
    This function applies:
    1. Threshold adjustment
    2. Rounding to nearest increment
    3. Minimum Transfer Amount (MTA) filter
    """
    # Calculate raw VM requirement
    vm_requirement = calculate_variation_margin(
        portfolio_mtm=portfolio_mtm,
        existing_collateral=existing_collateral,
        csa=csa,
        posted_by_party=posted_by_party,
    )

    if csa is None:
        # No CSA: return raw requirement
        return vm_requirement

    # Calculate new total collateral requirement
    party_label = csa.get_party_label(posted_by_party)
    threshold = csa.threshold_terms.get_threshold(party_label)
    required_collateral_gross = max(0.0, portfolio_mtm - threshold)

    # Apply rounding
    required_collateral_rounded = csa.apply_rounding(required_collateral_gross)

    # Apply MTA
    margin_call = csa.apply_mta(
        current_collateral=existing_collateral,
        required_collateral=required_collateral_rounded,
        posted_by=posted_by_party,
    )

    return margin_call


def calculate_vm_bilateral(
    our_exposure: float,
    their_exposure: float,
    our_posted_collateral: float = 0.0,
    their_posted_collateral: float = 0.0,
    csa: Optional[CSA] = None,
    our_party_id: Optional[str] = None,
    their_party_id: Optional[str] = None,
) -> tuple[float, float]:
    """Calculate bilateral variation margin (both directions).

    Parameters
    ----------
    our_exposure : float
        Our exposure to the counterparty (MTM from our perspective)
    their_exposure : float
        Their exposure to us (MTM from their perspective, should be -our_exposure)
    our_posted_collateral : float
        Collateral we have posted to them
    their_posted_collateral : float
        Collateral they have posted to us
    csa : CSA, optional
        CSA agreement
    our_party_id : str, optional
        Our party ID (required if CSA provided)
    their_party_id : str, optional
        Their party ID (required if CSA provided)

    Returns
    -------
    tuple[float, float]
        (we_post, they_post) - amounts to be posted by each party

    Notes
    -----
    In bilateral VM:
    - If our_exposure > 0, they owe us, so we receive VM
    - If their_exposure > 0 (our_exposure < 0), we owe them, so we post VM
    """
    # Calculate what they should post to us
    if our_exposure > 0:
        they_post = calculate_vm_call(
            portfolio_mtm=our_exposure,
            existing_collateral=their_posted_collateral,
            csa=csa,
            posted_by_party=their_party_id,
        )
    else:
        they_post = 0.0

    # Calculate what we should post to them
    if their_exposure > 0:
        we_post = calculate_vm_call(
            portfolio_mtm=their_exposure,
            existing_collateral=our_posted_collateral,
            csa=csa,
            posted_by_party=our_party_id,
        )
    else:
        we_post = 0.0

    return we_post, they_post


def calculate_vm_portfolio_level(
    trades_mtm: list[float],
    netting_enabled: bool = True,
) -> float:
    """Calculate portfolio-level variation margin before CSA adjustments.

    Parameters
    ----------
    trades_mtm : list[float]
        MTM values for all trades in the netting set
    netting_enabled : bool
        Whether bilateral netting is enabled (default True)

    Returns
    -------
    float
        Net portfolio MTM (basis for VM calculation)

    Notes
    -----
    With netting: VM based on sum(MTM)
    Without netting: VM based on max(0, sum(positive MTM))
    """
    if netting_enabled:
        # Bilateral netting: sum all MTMs
        return sum(trades_mtm)
    else:
        # No netting: only positive exposures
        return sum(max(0.0, mtm) for mtm in trades_mtm)
