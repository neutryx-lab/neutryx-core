"""Tests for NettingSet model."""
from __future__ import annotations

import pytest

from neutryx.portfolio.netting_set import NettingSet


def test_netting_set_basic():
    """Test basic NettingSet creation."""
    ns = NettingSet(
        id="NS001",
        name="Bank ABC - Derivatives",
        master_agreement_id="MA001",
        counterparty_id="CP001",
    )
    assert ns.id == "NS001"
    assert ns.name == "Bank ABC - Derivatives"
    assert ns.master_agreement_id == "MA001"
    assert ns.counterparty_id == "CP001"
    assert ns.is_empty()
    assert not ns.has_csa()


def test_netting_set_with_csa():
    """Test NettingSet with CSA."""
    ns = NettingSet(
        id="NS002",
        master_agreement_id="MA001",
        counterparty_id="CP001",
        csa_id="CSA001",
    )
    assert ns.has_csa()
    assert ns.csa_id == "CSA001"


def test_netting_set_add_remove_trades():
    """Test adding and removing trades."""
    ns = NettingSet(
        id="NS003",
        master_agreement_id="MA001",
        counterparty_id="CP001",
    )

    # Add trades
    ns.add_trade("TRD001")
    assert ns.num_trades() == 1
    assert ns.contains_trade("TRD001")
    assert not ns.is_empty()

    ns.add_trade("TRD002")
    ns.add_trade("TRD003")
    assert ns.num_trades() == 3

    # Adding duplicate should not increase count
    ns.add_trade("TRD001")
    assert ns.num_trades() == 3

    # Remove trade
    removed = ns.remove_trade("TRD002")
    assert removed
    assert ns.num_trades() == 2
    assert not ns.contains_trade("TRD002")

    # Remove non-existent trade
    removed_none = ns.remove_trade("TRD999")
    assert not removed_none
    assert ns.num_trades() == 2


def test_netting_set_cleared():
    """Test cleared netting set with clearinghouse."""
    ns = NettingSet(
        id="NS004",
        master_agreement_id="MA001",
        counterparty_id="CP001",
        is_cleared=True,
        clearinghouse_id="CCP001",
    )
    assert ns.is_cleared
    assert ns.clearinghouse_id == "CCP001"
    assert not ns.is_bilaterally_cleared()


def test_netting_set_bilateral():
    """Test bilateral (non-cleared) netting set."""
    ns = NettingSet(
        id="NS005",
        master_agreement_id="MA001",
        counterparty_id="CP001",
        is_cleared=False,
    )
    assert not ns.is_cleared
    assert ns.is_bilaterally_cleared()


def test_netting_set_clearinghouse_validation():
    """Test that clearinghouse_id is required when is_cleared=True."""
    # This should raise a validation error
    with pytest.raises(Exception):  # Pydantic validation error
        NettingSet(
            id="NS006",
            master_agreement_id="MA001",
            counterparty_id="CP001",
            is_cleared=True,
            # Missing clearinghouse_id
        )


def test_netting_set_get_display_name():
    """Test get_display_name method."""
    # With name
    ns1 = NettingSet(
        id="NS007",
        name="My Netting Set",
        master_agreement_id="MA001",
        counterparty_id="CP001",
    )
    assert ns1.get_display_name() == "My Netting Set"

    # Without name (fallback to ID)
    ns2 = NettingSet(
        id="NS008",
        master_agreement_id="MA001",
        counterparty_id="CP001",
    )
    assert ns2.get_display_name() == "NS008"


def test_netting_set_repr():
    """Test string representation."""
    ns = NettingSet(
        id="NS009",
        master_agreement_id="MA001",
        counterparty_id="CP001",
        csa_id="CSA001",
        trade_ids=["TRD001", "TRD002", "TRD003"],
    )
    repr_str = repr(ns)
    assert "NS009" in repr_str
    assert "CP001" in repr_str
    assert "trades=3" in repr_str
    assert "CSA=CSA001" in repr_str


def test_netting_set_trade_list():
    """Test initializing with trade list."""
    ns = NettingSet(
        id="NS010",
        master_agreement_id="MA001",
        counterparty_id="CP001",
        trade_ids=["TRD001", "TRD002", "TRD003", "TRD004"],
    )
    assert ns.num_trades() == 4
    assert ns.contains_trade("TRD003")
