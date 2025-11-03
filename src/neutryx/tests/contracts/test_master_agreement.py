"""Tests for Master Agreement models."""
from __future__ import annotations

from datetime import date

import pytest

from neutryx.contracts.master_agreement import (
    AgreementType,
    GoverningLaw,
    MasterAgreement,
    TerminationCurrency,
)


def test_master_agreement_basic():
    """Test basic MasterAgreement creation."""
    ma = MasterAgreement(
        id="MA001",
        agreement_type=AgreementType.ISDA_2002,
        party_a_id="CP001",
        party_b_id="CP002",
        effective_date=date(2020, 1, 1),
    )
    assert ma.id == "MA001"
    assert ma.agreement_type == AgreementType.ISDA_2002
    assert ma.party_a_id == "CP001"
    assert ma.party_b_id == "CP002"
    assert ma.governing_law == GoverningLaw.ENGLISH  # default
    assert ma.payment_netting
    assert ma.close_out_netting


def test_master_agreement_with_csa():
    """Test MasterAgreement with associated CSA."""
    ma = MasterAgreement(
        id="MA002",
        agreement_type=AgreementType.ISDA_2002,
        party_a_id="CP001",
        party_b_id="CP002",
        effective_date=date(2020, 1, 1),
        csa_id="CSA001",
    )
    assert ma.has_csa()
    assert ma.csa_id == "CSA001"


def test_master_agreement_governing_law():
    """Test different governing laws."""
    ma_ny = MasterAgreement(
        id="MA003",
        agreement_type=AgreementType.ISDA_1992,
        party_a_id="CP001",
        party_b_id="CP002",
        effective_date=date(2020, 1, 1),
        governing_law=GoverningLaw.NEW_YORK,
    )
    assert ma_ny.governing_law == GoverningLaw.NEW_YORK

    ma_jp = MasterAgreement(
        id="MA004",
        agreement_type=AgreementType.ISDA_2002,
        party_a_id="CP001",
        party_b_id="CP003",
        effective_date=date(2020, 1, 1),
        governing_law=GoverningLaw.JAPANESE,
    )
    assert ma_jp.governing_law == GoverningLaw.JAPANESE


def test_master_agreement_is_party():
    """Test is_party method."""
    ma = MasterAgreement(
        id="MA005",
        agreement_type=AgreementType.ISDA_2002,
        party_a_id="CP001",
        party_b_id="CP002",
        effective_date=date(2020, 1, 1),
    )
    assert ma.is_party("CP001")
    assert ma.is_party("CP002")
    assert not ma.is_party("CP003")


def test_master_agreement_get_other_party():
    """Test get_other_party method."""
    ma = MasterAgreement(
        id="MA006",
        agreement_type=AgreementType.ISDA_2002,
        party_a_id="CP001",
        party_b_id="CP002",
        effective_date=date(2020, 1, 1),
    )
    assert ma.get_other_party("CP001") == "CP002"
    assert ma.get_other_party("CP002") == "CP001"

    with pytest.raises(ValueError, match="not a party"):
        ma.get_other_party("CP003")


def test_master_agreement_allows_netting():
    """Test allows_netting method."""
    # Both payment and close-out netting enabled
    ma1 = MasterAgreement(
        id="MA007",
        agreement_type=AgreementType.ISDA_2002,
        party_a_id="CP001",
        party_b_id="CP002",
        effective_date=date(2020, 1, 1),
        payment_netting=True,
        close_out_netting=True,
    )
    assert ma1.allows_netting()

    # Only payment netting
    ma2 = MasterAgreement(
        id="MA008",
        agreement_type=AgreementType.ISDA_2002,
        party_a_id="CP001",
        party_b_id="CP002",
        effective_date=date(2020, 1, 1),
        payment_netting=True,
        close_out_netting=False,
    )
    assert ma2.allows_netting()

    # No netting
    ma3 = MasterAgreement(
        id="MA009",
        agreement_type=AgreementType.CUSTOM,
        party_a_id="CP001",
        party_b_id="CP002",
        effective_date=date(2020, 1, 1),
        payment_netting=False,
        close_out_netting=False,
    )
    assert not ma3.allows_netting()


def test_master_agreement_is_isda():
    """Test is_isda method."""
    ma_isda = MasterAgreement(
        id="MA010",
        agreement_type=AgreementType.ISDA_2002,
        party_a_id="CP001",
        party_b_id="CP002",
        effective_date=date(2020, 1, 1),
    )
    assert ma_isda.is_isda()

    ma_gmra = MasterAgreement(
        id="MA011",
        agreement_type=AgreementType.GMRA,
        party_a_id="CP001",
        party_b_id="CP002",
        effective_date=date(2020, 1, 1),
    )
    assert not ma_gmra.is_isda()


def test_master_agreement_get_isda_version():
    """Test get_isda_version method."""
    ma_1992 = MasterAgreement(
        id="MA012",
        agreement_type=AgreementType.ISDA_1992,
        party_a_id="CP001",
        party_b_id="CP002",
        effective_date=date(2020, 1, 1),
    )
    assert ma_1992.get_isda_version() == 1992

    ma_2002 = MasterAgreement(
        id="MA013",
        agreement_type=AgreementType.ISDA_2002,
        party_a_id="CP001",
        party_b_id="CP002",
        effective_date=date(2020, 1, 1),
    )
    assert ma_2002.get_isda_version() == 2002

    ma_2021 = MasterAgreement(
        id="MA014",
        agreement_type=AgreementType.ISDA_2021,
        party_a_id="CP001",
        party_b_id="CP002",
        effective_date=date(2020, 1, 1),
    )
    assert ma_2021.get_isda_version() == 2021

    ma_custom = MasterAgreement(
        id="MA015",
        agreement_type=AgreementType.CUSTOM,
        party_a_id="CP001",
        party_b_id="CP002",
        effective_date=date(2020, 1, 1),
    )
    assert ma_custom.get_isda_version() is None


def test_master_agreement_supports_bilateral_netting():
    """Test supports_bilateral_netting method."""
    # With close-out netting, no walkaway
    ma1 = MasterAgreement(
        id="MA016",
        agreement_type=AgreementType.ISDA_2002,
        party_a_id="CP001",
        party_b_id="CP002",
        effective_date=date(2020, 1, 1),
        close_out_netting=True,
        walkaway_clause=False,
    )
    assert ma1.supports_bilateral_netting()

    # With walkaway clause (non-bilateral)
    ma2 = MasterAgreement(
        id="MA017",
        agreement_type=AgreementType.ISDA_1992,
        party_a_id="CP001",
        party_b_id="CP002",
        effective_date=date(2020, 1, 1),
        close_out_netting=True,
        walkaway_clause=True,
    )
    assert not ma2.supports_bilateral_netting()

    # No close-out netting
    ma3 = MasterAgreement(
        id="MA018",
        agreement_type=AgreementType.ISDA_2002,
        party_a_id="CP001",
        party_b_id="CP002",
        effective_date=date(2020, 1, 1),
        close_out_netting=False,
        walkaway_clause=False,
    )
    assert not ma3.supports_bilateral_netting()


def test_master_agreement_termination_events():
    """Test additional termination events."""
    ma = MasterAgreement(
        id="MA019",
        agreement_type=AgreementType.ISDA_2002,
        party_a_id="CP001",
        party_b_id="CP002",
        effective_date=date(2020, 1, 1),
        additional_termination_events=[
            "Material Adverse Change",
            "Regulatory Event",
        ],
    )
    assert len(ma.additional_termination_events) == 2
    assert "Material Adverse Change" in ma.additional_termination_events


def test_master_agreement_credit_provisions():
    """Test credit event provisions."""
    ma = MasterAgreement(
        id="MA020",
        agreement_type=AgreementType.ISDA_2002,
        party_a_id="CP001",
        party_b_id="CP002",
        effective_date=date(2020, 1, 1),
        credit_event_upon_merger=True,
        cross_default_applicable=True,
    )
    assert ma.credit_event_upon_merger
    assert ma.cross_default_applicable


def test_master_agreement_automatic_early_termination():
    """Test automatic early termination."""
    ma_aet = MasterAgreement(
        id="MA021",
        agreement_type=AgreementType.ISDA_2002,
        party_a_id="CP001",
        party_b_id="CP002",
        effective_date=date(2020, 1, 1),
        automatic_early_termination=True,
    )
    assert ma_aet.automatic_early_termination


def test_master_agreement_termination_currency():
    """Test termination currency options."""
    ma_eur = MasterAgreement(
        id="MA022",
        agreement_type=AgreementType.ISDA_2002,
        party_a_id="CP001",
        party_b_id="CP002",
        effective_date=date(2020, 1, 1),
        termination_currency=TerminationCurrency.EUR,
    )
    assert ma_eur.termination_currency == TerminationCurrency.EUR


def test_master_agreement_repr():
    """Test string representation."""
    ma = MasterAgreement(
        id="MA023",
        agreement_type=AgreementType.ISDA_2002,
        party_a_id="CP001",
        party_b_id="CP002",
        effective_date=date(2020, 1, 1),
        csa_id="CSA001",
    )
    repr_str = repr(ma)
    assert "MA023" in repr_str
    assert "ISDA2002" in repr_str
    assert "CP001" in repr_str
    assert "CP002" in repr_str
    assert "CSA=CSA001" in repr_str


def test_agreement_type_enum():
    """Test AgreementType enum values."""
    assert AgreementType.ISDA_1992.value == "ISDA1992"
    assert AgreementType.ISDA_2002.value == "ISDA2002"
    assert AgreementType.ISDA_2021.value == "ISDA2021"
    assert AgreementType.GMRA.value == "GMRA"
    assert AgreementType.GMSLA.value == "GMSLA"


def test_governing_law_enum():
    """Test GoverningLaw enum values."""
    assert GoverningLaw.ENGLISH.value == "English"
    assert GoverningLaw.NEW_YORK.value == "NewYork"
    assert GoverningLaw.JAPANESE.value == "Japanese"


def test_termination_currency_enum():
    """Test TerminationCurrency enum values."""
    assert TerminationCurrency.USD.value == "USD"
    assert TerminationCurrency.EUR.value == "EUR"
    assert TerminationCurrency.BASE.value == "Base"
