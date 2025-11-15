import pytest

from neutryx.models.bs import price


@pytest.mark.fast
@pytest.mark.unit
def test_bs_call_atm():
    p = float(price(100.0, 100.0, 1.0, 0.01, 0.00, 0.2, "call"))
    assert 7.9 < p < 8.6
