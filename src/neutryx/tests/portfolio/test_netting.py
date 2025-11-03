"""Tests for netting calculations."""
from __future__ import annotations

from datetime import date

import jax.numpy as jnp
import pytest

from neutryx.portfolio import netting


def test_net_exposure_legacy():
    """Test legacy net_exposure function for backward compatibility."""
    payoffs = jnp.array([
        [100.0, 200.0, 150.0],  # Trade 1 across 3 paths
        [-50.0, -80.0, -60.0],  # Trade 2 across 3 paths
        [30.0, 40.0, 35.0],  # Trade 3 across 3 paths
    ])
    net = netting.net_exposure(payoffs)
    expected = jnp.array([80.0, 160.0, 125.0])
    assert jnp.allclose(net, expected)


def test_calculate_close_out_amount_basic():
    """Test basic close-out amount calculation."""
    mtm_values = [100_000.0, -50_000.0, 30_000.0]
    close_out = netting.calculate_close_out_amount(mtm_values)
    assert close_out == 80_000.0


def test_calculate_close_out_amount_with_termination_costs():
    """Test close-out with termination costs."""
    mtm_values = [100_000.0, -50_000.0]
    termination_costs = [5_000.0, 3_000.0]
    close_out = netting.calculate_close_out_amount(
        mtm_values,
        include_termination_costs=True,
        termination_costs=termination_costs,
    )
    # (100K - 50K) + (5K + 3K) = 58K
    assert close_out == 58_000.0


def test_calculate_close_out_amount_mismatched_lengths():
    """Test that mismatched lengths raise error."""
    mtm_values = [100_000.0, -50_000.0]
    termination_costs = [5_000.0]  # Wrong length
    with pytest.raises(ValueError, match="must match length"):
        netting.calculate_close_out_amount(
            mtm_values,
            include_termination_costs=True,
            termination_costs=termination_costs,
        )


def test_calculate_net_exposure_by_currency():
    """Test netting by currency."""
    mtm_by_currency = {
        "USD": 100_000.0,
        "EUR": -50_000.0,
        "GBP": 25_000.0,
    }
    net_exposure = netting.calculate_net_exposure_by_currency(mtm_by_currency)
    assert net_exposure["USD"] == 100_000.0
    assert net_exposure["EUR"] == -50_000.0
    assert net_exposure["GBP"] == 25_000.0


def test_calculate_payment_netting_by_currency():
    """Test payment netting with currency separation."""
    cash_flows = [
        {"date": date(2024, 6, 15), "currency": "USD", "amount": 100_000.0},
        {"date": date(2024, 6, 15), "currency": "USD", "amount": -30_000.0},
        {"date": date(2024, 6, 15), "currency": "EUR", "amount": 50_000.0},
        {"date": date(2024, 9, 15), "currency": "USD", "amount": 75_000.0},
    ]
    netted = netting.calculate_payment_netting(cash_flows, netting_by_currency=True)

    assert len(netted) == 3
    # Find the netted USD payment on 2024-06-15
    usd_june = [cf for cf in netted if cf["date"] == date(2024, 6, 15) and cf["currency"] == "USD"][0]
    assert usd_june["amount"] == 70_000.0  # 100K - 30K


def test_calculate_payment_netting_all_currencies():
    """Test payment netting across all currencies."""
    cash_flows = [
        {"date": date(2024, 6, 15), "currency": "USD", "amount": 100_000.0},
        {"date": date(2024, 6, 15), "currency": "EUR", "amount": 50_000.0},
        {"date": date(2024, 9, 15), "currency": "USD", "amount": -30_000.0},
    ]
    netted = netting.calculate_payment_netting(cash_flows, netting_by_currency=False)

    # Should have 2 dates, with amounts netted across currencies
    assert len(netted) == 2


def test_calculate_payment_netting_zero_amounts():
    """Test that zero netted amounts are filtered out."""
    cash_flows = [
        {"date": date(2024, 6, 15), "currency": "USD", "amount": 100_000.0},
        {"date": date(2024, 6, 15), "currency": "USD", "amount": -100_000.0},
    ]
    netted = netting.calculate_payment_netting(cash_flows)
    # Should be empty (amounts cancel out)
    assert len(netted) == 0


def test_calculate_payment_netting_sorted():
    """Test that netted cash flows are sorted by date."""
    cash_flows = [
        {"date": date(2024, 9, 15), "currency": "USD", "amount": 50_000.0},
        {"date": date(2024, 3, 15), "currency": "USD", "amount": 100_000.0},
        {"date": date(2024, 6, 15), "currency": "USD", "amount": 75_000.0},
    ]
    netted = netting.calculate_payment_netting(cash_flows)

    # Should be sorted by date
    assert netted[0]["date"] == date(2024, 3, 15)
    assert netted[1]["date"] == date(2024, 6, 15)
    assert netted[2]["date"] == date(2024, 9, 15)


def test_calculate_collateral_adjusted_exposure_basic():
    """Test basic collateral adjustment."""
    # Exposure of 1M, no threshold, no collateral
    adj_exposure = netting.calculate_collateral_adjusted_exposure(
        exposure=1_000_000.0,
        threshold=0.0,
        posted_collateral=0.0,
    )
    assert adj_exposure == 1_000_000.0


def test_calculate_collateral_adjusted_exposure_with_threshold():
    """Test exposure with threshold."""
    # Exposure 1M, threshold 500K, no collateral
    adj_exposure = netting.calculate_collateral_adjusted_exposure(
        exposure=1_000_000.0,
        threshold=500_000.0,
        posted_collateral=0.0,
    )
    # max(0, 1M - 500K) = 500K
    assert adj_exposure == 500_000.0


def test_calculate_collateral_adjusted_exposure_with_collateral():
    """Test exposure with posted collateral."""
    # Exposure 1M, threshold 200K, collateral 600K
    adj_exposure = netting.calculate_collateral_adjusted_exposure(
        exposure=1_000_000.0,
        threshold=200_000.0,
        posted_collateral=600_000.0,
    )
    # max(0, 1M - 200K - 600K) = max(0, 200K) = 200K
    assert adj_exposure == 200_000.0


def test_calculate_collateral_adjusted_exposure_with_independent_amount():
    """Test exposure with independent amount."""
    # Exposure 1M, threshold 500K, collateral 400K, IA 100K
    adj_exposure = netting.calculate_collateral_adjusted_exposure(
        exposure=1_000_000.0,
        threshold=500_000.0,
        posted_collateral=400_000.0,
        independent_amount=100_000.0,
    )
    # max(0, 1M - 500K - 400K) + 100K = 100K + 100K = 200K
    assert adj_exposure == 200_000.0


def test_calculate_collateral_adjusted_exposure_below_threshold():
    """Test exposure below threshold."""
    # Exposure 300K, threshold 500K, no collateral, IA 50K
    adj_exposure = netting.calculate_collateral_adjusted_exposure(
        exposure=300_000.0,
        threshold=500_000.0,
        posted_collateral=0.0,
        independent_amount=50_000.0,
    )
    # max(0, 300K - 500K) + 50K = 0 + 50K = 50K
    assert adj_exposure == 50_000.0


def test_apply_bilateral_netting_to_paths():
    """Test JAX bilateral netting across paths."""
    trade_values = jnp.array([
        [100.0, 200.0, -50.0, 150.0],  # Trade 1
        [-30.0, -80.0, 20.0, -40.0],  # Trade 2
        [20.0, 30.0, 10.0, 25.0],  # Trade 3
    ])
    net_values = netting.apply_bilateral_netting_to_paths(trade_values)
    expected = jnp.array([90.0, 150.0, -20.0, 135.0])
    assert jnp.allclose(net_values, expected)


def test_calculate_epe_with_netting():
    """Test EPE calculation with netting."""
    trade_values = jnp.array([
        [100.0, 200.0, -50.0, 150.0],
        [-30.0, -80.0, 20.0, -40.0],
        [20.0, 30.0, 10.0, 25.0],
    ])
    epe = netting.calculate_epe_with_netting(trade_values)
    # Net values: [90, 150, -20, 135]
    # Positive: [90, 150, 0, 135]
    # EPE = mean = (90 + 150 + 0 + 135) / 4 = 93.75
    assert abs(epe - 93.75) < 0.01


def test_calculate_epe_with_netting_and_discounting():
    """Test EPE with discount factors."""
    trade_values = jnp.array([
        [100.0, 200.0],
        [-30.0, -50.0],
    ])
    discount_factors = jnp.array([0.95, 0.90])
    epe = netting.calculate_epe_with_netting(trade_values, discount_factors)
    # Net values: [70, 150]
    # Discounted: [70 * 0.95, 150 * 0.90] = [66.5, 135]
    # EPE = (66.5 + 135) / 2 = 100.75
    assert abs(epe - 100.75) < 0.01


def test_calculate_netting_factor():
    """Test netting factor calculation."""
    gross_epe = 100_000.0
    net_epe = 30_000.0
    factor = netting.calculate_netting_factor(gross_epe, net_epe)
    assert factor == 0.3  # 70% netting benefit


def test_calculate_netting_factor_zero_gross():
    """Test netting factor with zero gross EPE."""
    factor = netting.calculate_netting_factor(0.0, 0.0)
    assert factor == 0.0


def test_calculate_netting_factor_no_benefit():
    """Test netting factor with no netting benefit."""
    gross_epe = 100_000.0
    net_epe = 100_000.0
    factor = netting.calculate_netting_factor(gross_epe, net_epe)
    assert factor == 1.0  # No netting benefit
