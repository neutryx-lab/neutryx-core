"""Tests for barrier option payoffs."""

import jax.numpy as jnp
import pytest

from neutryx.products.barrier import (
    DoubleBarrierCall,
    DoubleBarrierPut,
    DownAndInCall,
    DownAndInPut,
    DownAndOutCall,
    DownAndOutPut,
    UpAndInCall,
    UpAndInPut,
    UpAndOutCall,
    UpAndOutPut,
)


def create_path_up_barrier(S0=100.0, barrier=120.0, final=130.0, n_steps=100):
    """Create a path that hits an upper barrier."""
    # Path goes from S0 to final, passing through barrier
    path = jnp.linspace(S0, final, n_steps)
    # Ensure it actually hits the barrier
    path = path.at[n_steps // 2].set(barrier)
    return path


def create_path_down_barrier(S0=100.0, barrier=80.0, final=70.0, n_steps=100):
    """Create a path that hits a lower barrier."""
    path = jnp.linspace(S0, final, n_steps)
    path = path.at[n_steps // 2].set(barrier)
    return path


def create_path_no_barrier(S0=100.0, final=110.0, n_steps=100):
    """Create a path that doesn't hit barriers."""
    return jnp.linspace(S0, final, n_steps)


def test_up_and_out_call_knocked_out():
    """Test up-and-out call when barrier is hit."""
    option = UpAndOutCall(K=100.0, T=1.0, B=120.0)
    path = create_path_up_barrier(S0=100.0, barrier=120.0, final=130.0)

    payoff = option.payoff_path(path)

    # Should be knocked out (payoff = 0)
    assert payoff == 0.0


def test_up_and_out_call_not_knocked_out():
    """Test up-and-out call when barrier is not hit."""
    option = UpAndOutCall(K=100.0, T=1.0, B=120.0)
    path = create_path_no_barrier(S0=100.0, final=110.0)

    payoff = option.payoff_path(path)

    # Should have intrinsic value (110 - 100 = 10)
    assert jnp.isclose(payoff, 10.0)


def test_up_and_out_put_knocked_out():
    """Test up-and-out put when barrier is hit."""
    option = UpAndOutPut(K=100.0, T=1.0, B=120.0)
    path = create_path_up_barrier(S0=100.0, barrier=120.0, final=130.0)

    payoff = option.payoff_path(path)

    # Should be knocked out
    assert payoff == 0.0


def test_down_and_out_call_knocked_out():
    """Test down-and-out call when barrier is hit."""
    option = DownAndOutCall(K=80.0, T=1.0, B=70.0)
    path = create_path_down_barrier(S0=100.0, barrier=70.0, final=75.0)

    payoff = option.payoff_path(path)

    # Should be knocked out
    assert payoff == 0.0


def test_down_and_out_call_not_knocked_out():
    """Test down-and-out call when barrier is not hit."""
    option = DownAndOutCall(K=80.0, T=1.0, B=70.0)
    path = create_path_no_barrier(S0=100.0, final=90.0)

    payoff = option.payoff_path(path)

    # Should have intrinsic value (90 - 80 = 10)
    assert jnp.isclose(payoff, 10.0)


def test_down_and_out_put_knocked_out():
    """Test down-and-out put when barrier is hit."""
    option = DownAndOutPut(K=100.0, T=1.0, B=70.0)
    path = create_path_down_barrier(S0=100.0, barrier=70.0, final=60.0)

    payoff = option.payoff_path(path)

    # Should be knocked out
    assert payoff == 0.0


def test_up_and_in_call_knocked_in():
    """Test up-and-in call when barrier is hit."""
    option = UpAndInCall(K=100.0, T=1.0, B=120.0)
    path = create_path_up_barrier(S0=100.0, barrier=120.0, final=130.0)

    payoff = option.payoff_path(path)

    # Should be knocked in and have intrinsic value (130 - 100 = 30)
    assert jnp.isclose(payoff, 30.0)


def test_up_and_in_call_not_knocked_in():
    """Test up-and-in call when barrier is not hit."""
    option = UpAndInCall(K=100.0, T=1.0, B=120.0)
    path = create_path_no_barrier(S0=100.0, final=110.0)

    payoff = option.payoff_path(path)

    # Should not be knocked in (payoff = 0)
    assert payoff == 0.0


def test_up_and_in_put_knocked_in():
    """Test up-and-in put when barrier is hit."""
    option = UpAndInPut(K=100.0, T=1.0, B=120.0)
    path = create_path_up_barrier(S0=100.0, barrier=120.0, final=90.0)

    payoff = option.payoff_path(path)

    # Should be knocked in and have intrinsic value (100 - 90 = 10)
    assert jnp.isclose(payoff, 10.0)


def test_down_and_in_call_knocked_in():
    """Test down-and-in call when barrier is hit."""
    option = DownAndInCall(K=80.0, T=1.0, B=70.0)
    path = create_path_down_barrier(S0=100.0, barrier=70.0, final=90.0)

    payoff = option.payoff_path(path)

    # Should be knocked in and have intrinsic value (90 - 80 = 10)
    assert jnp.isclose(payoff, 10.0)


def test_down_and_in_call_not_knocked_in():
    """Test down-and-in call when barrier is not hit."""
    option = DownAndInCall(K=80.0, T=1.0, B=70.0)
    path = create_path_no_barrier(S0=100.0, final=90.0)

    payoff = option.payoff_path(path)

    # Should not be knocked in
    assert payoff == 0.0


def test_down_and_in_put_knocked_in():
    """Test down-and-in put when barrier is hit."""
    option = DownAndInPut(K=100.0, T=1.0, B=70.0)
    path = create_path_down_barrier(S0=100.0, barrier=70.0, final=80.0)

    payoff = option.payoff_path(path)

    # Should be knocked in and have intrinsic value (100 - 80 = 20)
    assert jnp.isclose(payoff, 20.0)


def test_double_barrier_call_knocked_out_upper():
    """Test double barrier call knocked out by upper barrier."""
    option = DoubleBarrierCall(K=100.0, T=1.0, B_lower=80.0, B_upper=120.0)
    path = create_path_up_barrier(S0=100.0, barrier=120.0, final=130.0)

    payoff = option.payoff_path(path)

    # Should be knocked out by upper barrier
    assert payoff == 0.0


def test_double_barrier_call_knocked_out_lower():
    """Test double barrier call knocked out by lower barrier."""
    option = DoubleBarrierCall(K=100.0, T=1.0, B_lower=80.0, B_upper=120.0)
    path = create_path_down_barrier(S0=100.0, barrier=80.0, final=90.0)

    payoff = option.payoff_path(path)

    # Should be knocked out by lower barrier
    assert payoff == 0.0


def test_double_barrier_call_not_knocked_out():
    """Test double barrier call when no barrier is hit."""
    option = DoubleBarrierCall(K=90.0, T=1.0, B_lower=80.0, B_upper=120.0)
    path = create_path_no_barrier(S0=100.0, final=110.0)

    payoff = option.payoff_path(path)

    # Should have intrinsic value (110 - 90 = 20)
    assert jnp.isclose(payoff, 20.0)


def test_double_barrier_put_knocked_out():
    """Test double barrier put knocked out."""
    option = DoubleBarrierPut(K=100.0, T=1.0, B_lower=80.0, B_upper=120.0)
    path = create_path_up_barrier(S0=100.0, barrier=120.0, final=130.0)

    payoff = option.payoff_path(path)

    # Should be knocked out
    assert payoff == 0.0


def test_double_barrier_put_not_knocked_out():
    """Test double barrier put when no barrier is hit."""
    option = DoubleBarrierPut(K=110.0, T=1.0, B_lower=80.0, B_upper=120.0)
    path = create_path_no_barrier(S0=100.0, final=95.0)

    payoff = option.payoff_path(path)

    # Should have intrinsic value (110 - 95 = 15)
    assert jnp.isclose(payoff, 15.0)


def test_in_out_parity():
    """Test that In + Out = Vanilla for same parameters."""
    K = 100.0
    T = 1.0
    B_up = 120.0

    # Create a path
    path = create_path_no_barrier(S0=100.0, final=110.0)

    # Up-and-out + Up-and-in should equal vanilla call
    out_call = UpAndOutCall(K=K, T=T, B=B_up)
    in_call = UpAndInCall(K=K, T=T, B=B_up)

    payoff_out = out_call.payoff_path(path)
    payoff_in = in_call.payoff_path(path)

    # Vanilla call payoff
    vanilla_payoff = jnp.maximum(path[-1] - K, 0.0)

    # In + Out should equal vanilla
    assert jnp.isclose(payoff_out + payoff_in, vanilla_payoff)


def test_barrier_option_otm():
    """Test barrier options that expire out-of-the-money."""
    # Call option: ST < K
    path = create_path_no_barrier(S0=100.0, final=90.0)

    call = UpAndOutCall(K=100.0, T=1.0, B=120.0)
    payoff = call.payoff_path(path)

    # Should be 0 (OTM)
    assert payoff == 0.0

    # Put option: ST > K
    path2 = create_path_no_barrier(S0=100.0, final=110.0)

    put = DownAndOutPut(K=100.0, T=1.0, B=80.0)
    payoff2 = put.payoff_path(path2)

    # Should be 0 (OTM)
    assert payoff2 == 0.0
