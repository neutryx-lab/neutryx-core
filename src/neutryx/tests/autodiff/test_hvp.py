import jax
import jax.numpy as jnp

from neutryx.core.autodiff import value_grad_hvp
from neutryx.models.bs import price, second_order_greeks

jax.config.update("jax_enable_x64", True)


def _finite_difference_gradient(func, x, eps=1e-5):
    x = jnp.asarray(x, dtype=jnp.float64)
    grad = []
    for i in range(x.shape[0]):
        basis = jnp.zeros_like(x)
        basis = basis.at[i].set(1.0)
        grad_i = func(x + eps * basis) - func(x - eps * basis)
        grad_i /= 2.0 * eps
        grad.append(grad_i)
    return jnp.stack(grad)


def _finite_difference_hvp(func, x, v, eps=1e-4):
    grad_plus = _finite_difference_gradient(func, x + eps * v)
    grad_minus = _finite_difference_gradient(func, x - eps * v)
    return (grad_plus - grad_minus) / (2.0 * eps)


def test_hessian_vector_product_matches_finite_difference():
    K, T, r, q = 100.0, 1.0, 0.02, 0.0

    def price_from_params(params):
        S, sigma = params
        return price(S, K, T, r, q, sigma)

    params = jnp.array([100.0, 0.2])
    vector = jnp.array([0.5, -0.3])

    value, grad, hvp = value_grad_hvp(price_from_params)(params)
    hvp_result = hvp(vector)
    fd_result = _finite_difference_hvp(price_from_params, params, vector)

    assert jnp.allclose(value, price_from_params(params))
    assert jnp.allclose(hvp_result, fd_result, rtol=5e-3, atol=5e-5)


def test_black_scholes_second_order_greeks_against_finite_difference():
    S, K, T, r, q, sigma = 100.0, 105.0, 2.0, 0.01, 0.0, 0.25
    eps = 1e-4

    greeks = second_order_greeks(S, K, T, r, q, sigma)

    def price_spot(spot):
        return price(spot, K, T, r, q, sigma)

    gamma_fd = (price_spot(S + eps) - 2.0 * price_spot(S) + price_spot(S - eps)) / (eps**2)

    def price_sigma(vol):
        return price(S, K, T, r, q, vol)

    vomma_fd = (price_sigma(sigma + eps) - 2.0 * price_sigma(sigma) + price_sigma(sigma - eps)) / (eps**2)

    def price_mixed(spot, vol):
        return price(spot, K, T, r, q, vol)

    vanna_fd = (
        price_mixed(S + eps, sigma + eps)
        - price_mixed(S + eps, sigma - eps)
        - price_mixed(S - eps, sigma + eps)
        + price_mixed(S - eps, sigma - eps)
    ) / (4.0 * eps * eps)

    assert jnp.isclose(greeks["gamma"], gamma_fd, rtol=5e-3, atol=1e-4)
    assert jnp.isclose(greeks["vomma"], vomma_fd, rtol=5e-3, atol=1e-4)
    assert jnp.isclose(greeks["vanna"], vanna_fd, rtol=5e-3, atol=1e-4)
