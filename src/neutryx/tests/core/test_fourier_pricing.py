import numpy as np

from neutryx.models.bs import price as bs_price
from neutryx.engines import (
    BlackScholesCharacteristicModel,
    carr_madan_fft,
    cos_method,
)


def _analytic_prices(strikes, spot, rate, dividend, sigma, maturity, kind):
    return np.array([
        float(bs_price(spot, strike, maturity, rate, dividend, sigma, kind))
        for strike in strikes
    ])


def test_carr_madan_matches_black_scholes():
    model = BlackScholesCharacteristicModel(spot=100.0, rate=0.01, dividend=0.0, volatility=0.2)
    maturity = 1.0
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])

    prices_fft = np.array(carr_madan_fft(model, maturity, strikes, alpha=1.5, grid_size=4096, eta=0.15))
    expected = _analytic_prices(strikes, 100.0, 0.01, 0.0, 0.2, maturity, "call")

    np.testing.assert_allclose(prices_fft, expected, rtol=1e-3, atol=1e-3)


def test_cos_method_call_and_put():
    model = BlackScholesCharacteristicModel(spot=100.0, rate=0.03, dividend=0.01, volatility=0.25)
    maturity = 0.75
    strikes = np.linspace(70.0, 130.0, 7)

    prices_call = np.array(cos_method(model, maturity, strikes, expansion_terms=512, truncation=8.0, option="call"))
    expected_call = _analytic_prices(strikes, 100.0, 0.03, 0.01, 0.25, maturity, "call")
    np.testing.assert_allclose(prices_call, expected_call, rtol=2e-4, atol=2e-4)

    prices_put = np.array(cos_method(model, maturity, strikes, expansion_terms=512, truncation=8.0, option="put"))
    expected_put = _analytic_prices(strikes, 100.0, 0.03, 0.01, 0.25, maturity, "put")
    np.testing.assert_allclose(prices_put, expected_put, rtol=2e-4, atol=2e-4)
