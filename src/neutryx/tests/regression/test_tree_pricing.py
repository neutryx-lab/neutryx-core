import numpy as np
import pytest

from neutryx.models import bs
from neutryx.pricing.tree import BinomialModel, price_binomial


def _vanilla_payoff(K: float, *, is_call: bool):
    def payoff(spot: np.ndarray) -> np.ndarray:
        if is_call:
            return np.maximum(spot - K, 0.0)
        return np.maximum(K - spot, 0.0)

    return payoff


def test_binomial_matches_bs_call():
    model = BinomialModel(S0=100.0, r=0.01, q=0.0, sigma=0.2, T=1.0, steps=256)
    payoff = _vanilla_payoff(100.0, is_call=True)
    price_tree = price_binomial(model, payoff, exercise="european")
    price_bs = float(bs.price(100.0, 100.0, 1.0, 0.01, 0.0, 0.2, "call"))
    assert abs(price_tree - price_bs) < 0.05


def test_american_call_matches_european_without_dividend():
    model = BinomialModel(S0=100.0, r=0.01, q=0.0, sigma=0.2, T=1.0, steps=200)
    payoff = _vanilla_payoff(100.0, is_call=True)
    price_eur = price_binomial(model, payoff, exercise="european")
    price_am = price_binomial(model, payoff, exercise="american")
    assert price_am == pytest.approx(price_eur, rel=1e-6)


def test_american_call_benefits_from_dividend_yield():
    model = BinomialModel(S0=100.0, r=0.02, q=0.12, sigma=0.25, T=0.75, steps=200)
    payoff = _vanilla_payoff(90.0, is_call=True)
    price_eur = price_binomial(model, payoff, exercise="european")
    price_am = price_binomial(model, payoff, exercise="american")
    assert price_am > price_eur
    assert price_am - price_eur > 1.0


def test_american_put_regression_value():
    model = BinomialModel(S0=42.0, r=0.05, q=0.0, sigma=0.25, T=1.25, steps=200)
    payoff = _vanilla_payoff(40.0, is_call=False)
    price = price_binomial(model, payoff, exercise="american")
    assert price == pytest.approx(2.7485556071437034, rel=1e-6)
