from fastapi.testclient import TestClient

from neutryx.api.rest import create_app

client = TestClient(create_app())


def test_rest_price_vanilla():
    response = client.post(
        "/price/vanilla",
        json={
            "spot": 100.0,
            "strike": 100.0,
            "maturity": 1.0,
            "rate": 0.01,
            "dividend": 0.0,
            "volatility": 0.2,
            "steps": 4,
            "paths": 128,
            "seed": 0,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["price"] > 0.0


def test_rest_cva():
    response = client.post(
        "/xva/cva",
        json={
            "epe": {"values": [1.0, 0.8, 0.6]},
            "discount": {"values": [0.99, 0.97, 0.95]},
            "default_probability": {"values": [0.0, 0.02, 0.05]},
            "lgd": 0.6,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["cva"] >= 0.0
