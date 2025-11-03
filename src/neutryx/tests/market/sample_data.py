"""Sample market data used for validation tests."""

import jax.numpy as jnp

DEPOSIT_FIXTURES = [
    {"maturity": 0.5, "rate": 0.05},
]

SWAP_FIXTURES = [
    {
        "fixed_rate": 0.055,
        "payment_times": [0.5, 1.0],
        "accrual_factors": [0.5, 0.5],
    }
]

# FRA fixtures for testing
FRA_FIXTURES = [
    {"start": 1.0, "end": 1.5, "rate": 0.058},
]

# Future fixtures for testing
FUTURE_FIXTURES = [
    {"start": 1.5, "end": 2.0, "price": 94.0, "convexity_adjustment": 0.0001},
]

EXPECTED_DISCOUNT_FACTORS = {
    0.5: 0.9756097560975611,
    1.0: 0.9471247997151503,
    # DF(1.5) = DF(1.0) / (1 + 0.058 * 0.5) = 0.9471247997151503 / 1.029
    1.5: 0.9204322640574834,
    # DF(2.0) = DF(1.5) / (1 + (0.06 - 0.0001) * 0.5) = 0.9204322640574834 / 1.02995
    2.0: 0.8936669392276163,
}

EXPECTED_ZERO_RATES = {
    1.0: 0.0543244294822216,
    # r(1.5) = -ln(DF(1.5))/1.5
    1.5: 0.0552745759487152,
    # r(2.0) = -ln(DF(2.0))/2.0
    2.0: 0.0562110692262650,
}

SABR_SURFACE_DATA = {
    "expiries": jnp.array([1.0, 2.0]),
    "forwards": jnp.array([0.025, 0.027]),
    "alphas": jnp.array([0.04, 0.045]),
    "betas": jnp.array([0.5, 0.5]),
    "rhos": jnp.array([-0.25, -0.2]),
    "nus": jnp.array([0.6, 0.55]),
}

SABR_VALIDATION_POINT = {
    "expiry": 1.5,
    "strike": 0.03,
    "expected_vol": 0.25760872045312216,
}

IMPLIED_VOL_SURFACE = {
    "expiries": jnp.array([0.5, 1.0, 2.0]),
    "strikes": jnp.array([80.0, 100.0, 120.0]),
    "vols": jnp.array(
        [
            [0.22, 0.2, 0.21],
            [0.24, 0.22, 0.215],
            [0.26, 0.24, 0.225],
        ]
    ),
    "validation": {
        "expiry": 1.25,
        "strike": 110.0,
        "expected": 0.22125,
    },
}
