"""Sample portfolios used for regression testing."""

SAMPLE_PORTFOLIO = {
    "IRS-1": {
        "IR_USD_1Y": {"delta": 95_000.0},
        "IR_USD_5Y": {"delta": -35_000.0},
    },
    "Cap-1": {
        "IR_USD_1Y": {"delta": 12_500.0, "vega": 750.0},
        "IR_USD_5Y": {"delta": 5_500.0},
    },
    "FXSwap-1": {
        "FX_EURUSD": {"delta": -42_000.0},
    },
    "EqOption-1": {
        "EQ_SPX": {"delta": 7_250.0, "vega": 620.0},
    },
}
