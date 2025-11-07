"""Interactive pricing dashboard powered by Gradio.

This dashboard exposes the Black-Scholes pricing routines implemented in
:mod:`neutryx.engines.fourier` and wraps them in an easy-to-use web UI.  Users
can explore how option prices respond to changes in spot, volatility and other
parameters and stress-test scenarios with a single click.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import gradio as gr
import jax.numpy as jnp
import pandas as pd

from neutryx.engines.fourier import (
    BlackScholesCharacteristicModel,
    carr_madan_fft,
    cos_method,
)


Number = float | int


@dataclass
class PricingConfig:
    """Container for method specific settings."""

    method: str
    option_type: str
    maturity: float
    rate: float
    dividend: float
    strike: float

    def price(self, spot: Number, volatility: Number, *, strike: Number | None = None) -> float:
        """Evaluate the option price for the provided parameters."""

        model = BlackScholesCharacteristicModel(
            spot=float(spot),
            rate=self.rate,
            volatility=float(volatility),
            dividend=self.dividend,
        )

        target_strike = float(strike if strike is not None else self.strike)
        strikes = jnp.asarray([target_strike], dtype=jnp.float32)

        if self.method == "Carr-Madan FFT":
            prices = carr_madan_fft(
                model,
                self.maturity,
                strikes,
                alpha=1.5,
                grid_size=2048,
                eta=0.25,
            )
        else:
            prices = cos_method(
                model,
                self.maturity,
                strikes,
                expansion_terms=256,
                truncation=10.0,
                option=self.option_type.lower(),
            )

        return float(prices[0])


def _scenario_rows(
    base_spot: float,
    base_vol: float,
    config: PricingConfig,
    *,
    spot_shift_pct: float,
    vol_shift_pct: float,
) -> list[dict[str, float]]:
    """Generate a set of pricing scenarios."""

    def shifted(value: float, pct: float, direction: float) -> float:
        return max(value * (1.0 + direction * pct / 100.0), 1e-8)

    scenarios: Iterable[tuple[str, float, float]] = [
        ("Base", base_spot, base_vol),
        ("Spot Down", shifted(base_spot, spot_shift_pct, -1.0), base_vol),
        ("Spot Up", shifted(base_spot, spot_shift_pct, 1.0), base_vol),
        ("Vol Down", base_spot, shifted(base_vol, vol_shift_pct, -1.0)),
        ("Vol Up", base_spot, shifted(base_vol, vol_shift_pct, 1.0)),
    ]

    rows: list[dict[str, float]] = []
    for label, scenario_spot, scenario_vol in scenarios:
        price = config.price(scenario_spot, scenario_vol)
        rows.append(
            {
                "Scenario": label,
                "Spot": float(scenario_spot),
                "Volatility": float(scenario_vol),
                "Price": price,
            }
        )

    return rows


def evaluate_dashboard(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
    volatility: float,
    option_type: str,
    method: str,
    spot_shift_pct: float,
    vol_shift_pct: float,
    strike_min: float,
    strike_max: float,
    strike_steps: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute scenario table and pricing curve for the dashboard."""

    config = PricingConfig(
        method=method,
        option_type=option_type,
        maturity=maturity,
        rate=rate,
        dividend=dividend,
        strike=strike,
    )

    scenario_rows = _scenario_rows(
        base_spot=spot,
        base_vol=volatility,
        config=config,
        spot_shift_pct=spot_shift_pct,
        vol_shift_pct=vol_shift_pct,
    )

    strike_grid = jnp.linspace(strike_min, strike_max, int(strike_steps))
    curve_prices = [config.price(spot, volatility, strike=strike_value) for strike_value in strike_grid]

    scenario_df = pd.DataFrame(scenario_rows)
    curve_df = pd.DataFrame({"Strike": strike_grid, "Price": curve_prices})

    return scenario_df, curve_df


def build_dashboard() -> gr.Blocks:
    """Create the Gradio Blocks layout."""

    with gr.Blocks(title="Neutryx Pricing Dashboard") as demo:
        gr.Markdown(
            """
            # Neutryx Pricing Dashboard

            Explore Black-Scholes pricing with FFT and COS methods. Adjust the
            market inputs on the left and review pre-built scenarios alongside a
            strike sweep of prices.
            """
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                spot = gr.Number(label="Spot", value=100.0)
                strike = gr.Number(label="Strike", value=100.0)
                maturity = gr.Slider(0.05, 5.0, value=1.0, step=0.05, label="Maturity (years)")
                rate = gr.Slider(-0.05, 0.20, value=0.01, step=0.0025, label="Risk-free rate")
                dividend = gr.Slider(0.0, 0.10, value=0.0, step=0.0025, label="Dividend yield")
                volatility = gr.Slider(0.05, 1.0, value=0.20, step=0.01, label="Volatility")
                option_type = gr.Radio(["Call", "Put"], value="Call", label="Option type")
                method = gr.Radio(["Carr-Madan FFT", "COS"], value="COS", label="Pricing method")
                spot_shift_pct = gr.Slider(0.0, 25.0, value=5.0, step=0.5, label="Spot shift (%)")
                vol_shift_pct = gr.Slider(0.0, 50.0, value=10.0, step=1.0, label="Volatility shift (%)")
                strike_min = gr.Number(label="Strike sweep min", value=60.0)
                strike_max = gr.Number(label="Strike sweep max", value=140.0)
                strike_steps = gr.Slider(10, 80, value=30, step=1, label="Strike sweep points")
                run_button = gr.Button("Run analysis", variant="primary")

            with gr.Column(scale=2):
                scenario_table = gr.Dataframe(
                    headers=["Scenario", "Spot", "Volatility", "Price"],
                    datatype=["str", "number", "number", "number"],
                    interactive=False,
                    wrap=True,
                    label="Scenario comparison",
                )
                curve_plot = gr.LinePlot(
                    x="Strike",
                    y="Price",
                    label="Strike sweep",
                    x_title="Strike",
                    y_title="Option price",
                )

        run_button.click(
            fn=evaluate_dashboard,
            inputs=[
                spot,
                strike,
                maturity,
                rate,
                dividend,
                volatility,
                option_type,
                method,
                spot_shift_pct,
                vol_shift_pct,
                strike_min,
                strike_max,
                strike_steps,
            ],
            outputs=[scenario_table, curve_plot],
        )

        # Trigger an initial evaluation when the app loads.
        demo.load(
            fn=evaluate_dashboard,
            inputs=[
                spot,
                strike,
                maturity,
                rate,
                dividend,
                volatility,
                option_type,
                method,
                spot_shift_pct,
                vol_shift_pct,
                strike_min,
                strike_max,
                strike_steps,
            ],
            outputs=[scenario_table, curve_plot],
        )

    return demo


def main() -> None:
    """Launch the dashboard when executed as a script."""

    demo = build_dashboard()
    demo.queue().launch()


if __name__ == "__main__":
    main()
