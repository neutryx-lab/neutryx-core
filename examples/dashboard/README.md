# Neutryx Pricing Dashboard

This example demonstrates how to wrap the Fourier-based Black-Scholes pricing
utilities provided by `neutryx` into an interactive dashboard. The app is built
with [Gradio](https://www.gradio.app/) and lets you explore how option prices
react to different market scenarios.

## Features

- Choose between the Carr–Madan FFT and COS pricing routines.
- Stress-test preconfigured spot and volatility scenarios.
- Sweep option strikes to visualise the price surface.

## Prerequisites

Install the project dependencies together with the dashboard extra:

```bash
pip install -r requirements.txt
pip install gradio pandas
```

If you already have the repository's development dependencies installed you only
need to add `gradio` because `pandas` is already part of the base requirements.

## Run locally

```bash
python examples/dashboard/app.py
```

Gradio will print a `http://127.0.0.1:7860` link in the console. Open it in a
browser to interact with the dashboard.

## Screenshot

The screenshot below shows the dashboard with the default configuration.

![Dashboard screenshot](./assets/dashboard-default.svg)
