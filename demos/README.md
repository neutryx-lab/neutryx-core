# Examples Overview

The `examples/` directory now contains two complementary sets of resources:

- **Focused API demos** (`01_bs_vanilla.py`, `02_path_dependents.py`, `03_american_lsm.py`)
  showcase the core Monte Carlo pricing engines and product payoffs in compact
  scripts.
- **Scenario-driven tutorials** under [`tutorials/`](tutorials/README.md) walk
  through end-to-end workflows that combine pricing, risk measurement, and
  reporting.

| Area | Script | Purpose |
| --- | --- | --- |
| Vanilla option pricing | `01_bs_vanilla.py` | Compare Monte Carlo prices with the Black-Scholes closed form. |
| Path-dependent payoffs | `02_path_dependents.py` | Price an Asian arithmetic call using sampled paths. |
| American exercise | `03_american_lsm.py` | Estimate an American put price with Longstaff-Schwartz Monte Carlo. |
| Interactive dashboards | `dashboard/app.py` | Explore scenario analysis and strike sweeps in a Gradio UI. |
| Scenario tutorial | `tutorials/01_vanilla_pricing/run.py` | Narrative walkthrough of Monte Carlo vs analytic pricing. |
| Scenario tutorial | `tutorials/02_asian_scenario/run.py` | Stress-test an average price guarantee across volatility regimes. |
| Scenario tutorial | `tutorials/03_counterparty_cva/run.py` | Estimate a counterparty CVA profile from simulated exposures. |

Use these scripts as starting points for experiments or as references when
integrating `neutryx` components into larger applications.
