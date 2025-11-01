# Tutorial 2 – Volatility scenario analysis for Asian structures

This tutorial stress-tests an average price guarantee (arithmetic Asian call)
under different volatility regimes. It shows how to repurpose the Monte Carlo
engine for scenario analysis rather than point estimates.

## Learning goals

- Reuse `simulate_gbm` to produce shared random paths for comparable scenarios.
- Evaluate an `AsianArithmetic` payoff and discount the result.
- Quantify the sensitivity of the product to volatility shocks and summarize the
  results in a quick scenario table.

## Running the script

```bash
python examples/tutorials/02_asian_scenario/run.py
```

The script reports one row per volatility assumption, re-using the same base
paths for an apples-to-apples comparison.

## Key takeaways

- Volatility has a first-order impact on the average price guarantee because the
  payoff looks at the entire path distribution.
- Holding the random seed constant isolates the effect of the parameter change
  when comparing scenarios.
- The tabular output is easy to share with stakeholders when discussing stress
  tests or calibration changes.
