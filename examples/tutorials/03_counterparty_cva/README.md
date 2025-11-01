# Tutorial 3 – Counterparty CVA estimation

This scenario demonstrates a lightweight counterparty credit valuation
adjustment (CVA) workflow. We simulate exposure profiles for an at-the-money
European call, map a flat hazard rate to default probabilities, and then
aggregate the discounted CVA charge.

## Learning goals

- Generate an exposure time series by sampling Monte Carlo paths and projecting
  intrinsic option values through time.
- Convert a constant hazard rate to a cumulative default probability term
  structure.
- Combine exposure, discount factors, and default probabilities with the library
  `cva` utility.

## Running the script

```bash
python examples/tutorials/03_counterparty_cva/run.py
```

The script prints the exposure curve, discount factors, marginal default
probabilities, and the final CVA number for easy inspection.

## Key takeaways

- Even simple exposure approximations are useful for directional CVA insights.
- Hazard rates translate into cumulative default probabilities via
  `hazard_to_pd`.
- Extending the scaffolding to more realistic credit curves or netting sets is a
  natural next step once the workflow is familiar.
