# Tutorial 1 – Vanilla option pricing diagnostic

This walkthrough revisits the canonical European call option and shows how to
validate Monte Carlo estimates against the Black–Scholes closed form. It is a
useful smoke test whenever model assumptions or simulation parameters change.

## Learning goals

- Build an `MCConfig` and generate geometric Brownian motion paths with JAX.
- Discount the terminal payoffs and compare them against the analytical
  Black–Scholes solution.
- Inspect the Monte Carlo sampling error and understand how the number of paths
  impacts convergence.

## Running the script

```bash
python examples/tutorials/01_vanilla_pricing/run.py
```

The script prints commentary after each step so you can trace the inputs,
intermediate statistics, and the final price comparison.

## Key takeaways

- Monte Carlo and analytic prices should agree within sampling error.
- Increasing the number of paths reduces the standard error of the Monte Carlo
  estimate.
- This diagnostic provides quick feedback that the RNG seed, drift, and discount
  conventions are wired correctly.
