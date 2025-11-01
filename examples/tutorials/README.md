# Scenario Tutorials

These tutorials build on the lightweight scripts in the parent directory by
adding written guidance, scenario motivations, and runnable code. Each folder
contains:

1. **Commentary (`README.md`)** – background, setup instructions, and guidance
   for interpreting results.
2. **Runnable script (`run.py`)** – the executable companion that reproduces the
   narrative steps.

## Available scenarios

1. [Vanilla option pricing diagnostic](01_vanilla_pricing/README.md)
2. [Volatility scenario analysis for Asian structures](02_asian_scenario/README.md)
3. [Counterparty CVA estimation](03_counterparty_cva/README.md)

Run any tutorial with:

```bash
python examples/tutorials/<folder>/run.py
```

Each script prints intermediate commentary so you can follow along with the
explanations in the matching README.
