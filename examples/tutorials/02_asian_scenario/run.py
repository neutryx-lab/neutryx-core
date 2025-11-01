"""Scenario tutorial: volatility sweep for an Asian arithmetic call."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from neutryx.core.engine import MCConfig, present_value, simulate_gbm
from neutryx.products.asian import AsianArithmetic


SCENARIOS = [
    ("Calm markets", 0.15),
    ("Base case", 0.20),
    ("Stressed vol", 0.30),
]


def main():
    print("🧭  Running Asian payoff volatility scenarios...")

    base_key = jax.random.PRNGKey(2024)
    S, K, T, r, q = 95.0, 100.0, 1.0, 0.02, 0.0
    cfg = MCConfig(steps=64, paths=120_000)
    product = AsianArithmetic(K=K, T=T, is_call=True)

    print(
        f"Shared setup -> steps={cfg.steps}, paths={cfg.paths}, maturity={T}y, "
        f"seed=2024"
    )
    print("(Each scenario reuses the same Gaussian draws for comparability.)")

    header = f"{'Scenario':<15}{'Volatility':>12}{'Price':>12}{'StdErr':>12}"
    print("\n" + header)
    print("-" * len(header))

    for label, sigma in SCENARIOS:
        paths = simulate_gbm(base_key, S, r - q, sigma, T, cfg)
        payoffs = jax.vmap(product.payoff)(paths)
        pv = present_value(payoffs, jnp.array(T), r)
        stderr = payoffs.std() * jnp.exp(-r * T) / jnp.sqrt(cfg.paths)
        print(f"{label:<15}{sigma:>12.2%}{float(pv):>12.4f}{float(stderr):>12.4f}")

    print("\n✨  Done. Swap in alternative strikes or forward-start dates as needed.")


if __name__ == "__main__":
    main()
