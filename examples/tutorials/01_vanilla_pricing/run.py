"""Scenario tutorial: Monte Carlo vs analytic pricing for a vanilla call."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from neutryx.core.engine import MCConfig, present_value, simulate_gbm
from neutryx.models.bs import price as bs_price


def main():
    print("🧭  Starting vanilla option pricing diagnostic...")

    key = jax.random.PRNGKey(123)
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.01, 0.00, 0.2
    cfg = MCConfig(steps=252, paths=120_000)

    print(f"\nConfig -> steps={cfg.steps}, paths={cfg.paths}, seed=123")
    paths = simulate_gbm(key, S, r - q, sigma, T, cfg)
    terminal_spots = paths[:, -1]

    payoffs = jnp.maximum(terminal_spots - K, 0.0)
    mc_price = present_value(payoffs, jnp.array(T), r)
    std_error = payoffs.std() * jnp.exp(-r * T) / jnp.sqrt(cfg.paths)

    bs_closed = bs_price(S, K, T, r, q, sigma, "call")

    print("\nInputs:")
    print(f"  Spot={S}, Strike={K}, Maturity={T}y, Rate={r}, Vol={sigma}")

    print("\nResults:")
    print(f"  Monte Carlo price  : {float(mc_price):.4f}")
    print(f"  Analytical (B&S)   : {float(bs_closed):.4f}")
    print(f"  Std. error (1σ)    : {float(std_error):.4f}")

    diff = jnp.abs(mc_price - bs_closed)
    within_error = diff <= 2 * std_error
    print("\nDiagnostic:")
    print(f"  Absolute difference: {float(diff):.4f}")
    print(
        "  Within 2x Monte Carlo standard error?",
        "✅ yes" if within_error else "⚠️ no",
    )

    print("\n✨  Done. Re-run with different seeds or path counts to see convergence.")


if __name__ == "__main__":
    main()
