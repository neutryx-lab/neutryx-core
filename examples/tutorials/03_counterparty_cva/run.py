"""Scenario tutorial: estimate a simple CVA charge for a vanilla call."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from neutryx.core.engine import MCConfig, simulate_gbm
from neutryx.valuations.cva import cva
from neutryx.valuations.utils import hazard_to_pd


def main():
    print("🧭  Estimating counterparty CVA for an at-the-money call...")

    key = jax.random.PRNGKey(7)
    S, K, T, r, q, sigma = 100.0, 100.0, 2.0, 0.01, 0.0, 0.25
    cfg = MCConfig(steps=24, paths=150_000)

    container = simulate_gbm(
        key,
        S,
        r - q,
        sigma,
        T,
        cfg,
        return_full=True,
    )

    # Drop t=0; exposures start after trade inception.
    times = container.times[1:]
    spot_paths = container.values[:, 1:]
    intrinsic = jnp.maximum(spot_paths - K, 0.0)
    epe_curve = intrinsic.mean(axis=0)

    df_curve = jnp.exp(-r * times)
    hazard_rate = 0.025
    pd_curve = hazard_to_pd(hazard_rate, times)
    marginal_pd = jnp.diff(jnp.concatenate([jnp.array([0.0]), pd_curve]))

    charge = cva(epe_curve, df_curve, pd_curve, lgd=0.6)

    print("\nAssumptions:")
    print(
        f"  Spot={S}, Strike={K}, Tenor={T}y, Rate={r}, Vol={sigma}, "
        f"LGD=60%, Hazard={hazard_rate:.2%}"
    )
    print(f"  Time buckets={cfg.steps} (uniform)")

    header = f"{'t (years)':>10}{'EPE':>12}{'DF':>10}{'Cum PD':>12}{'Marginal PD':>14}"
    print("\n" + header)
    print("-" * len(header))
    for t, e, df, pd, d_pd in zip(times, epe_curve, df_curve, pd_curve, marginal_pd):
        print(f"{float(t):>10.2f}{float(e):>12.4f}{float(df):>10.4f}{float(pd):>12.4f}{float(d_pd):>14.4f}")

    print("\nResults:")
    print(f"  Expected positive exposure (avg): {float(epe_curve.mean()):.4f}")
    print(f"  CVA charge: {float(charge):.4f}")

    print("\n✨  Done. Swap in term structures for more realistic what-if analysis.")


if __name__ == "__main__":
    main()
