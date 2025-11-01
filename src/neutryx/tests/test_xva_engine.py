import jax
import jax.numpy as jnp

from neutryx.core.engine import MCConfig, simulate_gbm
from neutryx.valuations.xva import AggregationEngine, CapitalCalculator, ExposureSimulator, XVAScenario


def _gbm_generator(key, params):
    return simulate_gbm(
        key,
        params["S0"],
        params["mu"],
        params["sigma"],
        params["T"],
        params["cfg"],
    )


def _call_exposure(paths, params):
    strike = params.get("K", params["S0"])
    return paths - strike


def _build_cube(seed=0):
    key = jax.random.PRNGKey(seed)
    cfg = MCConfig(steps=6, paths=2000)
    simulator = ExposureSimulator(_gbm_generator, _call_exposure)
    base_params = {"S0": 100.0, "mu": 0.02, "sigma": 0.2, "T": 1.0, "cfg": cfg, "K": 100.0}
    scenarios = [
        XVAScenario("base", {}),
        XVAScenario("bull", {"mu": 0.05}, weight=2.0),
        XVAScenario("stress", {"sigma": 0.35}, weight=1.5),
    ]
    return simulator.simulate(key, base_params, scenarios)


def test_exposure_cube_weighting():
    cube = _build_cube()
    weights = jnp.array([1.0, 2.0, 1.5])
    weights = weights / weights.sum()
    matrix = cube.expected_positive_matrix()
    manual = (weights[:, None] * matrix).sum(axis=0)
    aggregated = cube.aggregate_expected_positive()
    assert aggregated.shape == manual.shape
    assert jnp.allclose(aggregated, manual, rtol=1e-5, atol=1e-5)


def test_aggregation_engine_outputs():
    cube = _build_cube(seed=1)
    times = cube.times
    discount_curve = jnp.exp(-0.02 * times)
    default_probabilities = jnp.linspace(0.0, 0.1, times.shape[0])
    calculator = CapitalCalculator(
        discount_curve=discount_curve,
        default_probabilities=default_probabilities,
        funding_spread=0.01,
        hurdle_rate=0.12,
    )
    engine = AggregationEngine(calculator)
    summary = engine.summarize(cube, quantile=0.975, alpha=0.98)
    assert summary["epe"].shape == times.shape
    assert summary["ene"].shape == times.shape
    assert summary["pfe"].shape == times.shape
    assert summary["expected_shortfall"].shape == times.shape
    assert summary["cva"] >= 0.0
    assert summary["funding_cost"] >= 0.0
    assert summary["kva"] >= 0.0
