from pathlib import Path
import sys

import numpy as np
import pytest

SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from neutryx.research.portfolio import (
    BlackLittermanModel,
    CovarianceEstimator,
    MaximumSharpeRatioOptimizer,
    MinimumVarianceOptimizer,
    PortfolioView,
    ViewCollection,
)
from neutryx.research.portfolio.advanced import (
    MarketSimulationEnvironment,
    ReinforcementLearningPortfolioAgent,
    RobustMeanVarianceOptimizer,
)


@pytest.fixture
def toy_returns() -> np.ndarray:
    rng = np.random.default_rng(42)
    base = rng.normal(0.0, 0.02, size=(120, 3))
    signal = np.linspace(0.001, 0.003, 120)[:, None] * np.array([[1.0, 0.5, -0.25]])
    return base + signal


def test_covariance_estimator_ledoit_wolf(toy_returns: np.ndarray) -> None:
    estimator = CovarianceEstimator(method="ledoit_wolf", shrinkage_target="diagonal")
    cov = estimator.estimate(toy_returns)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals > 0)


def test_minimum_variance_optimizer(toy_returns: np.ndarray) -> None:
    covariance = CovarianceEstimator().estimate(toy_returns)
    optimizer = MinimumVarianceOptimizer()
    weights = optimizer.optimize(covariance)
    assert pytest.approx(weights.sum(), rel=1e-6, abs=1e-6) == 1.0
    assert np.all(weights >= -1e-6)


def test_maximum_sharpe_ratio_optimizer(toy_returns: np.ndarray) -> None:
    expected_returns = toy_returns.mean(axis=0)
    covariance = CovarianceEstimator().estimate(toy_returns)
    optimizer = MaximumSharpeRatioOptimizer(risk_free_rate=0.0005)
    weights = optimizer.optimize(expected_returns, covariance)
    assert pytest.approx(weights.sum(), rel=1e-6, abs=1e-6) == 1.0
    best_asset = int(np.argmax(expected_returns))
    assert weights[best_asset] == pytest.approx(max(weights), rel=1e-2)


def test_black_litterman_posterior(toy_returns: np.ndarray) -> None:
    asset_names = ["AssetA", "AssetB", "AssetC"]
    market_weights = np.array([0.5, 0.3, 0.2])
    model = BlackLittermanModel(asset_names=asset_names, market_weights=market_weights)
    model.fit(toy_returns)

    views = ViewCollection(asset_names)
    views.add(PortfolioView.relative("AssetA", "AssetC", expected_outperformance=0.015, confidence=0.7))
    posterior = model.posterior(views)

    assert posterior.mean.shape[0] == len(asset_names)
    assert posterior.covariance.shape == (3, 3)
    assert pytest.approx(posterior.weights.sum(), rel=1e-6, abs=1e-6) == 1.0
    assert posterior.weights[0] > posterior.weights[2]


def test_robust_optimizer_is_more_conservative(toy_returns: np.ndarray) -> None:
    expected_returns = toy_returns.mean(axis=0)
    covariance = CovarianceEstimator().estimate(toy_returns)

    classical = MaximumSharpeRatioOptimizer().optimize(expected_returns, covariance)
    robust = RobustMeanVarianceOptimizer(gamma=5.0).optimize(expected_returns, covariance)

    distance_classical = abs(classical[0] - 1.0 / 3.0)
    distance_robust = abs(robust[0] - 1.0 / 3.0)
    assert distance_robust < distance_classical


def test_reinforcement_learning_agent_converges() -> None:
    periods = 30
    returns = np.column_stack(
        [
            np.full(periods, 0.02),
            np.full(periods, -0.01),
            np.full(periods, 0.005),
        ]
    )
    env = MarketSimulationEnvironment(returns)
    actions = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([1 / 3, 1 / 3, 1 / 3])]
    agent = ReinforcementLearningPortfolioAgent(actions, learning_rate=0.2, discount=0.9, seed=123)
    agent.train(env, episodes=60)
    probs = agent.action_probabilities()
    assert probs[0] == pytest.approx(max(probs), rel=1e-2)
    assert probs[0] > 0.7
