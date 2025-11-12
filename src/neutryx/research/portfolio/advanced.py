"""Advanced portfolio optimization algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class RobustMeanVarianceOptimizer:
    """Ellipsoidal uncertainty set robust optimizer."""

    gamma: float = 1.0
    risk_aversion: float = 2.5
    allow_short: bool = False

    def optimize(self, expected_returns: np.ndarray, covariance: np.ndarray) -> np.ndarray:
        expected_returns = np.asarray(expected_returns, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        n_assets = expected_returns.shape[0]
        if covariance.shape != (n_assets, n_assets):
            raise ValueError("Covariance shape must be (n_assets, n_assets).")

        regularizer = self.gamma * np.eye(n_assets)
        adjusted_cov = covariance + regularizer
        weights = np.linalg.solve(self.risk_aversion * adjusted_cov, expected_returns)
        weights /= np.sum(weights)
        if not self.allow_short:
            weights = np.clip(weights, 0.0, None)
            if weights.sum() == 0:
                weights = np.full_like(weights, 1.0 / n_assets)
            else:
                weights /= weights.sum()
        return weights


class MarketSimulationEnvironment:
    """Simple vectorized market environment for reinforcement learning."""

    def __init__(self, returns: np.ndarray):
        if returns.ndim != 2:
            raise ValueError("Returns must have shape (n_periods, n_assets).")
        self.returns = returns
        self.n_periods, self.n_assets = returns.shape
        self._t = 0

    def reset(self) -> np.ndarray:
        self._t = 0
        return self._get_observation()

    def step(self, weights: np.ndarray) -> Tuple[float, np.ndarray, bool, dict]:
        if weights.shape[0] != self.n_assets:
            raise ValueError("Weight vector size does not match number of assets.")
        weights = np.asarray(weights, dtype=float)
        weights = weights / np.sum(weights)
        period_return = float(weights @ self.returns[self._t])
        self._t += 1
        done = self._t >= self.n_periods
        next_obs = self._get_observation() if not done else np.zeros(self.n_assets)
        return period_return, next_obs, done, {}

    def _get_observation(self) -> np.ndarray:
        return self.returns[self._t]


class ReinforcementLearningPortfolioAgent:
    """Policy-gradient agent that selects from a discrete action set of allocations."""

    def __init__(
        self,
        action_space: Sequence[np.ndarray],
        learning_rate: float = 0.1,
        discount: float = 0.95,
        entropy_coefficient: float = 1e-3,
        seed: Optional[int] = None,
    ) -> None:
        if len(action_space) == 0:
            raise ValueError("Action space must contain at least one allocation vector.")
        self.action_space = [np.asarray(action) / np.sum(action) for action in action_space]
        self.learning_rate = learning_rate
        self.discount = discount
        self.entropy_coefficient = entropy_coefficient
        self.rng = np.random.default_rng(seed)
        self._params = np.zeros(len(self.action_space))

    def policy(self) -> np.ndarray:
        logits = self._params - np.max(self._params)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum()

    def act(self) -> np.ndarray:
        probs = self.policy()
        idx = self.rng.choice(len(self.action_space), p=probs)
        return self.action_space[idx]

    def train(self, env: MarketSimulationEnvironment, episodes: int = 100) -> None:
        for _ in range(episodes):
            self._episode(env)

    def _episode(self, env: MarketSimulationEnvironment) -> None:
        env.reset()
        grads: List[np.ndarray] = []
        rewards: List[float] = []
        done = False
        t = 0
        while not done:
            probs = self.policy()
            action_idx = self.rng.choice(len(self.action_space), p=probs)
            action = self.action_space[action_idx]
            reward, _, done, _ = env.step(action)
            grads.append(self._grad_log_policy(probs, action_idx))
            rewards.append(reward)
            t += 1

        returns = self._discounted_returns(rewards)
        for grad, G in zip(grads, returns):
            update = self.learning_rate * (G * grad - self.entropy_coefficient * self._entropy_gradient())
            self._params += update

    def _discounted_returns(self, rewards: Iterable[float]) -> np.ndarray:
        returns = []
        cumulative = 0.0
        for reward in reversed(list(rewards)):
            cumulative = reward + self.discount * cumulative
            returns.append(cumulative)
        returns.reverse()
        return np.asarray(returns)

    def _grad_log_policy(self, probs: np.ndarray, action_idx: int) -> np.ndarray:
        grad = -probs
        grad[action_idx] += 1.0
        return grad

    def _entropy_gradient(self) -> np.ndarray:
        probs = self.policy()
        return -np.log(np.clip(probs, 1e-8, None)) - 1.0

    def action_probabilities(self) -> np.ndarray:
        return self.policy()
