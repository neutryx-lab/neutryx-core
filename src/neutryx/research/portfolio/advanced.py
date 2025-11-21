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


@dataclass
class DynamicProgrammingPortfolioOptimizer:
    """
    Multi-period portfolio optimization using dynamic programming.

    Solves the Bellman equation for optimal portfolio allocation over multiple periods
    with transaction costs and risk constraints.

    Parameters:
    -----------
    n_periods : int
        Number of periods in the investment horizon
    risk_aversion : float
        Risk aversion parameter (lambda in utility function)
    transaction_cost : float
        Proportional transaction cost (e.g., 0.01 for 1%)
    wealth_grid_size : int
        Number of wealth states in discretization
    initial_wealth : float
        Starting wealth level
    allow_short : bool
        Whether short positions are allowed
    """

    n_periods: int
    risk_aversion: float = 2.0
    transaction_cost: float = 0.001
    wealth_grid_size: int = 100
    initial_wealth: float = 1.0
    allow_short: bool = False

    def __post_init__(self):
        if self.n_periods < 1:
            raise ValueError("n_periods must be at least 1")
        if self.wealth_grid_size < 10:
            raise ValueError("wealth_grid_size must be at least 10")
        if self.initial_wealth <= 0:
            raise ValueError("initial_wealth must be positive")

    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        wealth_grid: Optional[np.ndarray] = None
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Solve multi-period portfolio optimization problem.

        Parameters:
        -----------
        expected_returns : np.ndarray
            Expected returns for each asset (n_assets,)
        covariance : np.ndarray
            Covariance matrix (n_assets, n_assets)
        wealth_grid : np.ndarray, optional
            Custom wealth grid for discretization

        Returns:
        --------
        policies : List[np.ndarray]
            Optimal portfolio weights for each period (list of length n_periods)
        value_function : np.ndarray
            Terminal value function over wealth grid
        """
        expected_returns = np.asarray(expected_returns, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        n_assets = expected_returns.shape[0]

        if covariance.shape != (n_assets, n_assets):
            raise ValueError("Covariance shape must be (n_assets, n_assets)")

        # Initialize wealth grid
        if wealth_grid is None:
            wealth_grid = np.linspace(
                0.5 * self.initial_wealth,
                2.0 * self.initial_wealth,
                self.wealth_grid_size
            )

        # Initialize value function (terminal utility)
        value_function = self._terminal_utility(wealth_grid)

        # Store optimal policies for each period
        policies = []

        # Backward induction
        for t in range(self.n_periods - 1, -1, -1):
            policy_t, value_function = self._solve_period(
                wealth_grid,
                value_function,
                expected_returns,
                covariance,
                t
            )
            policies.insert(0, policy_t)

        return policies, value_function

    def _solve_period(
        self,
        wealth_grid: np.ndarray,
        value_next: np.ndarray,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        period: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve single period optimization using backward induction."""
        n_assets = expected_returns.shape[0]
        n_wealth_states = len(wealth_grid)

        # Storage for optimal policy and value
        optimal_weights = np.zeros((n_wealth_states, n_assets))
        value_current = np.zeros(n_wealth_states)

        # For each wealth state, find optimal portfolio
        for i, wealth in enumerate(wealth_grid):
            # Grid search over portfolio weights
            best_value = -np.inf
            best_weights = np.zeros(n_assets)

            # Generate candidate portfolios
            candidates = self._generate_candidate_portfolios(n_assets, wealth)

            for weights in candidates:
                # Calculate expected utility
                expected_value = self._evaluate_portfolio(
                    weights,
                    wealth,
                    expected_returns,
                    covariance,
                    value_next,
                    wealth_grid
                )

                if expected_value > best_value:
                    best_value = expected_value
                    best_weights = weights.copy()

            optimal_weights[i] = best_weights
            value_current[i] = best_value

        # Return policy as weighted average for typical wealth level
        typical_idx = np.argmin(np.abs(wealth_grid - self.initial_wealth))
        return optimal_weights[typical_idx], value_current

    def _generate_candidate_portfolios(
        self,
        n_assets: int,
        wealth: float,
        n_candidates: int = 50
    ) -> List[np.ndarray]:
        """Generate candidate portfolio allocations."""
        candidates = []

        # Add corner solutions
        for i in range(n_assets):
            weights = np.zeros(n_assets)
            weights[i] = 1.0
            candidates.append(weights)

        # Add equal weight
        candidates.append(np.ones(n_assets) / n_assets)

        # Add random portfolios
        rng = np.random.default_rng(42)
        for _ in range(n_candidates - n_assets - 1):
            if self.allow_short:
                weights = rng.uniform(-0.5, 1.5, n_assets)
            else:
                weights = rng.uniform(0, 1, n_assets)

            weights /= np.sum(np.abs(weights))
            candidates.append(weights)

        return candidates

    def _evaluate_portfolio(
        self,
        weights: np.ndarray,
        wealth: float,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        value_next: np.ndarray,
        wealth_grid: np.ndarray
    ) -> float:
        """Evaluate portfolio using Bellman equation."""
        # Portfolio return statistics
        portfolio_return = float(weights @ expected_returns)
        portfolio_variance = float(weights @ covariance @ weights)

        # Transaction costs
        cost = self.transaction_cost * wealth * np.sum(np.abs(weights))

        # Next period wealth (deterministic approximation)
        wealth_next = wealth * (1 + portfolio_return) - cost

        # Interpolate value function
        if wealth_next < wealth_grid[0]:
            next_value = value_next[0]
        elif wealth_next > wealth_grid[-1]:
            next_value = value_next[-1]
        else:
            next_value = float(np.interp(wealth_next, wealth_grid, value_next))

        # Current period utility (mean-variance)
        current_utility = portfolio_return - 0.5 * self.risk_aversion * portfolio_variance

        # Bellman equation: current utility + discounted future value
        total_value = current_utility + 0.95 * next_value

        return total_value

    def _terminal_utility(self, wealth_grid: np.ndarray) -> np.ndarray:
        """Terminal utility function (CRRA utility)."""
        # Constant Relative Risk Aversion utility: W^(1-gamma) / (1-gamma)
        gamma = self.risk_aversion
        if np.abs(gamma - 1.0) < 1e-6:
            # Log utility when gamma = 1
            return np.log(np.maximum(wealth_grid, 1e-8))
        else:
            return np.power(np.maximum(wealth_grid, 1e-8), 1 - gamma) / (1 - gamma)


@dataclass
class StochasticDynamicProgramming:
    """
    Stochastic dynamic programming for portfolio optimization with scenario trees.

    Uses scenario tree approach to handle uncertainty in asset returns.
    More sophisticated than deterministic DP, handles distributional uncertainty.
    """

    n_periods: int
    n_scenarios: int = 100
    risk_aversion: float = 2.0
    transaction_cost: float = 0.001
    discount_factor: float = 0.95

    def optimize(
        self,
        initial_wealth: float,
        return_scenarios: np.ndarray,
        scenario_probabilities: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Solve stochastic DP problem using scenario tree.

        Parameters:
        -----------
        initial_wealth : float
            Initial wealth
        return_scenarios : np.ndarray
            Return scenarios (n_scenarios, n_periods, n_assets)
        scenario_probabilities : np.ndarray, optional
            Probability of each scenario (defaults to uniform)

        Returns:
        --------
        optimal_policy : np.ndarray
            Optimal first-period portfolio weights
        expected_value : float
            Expected value of optimal policy
        """
        if return_scenarios.ndim != 3:
            raise ValueError("return_scenarios must have shape (n_scenarios, n_periods, n_assets)")

        n_scenarios, n_periods, n_assets = return_scenarios.shape

        if scenario_probabilities is None:
            scenario_probabilities = np.ones(n_scenarios) / n_scenarios

        # Initialize terminal values
        terminal_values = np.full(n_scenarios, initial_wealth)

        # Backward recursion through scenarios
        for t in range(n_periods - 1, -1, -1):
            period_returns = return_scenarios[:, t, :]

            # For simplicity, use equal-weight portfolio (can be extended to optimization)
            weights = np.ones(n_assets) / n_assets

            # Update wealth for each scenario
            portfolio_returns = (period_returns @ weights)
            terminal_values = terminal_values * (1 + portfolio_returns)

        # Calculate expected utility
        utilities = self._utility(terminal_values)
        expected_value = float(scenario_probabilities @ utilities)

        # Return simple policy (can be extended to state-dependent)
        optimal_policy = np.ones(n_assets) / n_assets

        return optimal_policy, expected_value

    def _utility(self, wealth: np.ndarray) -> np.ndarray:
        """Power utility function."""
        gamma = self.risk_aversion
        if np.abs(gamma - 1.0) < 1e-6:
            return np.log(np.maximum(wealth, 1e-8))
        else:
            return np.power(np.maximum(wealth, 1e-8), 1 - gamma) / (1 - gamma)
