"""Advanced reinforcement learning algorithms for portfolio optimization.

This module implements state-of-the-art RL algorithms including:
- PPO (Proximal Policy Optimization)
- A3C (Asynchronous Advantage Actor-Critic)
- Additional utilities for training and evaluation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent for portfolio allocation.

    PPO is a policy gradient method that uses a clipped surrogate objective
    to prevent large policy updates, improving training stability.

    References:
    -----------
    Schulman et al. (2017): "Proximal Policy Optimization Algorithms"
    https://arxiv.org/abs/1707.06347

    Parameters:
    -----------
    state_dim : int
        Dimension of state space
    action_dim : int
        Dimension of action space (number of assets)
    hidden_dim : int
        Hidden layer size for neural networks
    learning_rate : float
        Learning rate for policy and value networks
    clip_epsilon : float
        Clipping parameter for PPO objective (typically 0.1 or 0.2)
    gamma : float
        Discount factor
    gae_lambda : float
        Lambda parameter for Generalized Advantage Estimation
    entropy_coef : float
        Entropy bonus coefficient for exploration
    value_coef : float
        Value loss coefficient
    """

    state_dim: int
    action_dim: int
    hidden_dim: int = 64
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    def __post_init__(self):
        if self.state_dim < 1:
            raise ValueError("state_dim must be at least 1")
        if self.action_dim < 1:
            raise ValueError("action_dim must be at least 1")

        # Initialize policy network parameters (simple linear model)
        rng = np.random.default_rng(42)
        self.policy_weights = rng.normal(0, 0.1, (self.state_dim, self.action_dim))
        self.policy_bias = np.zeros(self.action_dim)

        # Initialize value network parameters
        self.value_weights = rng.normal(0, 0.1, (self.state_dim, self.hidden_dim))
        self.value_hidden_to_out = rng.normal(0, 0.1, self.hidden_dim)
        self.value_bias = np.zeros(self.hidden_dim)
        self.value_out_bias = 0.0

    def policy(self, state: np.ndarray) -> np.ndarray:
        """Compute action probabilities given state."""
        state = np.asarray(state, dtype=float)
        logits = state @ self.policy_weights + self.policy_bias
        # Softmax
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)

    def value(self, state: np.ndarray) -> float:
        """Estimate state value."""
        state = np.asarray(state, dtype=float)
        hidden = np.tanh(state @ self.value_weights + self.value_bias)
        return float(hidden @ self.value_hidden_to_out + self.value_out_bias)

    def select_action(self, state: np.ndarray, rng: Optional[np.random.Generator] = None) -> Tuple[int, float]:
        """Select action and return action index and log probability."""
        if rng is None:
            rng = np.random.default_rng()

        probs = self.policy(state)
        action = rng.choice(self.action_dim, p=probs)
        log_prob = np.log(np.clip(probs[action], 1e-8, 1.0))

        return action, log_prob

    def compute_gae(
        self,
        rewards: List[float],
        states: List[np.ndarray],
        dones: List[bool]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).

        GAE provides a variance-reduced estimator of advantages by using
        a weighted combination of n-step returns.
        """
        values = np.array([self.value(s) for s in states])
        advantages = np.zeros(len(rewards))
        gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values

        return advantages, returns

    def update(
        self,
        states: List[np.ndarray],
        actions: List[int],
        old_log_probs: List[float],
        advantages: np.ndarray,
        returns: np.ndarray,
        n_epochs: int = 10,
        batch_size: int = 64
    ) -> dict:
        """
        Update policy and value networks using PPO objective.

        Returns:
        --------
        metrics : dict
            Training metrics including policy loss, value loss, entropy
        """
        states_arr = np.array(states)
        actions_arr = np.array(actions)
        old_log_probs_arr = np.array(old_log_probs)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(n_epochs):
            # Mini-batch updates
            indices = np.random.permutation(len(states))

            for start in range(0, len(states), batch_size):
                end = min(start + batch_size, len(states))
                batch_idx = indices[start:end]

                batch_states = states_arr[batch_idx]
                batch_actions = actions_arr[batch_idx]
                batch_old_log_probs = old_log_probs_arr[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # Compute current log probs
                batch_probs = np.array([self.policy(s) for s in batch_states])
                batch_log_probs = np.log(
                    np.clip(batch_probs[np.arange(len(batch_actions)), batch_actions], 1e-8, 1.0)
                )

                # Compute ratio for PPO objective
                ratio = np.exp(batch_log_probs - batch_old_log_probs)

                # Clipped surrogate objective
                surrogate1 = ratio * batch_advantages
                surrogate2 = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -np.mean(np.minimum(surrogate1, surrogate2))

                # Value loss
                batch_values = np.array([self.value(s) for s in batch_states])
                value_loss = np.mean((batch_values - batch_returns) ** 2)

                # Entropy bonus
                entropy = -np.mean(np.sum(batch_probs * np.log(np.clip(batch_probs, 1e-8, 1.0)), axis=1))

                # Total loss
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Gradient descent (simplified - in practice, use automatic differentiation)
                self._gradient_step(batch_states, batch_actions, batch_advantages, batch_returns)

                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_entropy += entropy
                n_updates += 1

        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates
        }

    def _gradient_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray
    ):
        """Perform gradient descent step (simplified)."""
        # Simplified gradient update
        # In practice, use JAX or PyTorch for automatic differentiation

        # Policy gradient (REINFORCE-style)
        for i, (state, action, advantage) in enumerate(zip(states, actions, advantages)):
            probs = self.policy(state)
            grad_log_prob = -probs
            grad_log_prob[action] += 1.0

            # Update policy weights
            self.policy_weights += self.learning_rate * np.outer(state, grad_log_prob * advantage)
            self.policy_bias += self.learning_rate * grad_log_prob * advantage

        # Value gradient
        for state, target in zip(states, returns):
            predicted = self.value(state)
            error = predicted - target

            # Backprop through value network
            hidden = np.tanh(state @ self.value_weights + self.value_bias)
            grad_out = error
            grad_hidden = grad_out * self.value_hidden_to_out
            grad_hidden_input = grad_hidden * (1 - hidden ** 2)

            self.value_weights -= self.learning_rate * self.value_coef * np.outer(state, grad_hidden_input)
            self.value_hidden_to_out -= self.learning_rate * self.value_coef * hidden * grad_out


@dataclass
class A3CAgent:
    """
    Asynchronous Advantage Actor-Critic (A3C) agent for portfolio optimization.

    A3C uses multiple parallel workers to collect experience and update
    a shared policy network asynchronously, improving sample efficiency.

    References:
    -----------
    Mnih et al. (2016): "Asynchronous Methods for Deep Reinforcement Learning"
    https://arxiv.org/abs/1602.01783

    Parameters:
    -----------
    state_dim : int
        Dimension of state space
    action_dim : int
        Dimension of action space (number of assets)
    hidden_dim : int
        Hidden layer size for neural networks
    learning_rate : float
        Learning rate for shared network
    gamma : float
        Discount factor
    entropy_coef : float
        Entropy bonus coefficient
    value_coef : float
        Value loss coefficient
    n_workers : int
        Number of parallel workers
    """

    state_dim: int
    action_dim: int
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    n_workers: int = 4
    max_grad_norm: float = 0.5

    def __post_init__(self):
        if self.state_dim < 1:
            raise ValueError("state_dim must be at least 1")
        if self.action_dim < 1:
            raise ValueError("action_dim must be at least 1")
        if self.n_workers < 1:
            raise ValueError("n_workers must be at least 1")

        # Initialize shared network parameters
        rng = np.random.default_rng(42)

        # Actor network (policy)
        self.actor_weights = rng.normal(0, 0.1, (self.state_dim, self.hidden_dim))
        self.actor_hidden_to_out = rng.normal(0, 0.1, (self.hidden_dim, self.action_dim))
        self.actor_bias = np.zeros(self.hidden_dim)
        self.actor_out_bias = np.zeros(self.action_dim)

        # Critic network (value function)
        self.critic_weights = rng.normal(0, 0.1, (self.state_dim, self.hidden_dim))
        self.critic_hidden_to_out = rng.normal(0, 0.1, self.hidden_dim)
        self.critic_bias = np.zeros(self.hidden_dim)
        self.critic_out_bias = 0.0

        # Training statistics
        self.episode_count = 0
        self.total_steps = 0

    def policy(self, state: np.ndarray) -> np.ndarray:
        """Compute action probabilities given state."""
        state = np.asarray(state, dtype=float)
        hidden = np.tanh(state @ self.actor_weights + self.actor_bias)
        logits = hidden @ self.actor_hidden_to_out + self.actor_out_bias
        # Softmax
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)

    def value(self, state: np.ndarray) -> float:
        """Estimate state value."""
        state = np.asarray(state, dtype=float)
        hidden = np.tanh(state @ self.critic_weights + self.critic_bias)
        return float(hidden @ self.critic_hidden_to_out + self.critic_out_bias)

    def select_action(self, state: np.ndarray, rng: Optional[np.random.Generator] = None) -> Tuple[int, float]:
        """Select action and return action index and log probability."""
        if rng is None:
            rng = np.random.default_rng()

        probs = self.policy(state)
        action = rng.choice(self.action_dim, p=probs)
        log_prob = np.log(np.clip(probs[action], 1e-8, 1.0))

        return action, log_prob

    def compute_returns(
        self,
        rewards: List[float],
        next_state: Optional[np.ndarray],
        done: bool
    ) -> np.ndarray:
        """Compute n-step returns with bootstrapping."""
        returns = np.zeros(len(rewards))

        if done:
            R = 0.0
        else:
            R = self.value(next_state) if next_state is not None else 0.0

        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.gamma * R
            returns[t] = R

        return returns

    def update_worker(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        next_state: Optional[np.ndarray],
        done: bool
    ) -> dict:
        """
        Update shared network based on worker's trajectory.

        This method would be called asynchronously by each worker in a
        true A3C implementation. Here we provide a synchronous version.

        Returns:
        --------
        metrics : dict
            Training metrics including policy loss, value loss, entropy
        """
        states_arr = np.array(states)
        actions_arr = np.array(actions)

        # Compute returns
        returns = self.compute_returns(rewards, next_state, done)

        # Compute advantages
        values = np.array([self.value(s) for s in states])
        advantages = returns - values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute losses
        probs = np.array([self.policy(s) for s in states])
        log_probs = np.log(np.clip(probs[np.arange(len(actions)), actions], 1e-8, 1.0))

        # Policy loss (advantage actor-critic)
        policy_loss = -np.mean(log_probs * advantages)

        # Value loss
        value_loss = np.mean((values - returns) ** 2)

        # Entropy bonus
        entropy = -np.mean(np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0)), axis=1))

        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        # Gradient update (simplified)
        self._gradient_step(states_arr, actions_arr, advantages, returns)

        self.episode_count += 1
        self.total_steps += len(states)

        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'total_loss': total_loss,
            'episode_count': self.episode_count,
            'total_steps': self.total_steps
        }

    def _gradient_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray
    ):
        """Perform gradient descent on shared parameters."""
        # Actor gradient
        for state, action, advantage in zip(states, actions, advantages):
            probs = self.policy(state)
            hidden = np.tanh(state @ self.actor_weights + self.actor_bias)

            # Gradient of log policy
            grad_log_prob = -probs
            grad_log_prob[action] += 1.0

            # Backprop through actor network
            grad_out = grad_log_prob * advantage
            grad_hidden = self.actor_hidden_to_out @ grad_out
            grad_hidden_input = grad_hidden * (1 - hidden ** 2)

            # Update actor parameters
            self.actor_weights -= self.learning_rate * np.outer(state, grad_hidden_input)
            self.actor_hidden_to_out -= self.learning_rate * np.outer(hidden, grad_out)

        # Critic gradient
        for state, target in zip(states, returns):
            predicted = self.value(state)
            error = predicted - target
            hidden = np.tanh(state @ self.critic_weights + self.critic_bias)

            # Backprop through critic network
            grad_out = error * self.value_coef
            grad_hidden = grad_out * self.critic_hidden_to_out
            grad_hidden_input = grad_hidden * (1 - hidden ** 2)

            # Update critic parameters
            self.critic_weights -= self.learning_rate * np.outer(state, grad_hidden_input)
            self.critic_hidden_to_out -= self.learning_rate * hidden * grad_out


class PortfolioTradingEnvironment:
    """
    Trading environment for RL portfolio optimization.

    Provides a gym-like interface for training RL agents on portfolio allocation tasks.
    Supports transaction costs, risk constraints, and realistic market dynamics.
    """

    def __init__(
        self,
        returns: np.ndarray,
        transaction_cost: float = 0.001,
        risk_penalty: float = 0.5
    ):
        """
        Initialize trading environment.

        Parameters:
        -----------
        returns : np.ndarray
            Historical returns (n_periods, n_assets)
        transaction_cost : float
            Proportional transaction cost
        risk_penalty : float
            Penalty for portfolio variance
        """
        if returns.ndim != 2:
            raise ValueError("Returns must have shape (n_periods, n_assets)")

        self.returns = returns
        self.n_periods, self.n_assets = returns.shape
        self.transaction_cost = transaction_cost
        self.risk_penalty = risk_penalty

        self.current_step = 0
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        self.wealth = 1.0

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = 0
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        self.wealth = 1.0
        return self._get_state()

    def step(self, action_weights: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one step in the environment.

        Parameters:
        -----------
        action_weights : np.ndarray
            New portfolio weights

        Returns:
        --------
        state : np.ndarray
            Next state
        reward : float
            Reward for this step
        done : bool
            Whether episode is finished
        info : dict
            Additional information
        """
        # Normalize weights
        action_weights = action_weights / np.sum(action_weights)

        # Calculate transaction cost
        weight_change = np.abs(action_weights - self.current_weights)
        cost = self.transaction_cost * self.wealth * np.sum(weight_change)

        # Calculate portfolio return
        period_returns = self.returns[self.current_step]
        portfolio_return = float(action_weights @ period_returns)

        # Update wealth
        self.wealth = self.wealth * (1 + portfolio_return) - cost

        # Calculate reward (risk-adjusted return)
        portfolio_variance = float(action_weights @ np.cov(self.returns.T) @ action_weights)
        reward = portfolio_return - self.risk_penalty * portfolio_variance - cost / self.wealth

        # Update state
        self.current_weights = action_weights
        self.current_step += 1

        done = self.current_step >= self.n_periods
        state = self._get_state() if not done else np.zeros(self.n_assets + 1)

        info = {
            'wealth': self.wealth,
            'portfolio_return': portfolio_return,
            'transaction_cost': cost,
            'portfolio_variance': portfolio_variance
        }

        return state, reward, done, info

    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        # State: current returns + wealth
        if self.current_step < self.n_periods:
            return np.concatenate([
                self.returns[self.current_step],
                [self.wealth]
            ])
        else:
            return np.zeros(self.n_assets + 1)
