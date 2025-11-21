"""Tests for reinforcement learning portfolio optimization."""

import numpy as np
import pytest

from neutryx.research.portfolio.reinforcement_learning import (
    PPOAgent,
    A3CAgent,
    PortfolioTradingEnvironment,
)


class TestPPOAgent:
    """Test suite for PPO agent."""

    def test_initialization(self):
        """Test PPO agent initialization."""
        agent = PPOAgent(
            state_dim=5,
            action_dim=3,
            hidden_dim=64,
            learning_rate=3e-4,
            clip_epsilon=0.2
        )

        assert agent.state_dim == 5
        assert agent.action_dim == 3
        assert agent.hidden_dim == 64
        assert agent.learning_rate == 3e-4
        assert agent.clip_epsilon == 0.2

        # Check parameter initialization
        assert agent.policy_weights.shape == (5, 3)
        assert agent.policy_bias.shape == (3,)
        assert agent.value_weights.shape == (5, 64)

    def test_invalid_dimensions(self):
        """Test validation of invalid dimensions."""
        with pytest.raises(ValueError, match="state_dim must be at least 1"):
            PPOAgent(state_dim=0, action_dim=3)

        with pytest.raises(ValueError, match="action_dim must be at least 1"):
            PPOAgent(state_dim=5, action_dim=0)

    def test_policy(self):
        """Test policy computation."""
        agent = PPOAgent(state_dim=4, action_dim=3)

        state = np.array([0.1, 0.2, 0.3, 0.4])
        probs = agent.policy(state)

        # Check output properties
        assert probs.shape == (3,)
        assert np.isclose(np.sum(probs), 1.0)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_value_function(self):
        """Test value function estimation."""
        agent = PPOAgent(state_dim=4, action_dim=3)

        state = np.array([0.1, 0.2, 0.3, 0.4])
        value = agent.value(state)

        # Check output
        assert isinstance(value, float)
        assert np.isfinite(value)

    def test_select_action(self):
        """Test action selection."""
        agent = PPOAgent(state_dim=4, action_dim=3)

        state = np.array([0.1, 0.2, 0.3, 0.4])
        rng = np.random.default_rng(42)

        action, log_prob = agent.select_action(state, rng)

        # Check outputs
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 3
        assert isinstance(log_prob, (float, np.floating))
        assert np.isfinite(log_prob)
        assert log_prob <= 0  # Log probabilities should be non-positive

    def test_compute_gae(self):
        """Test Generalized Advantage Estimation."""
        agent = PPOAgent(state_dim=4, action_dim=3, gamma=0.99, gae_lambda=0.95)

        states = [np.random.randn(4) for _ in range(10)]
        rewards = [0.1, 0.2, 0.3, 0.1, -0.1, 0.2, 0.3, 0.1, 0.2, 0.5]
        dones = [False] * 9 + [True]

        advantages, returns = agent.compute_gae(rewards, states, dones)

        # Check outputs
        assert advantages.shape == (10,)
        assert returns.shape == (10,)
        assert np.isfinite(advantages).all()
        assert np.isfinite(returns).all()

    def test_update(self):
        """Test policy and value network updates."""
        agent = PPOAgent(state_dim=4, action_dim=3, learning_rate=1e-3)

        # Generate fake trajectory
        states = [np.random.randn(4) for _ in range(20)]
        actions = [np.random.randint(0, 3) for _ in range(20)]
        old_log_probs = [np.log(0.3) for _ in range(20)]
        advantages = np.random.randn(20)
        returns = np.random.randn(20)

        # Perform update
        metrics = agent.update(
            states,
            actions,
            old_log_probs,
            advantages,
            returns,
            n_epochs=2,
            batch_size=10
        )

        # Check metrics
        assert 'policy_loss' in metrics
        assert 'value_loss' in metrics
        assert 'entropy' in metrics
        assert np.isfinite(metrics['policy_loss'])
        assert np.isfinite(metrics['value_loss'])
        assert np.isfinite(metrics['entropy'])


class TestA3CAgent:
    """Test suite for A3C agent."""

    def test_initialization(self):
        """Test A3C agent initialization."""
        agent = A3CAgent(
            state_dim=5,
            action_dim=3,
            hidden_dim=64,
            learning_rate=1e-3,
            n_workers=4
        )

        assert agent.state_dim == 5
        assert agent.action_dim == 3
        assert agent.hidden_dim == 64
        assert agent.learning_rate == 1e-3
        assert agent.n_workers == 4
        assert agent.episode_count == 0
        assert agent.total_steps == 0

        # Check parameter initialization
        assert agent.actor_weights.shape == (5, 64)
        assert agent.actor_hidden_to_out.shape == (64, 3)
        assert agent.critic_weights.shape == (5, 64)
        assert agent.critic_hidden_to_out.shape == (64,)

    def test_invalid_parameters(self):
        """Test validation of invalid parameters."""
        with pytest.raises(ValueError, match="state_dim must be at least 1"):
            A3CAgent(state_dim=0, action_dim=3)

        with pytest.raises(ValueError, match="action_dim must be at least 1"):
            A3CAgent(state_dim=5, action_dim=0)

        with pytest.raises(ValueError, match="n_workers must be at least 1"):
            A3CAgent(state_dim=5, action_dim=3, n_workers=0)

    def test_policy(self):
        """Test policy computation."""
        agent = A3CAgent(state_dim=4, action_dim=3)

        state = np.array([0.1, 0.2, 0.3, 0.4])
        probs = agent.policy(state)

        # Check output properties
        assert probs.shape == (3,)
        assert np.isclose(np.sum(probs), 1.0)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_value_function(self):
        """Test value function estimation."""
        agent = A3CAgent(state_dim=4, action_dim=3)

        state = np.array([0.1, 0.2, 0.3, 0.4])
        value = agent.value(state)

        # Check output
        assert isinstance(value, float)
        assert np.isfinite(value)

    def test_select_action(self):
        """Test action selection."""
        agent = A3CAgent(state_dim=4, action_dim=3)

        state = np.array([0.1, 0.2, 0.3, 0.4])
        rng = np.random.default_rng(42)

        action, log_prob = agent.select_action(state, rng)

        # Check outputs
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 3
        assert isinstance(log_prob, (float, np.floating))
        assert np.isfinite(log_prob)

    def test_compute_returns(self):
        """Test n-step returns computation."""
        agent = A3CAgent(state_dim=4, action_dim=3, gamma=0.99)

        rewards = [0.1, 0.2, 0.3, 0.4, 0.5]
        next_state = np.array([0.5, 0.6, 0.7, 0.8])

        # Episode not done
        returns = agent.compute_returns(rewards, next_state, done=False)

        assert returns.shape == (5,)
        assert np.isfinite(returns).all()
        # Returns should be monotonically increasing (backward from terminal)
        assert returns[-1] < returns[0]  # Discounting effect

    def test_compute_returns_terminal(self):
        """Test returns computation at terminal state."""
        agent = A3CAgent(state_dim=4, action_dim=3, gamma=0.99)

        rewards = [0.1, 0.2, 0.3]

        # Episode done (terminal)
        returns = agent.compute_returns(rewards, None, done=True)

        assert returns.shape == (3,)
        assert np.isfinite(returns).all()

    def test_update_worker(self):
        """Test worker update."""
        agent = A3CAgent(state_dim=4, action_dim=3, learning_rate=1e-3)

        # Generate fake trajectory
        states = [np.random.randn(4) for _ in range(10)]
        actions = [np.random.randint(0, 3) for _ in range(10)]
        rewards = [0.1] * 10
        next_state = np.random.randn(4)

        initial_episode_count = agent.episode_count
        initial_steps = agent.total_steps

        # Perform update
        metrics = agent.update_worker(states, actions, rewards, next_state, done=True)

        # Check metrics
        assert 'policy_loss' in metrics
        assert 'value_loss' in metrics
        assert 'entropy' in metrics
        assert 'total_loss' in metrics
        assert 'episode_count' in metrics
        assert 'total_steps' in metrics

        assert np.isfinite(metrics['policy_loss'])
        assert np.isfinite(metrics['value_loss'])
        assert np.isfinite(metrics['entropy'])

        # Check that counters were updated
        assert agent.episode_count == initial_episode_count + 1
        assert agent.total_steps == initial_steps + 10


class TestPortfolioTradingEnvironment:
    """Test suite for portfolio trading environment."""

    def test_initialization(self):
        """Test environment initialization."""
        returns = np.random.randn(100, 3) * 0.01
        env = PortfolioTradingEnvironment(
            returns=returns,
            transaction_cost=0.001,
            risk_penalty=0.5
        )

        assert env.n_periods == 100
        assert env.n_assets == 3
        assert env.transaction_cost == 0.001
        assert env.risk_penalty == 0.5
        assert env.wealth == 1.0

    def test_invalid_returns_shape(self):
        """Test validation of returns shape."""
        returns = np.random.randn(100)  # 1D array

        with pytest.raises(ValueError, match="Returns must have shape"):
            PortfolioTradingEnvironment(returns=returns)

    def test_reset(self):
        """Test environment reset."""
        returns = np.random.randn(50, 3) * 0.01
        env = PortfolioTradingEnvironment(returns=returns)

        # Advance environment
        env.current_step = 10
        env.wealth = 1.5

        # Reset
        state = env.reset()

        assert env.current_step == 0
        assert env.wealth == 1.0
        assert state.shape == (4,)  # 3 assets + 1 wealth
        assert np.allclose(env.current_weights, 1/3)

    def test_step(self):
        """Test environment step."""
        returns = np.random.randn(50, 3) * 0.01
        env = PortfolioTradingEnvironment(returns=returns)

        state = env.reset()

        # Take action
        action_weights = np.array([0.5, 0.3, 0.2])
        next_state, reward, done, info = env.step(action_weights)

        # Check outputs
        assert next_state.shape == (4,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert not done  # First step shouldn't be terminal

        # Check info dict
        assert 'wealth' in info
        assert 'portfolio_return' in info
        assert 'transaction_cost' in info
        assert 'portfolio_variance' in info

        assert np.isfinite(reward)
        assert env.current_step == 1

    def test_episode_completion(self):
        """Test that episode completes after all periods."""
        returns = np.random.randn(10, 3) * 0.01
        env = PortfolioTradingEnvironment(returns=returns)

        env.reset()
        done = False

        for _ in range(10):
            action_weights = np.array([0.4, 0.3, 0.3])
            _, _, done, _ = env.step(action_weights)

        # After 10 steps, episode should be done
        assert done
        assert env.current_step == 10

    def test_transaction_costs(self):
        """Test that transaction costs are applied."""
        returns = np.random.randn(50, 2) * 0.01
        env = PortfolioTradingEnvironment(
            returns=returns,
            transaction_cost=0.01  # 1% transaction cost
        )

        env.reset()
        initial_wealth = env.wealth

        # Take action with weight change
        action_weights = np.array([0.8, 0.2])  # Different from initial [0.5, 0.5]
        _, _, _, info = env.step(action_weights)

        # Transaction cost should have been applied
        assert info['transaction_cost'] > 0
        assert info['transaction_cost'] < initial_wealth

    def test_weight_normalization(self):
        """Test that action weights are normalized."""
        returns = np.random.randn(50, 3) * 0.01
        env = PortfolioTradingEnvironment(returns=returns)

        env.reset()

        # Unnormalized weights
        action_weights = np.array([2.0, 3.0, 1.0])
        _, _, _, info = env.step(action_weights)

        # Weights should have been normalized internally
        assert np.isclose(np.sum(env.current_weights), 1.0)

    def test_wealth_dynamics(self):
        """Test wealth accumulation over time."""
        # Positive returns scenario
        returns = np.ones((10, 2)) * 0.05  # 5% return each period
        env = PortfolioTradingEnvironment(
            returns=returns,
            transaction_cost=0.0,  # No transaction cost
            risk_penalty=0.0  # No risk penalty
        )

        env.reset()
        initial_wealth = env.wealth

        # Equal weight portfolio
        for _ in range(10):
            action_weights = np.array([0.5, 0.5])
            _, _, _, info = env.step(action_weights)

        # Wealth should have grown (compounding)
        final_wealth = info['wealth']
        assert final_wealth > initial_wealth


class TestIntegration:
    """Integration tests for RL portfolio optimization."""

    def test_ppo_training_loop(self):
        """Test PPO agent training loop."""
        returns = np.random.randn(100, 3) * 0.02
        env = PortfolioTradingEnvironment(returns=returns)

        agent = PPOAgent(
            state_dim=4,  # 3 assets + wealth
            action_dim=3,
            learning_rate=1e-3
        )

        # Run one episode
        state = env.reset()
        states, actions, log_probs, rewards, dones = [], [], [], [], []

        done = False
        while not done and len(states) < 50:
            action, log_prob = agent.select_action(state)

            # Map discrete action to weights (simplified)
            weights = np.zeros(3)
            weights[action] = 1.0

            next_state, reward, done, _ = env.step(weights)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)

            state = next_state

        # Compute advantages and update
        advantages, returns_arr = agent.compute_gae(rewards, states, dones)
        metrics = agent.update(
            states,
            actions,
            log_probs,
            advantages,
            returns_arr,
            n_epochs=2,
            batch_size=10
        )

        # Check that training ran successfully
        assert len(states) > 0
        assert 'policy_loss' in metrics

    def test_a3c_training_loop(self):
        """Test A3C agent training loop."""
        returns = np.random.randn(50, 3) * 0.02
        env = PortfolioTradingEnvironment(returns=returns)

        agent = A3CAgent(
            state_dim=4,
            action_dim=3,
            learning_rate=1e-3
        )

        # Run one episode
        state = env.reset()
        states, actions, rewards = [], [], []

        done = False
        while not done and len(states) < 30:
            action, _ = agent.select_action(state)

            # Map discrete action to weights
            weights = np.zeros(3)
            weights[action] = 1.0

            next_state, reward, done, _ = env.step(weights)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        # Update agent
        metrics = agent.update_worker(
            states,
            actions,
            rewards,
            next_state if not done else None,
            done
        )

        # Check that training ran successfully
        assert len(states) > 0
        assert 'policy_loss' in metrics
        assert agent.episode_count == 1
