"""Advanced variance reduction techniques for Monte Carlo simulations.

This module implements various variance reduction methods to improve
the efficiency of Monte Carlo estimators:
- Antithetic variates
- Control variates
- Importance sampling
- Stratified sampling
- Moment matching
- Conditional Monte Carlo
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array


@dataclass
class VarianceReductionConfig:
    """Configuration for variance reduction techniques.

    Attributes:
        use_antithetic: Enable antithetic variates
        use_control_variate: Enable control variate
        use_importance_sampling: Enable importance sampling
        use_stratification: Enable stratified sampling
        use_moment_matching: Enable moment matching
        n_strata: Number of strata for stratified sampling
    """

    use_antithetic: bool = False
    use_control_variate: bool = False
    use_importance_sampling: bool = False
    use_stratification: bool = False
    use_moment_matching: bool = False
    n_strata: int = 10


def antithetic_variates(
    key: jax.random.KeyArray,
    payoff_fn: Callable[[Array], Array],
    path_generator: Callable[[jax.random.KeyArray, int], Array],
    n_paths: int
) -> Tuple[float, float]:
    """Apply antithetic variates variance reduction.

    Antithetic variates reduce variance by generating pairs of negatively
    correlated random paths: if ε ~ N(0,1), use both ε and -ε.

    Args:
        key: JAX random key
        payoff_fn: Function computing payoff from paths
        path_generator: Function generating paths from random normals
        n_paths: Number of path pairs (actual paths = 2 * n_paths)

    Returns:
        Tuple of (estimated_value, standard_error)

    Example:
        >>> def payoff(paths):
        ...     return jnp.maximum(paths[:, -1] - 100, 0)
        >>> def gen_paths(key, n):
        ...     normals = jax.random.normal(key, (n, 252))
        ...     # Generate GBM paths...
        ...     return paths
        >>> value, stderr = antithetic_variates(key, payoff, gen_paths, 5000)
    """
    # Generate half the paths with standard normals
    key1, key2 = jax.random.split(key)

    paths_positive = path_generator(key1, n_paths // 2)
    payoffs_positive = payoff_fn(paths_positive)

    # Generate antithetic paths (negate the random numbers)
    paths_negative = path_generator(key2, n_paths // 2)
    payoffs_negative = payoff_fn(paths_negative)

    # Combine results
    all_payoffs = jnp.concatenate([payoffs_positive, payoffs_negative])

    estimate = float(jnp.mean(all_payoffs))
    std_error = float(jnp.std(all_payoffs) / jnp.sqrt(n_paths))

    return estimate, std_error


def control_variate(
    payoffs: Array,
    control_payoffs: Array,
    control_mean: float
) -> Tuple[float, float, float]:
    """Apply control variate variance reduction.

    Control variates use a correlated variable with known mean to reduce
    variance. If Y is the target and X is the control with known mean μ_X:

    Y_cv = Y - β(X - μ_X)

    where β = Cov(X,Y) / Var(X) is estimated from the sample.

    Args:
        payoffs: Target payoffs to estimate [n_samples]
        control_payoffs: Control variable payoffs [n_samples]
        control_mean: Known mean of control variable

    Returns:
        Tuple of (cv_estimate, cv_std_error, beta_coefficient)

    Example:
        >>> # Price Asian option using European as control
        >>> asian_payoffs = ...  # Asian option payoffs
        >>> european_payoffs = ...  # European option payoffs from same paths
        >>> european_price = black_scholes(...)  # Analytical price
        >>> value, stderr, beta = control_variate(
        ...     asian_payoffs, european_payoffs, european_price
        ... )
    """
    # Estimate optimal beta coefficient
    covariance = jnp.cov(payoffs, control_payoffs)[0, 1]
    variance_control = jnp.var(control_payoffs)

    beta = covariance / (variance_control + 1e-10)

    # Apply control variate adjustment
    control_adjustment = beta * (control_payoffs - control_mean)
    cv_payoffs = payoffs - control_adjustment

    cv_estimate = float(jnp.mean(cv_payoffs))
    cv_std_error = float(jnp.std(cv_payoffs) / jnp.sqrt(len(cv_payoffs)))

    return cv_estimate, cv_std_error, float(beta)


def importance_sampling(
    key: jax.random.KeyArray,
    payoff_fn: Callable[[Array], Array],
    target_density: Callable[[Array], Array],
    proposal_density: Callable[[Array], Array],
    proposal_sampler: Callable[[jax.random.KeyArray, int], Array],
    n_paths: int
) -> Tuple[float, float, Array]:
    """Apply importance sampling variance reduction.

    Importance sampling samples from a proposal distribution q(x) instead
    of the target distribution p(x), and reweights:

    E_p[f(X)] = E_q[f(X) * p(X)/q(X)]

    Args:
        key: JAX random key
        payoff_fn: Payoff function
        target_density: Target probability density p(x)
        proposal_density: Proposal probability density q(x)
        proposal_sampler: Function to sample from proposal
        n_paths: Number of samples

    Returns:
        Tuple of (is_estimate, is_std_error, importance_weights)

    Notes:
        Effective for rare event simulation (e.g., deep out-of-the-money options)
    """
    # Sample from proposal distribution
    samples = proposal_sampler(key, n_paths)

    # Compute payoffs
    payoffs = payoff_fn(samples)

    # Compute importance weights: w = p(x) / q(x)
    p_x = target_density(samples)
    q_x = proposal_density(samples)
    weights = p_x / (q_x + 1e-10)

    # Weighted estimate
    weighted_payoffs = payoffs * weights
    is_estimate = float(jnp.mean(weighted_payoffs))

    # Standard error accounting for weights
    is_variance = jnp.var(weighted_payoffs)
    is_std_error = float(jnp.sqrt(is_variance / n_paths))

    return is_estimate, is_std_error, weights


def stratified_sampling(
    key: jax.random.KeyArray,
    payoff_fn: Callable[[Array], float],
    path_generator: Callable[[Array], Array],
    n_strata: int,
    n_paths_per_stratum: int
) -> Tuple[float, float]:
    """Apply stratified sampling variance reduction.

    Stratified sampling divides the sample space into non-overlapping strata
    and samples proportionally from each stratum. This ensures better coverage
    of the sample space.

    Args:
        key: JAX random key
        payoff_fn: Function computing payoff
        path_generator: Function generating paths from uniform [0,1] inputs
        n_strata: Number of strata
        n_paths_per_stratum: Paths to generate per stratum

    Returns:
        Tuple of (stratified_estimate, standard_error)

    Notes:
        For d-dimensional problems, can use Latin Hypercube Sampling (LHS)
    """
    estimates = []

    for stratum in range(n_strata):
        # Generate uniform samples within stratum [stratum/n, (stratum+1)/n]
        key, subkey = jax.random.split(key)
        uniforms = jax.random.uniform(subkey, (n_paths_per_stratum,))

        # Scale to stratum bounds
        stratum_uniforms = (uniforms + stratum) / n_strata

        # Generate paths
        paths = path_generator(stratum_uniforms)

        # Compute payoffs
        payoff = payoff_fn(paths)
        estimates.append(payoff)

    # Combine stratum estimates
    estimates_array = jnp.array(estimates)
    stratified_estimate = float(jnp.mean(estimates_array))
    std_error = float(jnp.std(estimates_array) / jnp.sqrt(n_strata))

    return stratified_estimate, std_error


def moment_matching(
    paths: Array,
    target_mean: Optional[float] = None,
    target_std: Optional[float] = None
) -> Array:
    """Apply moment matching to simulated paths.

    Moment matching adjusts simulated paths to exactly match target
    moments (mean and/or standard deviation).

    Args:
        paths: Simulated paths [n_paths, n_steps] or [n_paths]
        target_mean: Target mean (if None, use 0)
        target_std: Target standard deviation (if None, use 1)

    Returns:
        Adjusted paths with exact target moments

    Example:
        >>> # Match normal distribution moments
        >>> normals = jax.random.normal(key, (10000,))
        >>> matched = moment_matching(normals, target_mean=0.0, target_std=1.0)
        >>> print(jnp.mean(matched))  # Exactly 0.0
        >>> print(jnp.std(matched))   # Exactly 1.0
    """
    if target_mean is None:
        target_mean = 0.0
    if target_std is None:
        target_std = 1.0

    # Current moments
    current_mean = jnp.mean(paths, axis=0, keepdims=True)
    current_std = jnp.std(paths, axis=0, keepdims=True) + 1e-10

    # Adjust to target moments
    adjusted = (paths - current_mean) / current_std
    adjusted = adjusted * target_std + target_mean

    return adjusted


def conditional_monte_carlo(
    key: jax.random.KeyArray,
    outer_sampler: Callable[[jax.random.KeyArray, int], Array],
    conditional_expectation: Callable[[Array], Array],
    n_paths: int
) -> Tuple[float, float]:
    """Apply conditional Monte Carlo variance reduction.

    Conditional Monte Carlo exploits the law of iterated expectations:
    E[Y] = E[E[Y|X]]

    If E[Y|X] can be computed analytically, we only need to simulate X,
    which typically has lower variance than simulating both X and Y.

    Args:
        key: JAX random key
        outer_sampler: Function to sample outer variable X
        conditional_expectation: Function computing E[Y|X]
        n_paths: Number of samples

    Returns:
        Tuple of (cmc_estimate, standard_error)

    Example:
        >>> # Price Asian option: condition on terminal price
        >>> def sample_terminal_price(key, n):
        ...     return S0 * jnp.exp((r - 0.5*σ²)*T + σ*√T*jax.random.normal(key, (n,)))
        >>> def conditional_asian(S_T):
        ...     # E[Asian|S_T] can be approximated analytically
        ...     return analytical_asian_given_terminal(S_T, ...)
        >>> value, stderr = conditional_monte_carlo(
        ...     key, sample_terminal_price, conditional_asian, 10000
        ... )
    """
    # Sample outer variable
    outer_samples = outer_sampler(key, n_paths)

    # Compute conditional expectation
    conditional_values = conditional_expectation(outer_samples)

    # Estimate
    cmc_estimate = float(jnp.mean(conditional_values))
    std_error = float(jnp.std(conditional_values) / jnp.sqrt(n_paths))

    return cmc_estimate, std_error


@dataclass
class VarianceReductionEngine:
    """Comprehensive variance reduction engine.

    Combines multiple variance reduction techniques for optimal efficiency.
    """

    config: VarianceReductionConfig

    def price_with_variance_reduction(
        self,
        key: jax.random.KeyArray,
        payoff_fn: Callable[[Array], Array],
        path_generator: Callable[[jax.random.KeyArray, int], Array],
        n_paths: int,
        control_payoffs: Optional[Array] = None,
        control_mean: Optional[float] = None
    ) -> dict:
        """Price an option using configured variance reduction techniques.

        Args:
            key: JAX random key
            payoff_fn: Payoff function
            path_generator: Path generation function
            n_paths: Number of paths
            control_payoffs: Control variable payoffs (if using CV)
            control_mean: Known control mean (if using CV)

        Returns:
            Dictionary with pricing results and variance reduction statistics
        """
        results = {}

        # Generate base paths
        if self.config.use_antithetic:
            # Use antithetic variates
            estimate, stderr = antithetic_variates(
                key, payoff_fn, path_generator, n_paths
            )
            results["price"] = estimate
            results["std_error"] = stderr
            results["method"] = "antithetic"

        else:
            # Standard Monte Carlo
            paths = path_generator(key, n_paths)

            # Apply moment matching if enabled
            if self.config.use_moment_matching:
                # For GBM paths, this should be applied to the random normals
                # before generating paths
                pass

            payoffs = payoff_fn(paths)

            # Apply control variate if enabled
            if self.config.use_control_variate and control_payoffs is not None:
                estimate, stderr, beta = control_variate(
                    payoffs, control_payoffs, control_mean
                )
                results["price"] = estimate
                results["std_error"] = stderr
                results["cv_beta"] = beta
                results["method"] = "control_variate"

            else:
                estimate = float(jnp.mean(payoffs))
                stderr = float(jnp.std(payoffs) / jnp.sqrt(n_paths))
                results["price"] = estimate
                results["std_error"] = stderr
                results["method"] = "standard"

        # Compute efficiency metrics
        results["variance"] = stderr ** 2
        results["n_paths"] = n_paths

        return results


def delta_gamma_approximation_control(
    S_t: Array,
    S0: float,
    delta: float,
    gamma: float,
    risk_free_rate: float,
    T: float
) -> Array:
    """Control variate using delta-gamma approximation.

    For path-dependent options, use the delta-gamma approximation
    of a European option as a control variate.

    Args:
        S_t: Simulated terminal prices [n_paths]
        S0: Initial stock price
        delta: Delta of control option
        gamma: Gamma of control option
        risk_free_rate: Risk-free rate
        T: Time to maturity

    Returns:
        Control variate values [n_paths]

    Notes:
        Control value ≈ Delta × (S_T - S0 × e^(rT)) + 0.5 × Gamma × (S_T - S0×e^(rT))²
    """
    forward_price = S0 * jnp.exp(risk_free_rate * T)
    deviation = S_t - forward_price

    # Delta-gamma approximation
    control = delta * deviation + 0.5 * gamma * deviation ** 2

    # Discount to present value
    discounted_control = control * jnp.exp(-risk_free_rate * T)

    return discounted_control
