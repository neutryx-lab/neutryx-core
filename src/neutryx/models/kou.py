"""Kou double exponential jump-diffusion model.

The Kou model extends the Merton jump-diffusion by using asymmetric double exponential
jump size distributions, allowing for different behavior in upward and downward jumps.

References:
    Kou, S. G. (2002). A jump-diffusion model for option pricing.
    Management Science, 48(8), 1086-1101.
"""
import jax
import jax.numpy as jnp

from neutryx.core.engine import Array, MCConfig


def simulate_kou(
    key: jax.random.KeyArray,
    S0: float,
    mu: float,
    sigma: float,
    lam: float,
    p: float,
    eta1: float,
    eta2: float,
    T: float,
    cfg: MCConfig,
) -> Array:
    """Simulate Kou double exponential jump-diffusion paths.

    The model is:
    dS/S = mu dt + sigma dW + dJ

    where J is a compound Poisson process with jump rate lam.
    Each jump has size Y - 1, where log(Y) has an asymmetric double exponential distribution:
    - With probability p: exponential distribution with rate eta1 (upward jumps)
    - With probability 1-p: exponential distribution with rate eta2 (downward jumps)

    Parameters
    ----------
    key : jax.random.KeyArray
        PRNG key
    S0 : float
        Initial asset price
    mu : float
        Drift (r - q)
    sigma : float
        Diffusion volatility
    lam : float
        Jump intensity (jumps per year)
    p : float
        Probability of upward jump (0 <= p <= 1)
    eta1 : float
        Rate parameter for upward jumps (> 1)
    eta2 : float
        Rate parameter for downward jumps (> 0)
    T : float
        Time to maturity
    cfg : MCConfig
        Monte Carlo configuration

    Returns
    -------
    Array
        Simulated paths of shape [paths, steps+1]
    """
    dtype = cfg.dtype
    dt = T / cfg.steps
    sqrt_dt = jnp.sqrt(dt)

    # Adjust drift for jump compensation
    # E[Y - 1] = p / (eta1 - 1) - (1 - p) / (eta2 + 1)
    jump_compensation = lam * (p / (eta1 - 1) - (1 - p) / (eta2 + 1))
    drift = (mu - 0.5 * sigma**2 - jump_compensation) * dt
    vol = sigma * sqrt_dt

    # Split keys for different random components
    key_norm, key_pois, key_jump_dir, key_jump_size = jax.random.split(key, 4)

    # Generate standard normals for diffusion
    normals = jax.random.normal(key_norm, (cfg.base_paths, cfg.steps), dtype=dtype)

    # Generate Poisson jumps
    jump_counts = jax.random.poisson(key_pois, lam=lam * dt,
                                     shape=(cfg.base_paths, cfg.steps))

    # Determine jump direction (up or down)
    jump_directions = jax.random.bernoulli(key_jump_dir, p=p,
                                          shape=(cfg.base_paths, cfg.steps))

    # Generate jump sizes from exponential distributions
    exp_up = jax.random.exponential(key_jump_size,
                                    shape=(cfg.base_paths, cfg.steps)) / eta1
    key_jump_size2 = jax.random.fold_in(key_jump_size, 1)
    exp_down = -jax.random.exponential(key_jump_size2,
                                       shape=(cfg.base_paths, cfg.steps)) / eta2

    # Combine: jump_size = direction ? exp_up : exp_down
    jump_sizes = jnp.where(jump_directions, exp_up, exp_down)

    # Total jump contribution: sum over all jumps in each time step
    # For simplicity, we approximate multiple jumps by scaling
    log_jump_total = jump_sizes * jump_counts.astype(dtype)

    # Combine diffusion and jumps
    increments = drift + vol * normals + log_jump_total

    if cfg.antithetic:
        anti_normals = -normals
        increments = jnp.concatenate(
            [increments, drift + vol * anti_normals + log_jump_total], axis=0
        )

    log_S0 = jnp.log(jnp.asarray(S0, dtype=dtype))
    total_paths = increments.shape[0]
    cum_returns = jnp.cumsum(increments, axis=1)
    log_paths = jnp.concatenate(
        [jnp.full((total_paths, 1), log_S0, dtype=dtype), log_S0 + cum_returns],
        axis=1,
    )

    return jnp.exp(log_paths)


def price_vanilla_kou_mc(
    key: jax.random.KeyArray,
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    lam: float,
    p: float,
    eta1: float,
    eta2: float,
    cfg: MCConfig,
    kind: str = "call",
) -> float:
    """Price vanilla European option under Kou model using Monte Carlo.

    Parameters
    ----------
    key : jax.random.KeyArray
        PRNG key
    S0 : float
        Initial asset price
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    q : float
        Dividend yield
    sigma : float
        Diffusion volatility
    lam : float
        Jump intensity
    p : float
        Probability of upward jump
    eta1 : float
        Rate parameter for upward jumps (> 1)
    eta2 : float
        Rate parameter for downward jumps (> 0)
    cfg : MCConfig
        Monte Carlo configuration
    kind : str
        "call" or "put"

    Returns
    -------
    float
        Option price
    """
    mu = r - q
    paths = simulate_kou(key, S0, mu, sigma, lam, p, eta1, eta2, T, cfg)
    ST = paths[:, -1]

    if kind == "call":
        payoffs = jnp.maximum(ST - K, 0.0)
    elif kind == "put":
        payoffs = jnp.maximum(K - ST, 0.0)
    else:
        raise ValueError(f"Unknown option kind: {kind}")

    discount = jnp.exp(-r * T)
    return float((discount * payoffs).mean())


__all__ = [
    "simulate_kou",
    "price_vanilla_kou_mc",
]
