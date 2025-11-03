"""Quasi-Monte Carlo drivers and Multi-Level Monte Carlo orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from scipy.stats import qmc

from neutryx.products.vanilla import European

from neutryx.core.engine import MCConfig, simulate_gbm

Array = jnp.ndarray
Payoff = Callable[[Array], Array]

__all__ = [
    "SobolGenerator",
    "HaltonGenerator",
    "MLMCLevel",
    "MLMCResult",
    "MLMCOrchestrator",
    "price_european_qmc",
    "price_european_mlmc",
]


@dataclass(frozen=True)
class SobolGenerator:
    """State-less Sobol sequence generator with normal transformation helpers."""

    scramble: bool = True
    seed: int | None = None
    dtype: jnp.dtype = jnp.float32

    def _sampler(self, dim: int) -> qmc.Sobol:
        if dim <= 0:
            raise ValueError("Sobol dimension must be positive.")
        return qmc.Sobol(d=dim, scramble=self.scramble, seed=self.seed)

    def sample(self, n: int, dim: int, *, skip: int = 0) -> Array:
        """Return Sobol points in ``[0, 1]^dim`` converted to ``jax`` arrays."""

        if n <= 0:
            raise ValueError("Number of Sobol samples must be positive.")
        sampler = self._sampler(dim)
        if skip:
            sampler.fast_forward(skip)
        points = sampler.random(n)
        return jnp.asarray(points, dtype=self.dtype)

    def normal(self, n: int, dim: int, *, skip: int = 0, center: bool = True) -> Array:
        """Return quasi-random standard normal draws using inverse-CDF mapping."""

        uniforms = self.sample(n, dim, skip=skip)
        eps = jnp.finfo(self.dtype).tiny
        clipped = jnp.clip(uniforms, eps, 1.0 - eps)
        normals = norm.ppf(clipped).astype(self.dtype)
        if center:
            normals = normals - jnp.mean(normals, axis=0, keepdims=True)
        return normals

    def spawn(self, seed_offset: int) -> "SobolGenerator":
        """Return a generator with a deterministically shifted seed."""

        base_seed = 0 if self.seed is None else self.seed
        return SobolGenerator(scramble=self.scramble, seed=base_seed + seed_offset, dtype=self.dtype)


@dataclass(frozen=True)
class HaltonGenerator:
    """Halton sequence generator with normal transformation helpers.

    Halton sequences are low-discrepancy quasi-random sequences based on
    coprime bases. They are particularly effective for low to moderate
    dimensional problems.

    Attributes:
        scramble: Whether to use random digit scrambling
        seed: Random seed for scrambling
        dtype: Data type for generated arrays
    """

    scramble: bool = True
    seed: int | None = None
    dtype: jnp.dtype = jnp.float32

    def _sampler(self, dim: int) -> qmc.Halton:
        """Create a Halton sampler for the given dimension."""
        if dim <= 0:
            raise ValueError("Halton dimension must be positive.")
        return qmc.Halton(d=dim, scramble=self.scramble, seed=self.seed)

    def sample(self, n: int, dim: int, *, skip: int = 0) -> Array:
        """Return Halton points in [0, 1]^dim converted to JAX arrays.

        Args:
            n: Number of samples
            dim: Dimension of each sample
            skip: Number of initial points to skip (for decorrelation)

        Returns:
            Array of shape [n, dim] with quasi-random samples
        """
        if n <= 0:
            raise ValueError("Number of Halton samples must be positive.")

        sampler = self._sampler(dim)

        if skip:
            sampler.fast_forward(skip)

        points = sampler.random(n)
        return jnp.asarray(points, dtype=self.dtype)

    def normal(self, n: int, dim: int, *, skip: int = 0, center: bool = True) -> Array:
        """Return quasi-random standard normal draws using inverse-CDF mapping.

        Args:
            n: Number of samples
            dim: Dimension of each sample
            skip: Number of initial points to skip
            center: Whether to center the normals to have zero mean

        Returns:
            Array of shape [n, dim] with quasi-random normal samples
        """
        uniforms = self.sample(n, dim, skip=skip)
        eps = jnp.finfo(self.dtype).tiny
        clipped = jnp.clip(uniforms, eps, 1.0 - eps)
        normals = norm.ppf(clipped).astype(self.dtype)

        if center:
            normals = normals - jnp.mean(normals, axis=0, keepdims=True)

        return normals

    def spawn(self, seed_offset: int) -> "HaltonGenerator":
        """Return a generator with a deterministically shifted seed.

        Args:
            seed_offset: Offset to add to the base seed

        Returns:
            New HaltonGenerator instance with shifted seed
        """
        base_seed = 0 if self.seed is None else self.seed
        return HaltonGenerator(
            scramble=self.scramble,
            seed=base_seed + seed_offset,
            dtype=self.dtype
        )


@dataclass(frozen=True)
class MLMCLevel:
    """Configuration for a single MLMC level."""

    steps: int
    paths: int

    def __post_init__(self) -> None:  # type: ignore[override]
        if self.steps <= 0:
            raise ValueError("MLMCLevel.steps must be positive.")
        if self.paths <= 0:
            raise ValueError("MLMCLevel.paths must be positive.")


@dataclass
class MLMCResult:
    """Summary of an MLMC simulation."""

    price: Array
    level_estimates: List[Array]
    level_variances: List[Array]
    total_paths: int


class MLMCOrchestrator:
    """Coordinate MLMC pricing using pseudo- or quasi-random drivers."""

    def __init__(
        self,
        levels: Sequence[MLMCLevel],
        *,
        generator: SobolGenerator | None = None,
        key: jax.random.KeyArray | None = None,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        if not levels:
            raise ValueError("At least one MLMC level is required.")
        self.levels = list(levels)
        self.generator = generator
        self.dtype = dtype
        self._key = key or jax.random.PRNGKey(0)
        self._validate_levels()

    def _validate_levels(self) -> None:
        prev_steps = None
        for idx, level in enumerate(self.levels):
            if prev_steps is not None:
                if level.steps % prev_steps != 0:
                    raise ValueError("Level steps must be multiples of previous level steps.")
            prev_steps = level.steps

    def _draw_normals(self, level_idx: int, n: int, dim: int) -> Array:
        if self.generator is not None:
            level_gen = self.generator.spawn(level_idx)
            return level_gen.normal(n, dim, center=False)
        key = jax.random.fold_in(self._key, level_idx)
        return jax.random.normal(key, (n, dim), dtype=self.dtype)

    def _simulate_level(
        self,
        level_idx: int,
        payoff_fn: Payoff,
        *,
        S0: float,
        mu: float,
        sigma: float,
        T: float,
    ) -> tuple[Array, Array] | tuple[Array]:
        level = self.levels[level_idx]
        normals_fine = self._draw_normals(level_idx, level.paths, level.steps)
        cfg_fine = MCConfig(steps=level.steps, paths=level.paths, dtype=self.dtype)
        paths_fine = simulate_gbm(
            jax.random.PRNGKey(0),
            S0,
            mu,
            sigma,
            T,
            cfg_fine,
            normal_draws=normals_fine,
        )
        fine_payoff = payoff_fn(paths_fine)
        if level_idx == 0:
            return (fine_payoff,)

        coarse_steps = self.levels[level_idx - 1].steps
        factor = level.steps // coarse_steps
        normals_coarse = normals_fine.reshape(level.paths, coarse_steps, factor)
        normals_coarse = normals_coarse.sum(axis=-1) / jnp.sqrt(float(factor))
        cfg_coarse = MCConfig(steps=coarse_steps, paths=level.paths, dtype=self.dtype)
        paths_coarse = simulate_gbm(
            jax.random.PRNGKey(0),
            S0,
            mu,
            sigma,
            T,
            cfg_coarse,
            normal_draws=normals_coarse,
        )
        coarse_payoff = payoff_fn(paths_coarse)
        return fine_payoff, coarse_payoff

    def run(
        self,
        payoff_fn: Payoff,
        *,
        S0: float,
        mu: float,
        sigma: float,
        T: float,
    ) -> MLMCResult:
        contributions: List[Array] = []
        variances: List[Array] = []
        total_paths = 0
        for idx, level in enumerate(self.levels):
            simulated = self._simulate_level(idx, payoff_fn, S0=S0, mu=mu, sigma=sigma, T=T)
            if idx == 0:
                payoffs = simulated[0]
            else:
                payoffs = simulated[0] - simulated[1]
            estimate = jnp.mean(payoffs)
            contributions.append(estimate)
            variances.append(jnp.var(payoffs))
            total_paths += level.paths
        price = jnp.sum(jnp.stack(contributions))
        return MLMCResult(price=price, level_estimates=contributions, level_variances=variances, total_paths=total_paths)


def _discounted_european_payoff(paths: Array, option: European, r: float) -> Array:
    ST = paths[:, -1]
    payoffs = option.payoff(ST)
    return jnp.exp(-r * option.T) * payoffs


def price_european_qmc(
    option: European,
    *,
    S0: float,
    r: float,
    q: float,
    sigma: float,
    generator: SobolGenerator | None = None,
    paths: int = 8192,
    steps: int = 128,
    dtype: jnp.dtype = jnp.float32,
) -> Array:
    """Price a European vanilla using quasi Monte Carlo."""

    mu = r - q
    gen = generator or SobolGenerator(dtype=dtype)
    normals = gen.normal(paths, steps, center=False)
    cfg = MCConfig(steps=steps, paths=paths, dtype=dtype)
    paths_sim = simulate_gbm(
        jax.random.PRNGKey(0),
        S0,
        mu,
        sigma,
        option.T,
        cfg,
        normal_draws=normals,
    )
    payoffs = _discounted_european_payoff(paths_sim, option, r)
    return jnp.mean(payoffs)


def price_european_mlmc(
    option: European,
    levels: Sequence[MLMCLevel],
    *,
    S0: float,
    r: float,
    q: float,
    sigma: float,
    generator: SobolGenerator | None = None,
    key: jax.random.KeyArray | None = None,
    dtype: jnp.dtype = jnp.float32,
) -> MLMCResult:
    """MLMC estimator for a European option under GBM dynamics."""

    payoff_fn = lambda paths: _discounted_european_payoff(paths, option, r)
    orchestrator = MLMCOrchestrator(
        levels,
        generator=generator,
        key=key,
        dtype=dtype,
    )
    mu = r - q
    return orchestrator.run(payoff_fn, S0=S0, mu=mu, sigma=sigma, T=option.T)
