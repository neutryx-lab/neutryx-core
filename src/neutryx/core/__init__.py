"""Core computational infrastructure for Neutryx.

This module provides Monte Carlo simulation, automatic differentiation,
configuration management, infrastructure, and utility functions.
"""

from . import autodiff, config, infrastructure, utils
from .engine import (
    Array,
    MCConfig,
    MCPaths,
    discount_factor,
    mc_expectation,
    present_value,
    price_vanilla_jump_diffusion_mc,
    price_vanilla_mc,
    resolve_schedule,
    simulate_gbm,
    simulate_jump_diffusion,
    time_grid,
)
from .execution import MeshConfig, mesh_context, mesh_named_sharding, mesh_pjit, mesh_xmap
from .grid import (
    log_uniform,
    merge_grids,
    refine_grid,
)
from .grid import (
    time_grid as grid_time_grid,
)
from .grid import (
    uniform as uniform_grid,
)
from .rng import KeySeq, split_key
from .rng import normal as rng_normal
from .rng import uniform as rng_uniform

__all__ = [
    "Array",
    "KeySeq",
    "MCConfig",
    "MCPaths",
    "MeshConfig",
    "autodiff",
    "config",
    "discount_factor",
    "grid_time_grid",
    "infrastructure",
    "log_uniform",
    "mc_expectation",
    "merge_grids",
    "mesh_context",
    "mesh_named_sharding",
    "mesh_pjit",
    "mesh_xmap",
    "present_value",
    "price_vanilla_jump_diffusion_mc",
    "price_vanilla_mc",
    "refine_grid",
    "resolve_schedule",
    "rng_normal",
    "rng_uniform",
    "simulate_gbm",
    "simulate_jump_diffusion",
    "split_key",
    "time_grid",
    "uniform_grid",
    "utils",
]
