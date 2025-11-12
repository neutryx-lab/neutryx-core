"""JAX PyTree-based market data grid for high-performance batch operations.

Pre-computes all market data (discount factors, volatilities, FX rates) on a fixed
time grid, eliminating runtime interpolation and enabling full JIT compilation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax import Array

from neutryx.data.indices import IndexMapping


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MarketDataGrid:
    """Pre-computed market data on a fixed time grid.

    All market data is pre-interpolated onto a common time grid, allowing
    O(1) lookups without runtime interpolation. This structure is registered
    as a JAX pytree, making it fully JIT-compilable.

    Attributes
    ----------
    time_grid : Array
        Common time grid [n_times] in years
    discount_factors : Array
        Pre-computed discount factors [n_currencies, n_times]
    zero_rates : Array
        Pre-computed zero rates [n_currencies, n_times]
    forward_rates : Array
        Pre-computed instantaneous forward rates [n_currencies, n_times]
    implied_vols : Array
        Pre-computed implied volatilities [n_assets, n_times, n_strikes]
    strike_grid : Array
        Strike grid used for vol surface [n_strikes]
    fx_spots : Array
        FX spot rates [n_fx_pairs]
    currency_mapping : IndexMapping
        Mapping from currency codes to indices
    asset_mapping : IndexMapping
        Mapping from asset identifiers to indices
    fx_pair_mapping : IndexMapping
        Mapping from FX pair tuples to indices
    metadata : dict[str, Any]
        Optional metadata (valuation date, grid construction params, etc.)

    Examples
    --------
    >>> # Build time grid for CVA calculation (600 steps, 5 years)
    >>> time_grid = jnp.linspace(0.0, 5.0, 600)
    >>>
    >>> # Pre-compute discount factors for all currencies
    >>> # Shape: [2 currencies × 600 times]
    >>> discount_factors = jnp.array([
    ...     [jnp.exp(-0.05 * t) for t in time_grid],  # USD
    ...     [jnp.exp(-0.03 * t) for t in time_grid],  # EUR
    ... ])
    >>>
    >>> # Create market data grid
    >>> grid = MarketDataGrid(
    ...     time_grid=time_grid,
    ...     discount_factors=discount_factors,
    ...     zero_rates=jnp.array([[0.05] * 600, [0.03] * 600]),
    ...     forward_rates=jnp.array([[0.05] * 600, [0.03] * 600]),
    ...     implied_vols=jnp.ones((1, 600, 10)) * 0.2,  # 1 asset, 10 strikes
    ...     strike_grid=jnp.linspace(80, 120, 10),
    ...     fx_spots=jnp.array([1.1]),  # EUR/USD
    ...     currency_mapping=IndexMapping.from_values(["EUR", "USD"]),
    ...     asset_mapping=IndexMapping.from_values(["SPX"]),
    ...     fx_pair_mapping=IndexMapping.from_values([("EUR", "USD")]),
    ... )
    >>>
    >>> # Fast lookup: get USD discount factor at time index 300
    >>> usd_idx = grid.currency_mapping.encode("USD")
    >>> df = grid.discount_factors[usd_idx, 300]  # O(1), JIT-compilable!

    Notes
    -----
    Performance Characteristics:
    - Grid construction: O(n_times × n_curves) - done once
    - Runtime lookup: O(1) - array indexing only
    - Memory: ~8 bytes × n_currencies × n_times (float64)
      - For 10 currencies × 600 times = 48KB
    - GPU transfer: Efficient (contiguous memory)

    Design Trade-offs:
    - Pre-computation cost amortized over many pricing calls
    - Fixed time grid (interpolation accuracy depends on grid density)
    - Memory scales with grid size, but typical grids are small
    """

    time_grid: Array
    discount_factors: Array
    zero_rates: Array
    forward_rates: Array
    implied_vols: Array
    strike_grid: Array
    fx_spots: Array
    currency_mapping: IndexMapping
    asset_mapping: IndexMapping
    fx_pair_mapping: IndexMapping
    metadata: dict[str, Any] = None

    def __post_init__(self) -> None:
        """Validate grid dimensions and consistency."""
        n_times = self.time_grid.shape[0]
        n_currencies = len(self.currency_mapping)
        n_assets = len(self.asset_mapping)
        n_strikes = self.strike_grid.shape[0]
        n_fx_pairs = len(self.fx_pair_mapping)

        # Validate time grid (skip validation if inside JAX transformation)
        try:
            if n_times == 0:
                raise ValueError("Time grid cannot be empty")
            if jnp.any(self.time_grid[1:] <= self.time_grid[:-1]):
                raise ValueError("Time grid must be strictly increasing")

            # Validate shapes
            expected_df_shape = (n_currencies, n_times)
            if self.discount_factors.shape != expected_df_shape:
                raise ValueError(
                    f"discount_factors shape {self.discount_factors.shape} != {expected_df_shape}"
                )

            if self.zero_rates.shape != expected_df_shape:
                raise ValueError(f"zero_rates shape {self.zero_rates.shape} != {expected_df_shape}")

            if self.forward_rates.shape != expected_df_shape:
                raise ValueError(f"forward_rates shape {self.forward_rates.shape} != {expected_df_shape}")

            expected_vol_shape = (n_assets, n_times, n_strikes)
            if self.implied_vols.shape != expected_vol_shape:
                raise ValueError(f"implied_vols shape {self.implied_vols.shape} != {expected_vol_shape}")

            expected_fx_shape = (n_fx_pairs,)
            if self.fx_spots.shape != expected_fx_shape:
                raise ValueError(f"fx_spots shape {self.fx_spots.shape} != {expected_fx_shape}")

            # Validate values
            if jnp.any(self.discount_factors <= 0) or jnp.any(self.discount_factors > 1):
                raise ValueError("Discount factors must be in (0, 1]")

            if jnp.any(self.fx_spots <= 0):
                raise ValueError("FX spots must be positive")
        except jax.errors.TracerBoolConversionError:
            # Skip validation inside JAX transformations
            pass

        # Initialize metadata
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

    @property
    def n_times(self) -> int:
        """Number of time points in the grid."""
        return int(self.time_grid.shape[0])

    @property
    def n_currencies(self) -> int:
        """Number of currencies."""
        return len(self.currency_mapping)

    @property
    def n_assets(self) -> int:
        """Number of assets."""
        return len(self.asset_mapping)

    @property
    def n_strikes(self) -> int:
        """Number of strike points in volatility grid."""
        return int(self.strike_grid.shape[0])

    @property
    def n_fx_pairs(self) -> int:
        """Number of FX pairs."""
        return len(self.fx_pair_mapping)

    @property
    def max_time(self) -> float:
        """Maximum time in the grid."""
        return float(self.time_grid[-1])

    # JAX Pytree implementation
    def tree_flatten(self) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Flatten for JAX pytree operations."""
        children = (
            self.time_grid,
            self.discount_factors,
            self.zero_rates,
            self.forward_rates,
            self.implied_vols,
            self.strike_grid,
            self.fx_spots,
        )
        aux_data = {
            "currency_mapping": self.currency_mapping,
            "asset_mapping": self.asset_mapping,
            "fx_pair_mapping": self.fx_pair_mapping,
            "metadata": self.metadata,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: dict[str, Any], children: tuple[Any, ...]) -> MarketDataGrid:
        """Reconstruct from flattened pytree."""
        return cls(
            time_grid=children[0],
            discount_factors=children[1],
            zero_rates=children[2],
            forward_rates=children[3],
            implied_vols=children[4],
            strike_grid=children[5],
            fx_spots=children[6],
            **aux_data,
        )

    def get_discount_factor(self, currency_idx: int, time_idx: int) -> float:
        """Get discount factor for a currency at a specific time index.

        Parameters
        ----------
        currency_idx : int
            Currency index
        time_idx : int
            Time grid index

        Returns
        -------
        float
            Discount factor

        Examples
        --------
        >>> usd_idx = grid.currency_mapping.encode("USD")
        >>> df = grid.get_discount_factor(usd_idx, 100)
        """
        return float(self.discount_factors[currency_idx, time_idx])

    def get_discount_factors_batch(self, currency_idx: Array, time_idx: Array) -> Array:
        """Get discount factors for multiple currencies and times.

        Parameters
        ----------
        currency_idx : Array
            Currency indices [n_queries]
        time_idx : Array
            Time grid indices [n_queries]

        Returns
        -------
        Array
            Discount factors [n_queries]

        Examples
        --------
        >>> # Get DFs for 10K trades (vectorized!)
        >>> currency_indices = portfolio.currency_idx  # [10K]
        >>> time_indices = jnp.searchsorted(grid.time_grid, maturities)  # [10K]
        >>> dfs = grid.get_discount_factors_batch(currency_indices, time_indices)
        """
        return self.discount_factors[currency_idx, time_idx]

    def get_implied_vol(self, asset_idx: int, time_idx: int, strike_idx: int) -> float:
        """Get implied volatility for an asset at specific time and strike.

        Parameters
        ----------
        asset_idx : int
            Asset index
        time_idx : int
            Time grid index
        strike_idx : int
            Strike grid index

        Returns
        -------
        float
            Implied volatility
        """
        return float(self.implied_vols[asset_idx, time_idx, strike_idx])

    def get_implied_vols_batch(
        self,
        asset_idx: Array,
        maturities: Array,
        strikes: Array,
    ) -> Array:
        """Interpolate implied volatilities for multiple asset/time/strike queries.

        Parameters
        ----------
        asset_idx : Array
            Asset indices [n_queries]
        maturities : Array
            Times in years [n_queries]
        strikes : Array
            Strike prices [n_queries]

        Returns
        -------
        Array
            Interpolated implied volatilities [n_queries]
        """

        asset_idx = jnp.asarray(asset_idx, dtype=jnp.int32)
        maturities = jnp.asarray(maturities, dtype=self.time_grid.dtype)
        strikes = jnp.asarray(strikes, dtype=self.strike_grid.dtype)

        min_time = self.time_grid[0]
        max_time = self.time_grid[-1]
        maturities = jnp.clip(maturities, min_time, max_time)

        min_strike = self.strike_grid[0]
        max_strike = self.strike_grid[-1]
        strikes = jnp.clip(strikes, min_strike, max_strike)

        eps_time = jnp.array(1e-12, dtype=self.time_grid.dtype)
        eps_strike = jnp.array(1e-12, dtype=self.strike_grid.dtype)

        if self.n_times == 1:
            time_left = time_right = jnp.zeros_like(asset_idx, dtype=jnp.int32)
            time_weight = jnp.zeros_like(maturities)
        else:
            time_right = jnp.searchsorted(self.time_grid, maturities, side="right")
            time_right = jnp.clip(time_right, 1, self.n_times - 1)
            time_left = time_right - 1
            t_left = self.time_grid[time_left]
            t_right = self.time_grid[time_right]
            denom_t = jnp.where(t_right > t_left, t_right - t_left, eps_time)
            time_weight = jnp.where(
                t_right > t_left,
                (maturities - t_left) / denom_t,
                jnp.zeros_like(maturities),
            )

        if self.n_strikes == 1:
            strike_left = strike_right = jnp.zeros_like(asset_idx, dtype=jnp.int32)
            strike_weight = jnp.zeros_like(strikes)
        else:
            strike_right = jnp.searchsorted(self.strike_grid, strikes, side="right")
            strike_right = jnp.clip(strike_right, 1, self.n_strikes - 1)
            strike_left = strike_right - 1
            k_left = self.strike_grid[strike_left]
            k_right = self.strike_grid[strike_right]
            denom_k = jnp.where(k_right > k_left, k_right - k_left, eps_strike)
            strike_weight = jnp.where(
                k_right > k_left,
                (strikes - k_left) / denom_k,
                jnp.zeros_like(strikes),
            )

        surfaces = self.implied_vols[asset_idx, :, :]

        def _bilinear(surface, t_l, t_r, k_l, k_r, t_w, k_w):
            vol_ll = surface[t_l, k_l]
            vol_lr = surface[t_l, k_r]
            vol_rl = surface[t_r, k_l]
            vol_rr = surface[t_r, k_r]

            vol_top = vol_ll + (vol_lr - vol_ll) * k_w
            vol_bottom = vol_rl + (vol_rr - vol_rl) * k_w
            return vol_top + (vol_bottom - vol_top) * t_w

        return jax.vmap(_bilinear)(
            surfaces, time_left, time_right, strike_left, strike_right, time_weight, strike_weight
        )

    def interpolate_strike(self, asset_idx: int, time_idx: int, strike: float) -> float:
        """Interpolate implied vol for a specific strike value.

        Uses linear interpolation on the pre-computed strike grid.

        Parameters
        ----------
        asset_idx : int
            Asset index
        time_idx : int
            Time grid index
        strike : float
            Strike price (will be interpolated)

        Returns
        -------
        float
            Interpolated implied volatility
        """
        vols_at_time = self.implied_vols[asset_idx, time_idx, :]
        return float(jnp.interp(strike, self.strike_grid, vols_at_time))

    def find_time_index(self, time: float) -> int:
        """Find the nearest time grid index for a given time.

        Parameters
        ----------
        time : float
            Time in years

        Returns
        -------
        int
            Nearest time grid index

        Examples
        --------
        >>> idx = grid.find_time_index(2.5)  # Find index for 2.5 years
        """
        return int(jnp.searchsorted(self.time_grid, time))

    def find_time_indices_batch(self, times: Array) -> Array:
        """Find time grid indices for multiple times.

        Parameters
        ----------
        times : Array
            Times in years [n_queries]

        Returns
        -------
        Array
            Time grid indices [n_queries]

        Examples
        --------
        >>> maturities = portfolio.trade_arrays.maturities  # [10K]
        >>> time_indices = grid.find_time_indices_batch(maturities)
        """
        return jnp.searchsorted(self.time_grid, times)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"MarketDataGrid(\n"
            f"  n_times={self.n_times},\n"
            f"  n_currencies={self.n_currencies},\n"
            f"  n_assets={self.n_assets},\n"
            f"  n_strikes={self.n_strikes},\n"
            f"  time_range=[{float(self.time_grid[0]):.2f}, {float(self.time_grid[-1]):.2f}],\n"
            f"  memory_size={(self.discount_factors.nbytes + self.implied_vols.nbytes) / 1024:.1f} KB\n"
            f")"
        )
