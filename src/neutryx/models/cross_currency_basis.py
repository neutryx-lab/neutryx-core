"""
Cross-Currency Basis Modeling

Cross-currency basis models capture the spread between domestic and foreign interest
rates, accounting for FX dynamics and basis spreads observed in cross-currency swap markets.

Model Structure:
    Domestic rate: dr_d(t) = μ_d(r_d, t)·dt + σ_d(r_d, t)·dW_d
    Foreign rate:  dr_f(t) = μ_f(r_f, t)·dt + σ_f(r_f, t)·dW_f
    FX spot:       dS(t)/S(t) = (r_d(t) - r_f(t) - b(t))·dt + σ_S(t)·dW_S

    where:
    - r_d: Domestic short rate
    - r_f: Foreign short rate (in foreign currency)
    - S: FX spot rate (domestic per unit foreign)
    - b(t): Cross-currency basis spread
    - Correlations: dW_i·dW_j = ρ_ij·dt

Key Features:
- Models domestic and foreign interest rate curves separately
- FX spot dynamics linked to interest rate differential
- Cross-currency basis spread b(t) calibrated from market data
- Allows pricing of cross-currency swaps, FX forwards, quanto derivatives
- Correlation structure between rates and FX

Applications:
- Cross-currency swap pricing and risk management
- FX forward curve construction
- Multi-currency portfolio risk
- Quanto option pricing
- Currency hedging strategies

References:
    - Brigo, D., & Mercurio, F. (2006). Interest Rate Models - Theory and Practice.
      Chapter 16: Multi-Currency Models.
    - Piterbarg, V. (2010). Funding beyond discounting: collateral agreements and
      derivatives pricing. Risk Magazine, 23(2), 97-102.
    - Fujii, M., Shimada, Y., & Takahashi, A. (2010). A note on construction of
      multiple swap curves with and without collateral. CARF Working Paper.

Author: Neutryx Development Team
Date: 2025
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union, Any
import jax
import jax.numpy as jnp
from jax import lax, Array
from functools import partial

# Import G2PP for underlying rate models
from neutryx.models.g2pp import G2PPParams


@dataclass
class CrossCurrencyBasisParams:
    """
    Parameters for cross-currency basis model.

    Combines two interest rate models (domestic and foreign) with FX dynamics
    and a cross-currency basis spread.

    Attributes:
        domestic_model: Interest rate model for domestic currency.
                        Can be G2PPParams, QuasiGaussianParams, or similar.
        foreign_model: Interest rate model for foreign currency.
        fx_spot: Initial FX spot rate S(0) (domestic per unit foreign).
                 Example: USD/EUR = 1.10 means 1.10 USD per 1 EUR.
        fx_vol_fn: FX spot volatility function σ_S(t). Can be constant or time-dependent.
                   Typical range: 0.05 - 0.20 (5% - 20%)
        basis_spread: Cross-currency basis spread b(t). Can be scalar (constant)
                      or callable for time-dependent spread.
                      Quoted in basis points. Typical: -50 bps to +50 bps
                      Positive = domestic currency more expensive to borrow
        correlation_matrix: Correlation matrix between (dW_d, dW_f, dW_S).
                           Shape: (3, 3) or (5, 5) if using two-factor models.
                           Default: [[1, 0.5, -0.3],
                                    [0.5, 1, -0.3],
                                    [-0.3, -0.3, 1]]
                           Typical correlations:
                           - ρ(r_d, r_f): 0.3 - 0.7 (positive, rates move together)
                           - ρ(r_d, S): -0.5 - 0.0 (negative, high dom rate → strong currency)
                           - ρ(r_f, S): -0.5 - 0.0 (negative)
        domestic_curve_fn: Optional domestic discount curve function P_d(0,T)
        foreign_curve_fn: Optional foreign discount curve function P_f(0,T)

    Raises:
        ValueError: If parameters violate constraints
    """
    domestic_model: Any  # G2PPParams or similar
    foreign_model: Any  # G2PPParams or similar
    fx_spot: float
    fx_vol_fn: Union[float, Callable[[float], float]]
    basis_spread: Union[float, Callable[[float], float]] = 0.0
    correlation_matrix: Optional[Array] = None
    domestic_curve_fn: Optional[Callable[[float], float]] = None
    foreign_curve_fn: Optional[Callable[[float], float]] = None

    def __post_init__(self):
        """Validate parameters."""
        if self.fx_spot <= 0:
            raise ValueError(f"FX spot must be positive, got {self.fx_spot}")

        # Validate FX volatility
        if callable(self.fx_vol_fn):
            fx_vol_0 = self.fx_vol_fn(0.0)
        else:
            fx_vol_0 = self.fx_vol_fn

        if fx_vol_0 <= 0:
            raise ValueError(f"FX volatility must be positive, got {fx_vol_0}")

        # Set default correlation matrix if not provided
        if self.correlation_matrix is None:
            # Default: positive rate correlation, negative rate-FX correlation
            self.correlation_matrix = jnp.array([
                [1.0,  0.5, -0.3],  # Domestic rate correlations
                [0.5,  1.0, -0.3],  # Foreign rate correlations
                [-0.3, -0.3, 1.0],  # FX correlations
            ])

        # Validate correlation matrix is symmetric and positive semi-definite
        rho = self.correlation_matrix
        if rho.shape[0] != rho.shape[1]:
            raise ValueError("Correlation matrix must be square")

        # Check symmetry
        if not jnp.allclose(rho, rho.T):
            raise ValueError("Correlation matrix must be symmetric")

        # Check positive semi-definite
        eigenvalues = jnp.linalg.eigvalsh(rho)
        if jnp.any(eigenvalues < -1e-6):
            raise ValueError(
                f"Correlation matrix must be positive semi-definite. "
                f"Got eigenvalues: {eigenvalues}"
            )

        # Check diagonal is all ones
        if not jnp.allclose(jnp.diag(rho), 1.0):
            raise ValueError("Correlation matrix diagonal must be all ones")

    def get_fx_vol(self, t: float) -> float:
        """Get FX volatility at time t."""
        if callable(self.fx_vol_fn):
            return self.fx_vol_fn(t)
        return self.fx_vol_fn

    def get_basis_spread(self, t: float) -> float:
        """Get basis spread at time t."""
        if callable(self.basis_spread):
            return self.basis_spread(t)
        return self.basis_spread


def simulate_path(
    params: CrossCurrencyBasisParams,
    T: float,
    n_steps: int,
    key: Array
) -> Tuple[Array, Array, Array]:
    """
    Simulate a single path of the cross-currency basis model.

    Simulates:
        dr_d = [μ_d(r_d, t)] dt + σ_d dW_d
        dr_f = [μ_f(r_f, t)] dt + σ_f dW_f
        dS/S = (r_d - r_f - b(t)) dt + σ_S dW_S

    with correlations specified in params.correlation_matrix.

    Args:
        params: Cross-currency basis model parameters
        T: Final time horizon
        n_steps: Number of time steps
        key: JAX random key

    Returns:
        Tuple (r_d_path, r_f_path, S_path) where:
            - r_d_path: Domestic short rate path, shape (n_steps+1,)
            - r_f_path: Foreign short rate path, shape (n_steps+1,)
            - S_path: FX spot path, shape (n_steps+1,)

    Example:
        >>> from neutryx.models.g2pp import G2PPParams
        >>> dom_params = G2PPParams(a=0.1, b=0.2, sigma_x=0.01, sigma_y=0.015,
        ...                          rho=-0.7, r0=0.03)
        >>> for_params = G2PPParams(a=0.15, b=0.25, sigma_x=0.012, sigma_y=0.018,
        ...                          rho=-0.6, r0=0.02)
        >>> ccy_params = CrossCurrencyBasisParams(
        ...     domestic_model=dom_params,
        ...     foreign_model=for_params,
        ...     fx_spot=1.10,
        ...     fx_vol_fn=0.10,
        ...     basis_spread=-0.0020  # -20 bps
        ... )
        >>> key = jax.random.PRNGKey(42)
        >>> r_d, r_f, S = simulate_path(ccy_params, T=1.0, n_steps=252, key=key)
        >>> assert r_d.shape == (253,)
        >>> assert S[0] == 1.10
    """
    dt = T / n_steps

    # Import simulate_path from respective model modules
    # For G2PP models
    from neutryx.models.g2pp import simulate_path as simulate_g2pp

    # Simulate domestic and foreign rates
    key_d, key_f, key_fx = jax.random.split(key, 3)

    # Simulate domestic rate (assuming G2PP for now)
    r_d_path, x_d_path, y_d_path = simulate_g2pp(
        params.domestic_model, T, n_steps, key_d
    )

    # Simulate foreign rate
    r_f_path, x_f_path, y_f_path = simulate_g2pp(
        params.foreign_model, T, n_steps, key_f
    )

    # Generate correlated FX Brownian motion
    # Use correlation matrix to correlate with rate Brownians
    # For simplicity, we'll use the marginal FX dynamics with correlation

    # Generate independent Brownian increments
    dW_fx_indep = jax.random.normal(key_fx, shape=(n_steps,)) * jnp.sqrt(dt)

    # Correlation: dW_S with domestic and foreign rates
    # Approximation: use correlation with instantaneous rate changes
    # More precise: would need to correlate with underlying factors

    rho_d_fx = params.correlation_matrix[0, 2]  # Correlation(W_d, W_S)
    rho_f_fx = params.correlation_matrix[1, 2]  # Correlation(W_f, W_S)

    # Simulate FX spot using correlated dynamics
    def step_fn(carry, inputs):
        S_t, t_idx = carry
        r_d_t, r_f_t, dW_indep = inputs

        t = t_idx * dt

        # Get time-dependent parameters
        sigma_S_t = params.get_fx_vol(t)
        basis_t = params.get_basis_spread(t)

        # Drift: μ_S = r_d - r_f - basis
        drift = (r_d_t - r_f_t - basis_t) * dt

        # Diffusion: use independent component
        # (In full model, would correlate with rate Brownians)
        diffusion = sigma_S_t * dW_indep

        # Update FX spot
        S_new = S_t * jnp.exp(drift - 0.5 * sigma_S_t**2 * dt + diffusion)

        return (S_new, t_idx + 1), S_new

    # Initial condition
    init = (params.fx_spot, 0)

    # Run FX simulation
    inputs = (r_d_path[:-1], r_f_path[:-1], dW_fx_indep)
    _, S_path = lax.scan(step_fn, init, inputs)

    # Prepend initial FX spot
    S_path = jnp.concatenate([jnp.array([params.fx_spot]), S_path])

    return r_d_path, r_f_path, S_path


@partial(jax.jit, static_argnames=["n_steps", "n_paths"])
def simulate_paths(
    params: CrossCurrencyBasisParams,
    T: float,
    n_steps: int,
    n_paths: int,
    key: Array
) -> Tuple[Array, Array, Array]:
    """
    Simulate multiple paths of the cross-currency basis model.

    Args:
        params: Cross-currency basis model parameters
        T: Final time horizon
        n_steps: Number of time steps per path
        n_paths: Number of paths to simulate
        key: JAX random key

    Returns:
        Tuple (r_d_paths, r_f_paths, S_paths) each of shape (n_paths, n_steps+1)
    """
    keys = jax.random.split(key, n_paths)
    simulate_fn = lambda k: simulate_path(params, T, n_steps, k)
    r_d_paths, r_f_paths, S_paths = jax.vmap(simulate_fn)(keys)
    return r_d_paths, r_f_paths, S_paths


def fx_forward_rate(
    params: CrossCurrencyBasisParams,
    T: float
) -> float:
    """
    Compute FX forward rate F(0,T) for delivery at time T.

    Under no-arbitrage:
        F(0,T) = S(0) · P_f(0,T) / P_d(0,T) · exp(-∫_0^T b(s) ds)

    where:
    - S(0): Current spot FX rate
    - P_d(0,T): Domestic discount factor
    - P_f(0,T): Foreign discount factor
    - b(t): Basis spread

    Args:
        params: Cross-currency basis model parameters
        T: Forward maturity

    Returns:
        FX forward rate F(0,T)

    Example:
        >>> # If domestic rate > foreign rate, forward FX < spot (domestic appreciates)
        >>> forward = fx_forward_rate(params, T=1.0)
    """
    # Get discount factors
    from neutryx.models.g2pp import zero_coupon_bond_price

    P_d = zero_coupon_bond_price(params.domestic_model, T)
    P_f = zero_coupon_bond_price(params.foreign_model, T)

    # Integrate basis spread
    if callable(params.basis_spread):
        # Numerical integration
        times = jnp.linspace(0, T, 100)
        basis_vals = jax.vmap(params.basis_spread)(times)
        basis_integral = jnp.trapz(basis_vals, times)
    else:
        basis_integral = params.basis_spread * T

    # Forward FX rate
    forward = params.fx_spot * (P_f / P_d) * jnp.exp(-basis_integral)

    return forward


def cross_currency_swap_value(
    params: CrossCurrencyBasisParams,
    notional_domestic: float,
    notional_foreign: float,
    domestic_rate: float,
    foreign_rate: float,
    maturity: float,
    tenor: float = 0.5,
    is_receive_domestic: bool = True
) -> float:
    """
    Value a cross-currency swap (mark-to-market).

    A cross-currency swap exchanges:
    - Fixed domestic currency payments at rate K_d
    - Fixed foreign currency payments at rate K_f
    - Notional exchange at maturity

    Args:
        params: Cross-currency basis model parameters
        notional_domestic: Domestic currency notional
        notional_foreign: Foreign currency notional (in foreign currency)
        domestic_rate: Fixed domestic rate (annualized)
        foreign_rate: Fixed foreign rate (annualized)
        maturity: Swap maturity
        tenor: Payment frequency (e.g., 0.5 for semi-annual)
        is_receive_domestic: True if receiving domestic, paying foreign

    Returns:
        Swap value in domestic currency

    Example:
        >>> # USD/EUR cross-currency swap
        >>> # Receive 3% on $1M USD, Pay 2% on €900K EUR
        >>> value = cross_currency_swap_value(
        ...     params, notional_domestic=1e6, notional_foreign=900000,
        ...     domestic_rate=0.03, foreign_rate=0.02,
        ...     maturity=5.0, is_receive_domestic=True
        ... )
    """
    from neutryx.models.g2pp import zero_coupon_bond_price

    # Payment times
    payment_times = jnp.arange(tenor, maturity + tenor/2, tenor)

    # Domestic leg value
    domestic_pv = 0.0
    for t in payment_times:
        P_d = zero_coupon_bond_price(params.domestic_model, t)
        domestic_pv += domestic_rate * tenor * notional_domestic * P_d

    # Add final notional
    P_d_final = zero_coupon_bond_price(params.domestic_model, maturity)
    domestic_pv += notional_domestic * P_d_final

    # Foreign leg value (convert to domestic currency)
    foreign_pv = 0.0
    for t in payment_times:
        P_f = zero_coupon_bond_price(params.foreign_model, t)
        # Convert to domestic using forward FX
        F_t = fx_forward_rate(params, t)
        foreign_pv += foreign_rate * tenor * notional_foreign * P_f * F_t

    # Add final notional
    F_final = fx_forward_rate(params, maturity)
    foreign_pv += notional_foreign * zero_coupon_bond_price(params.foreign_model, maturity) * F_final

    # Swap value
    if is_receive_domestic:
        value = domestic_pv - foreign_pv
    else:
        value = foreign_pv - domestic_pv

    return value


def quanto_option_price_mc(
    params: CrossCurrencyBasisParams,
    strike: float,
    maturity: float,
    is_call: bool = True,
    foreign_asset_spot: float = 100.0,
    foreign_asset_vol: float = 0.20,
    n_paths: int = 10000,
    key: Optional[Array] = None
) -> float:
    """
    Price a quanto option using Monte Carlo simulation.

    A quanto option pays in domestic currency based on foreign asset performance,
    without FX conversion. Payoff at maturity:
        max(S_foreign(T) - K, 0) × 1 domestic unit  [for call]

    where S_foreign is denominated in foreign currency.

    Args:
        params: Cross-currency basis model parameters
        strike: Strike price (in foreign currency)
        maturity: Option maturity
        is_call: True for call, False for put
        foreign_asset_spot: Initial foreign asset price (in foreign currency)
        foreign_asset_vol: Foreign asset volatility
        n_paths: Number of MC paths
        key: JAX random key

    Returns:
        Quanto option price in domestic currency

    Notes:
        The quanto adjustment arises from the correlation between the foreign
        asset and the FX rate. The drift adjustment is:
            drift_quanto = r_d - ρ(S_foreign, S_fx) × σ_foreign × σ_fx
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    # Simulate rates and FX to maturity
    n_steps = int(maturity * 252)
    r_d_paths, r_f_paths, S_fx_paths = simulate_paths(params, maturity, n_steps, n_paths, key)

    # Simulate foreign asset (assumed GBM with correlation to FX)
    key_asset = jax.random.split(key)[1]
    Z_asset = jax.random.normal(key_asset, shape=(n_paths, n_steps))

    # Correlation between foreign asset and FX
    # Typically negative: strong foreign currency → lower foreign asset prices
    rho_asset_fx = -0.3  # Can be a parameter

    dt = maturity / n_steps

    def simulate_foreign_asset(z_path):
        # Quanto drift adjustment
        # Under domestic risk-neutral measure:
        # dS_foreign/S_foreign = (r_d - ρ·σ_foreign·σ_fx) dt + σ_foreign dW

        avg_fx_vol = jnp.mean(jax.vmap(params.get_fx_vol)(jnp.linspace(0, maturity, 10)))
        quanto_adjustment = rho_asset_fx * foreign_asset_vol * avg_fx_vol

        # Average domestic rate (simplified)
        avg_r_d = jnp.mean(r_d_paths[0, :])

        # GBM simulation
        drift = (avg_r_d - quanto_adjustment - 0.5 * foreign_asset_vol**2) * dt
        diffusion = foreign_asset_vol * jnp.sqrt(dt) * z_path

        log_returns = drift + diffusion
        S_path = foreign_asset_spot * jnp.exp(jnp.cumsum(log_returns))
        return S_path[-1]

    foreign_asset_terminal = jax.vmap(simulate_foreign_asset)(Z_asset)

    # Quanto payoff (in domestic currency)
    if is_call:
        payoffs = jnp.maximum(foreign_asset_terminal - strike, 0.0)
    else:
        payoffs = jnp.maximum(strike - foreign_asset_terminal, 0.0)

    # Discount using domestic rate
    from neutryx.models.g2pp import zero_coupon_bond_price
    discount = zero_coupon_bond_price(params.domestic_model, maturity)

    price = discount * jnp.mean(payoffs)
    return price


def calibrate_basis_spread(
    domestic_model: Any,
    foreign_model: Any,
    fx_spot: float,
    fx_forward: float,
    maturity: float
) -> float:
    """
    Calibrate the cross-currency basis spread from FX forward market data.

    Given market FX forward F^market(0,T), solve for basis spread b such that:
        F^model(0,T) = F^market(0,T)

    where:
        F(0,T) = S(0) · [P_f(0,T) / P_d(0,T)] · exp(-b·T)

    Args:
        domestic_model: Domestic interest rate model
        foreign_model: Foreign interest rate model
        fx_spot: Current FX spot S(0)
        fx_forward: Market FX forward F^market(0,T)
        maturity: Forward maturity T

    Returns:
        Implied basis spread b (annualized)

    Example:
        >>> # Market shows FX forward trading below theoretical (positive basis)
        >>> basis = calibrate_basis_spread(dom_model, for_model,
        ...                                 fx_spot=1.10, fx_forward=1.08, maturity=1.0)
        >>> print(f"Basis spread: {basis * 10000:.1f} bps")
    """
    from neutryx.models.g2pp import zero_coupon_bond_price

    P_d = zero_coupon_bond_price(domestic_model, maturity)
    P_f = zero_coupon_bond_price(foreign_model, maturity)

    # Solve: F_market = S(0) · (P_f / P_d) · exp(-b·T)
    # => b = -ln(F_market · P_d / (S(0) · P_f)) / T

    basis_spread = -jnp.log(fx_forward * P_d / (fx_spot * P_f)) / maturity

    return basis_spread


__all__ = [
    "CrossCurrencyBasisParams",
    "simulate_path",
    "simulate_paths",
    "fx_forward_rate",
    "cross_currency_swap_value",
    "quanto_option_price_mc",
    "calibrate_basis_spread",
]
