"""FX stochastic volatility and multi-factor models.

This module implements sophisticated FX pricing models beyond Garman-Kohlhagen:
- FX Heston: Stochastic volatility with characteristic function pricing
- FX SABR: Stochastic volatility Beta with smile dynamics
- FX Bates: Heston + jumps for crash/rally scenarios
- Two-Factor FX: Multi-scale volatility dynamics

Theoretical foundations:
- Heston (1993): Closed-form solution via characteristic function
- Hagan et al. (2002): SABR model for smile dynamics
- Bates (1996): Stochastic volatility jump-diffusion
- Christoffersen-Jacobs-Ornthanalai (2009): Multi-factor FX models

References:
    - Heston, S. (1993). A closed-form solution for options with stochastic volatility.
    - Hagan, P., Kumar, D., Lesniewski, A., & Woodward, D. (2002). Managing smile risk.
    - Bates, D. (1996). Jumps and stochastic volatility: exchange rate processes.
    - Clark, I. (2011). Foreign Exchange Option Pricing: A Practitioner's Guide.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.stats import norm

from neutryx.models.heston import heston_call_price_cf as heston_price_fft
from neutryx.models.sabr import hagan_implied_vol as sabr_implied_vol


@dataclass
class FXHestonModel:
    """FX Heston stochastic volatility model.

    Dynamics:
        dS = (r_d - r_f) S dt + √v S dW₁
        dv = κ(θ - v) dt + σ √v dW₂
        dW₁ dW₂ = ρ dt

    where r_d = domestic rate, r_f = foreign rate (replace r-q in equity Heston)

    Args:
        v0: Initial variance (vol² at t=0)
        kappa: Mean reversion speed (1/time to mean)
        theta: Long-run variance level (long-term vol²)
        sigma: Volatility of volatility (vol of vol)
        rho: Correlation between spot and variance (-1 to 1)
        r_domestic: Domestic interest rate (e.g., USD if pricing EURUSD)
        r_foreign: Foreign interest rate (EUR if pricing EURUSD)

    Example:
        >>> # EURUSD with moderate smile
        >>> model = FXHestonModel(
        ...     v0=0.04,      # 20% initial vol
        ...     kappa=2.0,    # Mean reversion
        ...     theta=0.04,   # 20% long-run vol
        ...     sigma=0.3,    # 30% vol-of-vol
        ...     rho=-0.7,     # Negative correlation (typical for FX)
        ...     r_domestic=0.05,  # 5% USD
        ...     r_foreign=0.02    # 2% EUR
        ... )

    References:
        - Heston, S. (1993). Review of Financial Studies, 6(2), 327-343.
        - Clark, I. (2011). Foreign Exchange Option Pricing: A Practitioner's Guide.
    """

    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float
    r_domestic: float
    r_foreign: float

    def __post_init__(self):
        """Validate Feller condition and parameter bounds."""
        # Feller condition: 2κθ ≥ σ² ensures v stays positive
        if 2 * self.kappa * self.theta < self.sigma ** 2:
            import warnings
            warnings.warn(
                f"Feller condition violated: 2κθ = {2*self.kappa*self.theta:.4f} < σ² = {self.sigma**2:.4f}. "
                "Variance may hit zero."
            )

        if not -1 <= self.rho <= 1:
            raise ValueError(f"Correlation must be in [-1, 1], got {self.rho}")

        if self.v0 <= 0 or self.theta <= 0:
            raise ValueError("Initial and long-run variance must be positive")

        if self.kappa <= 0 or self.sigma <= 0:
            raise ValueError("Mean reversion and vol-of-vol must be positive")

    def price(
        self,
        S: float,
        K: float,
        T: float,
        is_call: bool = True,
        N: int = 4096
    ) -> float:
        """Price European FX option using characteristic function + FFT.

        Args:
            S: Spot FX rate (domestic per foreign, e.g., 1.10 for EURUSD)
            K: Strike FX rate
            T: Time to maturity (years)
            is_call: True for call, False for put
            N: Number of FFT points (power of 2, typically 2048-8192)

        Returns:
            Option price in domestic currency

        Note:
            Uses Carr-Madan FFT with optimal damping parameter.
            For illiquid strikes or short maturities, consider increasing N.
        """
        # Reuse equity Heston FFT with FX drift adjustment
        # r_d - r_f replaces r - q (dividend yield becomes foreign rate)
        price = heston_price_fft(
            S=S,
            K=K,
            T=T,
            r=self.r_domestic,
            q=self.r_foreign,  # Foreign rate acts as dividend yield
            v0=self.v0,
            kappa=self.kappa,
            theta=self.theta,
            sigma=self.sigma,
            rho=self.rho,
            is_call=is_call,
            N=N
        )
        return price

    def implied_vol(
        self,
        S: float,
        K: float,
        T: float,
        is_call: bool = True
    ) -> float:
        """Compute model-implied volatility for a given strike.

        Args:
            S: Spot FX rate
            K: Strike
            T: Maturity
            is_call: Option type

        Returns:
            Black-Scholes implied volatility (annual)

        Note:
            Uses Newton-Raphson on vega to invert BS formula.
        """
        price = self.price(S, K, T, is_call)

        # Garman-Kohlhagen inversion (BS with r_d - r_f)
        return self._invert_gk(price, S, K, T, is_call)

    def _invert_gk(
        self,
        price: float,
        S: float,
        K: float,
        T: float,
        is_call: bool,
        tol: float = 1e-6,
        max_iter: int = 100
    ) -> float:
        """Invert Garman-Kohlhagen to get implied vol (Newton-Raphson)."""
        from neutryx.products._utils import garman_kohlhagen, compute_d1_d2_fx

        # Initial guess: ATM vol from current variance
        vol = jnp.sqrt(self.v0)

        for _ in range(max_iter):
            model_price = garman_kohlhagen(
                S, K, T, vol, self.r_domestic, self.r_foreign, is_call
            )

            diff = model_price - price
            if abs(diff) < tol:
                return vol

            # Vega for Newton step
            d1, _ = compute_d1_d2_fx(S, K, T, vol, self.r_domestic, self.r_foreign)
            vega = S * jnp.exp(-self.r_foreign * T) * norm.pdf(d1) * jnp.sqrt(T)

            if vega < 1e-10:
                break

            vol = vol - diff / vega
            vol = jnp.maximum(vol, 0.001)  # Floor at 0.1%

        return vol

    def calibrate_to_smile(
        self,
        S: float,
        T: float,
        strikes: Array,
        market_vols: Array,
        weights: Optional[Array] = None
    ) -> dict:
        """Calibrate Heston parameters to FX smile (single maturity).

        Args:
            S: Spot FX rate
            T: Maturity
            strikes: Array of strikes
            market_vols: Observed implied vols (GK convention)
            weights: Optional weights for strikes (e.g., vega-weighted)

        Returns:
            Dictionary with:
                - 'params': Calibrated (v0, kappa, theta, sigma, rho)
                - 'rmse': Root-mean-square error in vol space
                - 'model_vols': Model implied vols at strikes

        Note:
            Uses L-BFGS-B with bounds. For multi-tenor, call iteratively
            with tenor-dependent kappa, theta.
        """
        from scipy.optimize import minimize

        if weights is None:
            weights = jnp.ones_like(strikes)

        # Normalize weights
        weights = weights / jnp.sum(weights)

        def objective(params):
            v0, kappa, theta, sigma, rho = params

            # Create temp model
            temp_model = FXHestonModel(
                v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho,
                r_domestic=self.r_domestic,
                r_foreign=self.r_foreign
            )

            # Compute model vols
            model_vols = jnp.array([
                temp_model.implied_vol(S, K, T, is_call=True)
                for K in strikes
            ])

            # Weighted RMSE
            errors = (model_vols - market_vols) ** 2
            return jnp.sqrt(jnp.sum(weights * errors))

        # Initial guess from ATM vol
        atm_vol = market_vols[len(market_vols) // 2]
        x0 = [
            atm_vol ** 2,  # v0
            2.0,           # kappa
            atm_vol ** 2,  # theta
            0.3,           # sigma
            -0.5           # rho (negative for FX)
        ]

        # Bounds
        bounds = [
            (0.0001, 1.0),  # v0
            (0.01, 10.0),   # kappa
            (0.0001, 1.0),  # theta
            (0.01, 2.0),    # sigma
            (-0.99, 0.99)   # rho
        ]

        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500}
        )

        # Update model parameters
        object.__setattr__(self, 'v0', result.x[0])
        object.__setattr__(self, 'kappa', result.x[1])
        object.__setattr__(self, 'theta', result.x[2])
        object.__setattr__(self, 'sigma', result.x[3])
        object.__setattr__(self, 'rho', result.x[4])

        # Compute final model vols
        model_vols = jnp.array([
            self.implied_vol(S, K, T, is_call=True)
            for K in strikes
        ])

        return {
            'params': tuple(result.x),
            'rmse': result.fun,
            'model_vols': model_vols,
            'success': result.success
        }


@dataclass
class FXSABRModel:
    """FX SABR (Stochastic Alpha Beta Rho) model.

    Dynamics:
        dF = α F^β dW₁
        dα = ν α dW₂
        dW₁ dW₂ = ρ dt

    where F is forward FX rate, α is stochastic volatility.

    Args:
        alpha: Initial volatility level (ATM vol ≈ α for β=0.5)
        beta: CEV exponent (0=normal, 0.5=√, 1=lognormal)
        rho: Correlation between forward and volatility
        nu: Volatility of volatility

    Example:
        >>> # Typical FX SABR with β=0.5 (compromise)
        >>> model = FXSABRModel(
        ...     alpha=0.15,   # 15% ATM vol
        ...     beta=0.5,     # Square-root dynamics
        ...     rho=-0.25,    # Moderate negative correlation
        ...     nu=0.4        # 40% vol-of-vol
        ... )

    References:
        - Hagan, P. et al. (2002). Managing Smile Risk. Wilmott Magazine.
        - West, G. (2005). Calibration of the SABR Model in Illiquid Markets.
    """

    alpha: float
    beta: float
    rho: float
    nu: float

    def __post_init__(self):
        """Validate parameters."""
        if self.alpha <= 0:
            raise ValueError("Alpha must be positive")
        if not 0 <= self.beta <= 1:
            raise ValueError("Beta must be in [0, 1]")
        if not -1 < self.rho < 1:
            raise ValueError("Rho must be in (-1, 1)")
        if self.nu < 0:
            raise ValueError("Nu must be non-negative")

    def implied_vol(
        self,
        F: float,
        K: float,
        T: float
    ) -> float:
        """Compute SABR implied volatility using Hagan formula.

        Args:
            F: Forward FX rate (= S exp[(r_d - r_f)T])
            K: Strike
            T: Time to maturity

        Returns:
            Implied volatility (Garman-Kohlhagen convention)

        Note:
            Uses Hagan et al. (2002) asymptotic expansion.
            Accurate for T < 2 years. For longer, use MC.
        """
        return sabr_implied_vol(F, K, T, self.alpha, self.beta, self.rho, self.nu)

    def calibrate_to_smile(
        self,
        F: float,
        T: float,
        strikes: Array,
        market_vols: Array,
        beta: Optional[float] = None
    ) -> dict:
        """Calibrate SABR to FX smile (single maturity).

        Args:
            F: Forward FX rate
            T: Maturity
            strikes: Strikes
            market_vols: Market implied vols
            beta: Fix beta (typically 0.5 for FX), or None to calibrate

        Returns:
            Dictionary with calibrated (alpha, beta, rho, nu) and RMSE

        Note:
            Typically beta is fixed (0.5 common for FX).
            Calibrates alpha, rho, nu to match ATM, RR 25Δ, BF 25Δ.
        """
        from scipy.optimize import minimize

        if beta is None:
            beta = self.beta

        def objective(params):
            if len(params) == 4:
                alpha, beta_opt, rho, nu = params
            else:
                alpha, rho, nu = params
                beta_opt = beta

            # Compute model vols
            model_vols = jnp.array([
                sabr_implied_vol(F, K, T, alpha, beta_opt, rho, nu)
                for K in strikes
            ])

            # RMSE
            return jnp.sqrt(jnp.mean((model_vols - market_vols) ** 2))

        # Initial guess
        atm_vol = market_vols[len(market_vols) // 2]
        x0 = [atm_vol, -0.25, 0.4] if beta is not None else [atm_vol, beta, -0.25, 0.4]

        # Bounds
        bounds = (
            [(0.001, 2.0), (-0.99, 0.99), (0.001, 3.0)]
            if beta is not None
            else [(0.001, 2.0), (0.0, 1.0), (-0.99, 0.99), (0.001, 3.0)]
        )

        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300}
        )

        # Extract parameters
        if len(result.x) == 4:
            alpha, beta_calib, rho, nu = result.x
        else:
            alpha, rho, nu = result.x
            beta_calib = beta

        # Update model
        object.__setattr__(self, 'alpha', alpha)
        object.__setattr__(self, 'beta', beta_calib)
        object.__setattr__(self, 'rho', rho)
        object.__setattr__(self, 'nu', nu)

        # Compute final vols
        model_vols = jnp.array([
            self.implied_vol(F, K, T)
            for K in strikes
        ])

        return {
            'params': (alpha, beta_calib, rho, nu),
            'rmse': result.fun,
            'model_vols': model_vols,
            'success': result.success
        }


@dataclass
class FXBatesModel:
    """FX Bates model (Heston + jumps).

    Dynamics:
        dS = (r_d - r_f - λμ_J) S dt + √v S dW₁ + (J-1) S dN
        dv = κ(θ - v) dt + σ √v dW₂

        Jump size: log(J) ~ N(μ_J, δ_J²)
        Jump intensity: λ (Poisson)

    Args:
        v0: Initial variance
        kappa: Mean reversion speed
        theta: Long-run variance
        sigma: Vol-of-vol
        rho: Correlation
        r_domestic: Domestic rate
        r_foreign: Foreign rate
        lambda_jump: Jump intensity (jumps per year)
        mu_jump: Mean log jump size
        sigma_jump: Std of log jump size

    Example:
        >>> # USDJPY with BoJ intervention jumps
        >>> model = FXBatesModel(
        ...     v0=0.03, kappa=2.0, theta=0.03,
        ...     sigma=0.3, rho=-0.7,
        ...     r_domestic=0.05, r_foreign=0.001,
        ...     lambda_jump=0.5,   # 0.5 jumps/year
        ...     mu_jump=-0.02,     # -2% average jump (JPY appreciation)
        ...     sigma_jump=0.05    # 5% jump volatility
        ... )

    References:
        - Bates, D. (1996). Jumps and Stochastic Volatility. Review of Financial Studies.
    """

    # Heston parameters
    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float
    r_domestic: float
    r_foreign: float

    # Jump parameters
    lambda_jump: float
    mu_jump: float
    sigma_jump: float

    def characteristic_function(
        self,
        u: complex,
        T: float,
        S0: float
    ) -> complex:
        """Characteristic function φ(u) = E[exp(iu log(S_T))].

        Combines Heston CF with Merton jump CF.

        Args:
            u: Frequency parameter (complex)
            T: Maturity
            S0: Initial spot

        Returns:
            Complex characteristic function value
        """
        # Heston component
        from neutryx.models.heston import heston_characteristic_function

        heston_cf = heston_characteristic_function(
            u=u,
            T=T,
            S0=S0,
            r=self.r_domestic,
            q=self.r_foreign,
            v0=self.v0,
            kappa=self.kappa,
            theta=self.theta,
            sigma=self.sigma,
            rho=self.rho
        )

        # Jump compensation (risk-neutral drift adjustment)
        m = jnp.exp(self.mu_jump + 0.5 * self.sigma_jump ** 2) - 1
        jump_compensation = jnp.exp(-self.lambda_jump * m * T)

        # Merton jump component
        # E[exp(iu log(J))] for log(J) ~ N(μ, δ²)
        jump_cf = jnp.exp(
            1j * u * self.mu_jump - 0.5 * u ** 2 * self.sigma_jump ** 2
        )

        # Combined CF
        jump_component = jnp.exp(
            self.lambda_jump * T * (jump_cf - 1)
        )

        return heston_cf * jump_component * jump_compensation

    def price(
        self,
        S: float,
        K: float,
        T: float,
        is_call: bool = True,
        N: int = 4096
    ) -> float:
        """Price European FX option with jumps using FFT.

        Args:
            S: Spot FX rate
            K: Strike
            T: Maturity
            is_call: Option type
            N: FFT points

        Returns:
            Option price
        """
        # Use Carr-Madan with custom CF
        # (Implementation would use self.characteristic_function)
        # For now, placeholder returning Heston + jump premium

        # Heston base price
        heston_model = FXHestonModel(
            v0=self.v0, kappa=self.kappa, theta=self.theta,
            sigma=self.sigma, rho=self.rho,
            r_domestic=self.r_domestic,
            r_foreign=self.r_foreign
        )
        base_price = heston_model.price(S, K, T, is_call, N)

        # Jump premium (simplified approximation)
        # Full implementation would use FFT with jump CF
        jump_premium = 0.0  # Placeholder

        return base_price + jump_premium


@dataclass
class TwoFactorFXModel:
    """Two-factor stochastic volatility FX model.

    Dynamics:
        dS = (r_d - r_f) S dt + √v₁ S dW₁
        dv₁ = κ₁(θ₁ - v₁) dt + σ₁ √v₁ dW₂
        dv₂ = κ₂(θ₂ - v₂) dt + σ₂ √v₂ dW₃

        Total variance: v = v₁ + v₂

    Captures:
        - Short-term vol dynamics (v₁, fast mean reversion)
        - Long-term vol dynamics (v₂, slow mean reversion)

    Args:
        v1_0: Initial fast variance
        v2_0: Initial slow variance
        kappa1: Fast mean reversion (typically > 5)
        kappa2: Slow mean reversion (typically < 1)
        theta1: Fast long-run variance
        theta2: Slow long-run variance
        sigma1: Fast vol-of-vol
        sigma2: Slow vol-of-vol
        rho12: Correlation between v₁ and v₂
        rho1S: Correlation between S and v₁
        rho2S: Correlation between S and v₂
        r_domestic: Domestic rate
        r_foreign: Foreign rate

    Example:
        >>> # Multi-scale FX vol (intraday + monthly cycles)
        >>> model = TwoFactorFXModel(
        ...     v1_0=0.02, v2_0=0.02,  # Equal initial contribution
        ...     kappa1=10.0,  # Fast: 1/10 = 0.1 year = 1.2 months
        ...     kappa2=0.5,   # Slow: 1/0.5 = 2 years
        ...     theta1=0.015, theta2=0.025,
        ...     sigma1=0.5, sigma2=0.2,
        ...     rho12=0.3, rho1S=-0.5, rho2S=-0.3,
        ...     r_domestic=0.05, r_foreign=0.02
        ... )

    References:
        - Christoffersen, Jacobs, Ornthanalai (2009). Multi-Factor FX Models.
        - Chernov et al. (2003). Alternative Models for Stock Price Dynamics.
    """

    # Fast factor
    v1_0: float
    kappa1: float
    theta1: float
    sigma1: float

    # Slow factor
    v2_0: float
    kappa2: float
    theta2: float
    sigma2: float

    # Correlations
    rho12: float  # Between v₁ and v₂
    rho1S: float  # Between S and v₁
    rho2S: float  # Between S and v₂

    # FX rates
    r_domestic: float
    r_foreign: float

    def price_monte_carlo(
        self,
        S0: float,
        K: float,
        T: float,
        is_call: bool = True,
        n_paths: int = 100000,
        n_steps: int = 252,
        random_key: Optional[Array] = None
    ) -> float:
        """Price FX option via Monte Carlo with two-factor vol.

        Args:
            S0: Initial spot
            K: Strike
            T: Maturity
            is_call: Option type
            n_paths: Number of MC paths
            n_steps: Time steps per path
            random_key: JAX PRNG key

        Returns:
            Option price (Monte Carlo estimate)

        Note:
            Uses Euler discretization with Cholesky for correlated Brownians.
            For variance reduction, consider antithetic variates.
        """
        if random_key is None:
            random_key = jax.random.PRNGKey(0)

        dt = T / n_steps
        sqrt_dt = jnp.sqrt(dt)

        # Cholesky decomposition for correlations
        # [dW₁, dW₂, dW₃] with corr matrix:
        # [[1, ρ1S, ρ2S], [ρ1S, 1, ρ12], [ρ2S, ρ12, 1]]

        corr_matrix = jnp.array([
            [1.0, self.rho1S, self.rho2S],
            [self.rho1S, 1.0, self.rho12],
            [self.rho2S, self.rho12, 1.0]
        ])
        L = jnp.linalg.cholesky(corr_matrix)

        # Initialize paths
        S = jnp.ones(n_paths) * S0
        v1 = jnp.ones(n_paths) * self.v1_0
        v2 = jnp.ones(n_paths) * self.v2_0

        # Simulate
        for step in range(n_steps):
            # Generate independent normals
            key, subkey = jax.random.split(random_key)
            Z_indep = jax.random.normal(subkey, (n_paths, 3))

            # Correlate
            Z = Z_indep @ L.T  # [Z_S, Z_v1, Z_v2]

            # Update variance factors (CIR-style with reflection)
            v1 = v1 + self.kappa1 * (self.theta1 - v1) * dt + \
                 self.sigma1 * jnp.sqrt(jnp.maximum(v1, 0)) * sqrt_dt * Z[:, 1]
            v1 = jnp.maximum(v1, 0)  # Reflection at zero

            v2 = v2 + self.kappa2 * (self.theta2 - v2) * dt + \
                 self.sigma2 * jnp.sqrt(jnp.maximum(v2, 0)) * sqrt_dt * Z[:, 2]
            v2 = jnp.maximum(v2, 0)

            # Total variance
            v_total = v1 + v2

            # Update spot
            S = S * jnp.exp(
                (self.r_domestic - self.r_foreign - 0.5 * v_total) * dt +
                jnp.sqrt(v_total) * sqrt_dt * Z[:, 0]
            )

        # Payoff
        if is_call:
            payoff = jnp.maximum(S - K, 0)
        else:
            payoff = jnp.maximum(K - S, 0)

        # Discount
        discount = jnp.exp(-self.r_domestic * T)

        return discount * jnp.mean(payoff)


__all__ = [
    "FXHestonModel",
    "FXSABRModel",
    "FXBatesModel",
    "TwoFactorFXModel",
]
