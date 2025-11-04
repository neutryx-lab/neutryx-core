"""Advanced credit spread curves and term structure modeling.

This module provides infrastructure for modeling credit spreads, hazard rates,
and survival probabilities for counterparty credit risk and credit derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp
from jax import Array
from scipy.optimize import minimize, root_scalar


class CreditCurveType(Enum):
    """Type of credit curve."""
    HAZARD_RATE = "hazard_rate"  # Piecewise constant hazard rates
    SURVIVAL_PROB = "survival_prob"  # Survival probabilities
    CDS_SPREAD = "cds_spread"  # CDS par spreads
    Z_SPREAD = "z_spread"  # Z-spread over risk-free


@dataclass
class HazardRateCurve:
    """Hazard rate (default intensity) curve.

    The hazard rate λ(t) is the instantaneous default probability:
    P(default in [t, t+dt] | survived to t) = λ(t) dt

    Survival probability: Q(t) = exp(-∫₀ᵗ λ(s) ds)

    Attributes:
        tenors: Pillar times [n_pillars]
        hazard_rates: Piecewise constant hazard rates [n_pillars]
        recovery_rate: Recovery rate on default (e.g., 0.4 = 40%)
    """

    tenors: Array
    hazard_rates: Array
    recovery_rate: float = 0.4

    def __post_init__(self):
        """Validate inputs."""
        self.tenors = jnp.asarray(self.tenors)
        self.hazard_rates = jnp.asarray(self.hazard_rates)

        if len(self.tenors) != len(self.hazard_rates):
            raise ValueError("Tenors and hazard rates must have same length")

        if jnp.any(self.hazard_rates < 0):
            raise ValueError("Hazard rates must be non-negative")

        if not 0 <= self.recovery_rate <= 1:
            raise ValueError("Recovery rate must be between 0 and 1")

    def survival_probability(self, t: float) -> float:
        """Compute survival probability Q(t) = P(no default by time t).

        Args:
            t: Time

        Returns:
            Survival probability
        """
        # Integrate piecewise constant hazard rate
        integral = 0.0
        prev_time = 0.0

        for i, tenor in enumerate(self.tenors):
            if t <= tenor:
                # Partial segment
                integral += float(self.hazard_rates[i]) * (t - prev_time)
                break
            else:
                # Full segment
                integral += float(self.hazard_rates[i]) * (tenor - prev_time)
                prev_time = tenor

        # Extrapolate with last hazard rate
        if t > self.tenors[-1]:
            integral += float(self.hazard_rates[-1]) * (t - float(self.tenors[-1]))

        return float(jnp.exp(-integral))

    def default_probability(self, t: float) -> float:
        """Compute cumulative default probability P(default by time t).

        Args:
            t: Time

        Returns:
            Default probability
        """
        return 1.0 - self.survival_probability(t)

    def marginal_default_probability(self, t1: float, t2: float) -> float:
        """Compute marginal default probability P(default in [t1, t2]).

        Args:
            t1: Start time
            t2: End time

        Returns:
            Marginal default probability
        """
        return self.survival_probability(t1) - self.survival_probability(t2)

    def forward_hazard_rate(self, t: float) -> float:
        """Get hazard rate at time t.

        Args:
            t: Time

        Returns:
            Hazard rate
        """
        # Piecewise constant interpolation
        return float(jnp.interp(t, self.tenors, self.hazard_rates))

    def expected_loss(self, t: float) -> float:
        """Compute expected loss E[LGD × 1_{default by t}].

        Args:
            t: Time

        Returns:
            Expected loss
        """
        lgd = 1.0 - self.recovery_rate
        return lgd * self.default_probability(t)


@dataclass
class CDSCurve:
    """CDS (Credit Default Swap) spread curve.

    Par CDS spreads are the market-quoted prices for credit protection.
    Used to bootstrap hazard rate curves.

    Attributes:
        tenors: CDS maturities [n_pillars]
        spreads: Par CDS spreads in bps [n_pillars]
        recovery_rate: Assumed recovery rate
        discount_curve: Risk-free discount curve
    """

    tenors: Array
    spreads: Array  # In basis points (100 = 1%)
    recovery_rate: float = 0.4
    discount_curve: Optional[Callable[[float], float]] = None

    def __post_init__(self):
        """Validate inputs."""
        self.tenors = jnp.asarray(self.tenors)
        self.spreads = jnp.asarray(self.spreads)

        if len(self.tenors) != len(self.spreads):
            raise ValueError("Tenors and spreads must have same length")

        if jnp.any(self.spreads < 0):
            raise ValueError("CDS spreads must be non-negative")

        # Default discount curve if not provided
        if self.discount_curve is None:
            self.discount_curve = lambda t: jnp.exp(-0.03 * t)

    def bootstrap_hazard_rates(self) -> HazardRateCurve:
        """Bootstrap hazard rate curve from CDS spreads.

        Uses iterative bootstrapping: solve for hazard rate at each tenor
        such that the CDS is at par (NPV = 0).

        Returns:
            Bootstrapped hazard rate curve
        """
        hazard_rates = []

        for i, (tenor, spread) in enumerate(zip(self.tenors, self.spreads)):
            # Build partial hazard curve up to this point
            if i == 0:
                partial_tenors = jnp.array([tenor])
                partial_hazards = jnp.array([0.01])  # Initial guess
            else:
                partial_tenors = jnp.concatenate([jnp.array(hazard_rates[: i]), jnp.array([tenor])])
                partial_hazards = jnp.concatenate([jnp.array(hazard_rates), jnp.array([0.01])])

            # Solve for hazard rate that makes CDS at par
            def objective(h):
                test_hazards = partial_hazards.at[i].set(h)
                temp_curve = HazardRateCurve(
                    partial_tenors[: i + 1], test_hazards[: i + 1], self.recovery_rate
                )
                return self._cds_npv(temp_curve, tenor, spread / 10000.0)  # Convert bps to decimal

            try:
                result = root_scalar(objective, bracket=[0.0001, 1.0], method="brentq")
                hazard_rates.append(result.root)
            except ValueError:
                # If bracket fails, use optimization
                result = minimize(lambda h: objective(h[0]) ** 2, x0=[0.01], bounds=[(0.0001, 1.0)])
                hazard_rates.append(result.x[0])

        return HazardRateCurve(self.tenors, jnp.array(hazard_rates), self.recovery_rate)

    def _cds_npv(self, hazard_curve: HazardRateCurve, maturity: float, spread: float) -> float:
        """Compute NPV of CDS contract.

        NPV = PV(premium leg) - PV(protection leg)

        For a par CDS, NPV = 0.

        Args:
            hazard_curve: Hazard rate curve
            maturity: CDS maturity
            spread: CDS spread (decimal, not bps)

        Returns:
            NPV of CDS
        """
        # Discretize into quarterly payments
        n_periods = int(maturity * 4)  # Quarterly
        dt = maturity / n_periods
        times = jnp.linspace(dt, maturity, n_periods)

        # Premium leg: sum of spread × DF(t) × Q(t) × dt
        premium_leg = 0.0
        for t in times:
            df = self.discount_curve(t)
            surv = hazard_curve.survival_probability(t)
            premium_leg += spread * df * surv * dt

        # Protection leg: sum of LGD × DF(t) × dQ(t)
        lgd = 1.0 - self.recovery_rate
        protection_leg = 0.0

        prev_surv = 1.0
        prev_time = 0.0

        for t in times:
            df = self.discount_curve(t)
            surv = hazard_curve.survival_probability(t)
            default_prob = prev_surv - surv

            # Approximate DF at default time (midpoint)
            df_mid = self.discount_curve((prev_time + t) / 2)
            protection_leg += lgd * df_mid * default_prob

            prev_surv = surv
            prev_time = t

        return premium_leg - protection_leg


@dataclass
class CreditSpreadCurve:
    """Credit spread curve (Z-spread or G-spread over benchmarks).

    Attributes:
        tenors: Pillar times
        spreads: Credit spreads in basis points
        spread_type: Type of spread ("z_spread", "g_spread", "oas")
    """

    tenors: Array
    spreads: Array  # In basis points
    spread_type: str = "z_spread"

    def __post_init__(self):
        """Validate inputs."""
        self.tenors = jnp.asarray(self.tenors)
        self.spreads = jnp.asarray(self.spreads)

        if len(self.tenors) != len(self.spreads):
            raise ValueError("Tenors and spreads must have same length")

    def get_spread(self, t: float) -> float:
        """Get credit spread at time t by interpolation.

        Args:
            t: Time

        Returns:
            Credit spread in basis points
        """
        return float(jnp.interp(t, self.tenors, self.spreads))

    def risky_discount_factor(self, t: float, risk_free_df: float) -> float:
        """Compute risky discount factor.

        Risky DF = Risk-free DF × exp(-spread × t)

        Args:
            t: Time
            risk_free_df: Risk-free discount factor at time t

        Returns:
            Risky discount factor
        """
        spread_decimal = self.get_spread(t) / 10000.0
        return risk_free_df * jnp.exp(-spread_decimal * t)


@dataclass
class CreditIndex:
    """Credit index (e.g., CDX, iTraxx) modeling.

    Attributes:
        name: Index name (e.g., "CDX.IG", "iTraxx Europe")
        n_names: Number of names in the index
        spreads: Index spreads by tenor
        recovery_rate: Assumed recovery rate
        individual_spreads: Optional individual name spreads
    """

    name: str
    n_names: int
    spreads: Dict[float, float]  # tenor -> spread in bps
    recovery_rate: float = 0.4
    individual_spreads: Optional[Dict[str, CDSCurve]] = None

    def index_hazard_curve(self) -> HazardRateCurve:
        """Bootstrap hazard curve for the index.

        Returns:
            Index hazard rate curve
        """
        tenors = sorted(self.spreads.keys())
        spread_values = [self.spreads[t] for t in tenors]

        cds_curve = CDSCurve(
            tenors=jnp.array(tenors), spreads=jnp.array(spread_values), recovery_rate=self.recovery_rate
        )

        return cds_curve.bootstrap_hazard_rates()

    def expected_losses(self, tenor: float) -> float:
        """Compute expected loss for the index.

        Args:
            tenor: Time horizon

        Returns:
            Expected loss rate
        """
        hazard_curve = self.index_hazard_curve()
        return hazard_curve.expected_loss(tenor)

    def tranche_pricing(
        self, attachment: float, detachment: float, tenor: float, n_simulations: int = 10000
    ) -> dict:
        """Price CDO tranche on the index.

        Args:
            attachment: Attachment point (e.g., 0.03 = 3%)
            detachment: Detachment point (e.g., 0.07 = 7%)
            tenor: Tranche maturity
            n_simulations: Number of Monte Carlo simulations

        Returns:
            Dictionary with tranche pricing metrics
        """
        import jax

        hazard_curve = self.index_hazard_curve()

        # Simulate default times for all names
        # Assume homogeneous pool with correlation
        correlation = 0.3  # Typical index correlation
        key = jax.random.PRNGKey(42)

        # Simple Gaussian copula
        key1, key2 = jax.random.split(key)
        common_factor = jax.random.normal(key1, (n_simulations,))
        idio_factors = jax.random.normal(key2, (n_simulations, self.n_names))

        sqrt_rho = jnp.sqrt(correlation)
        sqrt_1mrho = jnp.sqrt(1 - correlation)

        latent = sqrt_rho * common_factor[:, None] + sqrt_1mrho * idio_factors

        # Map to default times via inverse survival probability
        from scipy.stats import norm

        uniforms = norm.cdf(latent)

        # Default times: Q(τ) = U → τ = Q^{-1}(U)
        # For exponential: τ = -log(1-U) / λ
        default_times = -jnp.log(1 - uniforms + 1e-10) / hazard_curve.forward_hazard_rate(tenor / 2)

        # Count defaults by tenor
        defaults = (default_times <= tenor).astype(float)
        loss_rate = jnp.mean(defaults, axis=1)  # Portfolio loss for each simulation

        # Tranche loss
        lgd = 1.0 - self.recovery_rate
        portfolio_loss = loss_rate * lgd

        tranche_loss = jnp.maximum(0, jnp.minimum(portfolio_loss - attachment, detachment - attachment)) / (
            detachment - attachment
        )

        expected_loss = float(jnp.mean(tranche_loss))
        std_loss = float(jnp.std(tranche_loss))

        return {
            "expected_loss": expected_loss,
            "std_loss": std_loss,
            "attachment": attachment,
            "detachment": detachment,
            "tenor": tenor,
        }


def credit_spread_from_pd_lgd(pd: float, lgd: float, T: float) -> float:
    """Convert default probability and LGD to credit spread.

    Approximate formula: spread ≈ PD × LGD / T

    Args:
        pd: Cumulative default probability
        lgd: Loss given default
        T: Time horizon

    Returns:
        Credit spread (decimal, not bps)
    """
    return (pd * lgd) / T


def pd_from_credit_spread(spread: float, lgd: float, T: float) -> float:
    """Convert credit spread to default probability.

    Args:
        spread: Credit spread (decimal)
        lgd: Loss given default
        T: Time horizon

    Returns:
        Cumulative default probability
    """
    return (spread * T) / lgd


@dataclass
class SovereignCreditCurve:
    """Sovereign credit curve with CDS and bond spreads.

    Combines CDS and bond market information for sovereign credit risk.

    Attributes:
        country: Country code (e.g., "US", "DE", "IT")
        cds_curve: CDS spread curve
        bond_spreads: Government bond spreads over risk-free
        currency: Currency of denomination
    """

    country: str
    cds_curve: CDSCurve
    bond_spreads: Optional[Dict[float, float]] = None
    currency: str = "USD"

    def basis(self, tenor: float) -> float:
        """Compute CDS-bond basis.

        Basis = CDS spread - Bond spread

        Args:
            tenor: Maturity

        Returns:
            Basis in bps
        """
        if self.bond_spreads is None:
            return 0.0

        cds_spread = float(jnp.interp(tenor, self.cds_curve.tenors, self.cds_curve.spreads))

        # Interpolate bond spread
        bond_tenors = sorted(self.bond_spreads.keys())
        bond_spread_values = [self.bond_spreads[t] for t in bond_tenors]
        bond_spread = float(jnp.interp(tenor, jnp.array(bond_tenors), jnp.array(bond_spread_values)))

        return cds_spread - bond_spread
