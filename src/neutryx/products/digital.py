"""Digital (Binary) option pricing.

Digital options pay a fixed amount if the underlying crosses a barrier at maturity.
Also known as binary options or all-or-nothing options.
"""
import jax.numpy as jnp

from neutryx.core.engine import Array


def digital_payoff(ST: Array, K: float, payout: float = 1.0, is_call: bool = True) -> Array:
    """Digital option payoff: pays fixed amount if condition is met.

    Parameters
    ----------
    ST : Array
        Terminal asset prices
    K : float
        Strike price
    payout : float
        Fixed payout amount if condition is met
    is_call : bool
        If True, pays when ST >= K (call). If False, pays when ST <= K (put)

    Returns
    -------
    Array
        Payoff for each path
    """
    condition = ST >= K if is_call else ST <= K
    return jnp.where(condition, payout, 0.0)


def digital_call_payoff(ST: Array, K: float, payout: float = 1.0) -> Array:
    """Digital call payoff: pays fixed amount if ST >= K.

    Parameters
    ----------
    ST : Array
        Terminal asset prices
    K : float
        Strike price
    payout : float
        Fixed payout amount if condition is met

    Returns
    -------
    Array
        Payoff for each path
    """
    return digital_payoff(ST, K, payout, is_call=True)


def digital_put_payoff(ST: Array, K: float, payout: float = 1.0) -> Array:
    """Digital put payoff: pays fixed amount if ST <= K.

    Parameters
    ----------
    ST : Array
        Terminal asset prices
    K : float
        Strike price
    payout : float
        Fixed payout amount if condition is met

    Returns
    -------
    Array
        Payoff for each path
    """
    return digital_payoff(ST, K, payout, is_call=False)


def digital_analytical(S: float, K: float, T: float, r: float, q: float,
                       sigma: float, payout: float = 1.0, is_call: bool = True) -> float:
    """Analytical price for digital option using Black-Scholes formula.

    Parameters
    ----------
    S : float
        Current asset price
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    q : float
        Dividend yield
    sigma : float
        Volatility
    payout : float
        Fixed payout amount
    is_call : bool
        If True, prices digital call. If False, prices digital put

    Returns
    -------
    float
        Digital option price
    """
    from scipy.stats import norm

    d2 = (jnp.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * jnp.sqrt(T))

    # Digital call = exp(-rT) * payout * N(d2)
    # Digital put = exp(-rT) * payout * N(-d2)
    cdf_arg = float(d2) if is_call else float(-d2)
    price = jnp.exp(-r * T) * payout * norm.cdf(cdf_arg)

    return float(price)


def digital_call_analytical(S: float, K: float, T: float, r: float, q: float,
                            sigma: float, payout: float = 1.0) -> float:
    """Analytical price for digital call option using Black-Scholes formula.

    The digital call pays a fixed amount if S_T >= K at maturity.

    Parameters
    ----------
    S : float
        Current asset price
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    q : float
        Dividend yield
    sigma : float
        Volatility
    payout : float
        Fixed payout amount

    Returns
    -------
    float
        Digital call option price
    """
    return digital_analytical(S, K, T, r, q, sigma, payout, is_call=True)


def digital_put_analytical(S: float, K: float, T: float, r: float, q: float,
                           sigma: float, payout: float = 1.0) -> float:
    """Analytical price for digital put option using Black-Scholes formula.

    The digital put pays a fixed amount if S_T <= K at maturity.

    Parameters
    ----------
    S : float
        Current asset price
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    q : float
        Dividend yield
    sigma : float
        Volatility
    payout : float
        Fixed payout amount

    Returns
    -------
    float
        Digital put option price
    """
    return digital_analytical(S, K, T, r, q, sigma, payout, is_call=False)


def digital_mc(paths: Array, K: float, r: float, T: float,
               payout: float = 1.0, is_call: bool = True) -> float:
    """Monte Carlo pricing for digital option.

    Parameters
    ----------
    paths : Array
        Simulated price paths of shape [paths, steps+1]
    K : float
        Strike price
    r : float
        Risk-free rate
    T : float
        Time to maturity
    payout : float
        Fixed payout amount
    is_call : bool
        If True, prices digital call. If False, prices digital put

    Returns
    -------
    float
        Digital option price
    """
    ST = paths[:, -1]
    payoffs = digital_payoff(ST, K, payout, is_call)
    discount = jnp.exp(-r * T)
    return float((discount * payoffs).mean())


def digital_call_mc(paths: Array, K: float, r: float, T: float, payout: float = 1.0) -> float:
    """Monte Carlo pricing for digital call option.

    Parameters
    ----------
    paths : Array
        Simulated price paths of shape [paths, steps+1]
    K : float
        Strike price
    r : float
        Risk-free rate
    T : float
        Time to maturity
    payout : float
        Fixed payout amount

    Returns
    -------
    float
        Digital call option price
    """
    return digital_mc(paths, K, r, T, payout, is_call=True)


def digital_put_mc(paths: Array, K: float, r: float, T: float, payout: float = 1.0) -> float:
    """Monte Carlo pricing for digital put option.

    Parameters
    ----------
    paths : Array
        Simulated price paths of shape [paths, steps+1]
    K : float
        Strike price
    r : float
        Risk-free rate
    T : float
        Time to maturity
    payout : float
        Fixed payout amount

    Returns
    -------
    float
        Digital put option price
    """
    return digital_mc(paths, K, r, T, payout, is_call=False)


# Compound digital options (call spread approximation)
def digital_call_spread_approx(S: float, K: float, T: float, r: float, q: float,
                               sigma: float, spread: float, payout: float = 1.0) -> float:
    """Digital call approximated by call spread.

    This provides a smoother approximation to the digital payoff using
    a tight call spread. Better behaved for hedging and Greeks calculation.

    Parameters
    ----------
    S : float
        Current asset price
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    q : float
        Dividend yield
    sigma : float
        Volatility
    spread : float
        Width of the call spread (e.g., 0.01 * K)
    payout : float
        Fixed payout amount

    Returns
    -------
    float
        Approximate digital call price
    """
    from neutryx.models.bs import price as bs_price

    K1 = K
    K2 = K + spread

    call1 = bs_price(S, K1, T, r, q, sigma, kind="call")
    call2 = bs_price(S, K2, T, r, q, sigma, kind="call")

    # (Call(K1) - Call(K2)) / spread * payout
    return float((call1 - call2) / spread * payout)


__all__ = [
    # Unified functions
    "digital_payoff",
    "digital_analytical",
    "digital_mc",
    # Convenience wrappers for backward compatibility
    "digital_call_payoff",
    "digital_put_payoff",
    "digital_call_analytical",
    "digital_put_analytical",
    "digital_call_mc",
    "digital_put_mc",
    "digital_call_spread_approx",
]
