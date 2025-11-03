# Simple demonstration of LSM for an American put (educational; not production-ready).
import jax.numpy as jnp


def american_put_lsm(ST_paths, K, r, dt):
    # ST_paths: [paths, steps+1]
    # Basis: [1, S, S^2]
    paths, steps_plus = ST_paths.shape
    steps = steps_plus - 1
    cashflows = jnp.zeros((paths, steps_plus))
    exercise = jnp.zeros((paths, steps_plus), dtype=bool)

    # Immediate payoff matrix for put
    payoff = jnp.maximum(K - ST_paths, 0.0)

    V = jnp.zeros((paths,))
    for t in range(steps-1, 0, -1):
        itm_mask = payoff[:, t] > 0
        X = ST_paths[itm_mask, t]
        Y = jnp.exp(-r*dt) * V[itm_mask]  # discounted continuation from t+1
        if X.size > 0:
            A = jnp.vstack([jnp.ones_like(X), X, X*X]).T
            beta, *_ = jnp.linalg.lstsq(A, Y, rcond=None)
            cont = (beta[0] + beta[1]*X + beta[2]*X*X)
            ex = payoff[itm_mask, t] >= cont
            # update exercised states
            idx = jnp.where(itm_mask)[0]
            exercise = exercise.at[idx[ex], t].set(True)
            cashflows = cashflows.at[idx[ex], t].set(payoff[itm_mask, t][ex])
            # those not exercised carry V
            not_ex = ~ex
            V = V.at[idx[not_ex]].set(V[idx[not_ex]] * jnp.exp(-r*dt))
            # exercised paths set continuation to zero
            V = V.at[idx[ex]].set(0.0)
        else:
            V = V * jnp.exp(-r*dt)

    # Terminal payoff for unexercised
    cashflows = cashflows.at[:, -1].set(jnp.where(cashflows.sum(axis=1)==0, payoff[:, -1], 0.0))
    times = jnp.arange(steps_plus) * dt
    disc = jnp.exp(-r*times)
    pv = (cashflows * disc).sum(axis=1).mean()
    return pv
