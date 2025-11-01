import jax.numpy as jnp

def cva(epe_t, df_t, pd_t, lgd=0.6):
    # Discrete sum over time buckets: sum DF(t) * EPE(t) * dPD(t) * LGD
    dPD = jnp.diff(jnp.concatenate([jnp.array([0.0]), pd_t]))
    return (df_t * epe_t * dPD * lgd).sum()
