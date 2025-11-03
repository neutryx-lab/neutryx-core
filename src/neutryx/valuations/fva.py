def fva(epe_t, funding_spread, df_t):
    return (df_t * epe_t * funding_spread).sum()
