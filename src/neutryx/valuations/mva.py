def mva(initial_margin_profile, df_t, spread):
    return (df_t * initial_margin_profile * spread).sum()
