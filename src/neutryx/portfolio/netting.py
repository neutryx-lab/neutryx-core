def net_exposure(payoffs_matrix):
    # payoffs_matrix: [trades, paths] common maturity for toy demo
    return payoffs_matrix.sum(axis=0)
