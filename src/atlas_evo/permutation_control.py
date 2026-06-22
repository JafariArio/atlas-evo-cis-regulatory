import numpy as np
from .metrics import r2_score_np, rmse_np, mae_np


def permutation_r2_control(y_true, y_pred, n_permutations=500, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    observed = r2_score_np(y_true, y_pred)
    null = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        null[i] = r2_score_np(rng.permutation(y_true), y_pred)
    p_value = (np.sum(null >= observed) + 1) / (n_permutations + 1)
    return {
        "observed_r2": observed,
        "null_r2_median": float(np.median(null)),
        "null_r2_q025": float(np.quantile(null, 0.025)),
        "null_r2_q975": float(np.quantile(null, 0.975)),
        "empirical_p_null_r2_ge_observed": float(p_value),
        "n_permutations": int(n_permutations),
    }
