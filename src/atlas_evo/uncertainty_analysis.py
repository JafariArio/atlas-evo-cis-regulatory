import pandas as pd
import numpy as np


def add_abs_error(df, y_col="y_true", pred_col="y_pred"):
    out = df.copy()
    out["abs_error"] = (out[y_col].astype(float) - out[pred_col].astype(float)).abs()
    return out


def uncertainty_bins(df, uncertainty_col="uncertainty", error_col="abs_error", bins=5):
    out = df.copy()
    out["uncertainty_bin"] = pd.qcut(out[uncertainty_col], q=bins, duplicates="drop")
    return out.groupby("uncertainty_bin", observed=True).agg(
        n=(error_col, "size"),
        mean_uncertainty=(uncertainty_col, "mean"),
        mean_abs_error=(error_col, "mean"),
    ).reset_index()


def risk_coverage(df, uncertainty_col="uncertainty", error_col="abs_error", steps=20):
    out = df.sort_values(uncertainty_col, ascending=True).reset_index(drop=True)
    rows = []
    for coverage in np.linspace(0.05, 1.0, steps):
        n = max(int(round(len(out) * coverage)), 1)
        sub = out.iloc[:n]
        rows.append({"coverage": coverage, "n": n, "risk_mean_abs_error": float(sub[error_col].mean())})
    return pd.DataFrame(rows)
