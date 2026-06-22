import pandas as pd


def summarize_ablation(df, level_col="display_level", metric_col="r2"):
    return df.groupby(level_col).agg(
        n_tracks=("track_name", "nunique"),
        median_r2=(metric_col, "median"),
        mean_r2=(metric_col, "mean"),
        max_r2=(metric_col, "max"),
        min_r2=(metric_col, "min"),
    ).reset_index()
