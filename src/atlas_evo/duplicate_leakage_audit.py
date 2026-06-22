import pandas as pd
from .sequence_features import clean_sequence


def sequence_set(df, sequence_col="sequence"):
    return set(df[sequence_col].dropna().map(clean_sequence))


def pairwise_overlap(track_a, track_b, name_a="track_a", name_b="track_b", sequence_col="sequence"):
    a = sequence_set(track_a, sequence_col)
    b = sequence_set(track_b, sequence_col)
    inter = a & b
    union = a | b
    smaller = min(len(a), len(b)) if min(len(a), len(b)) else 1
    return {
        "track_a": name_a,
        "track_b": name_b,
        "seq_overlap_count": len(inter),
        "seq_jaccard": len(inter) / len(union) if union else 0.0,
        "seq_overlap_fraction_of_smaller_track": len(inter) / smaller,
    }


def summarize_duplicate_risk(df, risk_col="leakage_or_duplication_risk"):
    grouped = df.groupby(risk_col, dropna=False).agg(
        n_pairs=(risk_col, "size"),
        max_seq_jaccard=("seq_jaccard", "max"),
        max_overlap_fraction_smaller=("seq_overlap_fraction_of_smaller_track", "max"),
    ).reset_index()
    return grouped.sort_values("n_pairs", ascending=False)
