#!/usr/bin/env python3
"""
ATLAS-Evo Phase 5G: ATLAS nonlinear MLP heads

Purpose
-------
Runs MLPRegressor heads on ATLAS-Evo feature/proxy structures, especially for
Random_Promoter_DREAM_base_track_2 and Random_Promoter_DREAM_base_track_5.
This is a focused post-analysis/model-ceiling run. It does not alter the documented
Phase 4/Phase 5 results.

Outputs
-------
PHASE5G_ATLAS_MLP_ALL_METRICS.csv
PHASE5G_ATLAS_MLP_BEST_BY_TRACK.csv
PHASE5G_ATLAS_MLP_BOOTSTRAP_CI.csv
PHASE5G_ATLAS_MLP_SUMMARY.md
PHASE5G_ATLAS_MLP_PREDICTIONS.csv
PHASE5G_ATLAS_MLP_RUN_CONFIG.json
PHASE5G_ATLAS_MLP_RUN_LOG.csv
ATLAS_Evo_Phase5G_ATLAS_MLP_Heads_Workbook.xlsx
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge, ElasticNet

DNA = "ACGT"
VALID = set(DNA)

DEFAULT_TRACKS = [
    "Random_Promoter_DREAM_base_track_2",
    "Random_Promoter_DREAM_base_track_5",
]

TARGET_CANDIDATES = ["target", "activity", "expression", "score", "value", "y", "minus_plasmid"]
SEQUENCE_CANDIDATES = ["sequence", "seq", "dna_sequence", "promoter_sequence", "insert_sequence"]
CONDITION_CANDIDATES = ["condition", "context", "group", "cell_type", "cellline", "cell_line", "label"]
SPLIT_CANDIDATES = ["split", "partition", "subset", "train_test", "official_split"]
ID_CANDIDATES = ["id", "sequence_id", "uid", "name", "oligo_id"]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def safe_name(x: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(x))[:160]


def norm_name(x: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).lower()).strip("_")


def clean_sequence(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = re.sub(r"\s+", "", str(x).upper())
    return "".join(c for c in s if c in "ACGTN")


def find_col(cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower = {str(c).lower(): c for c in cols}
    normed = {norm_name(c): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
        nc = norm_name(cand)
        if nc in normed:
            return normed[nc]
    return None


def read_table(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in [".tsv", ".txt"]:
        try:
            return pd.read_csv(path, sep="\t", nrows=nrows, low_memory=False)
        except Exception:
            return pd.read_csv(path, nrows=nrows, low_memory=False)
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def load_config_tracks(config_json: Path) -> List[Dict[str, Any]]:
    cfg = json.loads(config_json.read_text(encoding="utf-8"))
    if isinstance(cfg, dict) and "tracks" in cfg:
        return [t for t in cfg.get("tracks", []) if t.get("enabled", True)]
    if isinstance(cfg, list):
        return cfg
    raise ValueError("Could not find tracks in config JSON.")


def track_specs_from_prepared_dir(prepared_dir: Path, track_names: List[str]) -> List[Dict[str, Any]]:
    specs = []
    files = list(prepared_dir.glob("*.csv"))
    # fallback map by known filename patterns
    candidates = {
        "Random_Promoter_DREAM_base_track_2": ["filtered_test_data_with_MAUDE_expression", "base_track_2"],
        "Random_Promoter_DREAM_base_track_5": ["test_labeled_maude", "base_track_5"],
    }
    for tn in track_names:
        chosen = None
        for p in files:
            s = p.name.lower()
            if any(k.lower() in s for k in candidates.get(tn, [tn.lower()])):
                chosen = p
                break
        if chosen:
            specs.append({
                "track_name": tn,
                "input_csv": str(chosen),
                "seq_col": "sequence",
                "target_col": "target",
                "condition_col": "condition",
                "split_col": "split",
                "id_col": "id",
                "claim_level": "headline_base_reference",
                "source_role": "base-track structured reference",
            })
    return specs


def infer_or_get(df: pd.DataFrame, given: Optional[str], candidates: Sequence[str]) -> Optional[str]:
    if given and given in df.columns:
        return given
    return find_col(list(df.columns), candidates)


def make_split(df: pd.DataFrame, split_col: Optional[str], test_size: float, random_state: int) -> Tuple[np.ndarray, np.ndarray, str]:
    n = len(df)
    if split_col and split_col in df.columns:
        s = df[split_col].astype(str).str.lower()
        train_mask = s.str.contains("train|training", regex=True)
        test_mask = s.str.contains("test|valid|validation|val|held", regex=True)
        if train_mask.sum() >= 5 and test_mask.sum() >= 5:
            return np.where(train_mask.values)[0], np.where(test_mask.values)[0], "provided_split_column"
    tr, te = train_test_split(np.arange(n), test_size=test_size, random_state=random_state)
    return tr, te, "random_split"


def build_vocab(k: int) -> List[str]:
    return ["".join(p) for p in itertools.product(DNA, repeat=k)]


def gc_len_features(seqs: Sequence[str]) -> np.ndarray:
    X = np.zeros((len(seqs), 2), dtype=np.float32)
    for i, s in enumerate(seqs):
        acgt = [c for c in s if c in VALID]
        L = len(acgt)
        X[i, 0] = 0.0 if L == 0 else (acgt.count("G") + acgt.count("C")) / L
        X[i, 1] = math.log1p(L)
    return X


def mut_prior_features(seqs: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    names = [
        "mut_gc", "mut_at", "mut_cpg", "mut_n_rate", "mut_entropy_dinuc",
        "mut_max_homopolymer_frac", "mut_purine_frac", "mut_transition_proxy",
        "mut_length_log1p", "mut_complexity",
    ]
    X = np.zeros((len(seqs), len(names)), dtype=np.float32)
    for i, s0 in enumerate(seqs):
        chars_all = [c for c in str(s0).upper() if c in "ACGTN"]
        ss = "".join(c for c in chars_all if c in VALID)
        L = len(ss)
        if L == 0:
            continue
        gc = (ss.count("G") + ss.count("C")) / L
        cpg = sum(1 for j in range(L-1) if ss[j:j+2] == "CG") / max(L-1, 1)
        dinucs = [ss[j:j+2] for j in range(L-1)]
        if dinucs:
            vc = pd.Series(dinucs).value_counts().to_numpy(float)
            p = vc / vc.sum()
            entropy = float(-(p * np.log2(p + 1e-12)).sum() / 4.0)
        else:
            entropy = 0.0
        max_run = 1
        cur = 1
        for a, b in zip(ss[:-1], ss[1:]):
            if a == b:
                cur += 1
                max_run = max(max_run, cur)
            else:
                cur = 1
        pur = (ss.count("A") + ss.count("G")) / L
        trans_proxy = 1.0 / 3.0
        comp = 1.0 - pd.Series(list(ss)).value_counts(normalize=True).max()
        X[i] = [gc, 1-gc, cpg, (len(chars_all)-L)/max(len(chars_all),1), entropy, max_run/L, pur, trans_proxy, math.log1p(L), comp]
    return X, names


def kmer_matrix(seqs: Sequence[str], k: int, vocab: List[str], max_features: Optional[int] = None) -> sparse.csr_matrix:
    vi = {km: j for j, km in enumerate(vocab)}
    rows, cols, vals = [], [], []
    for i, s in enumerate(seqs):
        if len(s) < k:
            continue
        counts: Dict[int, int] = {}
        total = 0
        for pos in range(len(s)-k+1):
            km = s[pos:pos+k]
            if all(c in VALID for c in km):
                j = vi.get(km)
                if j is not None:
                    counts[j] = counts.get(j, 0) + 1
                    total += 1
        if total:
            inv = 1.0 / total
            for j, c in counts.items():
                rows.append(i); cols.append(j); vals.append(c * inv)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(seqs), len(vocab)), dtype=np.float32)


def select_train_features(X: sparse.csr_matrix, y: np.ndarray, train_idx: np.ndarray, k: int) -> np.ndarray:
    k = int(min(max(1, k), X.shape[1]))
    sel = SelectKBest(f_regression, k=k)
    sel.fit(X[train_idx], y[train_idx])
    return np.where(sel.get_support())[0].astype(int)


def make_interactions_dense(Xsel: sparse.csr_matrix, source_k: int, max_interactions: int) -> Tuple[np.ndarray, List[Tuple[int,int]]]:
    source_k = min(source_k, Xsel.shape[1])
    pairs = []
    for a, b in itertools.combinations(range(source_k), 2):
        pairs.append((a, b))
        if len(pairs) >= max_interactions:
            break
    if not pairs:
        return np.zeros((Xsel.shape[0], 0), dtype=np.float32), pairs
    Xd = Xsel[:, :source_k].toarray().astype(np.float32)
    out = np.empty((Xd.shape[0], len(pairs)), dtype=np.float32)
    for j, (a, b) in enumerate(pairs):
        out[:, j] = Xd[:, a] * Xd[:, b]
    return out, pairs


def onehot_groups(groups: Sequence[str], train_idx: np.ndarray) -> np.ndarray:
    arr = np.asarray(groups, dtype=object).reshape(-1, 1)
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    enc.fit(arr[train_idx])
    return enc.transform(arr).astype(np.float32)


def standardize_dense(X: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return X


def r2_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() < 2:
        return {"r2": np.nan, "rmse": np.nan, "mae": np.nan, "n_test": int(ok.sum())}
    yt = y_true[ok]
    yp = y_pred[ok]
    r2 = np.nan if np.std(yt) <= 1e-12 else float(r2_score(yt, yp))
    return {"r2": r2, "rmse": float(mean_squared_error(yt, yp) ** 0.5), "mae": float(mean_absolute_error(yt, yp)), "n_test": int(ok.sum())}


def bootstrap_ci(y: np.ndarray, pred: np.ndarray, reps: int, seed: int) -> Tuple[float, float, int]:
    ok = np.isfinite(y) & np.isfinite(pred)
    y = y[ok]
    pred = pred[ok]
    n = len(y)
    if n < 3 or reps <= 0:
        return np.nan, np.nan, n
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        if np.std(y[idx]) <= 1e-12:
            continue
        vals.append(r2_score(y[idx], pred[idx]))
    if not vals:
        return np.nan, np.nan, n
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), n


def make_mlp(seed: int, hidden: Tuple[int, ...], max_iter: int, alpha: float, learning_rate_init: float) -> Any:
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=15,
            random_state=seed,
            batch_size="auto",
            verbose=False,
        ),
    )


def fit_predict_model(model: Any, X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    model.fit(X[train_idx], y[train_idx])
    pred = np.full(len(y), np.nan, dtype=float)
    pred[test_idx] = model.predict(X[test_idx])
    return pred


def observable_alignment_correction(pred_train: np.ndarray, pred_test: np.ndarray, y_train: np.ndarray, groups_train: np.ndarray, groups_test: np.ndarray) -> np.ndarray:
    residual = pd.Series(y_train - pred_train)
    offsets = residual.groupby(pd.Series(groups_train).astype(str)).mean().to_dict()
    off = np.array([offsets.get(str(g), 0.0) for g in groups_test], dtype=float)
    return pred_test + off


def run_one_track(spec: Dict[str, Any], args: argparse.Namespace, outdir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    track_name = spec.get("track_name") or Path(spec.get("input_csv", "track")).stem
    log(f"Running ATLAS-MLP heads for {track_name}")
    df = read_table(Path(spec["input_csv"]))
    if args.max_rows_per_track and len(df) > args.max_rows_per_track:
        df = df.sample(n=args.max_rows_per_track, random_state=args.random_state).reset_index(drop=True)
    seq_col = infer_or_get(df, spec.get("seq_col"), SEQUENCE_CANDIDATES)
    target_col = infer_or_get(df, spec.get("target_col"), TARGET_CANDIDATES)
    cond_col = infer_or_get(df, spec.get("condition_col") or spec.get("group_col"), CONDITION_CANDIDATES)
    split_col = infer_or_get(df, spec.get("split_col"), SPLIT_CANDIDATES)
    id_col = infer_or_get(df, spec.get("id_col"), ID_CANDIDATES)
    if not seq_col or not target_col:
        raise ValueError(f"Missing sequence or target column for {track_name}")
    work = pd.DataFrame({
        "sequence": df[seq_col].map(clean_sequence),
        "target": pd.to_numeric(df[target_col], errors="coerce"),
        "group": df[cond_col].astype(str) if cond_col else "ALL",
        "split": df[split_col].astype(str) if split_col else "",
        "id": df[id_col].astype(str) if id_col else [f"row_{i}" for i in range(len(df))],
    })
    work = work[(work["sequence"].str.len() >= args.kmer) & work["target"].notna()].reset_index(drop=True)
    if len(work) < 20:
        raise ValueError(f"Too few usable rows for {track_name}: {len(work)}")
    train_idx, test_idx, split_rule = make_split(work, "split", args.test_size, args.random_state)
    seqs = work["sequence"].tolist()
    y = work["target"].to_numpy(float)
    groups = work["group"].astype(str).to_numpy()

    vocab = build_vocab(args.kmer)
    X_gc = gc_len_features(seqs)
    X_mut, _ = mut_prior_features(seqs)
    X_full = kmer_matrix(seqs, args.kmer, vocab)
    selected_idx = select_train_features(X_full, y, train_idx, args.select_top_char6)
    X_char = X_full[:, selected_idx]
    X_inter, pairs = make_interactions_dense(X_char, args.interaction_source_k, args.max_multi_interactions)
    X_cond = onehot_groups(groups, train_idx) if pd.Series(groups).nunique() > 1 else np.zeros((len(groups), 0), dtype=np.float32)
    X_atlas = standardize_dense(np.hstack([X_gc, X_char.toarray().astype(np.float32), X_inter, X_cond]))
    X_t02 = standardize_dense(np.hstack([X_mut, X_atlas]))
    X_t04 = X_atlas
    X_t05 = X_atlas

    models: List[Tuple[str, str, str, np.ndarray, str]] = [
        ("B08_MLPRegressor_reproducer", "External MLP baseline on ATLAS feature matrix", "external_mlp_baseline", X_atlas, "plain"),
        ("N06_ATLAS_T04_ObservableAlignment_MLP", "ATLAS T04 observable-alignment proxy + MLP head", "atlas_observable_alignment_mlp", X_t04, "observable_alignment"),
        ("N07_ATLAS_T05_UncertaintyEnsemble_MLP", "ATLAS T05 uncertainty ensemble proxy + MLP heads", "atlas_uncertainty_ensemble_mlp", X_t05, "ensemble"),
        ("N08_ATLAS_T02_SelectionPotential_MLP", "ATLAS T02 selection-potential proxy + MLP residual head", "atlas_selection_potential_mlp", X_t02, "selection"),
        ("N09_ATLAS_FullTheory_MLP", "Combined grammar/proxy matrix + MLP head", "atlas_full_theory_mlp", np.hstack([X_t02, X_cond]), "plain"),
    ]

    metrics, preds_rows, log_rows = [], [], []
    for model_id, label, component, X, mode in models:
        try:
            start = time.time()
            if mode == "plain":
                model = make_mlp(args.random_state, tuple(args.hidden_layers), args.mlp_max_iter, args.mlp_alpha, args.learning_rate_init)
                pred = fit_predict_model(model, X, y, train_idx, test_idx)
            elif mode == "observable_alignment":
                model = make_mlp(args.random_state + 1, tuple(args.hidden_layers), args.mlp_max_iter, args.mlp_alpha, args.learning_rate_init)
                model.fit(X[train_idx], y[train_idx])
                pred = np.full(len(y), np.nan, dtype=float)
                tr_pred = model.predict(X[train_idx])
                te_base = model.predict(X[test_idx])
                pred[test_idx] = observable_alignment_correction(tr_pred, te_base, y[train_idx], groups[train_idx], groups[test_idx])
            elif mode == "ensemble":
                all_preds = []
                for j in range(args.ensemble_n):
                    model = make_mlp(args.random_state + 100 + j, tuple(args.hidden_layers), args.mlp_max_iter, args.mlp_alpha, args.learning_rate_init)
                    rng = np.random.default_rng(args.random_state + 1000 + j)
                    boot = rng.choice(train_idx, size=len(train_idx), replace=True)
                    model.fit(X[boot], y[boot])
                    all_preds.append(model.predict(X[test_idx]))
                P = np.vstack(all_preds)
                pred = np.full(len(y), np.nan, dtype=float)
                pred[test_idx] = P.mean(axis=0)
            elif mode == "selection":
                # mutational/accessibility prior predicts baseline; MLP learns residual selection signal.
                prior = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                prior.fit(X_mut[train_idx], y[train_idx])
                prior_tr = prior.predict(X_mut[train_idx])
                prior_te = prior.predict(X_mut[test_idx])
                residual = y[train_idx] - prior_tr
                mlp = make_mlp(args.random_state + 2, tuple(args.hidden_layers), args.mlp_max_iter, args.mlp_alpha, args.learning_rate_init)
                mlp.fit(X_atlas[train_idx], residual)
                pred = np.full(len(y), np.nan, dtype=float)
                pred[test_idx] = prior_te + mlp.predict(X_atlas[test_idx])
            else:
                raise ValueError("Unknown mode")
            met = r2_metrics(y[test_idx], pred[test_idx])
            lo, hi, n_boot = bootstrap_ci(y[test_idx], pred[test_idx], args.bootstrap_repeats, args.random_state)
            row = {
                "track_name": track_name,
                "model_id": model_id,
                "model_label": label,
                "atlas_component": component,
                "split_rule": split_rule,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "r2_ci_low": lo,
                "r2_ci_high": hi,
                "bootstrap_repeats": args.bootstrap_repeats,
                "seconds": round(time.time() - start, 3),
                **met,
            }
            metrics.append(row)
            for ii in test_idx:
                preds_rows.append({
                    "track_name": track_name,
                    "model_id": model_id,
                    "row_id": work.loc[ii, "id"],
                    "group": groups[ii],
                    "y_true": y[ii],
                    "y_pred": pred[ii],
                })
            log_rows.append({"track_name": track_name, "model_id": model_id, "status": "ok", "seconds": row["seconds"]})
            log(f"  {model_id}: R2={met['r2']:.6g}, CI=[{lo:.6g}, {hi:.6g}], n={met['n_test']}")
        except Exception as e:
            log_rows.append({"track_name": track_name, "model_id": model_id, "status": "error", "error": str(e), "traceback": traceback.format_exc()})
            log(f"  ERROR {model_id}: {e}")
    return metrics, preds_rows, log_rows


def write_outputs(outdir: Path, metrics: List[Dict[str, Any]], preds: List[Dict[str, Any]], logs: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    m = pd.DataFrame(metrics)
    p = pd.DataFrame(preds)
    l = pd.DataFrame(logs)
    m.to_csv(outdir / "PHASE5G_ATLAS_MLP_ALL_METRICS.csv", index=False)
    p.to_csv(outdir / "PHASE5G_ATLAS_MLP_PREDICTIONS.csv", index=False)
    l.to_csv(outdir / "PHASE5G_ATLAS_MLP_RUN_LOG.csv", index=False)
    if not m.empty:
        best = m.sort_values("r2", ascending=False).groupby("track_name", as_index=False).head(1)
        best.to_csv(outdir / "PHASE5G_ATLAS_MLP_BEST_BY_TRACK.csv", index=False)
        mat = m.pivot_table(index="model_id", columns="track_name", values="r2", aggfunc="max")
        mat.to_csv(outdir / "PHASE5G_ATLAS_MLP_R2_MATRIX.csv")
    else:
        pd.DataFrame().to_csv(outdir / "PHASE5G_ATLAS_MLP_BEST_BY_TRACK.csv", index=False)
        pd.DataFrame().to_csv(outdir / "PHASE5G_ATLAS_MLP_R2_MATRIX.csv")

    cfg = vars(args).copy()
    (outdir / "PHASE5G_ATLAS_MLP_RUN_CONFIG.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    lines = [
        "# ATLAS-Evo Phase 5G ATLAS MLP-head rerun\n\n",
        f"Generated: `{now()}`\n\n",
        "## Purpose\n\n",
        "This focused run tests whether the high MLP model-ceiling result can be attached to ATLAS-Evo feature/proxy heads, especially on `Random_Promoter_DREAM_base_track_2` and `_5`.\n\n",
    ]
    if not m.empty:
        lines.append("## Top rows\n\n")
        top = m.sort_values("r2", ascending=False).head(12)
        for _, r in top.iterrows():
            lines.append(f"- `{r['track_name']}` | `{r['model_id']}` | R2={r['r2']:.6g} | CI=[{r['r2_ci_low']:.6g}, {r['r2_ci_high']:.6g}] | n={int(r['n_test'])}\n")
        lines.append("\n## Interpretation rule\n\n")
        lines.append("If ATLAS-Evo MLP heads approach or exceed the external MLP baseline, the manuscript can state that the nonlinear predictor can be implemented as an ATLAS-Evo head rather than only as an external black-box benchmark. If the external MLP remains best, report it as model-ceiling evidence and keep ATLAS-Evo as the theory-aligned, interpretable framework.\n")
    (outdir / "PHASE5G_ATLAS_MLP_SUMMARY.md").write_text("".join(lines), encoding="utf-8")

    try:
        with pd.ExcelWriter(outdir / "ATLAS_Evo_Phase5G_ATLAS_MLP_Heads_Workbook.xlsx", engine="openpyxl") as writer:
            m.to_excel(writer, sheet_name="All_Metrics", index=False)
            if not m.empty:
                best.to_excel(writer, sheet_name="Best_By_Track", index=False)
                mat.to_excel(writer, sheet_name="R2_Matrix")
            l.to_excel(writer, sheet_name="Run_Log", index=False)
            pd.DataFrame([cfg]).to_excel(writer, sheet_name="Run_Config", index=False)
    except Exception as e:
        log(f"Could not write Excel workbook: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_json", default="", help="EFFECTIVE_TRACKS_CONFIG.json or RUN_CONFIG_USED.json from v7 run.")
    ap.add_argument("--prepared_tracks_dir", default="", help="Fallback prepared tracks directory.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--track_names", default=",".join(DEFAULT_TRACKS), help="Comma-separated track names to run.")
    ap.add_argument("--kmer", type=int, default=6)
    ap.add_argument("--max_rows_per_track", type=int, default=150000)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--select_top_char6", type=int, default=128)
    ap.add_argument("--interaction_source_k", type=int, default=64)
    ap.add_argument("--max_multi_interactions", type=int, default=512)
    ap.add_argument("--bootstrap_repeats", type=int, default=500)
    ap.add_argument("--mlp_max_iter", type=int, default=250)
    ap.add_argument("--hidden_layers", type=int, nargs="+", default=[256, 64])
    ap.add_argument("--mlp_alpha", type=float, default=0.0001)
    ap.add_argument("--learning_rate_init", type=float, default=0.001)
    ap.add_argument("--ensemble_n", type=int, default=3)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    tracks_wanted = [x.strip() for x in args.track_names.split(",") if x.strip()]

    if args.config_json:
        all_specs = load_config_tracks(Path(args.config_json))
        specs = [s for s in all_specs if s.get("track_name") in tracks_wanted]
    elif args.prepared_tracks_dir:
        specs = track_specs_from_prepared_dir(Path(args.prepared_tracks_dir), tracks_wanted)
    else:
        raise SystemExit("Provide --config_json or --prepared_tracks_dir")

    if not specs:
        raise SystemExit("No matching track specifications found.")

    all_metrics: List[Dict[str, Any]] = []
    all_preds: List[Dict[str, Any]] = []
    all_logs: List[Dict[str, Any]] = []
    for spec in specs:
        try:
            metrics, preds, logs = run_one_track(spec, args, outdir)
            all_metrics.extend(metrics)
            all_preds.extend(preds)
            all_logs.extend(logs)
        except Exception as e:
            all_logs.append({"track_name": spec.get("track_name"), "status": "track_error", "error": str(e), "traceback": traceback.format_exc()})
            log(f"TRACK ERROR {spec.get('track_name')}: {e}")
    write_outputs(outdir, all_metrics, all_preds, all_logs, args)
    log(f"Done. Outputs written to: {outdir}")


if __name__ == "__main__":
    main()
