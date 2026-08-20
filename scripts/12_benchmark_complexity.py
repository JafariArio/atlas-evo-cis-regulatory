#!/usr/bin/env python3
"""Controlled computational-complexity benchmark for ATLAS-Evo.

This script measures implementation scaling for representative stages of the
ATLAS-Evo case study on one model-ready track. The repository includes a small
synthetic input for smoke testing; reported measurements use the documented
public-data track. The script deliberately does not
recompute or reinterpret manuscript performance results. The benchmark uses a
timing-only random split and reports wall-clock time, process peak memory,
software/hardware metadata, and empirical log-log scaling exponents.

The script is an orchestrator and worker in one file. Each sample-size/repeat
combination is executed in a fresh subprocess so that peak-memory measurements
remain comparable across sample sizes.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


MAX_SUPPORTED_WORKERS = 8
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "complexity_benchmark_demo.json"
DEFAULT_CODE_DIR = SCRIPT_DIR / "case_study"
DEFAULT_OUTDIR = REPO_ROOT / "outputs" / "complexity"


def parse_args() -> argparse.Namespace:
    default_workers = min(MAX_SUPPORTED_WORKERS, os.cpu_count() or 1)
    parser = argparse.ArgumentParser(
        description="ATLAS-Evo controlled runtime and memory scaling benchmark"
    )
    parser.add_argument(
        "--config-json",
        default=str(DEFAULT_CONFIG),
        help="Track configuration JSON. Defaults to the repository demo configuration.",
    )
    parser.add_argument(
        "--code-dir",
        default=str(DEFAULT_CODE_DIR),
        help="Directory containing the included case-study feature/head module.",
    )
    parser.add_argument(
        "--atlas-root",
        default=str(REPO_ROOT),
        help="Optional root used to rebase legacy absolute paths in older configurations.",
    )
    parser.add_argument(
        "--track-name",
        default="demo_complexity_track",
        help="One sufficiently large model-ready track used only for timing and scaling.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100, 200],
        help="Requested row counts for the controlled scaling benchmark.",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=default_workers)
    parser.add_argument(
        "--outdir",
        default=str(DEFAULT_OUTDIR),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--kmer", type=int, default=6)
    parser.add_argument("--select-top-char6", type=int, default=128)
    parser.add_argument("--interaction-source-k", type=int, default=64)
    parser.add_argument("--max-multi-interactions", type=int, default=512)
    parser.add_argument("--n-estimators", type=int, default=160)
    parser.add_argument("--tree-max-depth", type=int, default=18)
    parser.add_argument("--min-samples-leaf", type=int, default=3)
    parser.add_argument("--hgb-max-iter", type=int, default=220)
    parser.add_argument("--mlp-max-iter", type=int, default=250)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[256, 64])
    parser.add_argument(
        "--models",
        default="ridge,extratrees,histgb,n09_mlp",
        help="Comma-separated timing models: ridge, extratrees, histgb, n09_mlp.",
    )
    parser.add_argument("--memory-sample-ms", type=int, default=50)

    # Internal worker options. Users do not need to set these directly.
    parser.add_argument("--worker-size", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-repeat", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-json", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.n_jobs <= MAX_SUPPORTED_WORKERS:
        raise SystemExit(f"--n-jobs must be between 1 and {MAX_SUPPORTED_WORKERS}.")
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1.")
    if not args.sizes or any(n < 100 for n in args.sizes):
        raise SystemExit("Every requested benchmark size must be at least 100 rows.")
    if not 0.05 <= args.test_size <= 0.5:
        raise SystemExit("--test-size must be between 0.05 and 0.5.")
    allowed = {"ridge", "extratrees", "histgb", "n09_mlp"}
    selected = {m.strip().lower() for m in args.models.split(",") if m.strip()}
    unknown = selected - allowed
    if unknown:
        raise SystemExit(f"Unknown --models values: {sorted(unknown)}")


def cap_numeric_threads(n_jobs: int) -> Dict[str, str]:
    """Cap common native numerical libraries before NumPy/scikit-learn import."""
    values = {
        "OMP_NUM_THREADS": str(n_jobs),
        "MKL_NUM_THREADS": str(n_jobs),
        "OPENBLAS_NUM_THREADS": str(n_jobs),
        "NUMEXPR_NUM_THREADS": str(n_jobs),
        "VECLIB_MAXIMUM_THREADS": str(n_jobs),
        "BLIS_NUM_THREADS": str(n_jobs),
    }
    for key, value in values.items():
        os.environ[key] = value
    return values


def import_from_path(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_track_spec(raw: Dict[str, Any]) -> Dict[str, Any]:
    track = dict(raw)
    track["track_name"] = track.get("track_name") or track.get("name")
    track["input_csv"] = track.get("input_csv") or track.get("path") or track.get("file")
    track["seq_col"] = track.get("seq_col") or track.get("sequence_col") or "sequence"
    track["target_col"] = track.get("target_col") or track.get("y_col") or "target"
    track["condition_col"] = track.get("condition_col") or track.get("group_col") or "condition"
    track["split_col"] = track.get("split_col") or "split"
    track["id_col"] = track.get("id_col") or "id"
    return track


def load_track_spec(config_path: Path, track_name: str) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    tracks = [normalize_track_spec(t) for t in raw.get("tracks", []) if t.get("enabled", True)]
    exact = [t for t in tracks if str(t.get("track_name")) == track_name]
    if exact:
        track = exact[0]
        raw_input = Path(str(track["input_csv"]))
        if not raw_input.is_absolute():
            track["input_csv"] = str((config_path.parent / raw_input).resolve())
        return track
    available = [str(t.get("track_name")) for t in tracks]
    preview = ", ".join(available[:20])
    raise KeyError(f"Track {track_name!r} was not found. First available names: {preview}")


def resolve_existing_path(raw_path: str, atlas_root: str) -> Path:
    direct = Path(raw_path)
    if direct.exists():
        return direct
    # Legacy configurations can be rebased under a user-supplied data root.
    win = PureWindowsPath(raw_path)
    parts_lower = [p.lower() for p in win.parts]
    if "atlas" in parts_lower:
        index = parts_lower.index("atlas")
        tail = win.parts[index + 1 :]
        candidate = Path(atlas_root).joinpath(*tail)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Input table does not exist at {raw_path!r}, and it could not be rebased under {atlas_root!r}."
    )


def sanitize_threadpool_info(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove installation paths while retaining reproducibility metadata."""
    keep = {
        "user_api",
        "internal_api",
        "num_threads",
        "version",
        "threading_layer",
        "architecture",
    }
    return [{key: value for key, value in record.items() if key in keep} for record in records]


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats so output remains strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def infer_separator(path: Path, configured: Any = None) -> Optional[str]:
    if configured:
        return str(configured)
    low = path.name.lower()
    if low.endswith((".tsv", ".tsv.gz", ".bed", ".bed.gz", ".txt", ".txt.gz")):
        return "\t"
    return None


class MemorySampler:
    """Sample RSS for this process and any child processes."""

    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = max(0.01, interval_seconds)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stage_peak = float("nan")
        self._overall_peak = float("nan")
        self._current = float("nan")
        self._psutil = None
        self._process = None
        try:
            import psutil  # type: ignore

            self._psutil = psutil
            self._process = psutil.Process(os.getpid())
        except Exception:
            pass

    @property
    def available(self) -> bool:
        return self._process is not None

    def _rss_mb(self) -> float:
        if self._process is None:
            return float("nan")
        total = 0
        processes = [self._process]
        try:
            processes.extend(self._process.children(recursive=True))
        except Exception:
            pass
        for process in processes:
            try:
                total += process.memory_info().rss
            except Exception:
                continue
        return total / (1024.0**2)

    def _run(self) -> None:
        while not self._stop.is_set():
            value = self._rss_mb()
            with self._lock:
                self._current = value
                if math.isnan(self._stage_peak) or (not math.isnan(value) and value > self._stage_peak):
                    self._stage_peak = value
                if math.isnan(self._overall_peak) or (not math.isnan(value) and value > self._overall_peak):
                    self._overall_peak = value
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        initial = self._rss_mb()
        with self._lock:
            self._current = initial
            self._stage_peak = initial
            self._overall_peak = initial
        if self.available:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def reset_stage(self) -> float:
        value = self._rss_mb()
        with self._lock:
            self._current = value
            self._stage_peak = value
            if math.isnan(self._overall_peak) or (not math.isnan(value) and value > self._overall_peak):
                self._overall_peak = value
        return value

    def stage_values(self, baseline: float) -> Tuple[float, float]:
        value = self._rss_mb()
        with self._lock:
            if math.isnan(self._stage_peak) or (not math.isnan(value) and value > self._stage_peak):
                self._stage_peak = value
            if math.isnan(self._overall_peak) or (not math.isnan(value) and value > self._overall_peak):
                self._overall_peak = value
            peak = self._stage_peak
        delta = peak - baseline if not math.isnan(peak) and not math.isnan(baseline) else float("nan")
        return peak, max(0.0, delta) if not math.isnan(delta) else delta

    def stop(self) -> float:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._lock:
            return self._overall_peak


def load_benchmark_rows(
    pd: Any,
    phase5g: Any,
    track: Dict[str, Any],
    input_path: Path,
    requested_n: int,
    random_state: int,
    kmer: int,
) -> Any:
    read_n = max(requested_n, int(math.ceil(requested_n * 1.15)))
    sep = infer_separator(input_path, track.get("sep"))
    kwargs = {"nrows": read_n, "low_memory": False}
    if sep is not None:
        kwargs["sep"] = sep
    try:
        source = pd.read_csv(input_path, **kwargs)
    except Exception:
        # Final flexible fallback for unusual public-source table formatting.
        source = pd.read_csv(input_path, sep=None, engine="python", nrows=read_n)
    if len(source.columns) == 1 and sep is None:
        source = pd.read_csv(input_path, sep="\t", nrows=read_n, low_memory=False)

    seq_col = phase5g.infer_or_get(source, track.get("seq_col"), phase5g.SEQUENCE_CANDIDATES)
    target_col = phase5g.infer_or_get(source, track.get("target_col"), phase5g.TARGET_CANDIDATES)
    cond_col = phase5g.infer_or_get(
        source,
        track.get("condition_col") or track.get("group_col"),
        phase5g.CONDITION_CANDIDATES,
    )
    if not seq_col or not target_col:
        raise ValueError("The selected table does not contain resolvable sequence and target columns.")

    work = pd.DataFrame(
        {
            "sequence": source[seq_col].map(phase5g.clean_sequence),
            "target": pd.to_numeric(source[target_col], errors="coerce"),
            "group": source[cond_col].astype(str) if cond_col else "ALL",
        }
    )
    work = work[(work["sequence"].str.len() >= kmer) & work["target"].notna()].copy()
    if len(work) > requested_n:
        work = work.sample(n=requested_n, random_state=random_state)
    work = work.reset_index(drop=True)
    if len(work) < 100:
        raise ValueError(f"Only {len(work)} usable rows were available for the requested size {requested_n}.")
    return work


def build_phase5g_features(
    np: Any,
    pd: Any,
    phase5g: Any,
    work: Any,
    train_idx: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    sequences = work["sequence"].tolist()
    targets = work["target"].to_numpy(float)
    groups = work["group"].astype(str).to_numpy()

    vocab = phase5g.build_vocab(args.kmer)
    x_gc = phase5g.gc_len_features(sequences)
    x_mut, _ = phase5g.mut_prior_features(sequences)
    x_full = phase5g.kmer_matrix(sequences, args.kmer, vocab)
    selected_idx = phase5g.select_train_features(
        x_full, targets, train_idx, args.select_top_char6
    )
    x_char = x_full[:, selected_idx]
    x_inter, pairs = phase5g.make_interactions_dense(
        x_char, args.interaction_source_k, args.max_multi_interactions
    )
    x_cond = (
        phase5g.onehot_groups(groups, train_idx)
        if pd.Series(groups).nunique() > 1
        else np.zeros((len(groups), 0), dtype=np.float32)
    )
    x_atlas = phase5g.standardize_dense(
        np.hstack([x_gc, x_char.toarray().astype(np.float32), x_inter, x_cond])
    )
    x_t02 = phase5g.standardize_dense(np.hstack([x_mut, x_atlas]))
    # This reproduces the N09 feature construction in the submitted case-study code.
    x_n09 = phase5g.standardize_dense(np.hstack([x_t02, x_cond]))
    return {
        "x_atlas": x_atlas,
        "x_n09": x_n09,
        "targets": targets,
        "sequences": sequences,
        "n_selected_char6": int(len(selected_idx)),
        "n_interactions": int(len(pairs)),
        "n_context_columns": int(x_cond.shape[1]),
    }


def model_specs(
    np: Any,
    phase5g: Any,
    args: argparse.Namespace,
    features: Dict[str, Any],
) -> List[Tuple[str, str, Any, Any, int]]:
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    selected = {m.strip().lower() for m in args.models.split(",") if m.strip()}
    specs: List[Tuple[str, str, Any, Any, int]] = []
    x_atlas = features["x_atlas"]
    if "ridge" in selected:
        specs.append(
            (
                "Representative_Ridge",
                "Representative linear head",
                make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
                x_atlas,
                args.n_jobs,
            )
        )
    if "extratrees" in selected:
        specs.append(
            (
                "Representative_ExtraTrees",
                "Representative tree-ensemble head",
                ExtraTreesRegressor(
                    n_estimators=args.n_estimators,
                    max_depth=args.tree_max_depth,
                    min_samples_leaf=args.min_samples_leaf,
                    n_jobs=args.n_jobs,
                    random_state=args.random_state,
                ),
                x_atlas,
                1,  # tree-level job parallelism already uses --n-jobs
            )
        )
    if "histgb" in selected:
        specs.append(
            (
                "Representative_HistGradientBoosting",
                "Representative boosting head",
                HistGradientBoostingRegressor(
                    max_iter=args.hgb_max_iter,
                    learning_rate=0.06,
                    max_leaf_nodes=31,
                    l2_regularization=0.01,
                    random_state=args.random_state,
                ),
                x_atlas,
                args.n_jobs,
            )
        )
    if "n09_mlp" in selected:
        specs.append(
            (
                "N09_ATLAS_FullTheory_MLP",
                "Combined grammar/proxy MLP",
                phase5g.make_mlp(
                    args.random_state,
                    tuple(args.hidden_layers),
                    args.mlp_max_iter,
                    0.0001,
                    0.001,
                ),
                features["x_n09"],
                args.n_jobs,
            )
        )
    return specs


def stage_row(
    args: argparse.Namespace,
    requested_n: int,
    actual_n: int,
    repeat: int,
    stage: str,
    model_id: str,
    model_label: str,
    seconds: float,
    baseline_mb: float,
    peak_mb: float,
    delta_mb: float,
    n_train: int,
    n_test: int,
    n_features: int,
    status: str = "ok",
    error: str = "",
) -> Dict[str, Any]:
    return {
        "track_name": args.track_name,
        "requested_n": requested_n,
        "n_rows": actual_n,
        "n_train": n_train,
        "n_test": n_test,
        "repeat": repeat,
        "stage": stage,
        "model_id": model_id,
        "model_label": model_label,
        "n_features": n_features,
        "seconds": seconds,
        "stage_baseline_rss_mb": baseline_mb,
        "stage_peak_rss_mb": peak_mb,
        "stage_incremental_peak_mb": delta_mb,
        "n_jobs_cap": args.n_jobs,
        "status": status,
        "error": error,
    }


def worker_run(args: argparse.Namespace) -> Dict[str, Any]:
    cap_numeric_threads(args.n_jobs)
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.model_selection import train_test_split
    from threadpoolctl import threadpool_info, threadpool_limits

    code_dir = Path(args.code_dir)
    module_path = code_dir / "ATLAS_Evo_Phase5G_ATLAS_MLP_Heads_v1.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Required case-study module was not found: {module_path}")
    phase5g = import_from_path("atlas_evo_phase5g_for_complexity", module_path)

    config_path = Path(args.config_json)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file was not found: {config_path}")
    track = load_track_spec(config_path, args.track_name)
    input_path = resolve_existing_path(str(track["input_csv"]), args.atlas_root)

    sampler = MemorySampler(args.memory_sample_ms / 1000.0)
    sampler.start()
    rows: List[Dict[str, Any]] = []
    requested_n = args.worker_size
    repeat = args.worker_repeat
    work = None
    features: Dict[str, Any] = {}
    train_idx = test_idx = np.asarray([], dtype=int)
    actual_n = 0
    try:
        baseline = sampler.reset_stage()
        started = time.perf_counter()
        work = load_benchmark_rows(
            pd,
            phase5g,
            track,
            input_path,
            requested_n,
            args.random_state + repeat,
            args.kmer,
        )
        actual_n = len(work)
        all_idx = np.arange(actual_n)
        train_idx, test_idx = train_test_split(
            all_idx,
            test_size=args.test_size,
            random_state=args.random_state + repeat,
            shuffle=True,
        )
        elapsed = time.perf_counter() - started
        peak, delta = sampler.stage_values(baseline)
        rows.append(
            stage_row(
                args,
                requested_n,
                actual_n,
                repeat,
                "data_loading_and_cleaning",
                "",
                "",
                elapsed,
                baseline,
                peak,
                delta,
                len(train_idx),
                len(test_idx),
                0,
            )
        )

        baseline = sampler.reset_stage()
        started = time.perf_counter()
        hashes = [hashlib.blake2b(s.encode("ascii", "ignore"), digest_size=16).digest() for s in work["sequence"]]
        duplicate_count = len(hashes) - len(set(hashes))
        elapsed = time.perf_counter() - started
        peak, delta = sampler.stage_values(baseline)
        hash_row = stage_row(
            args,
            requested_n,
            actual_n,
            repeat,
            "sequence_hash_and_exact_duplicate_check",
            "",
            "",
            elapsed,
            baseline,
            peak,
            delta,
            len(train_idx),
            len(test_idx),
            0,
        )
        hash_row["duplicate_rows_detected"] = duplicate_count
        rows.append(hash_row)
        del hashes

        baseline = sampler.reset_stage()
        started = time.perf_counter()
        features = build_phase5g_features(np, pd, phase5g, work, train_idx, args)
        elapsed = time.perf_counter() - started
        peak, delta = sampler.stage_values(baseline)
        feature_row = stage_row(
            args,
            requested_n,
            actual_n,
            repeat,
            "combined_grammar_proxy_feature_construction",
            "",
            "",
            elapsed,
            baseline,
            peak,
            delta,
            len(train_idx),
            len(test_idx),
            int(features["x_n09"].shape[1]),
        )
        feature_row.update(
            {
                "n_selected_char6": features["n_selected_char6"],
                "n_interactions": features["n_interactions"],
                "n_context_columns": features["n_context_columns"],
                "mean_sequence_length": float(work["sequence"].str.len().mean()),
            }
        )
        rows.append(feature_row)

        targets = features["targets"]
        for model_id, label, model, matrix, native_threads in model_specs(np, phase5g, args, features):
            baseline = sampler.reset_stage()
            started = time.perf_counter()
            try:
                with threadpool_limits(limits=max(1, native_threads)):
                    model.fit(matrix[train_idx], targets[train_idx])
                    fit_seconds = time.perf_counter() - started
                    predict_started = time.perf_counter()
                    predictions = model.predict(matrix[test_idx])
                    predict_seconds = time.perf_counter() - predict_started
                peak, delta = sampler.stage_values(baseline)
                row = stage_row(
                    args,
                    requested_n,
                    actual_n,
                    repeat,
                    "model_fit_and_prediction",
                    model_id,
                    label,
                    fit_seconds + predict_seconds,
                    baseline,
                    peak,
                    delta,
                    len(train_idx),
                    len(test_idx),
                    int(matrix.shape[1]),
                )
                row["fit_seconds"] = fit_seconds
                row["predict_seconds"] = predict_seconds
                row["prediction_checksum"] = float(np.mean(predictions))
                rows.append(row)
                del predictions, model
                gc.collect()
            except Exception as exc:
                peak, delta = sampler.stage_values(baseline)
                rows.append(
                    stage_row(
                        args,
                        requested_n,
                        actual_n,
                        repeat,
                        "model_fit_and_prediction",
                        model_id,
                        label,
                        time.perf_counter() - started,
                        baseline,
                        peak,
                        delta,
                        len(train_idx),
                        len(test_idx),
                        int(matrix.shape[1]),
                        status="error",
                        error=repr(exc),
                    )
                )
    finally:
        process_peak = sampler.stop()

    metadata = {
        "track_name": args.track_name,
        "input_file": input_path.name,
        "requested_n": requested_n,
        "actual_n": actual_n,
        "repeat": repeat,
        "process_peak_rss_mb": process_peak,
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "threadpool_info": sanitize_threadpool_info(threadpool_info()),
    }
    for row in rows:
        row["process_peak_rss_mb"] = process_peak
        row["input_file"] = input_path.name
    return {"rows": rows, "metadata": metadata}


def command_for_worker(args: argparse.Namespace, size: int, repeat: int, output_json: Path) -> List[str]:
    script = Path(__file__).resolve()
    command = [
        sys.executable,
        str(script),
        "--config-json",
        args.config_json,
        "--code-dir",
        args.code_dir,
        "--atlas-root",
        args.atlas_root,
        "--track-name",
        args.track_name,
        "--sizes",
        *[str(n) for n in args.sizes],
        "--repeats",
        str(args.repeats),
        "--n-jobs",
        str(args.n_jobs),
        "--outdir",
        args.outdir,
        "--random-state",
        str(args.random_state),
        "--test-size",
        str(args.test_size),
        "--kmer",
        str(args.kmer),
        "--select-top-char6",
        str(args.select_top_char6),
        "--interaction-source-k",
        str(args.interaction_source_k),
        "--max-multi-interactions",
        str(args.max_multi_interactions),
        "--n-estimators",
        str(args.n_estimators),
        "--tree-max-depth",
        str(args.tree_max_depth),
        "--min-samples-leaf",
        str(args.min_samples_leaf),
        "--hgb-max-iter",
        str(args.hgb_max_iter),
        "--mlp-max-iter",
        str(args.mlp_max_iter),
        "--hidden-layers",
        *[str(v) for v in args.hidden_layers],
        "--models",
        args.models,
        "--memory-sample-ms",
        str(args.memory_sample_ms),
        "--worker-size",
        str(size),
        "--worker-repeat",
        str(repeat),
        "--worker-json",
        str(output_json),
    ]
    return command


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    columns: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                columns.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: Sequence[float], q: float) -> float:
    clean = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not clean:
        return float("nan")
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return clean[lower]
    fraction = position - lower
    return clean[lower] * (1.0 - fraction) + clean[upper] * fraction


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (
            row.get("track_name"),
            row.get("n_rows"),
            row.get("stage"),
            row.get("model_id", ""),
            row.get("model_label", ""),
            row.get("n_features", 0),
        )
        groups.setdefault(key, []).append(row)
    summary: List[Dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: (item[0][1], item[0][2], item[0][3])):
        seconds = [r.get("seconds") for r in group]
        stage_memory = [r.get("stage_peak_rss_mb") for r in group]
        process_memory = [r.get("process_peak_rss_mb") for r in group]
        summary.append(
            {
                "track_name": key[0],
                "n_rows": key[1],
                "stage": key[2],
                "model_id": key[3],
                "model_label": key[4],
                "n_features": key[5],
                "n_repeats": len(group),
                "median_seconds": percentile(seconds, 0.5),
                "q1_seconds": percentile(seconds, 0.25),
                "q3_seconds": percentile(seconds, 0.75),
                "median_stage_peak_rss_mb": percentile(stage_memory, 0.5),
                "q1_stage_peak_rss_mb": percentile(stage_memory, 0.25),
                "q3_stage_peak_rss_mb": percentile(stage_memory, 0.75),
                "median_process_peak_rss_mb": percentile(process_memory, 0.5),
                "n_jobs_cap": group[0].get("n_jobs_cap"),
            }
        )
    return summary


def linear_slope_loglog(points: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    clean = [(math.log(x), math.log(y)) for x, y in points if x > 0 and y > 0]
    if len(clean) < 3:
        return float("nan"), float("nan")
    xs = [p[0] for p in clean]
    ys = [p[1] for p in clean]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= 0:
        return float("nan"), float("nan")
    slope = sum((x - x_mean) * (y - y_mean) for x, y in clean) / denom
    intercept = y_mean - slope * x_mean
    predicted = [intercept + slope * x for x in xs]
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, predicted))
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, r2


def scaling_rows(summary: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in summary:
        key = (str(row["stage"]), str(row.get("model_id", "")), str(row.get("model_label", "")))
        grouped.setdefault(key, []).append(row)
    results: List[Dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        points = [(float(r["n_rows"]), float(r["median_seconds"])) for r in group]
        slope, r2 = linear_slope_loglog(points)
        results.append(
            {
                "stage": key[0],
                "model_id": key[1],
                "model_label": key[2],
                "n_sizes": len({int(r["n_rows"]) for r in group}),
                "empirical_time_exponent": slope,
                "loglog_fit_r2": r2,
                "interpretation": "time approximately proportional to n^exponent over the measured range",
            }
        )
    return results


def theoretical_complexity_rows() -> List[Dict[str, str]]:
    return [
        {
            "component": "Track documentation and tier assignment",
            "time_complexity": "O(T + R)",
            "memory_complexity": "O(T + R)",
            "notes": "T tracks and R model-track result records.",
        },
        {
            "component": "Sequence normalization, hashing, and exact duplicate check",
            "time_complexity": "O(n*l)",
            "memory_complexity": "O(n)",
            "notes": "n sequences with mean length l; expected hash-table lookup is O(1).",
        },
        {
            "component": "6-mer counting",
            "time_complexity": "O(n*l)",
            "memory_complexity": "O(nnz) sparse; O(n*d) after dense selection",
            "notes": "d selected grammar features; full 6-mer vocabulary is fixed at 4^6.",
        },
        {
            "component": "Selected interaction construction",
            "time_complexity": "O(n*q)",
            "memory_complexity": "O(n*q)",
            "notes": "q retained multi-6-mer interactions (capped at 512 in the reported configuration).",
        },
        {
            "component": "Condition representation",
            "time_complexity": "O(n*c)",
            "memory_complexity": "O(n*c) dense or O(nnz) sparse",
            "notes": "c observed context categories.",
        },
        {
            "component": "Ridge/linear head",
            "time_complexity": "solver-dependent; approximately O(n*d^2 + d^3) for dense normal-equation solvers",
            "memory_complexity": "O(n*d + d^2)",
            "notes": "Iterative solvers may approach O(iterations*n*d).",
        },
        {
            "component": "Tree-ensemble head",
            "time_complexity": "approximately O(t*n*d*log n)",
            "memory_complexity": "approximately O(t*n)",
            "notes": "t trees; implementation and feature subsampling affect constants.",
        },
        {
            "component": "Combined grammar/proxy MLP head",
            "time_complexity": "O(e*n*sum(h_j*h_(j+1)))",
            "memory_complexity": "O(n*d + sum(h_j*h_(j+1)))",
            "notes": "e training epochs/iterations and layer widths h_j; early stopping can reduce e.",
        },
        {
            "component": "Uncertainty ensemble",
            "time_complexity": "O(m*C_fit)",
            "memory_complexity": "O(C_model + m*n_test)",
            "notes": "m fitted ensemble members and base-model fitting cost C_fit.",
        },
        {
            "component": "Bootstrap confidence interval",
            "time_complexity": "O(B*n_test)",
            "memory_complexity": "O(n_test)",
            "notes": "B bootstrap resamples; excluded from the controlled model-fitting benchmark.",
        },
    ]


def system_metadata(args: argparse.Namespace, worker_metadata: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    cpu_name = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "")
    try:
        import psutil  # type: ignore

        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True)
        ram_gb = psutil.virtual_memory().total / (1024.0**3)
    except Exception:
        physical = None
        logical = os.cpu_count()
        ram_gb = None
    return {
        "generated_local_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "processor": cpu_name,
        "physical_cpu_cores": physical,
        "logical_cpu_cores": logical,
        "total_ram_gb": ram_gb,
        "python": sys.version,
        "n_jobs_cap": args.n_jobs,
        "thread_environment": cap_numeric_threads(args.n_jobs),
        "benchmark_track": args.track_name,
        "requested_sizes": sorted(set(args.sizes)),
        "repeats": args.repeats,
        "models": args.models,
        "case_study_parameters": {
            "kmer": args.kmer,
            "select_top_char6": args.select_top_char6,
            "interaction_source_k": args.interaction_source_k,
            "max_multi_interactions": args.max_multi_interactions,
            "n_estimators": args.n_estimators,
            "tree_max_depth": args.tree_max_depth,
            "hgb_max_iter": args.hgb_max_iter,
            "mlp_max_iter": args.mlp_max_iter,
            "hidden_layers": args.hidden_layers,
        },
        "worker_metadata": list(worker_metadata),
        "important_scope_note": (
            "This controlled benchmark evaluates implementation scaling and resource use. "
            "Its random timing split and selected track are not used for predictive or biological claims."
        ),
    }


def markdown_report(
    args: argparse.Namespace,
    summary: Sequence[Dict[str, Any]],
    scaling: Sequence[Dict[str, Any]],
    failures: Sequence[Dict[str, Any]],
) -> str:
    lines = [
        "# ATLAS-Evo controlled computational-complexity benchmark\n\n",
        "This benchmark measures runtime and process peak memory for representative stages of the "
        "ATLAS-Evo case-study workflow. It is a computational assessment, not a new predictive "
        "performance experiment.\n\n",
        f"- Timing track: `{args.track_name}`\n",
        f"- CPU-worker cap: `{args.n_jobs}`\n",
        f"- Repeats per sample size: `{args.repeats}`\n",
        f"- Requested sizes: `{', '.join(str(n) for n in sorted(set(args.sizes)))}`\n\n",
        "## Median runtime and peak memory\n\n",
        "| Rows | Stage/model | Median time (s) | IQR time (s) | Median stage peak (MB) | Worker peak (MB) |\n",
        "|---:|---|---:|---:|---:|---:|\n",
    ]
    for row in summary:
        label = row.get("model_label") or row.get("stage")
        lines.append(
            f"| {int(row['n_rows'])} | {label} | {float(row['median_seconds']):.4g} | "
            f"{float(row['q1_seconds']):.4g}-{float(row['q3_seconds']):.4g} | "
            f"{float(row['median_stage_peak_rss_mb']):.4g} | "
            f"{float(row['median_process_peak_rss_mb']):.4g} |\n"
        )
    lines.extend(
        [
            "\n## Empirical scaling\n\n",
            "The exponent is obtained from a log-log fit of median wall time against row count. "
            "It describes only the measured range and implementation.\n\n",
            "| Stage/model | Time exponent | Log-log fit R2 |\n",
            "|---|---:|---:|\n",
        ]
    )
    for row in scaling:
        label = row.get("model_label") or row.get("stage")
        exponent = row.get("empirical_time_exponent")
        fit_r2 = row.get("loglog_fit_r2")
        exp_text = "NA" if exponent is None or not math.isfinite(float(exponent)) else f"{float(exponent):.3f}"
        r2_text = "NA" if fit_r2 is None or not math.isfinite(float(fit_r2)) else f"{float(fit_r2):.3f}"
        lines.append(f"| {label} | {exp_text} | {r2_text} |\n")
    lines.extend(
        [
            "\n## Interpretation boundary\n\n",
            "The benchmark uses a random train/test split solely to create fitting and prediction "
            "workloads. No scores from this run should be added to the manuscript's biological or "
            "transfer-facing results. The theoretical complexity table should be interpreted together "
            "with these hardware-dependent measurements.\n",
        ]
    )
    if failures:
        lines.append("\n## Failed worker runs\n\n")
        for failure in failures:
            lines.append(
                f"- requested_n={failure.get('requested_n')}, repeat={failure.get('repeat')}: "
                f"{failure.get('error')}\n"
            )
    return "".join(lines)


def orchestrator_run(args: argparse.Namespace) -> None:
    cap_numeric_threads(args.n_jobs)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    worker_dir = outdir / "worker_records"
    worker_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: List[Dict[str, Any]] = []
    worker_metadata: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    tasks = [(n, repeat) for n in sorted(set(args.sizes)) for repeat in range(1, args.repeats + 1)]
    for index, (size, repeat) in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] rows={size}, repeat={repeat}, workers<= {args.n_jobs}", flush=True)
        output_json = worker_dir / f"worker_n{size}_r{repeat}.json"
        command = command_for_worker(args, size, repeat, output_json)
        environment = os.environ.copy()
        environment.update(cap_numeric_threads(args.n_jobs))
        completed = subprocess.run(command, text=True, capture_output=True, env=environment)
        log_path = worker_dir / f"worker_n{size}_r{repeat}.log"
        log_path.write_text(
            "COMMAND\n" + subprocess.list2cmdline(command) + "\n\nSTDOUT\n" + completed.stdout + "\nSTDERR\n" + completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode != 0 or not output_json.exists():
            failures.append(
                {
                    "requested_n": size,
                    "repeat": repeat,
                    "returncode": completed.returncode,
                    "error": completed.stderr[-4000:] or "Worker did not produce its JSON output.",
                    "log": str(log_path),
                }
            )
            print(f"  failed; see {log_path}", flush=True)
            continue
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        raw_rows.extend(payload.get("rows", []))
        worker_metadata.append(payload.get("metadata", {}))

    if not raw_rows:
        raise SystemExit("No successful benchmark records were produced. Inspect worker_records/*.log.")
    summary = summarize_rows(raw_rows)
    scaling = scaling_rows(summary)
    theory = theoretical_complexity_rows()

    write_csv(outdir / "ATLAS_Evo_complexity_raw.csv", raw_rows)
    write_csv(outdir / "ATLAS_Evo_complexity_summary.csv", summary)
    write_csv(outdir / "ATLAS_Evo_empirical_scaling.csv", scaling)
    write_csv(outdir / "ATLAS_Evo_theoretical_complexity.csv", theory)
    if failures:
        write_csv(outdir / "ATLAS_Evo_complexity_failures.csv", failures)
    metadata = system_metadata(args, worker_metadata)
    (outdir / "ATLAS_Evo_complexity_system_info.json").write_text(
        json.dumps(json_safe(metadata), indent=2, allow_nan=False), encoding="utf-8"
    )
    (outdir / "ATLAS_Evo_complexity_report.md").write_text(
        markdown_report(args, summary, scaling, failures), encoding="utf-8"
    )
    print(f"Completed. Results: {outdir}", flush=True)


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.worker_size > 0:
        if not args.worker_json:
            raise SystemExit("Internal worker mode requires --worker-json.")
        try:
            payload = worker_run(args)
        except Exception as exc:
            payload = {
                "rows": [],
                "metadata": {
                    "requested_n": args.worker_size,
                    "repeat": args.worker_repeat,
                    "status": "error",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
            }
            Path(args.worker_json).write_text(
                json.dumps(json_safe(payload), indent=2, allow_nan=False), encoding="utf-8"
            )
            raise
        Path(args.worker_json).write_text(
            json.dumps(json_safe(payload), indent=2, allow_nan=False), encoding="utf-8"
        )
        return
    orchestrator_run(args)


if __name__ == "__main__":
    main()
