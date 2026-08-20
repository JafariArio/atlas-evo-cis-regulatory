# ATLAS-Evo cis-regulatory benchmark

This repository contains code, configuration files, and result tables for the ATLAS-Evo public-data benchmark for condition-aware cis-regulatory prediction.

## What ATLAS-Evo is

ATLAS-Evo is an evidence-tiered benchmarking and reporting framework rather than a single prediction architecture. It defines how dataset branches are documented, checked for duplication and relatedness, evaluated using selected candidate models, assigned to evidence tiers, and interpreted according to the claims supported by those tiers.

The grammar, theory-inspired proxy, external-comparator, and nonlinear model families included in this repository form the empirical case study. They are not mandatory components of the ATLAS-Evo framework and may be replaced by other model families.

## How ATLAS-Evo is used

The workflow follows five stages:

**Track definition → Provenance and relatedness checks → Candidate model evaluation → Evidence-tier assignment → Claim-level interpretation**

Users first define each sequence-target branch and document its source, target construction, biological or assay context, and data split. They then check sequence and identifier overlap, target reuse, and branch relationships; evaluate their selected candidate models; assign each result to a prespecified evidence tier; and report performance according to the scope of that tier.

## Core deliverables

ATLAS-Evo produces:

- Track-level provenance records
- Duplicate, overlap, target-reuse, and relatedness checks
- Evidence-tier assignments
- Tier-aware model-track performance tables
- Control and sensitivity summaries
- An explicit boundary between transfer-facing evidence and supporting model-capacity results

## Evidence categories

Results are interpreted within four broad evidence categories:

- Transfer-facing independent/context evidence
- Related-branch or proxy support
- Model- or proxy-capacity evidence
- Prediction-derived, exploratory, or boundary evidence

These categories prevent related, duplicated, prediction-derived, or capacity-oriented results from being interpreted automatically as independent transfer.

## Contents

- `src/atlas_evo/`: reusable Python utilities for sequence features, metrics, evidence-tier checks, duplicate and relatedness checks, ablation summaries, uncertainty summaries, and target-permutation controls.
- `scripts/`: command-line entry points for checking the environment and regenerating result summaries from tabular inputs.
- `scripts/12_benchmark_complexity.py`: controlled runtime and memory benchmark used for the computational-complexity assessment.
- `scripts/case_study/`: case-study feature and prediction-head functions required by the complexity benchmark.
- `configs/`: track, model, proxy, evidence-tier, and complexity-benchmark configurations.
- `data/`: public dataset links, processed metadata, and small demonstration inputs.
- `results/tables/`: CSV result tables from the benchmark, control analyses, and complexity assessment.
- `results/metadata/`: sanitized computational-environment metadata.
- `docs/`: reproduction, data-availability, and computational-complexity documentation.
- `tests/`: lightweight tests for the core utility and benchmark functions.

## Quick start

```bash
python -m pip install -r requirements.txt
python scripts/00_check_environment.py
python scripts/11_export_result_tables.py --results-dir results/tables --out-dir outputs
```

## Computational-complexity benchmark

Run the included smoke test with the deterministic synthetic input:

```bash
python scripts/13_generate_complexity_demo.py
python scripts/12_benchmark_complexity.py
```

The reported controlled benchmark used 5,000, 10,000, 25,000, and 50,000 sequences, three timing runs per size, and an eight-worker limit. A full run can be reproduced with a model-ready public-data table and its track configuration:

```bash
python scripts/12_benchmark_complexity.py \
  --config-json path/to/EFFECTIVE_TRACKS_CONFIG.json \
  --atlas-root path/to/ATLAS \
  --track-name Random_Promoter_DREAM_base_track_2 \
  --sizes 5000 10000 25000 50000 \
  --repeats 3 \
  --n-jobs 8 \
  --n-estimators 160 \
  --hgb-max-iter 220 \
  --mlp-max-iter 250 \
  --hidden-layers 256 64 \
  --models ridge,extratrees,histgb,n09_mlp \
  --outdir outputs/complexity
```

This benchmark assesses implementation scaling and resource use only. Its random timing split and predictions are not used for biological, predictive-performance, or transfer-facing claims. The methods and reported values are provided in `docs/computational_complexity.md`.

## Scope

The repository supports reproduction and inspection of the benchmark workflow and its reported summaries. Public raw datasets remain available from their original sources, and large raw sequence tables are not included in this repository.

## Core interpretation rule

The highest numerical score is not automatically the strongest evidence. Independent/context evidence, related-branch or proxy support, model-capacity results, and prediction-derived or exploratory analyses must be interpreted separately.
