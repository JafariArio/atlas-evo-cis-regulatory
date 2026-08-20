# ATLAS-Evo cis-regulatory benchmark

This repository contains code, configuration files, and neutral result tables for the ATLAS-Evo public-data benchmark for condition-aware cis-regulatory sequence modeling.

ATLAS-Evo is organized as an evidence-tiered benchmarking and reporting framework rather than as a single prediction architecture. It documents benchmark branches, checks duplication and relatedness, evaluates selected candidate models, assigns results to evidence tiers, and interprets performance within those tiers. Its purpose is to keep independent/context evidence separate from related-branch support, prediction-derived capacity checks, boundary cases, and control analyses. The grammar, theory-inspired proxy, external-comparator, and nonlinear model families included here form the empirical case study and are not mandatory components of the framework. The repository is intended for reproducibility and inspection of the benchmark workflow, not as a public archive of submission files.

## Contents

* `src/atlas_evo/`: reusable Python utilities for sequence features, metrics, evidence-tier checks, duplicate and relatedness checks, ablation summaries, uncertainty summaries, and target-permutation controls.
* `scripts/`: command-line entry points for checking the environment and regenerating result summaries from tabular inputs.
* `configs/`: track, model, proxy, and evidence-tier dictionaries.
* `data/`: public dataset links, processed metadata, and small demo inputs.
* `results/tables/`: neutral CSV result tables from the benchmark and control analyses.
* `tests/`: lightweight tests for core utility functions.

## Quick start

```bash
python -m pip install -r requirements.txt
python scripts/00_check_environment.py
python scripts/11_export_result_tables.py --results-dir results/tables --out-dir outputs
```

## Scope

The repository reports benchmark structure and reproducible summaries. Public raw datasets remain at their original public sources. Large raw sequence tables are not bundled here.

## Core interpretation rule

The highest numerical score is not automatically the strongest evidence. Independent/context evidence, related/proxy support, prediction-derived capacity checks, and boundary analyses must be interpreted separately.

