# ATLAS-Evo cis-regulatory benchmark

This repository contains code, configuration files, and result tables for the ATLAS-Evo public-data benchmark for condition-aware cis-regulatory prediction.

## What ATLAS-Evo is

ATLAS-Evo is an evidence-tiered benchmarking and reporting framework rather than a single prediction architecture. It defines how dataset branches are documented, checked for duplication and relatedness, evaluated using selected candidate models, assigned to evidence tiers, and interpreted according to the claims supported by those tiers.

The grammar, theory-inspired proxy, external-comparator, and nonlinear model families included in this repository form the empirical case study. They are not mandatory components of the ATLAS-Evo framework and may be replaced by other model families.

## How ATLAS-Evo is used

The workflow follows five stages:

**Track definition → Provenance and relatedness checks → Candidate model evaluation → Evidence-tier assignment → Claim-level interpretation**

Users first define each sequence–target branch and document its source, target construction, biological or assay context, and data split. They then check sequence and identifier overlap, target reuse, and branch relationships; evaluate their selected candidate models; assign each result to a prespecified evidence tier; and report performance according to the scope of that tier.

## Core deliverables

ATLAS-Evo produces:

* Track-level provenance records
* Duplicate, overlap, target-reuse, and relatedness checks
* Evidence-tier assignments
* Tier-aware model–track performance tables
* Control and sensitivity summaries
* An explicit boundary between transfer-facing evidence and supporting model-capacity results

## Evidence categories

Results are interpreted within four broad evidence categories:

* Transfer-facing independent/context evidence
* Related-branch or proxy support
* Model- or proxy-capacity evidence
* Prediction-derived, exploratory, or boundary evidence

These categories prevent related, duplicated, prediction-derived, or capacity-oriented results from being interpreted automatically as independent transfer.

## Contents

* `src/atlas_evo/`: reusable Python utilities for sequence features, metrics, evidence-tier checks, duplicate and relatedness checks, ablation summaries, uncertainty summaries, and target-permutation controls.
* `scripts/`: command-line entry points for checking the environment and regenerating result summaries from tabular inputs.
* `configs/`: track, model, proxy, and evidence-tier dictionaries.
* `data/`: public dataset links, processed metadata, and small demonstration inputs.
* `results/tables/`: CSV result tables from the benchmark and control analyses.
* `tests/`: lightweight tests for the core utility functions.

## Quick start

```bash
python -m pip install -r requirements.txt
python scripts/00_check_environment.py
python scripts/11_export_result_tables.py --results-dir results/tables --out-dir outputs
```

## Scope

The repository supports reproduction and inspection of the benchmark workflow and its reported summaries. Public raw datasets remain available from their original sources, and large raw sequence tables are not included in this repository.

## Core interpretation rule

The highest numerical score is not automatically the strongest evidence. Independent/context evidence, related-branch or proxy support, model-capacity results, and prediction-derived or exploratory analyses must be interpreted separately.
