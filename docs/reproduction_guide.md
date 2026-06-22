# Reproduction guide

## 1. Install environment

```bash
python -m pip install -r requirements.txt
python scripts/00_check_environment.py
```

## 2. Inspect dictionaries

Track and model dictionaries are in `configs/`. Evidence-tier rules are in `configs/evidence_tier_rules.yaml`.

## 3. Recompute summary checks

```bash
python scripts/11_export_result_tables.py --results-dir results/tables --out-dir outputs
python scripts/07_run_duplicate_leakage_audit.py --input results/tables/duplicate_overlap_audit.csv --out outputs/duplicate_audit_summary.csv
python scripts/08_run_ablation_analysis.py --input results/tables/ablation_by_track.csv --out outputs/ablation_summary.csv
python scripts/09_run_uncertainty_analysis.py --input results/tables/uncertainty_bins.csv --out outputs/uncertainty_summary.csv
python scripts/10_run_permutation_control.py --input results/tables/permutation_control_summary.csv --out outputs/permutation_summary.csv
```

These commands operate on tabular benchmark outputs. Full raw-data reruns require downloading the original public datasets.
