# Result table dictionary

- `full_model_track_metrics.csv`: model-by-track metrics, sample sizes, evidence class, and bootstrap fields where available.
- `bootstrap_ci_proxy.csv`: bootstrap confidence intervals for proxy-level metrics.
- `strict_nonduplicate_results.csv`: rows passing strict nonduplicate criteria.
- `best_model_per_track.csv`: best-performing model per track after audit fields are retained.
- `evidence_hierarchy.csv`: claim-level interpretation table.
- `key_results.csv`: selected benchmark endpoints and interpretation notes.
- `track_provenance.csv`: processed-row, unique-sequence, and target-column metadata by track.
- `duplicate_overlap_audit.csv`: pairwise sequence and identifier overlap audit.
- `duplicate_audit_summary.csv`: compact duplicate-risk summary.
- `ablation_summary.csv`: ablation-level summary across tracks.
- `ablation_by_track.csv`: track-level ablation records.
- `uncertainty_error_summary.csv`: uncertainty-bin error summaries.
- `uncertainty_high_low_summary.csv`: high-uncertainty versus low-uncertainty error comparison.
- `risk_coverage_summary.csv`: optional risk-coverage records where prediction-level data were available.
- `permutation_control_summary.csv`: target-permutation control results.
- `permutation_selected_rows.csv`: selected compact permutation-control rows.
