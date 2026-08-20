# Computational complexity and controlled runtime assessment

ATLAS-Evo is a benchmarking and reporting framework rather than a single predictive architecture. Its computational cost is therefore reported by workflow stage and representative case-study model family.

## Interpretation boundary

The controlled run uses a random 80:20 split only to create model-fitting and prediction workloads. It does not generate or revise biological, predictive-performance, or transfer-facing results.

## Theoretical complexity

Let `T` be the number of tracks, `R` the number of model-track records, `n` the number of sequences, `l` mean sequence length, `d` retained features, `q` selected interactions, `c` context categories, `t` trees, `e` MLP iterations, and `h_j` successive layer widths.

| Component | Time complexity | Memory complexity |
|---|---|---|
| Track documentation and tier assignment | `O(T + R)` | `O(T + R)` |
| Sequence normalization, hashing, and exact-duplicate checking | `O(n*l)` | `O(n)` |
| 6-mer counting | `O(n*l)` | `O(nnz)` sparse or `O(n*d)` dense |
| Selected interaction construction | `O(n*q)` | `O(n*q)` |
| Context representation | `O(n*c)` | `O(n*c)` dense or `O(nnz)` sparse |
| Linear/Ridge head | Solver-dependent; approximately `O(n*d^2 + d^3)` | `O(n*d + d^2)` |
| Tree-ensemble head | Approximately `O(t*n*d*log n)` | Approximately `O(t*n)` |
| Combined grammar/proxy MLP | `O(e*n*sum(h_j*h_(j+1)))` | `O(n*d + sum(h_j*h_(j+1)))` |
| Uncertainty ensemble with `m` members | `O(m*C_fit)` | `O(C_model + m*n_test)` |
| Bootstrap interval with `B` replicates | `O(B*n_test)` | `O(n_test)` |

## Controlled benchmark configuration

- Timing branch: `Random_Promoter_DREAM_base_track_2`
- Sequence counts: 5,000, 10,000, 25,000, and 50,000
- Repeats: three per size
- Split: random 80:20 timing split
- Worker limit: eight
- Hardware: Windows 11 workstation, 8 physical cores, 16 logical processors, and 31.3 GB RAM
- Software: Python 3.13.12, NumPy 2.4.4, pandas 3.0.1, and scikit-learn 1.8.0
- Feature settings: 6-mers, 128 retained 6-mers, 64 interaction sources, and at most 512 interactions
- Model settings: 160 ExtraTrees estimators; histogram-gradient-boosting maximum 220 iterations; MLP layers 256 and 64 with maximum 250 iterations

## Reported measurements

Values are medians with interquartile ranges from three timing runs.

| Rows | Feature construction (s) | Linear head (s) | ExtraTrees (s) | Boosting (s) | Combined grammar/proxy MLP (s) | Process peak memory |
|---:|---:|---:|---:|---:|---:|---:|
| 5,000 | 4.14 (4.14-4.64) | 0.110 (0.108-0.142) | 3.10 (2.98-3.60) | 6.12 (5.81-6.72) | 10.00 (6.60-10.06) | 311 MB |
| 10,000 | 8.67 (8.61-8.71) | 0.139 (0.138-0.140) | 6.73 (6.72-6.89) | 5.42 (5.37-5.43) | 17.11 (13.42-21.49) | 392 MB |
| 25,000 | 23.42 (23.15-25.68) | 0.286 (0.275-0.288) | 33.79 (33.54-34.37) | 7.65 (7.51-7.88) | 29.52 (27.62-34.22) | 691 MB |
| 50,000 | 55.61 (55.57-55.67) | 0.519 (0.508-0.527) | 100.32 (100.06-100.46) | 11.00 (10.89-11.19) | 64.94 (59.03-83.25) | 1.10 GB |

Feature construction had an empirical time exponent of 1.12 (`R^2 = 0.999`) over the measured range. ExtraTrees had the strongest observed increase, with an exponent of 1.54 (`R^2 = 0.994`). These exponents describe the tested implementation, hardware, sample-size range, and fixed parameter settings; they are not asymptotic guarantees.

## Repository outputs

- `results/tables/computational_complexity_summary.csv`
- `results/tables/computational_complexity_empirical_scaling.csv`
- `results/tables/computational_complexity_theoretical.csv`
- `results/tables/computational_complexity_si_table.csv`
- `results/metadata/computational_complexity_system_info.json`

Raw public sequence tables are not included. Local worker logs and absolute filesystem paths should not be committed.
