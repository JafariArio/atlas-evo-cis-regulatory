import argparse
from atlas_evo.io_utils import read_table, write_table

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--min-n", type=int, default=100)
args = parser.parse_args()

df = read_table(args.input)
if "n_used" in df.columns:
    df = df[df["n_used"] >= args.min_n].copy()
cols = [c for c in ["track_name", "model_name", "n_used", "observed_r2", "null_r2_median", "observed_minus_null_median_r2", "empirical_p_null_r2_ge_observed", "n_permutations"] if c in df.columns]
write_table(df[cols], args.out)
print(f"wrote {args.out}")
