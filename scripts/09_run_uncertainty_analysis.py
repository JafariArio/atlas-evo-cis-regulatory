import argparse
from atlas_evo.io_utils import read_table, write_table

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

df = read_table(args.input)
cols = [c for c in df.columns if c in {"track_name", "model_name", "highest_minus_lowest_abs_error", "spearman_bin_uncertainty_abs_error", "total_n"}]
summary = df[cols].copy() if cols else df.copy()
write_table(summary, args.out)
print(f"wrote {args.out}")
