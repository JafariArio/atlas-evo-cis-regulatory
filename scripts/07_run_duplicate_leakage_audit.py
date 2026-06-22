import argparse
from atlas_evo.io_utils import read_table, write_table
from atlas_evo.duplicate_leakage_audit import summarize_duplicate_risk

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

df = read_table(args.input)
summary = summarize_duplicate_risk(df)
write_table(summary, args.out)
print(f"wrote {args.out}")
