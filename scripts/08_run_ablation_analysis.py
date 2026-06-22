import argparse
from atlas_evo.io_utils import read_table, write_table
from atlas_evo.ablation_analysis import summarize_ablation

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

df = read_table(args.input)
summary = summarize_ablation(df)
write_table(summary, args.out)
print(f"wrote {args.out}")
