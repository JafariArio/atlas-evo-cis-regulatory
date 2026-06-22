import argparse
from pathlib import Path
from atlas_evo.table_checks import table_inventory
from atlas_evo.io_utils import write_table

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", default="results/tables")
parser.add_argument("--out-dir", default="outputs")
args = parser.parse_args()

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
inv = table_inventory(args.results_dir)
write_table(inv, out_dir / "table_inventory.csv")
print(f"wrote {out_dir / 'table_inventory.csv'}")
