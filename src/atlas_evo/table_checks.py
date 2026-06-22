from pathlib import Path
import pandas as pd


def table_inventory(directory):
    rows = []
    for path in sorted(Path(directory).glob("*.csv")):
        df = pd.read_csv(path)
        rows.append({"file": path.name, "rows": len(df), "columns": len(df.columns)})
    return pd.DataFrame(rows)
