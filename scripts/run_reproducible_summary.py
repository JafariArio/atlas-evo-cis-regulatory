from pathlib import Path
import subprocess
import sys

commands = [
    [sys.executable, "scripts/00_check_environment.py"],
    [sys.executable, "scripts/11_export_result_tables.py", "--results-dir", "results/tables", "--out-dir", "outputs"],
]
for cmd in commands:
    print("running", " ".join(cmd))
    subprocess.check_call(cmd)
print("summary complete")
