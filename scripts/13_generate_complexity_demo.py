#!/usr/bin/env python3
"""Generate a deterministic synthetic smoke-test input for the complexity runner."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=300)
    parser.add_argument("--length", type=int, default=110)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "demo" / "complexity_demo.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows < 200 or args.length < 12:
        raise SystemExit("Use at least 200 rows and sequence length 12.")
    rng = random.Random(42)
    rows = []
    for index in range(args.rows):
        sequence = "".join(rng.choice("ACGT") for _ in range(args.length))
        gc = (sequence.count("G") + sequence.count("C")) / len(sequence)
        motif = sequence.count("TATAAA") + 0.5 * sequence.count("CG")
        target = 1.8 * gc + 0.08 * motif + rng.gauss(0.0, 0.05)
        rows.append(
            {
                "id": f"demo_{index:04d}",
                "sequence": sequence,
                "condition": "demo",
                "split": "timing_only",
                "target": f"{target:.8f}",
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
