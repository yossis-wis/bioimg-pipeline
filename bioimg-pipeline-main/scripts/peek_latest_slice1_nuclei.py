"""Peek the nuclei table from the most recent Slice1 run.

Usage:
    conda activate bioimg-slice0
    python scripts/peek_latest_slice1_nuclei.py

This avoids having to manually copy the timestamped run folder name.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def _data_root() -> Path:
    root = os.environ.get("BIOIMG_DATA_ROOT")
    if not root:
        raise RuntimeError("BIOIMG_DATA_ROOT is not set")
    return Path(root).expanduser().resolve()


def _latest_run_dir(runs_dir: Path) -> Path:
    candidates = sorted(runs_dir.glob("*__slice1_nuclei"), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No Slice1 runs found in: {runs_dir}")
    return candidates[0]


def main() -> int:
    data_root = _data_root()
    runs_dir = (data_root / "runs").resolve()

    run_dir = _latest_run_dir(runs_dir)
    parquet_path = run_dir / "nuclei.parquet"
    csv_path = run_dir / "nuclei.csv"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        src = parquet_path
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
        src = csv_path
    else:
        raise FileNotFoundError(
            f"No nuclei table found in {run_dir}. Did you set write_nuclei_table: true?"
        )

    print(f"Latest Slice1 run: {run_dir}")
    print(f"Loaded: {src.name}")
    print(f"Rows: {len(df)}\nColumns: {list(df.columns)}\n")
    print(df.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
