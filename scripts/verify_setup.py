from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path


# Modules required for integrated workflow + file I/O
REQUIRED_IMPORTS = [
    "numpy",
    "pandas",
    "tifffile",
    "yaml",
    "pyarrow",
    "skimage",
    "matplotlib",
    "h5py",  # needed for reading Imaris .ims
    "stardist",
    "csbdeep",
    "tensorflow",
]


def try_git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        return out
    except Exception:
        return None


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    print("=== bioimg-pipeline :: verify_setup ===")
    print(f"repo_root: {repo_root}")
    print(f"python: {sys.executable}")
    print(f"python_version: {sys.version.split()[0]}")
    print(f"conda_env: {os.environ.get('CONDA_DEFAULT_ENV', '(not set)')}")

    # 1) Env var
    data_root = os.environ.get("BIOIMG_DATA_ROOT")
    if not data_root:
        print("ERROR: BIOIMG_DATA_ROOT is not set.")
        print(
            r"Fix (PowerShell): [Environment]::SetEnvironmentVariable('BIOIMG_DATA_ROOT','D:\bioimg-data','User')"
        )
        return 2

    data_root = Path(data_root)
    print(f"BIOIMG_DATA_ROOT: {data_root}")
    if not data_root.exists():
        print(f"ERROR: BIOIMG_DATA_ROOT path does not exist: {data_root}")
        return 2

    # 2) Expected bench folders
    expected = ["raw_staging", "runs", "cache", "models"]
    for name in expected:
        p = data_root / name
        if not p.exists():
            print(f"ERROR: missing bench folder: {p}")
            return 2
    print("bench folders: OK")

    # 3) Required repo files
    required_repo_files = [
        "environment.yml",
        "docs/CONTRACTS.md",
        "configs/integrated_sim.yaml",
        "configs/integrated_ims.yaml",
    ]
    for rel in required_repo_files:
        p = repo_root / rel
        if not p.exists():
            print(f"ERROR: missing repo file: {p}")
            return 2
    print("repo files: OK")

    # 4) Required imports
    for mod in REQUIRED_IMPORTS:
        try:
            __import__(mod)
            print(f"import {mod}: OK")
        except Exception as e:
            print(f"ERROR: import {mod} failed: {e}")
            return 2

    # 5) Write test to runs/
    test_dir = data_root / "runs" / "_setup_write_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "touch.txt").write_text("ok\n", encoding="utf-8")
    print(f"\nwrite test: OK ({test_dir})")

    # 6) Git commit (optional)
    commit = try_git_commit(repo_root)
    if commit:
        print(f"git_commit: {commit[:12]}")
    else:
        print("git_commit: (unavailable in this shell)")

    print("SETUP OK ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
