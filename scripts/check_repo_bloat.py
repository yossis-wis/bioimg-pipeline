#!/usr/bin/env python3
"""Check that the git-tracked repo stays small + text-only.

Why this exists
---------------
This repo is frequently *flattened and attached to LLM prompts*, so large tracked
assets (especially exploded Matplotlib SVGs or accidentally-committed PNG/PDFs)
quickly become painful.

This script is intentionally conservative:
- it fails if a tracked file looks binary (by extension) or is larger than a
  configurable size budget.
- it prints the top-N largest tracked files for quick cleanup.

Typical use
-----------
    python scripts/check_repo_bloat.py

Optional:
    python scripts/check_repo_bloat.py --max-kb 250 --top 30
"""  # noqa: D401

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


# Conservative set of binary / archive extensions that should not be tracked.
BANNED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".pdf",
    ".pptx",
    ".docx",
    ".xlsx",
    ".zip",
    ".7z",
    ".tar",
    ".gz",
    ".parquet",
    ".feather",
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".mat",
    ".bin",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".ipynb",  # notebooks should be Jupytext .py
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_ls_files(repo_root: Path) -> list[str]:
    try:
        out = subprocess.check_output(["git", "ls-files", "-z"], cwd=str(repo_root))
    except Exception as exc:
        print(f"ERROR: not a git repo (or git unavailable): {exc}")
        return []
    parts = out.split(b"\x00")
    files: list[str] = []
    for p in parts:
        if not p:
            continue
        try:
            files.append(p.decode("utf-8"))
        except Exception:
            files.append(p.decode("utf-8", errors="replace"))
    return files


def _iter_sizes(repo_root: Path, relpaths: Iterable[str]) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for rp in relpaths:
        p = repo_root / rp
        try:
            sz = p.stat().st_size
        except FileNotFoundError:
            # Should not happen for tracked files, but guard anyway.
            continue
        rows.append((int(sz), rp))
    rows.sort(key=lambda t: t[0])
    return rows


def _ext_lower(path: str) -> str:
    return Path(path).suffix.lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--max-kb",
        type=int,
        default=250,
        help="Fail if any tracked file is larger than this many KB (default: 250)",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=20,
        help="Show the top-N largest tracked files (default: 20)",
    )
    args = ap.parse_args()

    repo_root = _repo_root()
    tracked = _git_ls_files(repo_root)
    if not tracked:
        print("No tracked files found (or not a git repo).")

        return 2

    max_bytes = int(args.max_kb) * 1024

    sizes = _iter_sizes(repo_root, tracked)
    total_bytes = sum(sz for sz, _ in sizes)

    violations: list[str] = []

    for sz, rp in sizes:
        ext = _ext_lower(rp)
        if ext in BANNED_EXTENSIONS:
            violations.append(f"BINARY EXT: {rp} ({sz/1024:.1f} KB)")
        elif sz > max_bytes:
            violations.append(f"TOO LARGE: {rp} ({sz/1024:.1f} KB > {args.max_kb} KB)")

    print("=== check_repo_bloat ===")
    print(f"repo_root: {repo_root}")
    print(f"tracked_files: {len(tracked)}")
    print(f"total_tracked_size: {total_bytes/1024:.1f} KB")
    print(f"budget_max_file: {args.max_kb} KB")
    print()

    print(f"Top {int(args.top)} largest tracked files:")
    for sz, rp in list(reversed(sizes))[: int(args.top)]:
        print(f"  {sz:>9d}  {rp}")
    print()

    if violations:
        print("VIOLATIONS:")
        for v in violations:
            print(f"- {v}")
        print()
        print("Suggested cleanup:")
        print("- If a binary file is already tracked, untrack it with:")
        print("    git rm --cached <path>  # keep local copy\n")
        print("- For large generated figures, prefer:")
        print("    docs/figures/generated/   (gitignored)")
        print("  or $BIOIMG_DATA_ROOT/reports/\n")
        print("- Then commit the removal and rerun this script.")
        return 2

    print("OK: repo looks small + text-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
