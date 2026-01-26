#!/usr/bin/env python3
"""Inspect XY pixel size (nm/px) from TIFF or Imaris .ims metadata.

This is a lightweight helper that prints a suggested config entry:

  spot_pixel_size_nm: <value>

Notes
-----
- Supports OME-TIFF (reads OME-XML PhysicalSizeX/Y).
- Supports Imaris .ims (HDF5) via DataSetInfo/Image extents.
- Imaris sometimes stores numeric attributes as arrays of single-character strings (dtype |S1).
  This script handles that case robustly.

Paths
-----
If --input is not absolute, we try (in order):

1) As given, relative to the current working directory.
2) Under --data-root, if provided.
3) Under $BIOIMG_DATA_ROOT, if set.
4) Under $BIOIMG_DATA_ROOT/raw_staging/ (common convention).

"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional


def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(10):
        if (cur / "src").exists() and (cur / "scripts").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve()


REPO_ROOT = _find_repo_root(Path(__file__).parent)
sys.path.insert(0, str(REPO_ROOT / "src"))

from pixel_size_utils import infer_pixel_size_nm  # noqa: E402


def _resolve_input_path(raw: str, *, data_root: Optional[Path]) -> Path:
    p = Path(raw).expanduser()

    tried: list[Path] = []

    # 1) Absolute path
    if p.is_absolute():
        tried.append(p)
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"Input not found (absolute): {p}")

    # 2) Relative to CWD
    tried.append(p)
    if p.exists():
        return p.resolve()

    # 3) Under explicit data_root / env data_root
    if data_root is not None:
        cand = (data_root / p).resolve()
        tried.append(cand)
        if cand.exists():
            return cand

        # 4) Convenience fallback: raw_staging/<basename>
        cand2 = (data_root / "raw_staging" / p.name).resolve()
        tried.append(cand2)
        if cand2.exists():
            return cand2

    msg = "Input not found. Tried:\n" + "\n".join([f"  - {t}" for t in tried])
    raise FileNotFoundError(msg)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Inspect XY pixel size (nm/px) from TIFF or Imaris .ims metadata.",
    )
    ap.add_argument(
        "--input",
        required=True,
        help="Input image path (.tif/.tiff/.ims). If relative, resolved using --data-root or $BIOIMG_DATA_ROOT.",
    )
    ap.add_argument(
        "--data-root",
        default=None,
        help="Optional data root. If not set, uses $BIOIMG_DATA_ROOT.",
    )
    args = ap.parse_args()

    data_root: Optional[Path] = None
    if args.data_root:
        data_root = Path(args.data_root).expanduser().resolve()
    else:
        env_root = os.environ.get("BIOIMG_DATA_ROOT")
        if env_root:
            data_root = Path(env_root).expanduser().resolve()

    p = _resolve_input_path(args.input, data_root=data_root)

    py_nm, px_nm, note = infer_pixel_size_nm(p)

    print("=== inspect_pixel_size ===")
    print("input:", p)
    print("format:", p.suffix.lower())
    print("note:", note)
    print()

    if py_nm is None or px_nm is None:
        print("ERROR: could not determine pixel size from metadata.")
        print("You can still set spot_pixel_size_nm manually if you know the calibration.")
        return 2

    mean_nm = 0.5 * (float(py_nm) + float(px_nm))

    print(f"pixel_size_y: {float(py_nm):.3f} nm/px")
    print(f"pixel_size_x: {float(px_nm):.3f} nm/px")
    print(f"pixel_size_xy_mean: {mean_nm:.3f} nm/px")
    print()
    print("Suggested config entry (nm/px):")
    print(f"  spot_pixel_size_nm: {mean_nm:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
