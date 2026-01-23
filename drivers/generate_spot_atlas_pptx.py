from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import pyarrow  # noqa: F401
import yaml

# Matplotlib: write PNGs without needing a display
import matplotlib

matplotlib.use("Agg")

# Allow running without packaging
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT / "src"))

from qc_spot_atlas import SpotAtlasParams, build_spot_atlas_pptx  # noqa: E402


def _data_root() -> Path:
    root = os.environ.get("BIOIMG_DATA_ROOT")
    if not root:
        raise RuntimeError("BIOIMG_DATA_ROOT is not set (see docs/SETUP_WINDOWS.md)")
    return Path(root).expanduser().resolve()


def _resolve_under_runs_dir(value: str, data_root: Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p.resolve()
    # allow either "<batch_name>" or "runs/<batch_name>"
    if str(p).replace("\\", "/").startswith("runs/"):
        return (data_root / p).resolve()
    return (data_root / "runs" / p).resolve()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML dict: {path}")
    return data


def _load_spots_from_batch_dir(batch_dir: Path) -> pd.DataFrame:
    batch_dir = batch_dir.expanduser().resolve()

    # Prefer the fast path if the integrated driver already wrote an aggregate.
    agg_path = batch_dir / "spots_aggregate.parquet"
    if agg_path.exists():
        df = pd.read_parquet(agg_path)
        # Ensure output_dir exists as a column (needed for manifests)
        if "output_dir" not in df.columns:
            # best effort: if all spots came from a single batch dir, infer per-run output dirs
            df["output_dir"] = ""
        return df

    manifest_path = batch_dir / "batch_manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing batch_manifest.yaml: {manifest_path}")

    bm = _load_yaml(manifest_path)
    runs = bm.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError(f"batch_manifest.yaml missing 'runs' list: {manifest_path}")

    tables = []
    for entry in runs:
        if not isinstance(entry, dict):
            continue
        if entry.get("status") not in {"completed", "skipped"}:
            continue
        out_dir = Path(str(entry.get("output_dir", "")))
        if not out_dir.exists():
            continue
        spots_path = out_dir / "spots.parquet"
        if not spots_path.exists():
            continue
        df = pd.read_parquet(spots_path)
        if df.empty:
            continue
        df["input_path"] = str(entry.get("input_path", ""))
        df["output_dir"] = str(out_dir)
        try:
            df["condition"] = Path(str(entry.get("input_path", ""))).parent.name
        except Exception:
            df["condition"] = ""
        tables.append(df)

    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Generate a MATLAB-style 'spot atlas' PowerPoint for batch QC. "
            "Point this at a batch run folder under $BIOIMG_DATA_ROOT/runs/."
        )
    )
    p.add_argument(
        "--batch-dir",
        type=str,
        required=True,
        help="Batch output folder name under $BIOIMG_DATA_ROOT/runs/ (or an absolute path).",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PPTX path (default: <batch_dir>/qc_spot_atlas_batch.pptx).",
    )
    p.add_argument(
        "--group-by",
        type=str,
        default=None,
        help="Optional grouping column, e.g. condition or spot_channel.",
    )
    p.add_argument("--u0-min", type=float, default=30.0)
    p.add_argument("--spots-per-slide", type=int, default=15)
    p.add_argument(
        "--sort-by",
        type=str,
        default="intensity",
        help="intensity | snr | random | input_path",
    )
    p.add_argument(
        "--fixed-clim",
        type=str,
        default=None,
        help="Optional fixed contrast as 'vmin,vmax' (example: 125,175).",
    )
    p.add_argument(
        "--fixed-percentiles",
        type=str,
        default="1,99.5",
        help="Percentiles for fixed contrast when --fixed-clim is not set (default: 1,99.5).",
    )
    p.add_argument(
        "--nuclei-percentiles",
        type=str,
        default="1,99.8",
        help="Percentiles for nuclei row scaling (default: 1,99.8).",
    )
    args = p.parse_args()

    data_root = _data_root()
    batch_dir = _resolve_under_runs_dir(args.batch_dir, data_root)
    if not batch_dir.exists():
        raise FileNotFoundError(f"batch_dir not found: {batch_dir}")

    spots_df = _load_spots_from_batch_dir(batch_dir)
    if spots_df.empty:
        raise RuntimeError(f"No spots found under batch_dir={batch_dir}")

    fixed_clim = None
    if args.fixed_clim:
        parts = [p.strip() for p in str(args.fixed_clim).split(",")]
        if len(parts) != 2:
            raise ValueError("--fixed-clim must be 'vmin,vmax'")
        fixed_clim = (float(parts[0]), float(parts[1]))

    fp = [p.strip() for p in str(args.fixed_percentiles).split(",")]
    if len(fp) != 2:
        raise ValueError("--fixed-percentiles must be 'pmin,pmax'")
    fixed_percentiles = (float(fp[0]), float(fp[1]))

    npct = [p.strip() for p in str(args.nuclei_percentiles).split(",")]
    if len(npct) != 2:
        raise ValueError("--nuclei-percentiles must be 'pmin,pmax'")
    nuclei_percentiles = (float(npct[0]), float(npct[1]))

    params = SpotAtlasParams(
        spots_per_slide=int(args.spots_per_slide),
        u0_min=float(args.u0_min),
        sort_by=str(args.sort_by),
        fixed_clim=fixed_clim,
        fixed_percentiles=fixed_percentiles,
        nuclei_percentiles=nuclei_percentiles,
    )

    out_path = Path(args.out) if args.out else (batch_dir / "qc_spot_atlas_batch.pptx")
    if not out_path.is_absolute():
        out_path = (batch_dir / out_path).resolve()

    print(f"Loading {len(spots_df)} spots...")
    print(f"Writing PPTX: {out_path}")

    build_spot_atlas_pptx(
        spots_df,
        out_pptx=out_path,
        params=params,
        group_by=args.group_by,
        deck_title=batch_dir.name,
    )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
