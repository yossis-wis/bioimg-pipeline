from __future__ import annotations

"""Slice1 driver: nucleus segmentation (StarDist).

This driver is intentionally analogous to drivers/run_slice0.py:
- resolve $BIOIMG_DATA_ROOT
- load a single 2D image plane (TIFF or .ims)
- load a StarDist model (disk I/O)
- run segmentation kernel
- write run folder artifacts + manifest

Outputs (run folder)
--------------------
Required:
- nuclei_labels.tif
- qc_overlay.png
- run_manifest.yaml

Optional (future use):
- nuclei.parquet (region properties)
"""

import argparse
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import tifffile
import yaml

# Matplotlib: write PNGs without needing a display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from skimage.segmentation import find_boundaries  # noqa: E402
from skimage.measure import regionprops_table  # noqa: E402

# Allow running without packaging
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT / "src"))

from image_io import PlaneSelection, read_image_2d  # noqa: E402
from stardist_utils import StardistModelRef, load_stardist2d, get_model_thresholds  # noqa: E402
from slice1_nuclei_kernel import Slice1NucleiParams, segment_nuclei_stardist  # noqa: E402


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _data_root() -> Path:
    root = os.environ.get("BIOIMG_DATA_ROOT")
    if not root:
        raise RuntimeError("BIOIMG_DATA_ROOT is not set (see docs/SETUP_WINDOWS.md)")
    return Path(root).expanduser().resolve()


def _try_git_commit(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def _resolve_model_dir(cfg: Dict[str, Any], data_root: Path) -> Path:
    """Resolve the StarDist model folder.

    Config options (choose ONE):
      - stardist_model_dir: path (absolute OR relative to $BIOIMG_DATA_ROOT)

    Recommended convention:
      $BIOIMG_DATA_ROOT/models/<model_name>/
    """

    model_dir_raw = cfg.get("stardist_model_dir")
    if not model_dir_raw:
        raise ValueError(
            "Config must set stardist_model_dir (absolute or relative to $BIOIMG_DATA_ROOT)."
        )

    model_dir = Path(str(model_dir_raw))
    if not model_dir.is_absolute():
        model_dir = (data_root / model_dir).resolve()
    return model_dir


def _write_qc_overlay(image_2d: np.ndarray, labels: np.ndarray, out_png: Path) -> None:
    """Write a simple QC overlay: boundaries (red) on image."""

    b = find_boundaries(labels > 0, mode="outer")

    vmin, vmax = np.percentile(image_2d.astype(np.float32), [1, 99])
    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = fig.add_subplot(111)
    ax.imshow(image_2d, cmap="gray", vmin=vmin, vmax=vmax)

    # Masked overlay so background is transparent
    overlay = np.ma.masked_where(~b, b)
    ax.imshow(overlay, cmap="Reds", alpha=0.8)

    ax.set_axis_off()
    ax.set_title(f"Slice1 nuclei: {int(labels.max())} instances")
    fig.tight_layout(pad=0)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def run_slice1_nuclei(config_path: Path) -> Path:
    cfg = _load_config(config_path)
    data_root = _data_root()

    input_relpath = cfg.get("input_relpath")
    if not input_relpath:
        raise ValueError("Config must set input_relpath")

    input_path = (data_root / str(input_relpath)).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Output folder
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runs_dir = (data_root / str(cfg.get("output_runs_dir", "runs"))).resolve()
    out_dir = runs_dir / f"{stamp}__slice1_nuclei"
    out_dir.mkdir(parents=True, exist_ok=False)

    # Read a single 2D plane
    sel = PlaneSelection(
        channel=int(cfg.get("channel", 1)),
        ims_resolution_level=int(cfg.get("ims_resolution_level", 0)),
        ims_time_index=int(cfg.get("ims_time_index", 0)),
        ims_z_index=int(cfg.get("ims_z_index", 0)),
    )
    image_2d = read_image_2d(input_path, selection=sel)

    # Load StarDist model
    model_dir = _resolve_model_dir(cfg, data_root)
    model_ref = StardistModelRef(model_dir=model_dir)
    model = load_stardist2d(model_ref)

    # Segmentation params
    params = Slice1NucleiParams(
        normalize_pmin=float(cfg.get("normalize_pmin", 1.0)),
        normalize_pmax=float(cfg.get("normalize_pmax", 99.8)),
        prob_thresh=cfg.get("prob_thresh", None),
        nms_thresh=cfg.get("nms_thresh", None),
    )

    labels, override_thresholds = segment_nuclei_stardist(image_2d, model=model, params=params)

    # Determine final thresholds used (model defaults, optionally overridden)
    model_thresholds = get_model_thresholds(model)
    final_thresholds: Dict[str, float] = {}
    if "prob" in model_thresholds:
        final_thresholds["prob"] = float(model_thresholds["prob"])
    if "nms" in model_thresholds:
        final_thresholds["nms"] = float(model_thresholds["nms"])
    if params.prob_thresh is not None:
        final_thresholds["prob"] = float(params.prob_thresh)
    if params.nms_thresh is not None:
        final_thresholds["nms"] = float(params.nms_thresh)

    # Write label image
    labels_path = out_dir / "nuclei_labels.tif"
    # Save as uint16 when possible, fall back to uint32 if needed
    if int(labels.max()) <= np.iinfo(np.uint16).max:
        tifffile.imwrite(labels_path, labels.astype(np.uint16, copy=False))
    else:
        tifffile.imwrite(labels_path, labels.astype(np.uint32, copy=False))

    # QC overlay
    qc_path = out_dir / "qc_overlay.png"
    _write_qc_overlay(image_2d, labels, qc_path)

    # Optional nuclei table
    if bool(cfg.get("write_nuclei_table", False)):
        props = regionprops_table(
            labels,
            intensity_image=image_2d.astype(np.float32, copy=False),
            properties=(
                "label",
                "area",
                "centroid",
                "bbox",
                "mean_intensity",
                "max_intensity",
            ),
        )
        df = pd.DataFrame(props)
        df.to_parquet(out_dir / "nuclei.parquet", index=False)

    # Manifest
    manifest: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "input_path": str(input_path),
        "output_dir": str(out_dir),
        "git_commit": _try_git_commit(REPO_ROOT),
        "env_name": os.environ.get("CONDA_DEFAULT_ENV"),
        "config_snapshot": cfg,
        "image_shape": [int(s) for s in image_2d.shape],
        "image_dtype": str(image_2d.dtype),
        "num_nuclei": int(labels.max()),
        "nuclei_labels": str(labels_path),
        "qc_overlay": str(qc_path),
        "stardist_model_dir": str(model_dir),
        "stardist_model_thresholds": model_thresholds,
        "override_thresholds": override_thresholds,
        "final_thresholds": final_thresholds,
    }

    with (out_dir / "run_manifest.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    print(f"Slice1 nuclei complete: {out_dir}")
    return out_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Slice1 nucleus segmentation (StarDist)")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/slice1_nuclei.yaml"),
        help="Path to a YAML config file",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    run_slice1_nuclei(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
