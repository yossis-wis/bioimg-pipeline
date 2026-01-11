from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
import yaml

# Parquet support (required by docs/CONTRACTS.md)
try:
    import pyarrow  # noqa: F401
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "pyarrow is required to write spots.parquet. "
        "Install the conda env from environment.yml and re-run scripts/verify_setup.py."
    ) from e

# Matplotlib: write PNGs without needing a display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Allow running without packaging
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT / "src"))

from slice0_kernel import Slice0Params, detect_spots, REQUIRED_COLUMNS  # noqa: E402


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


def _select_2d_plane(arr: np.ndarray, channel: Optional[int]) -> np.ndarray:
    """Best-effort: choose a 2D plane for Slice0.

    Rules:
      - If arr is already 2D -> return it.
      - If arr is 3D and channel is provided -> treat axis0 as channel-like if possible.
      - Otherwise, fall back to the first plane (arr[0]).
    """
    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        if channel is None:
            return arr[0]
        # treat channel as 1-based if >=1; allow channel=0 to mean first channel
        ch = int(channel)
        idx = ch - 1 if ch >= 1 else 0
        if 0 <= idx < arr.shape[0]:
            return arr[idx]
        return arr[0]

    # For higher dims, flatten leading dims and take first plane
    flat = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
    return flat[0]


def _write_qc_overlay(image_2d: np.ndarray, spots: pd.DataFrame, out_path: Path) -> None:
    img = np.asarray(image_2d)

    # Contrast stretch for view
    vmin, vmax = np.percentile(img.astype(float), [1, 99]) if img.size else (0, 1)

    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    if not spots.empty:
        plt.scatter(spots["x_px"], spots["y_px"], s=30, facecolors="none", edgecolors="red", linewidths=1.0)
        plt.title(f"Slice0: {len(spots)} detected spots")
    else:
        plt.title("Slice0: 0 detected spots")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def run_slice0(config_path: Path) -> Path:
    cfg = _load_config(config_path)
    data_root = _data_root()

    # Resolve input TIFF path
    input_relpath = cfg.get("input_relpath")
    if not input_relpath:
        raise ValueError("config missing required key: input_relpath")
    input_path = (data_root / input_relpath).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input TIFF not found: {input_path}")

    # Read TIFF
    arr = tifffile.imread(str(input_path))
    img2d = _select_2d_plane(np.asarray(arr), channel=cfg.get("channel"))

    # Detect
    params = Slice0Params(
        threshold=float(cfg.get("threshold", 0.0)),
        min_distance=int(cfg.get("min_distance", 3)),
        smooth_sigma=float(cfg.get("smooth_sigma", 1.0)),
    )
    spots = detect_spots(img2d, params)

    # Create run folder
    runs_dir = (data_root / cfg.get("output_runs_dir", "runs")).resolve()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (runs_dir / f"{ts}__slice0").resolve()
    out_dir.mkdir(parents=True, exist_ok=False)

    # Outputs
    spots_path = out_dir / "spots.parquet"
    spots.to_parquet(spots_path, index=False)

    qc_path = out_dir / "qc_overlay.png"
    _write_qc_overlay(img2d, spots, qc_path)

    manifest: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "input_path": str(input_path),
        "config_snapshot": cfg,
        "output_dir": str(out_dir),
        "git_commit": _try_git_commit(REPO_ROOT),
        "env_name": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        # small helpful extras (safe additions)
        "image_shape": list(np.asarray(img2d).shape),
        "image_dtype": str(np.asarray(img2d).dtype),
        "num_spots": int(len(spots)),
        "qc_overlay": str(qc_path),
        "kernel_params": asdict(params),
    }

    manifest_path = out_dir / "run_manifest.yaml"
    with manifest_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Slice0 on a TIFF image")
    ap.add_argument("--config", default="configs/dev.yaml", help="Path to YAML config (relative or absolute)")
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()

    out_dir = run_slice0(config_path)
    print(f"Slice0 complete: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
