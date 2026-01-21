# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Review a run folder
#
# This notebook helps you quickly review QC artifacts written by the command-line drivers:
#
# - `qc_overlay*.png` (full-frame overlay)
# - `qc_cutouts*.tif` (montage of 80×80 spot-centered cutouts)

# %%
from __future__ import annotations

import os
import sys
from pathlib import Path
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import yaml

def _find_repo_root() -> Path:
    cwd = Path.cwd().resolve()
    for parent in (cwd, *cwd.parents):
        if (parent / "src").is_dir():
            return parent
    raise RuntimeError("Could not find repo root (expected a src/ directory).")


REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT / "src"))

from vis_utils import merge_nuclei_spots_rgb

# %%
bioimg_data_root_raw = os.environ.get("BIOIMG_DATA_ROOT")
if not bioimg_data_root_raw:
    raise RuntimeError("BIOIMG_DATA_ROOT is not set.")
bioimg_data_root = Path(bioimg_data_root_raw)

# Point this to a completed run folder
run_dir = bioimg_data_root / "runs" / "<timestamp>__integrated"

if not run_dir.exists():
    raise FileNotFoundError(
        f"Run folder not found: {run_dir}\n"
        "Set run_dir to an existing folder under BIOIMG_DATA_ROOT/runs/."
    )

print("run_dir:", run_dir)

# %% [markdown]
# ## Load manifest + spots table

# %%
manifest_path = run_dir / "run_manifest.yaml"
spots_path = run_dir / "spots.parquet"

if not manifest_path.exists():
    raise FileNotFoundError(f"Missing: {manifest_path}")
if not spots_path.exists():
    raise FileNotFoundError(f"Missing: {spots_path}")

with manifest_path.open("r", encoding="utf-8") as handle:
    manifest = yaml.safe_load(handle) or {}

spots_df = pd.read_parquet(spots_path)

print("spots_df rows:", len(spots_df))
spots_df.head()

# %% [markdown]
# ## QC overlay PNG(s)

# %%
qc_pngs = sorted(glob(str(run_dir / "qc_overlay*.png")))
print("qc overlays:", len(qc_pngs))
qc_pngs[:5]

# %%
def _show_png(path: Path, title: str | None = None) -> None:
    img = plt.imread(path)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    ax.set_title(title or path.name)
    ax.axis("off")
    plt.show()

for p in qc_pngs:
    _show_png(Path(p))

# %% [markdown]
# ## Cutout montage TIFF(s)
#
# These are written as ImageJ TIFF with axes `C,Y,X`.
#
# By default:
# - Channel 1 = nuclei (raw)
# - Channel 2 = spots (raw)

# %%
qc_tifs = sorted(glob(str(run_dir / "qc_cutouts*.tif")))
print("qc montages:", len(qc_tifs))
qc_tifs

# %%
qc_cfg = manifest.get("qc", {}) if isinstance(manifest, dict) else {}
crop_size = int(qc_cfg.get("cutout_size", 80))

def _preview_montage(tif_path: Path) -> None:
    arr = tifffile.imread(tif_path)
    if arr.ndim == 2:
        # Single-channel montage (unexpected, but handle)
        nuclei_plane = arr
        spots_plane = np.zeros_like(arr)
    elif arr.ndim == 3:
        # (C, Y, X)
        if arr.shape[0] == 1:
            nuclei_plane = arr[0]
            spots_plane = np.zeros_like(arr[0])
        else:
            nuclei_plane = arr[0]
            spots_plane = arr[1]
    else:
        raise ValueError(f"Unexpected montage shape for {tif_path.name}: {arr.shape}")

    rgb = merge_nuclei_spots_rgb(nuclei_plane, spots_plane, spots_to_cyan=False)

    h, w = rgb.shape[:2]
    fig_w = 12.0
    fig_h = fig_w * (h / w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(rgb, interpolation="nearest")

    # Add cyan rings at the center of each non-empty tile
    from matplotlib.patches import Circle

    crop = crop_size
    n_rows = h // crop
    n_cols = w // crop
    half = crop // 2
    ring_r = 5

    for rr in range(n_rows):
        for cc in range(n_cols):
            tile = spots_plane[rr * crop : (rr + 1) * crop, cc * crop : (cc + 1) * crop]
            if float(tile.max()) <= 0.0:
                continue
            cx = cc * crop + half
            cy = rr * crop + half
            ax.add_patch(
                Circle(
                    (cx, cy),
                    radius=ring_r,
                    fill=False,
                    edgecolor="cyan",
                    linewidth=1.0,
                    alpha=0.9,
                )
            )

    ax.set_title(tif_path.name + " (merge preview + rings)")
    ax.axis("off")
    plt.show()


for p in qc_tifs:
    _preview_montage(Path(p))
