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
# # Step-by-step integrated QC
#
# This notebook walks through a **single 2D plane**:
# 1) load nuclei + spot channels
# 2) segment nuclei (StarDist)
# 3) detect spots (LoG)
# 4) visualize overlays

# %%
from __future__ import annotations

import os
import sys
from pathlib import Path
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

def _find_repo_root() -> Path:
    cwd = Path.cwd().resolve()
    for parent in (cwd, *cwd.parents):
        if (parent / "src").is_dir():
            return parent
    raise RuntimeError(
        "Could not find repo root (expected a src/ directory). "
        f"Start this notebook from the repo root or a subdirectory. cwd={cwd}"
    )


REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT / "src"))

from image_io import PlaneSelection, read_image_2d
from slice0_kernel import Slice0Params, detect_spots
from slice1_nuclei_kernel import Slice1NucleiParams, segment_nuclei_stardist
from stardist_utils import StardistModelRef, load_stardist2d

# %% [markdown]
# ## Inputs
#
# Update the paths below to match your environment. If a local config exists at
# `configs/local/integrated_ims.local.yaml`, the notebook will load it and
# reuse its parameters (model path, channels, plane selection, etc.).

# %%
# Data roots
bioimg_data_root_raw = os.environ.get("BIOIMG_DATA_ROOT")
if not bioimg_data_root_raw:
    raise RuntimeError("BIOIMG_DATA_ROOT is not set. See docs/SETUP_WINDOWS.md.")
bioimg_data_root = Path(bioimg_data_root_raw)

# Optional: load parameters from a local config file
config_path = REPO_ROOT / "configs" / "local" / "integrated_ims.local.yaml"
config = {}
if config_path.exists():
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}


def _resolve_cfg_path(value: str | Path) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = (bioimg_data_root / path).resolve()
    return path


def _resolve_input_path(cfg: dict) -> Path | None:
    if cfg.get("input_relpath"):
        return _resolve_cfg_path(cfg["input_relpath"])
    if cfg.get("input_relpaths"):
        paths = cfg["input_relpaths"]
        if isinstance(paths, list) and paths:
            return _resolve_cfg_path(paths[0])
    if cfg.get("input_glob"):
        pattern = str(cfg["input_glob"])
        pattern_path = Path(pattern)
        if not pattern_path.is_absolute():
            pattern = str((bioimg_data_root / pattern_path).resolve())
        matches = sorted(glob(pattern))
        if matches:
            return Path(matches[0]).resolve()
    return None


model_dir = _resolve_cfg_path(
    config.get("stardist_model_dir", "models/y22m01d12_model_0")
)

# Input Imaris file
input_path = _resolve_input_path(config) or Path(
    "S:/BIC/<user>/equipment/<instrument>/<date>/<sample>.ims"
)

if not model_dir.exists():
    raise FileNotFoundError(f"StarDist model_dir not found: {model_dir}")
if not input_path.exists():
    raise FileNotFoundError(f"Input file not found: {input_path}")

# Output directory (optional)
output_dir = bioimg_data_root / "runs" / "_notebook_qc"
output_dir.mkdir(parents=True, exist_ok=True)

# Channel selection (1-based)
channel_nuclei = int(config.get("channel_nuclei", 1))
channel_spots_raw = config.get("channel_spots", 2)
if isinstance(channel_spots_raw, (list, tuple)) and channel_spots_raw:
    channel_spots = int(channel_spots_raw[0])
else:
    channel_spots = int(channel_spots_raw)

# Plane selection
ims_resolution_level = int(config.get("ims_resolution_level", 0))
ims_time_index = int(config.get("ims_time_index", 0))
ims_z_index = int(config.get("ims_z_index", 0))

# %% [markdown]
# ## Load nuclei + spot planes

# %%
selection_nuclei = PlaneSelection(
    channel=channel_nuclei,
    ims_resolution_level=ims_resolution_level,
    ims_time_index=ims_time_index,
    ims_z_index=ims_z_index,
)
selection_spots = PlaneSelection(
    channel=channel_spots,
    ims_resolution_level=ims_resolution_level,
    ims_time_index=ims_time_index,
    ims_z_index=ims_z_index,
)

nuclei_img = read_image_2d(input_path, selection_nuclei)
spots_img = read_image_2d(input_path, selection_spots)

print("nuclei_img:", nuclei_img.shape, nuclei_img.dtype)
print("spots_img:", spots_img.shape, spots_img.dtype)

# %% [markdown]
# ## Segment nuclei (StarDist)

# %%
model_ref = StardistModelRef(model_dir=model_dir)
model = load_stardist2d(model_ref)

nuclei_params = Slice1NucleiParams(
    normalize_pmin=float(config.get("nuc_normalize_pmin", 1.0)),
    normalize_pmax=float(config.get("nuc_normalize_pmax", 99.8)),
    prob_thresh=config.get("nuc_prob_thresh", 0.3),
    nms_thresh=config.get("nuc_nms_thresh", None),
)

nuclei_labels, nuclei_meta = segment_nuclei_stardist(nuclei_img, model, nuclei_params)
print("nuclei_labels:", nuclei_labels.shape, nuclei_labels.dtype)
print("nuclei_meta:", nuclei_meta)

# %% [markdown]
# ## Detect spots (LoG)

# %%
spot_params = Slice0Params(
    zR=float(config.get("spot_zR", 344.5)),
    lambda_nm=float(config.get("spot_lambda_nm", 667.0)),
    pixel_size_nm=float(config.get("spot_pixel_size_nm", 65.0)),
    u0_min=float(config.get("spot_u0_min", 30.0)),
)

spots_df = detect_spots(spots_img, spot_params, nuclei_labels=nuclei_labels)
spots_df.head()

# %% [markdown]
# ## Visualize overlays

# %%
try:
    from skimage.segmentation import find_boundaries
except Exception as exc:
    raise ImportError("scikit-image is required for boundary overlays") from exc

boundaries = find_boundaries(nuclei_labels, mode="outer")

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(spots_img, cmap="gray")
ax.imshow(np.ma.masked_where(~boundaries, boundaries), cmap="autumn", alpha=0.6)
ax.scatter(spots_df["x_px"], spots_df["y_px"], s=12, c="cyan", alpha=0.8)
ax.set_title("Spots (cyan) + nuclei boundaries (red)")
ax.axis("off")
plt.show()

# %% [markdown]
# ## Save a QC snapshot (optional)

# %%
qc_path = output_dir / "qc_overlay_notebook.png"
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(spots_img, cmap="gray")
ax.imshow(np.ma.masked_where(~boundaries, boundaries), cmap="autumn", alpha=0.6)
ax.scatter(spots_df["x_px"], spots_df["y_px"], s=12, c="cyan", alpha=0.8)
ax.axis("off")
fig.savefig(qc_path, dpi=200, bbox_inches="tight")
print("wrote", qc_path)
