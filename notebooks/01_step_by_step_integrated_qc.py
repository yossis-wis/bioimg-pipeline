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
#
# 1) load nuclei + spot channels  
# 2) segment nuclei (StarDist)  
# 3) detect spots (LoG)  
# 4) visualize overlays **without hiding raw spot data** (ring markers)  
# 5) build an **80×80 cutout montage** (open in Fiji or view here as a merge)

# %%
from __future__ import annotations

import os
import sys
from pathlib import Path
from glob import glob

import matplotlib

if "ipykernel" in sys.modules:
    try:
        if matplotlib.get_backend().lower() == "agg":
            matplotlib.use("module://matplotlib_inline.backend_inline", force=True)
    except Exception:
        pass

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
from vis_utils import create_cutout_montage, merge_nuclei_spots_rgb

# %% [markdown]
# ## Inputs
#
# If a local config exists at `configs/local/integrated_ims.local.yaml`,
# the notebook loads it and reuses its parameters (model path, channels, plane selection, etc.).

# %%
bioimg_data_root_raw = os.environ.get("BIOIMG_DATA_ROOT")
if not bioimg_data_root_raw:
    raise RuntimeError("BIOIMG_DATA_ROOT is not set. See docs/SETUP_WINDOWS.md.")
bioimg_data_root = Path(bioimg_data_root_raw)

# Optional: load parameters from a local config file
config_path = REPO_ROOT / "configs" / "local" / "integrated_ims.local.yaml"
config: dict = {}
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


# StarDist model folder under BIOIMG_DATA_ROOT/
model_dir = _resolve_cfg_path(
    config.get("stardist_model_dir", "models/y22m01d12_model_0")
)

# Input image (.ims or .tif)
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

# Plane selection (for .ims only; ignored for TIFF)
ims_resolution_level = int(config.get("ims_resolution_level", 0))
ims_time_index = int(config.get("ims_time_index", 0))
ims_z_index = int(config.get("ims_z_index", 0))

print("input_path:", input_path)
print("model_dir:", model_dir)
print("channel_nuclei:", channel_nuclei, "channel_spots:", channel_spots)
print("ims selection:", dict(r=ims_resolution_level, t=ims_time_index, z=ims_z_index))

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

print("nuclei_img:", nuclei_img.shape, nuclei_img.dtype, "min/max:", int(np.min(nuclei_img)), int(np.max(nuclei_img)))
print("spots_img:", spots_img.shape, spots_img.dtype, "min/max:", int(np.min(spots_img)), int(np.max(spots_img)))

# %% [markdown]
# ### Quick view of raw channels (percentile-scaled)

# %%
def _show_gray(img: np.ndarray, title: str, *, pmin: float = 1.0, pmax: float = 99.0) -> None:
    vmin, vmax = np.percentile(img.astype(float), [pmin, pmax])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    plt.show()

_show_gray(nuclei_img, "Nuclei channel (raw)", pmin=1, pmax=99.8)
_show_gray(spots_img, "Spots channel (raw)", pmin=1, pmax=99)

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

num_nuclei = int(np.max(nuclei_labels)) if nuclei_labels.size else 0
print("nuclei_labels:", nuclei_labels.shape, nuclei_labels.dtype, "num_nuclei:", num_nuclei)
print("nuclei_meta:", nuclei_meta)

if num_nuclei == 0:
    print(
        "\nWARNING: StarDist produced 0 nuclei labels.\n"
        "  - Check that channel_nuclei points to the DAPI/nuclear channel (1-based).\n"
        "  - Check that the StarDist model matches your imaging modality.\n"
        "  - Consider lowering nuc_prob_thresh in your local YAML.\n"
    )

# %% [markdown]
# ## Detect spots (LoG)

# %%
spot_params = Slice0Params(
    zR=float(config.get("spot_zR", 344.5)),
    lambda_nm=float(config.get("spot_lambda_nm", 667.0)),
    pixel_size_nm=float(config.get("spot_pixel_size_nm", 65.0)),
    u0_min=float(config.get("spot_u0_min", 30.0)),
    # TrackMate-style candidate stage controls:
    spot_radius_nm=(float(config["spot_radius_nm"]) if config.get("spot_radius_nm") is not None else None),
    do_median_filter=bool(config.get("spot_do_median_filter", False)),
    do_subpixel_localization=bool(config.get("spot_do_subpixel_localization", False)),
    q_min=float(config.get("spot_q_min", 1.0)),
    se_size=int(config.get("spot_se_size", 3)),
)

# TrackMate GUI uses an *estimated blob diameter* (in calibrated units), but internally stores a radius.
# For QC: the equivalent GUI value implied by this config is:
radius_nm = spot_params.spot_radius_nm
if radius_nm is None:
    radius_nm = float(np.sqrt(spot_params.lambda_nm * spot_params.zR / np.pi))
tm_diameter_nm = 2.0 * radius_nm
tm_diameter_um = tm_diameter_nm / 1000.0
tm_diameter_px = tm_diameter_nm / spot_params.pixel_size_nm
print(f"TrackMate GUI blob diameter ≈ {tm_diameter_um:.4f} µm ({tm_diameter_nm:.1f} nm; {tm_diameter_px:.2f} px)")

spots_df = detect_spots(spots_img, spot_params, nuclei_labels=nuclei_labels)

print("num_spots:", len(spots_df))
spots_df.head()

# %% [markdown]
# ## Overlay QC (rings that don't hide the spot peak)

# %%
from skimage.morphology import dilation, disk
from skimage.segmentation import find_boundaries

img = spots_img.astype(float)
vmin, vmax = np.percentile(img, [1, 99])

boundaries = find_boundaries(nuclei_labels > 0, mode="outer")
boundaries = dilation(boundaries, disk(1))

rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=float)
rgba[boundaries, 0] = 1.0   # red
rgba[boundaries, 3] = 0.6   # alpha

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
ax.imshow(rgba, interpolation="nearest")

if not spots_df.empty:
    ax.scatter(
        spots_df["x_px"],
        spots_df["y_px"],
        s=60,
        facecolors="none",
        edgecolors="cyan",
        linewidths=1.2,
        alpha=0.9,
    )
ax.set_title(f"Spots (cyan rings): {len(spots_df)}    Nuclei: {num_nuclei}")
ax.axis("off")
plt.show()

# %% [markdown]
# ## 80×80 cutout montage (TIFF for Fiji + merge preview here)
#
# The montage is written as a **2-channel ImageJ TIFF** with axes `C,Y,X`:
# - Channel 1: nuclei (raw)  
# - Channel 2: spots (raw)
#
# Tip: open the TIFF in Fiji and switch to **Image → Color → Channels Tool...**
# to adjust LUTs/contrast interactively.

# %%
qc_cutout_size = int(config.get("qc_cutout_size", 80))
qc_max_cutouts = int(config.get("qc_max_cutouts", 50))
qc_montage_cols = int(config.get("qc_montage_cols", 10))

qc_sample_seed_raw = config.get("qc_sample_seed", None)
if qc_sample_seed_raw in (None, "", "none", "None"):
    qc_sample_seed = None
else:
    qc_sample_seed = int(qc_sample_seed_raw)

montage, montage_count = create_cutout_montage(
    nuclei_img,
    spots_img,
    spots_df,
    crop_size=qc_cutout_size,
    max_cutouts=qc_max_cutouts,
    n_cols=qc_montage_cols,
    sample_seed=qc_sample_seed,
)

print("montage_count:", montage_count)
if montage is None:
    print("No montage written (no spots).")
else:
    montage_path = output_dir / "qc_cutouts_notebook.tif"
    tifffile.imwrite(montage_path, montage, imagej=True, metadata={"axes": "CYX"})
    print("wrote", montage_path)

    # Merge preview (Fiji-like)
    rgb = merge_nuclei_spots_rgb(montage[0], montage[1], spots_to_cyan=False)

    # Figure size matched to aspect ratio
    h, w = rgb.shape[:2]
    fig_w = 12.0
    fig_h = fig_w * (h / w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(rgb, interpolation="nearest")

    # Draw a small ring at the center of each populated tile (outline only)
    from matplotlib.patches import Circle

    crop = qc_cutout_size
    n_rows = h // crop
    n_cols = w // crop
    half = crop // 2
    ring_r = 5  # px

    spot_plane = montage[1]
    for rr in range(n_rows):
        for cc in range(n_cols):
            tile = spot_plane[rr * crop : (rr + 1) * crop, cc * crop : (cc + 1) * crop]
            if float(tile.max()) <= 0.0:
                continue
            cx = cc * crop + half
            cy = rr * crop + half
            ax.add_patch(Circle((cx, cy), radius=ring_r, fill=False, edgecolor="cyan", linewidth=1.0, alpha=0.9))

    ax.set_title("Cutout montage merge (nuclei=red, spots=green) + cyan rings")
    ax.axis("off")
    plt.show()

# %% [markdown]
# ## Save a QC snapshot (optional)

# %%
qc_path = output_dir / "qc_overlay_notebook.png"
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
ax.imshow(rgba, interpolation="nearest")
if not spots_df.empty:
    ax.scatter(
        spots_df["x_px"],
        spots_df["y_px"],
        s=60,
        facecolors="none",
        edgecolors="cyan",
        linewidths=1.2,
        alpha=0.9,
    )
ax.axis("off")
fig.savefig(qc_path, dpi=200, bbox_inches="tight")
print("wrote", qc_path)
plt.close(fig)


