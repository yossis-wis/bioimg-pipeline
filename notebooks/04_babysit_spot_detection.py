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
# # Babysit spot detection (Slice0, LoG + mean-threshold)
#
# This notebook is meant to **build trust** in the spot detection by walking through it
# in **small, visual, incremental steps**:
#
# 1) Load a single 2D plane (nuclei channel + spot channel)
# 2) Segment nuclei (Slice1 / StarDist) or load precomputed nuclei labels
# 3) Run LoG-based candidate detection (Slice0):
#    - derive $\sigma_0$ from $(z_R,\lambda)$ and pixel size
#    - build LoG kernel, convolve
#    - non-maximum suppression on the LoG response (TrackMate-style kernel yields positive peaks)
#    - quality threshold $q_{\min}$
# 4) Compute per-candidate measurements (exactly like the realtime script):
#    - background median in the thin ring mask (out0)
#    - mean inside the small disk mask (in5)
#    - $u_0 = \langle I\rangle_{\mathrm{in5}} - \mathrm{median}(I)_{\mathrm{out0}}$
#    - keep if $u_0 > u_{0,\min}$
# 5) Apply nuclei inclusion + assign `nucleus_label` to each spot
#
# The intent is that you can run this on a representative real file (e.g. an `.ims` plane),
# and visually confirm that each intermediate step behaves as expected.

# %%
from __future__ import annotations

import os
import sys
from glob import glob
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

# --- Matplotlib backend safety for notebooks ---
# If some prior import forced a headless backend (Agg), switch back to inline in Jupyter.
import matplotlib

if "ipykernel" in sys.modules:
    try:
        if matplotlib.get_backend().lower() == "agg":
            matplotlib.use("module://matplotlib_inline.backend_inline", force=True)
    except Exception:
        pass

import matplotlib.pyplot as plt

try:
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover
    def display(x):  # type: ignore
        print(x)


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

from image_io import PlaneSelection, read_image_2d  # noqa: E402
from slice0_kernel import Slice0Params, detect_spots_debug  # noqa: E402
from slice1_nuclei_kernel import Slice1NucleiParams, segment_nuclei_stardist  # noqa: E402
from stardist_utils import StardistModelRef, load_stardist2d  # noqa: E402
from vis_utils import merge_nuclei_spots_rgb  # noqa: E402

# We'll reuse the exact pixel-edge outline helper used by the PPTX atlas QC.
from qc_spot_atlas import _mask_outline_xy  # type: ignore  # noqa: E402

import tifffile  # noqa: E402


# %% [markdown]
# ## Inputs
#
# This notebook can reuse the same local config as the integrated driver:
#
# - `configs/local/integrated_ims.local.yaml`
#
# If that file exists, we load it and reuse its parameters (model path, channels, plane selection, etc.).
# Otherwise, we fall back to placeholders you can edit here.

# %%
bioimg_root_raw = os.environ.get("BIOIMG_DATA_ROOT")
if not bioimg_root_raw:
    raise RuntimeError("BIOIMG_DATA_ROOT is not set. See docs/SETUP_WINDOWS.md.")
BIOIMG_DATA_ROOT = Path(bioimg_root_raw).expanduser().resolve()

config_path = REPO_ROOT / "configs" / "local" / "integrated_ims.local.yaml"
config: dict = {}
if config_path.exists():
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
print("loaded config:", config_path if config_path.exists() else "(none)")


def _resolve_cfg_path(value: str | Path) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = (BIOIMG_DATA_ROOT / path).resolve()
    return path


def _resolve_input_path(cfg: dict) -> Optional[Path]:
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
            pattern = str((BIOIMG_DATA_ROOT / pattern_path).resolve())
        matches = sorted(glob(pattern))
        if matches:
            return Path(matches[0]).resolve()
    return None


# --- StarDist model folder under BIOIMG_DATA_ROOT/ ---
model_dir = _resolve_cfg_path(config.get("stardist_model_dir", "models/y22m01d12_model_0"))

# --- Input image (.ims or .tif) ---
input_path = _resolve_input_path(config) or Path(
    "S:/BIC/<user>/equipment/<instrument>/<date>/<sample>.ims"
)

if not input_path.exists():
    raise SystemExit(
        f"Input file not found: {input_path}\n"
        "Fix by either:\n"
        "  1) setting input_relpath/input_relpaths/input_glob in configs/local/integrated_ims.local.yaml, or\n"
        "  2) editing input_path in this notebook.\n"
    )

# Channels are 1-based (as in the integrated config template).
channel_nuclei = int(config.get("channel_nuclei", 1))

# channel_spots can be an int or list; here we focus on the *first* spot channel.
channel_spots_raw = config.get("channel_spots", 2)
if isinstance(channel_spots_raw, (list, tuple)) and channel_spots_raw:
    channel_spots = int(channel_spots_raw[0])
else:
    channel_spots = int(channel_spots_raw)

# Plane selection for .ims (ignored for TIFF)
ims_resolution_level = int(config.get("ims_resolution_level", 0))
ims_time_index = int(config.get("ims_time_index", 0))
ims_z_index = int(config.get("ims_z_index", 0))

print("input_path:", input_path)
print("channel_nuclei:", channel_nuclei, "| channel_spots:", channel_spots)
print("ims selection:", dict(rl=ims_resolution_level, t=ims_time_index, z=ims_z_index))

# Optional: restrict detection to a valid mask (nonzero = valid)
valid_mask: Optional[np.ndarray] = None
if config.get("valid_mask_relpath"):
    vm_path = _resolve_cfg_path(config["valid_mask_relpath"])
    if not vm_path.exists():
        raise FileNotFoundError(f"valid_mask_relpath not found: {vm_path}")
    valid_mask = tifffile.imread(str(vm_path)).astype(bool)
    print("loaded valid_mask:", vm_path, "shape:", valid_mask.shape)

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

nuclei_plane = read_image_2d(input_path, selection_nuclei)
spots_plane = read_image_2d(input_path, selection_spots)

print("nuclei_plane:", nuclei_plane.shape, nuclei_plane.dtype)
print("spots_plane :", spots_plane.shape, spots_plane.dtype)

# Quick visuals (use robust percentile contrast)
def _imshow(ax, img, title: str, p_lo=1.0, p_hi=99.8):
    vmin, vmax = np.percentile(img.astype(float), [p_lo, p_hi])
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_axis_off()


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
_imshow(axes[0], nuclei_plane, f"Nuclei channel (ch{channel_nuclei})", p_lo=1.0, p_hi=99.8)
_imshow(axes[1], spots_plane, f"Spot channel (ch{channel_spots})", p_lo=1.0, p_hi=99.5)
plt.show()

# %% [markdown]
# ## Segment nuclei (Slice1 / StarDist)
#
# If your config provides `nuclei_labels_relpath`, we load it (skipping StarDist).
# Otherwise we run StarDist on the nuclei plane.

# %%
nuclei_labels: Optional[np.ndarray] = None

if config.get("nuclei_labels_relpath"):
    labels_path = _resolve_cfg_path(config["nuclei_labels_relpath"])
    nuclei_labels = np.asarray(tifffile.imread(str(labels_path)))
    nuclei_labels = np.squeeze(nuclei_labels)
    if nuclei_labels.ndim != 2:
        raise ValueError(f"nuclei_labels_relpath must be 2D; got shape={nuclei_labels.shape}")
    nuclei_labels = nuclei_labels.astype(np.int32, copy=False)
    nuclei_meta = {"source": "precomputed", "path": str(labels_path)}
    print("loaded nuclei_labels:", labels_path, nuclei_labels.shape, nuclei_labels.dtype)
else:
    if not model_dir.exists():
        raise SystemExit(
            f"StarDist model_dir not found: {model_dir}\n"
            "Either download/copy the model under BIOIMG_DATA_ROOT/models/ or set nuclei_labels_relpath."
        )

    model_ref = StardistModelRef(model_dir)
    model = load_stardist2d(model_ref)

    prob_thresh_raw = config.get("nuc_prob_thresh", None)
    if prob_thresh_raw in (None, "", "none", "None"):
        prob_thresh = None
    else:
        prob_thresh = float(prob_thresh_raw)

    nms_thresh_raw = config.get("nuc_nms_thresh", None)
    if nms_thresh_raw in (None, "", "none", "None"):
        nms_thresh = None
    else:
        nms_thresh = float(nms_thresh_raw)

    nuc_params = Slice1NucleiParams(
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
        normalize_pmin=float(config.get("nuc_normalize_pmin", 1.0)),
        normalize_pmax=float(config.get("nuc_normalize_pmax", 99.8)),
    )

    nuclei_labels, nuclei_meta = segment_nuclei_stardist(nuclei_plane, model=model, params=nuc_params)
    print("segmented nuclei_labels:", nuclei_labels.shape, nuclei_labels.dtype)
    print("nuclei_meta:", nuclei_meta)

assert nuclei_labels is not None

# Visualize labels as an outline overlay on nuclei intensity.
#
# Use the same boundary logic as the integrated QC overlay (pixel-exact, no interpolation).
from skimage.segmentation import find_boundaries

fig, ax = plt.subplots(figsize=(6, 6))
_imshow(ax, nuclei_plane, "Nuclei intensity + segmentation outline", p_lo=1.0, p_hi=99.8)

# Compute boundaries on the *label image* (not a boolean union mask),
# so you get an outline around **each** nucleus.
boundaries = find_boundaries(nuclei_labels, mode="outer")

rgba = np.zeros((boundaries.shape[0], boundaries.shape[1], 4), dtype=float)
rgba[boundaries, 0] = 1.0
rgba[boundaries, 3] = 0.90  # 1 px outline (no dilation, no fill)

ax.imshow(rgba, interpolation="nearest")
plt.show()


# %% [markdown]
# ## Configure Slice0 parameters (spot detection)
#
# The optics-derived scale comes from the Rayleigh range relation:
#
# $$
# z_R = \frac{\pi w_0^2}{\lambda}
# \quad\Rightarrow\quad
# w_0 = \sqrt{\frac{\lambda z_R}{\pi}}.
# $$
#
# If the PSF is approximated as a Gaussian with $1/e^2$ radius $w_0$, then
# $\sigma = w_0/\sqrt{2}$ in the $\exp(-r^2/2\sigma^2)$ convention.
#
# Slice0 uses:
#
# $$
# \sigma_0\;[\mathrm{px}] = \frac{w_0/\sqrt{2}}{p},
# $$
#
# where $p$ is the pixel size (same length units as $w_0$).

# %%
params = Slice0Params(
    zR=float(config.get("spot_zR", 344.5)),
    lambda_nm=float(config.get("spot_lambda_nm", 667.0)),
    pixel_size_nm=float(config.get("spot_pixel_size_nm", 65.0)),
    u0_min=float(config.get("spot_u0_min", 30.0)),
    # You can tune these if needed:
    q_min=float(config.get("spot_q_min", 1.0)),
    se_size=int(config.get("spot_se_size", 3)),
)

print(params)

# %% [markdown]
# ## Run spot detection with debug outputs

# %%
spots_df, dbg = detect_spots_debug(
    spots_plane,
    params,
    valid_mask=valid_mask,
    nuclei_labels=nuclei_labels,
)

print("Spots after u0 threshold (and inside nuclei):", len(spots_df))
display(spots_df.head())

# Quick accounting through the pipeline:
print("\nPipeline counts:")
print("  local maxima (raw, no masks):", int(dbg.ys_raw.size))
print("  local maxima (after masks):  ", int(dbg.ys_masked.size))
print("  after q_min:", int(dbg.ys_q.size), f"(q_min={params.q_min})")
print("  after u0_min:", int(len(spots_df)), f"(u0_min={params.u0_min})")

# %% [markdown]
# ## Step 1 — Inspect the LoG kernel and the LoG response
#
# Slice0 constructs a **TrackMate-style LoG kernel** (a normalized version of $-\nabla^2 G$),
# matching TrackMate’s implementation in `DetectionUtils.createLoGKernel`.
#
# TrackMate writes the kernel (in calibrated coordinates) as:
#
# $$
# h(\mathbf{r}) = -C\,m(\mathbf{r})\,\exp\!\left(-\frac{\|\mathbf{r}\|^2}{2\sigma^2}\right),
# $$
#
# where in 2D:
#
# $$
# m(\mathbf{r}) =
# \sum_{d\in\{x,y\}}
# \frac{1}{\sigma_{\mathrm{px},d}^2}
# \left(\frac{x_d^2}{\sigma^2} - 1\right),
# \qquad
# C = \frac{1}{\pi\,\sigma_{\mathrm{px},x}^2}.
# $$
#
# In the common isotropic case this reduces (up to a constant factor) to the familiar form:
#
# $$
# h(r) \propto \left(2 - \frac{r^2}{\sigma^2}\right)\exp\!\left(-\frac{r^2}{2\sigma^2}\right),
# $$
#
# which yields **positive peaks** for bright, in-focus spots.
#
# The image is convolved with $h$, and candidates are found as **strict** local maxima in $(\mathrm{LoG} * I)$.

# %%
print(f"Derived: w0 = {dbg.w0_nm:.2f} nm, sigma0 = {dbg.sigma0_px:.2f} px, kernel size = {dbg.log_filter.shape}")

# TrackMate GUI uses an *estimated blob diameter* (in calibrated units),
# but internally stores a radius. For QC, the equivalent GUI value is:
tm_diameter_nm = 2.0 * dbg.w0_nm
tm_diameter_um = tm_diameter_nm / 1000.0
tm_diameter_px = tm_diameter_nm / params.pixel_size_nm
print(f"TrackMate GUI blob diameter ≈ {tm_diameter_um:.4f} µm ({tm_diameter_nm:.1f} nm; {tm_diameter_px:.2f} px)")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(dbg.log_filter, interpolation="nearest")
axes[0].set_title("LoG kernel h (numeric values)")
axes[0].set_axis_off()

# Show the LoG response map (padded back to image size)
resp = dbg.image_conv_padded  # larger => more spot-like (TrackMate-style kernel yields bright spots directly)
vmin, vmax = np.percentile(resp.astype(float), [1, 99.5])
axes[1].imshow(resp, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
axes[1].set_title(r"$(\mathrm{LoG} * I)$ (padded)")
axes[1].set_axis_off()

# Note: use `set_axis_off()` (not `axis("off")`) to avoid printing axis limits in some notebook frontends.
plt.show()
# %% [markdown]
# ### (Optional) View LoG as “Gaussian smoothing” + “Laplacian”
#
# In continuous space, applying a LoG is equivalent to applying a Laplacian after Gaussian smoothing:
#
# $$
# \nabla^2\!\left(G_{\sigma_0} * I\right) \equiv \left(\nabla^2 G_{\sigma_0}\right) * I.
# $$
#
# Slice0 uses the **direct LoG kernel convolution** (right-hand form), but it can be helpful
# to visualize the intermediate “Gaussian-smoothed” image as an intuition aid.

# %%
try:
    from scipy.ndimage import gaussian_filter, laplace  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("SciPy is required for this visualization.") from e

g = gaussian_filter(spots_plane.astype(np.float32, copy=False), sigma=float(dbg.sigma0_px))
lap = laplace(g)  # discrete Laplacian of the smoothed image

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
vmin, vmax = np.percentile(g.astype(float), [1, 99.5])
axes[0].imshow(g, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
axes[0].set_title(r"$G_{\sigma_0} * I$ (Gaussian-smoothed)")
axes[0].set_axis_off()

score2 = -lap
vmin2, vmax2 = np.percentile(score2.astype(float), [1, 99.5])
axes[1].imshow(score2, cmap="gray", vmin=vmin2, vmax=vmax2, interpolation="nearest")
axes[1].set_title(r"$-\nabla^2(G_{\sigma_0} * I)$ (spot-like score)")
axes[1].set_axis_off()
plt.show()

plt.show()

# %% [markdown]
# ## Step 2 — Non-maximum suppression (candidate maxima)
#
# Slice0 performs non-maximum suppression by comparing each pixel to the maximum
# in a square neighborhood (size `se_size`) **on the LoG response** (TrackMate-style kernel yields bright spots).
#
# We can visualize where these candidates fall (before and after the nuclei mask).

# %%
# Plot candidates over the spot image (raw maxima)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

_imshow(axes[0], spots_plane, "Spot image + raw LoG maxima (no masks)", p_lo=1.0, p_hi=99.5)
axes[0].scatter(dbg.xs_raw, dbg.ys_raw, s=2, facecolors="none", edgecolors="cyan", linewidths=0.5)

_imshow(axes[1], spots_plane, "Spot image + maxima after masks (inside nuclei [+valid mask])", p_lo=1.0, p_hi=99.5)
axes[1].scatter(dbg.xs_masked, dbg.ys_masked, s=2, facecolors="none", edgecolors="cyan", linewidths=0.5)

plt.show()

# %% [markdown]
# ## Step 3 — Quality threshold $q_{\min}$
#
# Slice0 defines the quality at each candidate as:
#
# $$
# q = (\mathrm{LoG}*I)(y_0,x_0),
# $$
#
# and keeps candidates with $q > q_{\min}$.

# %%
q = dbg.quality_masked.astype(float)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.hist(q, bins=60)
ax.axvline(params.q_min, linestyle="--")
ax.set_xlabel("q  (=LoG response at candidate)")
ax.set_ylabel("count")
ax.set_title("Candidate quality distribution")
plt.show()

# %% [markdown]
# ## Step 4 — Mean-threshold step ($u_0$) and the **pixel-exact measurement mask**
#
# For each candidate (after $q_{\min}$), Slice0 extracts a `(2*window_radius_px+1)^2` crop (default: 31×31),
# and computes:
#
# $$
# \begin{aligned}
# b &= \mathrm{median}\{ I(\mathbf{r}) : \mathbf{r}\in\mathrm{out0}\},\\
# \langle I\rangle_{\mathrm{in5}} &= \frac{1}{|\mathrm{in5}|}\sum_{\mathbf{r}\in\mathrm{in5}} I(\mathbf{r}),\\
# u_0 &= \langle I\rangle_{\mathrm{in5}} - b.
# \end{aligned}
# $$
#
# The **yellow outline** used in the PowerPoint atlas corresponds to the *pixel mask* `in5`.
#
# Below we compute $u_0$ for the $q$-filtered candidates (including those that may fail the $u_{0,\min}$ threshold),
# and then inspect a representative spot at pixel level.

# %%
wr = int(dbg.window_radius_px)
win = 2 * wr + 1

ys_q = dbg.ys_q.astype(int)
xs_q = dbg.xs_q.astype(int)

u0_all = np.full(ys_q.shape, np.nan, dtype=float)
u1_all = np.full(ys_q.shape, np.nan, dtype=float)

bkg_all = np.full(ys_q.shape, np.nan, dtype=float)
mean_in5_all = np.full(ys_q.shape, np.nan, dtype=float)

window_range = np.arange(-wr, wr + 1)

for i, (y0, x0) in enumerate(zip(ys_q.tolist(), xs_q.tolist())):
    ys0 = y0 + window_range
    xs0 = x0 + window_range
    if ys0.min() < 0 or xs0.min() < 0 or ys0.max() >= spots_plane.shape[0] or xs0.max() >= spots_plane.shape[1]:
        continue
    box = spots_plane[np.ix_(ys0, xs0)].astype(np.float32, copy=False)
    bkg = float(np.median(box[dbg.out0_mask]))
    mean_in5 = float(box[dbg.in5_mask].mean())
    mean_in7 = float(box[dbg.in7_mask].mean())

    u0_all[i] = mean_in5 - bkg
    u1_all[i] = mean_in7 - bkg
    bkg_all[i] = bkg
    mean_in5_all[i] = mean_in5

# Histogram of u0 over q-filtered candidates
fig, ax = plt.subplots(figsize=(6, 3.5))
vals = u0_all[np.isfinite(u0_all)]
ax.hist(vals, bins=60)
ax.axvline(params.u0_min, linestyle="--")
ax.set_xlabel("u0  (mean(in5) - median(out0))")
ax.set_ylabel("count")
ax.set_title("u0 distribution over q-filtered candidates")
plt.show()

print("q-filtered candidates:", len(vals))
print("pass u0_min:", int(np.sum(vals > params.u0_min)), f"(u0_min={params.u0_min})")

# %% [markdown]
# ### Pixel-level inspection of one spot
#
# We'll choose a representative detected spot (by default: the highest-intensity one).
# You can change the selection logic below.

# %%
if len(spots_df) == 0:
    raise RuntimeError("No detected spots after u0_min; choose a different plane or relax thresholds.")

# Pick a spot (highest intensity by default)
pick = spots_df.sort_values("intensity", ascending=False).iloc[0]
y0 = int(round(float(pick["y_px"])))
x0 = int(round(float(pick["x_px"])))
nuc_id = int(pick.get("nucleus_label", 0))

print("picked spot:")
print("  (y,x) =", (y0, x0), "| u0 =", float(pick["intensity"]), "| nucleus_label =", nuc_id)

ys0 = y0 + window_range
xs0 = x0 + window_range
box = spots_plane[np.ix_(ys0, xs0)].astype(np.float32, copy=False)

bkg = float(np.median(box[dbg.out0_mask]))
mean_in5 = float(box[dbg.in5_mask].mean())
u0 = mean_in5 - bkg

print("recomputed (from crop):")
print("  median(out0) =", bkg)
print("  mean(in5)    =", mean_in5)
print("  u0           =", u0)

# %% [markdown]
# #### Show the crop with **pixel-edge** in5 outline (yellow)
#
# The outline is rendered in the same coordinate convention as the PPTX atlas:
# pixel centers at integers (1..W), pixel edges at half-integers.

# %%
from skimage.measure import find_contours

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Left: crop intensity
vmin, vmax = np.percentile(box.astype(float), [1, 99.5])
axes[0].imshow(
    box,
    cmap="gray",
    vmin=vmin,
    vmax=vmax,
    interpolation="nearest",
    extent=(0.5, win + 0.5, win + 0.5, 0.5),  # MATLAB-like pixel centers at integers
)
axes[0].set_title("31×31 spot crop + in5 (yellow) + out0 (cyan)")
axes[0].set_aspect("equal")
axes[0].set_axis_off()

# in5 outline (yellow): single simply-connected mask -> use the atlas helper
x_in5, y_in5 = _mask_outline_xy(dbg.in5_mask)
if x_in5.size:
    axes[0].plot(x_in5, y_in5, linewidth=1.5, color="yellow")

# out0 outline (cyan): ring mask -> may have multiple contours (outer + inner)
for c in find_contours(dbg.out0_mask.astype(float), 0.5):
    axes[0].plot(c[:, 1] + 1.0, c[:, 0] + 1.0, linewidth=1.2, color="cyan")

# Middle: show the in5 mask itself (for absolute clarity)
axes[1].imshow(
    dbg.in5_mask.astype(int),
    interpolation="nearest",
    cmap="gray",
    extent=(0.5, win + 0.5, win + 0.5, 0.5),
)
axes[1].set_title("in5 mask (1 = included)")
axes[1].set_aspect("equal")
axes[1].set_axis_off()

# Right: show the out0 mask itself (for absolute clarity)
axes[2].imshow(
    dbg.out0_mask.astype(int),
    interpolation="nearest",
    cmap="gray",
    extent=(0.5, win + 0.5, win + 0.5, 0.5),
)
axes[2].set_title("out0 mask (1 = included)")
axes[2].set_aspect("equal")
axes[2].set_axis_off()

plt.show()


# %% [markdown]
# ## Step 5 — Nucleus assignment sanity check
#
# `nucleus_label` is assigned by sampling the nuclei label image at the spot center:
#
# $$
# \mathrm{nucleus\_label}(y_0,x_0) = L(y_0,x_0),
# $$
#
# where $L$ is the integer label image from Slice1.

# %%
# Visualize an ROI around the selected spot on both channels.
#
# Prefer an ROI that contains the *entire* assigned nucleus (so the outline is a closed loop),
# falling back to a spot-centered crop if `nucleus_label==0`.
H, W = spots_plane.shape
fallback_pad = 60
nuc_margin = int(config.get("nuc_roi_margin_px", 20))

if nuc_id > 0:
    ys_n, xs_n = np.where(nuclei_labels == nuc_id)
    if ys_n.size:
        y1 = max(0, int(ys_n.min()) - nuc_margin)
        y2 = min(H, int(ys_n.max()) + nuc_margin + 1)
        x1 = max(0, int(xs_n.min()) - nuc_margin)
        x2 = min(W, int(xs_n.max()) + nuc_margin + 1)
    else:
        # Unexpected: nuc_id > 0 but no pixels found; fall back.
        y1, y2 = max(0, y0 - fallback_pad), min(H, y0 + fallback_pad)
        x1, x2 = max(0, x0 - fallback_pad), min(W, x0 + fallback_pad)
else:
    y1, y2 = max(0, y0 - fallback_pad), min(H, y0 + fallback_pad)
    x1, x2 = max(0, x0 - fallback_pad), min(W, x0 + fallback_pad)

roi_nuc = nuclei_plane[y1:y2, x1:x2]
roi_spot = spots_plane[y1:y2, x1:x2]

# Build a pixel-exact boundary overlay.
from skimage.measure import find_contours
from skimage.segmentation import find_boundaries

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
_imshow(axes[0], roi_nuc, "Nuclei ROI + nucleus outline", p_lo=1.0, p_hi=99.8)
_imshow(axes[1], roi_spot, "Spot ROI + nucleus outline", p_lo=1.0, p_hi=99.5)

# Plot a *pixel-edge* outline (no dilation, no filled alpha mask),
# so you can still inspect raw pixel values under the cursor.
if nuc_id > 0:
    nuc_mask_roi = nuclei_labels[y1:y2, x1:x2] == nuc_id
    for c in find_contours(nuc_mask_roi.astype(float), 0.5):
        axes[0].plot(c[:, 1], c[:, 0], linewidth=1.5, color="red")
        axes[1].plot(c[:, 1], c[:, 0], linewidth=1.5, color="red")
else:
    # If the spot isn't assigned to a nucleus, show all nucleus boundaries for context.
    boundaries_roi = find_boundaries(nuclei_labels[y1:y2, x1:x2], mode="outer")
    rgba = np.zeros((roi_spot.shape[0], roi_spot.shape[1], 4), dtype=float)
    rgba[boundaries_roi, 0] = 1.0
    rgba[boundaries_roi, 3] = 0.90
    axes[0].imshow(rgba, interpolation="nearest")
    axes[1].imshow(rgba, interpolation="nearest")

# Spot center marker
axes[0].scatter([x0 - x1], [y0 - y1], s=40, facecolors="none", edgecolors="yellow", linewidths=1.5)
axes[1].scatter([x0 - x1], [y0 - y1], s=40, facecolors="none", edgecolors="yellow", linewidths=1.5)

plt.show()


print("nucleus_label at (y0,x0):", int(nuclei_labels[y0, x0]))
print("expected nucleus_label from spots table:", nuc_id)

# %% [markdown]
# ## Final visual summary — merged overlay (nuclei + spots)
#
# This is similar to the integrated QC overlay, but here it is just a quick sanity check
# after the detailed "babysit" steps.

# %%
rgb = merge_nuclei_spots_rgb(nuclei_plane, spots_plane, spots_to_cyan=True)

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(rgb, interpolation="nearest")
ax.set_title("Merged overlay (nuclei + spot channel)")
ax.set_axis_off()

# overlay detected spots (u0-filtered) as cyan rings
xs = spots_df["x_px"].astype(float).values
ys = spots_df["y_px"].astype(float).values
ax.scatter(xs, ys, s=25, facecolors="none", edgecolors="cyan", linewidths=0.8, alpha=0.9)

plt.show()

# %% [markdown]
# ## Notes / next steps
#
# - If you want to "babysit" a **different** spot channel (e.g. protein vs DNA), set `channel_spots`
#   above (or in `integrated_ims.local.yaml`) accordingly.
# - For high-throughput trust building across hundreds of files, use:
#   - integrated batch runs (`drivers/run_integrated.py` with `batch_aggregate_spots: true`)
#   - the aggregate PPTX atlas (`drivers/generate_spot_atlas_pptx.py`) or notebook `03_generate_batch_spot_atlas_qc.py`.

