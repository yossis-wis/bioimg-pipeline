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
# # Excitation speckle → Slice0 FP/FN proxy (single-protein widefield)
#
# This notebook is the “stable context” notebook for reasoning about:
#
# - **multimode fiber + scrambler + square field stop** widefield excitation, and
# - how excitation **speckle / spillover tails** can translate into **false positives / false negatives**
#   in your **Slice0** spot detector.
#
# It is intentionally written in **Jupytext percent format** so you can run it in:
# - **JupyterLab** (notebook feel, easy figure zoom), and
# - **Spyder** (variable explorer + interactive plots).
#
# Companion doc:
# - `docs/ILLUMINATION_SPECKLE.md`
#
# For the **500 µs** design regime (10 µm × 10 µm, 10–30 kW/cm²) and explicit
# parameter sweeps (power budget, stop sizing, spectral vs temporal speckle
# averaging), see:
# - `notebooks/06_mmf_illumination_500us_design.py`

# %% [markdown]
# ## 0) Imports + repo plumbing

# %%
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import gamma as gamma_dist

# Make inline plots work nicely in Jupyter, while staying interactive in Spyder.
if "ipykernel" in sys.modules:
    try:
        if matplotlib.get_backend().lower() == "agg":
            matplotlib.use("module://matplotlib_inline.backend_inline", force=True)
    except Exception:
        pass


def find_repo_root(start: Path) -> Path:
    """Find repo root by walking upward until we see (src/, environment.yml)."""
    p = start.resolve()
    for parent in [p, *p.parents]:
        if (parent / "src").is_dir() and (parent / "environment.yml").exists():
            return parent
    return p


REPO_ROOT = find_repo_root(Path.cwd())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Repo kernels (pure computation)
from src.excitation_speckle_sim import simulate_excitation_speckle_field, square_roi_mask  # noqa: E402
from src.slice0_kernel import Slice0Params, detect_spots  # noqa: E402

# %% [markdown]
# ## 1) Physical / design parameters (edit these)
#
# Key conceptual split:
#
# - **ROI size** is set by a **field stop** (field plane).
# - **Speckle grain size** and **edge roll-off** are set by **effective illumination NA**
#   \(\mathrm{NA}_{\mathrm{illum}}\) (pupil plane / BFP underfill).
#
# For your use case:
# - \(\lambda_{\rm exc}=640\,\mathrm{nm}\)
# - \(\tau = 5\,\mathrm{ms}\)
# - ROI = 30 µm × 30 µm
# - pixel size: 6.5 µm on camera, 100× → 65 nm at sample

# %%
# ---------------- Excitation (sample plane) ----------------
lambda_exc_nm = 640.0
roi_um = 30.0

exposure_ms = 5.0
scrambler_f_khz = 10.0  # Newport-style 10 kHz is a common reference point

# Effective illumination NA at the sample (UNDERFILL knob).
# Start with your estimate ~0.05 (very underfilled).
NA_illum = 0.05

# “M^2 proxy” (see docs/ILLUMINATION_SPECKLE.md).
# We map M^2 → N_src ≈ round(M^2), i.e. extra incoherent diversity.
# - M2 ~ 1.1 (near TEM00) → N_src ≈ 1
# - M2 ~ 10 (very poor beam) → N_src ≈ 10
M2_proxy = 1.1

# ---------------- Microscope sampling ----------------
M_obj = 100.0
camera_pixel_um = 6.5
dx_um = camera_pixel_um / M_obj  # sample-plane pixel pitch

# Simulation grid (FOV ≈ N_grid * dx_um). Must be large enough for ROI + spillover.
N_grid = 768  # ~50 µm FOV at 65 nm/px

# Visualization patch at the ROI edge
patch_px = 64
outside_cols_px = 16  # rightmost columns are outside the ideal ROI

# Inner ROI margin: exclude edge roll-off when computing “inside ROI” stats
inner_margin_um = 2.0

seed = 0  # reproducibility

# %% [markdown]
# ### 1.1 Representative fiber choice (for context)
#
# We keep using a single representative multimode fiber spec (edit if needed):
#
# - core: **400 µm**
# - \(\mathrm{NA}_{\rm fiber}=0.22\)
#
# This matches a very common “easy coupling / many modes” class. The notebook does **not**
# simulate exact guided modes; it uses speckle statistics + Fourier low-pass filtering instead.

# %%
fiber_core_um = 400.0
NA_fiber = 0.22

# Step-index V-number (context only)
lambda_um = lambda_exc_nm * 1e-3
a_um = 0.5 * fiber_core_um
V = 2.0 * np.pi * a_um * NA_fiber / lambda_um
M_modes_est = 0.5 * V * V  # ~V^2/2 (order-of-magnitude, includes polarization loosely)

print(f"Fiber context: core={fiber_core_um:.0f} µm, NA={NA_fiber:.2f}, V≈{V:.1f}, modes≈{M_modes_est:,.0f}")

# %% [markdown]
# ## 2) Simulate one **excitation-only** frame (time-averaged over 5 ms)
#
# We synthesize the excitation intensity field \(I_{\rm exc}(x,y)\) in arbitrary units.
# We then normalize by the **inner ROI mean** so that the inside-ROI distribution is easy to read.

# %%
exposure_s = exposure_ms * 1e-3
scrambler_hz = scrambler_f_khz * 1e3
N_src = max(1, int(np.round(M2_proxy)))  # intentionally simple mapping

I_exc_raw, meta = simulate_excitation_speckle_field(
    n=N_grid,
    dx_um=dx_um,
    roi_um=roi_um,
    lambda_um=lambda_um,
    na_illum=NA_illum,
    exposure_s=exposure_s,
    scrambler_hz=scrambler_hz,
    n_src=N_src,
    seed=seed,
)

roi_mask = square_roi_mask(N_grid, dx_um, roi_um)
coords_um = (np.arange(N_grid) - N_grid // 2) * dx_um
X_um, Y_um = np.meshgrid(coords_um, coords_um, indexing="xy")

inner_mask = roi_mask.copy()
inner_mask &= (np.abs(X_um) <= 0.5 * roi_um - inner_margin_um) & (np.abs(Y_um) <= 0.5 * roi_um - inner_margin_um)

mean_inner = float(I_exc_raw[inner_mask].mean())
I_exc = I_exc_raw / max(mean_inner, 1e-12)

C_inner = float(I_exc[inner_mask].std() / I_exc[inner_mask].mean())

# Speckle grain size estimate
delta_speckle_um = lambda_um / (2.0 * NA_illum)
delta_speckle_px = delta_speckle_um / dx_um

q05, q50, q95 = np.quantile(I_exc[inner_mask], [0.05, 0.50, 0.95])

print(
    "Excitation summary\n"
    f"  NA_illum={NA_illum:.3f}\n"
    f"  exposure={exposure_ms:.1f} ms, scrambler={scrambler_f_khz:.1f} kHz → N_time={meta.n_time}, N_src={meta.n_src}, N_eff={meta.n_eff}\n"
    f"  C_inner≈{C_inner:.3f}\n"
    f"  speckle grain ~ λ/(2 NA) ≈ {delta_speckle_um:.2f} µm ≈ {delta_speckle_px:.0f} px\n"
    f"  inner ROI quantiles: q05={q05:.3f}, q50={q50:.3f}, q95={q95:.3f}"
)

# %% [markdown]
# ## 3) Visualize excitation field
#
# We show three things:
# 1) full field (ROI + surroundings),
# 2) a **pixel-level 64×64 edge patch** (rightmost 16 px outside),
# 3) the **full inside-ROI distribution shape** (histogram) + a gamma-model overlay.
#
# The gamma overlay is not “truth,” but it is a good sanity check:
# averaging \(N_{\rm eff}\) independent speckles gives a gamma-like intensity distribution.

# %%
def extract_edge_patch(I: np.ndarray) -> tuple[np.ndarray, float]:
    # Patch centered vertically at ROI center, and horizontally at the right ROI edge.
    y0 = N_grid // 2 - patch_px // 2
    y1 = y0 + patch_px

    # ROI right edge in pixels:
    half_roi_px = int(round(0.5 * roi_um / dx_um))
    x_edge = N_grid // 2 + half_roi_px  # first pixel just outside ideal ROI (approximately)

    # Patch spans from (x_edge - (patch_px - outside_cols_px)) to (x_edge + outside_cols_px)
    x0 = x_edge - (patch_px - outside_cols_px)
    x1 = x0 + patch_px

    patch = I[y0:y1, x0:x1]
    x_rel_um = (np.arange(patch_px) - (patch_px - outside_cols_px)) * dx_um
    return patch, float(x_rel_um[patch_px - outside_cols_px])  # x_rel at ideal edge (≈0)


patch, _ = extract_edge_patch(I_exc)

# Gamma overlay: if I is normalized to mean=1, then Gamma(k, theta=1/k) has mean 1.
k_eff = meta.n_eff
xs = np.linspace(0.0, np.quantile(I_exc[inner_mask], 0.999), 400)
pdf_gamma = gamma_dist.pdf(xs, a=k_eff, scale=1.0 / k_eff)

fig = plt.figure(figsize=(15, 5), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.2])

# (1) full field
ax0 = fig.add_subplot(gs[0, 0])
im0 = ax0.imshow(
    I_exc,
    origin="lower",
    extent=[coords_um[0], coords_um[-1], coords_um[0], coords_um[-1]],
    interpolation="nearest",
)
ax0.add_patch(
    plt.Rectangle(
        (-0.5 * roi_um, -0.5 * roi_um),
        roi_um,
        roi_um,
        fill=False,
        linewidth=2,
        color="w",
    )
)
ax0.set_title("Excitation field (normalized)\nROI outline")
ax0.set_xlabel("x [µm]")
ax0.set_ylabel("y [µm]")
plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04, label=r"$I_{\rm exc}/\langle I\rangle_{\rm inner}$")

# (2) edge patch
ax1 = fig.add_subplot(gs[0, 1])
im1 = ax1.imshow(
    patch,
    origin="lower",
    extent=[-(patch_px - outside_cols_px) * dx_um, outside_cols_px * dx_um, -(patch_px / 2) * dx_um, (patch_px / 2) * dx_um],
    interpolation="nearest",
)
ax1.axvline(0.0, color="w", linestyle="--", linewidth=2)
ax1.set_title(f"{patch_px}×{patch_px} edge patch (pixel-level)\n(rightmost {outside_cols_px} px outside ROI)")
ax1.set_xlabel("x_rel to ideal edge [µm]")
ax1.set_ylabel("y [µm]")
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

# (3) distribution
ax2 = fig.add_subplot(gs[0, 2])
vals = I_exc[inner_mask].ravel()
ax2.hist(vals, bins=80, density=True, alpha=0.7, label="sim (inner ROI)")
ax2.plot(xs, pdf_gamma, linewidth=3, label=f"Gamma model (k={k_eff})")
for q, ls, lab in [(q05, ":", "q05"), (q50, "--", "q50"), (q95, ":", "q95")]:
    ax2.axvline(q, color="k", linestyle=ls, linewidth=1.5)
ax2.set_title("Inside-ROI intensity distribution")
ax2.set_xlabel(r"$I_{\rm exc}/\langle I\rangle_{\rm inner}$")
ax2.set_ylabel("PDF")
ax2.legend()

fig.suptitle(
    f"Excitation field | λ={lambda_exc_nm:.0f} nm | ROI={roi_um:.0f} µm | NA_illum={NA_illum:.3f} | "
    f"τ={exposure_ms:.1f} ms | f_scr={scrambler_f_khz:.1f} kHz | N_eff≈{meta.n_eff} | C_inner≈{C_inner:.3f}",
    fontsize=12,
)
plt.show()

# %% [markdown]
# ## 4) Build a sparse-emitter fluorescence frame (single 5 ms frame)
#
# We synthesize a single 2D fluorescence image with:
# - sparse non-overlapping emitters in the ROI,
# - brightness proportional to local excitation intensity,
# - background proportional to excitation (optional),
# - Poisson shot noise.
#
# Then we run the **actual Slice0 detector** (TrackMate-style LoG + in5/out0 u0 threshold)
# and compute TP/FP/FN plus a finite-TN proxy via “decoy sites”.

# %%
@dataclass(frozen=True)
class EmitterSimConfig:
    n_emitters: int = 60
    min_separation_px: int = 18  # ensure non-overlap (rough)
    photons_per_emitter_mean: float = 1200.0
    photons_per_emitter_sigma_frac: float = 0.25  # lognormal-ish variability (implemented as normal clamp)
    bg_photons_flat: float = 5.0
    bg_photons_scale_exc: float = 15.0  # background ∝ excitation

    psf_sigma_px: float = 2.8  # should be comparable to Slice0 LoG sigma
    psf_kernel_radius_px: int = 12  # kernel size = (2R+1)


cfg = EmitterSimConfig()

# Slice0 params (tune u0_min here)
params = Slice0Params(
    pixel_size_nm=dx_um * 1e3,  # µm → nm
    spot_radius_nm=270.0,       # sets LoG scale in calibrated units
    q_min=1.0,
    u0_min=30.0,
)

# Derive a PSF sigma that roughly matches the detector's assumed scale
psf_sigma_px = cfg.psf_sigma_px

# Precompute a Gaussian PSF kernel (normalized to sum=1)
R = cfg.psf_kernel_radius_px
yy, xx = np.mgrid[-R : R + 1, -R : R + 1]
psf = np.exp(-(xx * xx + yy * yy) / (2.0 * psf_sigma_px * psf_sigma_px))
psf /= psf.sum()

# Utility: place emitters with a minimum separation in px
rng = np.random.default_rng(seed + 123)


def place_emitters_nonoverlap(
    n_emitters: int,
    roi_mask: np.ndarray,
    min_sep_px: int,
    rng: np.random.Generator,
) -> np.ndarray:
    ys, xs = np.where(roi_mask)
    coords = np.stack([ys, xs], axis=1)
    chosen: list[tuple[int, int]] = []

    # random order
    idx = rng.permutation(coords.shape[0])
    for j in idx:
        y, x = int(coords[j, 0]), int(coords[j, 1])
        ok = True
        for (y0, x0) in chosen:
            if (y - y0) * (y - y0) + (x - x0) * (x - x0) < min_sep_px * min_sep_px:
                ok = False
                break
        if ok:
            chosen.append((y, x))
            if len(chosen) >= n_emitters:
                break
    if len(chosen) < n_emitters:
        raise RuntimeError(f"Could only place {len(chosen)} emitters with min_sep_px={min_sep_px}")
    return np.array(chosen, dtype=int)


# Place emitters inside *inner* ROI to avoid edge effects (optional)
emit_xy = place_emitters_nonoverlap(cfg.n_emitters, inner_mask, cfg.min_separation_px, rng)
emit_y = emit_xy[:, 0]
emit_x = emit_xy[:, 1]

# Ground-truth per-emitter brightness
photons = cfg.photons_per_emitter_mean * (1.0 + cfg.photons_per_emitter_sigma_frac * rng.standard_normal(cfg.n_emitters))
photons = np.clip(photons, 0.2 * cfg.photons_per_emitter_mean, 5.0 * cfg.photons_per_emitter_mean)

# Local excitation factor
exc_loc = I_exc[emit_y, emit_x]

# Build noiseless image (photons)
img = np.zeros((N_grid, N_grid), dtype=np.float64)

# Background
img += cfg.bg_photons_flat + cfg.bg_photons_scale_exc * I_exc

# Add emitters (convolve by PSF kernel)
for y0, x0, p0, e0 in zip(emit_y, emit_x, photons, exc_loc):
    amp = float(p0 * e0)
    y1 = y0 - R
    y2 = y0 + R + 1
    x1 = x0 - R
    x2 = x0 + R + 1
    if y1 < 0 or x1 < 0 or y2 > N_grid or x2 > N_grid:
        continue
    img[y1:y2, x1:x2] += amp * psf

# Poisson shot noise
img_noisy = rng.poisson(img).astype(np.float32)

# Optional read noise (small; set to 0 to disable)
read_noise_sigma = 0.0
if read_noise_sigma > 0:
    img_noisy = img_noisy + rng.normal(0.0, read_noise_sigma, img_noisy.shape).astype(np.float32)

# %% [markdown]
# ## 5) Run Slice0 spot detection + evaluate FP/FN
#
# We match detections to ground truth emitters within a match radius.

# %%
df = detect_spots(img_noisy, params)

# Match detections to GT
match_r_px = 3.0

det_xy = df[["y_px", "x_px"]].to_numpy(dtype=float)
gt_xy = np.stack([emit_y.astype(float), emit_x.astype(float)], axis=1)

# For each detection, nearest GT distance
tp_det = np.zeros(det_xy.shape[0], dtype=bool)
gt_matched = np.zeros(gt_xy.shape[0], dtype=bool)

for i, (y, x) in enumerate(det_xy):
    dy = gt_xy[:, 0] - y
    dx = gt_xy[:, 1] - x
    d2 = dy * dy + dx * dx
    j = int(np.argmin(d2))
    if d2[j] <= match_r_px * match_r_px and not gt_matched[j]:
        tp_det[i] = True
        gt_matched[j] = True

fp_det = ~tp_det
TP = int(tp_det.sum())
FP = int(fp_det.sum())
FN = int((~gt_matched).sum())

precision = TP / max(TP + FP, 1)
recall = TP / max(TP + FN, 1)

# Finite TN proxy: sample decoy sites in the inner ROI that are not near any GT emitter
n_decoy = 200
decoy_xy: list[tuple[int, int]] = []
tries = 0
while len(decoy_xy) < n_decoy and tries < 200000:
    tries += 1
    y = int(rng.integers(0, N_grid))
    x = int(rng.integers(0, N_grid))
    if not inner_mask[y, x]:
        continue
    # far from GT
    dy = emit_y - y
    dx = emit_x - x
    if np.any(dy * dy + dx * dx <= (match_r_px * 2.0) ** 2):
        continue
    decoy_xy.append((y, x))

decoy_xy_arr = np.array(decoy_xy, dtype=float)
TN = 0
FP_decoy = 0
for (y, x) in decoy_xy_arr:
    if det_xy.shape[0] == 0:
        TN += 1
        continue
    dy = det_xy[:, 0] - y
    dx = det_xy[:, 1] - x
    if np.any(dy * dy + dx * dx <= match_r_px * match_r_px):
        FP_decoy += 1
    else:
        TN += 1

print(
    f"Slice0 results\n"
    f"  detections={len(df)} | GT={cfg.n_emitters}\n"
    f"  TP={TP}, FP={FP}, FN={FN}, TN*={TN} (decoy sites)\n"
    f"  precision={precision:.3f}, recall={recall:.3f}"
)

# %% [markdown]
# ## 6) Plots: synthetic frame, confusion matrix proxy, u0 ECDF
#
# - Top-left: noisy frame + ROI outline + GT emitter locations + TP/FP detections
# - Top-right: confusion matrix proxy (TN is from decoy sites)
# - Bottom-left: ECDF of detected-candidate u0, colored by TP vs FP
# - Bottom-right: “why FNs happen” proxy plot: local excitation vs GT emitter brightness (u0-like)

# %%
# Prepare classifications for plotting
df = df.copy()
df["is_tp"] = tp_det
df["is_fp"] = fp_det

# Confusion matrix (proxy)
cm = np.array([[TP, FN], [FP_decoy, TN]], dtype=int)

fig = plt.figure(figsize=(13, 9), constrained_layout=True)
gs = fig.add_gridspec(2, 2)

# (A) Synthetic frame
axA = fig.add_subplot(gs[0, 0])
imA = axA.imshow(
    img_noisy,
    origin="lower",
    extent=[coords_um[0], coords_um[-1], coords_um[0], coords_um[-1]],
    interpolation="nearest",
)
axA.add_patch(
    plt.Rectangle(
        (-0.5 * roi_um, -0.5 * roi_um),
        roi_um,
        roi_um,
        fill=False,
        linewidth=2,
        color="w",
    )
)
axA.scatter(
    (emit_x - N_grid // 2) * dx_um,
    (emit_y - N_grid // 2) * dx_um,
    s=30,
    facecolors="none",
    edgecolors="w",
    linewidths=1.5,
    label="GT emitters",
)
axA.scatter(
    (df.loc[df["is_tp"], "x_px"] - N_grid // 2) * dx_um,
    (df.loc[df["is_tp"], "y_px"] - N_grid // 2) * dx_um,
    s=20,
    c="lime",
    label="TP detections",
)
axA.scatter(
    (df.loc[df["is_fp"], "x_px"] - N_grid // 2) * dx_um,
    (df.loc[df["is_fp"], "y_px"] - N_grid // 2) * dx_um,
    s=40,
    marker="x",
    c="r",
    label="FP detections",
)
axA.set_title("Synthetic fluorescence frame (ROI + spillover)")
axA.set_xlabel("x [µm]")
axA.set_ylabel("y [µm]")
plt.colorbar(imA, ax=axA, fraction=0.046, pad=0.04, label="counts (arb.)")
axA.legend(loc="upper right", framealpha=0.85)

# (B) Confusion matrix proxy
axB = fig.add_subplot(gs[0, 1])
imB = axB.imshow(cm, interpolation="nearest")
axB.set_xticks([0, 1])
axB.set_yticks([0, 1])
axB.set_xticklabels(["pred spot", "pred none"])
axB.set_yticklabels(["true spot", "true none*"])
axB.set_title("Confusion matrix (proxy)\n(*true none = decoy sites)")
for (i, j), v in np.ndenumerate(cm):
    axB.text(j, i, str(v), ha="center", va="center", color="w" if v > cm.max() / 2 else "k", fontsize=14)
plt.colorbar(imB, ax=axB, fraction=0.046, pad=0.04)

# (C) u0 ECDF
axC = fig.add_subplot(gs[1, 0])
if len(df) > 0:
    u0 = df["u0"].to_numpy(dtype=float)
    order = np.argsort(u0)
    u0s = u0[order]
    ecdf = (np.arange(len(u0s)) + 1) / len(u0s)

    colors = np.where(df["is_tp"].to_numpy()[order], "lime", "red")
    axC.plot(u0s, ecdf, "-", alpha=0.4)
    axC.scatter(u0s, ecdf, s=30, c=colors, edgecolors="k", linewidths=0.3)

axC.axvline(params.u0_min, color="k", linestyle="--", linewidth=1.5, label=f"u0_min={params.u0_min:g}")
axC.set_title("Candidate u0 ECDF (accepted candidates only)")
axC.set_xlabel(r"$u_0=\langle I\rangle_{\rm in5}-\mathrm{median}(I)_{\rm out0}$")
axC.set_ylabel("ECDF")
axC.legend(loc="lower right")

# (D) Why FNs happen: local excitation vs GT emitter amplitude
axD = fig.add_subplot(gs[1, 1])
axD.scatter(exc_loc, photons, s=35, facecolors="none", edgecolors="k", label="GT emitters")
axD.scatter(exc_loc[gt_matched], photons[gt_matched], s=35, c="lime", label="GT detected (TP)")
axD.scatter(exc_loc[~gt_matched], photons[~gt_matched], s=80, marker="x", c="orange", label="GT missed (FN)")
axD.set_title("Why FNs happen (proxy)\nGT brightness vs local excitation")
axD.set_xlabel(r"local excitation $I_{\rm exc}/\langle I\rangle_{\rm inner}$")
axD.set_ylabel("emitter photons (drawn)")
axD.legend(loc="lower right")

fig.suptitle(
    f"Sparse emitter sim | λ_exc={lambda_exc_nm:.0f} nm | ROI={roi_um:.0f} µm | NA_illum={NA_illum:.3f} | "
    f"τ={exposure_ms:.1f} ms | f_scr={scrambler_f_khz:.1f} kHz | N_eff≈{meta.n_eff} | "
    f"TP={TP}, FP={FP}, FN={FN} | precision={precision:.3f}, recall={recall:.3f}",
    fontsize=12,
)
plt.show()

# %% [markdown]
# ## 7) Next steps (recommended exploration ladder)
#
# 1) Sweep **scrambler frequency** (e.g. 0 → 1 → 10 → 30 kHz) at fixed \(\mathrm{NA}_{\rm illum}\).
# 2) Sweep **\(\mathrm{NA}_{\rm illum}\)** by changing BFP fill (keep ROI constant).
# 3) Sweep **M² proxy** cautiously (treat as a knob for extra incoherent diversity).
# 4) Track how:
#    - the *excitation-only* inner-ROI distribution (hist/ECDF) changes,
#    - the edge patch changes (spillover),
#    - TP/FP/FN changes for the *same* Slice0 parameters.
#
# For realism:
# - Measure an actual BFP pupil-fill image and set \(\mathrm{NA}_{\rm illum}\) from that underfill ratio.
# - Measure speckle contrast vs exposure and scrambler state to calibrate \(N_{\rm eff}\).

