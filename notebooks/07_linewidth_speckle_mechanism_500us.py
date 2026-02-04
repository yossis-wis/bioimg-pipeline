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
# # Linewidth → speckle averaging at $500\,\mu\mathrm{s}$
#
# This notebook focuses on a single question:
#
# > How does **spectral linewidth** (or an intentional wavelength sweep) reduce
# > the **speckle level** for a **$500\,\mu\mathrm{s}$** exposure?
#
# We treat two regions separately:
#
# 1. **Inner ROI ("internal speckled region")** — the part you care about for single-molecule excitation.
# 2. **Edge band** around the illumination boundary — where diffraction/speckle can create a rough transition.
#
# The output is intentionally **visual**:
#
# - a small **Fourier-optics pipeline** view (stop → pupil → sample),
# - a **linewidth sweep** showing speckle metrics vs $\Delta\lambda_{\mathrm{src}}$,
# - side-by-side images and an **edge cross-section**.

# %% [markdown]
# ## 0) Imports + repo plumbing

# %%
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

from src.excitation_speckle_sim import (  # noqa: E402
    make_circular_pupil,
    simulate_excitation_speckle_field,
    square_roi_mask,
)
from src.illumination_speckle_metrics import (  # noqa: E402
    normalize_by_mask_mean,
    speckle_contrast,
    square_roi_region_masks,
)
from src.speckle_diversity_models import (  # noqa: E402
    estimate_n_lambda,
    estimate_speckle_spectral_corr_width_nm,
)

# %% [markdown]
# ## 1) Parameters (edit these)
#
# We keep the geometry simple and representative:
#
# - excitation wavelength: $\lambda_0 = 640\,\mathrm{nm}$
# - exposure: $\tau = 500\,\mu\mathrm{s}$
# - square ROI: $10\,\mu\mathrm{m}\times10\,\mu\mathrm{m}$
# - illumination NA: $\mathrm{NA}_{\mathrm{illum}}\approx0.3\,\mathrm{NA}_{\mathrm{obj}}$ (edit directly)
#
# We sweep the *effective source span* $\Delta\lambda_{\mathrm{src}}$.
#
# The mapping from $\Delta\lambda_{\mathrm{src}}$ to an effective number of independent spectral patterns
# is controlled by the **speckle spectral correlation width**
#
# $$
# \Delta\lambda_c \sim \frac{\lambda_0^2}{\Delta\mathrm{OPL}},
# $$
#
# where $\Delta\mathrm{OPL}$ is an *effective* optical-path-length spread of the interfering contributions.
# We treat $\Delta\mathrm{OPL}$ as a parameter (because it depends on your MMF, launch conditions,
# and how much mode mixing you actually have).

# %%
lambda0_nm = 640.0
lambda0_um = lambda0_nm * 1e-3

exposure_us = 500.0
exposure_s = exposure_us * 1e-6

roi_um = 10.0

# Example: if you fill 0.3 of a 1.45 NA objective pupil, NA_illum ≈ 0.435.
# Replace with your best estimate from BFP images.
na_illum = 0.43

# Speckle averaging in time (scrambler). Keep fixed to isolate the linewidth effect.
scrambler_hz = 10e3

# Key uncertainty: effective optical-path-length spread (ΔOPL) in meters.
# A few cm of ΔOPL makes Δλ_c ~ 0.01 nm at 640 nm (order-of-magnitude).
delta_opl_m = 3.0e-2

# Sampling: 100× objective with 6.5 µm camera pixel -> 65 nm at sample.
M_obj = 100.0
camera_pixel_um = 6.5
dx_um = camera_pixel_um / M_obj

# Grid: keep modest so the sweep is fast.
N_grid = 512

# Regions for metrics
inner_margin_um = 2.0
edge_band_um = 1.0

masks = square_roi_region_masks(
    n=N_grid,
    dx_um=dx_um,
    roi_um=roi_um,
    inner_margin_um=inner_margin_um,
    edge_band_um=edge_band_um,
)

coords_um = (np.arange(N_grid) - N_grid // 2) * dx_um

# %% [markdown]
# ## 2) Fourier-optics "mechanism" snapshot
#
# The kernel in `src/excitation_speckle_sim.py` is intentionally minimal:
#
# 1. Draw a random complex field in the stop plane.
# 2. Multiply by the square ROI mask (field stop).
# 3. FFT to the pupil-conjugate plane.
# 4. Apply a circular pupil low-pass set by $\mathrm{NA}_{\mathrm{illum}}$.
# 5. IFFT back to the sample plane and take $I = |U|^2$.
#
# The *speckle grain size* is set mainly by the pupil cutoff
# $f_c = \mathrm{NA}_{\mathrm{illum}}/\lambda$.
#
# Spectral linewidth enters through **incoherent averaging**:
# if the source spans many independent spectral correlation widths, the camera sees
# something closer to an **incoherent sum of intensities** from many effectively independent patterns.

# %%
rng = np.random.default_rng(0)

pupil = make_circular_pupil(n=N_grid, dx_um=dx_um, lambda_um=lambda0_um, na_illum=na_illum)
stop = square_roi_mask(n=N_grid, dx_um=dx_um, roi_um=roi_um).astype(float)

u0 = (rng.standard_normal((N_grid, N_grid)) + 1j * rng.standard_normal((N_grid, N_grid))) * stop
U0 = np.fft.fft2(u0)
U1 = U0 * pupil
u_img = np.fft.ifft2(U1)
I0 = (u_img.real * u_img.real + u_img.imag * u_img.imag)
I0n = normalize_by_mask_mean(I0, masks.inner)

fig = plt.figure(figsize=(12, 7), constrained_layout=True)
gs = fig.add_gridspec(2, 3)

ax = fig.add_subplot(gs[0, 0])
ax.imshow(stop, origin="lower")
ax.set_title("Field stop mask")
ax.set_xticks([])
ax.set_yticks([])

ax = fig.add_subplot(gs[0, 1])
ax.imshow(np.log10(1e-12 + np.abs(np.fft.fftshift(U0))), origin="lower")
ax.set_title(r"$\log_{10}|\mathcal{F}\{U\}|$ (before pupil)")
ax.set_xticks([])
ax.set_yticks([])

ax = fig.add_subplot(gs[0, 2])
ax.imshow(np.fft.fftshift(pupil.astype(float)), origin="lower")
ax.set_title("Pupil low-pass")
ax.set_xticks([])
ax.set_yticks([])

ax = fig.add_subplot(gs[1, 0])
ax.imshow(I0n, origin="lower", vmin=0, vmax=np.percentile(I0n[masks.roi], 99))
ax.set_title("Sample-plane intensity (1 realization)\nnormalized to inner mean")
ax.set_xticks([])
ax.set_yticks([])

ax = fig.add_subplot(gs[1, 1:3])
center = N_grid // 2
half = int(round((roi_um / dx_um) * 0.6))
sl = slice(center - half, center + half)
ax.imshow(I0n[sl, sl], origin="lower", vmin=0, vmax=np.percentile(I0n[masks.roi], 99))
ax.set_title("Zoom near ROI (shows grain scale)")
ax.set_xticks([])
ax.set_yticks([])

plt.show()

# %% [markdown]
# ## 3) Sweep linewidth and measure two region metrics
#
# We map linewidth to an effective number of independent spectral patterns via:
#
# $$
# \Delta\lambda_c \sim \frac{\lambda_0^2}{\Delta\mathrm{OPL}},\qquad
# N_{\lambda} \approx \left\lceil \frac{\Delta\lambda_{\mathrm{src}}}{\Delta\lambda_c} \right\rceil.
# $$
#
# Then, keeping the time diversity fixed ($f_{\mathrm{scr}}\approx10\,\mathrm{kHz}$),
# we simulate fields with $N_{\mathrm{src}}=N_{\lambda}$ and measure:
#
# - $C_{\mathrm{inner}} = \sigma/\mu$ in the inner ROI.
# - $C_{\mathrm{edge}}  = \sigma/\mu$ in an edge band inside the ROI.
#
# The key qualitative expectation:
#
# - increasing $\Delta\lambda_{\mathrm{src}}$ mainly reduces **contrast** (variance), not the **mean** edge roll-off.

# %%
delta_lambda_c_nm = estimate_speckle_spectral_corr_width_nm(lambda0_nm=lambda0_nm, optical_path_spread_m=delta_opl_m)
print(f"Δλ_c ≈ {delta_lambda_c_nm:.4f} nm (λ0={lambda0_nm:.0f} nm, ΔOPL={delta_opl_m:g} m)")

span_nm_list = [0.01, 0.03, 0.1, 0.3, 1.0, 2.0, 5.0]

# Average over a few seeds to stabilize the curve.
seeds = [0, 1, 2]

rows = []
for span_nm in span_nm_list:
    n_lambda = estimate_n_lambda(source_span_nm=float(span_nm), speckle_corr_width_nm=delta_lambda_c_nm, n_lines=1)
    for seed in seeds:
        I, meta = simulate_excitation_speckle_field(
            n=N_grid,
            dx_um=dx_um,
            roi_um=roi_um,
            lambda_um=lambda0_um,
            na_illum=na_illum,
            exposure_s=exposure_s,
            scrambler_hz=scrambler_hz,
            n_src=n_lambda,
            seed=int(seed),
        )
        In = normalize_by_mask_mean(I, masks.inner)
        rows.append(
            {
                "span_nm": span_nm,
                "N_lambda": n_lambda,
                "seed": seed,
                "N_eff_sim": meta.n_eff,
                "C_inner": speckle_contrast(In, masks.inner),
                "C_edge_in": speckle_contrast(In, masks.edge_in),
            }
        )

df = pd.DataFrame(rows)
df_mean = df.groupby(["span_nm", "N_lambda"], as_index=False).agg(
    C_inner_mean=("C_inner", "mean"),
    C_inner_std=("C_inner", "std"),
    C_edge_in_mean=("C_edge_in", "mean"),
    C_edge_in_std=("C_edge_in", "std"),
)
df_mean

# %%
fig = plt.figure(figsize=(9, 5), constrained_layout=True)
ax = fig.add_subplot(111)

ax.semilogx(df_mean["span_nm"], df_mean["C_inner_mean"], marker="o", label="inner ROI")
ax.semilogx(df_mean["span_nm"], df_mean["C_edge_in_mean"], marker="o", label="edge band (inside)")

ax.set_xlabel(r"effective source span $\Delta\lambda_{\mathrm{src}}$ [nm]")
ax.set_ylabel(r"measured contrast $C=\sigma/\mu$")
ax.set_title(r"Simulated contrast vs linewidth (fixed $\tau=500\,\mu\mathrm{s}$, $f_{\mathrm{scr}}=10\,\mathrm{kHz}$)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.show()

# %% [markdown]
# ## 4) Side-by-side images for three linewidths + an edge cross-section

# %%
pick_spans_nm = [0.01, 0.1, 2.0]

def edge_profile(In: np.ndarray, *, y_half_um: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Mean profile across x near the center (average over a thin horizontal strip)."""

    y_half_px = max(1, int(round(y_half_um / dx_um)))
    center = N_grid // 2
    sl_y = slice(center - y_half_px, center + y_half_px + 1)
    prof = np.mean(In[sl_y, :], axis=0)
    return coords_um, prof


fig, axes = plt.subplots(2, len(pick_spans_nm), figsize=(4.3 * len(pick_spans_nm), 7), constrained_layout=True)
if len(pick_spans_nm) == 1:
    axes = np.array([[axes[0]], [axes[1]]])

profiles = []
for j, span_nm in enumerate(pick_spans_nm):
    n_lambda = estimate_n_lambda(source_span_nm=float(span_nm), speckle_corr_width_nm=delta_lambda_c_nm, n_lines=1)
    I, _ = simulate_excitation_speckle_field(
        n=N_grid,
        dx_um=dx_um,
        roi_um=roi_um,
        lambda_um=lambda0_um,
        na_illum=na_illum,
        exposure_s=exposure_s,
        scrambler_hz=scrambler_hz,
        n_src=n_lambda,
        seed=0,
    )
    In = normalize_by_mask_mean(I, masks.inner)

    ax = axes[0, j]
    im = ax.imshow(In, origin="lower", vmin=0, vmax=np.percentile(In[masks.roi], 99))
    ax.set_title(f"Δλ={span_nm:g} nm (Nλ={n_lambda})")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Edge zoom: right edge patch
    half_roi_px = int(round(0.5 * roi_um / dx_um))
    cx = N_grid // 2
    cy = N_grid // 2
    x0 = cx + half_roi_px - 64
    x1 = cx + half_roi_px + 64
    y0 = cy - 64
    y1 = cy + 64
    patch = In[y0:y1, x0:x1]

    ax = axes[1, j]
    im2 = ax.imshow(patch, origin="lower", vmin=0, vmax=np.percentile(In[masks.roi], 99))
    ax.set_title("edge patch")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    x_um, prof = edge_profile(In)
    profiles.append((span_nm, x_um, prof))

plt.show()

# %%
fig = plt.figure(figsize=(9, 4.5), constrained_layout=True)
ax = fig.add_subplot(111)

for span_nm, x_um, prof in profiles:
    ax.plot(x_um, prof, label=f"Δλ={span_nm:g} nm")

ax.axvline(0.5 * roi_um, linestyle="--", linewidth=1, label="ROI edge")
ax.axvline(-0.5 * roi_um, linestyle="--", linewidth=1)

ax.set_xlim(-0.8 * roi_um, 0.8 * roi_um)
ax.set_xlabel("x [µm]")
ax.set_ylabel(r"normalized intensity")
ax.set_title("Mean cross-section across the ROI (edge behavior)")
ax.grid(True, alpha=0.3)
ax.legend()
plt.show()

# %% [markdown]
# ## 5) Interpretation checklist
#
# - If $C_{\mathrm{inner}}$ decreases strongly with $\Delta\lambda_{\mathrm{src}}$ in your measurements,
#   then linewidth (or a small wavelength sweep) is a powerful way to make $500\,\mu\mathrm{s}$ viable.
#
# - If the **edge** contrast decreases much less than the **inner** contrast, the edge is probably dominated
#   by the deterministic roll-off from the finite illumination NA and relay aberrations rather than speckle.
#
# - If your camera image of the fiber near-field is not uniform even with a large linewidth:
#   that can be a **modal power distribution** problem (not an interference problem).
#   A wide linewidth suppresses intermode interference, but it does not guarantee a flat near-field intensity.
