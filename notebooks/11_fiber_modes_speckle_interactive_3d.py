# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.20.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fiber modes + multimode speckle (interactive 3D intuition builder)
#
# This notebook is designed for **visual learners**.
#
# It builds intuition in small steps:
#
# 1. **What is a “mode”** (in a fiber) and what do mode patterns look like?
# 2. **Single-mode fiber**: why the output is stable (no modal speckle).
# 3. **Multimode fiber**: why superposition of many modes creates a granular (speckle-like) pattern.
# 4. **Modal noise / speckle drift**: why tiny phase changes can drastically change the pattern.
# 5. **Averaging** (time / spectrum / polarization): why contrast drops like $C\\sim 1/\\sqrt{N_{\\mathrm{eff}}}$.
#
# This notebook complements:
#
# - `notebooks/09_mmf_wide_linewidth_scrambling_fourier_optics.py` (dispersion + linewidth scaling)
# - `notebooks/10_mmf_robust_setup_linewidth_stepindex_kohler.py` (practical failure modes)
# - `notebooks/08_cni_laser_system_diagrams.py` (your Approach A: SM fibers, Approach B: common MMF)
#
# **Important scope note:** the “modes” here are a clean *disk-basis surrogate* (Bessel modes),
# not a full vector LP-mode solver. They are accurate enough for understanding *why* modes matter.

# %% [markdown]
# ## 0) Imports + repo plumbing

# %%
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Plotly is used for interactive 3D figures.
import plotly.io as pio

# If you prefer a specific renderer (browser / notebook), set it here.
# Examples:
#   pio.renderers.default = "browser"
#   pio.renderers.default = "notebook"
#   pio.renderers.default = "vscode"
pio.renderers.default = pio.renderers.default  # no-op; keeps the cell explicit


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

from src.fiber_modes import (  # noqa: E402
    DiskModeIndex,
    average_uncorrelated_intensities,
    disk_bessel_mode_field,
    disk_mode_indices,
    intensity_from_field,
    make_core_grid,
    precompute_mode_stack,
    random_complex_coeffs,
    speckle_contrast,
    superpose_modes,
)
from src.fiber_modes_plotly import (  # noqa: E402
    make_surface_stack,
    surface_stack_figure,
)

from src.mmf_fiber_speckle import (  # noqa: E402
    MultimodeFiber,
    speckle_spectral_corr_width_nm_for_fiber,
)
from src.speckle_diversity_models import (  # noqa: E402
    DiversityBudget,
    estimate_n_eff,
    n_time_samples,
    speckle_contrast_from_n_eff,
)


# %% [markdown]
# ## 1) What is a mode?
#
# A (guided) **mode** is a transverse field pattern that can propagate down the fiber
# while keeping the same shape (up to a phase factor).
#
# In a **single-mode** fiber, only the fundamental mode exists (approximately a Gaussian).
# In a **multimode** fiber, many orthogonal patterns can propagate.
#
# Here we visualize a *disk-basis* surrogate for fiber-core modes:
#
# - Each mode has a distinct number of **radial rings** ($m$) and **angular lobes** ($\\ell$).
# - For $\\ell>0$ there are two angular variants (cos/sin), reflecting a common degeneracy in real fibers.

# %%
# Grid representing a 400 µm-core MMF (radius 200 µm).
n_grid = 201
core_radius_um = 200.0

x_um, y_um, mask, dx_um = make_core_grid(n=n_grid, core_radius_um=core_radius_um)

# Build a small gallery: ℓ up to 3, m up to 2 -> ~14 modes (including cos/sin degeneracy).
modes_gallery: list[DiskModeIndex] = disk_mode_indices(max_l=3, max_m=2, include_sin=True)

z_list = []
labels = []

abs_max = 0.0
for mode in modes_gallery:
    u = disk_bessel_mode_field(mode, x_um=x_um, y_um=y_um, core_radius_um=core_radius_um, mask=mask, normalize=True)
    u = np.where(mask, u, np.nan)
    abs_max = max(abs_max, float(np.nanmax(np.abs(u))))
    z_list.append(u)
    labels.append(mode.label)

surf = make_surface_stack(x_um=x_um, y_um=y_um, z_list=z_list, labels=labels)

fig_modes = surface_stack_figure(
    surf,
    title="Mode gallery (fiber-core surrogate). Rotate + use the slider.",
    z_title="Mode amplitude (a.u.)",
    colorscale="RdBu",
    show_colorbar=False,
    z_range=(-abs_max, abs_max),
    aspectmode="data",
)

fig_modes

# %% [markdown]
# ## 2) Single-mode vs multimode: why speckle appears in MMF
#
# A simple mental model:
#
# - **Single-mode:** there is effectively one transverse field pattern, so the output intensity is stable.
# - **Multimode:** many patterns exist. The output field is a *sum* of these modes:
#
# $$U(x,y) = \sum_{k=1}^{M} a_k\\,u_k(x,y)\\,e^{i\\phi_k}.$$
#
# The phases $\\phi_k$ are typically very sensitive to fiber bending/temperature/vibration.
# When many modes contribute with “effectively random” phases, the intensity
# $I = |U|^2$ becomes granular (speckle-like).

# %%
# "Multimode" demo: superpose many modes with random phases.

n_speckle_modes = 30
modes_all = disk_mode_indices(max_l=6, max_m=6, include_sin=True)
modes = modes_all[:n_speckle_modes]

mode_stack = precompute_mode_stack(modes, x_um=x_um, y_um=y_um, core_radius_um=core_radius_um, mask=mask)

coeffs = random_complex_coeffs(n_modes=mode_stack.shape[0], seed=0)
u = superpose_modes(mode_stack, coeffs)
I = intensity_from_field(u)
I_norm = I / float(np.mean(I[mask]))

C = speckle_contrast(I_norm, mask)
C

# %%
surf_single = make_surface_stack(
    x_um=x_um,
    y_um=y_um,
    z_list=[np.where(mask, I_norm, np.nan)],
    labels=[f"Random superposition of {n_speckle_modes} modes (C={C:.2f})"],
)

fig_mmf = surface_stack_figure(
    surf_single,
    title="Multimode output intensity (near-field at fiber face): speckle-like structure",
    z_title="Normalized intensity (mean=1)",
    colorscale="Viridis",
    show_colorbar=False,
    z_range=None,
)

fig_mmf

# %% [markdown]
# ## 3) Why MMF speckle can drift: tiny phase noise changes the pattern
#
# This section is the key "potential issue" for the common-MMF approach:
#
# - If your setup leaves you with a **static speckle pattern**, it is *not* reliably calibratable,
#   because the pattern can decorrelate from **very small** mechanical/thermal perturbations.
#
# Below, we keep the same mode amplitudes but add a per-mode phase perturbation with RMS $\\sigma_\\phi$.
# Watch how quickly the pattern decorrelates as $\\sigma_\\phi$ reaches fractions of a radian.

# %%
# Build a slider over phase-noise RMS levels.

rng = np.random.default_rng(1)
delta_unit = rng.normal(loc=0.0, scale=1.0, size=mode_stack.shape[0])

# Baseline
u0 = superpose_modes(mode_stack, coeffs)
I0 = intensity_from_field(u0)
I0n = I0 / float(np.mean(I0[mask]))

delta_list = [0.0, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60]

z_list = []
labels = []
for delta_rms in delta_list:
    coeffs_d = coeffs * np.exp(1j * (delta_unit * float(delta_rms)))
    u_d = superpose_modes(mode_stack, coeffs_d)
    I_d = intensity_from_field(u_d)
    I_dn = I_d / float(np.mean(I_d[mask]))

    a = I0n[mask].ravel()
    b = I_dn[mask].ravel()
    corr = float(np.corrcoef(a, b)[0, 1])

    C_d = speckle_contrast(I_dn, mask)

    z_list.append(np.where(mask, I_dn, np.nan))
    labels.append(f"σϕ={delta_rms:.2f} rad (corr={corr:.2f}, C={C_d:.2f})")

surf_drift = make_surface_stack(x_um=x_um, y_um=y_um, z_list=z_list, labels=labels)

fig_drift = surface_stack_figure(
    surf_drift,
    title="Speckle drift vs per-mode phase noise (MMF modal noise intuition)",
    z_title="Normalized intensity (mean=1)",
    colorscale="Viridis",
    show_colorbar=False,
)

fig_drift

# %% [markdown]
# ## 4) Why averaging helps: $C \\sim 1/\\sqrt{N}$
#
# If you can average $N$ **independent** speckle realizations (incoherently), contrast drops roughly as:
#
# $$C \\approx \\frac{1}{\\sqrt{N}}.$$
#
# In your MMF approach, “independent realizations” can come from:
#
# - time scrambling ($N_t$) during a 500 µs exposure
# - spectral diversity ($N_\\lambda$) if linewidth/sweep spans many decorrelation widths
# - polarization diversity ($N_{\\mathrm{pol}}$)
# - angular diversity ($N_{\\mathrm{angle}}$) depending on relay / pupil fill behavior
#
# Below we show the simplest case: averaging $N$ uncorrelated random-phase patterns.

# %%
n_avg_list = [1, 2, 4, 8, 16, 32, 64]

z_list = []
labels = []
for n_avg in n_avg_list:
    Iavg = average_uncorrelated_intensities(mode_stack, n_avg=int(n_avg), seed=0)
    Iavgn = Iavg / float(np.mean(Iavg[mask]))
    Cn = speckle_contrast(Iavgn, mask)
    z_list.append(np.where(mask, Iavgn, np.nan))
    labels.append(f"N={n_avg}  (C={Cn:.2f}, 1/√N={1/np.sqrt(n_avg):.2f})")

surf_avg = make_surface_stack(x_um=x_um, y_um=y_um, z_list=z_list, labels=labels)

fig_avg = surface_stack_figure(
    surf_avg,
    title="Speckle averaging demo: incoherent average of N independent patterns",
    z_title="Normalized intensity (mean=1)",
    colorscale="Viridis",
    show_colorbar=False,
)

fig_avg

# %% [markdown]
# ## 5) Tie back to your 500 µs design budget (time + spectral diversity)
#
# Your repo already contains the **bookkeeping model**:
#
# $$N_{\\mathrm{eff}} \\approx N_t\\,N_\\lambda\\,N_{\\mathrm{pol}}\\,N_{\\mathrm{angle}},\\qquad
# C \\approx 1/\\sqrt{N_{\\mathrm{eff}}}.$$
#
# Here we compute rough values using:
#
# - exposure: 500 µs
# - scrambler: 10 kHz
# - a 3 m, NA=0.22 MMF (step-index upper bound) as a starting point
#
# Then we compute the speckle spectral decorrelation width $\\Delta\\lambda_c$ and estimate
# $N_\\lambda$ for a few linewidth examples.

# %%
exposure_s = 500e-6
scrambler_hz = 10e3

fiber = MultimodeFiber(core_diameter_um=400.0, na=0.22, length_m=3.0, n_core=1.46, modal_delay_scale=1.0)
delta_lambda_c_nm = speckle_spectral_corr_width_nm_for_fiber(lambda0_nm=640.0, fiber=fiber)

n_t = n_time_samples(exposure_s, scrambler_hz)

delta_lambda_c_nm, n_t

# %%
# Try a few example source spans (nm) and compute an N_eff + contrast prediction.
span_list_nm = [0.01, 0.1, 1.0, 2.0, 10.0]

rows = []
for span_nm in span_list_nm:
    # N_lambda ≈ ceil(span / Δλ_c), clamp >=1
    n_lambda = max(1, int(np.ceil(float(span_nm) / float(delta_lambda_c_nm))))
    diversity = DiversityBudget(n_lambda=n_lambda, n_pol=2, n_angle=1)
    n_eff = estimate_n_eff(exposure_s=exposure_s, scrambler_hz=scrambler_hz, diversity=diversity)
    C_pred = speckle_contrast_from_n_eff(n_eff)
    rows.append(
        dict(
            span_nm=float(span_nm),
            delta_lambda_c_nm=float(delta_lambda_c_nm),
            n_lambda=int(n_lambda),
            n_t=int(n_t),
            n_eff=float(n_eff),
            C_pred=float(C_pred),
        )
    )

rows

# %% [markdown]
# ### Interpreting this for your two approaches
#
# **Approach A (separate single-mode fibers):**
#
# - Each fiber output is close to a stable Gaussian.
# - Speckle from *fiber modes* is not the dominant problem.
# - Remaining risks are mostly *coherent free-space artifacts* (fringes from hard stops,
#   back-reflections) and Gaussian nonuniformity unless you oversize the beam.
#
# **Approach B (common multimode fiber):**
#
# - Many guided modes create granular intensity (speckle-like structure).
# - The pattern is **extremely sensitive** to phase perturbations (see the drift slider).
# - To make it “feel” uniform at 500 µs, you need enough **effective averaging**:
#   - fast scrambling ($N_t$),
#   - and/or enough spectral diversity ($N_\\lambda$),
#   - plus whatever polarization/angle diversity your optics naturally provides.
#
# Practical failure modes to watch (and design around)
# ---------------------------------------------------
#
# 1. **Not enough excited modes** (launch underfills NA or only excites a subset): higher contrast than expected.
# 2. **Patterns not truly independent** (scrambler too slow / correlated): $N_t$ smaller than the naive count.
# 3. **Graded-index MMF** (much lower modal dispersion): $\\Delta\\lambda_c$ larger → linewidth helps less.
#    - In your repo this is the `modal_delay_scale << 1` scenario in `src/mmf_fiber_speckle.py`.
# 4. **Near-field vs far-field relay choice**: imaging the fiber face vs imaging the angular distribution can
#    change which “speckle” you see at the sample plane.
#
# If you want standalone HTML (no notebook needed), run:
#
# ```bash
# python scripts/generate_fiber_modes_3d_viz.py --out-dir reports/fiber_modes_3d
# ```
#
# The HTML files are written under `reports/` (ignored by git) so you can keep them local.
