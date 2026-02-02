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
# # MMF widefield illumination @ 500 µs: Fourier-optics + speckle-averaging design sweeps
#
# Goal: connect the *sample-plane* constraint
#
# - exposure: **500 µs**
# - ROI: **10 µm × 10 µm**
# - irradiance: **10–30 kW/cm²**
# - excitation: **~640 nm** (but dyes allow ~637–650 nm)
#
# to the illumination train choices:
#
# - field stop size (sample-conjugate image plane)
# - pupil fill (fraction of objective BFP)
# - fiber collimator focal length and beam diameter
# - speckle-averaging strategy: time scrambling, spectral diversity, polarization, pupil-angle hopping
#
# This notebook is written in **Jupytext percent format**.

# %% [markdown]
# ## 0) Imports + repo plumbing

# %%
from __future__ import annotations

import sys
from dataclasses import asdict
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

from src.excitation_speckle_sim import simulate_excitation_speckle_field  # noqa: E402
from src.illumination_design_params import (  # noqa: E402
    PowerBudget,
    bfp_beam_diameter_mm,
    collimated_beam_diameter_mm,
    field_stop_size_mm,
    illumination_na,
    objective_focal_length_mm,
    objective_pupil_diameter_mm,
    required_fiber_exit_power_mw,
    required_sample_power_mw,
    speckle_grain_size_um,
)
from src.speckle_diversity_models import (  # noqa: E402
    DiversityBudget,
    estimate_n_eff,
    estimate_n_lambda,
    estimate_speckle_spectral_corr_width_nm,
    required_n_eff_for_contrast,
    speckle_contrast_from_n_eff,
)

# %% [markdown]
# ## 1) Sample-plane requirements → power + field stop size
#
# Power needed at the *sample* is set by irradiance × area.
#
# For ROI = \(10\,\mu\mathrm{m}\times10\,\mu\mathrm{m}\),
#
# ```math
# A = 10^{-6}\,\mathrm{cm}^2.
# ```
#
# So 10–30 kW/cm² corresponds to 10–30 mW at the sample.

# %%
lambda_exc_nm = 640.0
roi_um = (10.0, 10.0)
exposure_us = 500.0
exposure_s = exposure_us * 1e-6

irradiance_kw_cm2_list = [10.0, 20.0, 30.0]
p_sample_mw_list = [required_sample_power_mw(E, roi_um) for E in irradiance_kw_cm2_list]
pd.DataFrame({"E_kW_cm2": irradiance_kw_cm2_list, "P_sample_mW": p_sample_mw_list})

# %% [markdown]
# A **1 mm × 1 mm** field stop at a 100× sample-conjugate image plane is the right scale:
#
# ```math
# D_{\mathrm{stop}} \approx M\,D_{\mathrm{sample}} \approx 100\times 10\,\mu\mathrm{m} = 1\,\mathrm{mm}.
# ```
#
# If you skip the field stop and rely on a digital ROI, you bleach outside the ROI
# (often unacceptable at 10–30 kW/cm²).

# %%
M_obj = 100.0
stop_mm = field_stop_size_mm(roi_um, sample_to_stop_magnification=M_obj)
print(f"Field stop for ROI {roi_um[0]:.0f}×{roi_um[1]:.0f} µm at {M_obj:.0f}×: {stop_mm[0]:.3f} × {stop_mm[1]:.3f} mm")

# %% [markdown]
# ## 2) Objective pupil + chosen pupil fill → illumination NA + speckle grain size
#
# For an infinity system, the objective focal length is approximately
#
# ```math
# f_{\mathrm{obj}} \approx \frac{f_{\mathrm{TL}}}{M}.
# ```
#
# A useful pupil diameter estimate is
#
# ```math
# D_{\mathrm{pupil}} \approx 2 f_{\mathrm{obj}}\,\mathrm{NA}_{\mathrm{obj}}.
# ```
#
# If you fill a fraction \(\rho\) of the pupil diameter, then
#
# ```math
# \mathrm{NA}_{\mathrm{illum}} \approx \rho\,\mathrm{NA}_{\mathrm{obj}}.
# ```

# %%
na_obj = 1.45
tube_lens_f_mm = 200.0
pupil_fill = 0.30  # your stated target: "fill 0.3 of the BFP"

f_obj_mm = objective_focal_length_mm(tube_lens_f_mm=tube_lens_f_mm, magnification=M_obj)
d_pupil_mm = objective_pupil_diameter_mm(na_obj=na_obj, f_obj_mm=f_obj_mm)
d_bfp_mm = bfp_beam_diameter_mm(pupil_fill_fraction=pupil_fill, pupil_diameter_mm=d_pupil_mm)
na_illum = illumination_na(na_obj=na_obj, pupil_fill_fraction=pupil_fill)

grain_um = speckle_grain_size_um(lambda_nm=lambda_exc_nm, na_illum=na_illum)

pd.DataFrame(
    [
        {
            "f_obj_mm": f_obj_mm,
            "D_pupil_mm": d_pupil_mm,
            "pupil_fill": pupil_fill,
            "D_beam@BFP_mm": d_bfp_mm,
            "NA_illum": na_illum,
            "speckle_grain_um~λ/(2NA)": grain_um,
        }
    ]
)

# %% [markdown]
# ## 3) Fiber collimator: what f_coll gives ~1.3 mm beam?
#
# Using the far-field cone of the MMF (NA_fiber) and a collimator focal length:
#
# ```math
# D_{\mathrm{coll}} \approx 2 f_{\mathrm{coll}} \tan\!\left(\arcsin\mathrm{NA}_{\mathrm{fiber}}\right).
# ```
#
# With NA_fiber ~ 0.22, f_coll ≈ 3 mm gives ~1.3 mm diameter. That lines up with your intuition.

# %%
na_fiber = 0.22

for f_coll_mm in [2.0, 2.75, 3.0, 4.5, 8.0]:
    d = collimated_beam_diameter_mm(f_coll_mm=f_coll_mm, fiber_na=na_fiber)
    print(f"f_coll={f_coll_mm:>4.2f} mm -> D_coll≈{d:>5.2f} mm")

# %% [markdown]
# If you want \(D_{\mathrm{beam@BFP}}\approx 1.7\,\mathrm{mm}\) and the fiber collimator gives \(D_{\mathrm{coll}}\approx 1.3\,\mathrm{mm}\),
# you need only a modest telescope (≈1.3× beam expansion).
#
# In other words, the **geometrical** parts of the optical train are not the hard part here.

# %%
d_coll_mm = collimated_beam_diameter_mm(f_coll_mm=3.0, fiber_na=na_fiber)
print(f"Example: D_coll={d_coll_mm:.2f} mm, target D@BFP={d_bfp_mm:.2f} mm -> telescope ratio ≈ {d_bfp_mm/d_coll_mm:.2f}×")

# %% [markdown]
# ## 4) Power budget: sample power → fiber exit power
#
# The sample needs 10–30 mW. Required fiber exit power depends on throughput.
# Below is a *typical* multiplicative budget (edit it to match your lab reality).

# %%
budget = PowerBudget(
    coupling_into_fiber=0.6,
    fiber_to_collimator=0.95,
    stop_and_relays=0.7,
    objective_and_misc=0.8,
)
T = budget.total_throughput()
print(f"Example throughput T_total ≈ {T:.3f} (from {asdict(budget)})")

p_fiber_mw_list = [required_fiber_exit_power_mw(p, throughput=T) for p in p_sample_mw_list]
pd.DataFrame({"E_kW_cm2": irradiance_kw_cm2_list, "P_sample_mW": p_sample_mw_list, "P_fiber_exit_mW": p_fiber_mw_list})

# %% [markdown]
# If you end up needing **100–500 mW at the fiber exit**, that implies \(T_{\mathrm{total}}\lesssim 0.1	ext{–}0.3\) for the 10–30 mW sample target.
# That can happen (extra AOMs, poor coupling, heavy clipping), but it is not automatic.

# %% [markdown]
# ## 5) Speckle averaging: what must be true at 500 µs?
#
# For fully developed speckle, a useful approximation is:
#
# ```math
# C \approx \frac{1}{\sqrt{N_{\mathrm{eff}}}},\qquad
# N_{\mathrm{eff}} \approx N_t\,N_\lambda\,N_{\mathrm{pol}}\,N_{\mathrm{angle}}.
# ```
#
# If you want \(C\lesssim 0.1\), you need \(N_{\mathrm{eff}}\gtrsim 100\).

# %%
c_target = 0.10
n_eff_needed = required_n_eff_for_contrast(c_target)
print(f"Target contrast C≤{c_target:.2f} -> N_eff ≥ {n_eff_needed:d}")

# %% [markdown]
# ### 5.1 Time diversity: why 10 kHz scramblers look bad at 500 µs

# %%
scrambler_hz_list = [0.0, 10e3, 50e3, 100e3, 1e6]
rows = []
for f in scrambler_hz_list:
    n_eff = estimate_n_eff(
        exposure_s=exposure_s,
        scrambler_hz=f,
        diversity=DiversityBudget(n_lambda=1, n_pol=1, n_angle=1),
        successive_pattern_correlation=0.0,
    )
    rows.append({"scrambler_hz": f, "N_eff(time_only)": n_eff, "C~1/sqrt(N_eff)": speckle_contrast_from_n_eff(n_eff)})
pd.DataFrame(rows)

# %% [markdown]
# At 10 kHz, \(N_t\approx 5\). That alone cannot reach \(N_{\mathrm{eff}}\gtrsim100\).
#
# So the question becomes: can we get the remaining factor from **spectral diversity** and/or **cheap faster scrambling**?

# %% [markdown]
# ### 5.2 Spectral diversity: broad diode linewidth or deliberate wavelength sweep
#
# A simple spectral decorrelation estimate is
#
# ```math
# \Delta\lambda_c \sim \frac{\lambda^2}{\Delta\mathrm{OPL}}.
# ```
#
# In an MMF, different guided paths can have different optical path lengths; if your laser spectrum spans many
# independent speckle correlation widths, then \(N_\lambda\) can be large even during a short exposure.
#
# The key uncertainty is the effective optical-path-length spread \(\Delta\mathrm{OPL}\) of *the interfering contributions*.
# We therefore sweep it.

# %%
lambda0_nm = lambda_exc_nm
source_span_nm_list = [0.02, 0.2, 1.0, 2.0, 10.0, 13.0]  # 13 nm ~ (637→650) sweep possibility
opl_spread_m_list = [1e-3, 1e-2, 3e-2, 1e-1]  # 1 mm .. 10 cm of optical-path spread

rows = []
for opl in opl_spread_m_list:
    dlam_c = estimate_speckle_spectral_corr_width_nm(lambda0_nm=lambda0_nm, optical_path_spread_m=opl)
    for span in source_span_nm_list:
        nlam = estimate_n_lambda(source_span_nm=span, speckle_corr_width_nm=dlam_c, n_lines=1)
        rows.append(
            {
                "OPL_spread_m": opl,
                "Δλ_c_nm": dlam_c,
                "source_span_nm": span,
                "N_lambda": nlam,
            }
        )

df_spec = pd.DataFrame(rows)
df_spec

# %% [markdown]
# Interpretation:
#
# - If your source behaves like a **narrow-line** laser (effective span ≪ 0.1 nm), \(N_\lambda\) is small.
# - If your **diode** behaves like a ~1–2 nm source *and* the fiber induces **cm-scale optical path spread**, \(N_\lambda\) can be tens-to-hundreds.
#
# That is the first “escape hatch” for 500 µs: you may get enough averaging **without** a super-fast mechanical scrambler.
#
# The second escape hatch is to make \(N_t\) faster (piezo fiber shaker, AOM pupil hops), which we treat as \(N_t\) or \(N_{\mathrm{angle}}\).

# %% [markdown]
# ### 5.3 Combine diversity channels into \(N_{\mathrm{eff}}\)
#
# We'll compare a few realistic scenarios:
#
# 1. **Pessimistic**: narrow-line, slow scrambler (10 kHz) → high contrast
# 2. **Spectral helps**: ~1–2 nm diode spectrum (or modest sweep), still 10 kHz scrambler
# 3. **Fast N_t**: piezo/ultrasonic fiber agitator (100 kHz–1 MHz class), narrow spectrum
# 4. **Angle hops**: beam steering (if you already have an AOM/resonant scanner), + modest spectrum

# %%
scenarios = [
    dict(name="narrow + 10kHz", scrambler_hz=10e3, n_lambda=1, n_pol=1, n_angle=1),
    dict(name="1nm + 10kHz", scrambler_hz=10e3, n_lambda=50, n_pol=2, n_angle=1),
    dict(name="2nm + 10kHz", scrambler_hz=10e3, n_lambda=100, n_pol=2, n_angle=1),
    dict(name="narrow + 100kHz", scrambler_hz=100e3, n_lambda=1, n_pol=1, n_angle=1),
    dict(name="narrow + 1MHz", scrambler_hz=1e6, n_lambda=1, n_pol=1, n_angle=1),
    dict(name="narrow + 10kHz + 20 angle hops", scrambler_hz=10e3, n_lambda=1, n_pol=1, n_angle=20),
]

rows = []
for s in scenarios:
    div = DiversityBudget(n_lambda=s["n_lambda"], n_pol=s["n_pol"], n_angle=s["n_angle"])
    n_eff = estimate_n_eff(exposure_s=exposure_s, scrambler_hz=s["scrambler_hz"], diversity=div)
    rows.append(
        {
            "scenario": s["name"],
            "scrambler_hz": s["scrambler_hz"],
            "n_lambda": s["n_lambda"],
            "n_pol": s["n_pol"],
            "n_angle": s["n_angle"],
            "N_eff": n_eff,
            "C_pred": speckle_contrast_from_n_eff(n_eff),
        }
    )

df = pd.DataFrame(rows).sort_values("C_pred")
df

# %% [markdown]
# If any of the “helpful” scenarios look plausible *for your hardware*, then MMF is not automatically disqualified at 500 µs.
#
# The point is not that the numbers above are gospel — they’re placeholders you should replace with **your measured**
# speckle contrast \(C\) and your measured diode spectrum (or deliberate sweep).
#
# The notebook’s job is to make those dependencies explicit.

# %% [markdown]
# ## 6) Fourier-optics simulation: visualize a few \(N_{\mathrm{eff}}\) cases
#
# We use `simulate_excitation_speckle_field(...)` which implements:
# - a square ROI mask (field stop)
# - a pupil low-pass parameterized by NA_illum
# - incoherent averaging over time (scrambler_hz × exposure_s) and over `n_src` independent realizations

# %%
# Simulation sampling: match the 100× / 6.5 µm camera pixel -> 65 nm at sample.
camera_pixel_um = 6.5
dx_um = camera_pixel_um / M_obj

# Keep grid modest; we only need a little larger than the ROI.
N_grid = 512  # ~33 µm field at 65 nm/px

lambda_um = lambda_exc_nm * 1e-3

def sim_case(label: str, scrambler_hz: float, n_src: int) -> np.ndarray:
    I, _ = simulate_excitation_speckle_field(
        n=N_grid,
        dx_um=dx_um,
        roi_um=roi_um[0],  # kernel currently expects scalar square ROI; our ROI is square here.
        lambda_um=lambda_um,
        na_illum=na_illum,
        exposure_s=exposure_s,
        scrambler_hz=scrambler_hz,
        n_src=n_src,
        seed=0,
    )
    # normalize to mean inside a central patch (avoid edge roll-off bias)
    center = N_grid // 2
    half = int((roi_um[0] / dx_um) * 0.35)
    patch = I[center - half : center + half, center - half : center + half]
    return I / np.mean(patch), label

viz_cases = [
    ("narrow + 10kHz (N_eff~5)", 10e3, 1),
    ("narrow + 1MHz (N_eff~500)", 1e6, 1),
    ("spectral help (N_src=100) + 10kHz", 10e3, 100),
]

fig, axes = plt.subplots(1, len(viz_cases), figsize=(4.5 * len(viz_cases), 4))
if len(viz_cases) == 1:
    axes = [axes]

for ax, (label, scr_hz, n_src) in zip(axes, viz_cases, strict=True):
    I_norm, _ = sim_case(label, scrambler_hz=scr_hz, n_src=n_src)
    im = ax.imshow(I_norm, vmin=0, vmax=np.percentile(I_norm, 99))
    ax.set_title(label)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()

# %% [markdown]
# You should see that as \(N_{\mathrm{eff}}\) increases, the intensity field becomes more uniform.
#
# **Important:** this is an *idealized* model:
# - it treats independent patterns as truly independent
# - it does not simulate exact fiber guided modes
# - it assumes the field stop is well imaged and that pupil fill is clean
#
# Still, it is a correct “Fourier optics + statistics” framework for checking whether your design assumptions are internally consistent.

# %% [markdown]
# ## 7) What to measure next (so the model collapses to your hardware)
#
# 1. Measure the diode spectrum (or effective spectral span during an exposure).
# 2. With MMF illumination, record many short-exposure frames at low camera noise.
# 3. Compute speckle contrast \(C\) in a central sub-ROI (avoid edges).
# 4. Infer \(N_{\mathrm{eff}}\approx 1/C^2\).
#
# Then update:
# - `source_span_nm` and/or the “sweep” concept (637–650 nm is allowed by your dyes)
# - `scrambler_hz` or “angle hops per exposure”
# - the optical-path spread assumption (or simply treat \(N_\lambda\) as a fitted parameter)
