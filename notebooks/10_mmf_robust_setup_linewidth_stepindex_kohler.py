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
# # MMF illumination robustness: linewidth, RF broadening, step-index vs graded-index, and what can break
#
# This notebook addresses the practical questions you raised after talking with your professor.
#
# The focus is the specific 640 nm, high-power, short-exposure use-case:
#
# - **Wavelength:** 640 nm (e.g. exciting JFX650)
# - **Power:** 1–2 W class source
# - **ROI:** 30 µm × 30 µm at the sample
# - **Exposure:** 500 µs (and why 5 ms is easier)
# - **Delivery:** large-core MMF (e.g. 400 µm, NA≈0.22), ~1–3 m
# - **Conditioning:** fiber scrambler (~10 kHz) + rotating diffuser + Köhler-like relay
#
# As in Notebook 09, the style is:
#
# 1. **Visual**
# 2. **Text**
# 3. **Pseudocode**
# 4. **Equations** (only when fundamental)

# %% [markdown]
# ## 0) Imports

# %%
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import asdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

from src.excitation_speckle_sim import simulate_excitation_speckle_field
from src.illumination_design_params import illumination_na, speckle_grain_size_um
from src.mmf_fiber_speckle import (
    MultimodeFiber,
    estimate_n_lambda_from_fiber,
    intermodal_group_delay_spread_s,
    optical_path_spread_m,
    speckle_spectral_corr_width_nm_for_fiber,
)
from src.speckle_diversity_models import (
    DiversityBudget,
    estimate_n_eff,
    required_n_eff_for_contrast,
    speckle_contrast_from_n_eff,
)
from src.speckle_weighting import effective_n_from_weights
from src.temporal_coherence import coherence_length_m_from_linewidth_nm

if "ipykernel" in sys.modules:
    try:
        if matplotlib.get_backend().lower() == "agg":
            matplotlib.use("module://matplotlib_inline.backend_inline", force=True)
    except Exception:
        pass


# %% [markdown]
# ## 1) A concrete "robust MMF" concept (the one you described)
#
# You proposed a setup that sounds like this:
#
# - **Laser linewidth:** somewhere between 2 nm and 20 nm
# - **Fiber:** step-index MMF, ~400 µm core, NA≈0.22, **~3 m**
# - **Mode mixing:** a commercial fiber scrambler (~10 kHz)
# - **Additional speckle suppression:** rotating diffuser
# - **Clean-up:** quad filter to suppress any fiber/connector autofluorescence
# - **Geometry:** image the **field stop** (not the fiber near-field) onto the sample
# - **Pupil fill:** underfill objective BFP to ~0.3 (illumination NA reduced)
#
# We'll turn this into numbers.

# %%
# Baseline microscope + illumination geometry
lambda0_nm = 640.0
roi_um = 30.0
exposure_us = 500.0
exposure_s = exposure_us * 1e-6

na_obj = 1.45
pupil_fill = 0.30
na_illum = illumination_na(na_obj=na_obj, pupil_fill_fraction=pupil_fill)

scrambler_hz = 10_000.0

# Baseline fiber spec (matches common "homogenizing" fibers, and CNI FC-640 default)
fiber_base = MultimodeFiber(
    core_diameter_um=400.0,
    na=0.22,
    length_m=3.0,
    n_core=1.46,
    modal_delay_scale=1.0,
)

pd.DataFrame(
    [
        {
            "lambda0_nm": lambda0_nm,
            "roi_um": roi_um,
            "exposure_us": exposure_us,
            "na_obj": na_obj,
            "pupil_fill": pupil_fill,
            "na_illum": na_illum,
            "scrambler_hz": scrambler_hz,
            **asdict(fiber_base),
        }
    ]
)

# %% [markdown]
# ### 1.1 First visual: what speckle grain size does NA=0.3 imply?
#
# A quick rule-of-thumb for lateral speckle grain size in the sample plane:
#
# $$
# d_{\mathrm{speckle}} \sim \frac{\lambda}{2\,\mathrm{NA}_{\mathrm{illum}}}.
# $$
#
# This matters for the **PSF-scale false-positive / false-negative** concern:
#
# - If speckle grains were PSF-sized, they'd modulate spot brightness strongly.
# - If grains are much larger than the PSF, the main effect is slow shading (flat-field-like).

# %%
grain_um = speckle_grain_size_um(lambda_nm=lambda0_nm, na_illum=na_illum)

pd.DataFrame(
    [
        {
            "na_illum": na_illum,
            "speckle_grain_um (rule-of-thumb)": grain_um,
        }
    ]
)

# %% [markdown]
# **Interpretation:** with a 1.45 NA objective and 0.3 pupil fill, the effective illumination NA is ~0.44.
# That predicts speckle grains on the order of **~0.7–0.8 µm**, i.e. typically **larger than** a
# diffraction-limited emission PSF (~0.25–0.35 µm), but not orders of magnitude larger.
#
# That means:
#
# - You do want **low speckle contrast** (averaging), because residual grains are not *so* large that they are
#   purely a "flat-field".
# - But underfilling the pupil is already helping by pushing structure to somewhat larger scales.

# %% [markdown]
# ## 2) 2 nm vs 20 nm linewidth: coherence length, not folklore
#
# Your professor's "1% of λ" comment is trying to get at **temporal coherence**.
# The actual question is:
#
# > Is the source's coherence length short compared to the relevant optical-path differences (OPL spreads)
# > that create interference?
#
# We'll compute coherence length for 2 nm and 20 nm and compare it to fiber intermodal OPL spread.

# %%
linewidth_cases_nm = {
    "2 nm": 2.0,
    "20 nm": 20.0,
}

rows = []
for label, dlam in linewidth_cases_nm.items():
    lc = coherence_length_m_from_linewidth_nm(lambda0_nm=lambda0_nm, fwhm_nm=dlam, profile="gaussian")
    rows.append({"case": label, "Δλ_FWHM (nm)": dlam, "coherence_length_Lc (µm)": lc * 1e6})

pd.DataFrame(rows)

# %% [markdown]
# ### 2.1 Visual: coherence length vs linewidth

# %%
dlam_sweep = np.logspace(-2, np.log10(30.0), 200)  # 0.01 .. 30 nm
lc_sweep_um = np.array(
    [
        coherence_length_m_from_linewidth_nm(lambda0_nm=lambda0_nm, fwhm_nm=float(d), profile="gaussian") * 1e6
        for d in dlam_sweep
    ]
)

fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogx(dlam_sweep, lc_sweep_um)
ax.set_title("Coherence length (Gaussian, FWHM) vs linewidth at 640 nm")
ax.set_xlabel("Linewidth Δλ (nm)")
ax.set_ylabel("Coherence length Lc (µm)")
ax.grid(True, which="both", alpha=0.3)

for label, dlam in linewidth_cases_nm.items():
    ax.axvline(dlam, linestyle="--", alpha=0.7)
    ax.text(dlam, ax.get_ylim()[0] * 1.3, label, rotation=90, va="bottom", ha="right")

plt.show()

# %% [markdown]
# ### 2.2 Text
#
# For a 640 nm source:
#
# - **2 nm** FWHM corresponds to a coherence length on the order of **~0.1 mm**.
# - **20 nm** FWHM corresponds to **~10 µm**.
#
# Those are extremely short compared to meter-scale optical paths.
#
# The practical consequence is: if your system generates **centimeter-scale optical-path spreads**,
# then even "only" a few nm linewidth is already far into the "low temporal coherence" regime
# relative to those path differences.

# %% [markdown]
# ## 3) OPL from vendor specs: what sets its scale?
#
# The CNI FC-640 spec sheet (fiber-coupled 640 nm, 1–3 W) lists:
#
# - Fiber core diameter: **400 µm**
# - Fiber NA: **0.22**
# - Default fiber length: **1 m**
#
# The intermodal optical-path spread for a step-index fiber can be estimated (geometric optics) as:
#
# $$
# \Delta\mathrm{OPL} \approx \frac{\mathrm{NA}^2}{2 n_{\mathrm{core}}}\,L
# $$
#
# and the absolute optical path length is simply:
#
# $$
# \mathrm{OPL} = n_{\mathrm{core}}\,L.
# $$
#
# We'll compute these for 1 m and 3 m.

# %%
rows = []
for L in [1.0, 3.0]:
    fib = MultimodeFiber(
        core_diameter_um=400.0,
        na=0.22,
        length_m=L,
        n_core=1.46,
        modal_delay_scale=1.0,
    )
    dt = intermodal_group_delay_spread_s(fib)
    dopl = optical_path_spread_m(fib)
    rows.append(
        {
            "length_m": L,
            "OPL=nL (m)": fib.n_core * fib.length_m,
            "ΔOPL (cm)": dopl * 100.0,
            "Δτ (ps)": dt * 1e12,
        }
    )

pd.DataFrame(rows)

# %% [markdown]
# ### 3.1 Interpretation
#
# For NA≈0.22 and silica index ~1.46:
#
# - **1 m** gives $
#   \Delta\mathrm{OPL}\sim 1.7\,\mathrm{cm}$.
# - **3 m** gives $
#   \Delta\mathrm{OPL}\sim 5\,\mathrm{cm}$.
#
# Compare that to the coherence lengths above:
#
# - 2 nm: $L_c\sim 0.01\,\mathrm{cm}$
# - 20 nm: $L_c\sim 0.001\,\mathrm{cm}$
#
# So even the **default 1 m** fiber length already generates optical-path spreads that are orders of magnitude
# larger than the coherence length of a multi-nm diode.
#
# This is why the professor's "fiber must be very long" claim is *not universal*; it depends strongly on what
# they are implicitly assuming about modal dispersion (step-index vs graded-index) and about linewidth.

# %% [markdown]
# ## 4) Step-index vs graded-index: why you should care (and how to order it)
#
# **Key point:** graded-index (GI) fibers are engineered to *reduce* intermodal delay spread.
# That's good for telecom bandwidth, but it can reduce the spectral diversity you get "for free".
#
# In the repo we model that uncertainty with a single scale factor $s$:
#
# - $s=1$: step-index-like (upper bound on modal dispersion)
# - $s\ll 1$: strong GI behavior (lower dispersion)
#
# We'll compute the speckle spectral correlation width $\Delta\lambda_c$ and the implied number of
# independent wavelength "bins" $N_\lambda$ for 2 nm and 20 nm.

# %%
fiber_scales = {
    "step-index-ish (s=1)": 1.0,
    "moderate GI (s=0.1)": 0.1,
    "strong GI (s=0.02)": 0.02,
}

rows = []
for s_label, s in fiber_scales.items():
    for L in [1.0, 3.0]:
        fib = MultimodeFiber(
            core_diameter_um=400.0,
            na=0.22,
            length_m=L,
            n_core=1.46,
            modal_delay_scale=s,
        )
        dlam_c = speckle_spectral_corr_width_nm_for_fiber(lambda0_nm=lambda0_nm, fiber=fib)
        for lw_label, lw_nm in linewidth_cases_nm.items():
            nlam = estimate_n_lambda_from_fiber(lambda0_nm=lambda0_nm, source_span_nm=lw_nm, fiber=fib)
            rows.append(
                {
                    "fiber": s_label,
                    "L (m)": L,
                    "Δλ_c (nm)": dlam_c,
                    "linewidth": lw_label,
                    "N_lambda": nlam,
                }
            )

df_nlam = pd.DataFrame(rows)
df_nlam

# %% [markdown]
# ### 4.1 Visual: $N_\lambda$ vs dispersion scale

# %%
fig, ax = plt.subplots(figsize=(7, 4))

for lw_label in linewidth_cases_nm:
    subset = df_nlam[(df_nlam["linewidth"] == lw_label) & (df_nlam["L (m)"] == 3.0)]
    ax.plot(subset["fiber"], subset["N_lambda"], marker="o", label=f"{lw_label}, L=3 m")

ax.set_title("Spectral diversity N_lambda depends strongly on GI vs step-index")
ax.set_ylabel("N_lambda (estimated)")
ax.grid(True, alpha=0.3)
ax.legend()
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.2 Text: what this means for procurement
#
# 1. **Specifying step-index vs graded-index is usually explicit**.
#    Many vendors list it as a categorical type ("step-index" / "graded-index") rather than a single number.
#
# 2. The Schaefer+Kirchhoff catalog snippet you shared uses exactly this kind of explicit type marking
#    ("S" for step-index, "G" for graded-index).
#
# 3. For this specific imaging application, step-index is *not* a "downside".
#    The main downside of step-index is high intermodal dispersion for communications, but you are not doing
#    high-bandwidth data transmission. In our use-case, higher modal dispersion can actually be helpful.
#
# 4. Price is vendor-dependent, but a practical heuristic:
#
#    - Telecom GI fibers (50/125) are commodity.
#    - Large-core step-index silica MMFs (200–600 µm) are also commodity in the high-power delivery world.
#    - Large-core **graded-index** "homogenizers" can be more specialized.
#
#    So it is *not safe* to assume GI is always cheaper.

# %% [markdown]
# ## 5) Do you really need >>100 kHz scrambling?
#
# Let's connect your **hardware** (10 kHz scrambler, rotating diffuser, linewidth) to the short exposure.
#
# A simple speckle bookkeeping model is:
#
# $$
# C \approx \frac{1}{\sqrt{N_{\mathrm{eff}}}},\qquad
# N_{\mathrm{eff}} \approx N_{t,\mathrm{eff}}\,N_\lambda\,N_{\mathrm{pol}}\,N_{\mathrm{angle}}.
# $$

# %%
# We'll compare a few scenarios.

contrast_target = 0.10
n_pol = 2  # polarization diversity can help (depends on downstream optics)
n_angle = 1

rows = []

for s_label, s in fiber_scales.items():
    fib = MultimodeFiber(
        core_diameter_um=400.0,
        na=0.22,
        length_m=3.0,
        n_core=1.46,
        modal_delay_scale=s,
    )
    for lw_label, lw_nm in linewidth_cases_nm.items():
        nlam = estimate_n_lambda_from_fiber(lambda0_nm=lambda0_nm, source_span_nm=lw_nm, fiber=fib)
        # Base: only the fiber scrambler at 10 kHz.
        n_eff = estimate_n_eff(
            exposure_s=exposure_s,
            scrambler_hz=scrambler_hz,
            diversity=DiversityBudget(n_lambda=nlam, n_pol=n_pol, n_angle=n_angle),
            successive_pattern_correlation=0.0,
        )
        rows.append(
            {
                "fiber": s_label,
                "linewidth": lw_label,
                "N_lambda": nlam,
                "scrambler_hz": scrambler_hz,
                "N_eff": n_eff,
                "C_pred": speckle_contrast_from_n_eff(n_eff),
            }
        )

df_budget = pd.DataFrame(rows)
df_budget

# %% [markdown]
# ### 5.1 Visual: predicted speckle contrast for the 500 µs case

# %%
fig, ax = plt.subplots(figsize=(7, 4))

for lw_label in linewidth_cases_nm:
    subset = df_budget[df_budget["linewidth"] == lw_label]
    ax.plot(subset["fiber"], subset["C_pred"], marker="o", label=f"{lw_label}")

ax.axhline(contrast_target, linestyle="--", alpha=0.7)
ax.text(0, contrast_target * 1.05, "target C=0.10", va="bottom")

ax.set_title("Predicted speckle contrast at 500 µs (10 kHz scrambler, N_pol=2)")
ax.set_ylabel("C_pred")
ax.grid(True, alpha=0.3)
ax.legend()
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5.2 Text
#
# This plot is the cleanest answer to "do I need >>100 kHz?":
#
# - If $N_\lambda$ is already large (step-index-like and/or wide linewidth), then even 10 kHz is plenty.
# - If GI behavior is very strong **and** linewidth is only a couple nm, you may land near the edge of
#   your contrast target — and then either:
#   - increase linewidth (or deliberately sweep wavelength),
#   - increase effective scrambling (scrambler + diffuser), or
#   - accept slightly higher contrast and rely on flat-field calibration.
#
# A crucial takeaway: the scrambler frequency requirement is **not an absolute number**; it scales with how
# much spectral (and polarization) diversity you already have.

# %% [markdown]
# ## 6) Mode-mixing and modal power distribution: why "M is huge" is not the whole story
#
# Your professor said:
#
# - "There are not enough uncorrelated modes."
# - "Not enough uncorrelated realizations."
#
# For a 400 µm MMF at 640 nm, the *count of supported modes* is enormous. But the important quantity
# for averaging is the **effective number of independent contributions**.
#
# One convenient way to see this is to model unequal weights.
#
# If you incoherently sum independent patterns with weights $w_i$ (power fractions), then the effective
# count is:
#
# $$
# N_{\mathrm{eff}} = \frac{(\sum_i w_i)^2}{\sum_i w_i^2}.
# $$
#
# If a few modes (or a few wavelength bins) carry most of the power, $N_{\mathrm{eff}}$ can be far smaller
# than the raw count.

# %%
def normalized(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 0.0, None)
    return w / float(w.sum())


def exp_mode_weights(n: int, decay: float) -> np.ndarray:
    # i=0 is strongest mode, higher i weaker.
    i = np.arange(n, dtype=float)
    return normalized(np.exp(-decay * i))


cases = {
    "uniform (100)": normalized(np.ones(100)),
    "10 strong + 90 weak": normalized(np.concatenate([np.ones(10), 0.1 * np.ones(90)])),
    "exponential decay (100, decay=0.1)": exp_mode_weights(100, decay=0.1),
    "exponential decay (100, decay=0.3)": exp_mode_weights(100, decay=0.3),
}

rows = []
for name, w in cases.items():
    rows.append({"case": name, "N_raw": int(w.size), "N_eff_from_weights": effective_n_from_weights(w)})

pd.DataFrame(rows)

# %% [markdown]
# ### 6.1 Visual: weight distributions

# %%
fig, ax = plt.subplots(figsize=(7, 4))

for name, w in cases.items():
    ax.plot(w[:60], label=name)

ax.set_title("Example mode/wavelength weight distributions")
ax.set_xlabel("index")
ax.set_ylabel("normalized weight")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6.2 Text
#
# This is how to reconcile two statements that can both be true:
#
# - "The fiber supports ~10^5 modes." (true)
# - "Effective mode diversity is not that large." (also can be true)
#
# The **solution space** is also clear:
#
# - Launch conditions that excite more modes (fill the fiber NA more, use a diffuser before coupling, etc.)
# - Intentional mode mixing (scrambler, bends, agitation)
# - Avoid optics that introduce strong mode-dependent loss

# %% [markdown]
# ## 7) Similarities and differences to Köhler illumination
#
# You're already thinking in the right direction:
#
# - *Critical illumination* images the source (fiber near-field) onto the sample.
#   This tends to imprint fiber structure onto the field.
#
# - *Köhler illumination* images the source onto a pupil plane, and images the field stop onto the sample.
#   This tends to de-couple source structure from field uniformity.
#
# In your proposed design:
#
# - the **field stop image** sets the 30×30 µm ROI
# - the **pupil fill** (0.3) sets $\mathrm{NA}_{\mathrm{illum}}$ and therefore speckle grain scale
# - the MMF + scrambler + diffuser provides angular and temporal diversity
#
# That's Köhler-like in the sense that you are trying to avoid mapping the fiber near-field to the sample.

# %% [markdown]
# ## 8) A quick PSF-scale sanity check (simulation)
#
# This is not a full microscope model; it's a quick way to answer:
#
# > If I have a target speckle contrast C≈0.1, what does the illumination field look like inside a 30 µm ROI?
#
# The simulator makes a square field stop, applies a circular pupil low-pass (illumination NA), and averages
# independent random complex fields.
#
# We will compare three "effective diversity" values by changing $N_{\mathrm{eff}}$.

# %%
n = 256
dx_um = 0.065  # 65 nm sampling

def show_field(I: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(I, origin="lower")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.show()


def summarize_field(I: np.ndarray) -> dict[str, float]:
    roi = I
    m = float(np.mean(roi))
    s = float(np.std(roi))
    return {
        "mean": m,
        "std": s,
        "contrast_C=std/mean": s / m,
        "p99/p1": float(np.quantile(roi, 0.99) / np.quantile(roi, 0.01)),
    }


scenarios = [
    {"label": "coherent-ish (N_eff=1)", "n_src": 1, "scr_hz": 0.0},
    {"label": "moderate averaging (N_eff~20)", "n_src": 4, "scr_hz": 10_000.0},
    {"label": "heavy averaging (N_eff~200)", "n_src": 40, "scr_hz": 10_000.0},
]

rows = []
for sc in scenarios:
    I, meta = simulate_excitation_speckle_field(
        n=n,
        dx_um=dx_um,
        roi_um=roi_um,
        lambda_um=lambda0_nm * 1e-3,
        na_illum=na_illum,
        exposure_s=exposure_s,
        scrambler_hz=sc["scr_hz"],
        n_src=sc["n_src"],
        seed=0,
    )
    I_norm = I / float(np.mean(I))
    show_field(I_norm, f"{sc['label']}\n(meta n_eff={meta.n_eff})")
    rows.append({"scenario": sc["label"], **summarize_field(I_norm), **asdict(meta)})

pd.DataFrame(rows)

# %% [markdown]
# ### 8.1 Text
#
# This simulation is meant to build intuition, not to be the final word.
#
# Still, it captures three relevant design facts:
#
# - With no averaging, speckle contrast is high.
# - With even modest averaging (tens of independent states), contrast drops substantially.
# - With heavy averaging (hundreds), the field becomes close to uniform inside the ROI.
#
# If your target is 500 µs exposures, this is the kind of "what does it look like" evidence you can show
# in a meeting.

# %% [markdown]
# ## 9) Why 5 ms is easier if 500 µs works
#
# Time averaging scales roughly as $C\propto 1/\sqrt{N_t}$.
# If you increase exposure from 500 µs to 5 ms (10×), then $N_t$ increases by 10× for the same scrambler,
# and speckle contrast drops by $\sqrt{10}\approx 3.2$.
#
# So if you can make the **500 µs** case work, the 5 ms case is almost always "in the bag".

# %% [markdown]
# ## 10) What can break the robustness? (and how to mitigate)
#
# Below is a deliberately critical checklist.
#
# ### 10.1 Failure mode: accidentally operating in critical illumination
#
# **Symptom:** the *mean* illumination (long exposure average) shows strong structure that tracks the fiber output.
#
# **Mitigations:**
#
# - ensure the field stop is imaged to the sample (field conjugate)
# - place the diffuser / fiber tip in a plane conjugate to the objective pupil (source conjugate)
# - diagnose by imaging:
#   - the fiber near-field,
#   - the pupil plane,
#   - and the sample plane.
#
# ### 10.2 Failure mode: insufficient effective diversity at 500 µs
#
# **Symptom:** high speckle contrast remains at short exposures.
#
# **Mitigations:**
#
# - increase linewidth (or sweep wavelength over the exposure)
# - use step-index fiber (or longer fiber if GI)
# - add/optimize diffuser + scrambler (increase effective decorrelation rate)
# - ensure you are not under-filling the fiber NA too aggressively (excite more modes)
#
# ### 10.3 Failure mode: mode-dependent loss / incomplete mode mixing
#
# **Symptom:** the average field has a ring/bright-center or otherwise non-flat profile that changes with fiber bends.
#
# **Mitigations:**
#
# - avoid tight bends; respect minimum bend radius
# - add a mode mixer (scrambler) section downstream of any tight routing
# - couple with a diffuser or with higher NA launch to excite more modes
#
# ### 10.4 Failure mode: laser instability from back-reflections
#
# **Symptom:** excess intensity noise, mode hopping, unstable spectrum.
#
# **Mitigation:** a good isolator (and careful connector cleanliness).
#
# ### 10.5 Failure mode: fiber autofluorescence / background
#
# At 640 nm this is usually much less severe than at 488/405, but can still happen depending on fiber material.
# Your "quad filter" concept is a reasonable precaution.

# %% [markdown]
# ## 11) 2 nm vs 20 nm: cost and "RF broadening" (engineering view, not a vendor quote)
#
# You asked whether a 20 nm source is cheaper or more expensive than a 2 nm source, and whether RF modulation
# can turn 2 nm into 20 nm economically.
#
# A cautious, practical answer:
#
# 1. **At fixed power and beam quality, narrower linewidth is often harder**, because it requires additional
#    cavity/control engineering (external cavities, volume Bragg gratings, etc.).
#
# 2. **But** very *broad* linewidth at high power can also be specialized. If you go all the way to 20 nm,
#    you may cross into "ASE / SLD / multi-emitter" territory. Those can be more expensive than a plain
#    multimode diode.
#
# 3. **RF / fast modulation** is a real technique for coherence reduction. In practice it is usually used to:
#
#    - suppress mode hopping artifacts,
#    - broaden the spectrum modestly (often ~1–2 nm class),
#    - and reduce speckle/etalon interference.
#
#    Whether it can economically yield a stable 20 nm span at ~1–2 W depends strongly on the laser architecture.
#
# The safe takeaway for optical design is:
#
# - You don't need to bet everything on any *single* mechanism.
# - A robust design uses **some linewidth** (even a few nm), plus **scrambling**, plus **a proper Köhler relay**.
