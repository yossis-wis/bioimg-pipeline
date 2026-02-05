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
# # MMF + wide linewidth + scrambling: a Fourier-optics sanity check
#
# This notebook is written to address a set of common objections to a
# **multimode-fiber (MMF) widefield illumination** approach:
#
# - "An MMF approach for a homogeneous field is not practical."
# - "Speckle will be a show-stopper."
# - "You'd need an impractically long fiber."
# - "There aren't enough uncorrelated modes."
# - "A couple of nanometers linewidth is not enough." / "Maybe 20 nm?" / "You need ~1% of λ."
#
# The goal is not to "win an argument"—it is to make the *assumptions explicit* and
# quantify which ones matter.
#
# ---
#
# ## How to read this notebook
#
# For each key idea, the presentation order is:
#
# 1. **Visual** (plots / pictures)
# 2. **Text** (plain-language explanation)
# 3. **Pseudocode** (what you'd implement)
# 4. **Equations** (only when they are fundamental)
#
# This makes it easier to build intuition without drowning in math.

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

from src.excitation_speckle_sim import simulate_excitation_speckle_field  # noqa: E402
from src.illumination_speckle_metrics import (  # noqa: E402
    normalize_by_mask_mean,
    speckle_contrast,
    square_roi_region_masks,
)
from src.mmf_fiber_speckle import (  # noqa: E402
    MultimodeFiber,
    approx_num_guided_modes_step_index,
    estimate_n_lambda_from_fiber,
    intermodal_group_delay_spread_s,
    optical_path_spread_m,
    required_fiber_length_m_for_target_n_lambda,
    speckle_spectral_corr_width_nm_for_fiber,
    v_number,
)

# %% [markdown]
# ## 1) The concrete scenario (based on the recent quote)
#
# The CNI quote (via Roshel) for the MMF concept was:
#
# - **Homogenizing fiber**: 400 µm core, 3 m length, SMA905
# - **640 nm**: 2 W, linewidth ~2–3 nm
# - **488 nm**: 100 mW, linewidth ~2 nm
# - **Scrambler**: ~10 kHz (question: provided by customer)
#
# For the microscope side, we care about a representative **short exposure** use-case:
#
# - **Exposure**: 500 µs
# - **ROI**: 10 µm × 10 µm
# - **Illumination NA**: set by pupil underfill (choose a representative value)

# %%
# --- Illumination / camera assumptions ---
exposure_us = 500.0
exposure_s = exposure_us * 1e-6

roi_um = 10.0

# Effective illumination NA at the sample (NOT the fiber NA).
# This depends on how much of the objective BFP you fill.
na_illum = 0.43

# Sampling: 100× objective with 6.5 µm camera pixel -> 65 nm at sample.
M_obj = 100.0
camera_pixel_um = 6.5

dx_um = camera_pixel_um / M_obj

# Grid (keep moderate so the notebook runs quickly)
N_grid = 256

masks = square_roi_region_masks(
    n=N_grid,
    dx_um=dx_um,
    roi_um=roi_um,
    inner_margin_um=2.0,
    edge_band_um=1.0,
)

# --- Fiber assumptions (typical large-core silica MMF NA) ---
# The quote did not explicitly list NA. 0.22 is a very common value.
fiber = MultimodeFiber(core_diameter_um=400.0, na=0.22, length_m=3.0, n_core=1.46, modal_delay_scale=1.0)

# --- Quoted linewidths (nm) ---
linewidths_nm = {
    "2 nm": 2.0,
    "3 nm": 3.0,
    "1% of 640 nm (~6.4 nm)": 6.4,
    "20 nm": 20.0,
}

scrambler_hz = 10e3

# %% [markdown]
# ## 2) First principle: what *speckle averaging* looks like
#
# ### 2.1 Visual
#
# Below we generate a *Fourier-limited* square ROI illumination field and average
# different numbers of **independent speckle realizations**.
#
# - $N_{\mathrm{eff}}=1$ → fully developed speckle
# - $N_{\mathrm{eff}}\gg 1$ → smoother field
#
# This is the fundamental phenomenon we want from:
#
# - **linewidth diversity** ($N_{\lambda}$)
# - **scrambler / agitation** ($N_t$)
# - **polarization mixing** ($N_{\mathrm{pol}}$)

# %%
N_eff_demo = [1, 5, 25, 125]

fig, axes = plt.subplots(1, len(N_eff_demo), figsize=(4.2 * len(N_eff_demo), 4), constrained_layout=True)
if len(N_eff_demo) == 1:
    axes = [axes]

for ax, n_eff in zip(axes, N_eff_demo):
    # We force n_time=1 and use n_src=n_eff to mean "average n_eff independent patterns".
    I, meta = simulate_excitation_speckle_field(
        n=N_grid,
        dx_um=dx_um,
        roi_um=roi_um,
        lambda_um=0.640,
        na_illum=na_illum,
        exposure_s=exposure_s,
        scrambler_hz=0.0,
        n_src=int(n_eff),
        seed=0,
    )
    In = normalize_by_mask_mean(I, masks.inner)
    C = speckle_contrast(In, masks.inner)

    im = ax.imshow(In, origin="lower", vmin=0, vmax=np.percentile(In[masks.roi], 99))
    ax.set_title(f"N_eff={n_eff}\nC={C:.3f} (measured)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.show()

# %% [markdown]
# ### 2.2 Text
#
# If you only remember one thing:
#
# - For fully developed speckle, the contrast is roughly
#   $C \approx 1/\sqrt{N_{\mathrm{eff}}}$.
#
# So the *only* real question is:
#
# > In your actual MMF setup, what is a defensible estimate of $N_{\mathrm{eff}}$ during one exposure?
#
# ---
#
# ### 2.3 Pseudocode
#
# ```text
# N_eff = N_time * N_lambda * N_pol * N_angle
# contrast ≈ 1 / sqrt(N_eff)
# ```
#
# ---
#
# ### 2.4 Equations
#
# $$
# C \equiv \frac{\sigma_I}{\langle I \rangle} \approx \frac{1}{\sqrt{N_{\mathrm{eff}}}}.
# $$

# %% [markdown]
# ## 3) "Not enough uncorrelated modes": mode count is *not* the bottleneck here
#
# One objection was "there aren't enough uncorrelated modes".
#
# It's important to separate two different questions:
#
# 1. **Does the fiber support many guided modes?** (geometry / V-number)
# 2. **Do we average enough *independent realizations* during a 500 µs exposure?** ($N_{\mathrm{eff}}$)
#
# The first is easy to answer from fiber specs.

# %%
# Mode count estimate for the quoted fiber core size at 640 nm and 488 nm.
rows = []
for lambda_nm in [640.0, 488.0]:
    v = v_number(core_radius_um=0.5 * fiber.core_diameter_um, na=fiber.na, lambda_um=lambda_nm * 1e-3)
    m = approx_num_guided_modes_step_index(v)
    rows.append({"lambda_nm": lambda_nm, "V": v, "M_modes~V^2/2": m})

pd.DataFrame(rows)

# %% [markdown]
# **Interpretation:** for a 400 µm-core, NA≈0.22 fiber at visible wavelengths, $V$ is **hundreds** and the
# supported mode count is on the order of **$10^5$**.
#
# So if the claim is literally "there are not enough modes", it is just false.
#
# The real bottleneck (if any) is *not* mode count—it is whether the speckle pattern is averaged
# fast enough during 500 µs.

# %% [markdown]
# ## 4) The key lever: spectral decorrelation width $\Delta\lambda_c$
#
# The professor's comments about "fiber length" and "a couple nm is not enough" are really about this.
#
# ### 4.1 Visual: how $\Delta\lambda_c$ scales with fiber length
#
# A standard (geometric optics) estimate for a step-index MMF is:
#
# $$
# \Delta\lambda_c \sim \frac{2 n_{\rm core}\,\lambda_0^2}{\mathrm{NA}^2 L}.
# $$
#
# In words:
#
# - Longer fiber → smaller $\Delta\lambda_c$ → more independent spectral patterns.
# - Higher NA → smaller $\Delta\lambda_c$ → more spectral diversity.
#
# We also include a *modal dispersion scale* $s$ to represent graded-index behavior:
#
# - $s=1$ means step-index-like dispersion (large modal delay spread)
# - $s \ll 1$ means graded-index-like dispersion (reduced modal delay spread)

# %%
lengths_m = np.logspace(-1, 1.3, 200)  # 0.1 m .. 20 m
scales = [1.0, 0.1, 0.02]

lambda0_nm = 640.0

fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)

for s in scales:
    dlam = []
    for L in lengths_m:
        fib = MultimodeFiber(core_diameter_um=fiber.core_diameter_um, na=fiber.na, length_m=float(L), n_core=fiber.n_core, modal_delay_scale=float(s))
        dlam.append(speckle_spectral_corr_width_nm_for_fiber(lambda0_nm=lambda0_nm, fiber=fib))
    ax.loglog(lengths_m, dlam, label=f"modal_delay_scale={s:g}")

ax.set_xlabel("fiber length L [m]")
ax.set_ylabel(r"speckle corr width $\Delta\lambda_c$ [nm]")
ax.set_title(r"Estimated $\Delta\lambda_c$ vs fiber length (640 nm, NA=0.22)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.show()

# %% [markdown]
# ### 4.2 Text
#
# This plot is the cleanest way to respond to "the fiber would need to be longer than practical":
#
# - For **step-index-like** dispersion ($s\approx1$), even a **few meters** gives *very small*
#   $\Delta\lambda_c$ (typically **hundredths to thousandths of a nm**).
#   That means even a **2 nm**-wide source spans **many** correlation widths.
#
# - For **strong graded-index** behavior ($s\ll 1$), $\Delta\lambda_c$ can be **larger**, and linewidth
#   matters more. In that case, moving from 2 nm to 20 nm *can* be the difference between
#   "barely helps" and "clearly enough".
#
# The important point is: you do not have to argue abstractly. You can treat $s$ as an uncertainty
# and sweep it.
#
# ---
#
# ### 4.3 Pseudocode
#
# ```text
# delta_tau_step ≈ (NA^2 / (2*n_core*c)) * L
# delta_tau ≈ modal_delay_scale * delta_tau_step
# delta_OPL ≈ c * delta_tau
# delta_lambda_c ≈ lambda0^2 / delta_OPL
# N_lambda ≈ ceil(linewidth / delta_lambda_c)
# ```

# %% [markdown]
# ## 5) Put numbers on the quoted fiber (3 m) and linewidth claims
#
# We'll compute:
#
# - group-delay spread $\Delta\tau$
# - optical path spread $\Delta\mathrm{OPL}$
# - spectral correlation width $\Delta\lambda_c$
# - implied spectral diversity $N_{\lambda}$ for several linewidths

# %%
# Fiber dispersions to consider:
# - 1.0: step-index-like (maximal modal dispersion)
# - 0.1: fairly strong GI reduction
# - 0.02: very strong GI reduction

fiber_scales = [1.0, 0.1, 0.02]

rows = []
for s in fiber_scales:
    fib = MultimodeFiber(core_diameter_um=fiber.core_diameter_um, na=fiber.na, length_m=fiber.length_m, n_core=fiber.n_core, modal_delay_scale=s)
    dt = intermodal_group_delay_spread_s(fib)
    dopl = optical_path_spread_m(fib)
    dlam_c = speckle_spectral_corr_width_nm_for_fiber(lambda0_nm=640.0, fiber=fib)
    for label, span in linewidths_nm.items():
        nlam = estimate_n_lambda_from_fiber(lambda0_nm=640.0, source_span_nm=span, fiber=fib)
        rows.append(
            {
                "modal_delay_scale": s,
                "Δτ (ps)": dt * 1e12,
                "ΔOPL (cm)": dopl * 100.0,
                "Δλ_c (nm)": dlam_c,
                "linewidth_case": label,
                "Δλ_src (nm)": span,
                "N_lambda": nlam,
            }
        )

pd.DataFrame(rows)

# %% [markdown]
# ### Quick read
#
# - With **step-index-like** behavior ($s=1$): $\Delta\lambda_c$ is typically around **0.01 nm** for 3 m.
#   So **2 nm** already implies $N_{\lambda}$ in the **hundreds**.
#
# - With **strong GI** reduction ($s=0.02$): $\Delta\lambda_c$ can be on the order of **0.4 nm**.
#   In that case:
#   - **2 nm** gives $N_{\lambda} \sim 5$ (helpful but not magical)
#   - **20 nm** gives $N_{\lambda} \sim 50$ (a big difference)
#
# This is exactly why the professor's intuition can flip depending on what (implicitly) assumed about
# modal dispersion.

# %% [markdown]
# ## 6) Turn $N_\lambda$ into an actual speckle-contrast prediction
#
# We use the simple bookkeeping:
#
# $$
# N_{\mathrm{eff}} \approx N_t\,N_{\lambda}\,N_{\mathrm{pol}},\qquad C \approx 1/\sqrt{N_{\mathrm{eff}}}.
# $$
#
# For a 500 µs exposure and a 10 kHz scrambler:
#
# $$
# N_t \approx f_{\mathrm{scr}}\,\tau \approx 10^4 \times 5\times 10^{-4} \approx 5.
# $$
#
# We'll compute predictions for $N_{\mathrm{pol}}=1$ and $2$.

# %%
N_t = int(round(scrambler_hz * exposure_s))

rows = []
for s in fiber_scales:
    fib = MultimodeFiber(core_diameter_um=fiber.core_diameter_um, na=fiber.na, length_m=fiber.length_m, n_core=fiber.n_core, modal_delay_scale=s)
    for label, span in linewidths_nm.items():
        nlam = estimate_n_lambda_from_fiber(lambda0_nm=640.0, source_span_nm=span, fiber=fib)
        for n_pol in [1, 2]:
            n_eff = max(1, N_t) * nlam * n_pol
            c_pred = 1.0 / np.sqrt(n_eff)
            rows.append(
                {
                    "modal_delay_scale": s,
                    "Δλ_src": label,
                    "N_t": N_t,
                    "N_lambda": nlam,
                    "N_pol": n_pol,
                    "N_eff": int(n_eff),
                    "C_pred~1/sqrt(N_eff)": c_pred,
                }
            )

pred = pd.DataFrame(rows).sort_values(["modal_delay_scale", "Δλ_src", "N_pol"])
pred

# %% [markdown]
# ### Interpretation (what this says about the professor's bullets)
#
# - If the fiber behaves closer to **step-index** ($s\sim1$), then *even* "a couple nm" easily gives
#   $C \ll 0.1$ at 500 µs.
#
# - If the fiber behaves like an **excellent graded-index homogenizer** ($s\ll 1$), then 2 nm might not be
#   enough by itself—but 20 nm is strongly helpful.
#
# - The "~1% of wavelength" heuristic corresponds to ~6–7 nm at 640 nm.
#   In this model it is not a universal threshold—it is a (sometimes reasonable) safety-margin
#   that assumes lower modal dispersion.

# %% [markdown]
# ## 7) Fourier-optics simulation: sanity check the scaling (without waiting forever)
#
# The simulator in this repo generates a sample-plane field by:
#
# 1. Applying a **square field stop** mask.
# 2. Applying a **circular pupil low-pass** set by $\mathrm{NA}_{\mathrm{illum}}$.
# 3. Averaging intensities of independent complex random fields.
#
# It does not attempt to propagate actual guided LP modes, but it *does* capture the
# core Fourier-optics fact: finite NA sets the spatial frequency content, and averaging reduces speckle.
#
# We'll simulate a small set of $N_{\mathrm{eff}}$ values that cover the range from "bad" to "good".

# %%
# Pick a few representative N_eff values (keep runtime reasonable).
N_eff_sim_list = [1, 5, 50, 250]

rows = []
fig, axes = plt.subplots(1, len(N_eff_sim_list), figsize=(4.2 * len(N_eff_sim_list), 4), constrained_layout=True)
if len(N_eff_sim_list) == 1:
    axes = [axes]

for ax, n_eff in zip(axes, N_eff_sim_list):
    I, meta = simulate_excitation_speckle_field(
        n=N_grid,
        dx_um=dx_um,
        roi_um=roi_um,
        lambda_um=0.640,
        na_illum=na_illum,
        exposure_s=exposure_s,
        scrambler_hz=0.0,
        n_src=int(n_eff),
        seed=0,
    )
    In = normalize_by_mask_mean(I, masks.inner)
    C_meas = speckle_contrast(In, masks.inner)
    C_pred = 1.0 / np.sqrt(n_eff)

    rows.append({"N_eff": n_eff, "C_meas": C_meas, "C_pred": C_pred})

    im = ax.imshow(In, origin="lower", vmin=0, vmax=np.percentile(In[masks.roi], 99))
    ax.set_title(f"N_eff={n_eff}\nC_meas={C_meas:.3f}\nC_pred={C_pred:.3f}")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.show()

pd.DataFrame(rows)

# %% [markdown]
# This confirms the expected scaling in the simplified Fourier-optics model.
#
# **Practical take-away:** once you can defend $N_{\mathrm{eff}}$ in the hundreds or thousands,
# the *residual speckle contrast* is usually not the limiting factor.
#
# The remaining risks are typically:
#
# - non-uniform **mean** intensity profile (mode fill / near-field envelope),
# - wavelength-dependent throughput,
# - deterministic diffraction structure at the ROI edge.

# %% [markdown]
# ## 8) "How long does the fiber need to be?" (solve the inverse problem)
#
# This section directly answers the "impractically long fiber" claim.
#
# Suppose you want to guarantee $N_{\lambda} \ge 100$ from a 2 nm source.
# What fiber length does that imply?

# %%
rows = []
for s in fiber_scales:
    L_need = required_fiber_length_m_for_target_n_lambda(
        lambda0_nm=640.0,
        na=fiber.na,
        n_core=fiber.n_core,
        source_span_nm=2.0,
        target_n_lambda=100,
        modal_delay_scale=s,
    )
    rows.append({"modal_delay_scale": s, "L_needed_for_Nλ=100 (m)": L_need})

pd.DataFrame(rows)

# %% [markdown]
# If your fiber is step-index-like, you need *well under* a few meters.
# If it's a very good GI homogenizer, you might need tens of meters to get $N_{\lambda}=100$ from 2 nm.
# But (a) you can also increase linewidth, and (b) you can rely on scrambler/time/polarization diversity.
#
# In other words: "fiber must be impractically long" is not a universal truth—it is a conditional claim.

# %% [markdown]
# ## 9) Addressing the professor's bullets, explicitly
#
# Below is a compact "response sheet" you can take back to the conversation.
# It is written to be falsifiable (you can measure each term).
#
# 1) **"MMF homogeneous field is not practical"**
#    - Practicality depends on whether the *mean* near-field / Köhler source is uniform.
#      Speckle contrast is a separate issue and can be made small with averaging.
#
# 2) **"Speckling would be an issue"**
#    - True for narrow-line, no-averaging illumination ($N_{\mathrm{eff}}\approx1$).
#    - False if you can defend $N_{\mathrm{eff}}\gtrsim100$; then $C\lesssim0.1$.
#
# 3) **"Fiber would need to be longer than practical"**
#    - For step-index-like MMF with NA≈0.22, 3 m already gives $\Delta\lambda_c\sim10^{-2}$ nm.
#      A 2 nm source spans hundreds of correlation widths.
#    - If the fiber is strongly GI (modal dispersion reduced), required length grows as $1/s$.
#      That is why sweeping the dispersion scale $s$ is the right way to argue quantitatively.
#
# 4) **"Not enough uncorrelated modes"**
#    - A 400 µm, NA≈0.22 fiber supports on the order of $10^5$ guided modes at visible wavelengths.
#      "Mode count" is not the bottleneck.
#    - The bottleneck is $N_{\mathrm{eff}}$ during one exposure.
#
# 5) **"A couple of nm linewidth is not enough"**
#    - Under step-index-like dispersion, 2 nm is *massively* larger than $\Delta\lambda_c$.
#    - Under strong GI dispersion reduction, 2 nm may or may not be enough; 20 nm is much safer.
#
# 6) **"~1% of wavelength"**
#    - 1% is ~6.4 nm at 640 nm. In this bookkeeping it is a *heuristic safety margin*,
#      not a fundamental threshold.
#
# ---
#
# ## What to measure (fast, decisive)
#
# If you want to settle this experimentally in a day:
#
# - Measure speckle contrast $C$ in the inner ROI for several exposures.
# - If possible, insert a narrowband filter to reduce linewidth and watch C increase.
# - Compare: scrambler off vs on (or fiber gently agitated vs not).
#
# Then compute $N_{\mathrm{eff}} \approx 1/C^2$ and compare to the budget in this notebook.
