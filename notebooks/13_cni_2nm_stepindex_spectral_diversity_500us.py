
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
# # CNI FC-640-class (2–3 nm linewidth) + 3 m step-index MMF: will **spectral diversity alone** make 500 µs work?
#
# This notebook is an intentionally *engineering-first* “chapter” analysis of the specific, practical question:
#
# > You have a quote for a **fiber-coupled 640 nm source** with **spectral linewidth ~2–3 nm** (CNI FC-640-class).
# > You plan to deliver it through a **3 m, step-index, 400 µm core multimode fiber**.
# > You want to know whether **spectral diversity by itself** (no scramblers, no diffusers, no moving parts,
# > no polarization diversity, no angle diversity) can yield **sufficiently homogeneous excitation** for:
# >
# > - ROI: **30 µm × 30 µm** (100× objective)
# > - exposure: **500 µs**
# > - irradiance: **10–30 kW/cm²**
#
# What we do here
# --------------
# 1. Translate the fiber spec into a **speckle spectral correlation width** $`\Delta\lambda_c`$.
# 2. Translate the vendor “2–3 nm” statement into several **plausible spectral scenarios**
#    (continuous envelope vs a comb of discrete longitudinal modes vs “linewidth” that is only time-averaged).
# 3. For each scenario, compute **two outputs**:
#    - **$C$**, a speckle-contrast metric inside the ROI (homogeneity proxy).
#    - a **Slice0 confusion-matrix proxy** (TP/FP/FN plus a finite-TN “decoy site” proxy),
#      measured on a synthetic sparse-emitter frame.
# 4. Show representative **component speckle fields** and the **summed/averaged field** for each scenario.
#
# Scope / constraints (per your request)
# -------------------------------------
# - Step-index MMF only (no graded-index discussion).
# - Spectral diversity only.
#   - No vibrating fiber, no mechanical scrambler, no diffuser, no angle diversity, no polarization diversity.
# - We care about the **500 µs simultaneity** question:
#   - Are all spectral components present *simultaneously* during a single 500 µs exposure?
#
# **Important realism warning**
# ----------------------------
# Vendor “spectral linewidth ~2–3 nm” can mean different things:
#
# - (Best case) a truly **multi-longitudinal-mode** source emitting many lines at once.
# - (Worst case) a **narrow** source that **mode-hops** over time so the *OSA trace* looks broad,
#   but at any instant (500 µs) only ~1 line is present.
#
# Because of this ambiguity, the correct output of this notebook is not a single number.
# It is a **scenario map**: scenario → (C, confusion matrix).

# %% [markdown]
# ## 0) Imports + repo plumbing

# %%
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# If we are in a notebook, prefer inline backend.
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
    square_roi_mask,
    simulate_excitation_speckle_field,
)
from src.mmf_fiber_speckle import (  # noqa: E402
    MultimodeFiber,
    max_guided_meridional_ray_angle_rad,
    optical_path_spread_geometric_m,
    speckle_spectral_corr_width_nm,
)
from src.speckle_weighting import (  # noqa: E402
    effective_n_from_weights,
    gaussian_spectrum_bins,
    speckle_contrast_from_weights,
    uniform_top_hat_spectrum_bins,
)
from src.slice0_kernel import Slice0Params, detect_spots  # noqa: E402

# Make plots a bit larger by default.
plt.rcParams.update({"figure.figsize": (8.0, 5.0), "figure.dpi": 120})

# %% [markdown]
# ## 1) Configuration (all user-editable)
#
# We separate **what comes from the vendor** (spec-sheet-type values) from
# **what comes from physical materials** (base units).

# %%
# ---------------- Vendor-like inputs ----------------
lambda0_nm = 640.0
linewidth_fwhm_nm_nominal = 2.5  # interpret "2–3 nm" as a nominal FWHM

# From the quote / target requirement
roi_um = 30.0
exposure_us = 500.0

# ---------------- Fiber (step-index) ----------------
fiber = MultimodeFiber(
    core_diameter_um=400.0,
    na=0.22,
    length_m=3.0,
    n_core=1.46,
    modal_delay_scale=1.0,  # step-index assumption
)

# ---------------- Microscope sampling (for Slice0 sim) ----------------
M_obj = 100.0
camera_pixel_um = 6.5
# sample-plane pixel pitch
dx_um = camera_pixel_um / M_obj

# Simulation grid
# 512 is faster but leaves very little margin around a 30 µm ROI at 65 nm/px.
# 640 gives a more comfortable margin while still being reasonably fast.
N_grid = 640

# Inner ROI margin for C computation (exclude field-stop edge roll-off)
inner_margin_um = 2.0

# Effective illumination NA at the sample (BFP underfill knob).
# This is NOT the fiber NA. It is set by your relay optics and underfill fraction.
# We'll keep a single baseline and then do a small sweep later.
NA_obj = 1.40
underfill_ratio = 0.10  # 0.10 means fill ~10% of the objective pupil diameter
NA_illum = underfill_ratio * NA_obj

# RNG seed used for representative visualization fields (separate from Monte Carlo below)
seed_vis = 0

# %% [markdown]
# ### 1.1 Quick power sanity check (not the main topic)
#
# Area of a 30 µm × 30 µm ROI is:
#
# $$
# A = (30\thinspace\mu\mathrm{m})^2 = 900\thinspace\mu\mathrm{m}^2 = 9\times 10^{-6}\thinspace\mathrm{cm}^2.
# $$
#
# So 10–30 kW/cm² corresponds to 90–270 mW at the sample.
# A 2 W class source leaves plenty of margin for losses.

# %%
area_um2 = roi_um * roi_um
area_cm2 = area_um2 / 1e8
for irr_kw_cm2 in [10.0, 30.0]:
    p_w = (irr_kw_cm2 * 1e3) * area_cm2
    print(f"ROI {roi_um:.0f}×{roi_um:.0f} µm² @ {irr_kw_cm2:.0f} kW/cm² -> {p_w*1e3:.0f} mW at sample")

# %% [markdown]
# ## 2) From **base fiber properties** to $`\Delta\lambda_c`$ (speckle spectral correlation width)
#
# **Derivation companion:** if you want the step-by-step geometry, approximation error checks,
# and the whiteboard coherence-time/length intuition in one place, see:
#
# - `notebooks/14_stepindex_mmf_spectral_linewidth_physics.py`
#
#
# The core modeling idea is extremely simple:
#
# - A step-index MMF supports rays/modes at different angles $\theta$.
# - Different angles correspond to different **axial group delays** (different transit times) and thus different
#   **optical path lengths**.
# - If the source has finite linewidth, those different delays wash out coherence between modal contributions.
#
# We compute (geometric optics):
#
# $$
# \theta_{\max} = \arcsin\left(\frac{\mathrm{NA}}{n_{\mathrm{core}}}\right)
# $$
#
# $$
# \Delta\mathrm{OPL} = n_{\mathrm{core}}\,L\left(\frac{1}{\cos\theta_{\max}} - 1\right)
# \approx \frac{\mathrm{NA}^2}{2n_{\mathrm{core}}}\,L
# $$
#
# and then the speckle spectral correlation width:
#
# $$
# \Delta\lambda_c \sim \frac{\lambda_0^2}{\Delta\mathrm{OPL}}.
# $$
#
# This gives an explicit dependency of the *homogeneity proxy* $C$ on base units.
# In the equal-weight, independent-bin limit:
#
# $$
# N_\lambda \approx \frac{\Delta\lambda_{\mathrm{src}}}{\Delta\lambda_c},\qquad
# C \sim \frac{1}{\sqrt{N_\lambda}}\sim \sqrt{\frac{\Delta\lambda_c}{\Delta\lambda_{\mathrm{src}}}}.
# $$
#
# Substituting the small-angle step-index approximation yields:
#
# $$
# C \sim \sqrt{\frac{2 n_{\mathrm{core}}\,\lambda_0^2}{L\,\mathrm{NA}^2\,\Delta\lambda_{\mathrm{src}}}}.
# $$
#
# That equation is the “explicit physical parameter” form:
# $C$ depends on $n_{\mathrm{core}}$, NA, length, and linewidth.

# %%
# Base-parameter computations

theta_max_rad = max_guided_meridional_ray_angle_rad(na=fiber.na, n_core=fiber.n_core)
delta_opl_m = optical_path_spread_geometric_m(fiber)
dlam_c_nm = speckle_spectral_corr_width_nm(lambda0_nm=lambda0_nm, delta_opl_m=delta_opl_m)

# Also compute n_clad implied by NA (air outside): NA^2 = n_core^2 - n_clad^2
n_clad = math.sqrt(max(fiber.n_core * fiber.n_core - fiber.na * fiber.na, 0.0))

df_fiber = pd.DataFrame(
    [
        {
            "lambda0_nm": lambda0_nm,
            "L_m": fiber.length_m,
            "core_diameter_um": fiber.core_diameter_um,
            "n_core": fiber.n_core,
            "n_clad (from NA)": n_clad,
            "NA": fiber.na,
            "theta_max_deg": theta_max_rad * 180.0 / math.pi,
            "ΔOPL (mm)": delta_opl_m * 1e3,
            "Δλ_c (nm)": dlam_c_nm,
        }
    ]
)

df_fiber

# %% [markdown]
# ### 2.1 What does $`\Delta\lambda_c`$ mean operationally?
#
# - If two wavelengths differ by **much less** than $`\Delta\lambda_c`$, they generate **highly correlated speckle**.
#   Their intensities do *not* average down much.
# - If they differ by **much more** than $`\Delta\lambda_c`$, they generate **nearly independent speckle**.
#   Their intensities average down like $C\sim 1/\sqrt{N}$.
#
# With the 3 m, NA=0.22 step-index fiber, $`\Delta\lambda_c`$ is typically **tiny** (often ~0.01 nm scale).
# That means even a “small” linewidth (2–3 nm) spans **hundreds** of correlation widths.
#
# The catch is that this is only useful if the spectrum is **present simultaneously** within the exposure.

# %%
print(f"Computed speckle spectral correlation width: Δλ_c ≈ {dlam_c_nm:.4f} nm")
print(f"Nominal vendor linewidth: Δλ_src ≈ {linewidth_fwhm_nm_nominal:.2f} nm")
print(f"Span / corr_width ≈ {linewidth_fwhm_nm_nominal / dlam_c_nm:.0f} independent bins (upper bound)")

# %% [markdown]
# ## 3) What does “spectral linewidth ~2–3 nm” *actually* mean?
#
# Here are the main interpretations that matter for your **spectral diversity** approach.
# These are not philosophical distinctions — they change $N_\lambda$ by orders of magnitude.
#
# ### Interpretation A: Continuous-ish spectrum (best case)
# A broad envelope is present **simultaneously** (ASE-like, or a very dense comb).
# Then $N_\lambda$ is limited mainly by the fiber’s $`\Delta\lambda_c`$.
#
# ### Interpretation B: Discrete longitudinal modes, all lasing simultaneously (common case)
# A Fabry–Perot diode can lase on **many longitudinal modes** at once.
# The OSA trace shows multiple spikes.
# Then $N_\lambda$ is roughly the **number of spikes** (if each spike is narrower than $`\Delta\lambda_c`$),
# unless the spikes are so dense that they fill the band.
#
# ### Interpretation C: “Linewidth” is time-averaged (worst case for *within-exposure* averaging)
# If the laser mode-hops slowly, an OSA trace may show 2–3 nm wide output, but at any instant
# there may be only ~1 mode (or a few).
# In that case, within a **single 500 µs exposure**, you do *not* get large $N_\lambda$.
# Instead you get time-varying speckle frame-to-frame (multiplicative noise), which is exactly what
# you were trying to avoid.
#
# Because you don’t yet have a time-resolved spectrum measurement for the CNI source, the correct move is:
#
# - define a small set of **scenarios**,
# - compute (C, confusion matrix) for each, and
# - ask the vendor for just enough additional information to collapse the uncertainty.

# %% [markdown]
# ## 4) Scenario definitions (spectrum → weights → $N_{\mathrm{eff}}$ → predicted $C$)
#
# We represent the spectrum by a list of **spectral components** $\{w_i\}$ whose weights sum to 1.
# If each component generates an independent speckle realization,
# then for fully developed speckle the residual contrast is:
#
# $$
# C = \sqrt{\sum_i w_i^2} = \frac{1}{\sqrt{N_{\mathrm{eff}}}},\qquad
# N_{\mathrm{eff}} = \frac{1}{\sum_i w_i^2}.
# $$
#
# This handles both:
# - equal-power spikes (all $w_i=1/N$ → $C=1/\sqrt{N}$), and
# - unequal powers (one line dominates → $C$ stays high).
#
# We build a few scenarios that cover the plausible range for a “2–3 nm linewidth” vendor statement.

# %%
@dataclass(frozen=True)
class Scenario:
    name: str
    kind: str  # "single" | "spikes" | "top_hat" | "gaussian"
    # Spectrum parameters
    span_nm: Optional[float] = None  # for top-hat
    fwhm_nm: Optional[float] = None  # for gaussian
    n_spikes: Optional[int] = None
    spike_weights: Optional[np.ndarray] = None  # if provided, overrides equal weights
    notes: str = ""


def make_equal_spike_weights(n: int) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be >= 1")
    return np.full(n, 1.0 / n, dtype=np.float64)


def make_skewed_spike_weights(n: int, dominant_frac: float) -> np.ndarray:
    """One dominant spike + flat remainder."""
    if n < 2:
        raise ValueError("n must be >= 2")
    if not (0.0 < dominant_frac < 1.0):
        raise ValueError("dominant_frac must be in (0,1)")
    rest = 1.0 - dominant_frac
    w = np.full(n, rest / (n - 1), dtype=np.float64)
    w[0] = dominant_frac
    return w


def make_gaussian_envelope_spike_weights(
    n: int,
    *,
    fwhm_nm: float,
    span_nm: Optional[float] = None,
) -> np.ndarray:
    """Gaussian power envelope across discrete spikes.

    This is a simple surrogate for a multi-longitudinal-mode Fabry–Perot diode:
    many narrow modes, but with a smooth gain-envelope so central modes are stronger.

    We place `n` spikes uniformly across `span_nm` (default: `fwhm_nm`) and assign
    weights proportional to a Gaussian with the requested FWHM.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if fwhm_nm <= 0:
        raise ValueError("fwhm_nm must be > 0")
    if span_nm is None:
        span_nm = float(fwhm_nm)
    if span_nm <= 0:
        raise ValueError("span_nm must be > 0")

    sigma_nm = float(fwhm_nm) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    lam = np.linspace(-0.5 * float(span_nm), 0.5 * float(span_nm), int(n), dtype=np.float64)
    w = np.exp(-0.5 * (lam / sigma_nm) ** 2)
    w = w / float(w.sum())
    return w


# A small set of scenarios spanning "optimistic" to "pessimistic" interpretations.
scenarios: list[Scenario] = [
    Scenario(
        name="C0: instantaneous single-line (worst case)",
        kind="single",
        notes=(
            "OSA may show 2–3 nm because of slow mode hopping, but within 500 µs the output is effectively single-mode."
        ),
    ),
    Scenario(
        name="C1: 10 spikes, equal power",
        kind="spikes",
        n_spikes=10,
        notes="Low-count Fabry–Perot case (or filtered).",
    ),
    Scenario(
        name="C2: 20 spikes, equal power",
        kind="spikes",
        n_spikes=20,
        notes=(
            "Plausible for a short-cavity diode where the 2–3 nm envelope contains ~10–30 longitudinal modes."
        ),
    ),
    Scenario(
        name="C3: 100 spikes, equal power",
        kind="spikes",
        n_spikes=100,
        notes=(
            "Optimistic discrete-line case (dense comb or longer cavity)."
        ),
    ),
    Scenario(
        name="C3b: 100 spikes, Gaussian envelope",
        kind="spikes",
        n_spikes=100,
        spike_weights=make_gaussian_envelope_spike_weights(100, fwhm_nm=linewidth_fwhm_nm_nominal),
        notes=(
            "Many simultaneous longitudinal modes, but with a smooth gain-envelope (central modes stronger)."
        ),
    ),

    Scenario(
        name="C4: 20 spikes, one dominates 50%",
        kind="spikes",
        n_spikes=20,
        spike_weights=make_skewed_spike_weights(20, dominant_frac=0.50),
        notes="If most power is in one line, spectral diversity is much weaker than the FWHM number suggests.",
    ),
    Scenario(
        name="C5: top-hat continuum over 2.5 nm (upper bound)",
        kind="top_hat",
        span_nm=linewidth_fwhm_nm_nominal,
        notes=(
            "Best-case interpretation: spectrum is simultaneously present and roughly flat across 2–3 nm."
        ),
    ),
]


def scenario_weights_and_neff(
    sc: Scenario,
    *,
    lambda0_nm: float,
    corr_width_nm: float,
) -> tuple[np.ndarray, float, str]:
    """Return (weights, N_eff, label) for speckle averaging."""

    if sc.kind == "single":
        w = np.array([1.0], dtype=np.float64)
        return w, 1.0, "single"

    if sc.kind == "spikes":
        if sc.spike_weights is not None:
            w = np.asarray(sc.spike_weights, dtype=np.float64)
        else:
            if sc.n_spikes is None:
                raise ValueError("spikes scenario requires n_spikes")
            w = make_equal_spike_weights(int(sc.n_spikes))
        w = w / float(w.sum())
        neff = float(effective_n_from_weights(w))
        return w, neff, f"{len(w)} spikes"

    if sc.kind == "top_hat":
        if sc.span_nm is None:
            raise ValueError("top_hat scenario requires span_nm")
        bins = uniform_top_hat_spectrum_bins(lambda0_nm=lambda0_nm, span_nm=float(sc.span_nm), corr_width_nm=corr_width_nm)
        w = bins.weights
        neff = float(effective_n_from_weights(w))
        return w, neff, f"top-hat ({len(w)} bins)"

    if sc.kind == "gaussian":
        if sc.fwhm_nm is None:
            raise ValueError("gaussian scenario requires fwhm_nm")
        bins = gaussian_spectrum_bins(lambda0_nm=lambda0_nm, fwhm_nm=float(sc.fwhm_nm), corr_width_nm=corr_width_nm, n_std=2.0)
        w = bins.weights
        neff = float(effective_n_from_weights(w))
        return w, neff, f"gaussian ({len(w)} bins)"

    raise ValueError(f"Unknown scenario kind: {sc.kind}")


rows = []
for sc in scenarios:
    w, neff, label = scenario_weights_and_neff(sc, lambda0_nm=lambda0_nm, corr_width_nm=dlam_c_nm)
    C_pred = float(speckle_contrast_from_weights(w))
    rows.append(
        {
            "scenario": sc.name,
            "spectrum_model": label,
            "N_eff (from weights)": neff,
            "C_pred": C_pred,
            "notes": sc.notes,
        }
    )

df_scen = pd.DataFrame(rows)
df_scen

# %% [markdown]
# ### 4.1 A practical “sanity line” for $C$
#
# A common intuition anchor:
#
# - $C=0.10$ means **10% RMS** intensity fluctuations (relative to mean).
# - $C=0.05$ means **5% RMS**.
#
# Many people will *visually* call something “homogeneous” somewhere in the 0.05–0.10 ballpark,
# but **spot detection** can care about both:
#
# - amplitude (contrast), and
# - correlation length (speckle grain size vs PSF scale).
#
# That’s why we compute both $C$ and a detector-level metric.

# %%
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.axhline(0.10, linestyle="--", alpha=0.7, label="C=0.10 (10% RMS)")
ax.axhline(0.05, linestyle=":", alpha=0.9, label="C=0.05 (5% RMS)")
ax.plot(np.arange(len(df_scen)), df_scen["C_pred"].to_numpy(), marker="o")
ax.set_xticks(np.arange(len(df_scen)))
ax.set_xticklabels([sc.name.split(":")[0] for sc in scenarios])
ax.set_ylabel("Predicted speckle contrast C")
ax.set_title("Scenario-level predicted C from spectral weights")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right")
plt.show()

# %% [markdown]
# ## 5) Simulate excitation fields: component patterns + summed field
#
# We now generate actual 2D excitation fields for a subset of scenarios.
#
# Key point: **averaging does not change speckle grain size**.
# It reduces the *amplitude* of the fluctuations.
#
# So if a single-wavelength speckle pattern is “fine”, the averaged pattern remains “fine” but with lower contrast.
# Whether that is acceptable depends on the PSF scale and your detector thresholding.

# %%
coords_um = (np.arange(N_grid) - N_grid // 2) * dx_um
X_um, Y_um = np.meshgrid(coords_um, coords_um, indexing="xy")

roi_mask = square_roi_mask(N_grid, dx_um, roi_um)
inner_mask = roi_mask & (np.abs(X_um) <= 0.5 * roi_um - inner_margin_um) & (np.abs(Y_um) <= 0.5 * roi_um - inner_margin_um)


def speckle_contrast_inner(I: np.ndarray) -> float:
    vals = I[inner_mask]
    return float(vals.std() / max(vals.mean(), 1e-12))


def simulate_weighted_excitation_field(
    *,
    weights: np.ndarray,
    seed0: int,
    n: int,
    dx_um: float,
    roi_um: float,
    lambda_um: float,
    na_illum: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Simulate weighted incoherent sum of independent speckle intensities.

    Returns (I_sum, components) where:
    - I_sum is normalized to mean(inner ROI)=1
    - components are also normalized the same way (so you can compare visually)

    Notes
    -----
    We enforce independence by using different RNG seeds per component.
    """

    w = np.asarray(weights, dtype=np.float64)
    w = w / float(w.sum())

    # Generate each component as a coherent single realization
    comps: list[np.ndarray] = []
    for k in range(len(w)):
        I_k, _ = simulate_excitation_speckle_field(
            n=n,
            dx_um=dx_um,
            roi_um=roi_um,
            lambda_um=lambda_um,
            na_illum=na_illum,
            exposure_s=1.0,      # irrelevant because scrambler_hz=0
            scrambler_hz=0.0,    # force 1 realization
            n_src=1,
            seed=seed0 + 1000 * k,
        )
        comps.append(I_k)

    # Weighted sum of intensities
    I_sum = np.zeros((n, n), dtype=np.float64)
    for wk, Ik in zip(w, comps):
        I_sum += wk * Ik

    # Normalize by inner ROI mean for easy comparison
    mean_inner = float(I_sum[inner_mask].mean())
    I_sum_n = I_sum / max(mean_inner, 1e-12)
    comps_n = [Ik / max(mean_inner, 1e-12) for Ik in comps]
    return I_sum_n, comps_n


def show_component_and_sum(
    *,
    scenario_name: str,
    I_sum: np.ndarray,
    components: list[np.ndarray],
    n_show_components: int = 3,
) -> None:
    """Visual summary: a few components + the sum + histogram."""

    n_show = min(n_show_components, len(components))
    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, n_show + 2)

    # Components
    for i in range(n_show):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(
            components[i],
            origin="lower",
            extent=[coords_um[0], coords_um[-1], coords_um[0], coords_um[-1]],
            interpolation="nearest",
        )
        ax.set_title(f"component {i+1}")
        ax.add_patch(
            plt.Rectangle(
                (-0.5 * roi_um, -0.5 * roi_um),
                roi_um,
                roi_um,
                fill=False,
                linewidth=1.5,
                color="w",
            )
        )
        ax.set_xlabel("x [µm]")
        ax.set_ylabel("y [µm]")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Sum field
    axS = fig.add_subplot(gs[0, n_show])
    imS = axS.imshow(
        I_sum,
        origin="lower",
        extent=[coords_um[0], coords_um[-1], coords_um[0], coords_um[-1]],
        interpolation="nearest",
    )
    axS.add_patch(
        plt.Rectangle(
            (-0.5 * roi_um, -0.5 * roi_um),
            roi_um,
            roi_um,
            fill=False,
            linewidth=2.0,
            color="w",
        )
    )
    axS.set_title("weighted sum")
    axS.set_xlabel("x [µm]")
    axS.set_ylabel("y [µm]")
    plt.colorbar(imS, ax=axS, fraction=0.046, pad=0.04)

    # Histogram (inner ROI)
    axH = fig.add_subplot(gs[0, n_show + 1])
    vals = I_sum[inner_mask].ravel()
    axH.hist(vals, bins=80, density=True, alpha=0.8)
    axH.set_title("inner ROI PDF")
    axH.set_xlabel(r"$I/\langle I\rangle_{\rm inner}$")
    axH.set_ylabel("PDF")

    # A zoom patch centered in the ROI (to see grain)
    patch_half_um = 5.0
    patch = (
        (np.abs(X_um) <= patch_half_um) & (np.abs(Y_um) <= patch_half_um)
    )
    ys, xs = np.where(patch)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    axZ = fig.add_subplot(gs[1, : n_show + 1])
    imZ = axZ.imshow(
        I_sum[y0:y1, x0:x1],
        origin="lower",
        extent=[-patch_half_um, patch_half_um, -patch_half_um, patch_half_um],
        interpolation="nearest",
    )
    axZ.set_title("zoom (10 µm × 10 µm patch in center)")
    axZ.set_xlabel("x [µm]")
    axZ.set_ylabel("y [µm]")
    plt.colorbar(imZ, ax=axZ, fraction=0.046, pad=0.04)

    C_meas = speckle_contrast_inner(I_sum)
    fig.suptitle(f"{scenario_name} | NA_illum={NA_illum:.3f} | C_inner={C_meas:.3f}")
    plt.show()


# Choose a subset to visualize (full set can be slow)
scenario_names_to_visualize = {
    "C0: instantaneous single-line (worst case)",
    "C2: 20 spikes, equal power",
    "C4: 20 spikes, one dominates 50%",
    "C5: top-hat continuum over 2.5 nm (upper bound)",
}

vis_results = {}

for sc in scenarios:
    if sc.name not in scenario_names_to_visualize:
        continue

    w, neff, label = scenario_weights_and_neff(sc, lambda0_nm=lambda0_nm, corr_width_nm=dlam_c_nm)

    # For very large bin counts (e.g. top-hat continuum), generating every component separately is expensive.
    # If weights are uniform, we can generate the sum directly by averaging N independent realizations.
    uniform_weights = np.allclose(w, w[0])

    if uniform_weights and len(w) > 200:
        # Direct average using n_src = N (equal weights)
        I_sum, _meta = simulate_excitation_speckle_field(
            n=N_grid,
            dx_um=dx_um,
            roi_um=roi_um,
            lambda_um=lambda0_nm * 1e-3,
            na_illum=NA_illum,
            exposure_s=1.0,
            scrambler_hz=0.0,
            n_src=int(len(w)),
            seed=seed_vis,
        )
        I_sum = I_sum / float(I_sum[inner_mask].mean())

        # Still generate a few individual components for display
        comps = []
        for k in range(3):
            I_k, _ = simulate_excitation_speckle_field(
                n=N_grid,
                dx_um=dx_um,
                roi_um=roi_um,
                lambda_um=lambda0_nm * 1e-3,
                na_illum=NA_illum,
                exposure_s=1.0,
                scrambler_hz=0.0,
                n_src=1,
                seed=seed_vis + 1234 * (k + 1),
            )
            comps.append(I_k / float(I_sum[inner_mask].mean()))
    else:
        I_sum, comps = simulate_weighted_excitation_field(
            weights=w,
            seed0=seed_vis,
            n=N_grid,
            dx_um=dx_um,
            roi_um=roi_um,
            lambda_um=lambda0_nm * 1e-3,
            na_illum=NA_illum,
        )

    vis_results[sc.name] = {
        "weights": w,
        "N_eff": neff,
        "I_sum": I_sum,
        "components": comps,
        "C_inner": speckle_contrast_inner(I_sum),
        "label": label,
    }

    # NOTE: `sc.name` is the scenario identifier; a previous revision used an undefined
    # `scenario_name` variable, which raised a NameError when this cell was run.
    show_component_and_sum(scenario_name=sc.name, I_sum=I_sum, components=comps, n_show_components=3)

# %% [markdown]
# ## 6) Scenario → (C, confusion matrix): Slice0 FP/FN proxy
#
# The question you care about is not purely “is the field pretty?”.
# It is: **does the nonuniformity cause missed spots or false positives?**
#
# We reuse the synthetic-emitter setup from `notebooks/05_excitation_speckle_fpfn_proxy.py`:
#
# - emitters are placed randomly (non-overlapping) inside the inner ROI
# - brightness is proportional to local excitation intensity
# - background can include a component proportional to excitation (to emulate structured background)
# - we run the actual `Slice0` detector and compute TP/FP/FN
# - TN is approximated via “decoy sites” (random points that should be empty)
#
# We do **not** claim this is a full physical microscopy simulator.
# It is a *consistent, reproducible* way to measure whether different speckle scenarios produce meaningfully
# different detector behavior.

# %%
@dataclass(frozen=True)
class EmitterSimConfig:
    """Lightweight synthetic frame model for Slice0 FP/FN intuition.

    The goal is *not* a full microscope simulator. The goal is to generate frames
    whose spot brightness is in the right ballpark for the *real* Slice0 acceptance
    threshold `u0_min≈30` (mean in5 minus background ring).

    If `photons_per_emitter_mean` is too low, even a perfectly uniform illumination
    field will produce near-zero recall, which is a misleading failure mode for the
    speckle study. So we intentionally choose photon counts such that the **uniform
    illumination positive control** yields near-perfect detection.
    """

    n_emitters: int = 60
    min_separation_px: int = 18

    # Total signal photons per emitter (before PSF spreading), scaled so that
    # u0 ≈ mean(in5) - bg comfortably exceeds u0_min for typical emitters at I_exc≈1.
    photons_per_emitter_mean: float = 5000.0
    photons_per_emitter_sigma_frac: float = 0.15
    photons_per_emitter_min_frac: float = 0.5
    photons_per_emitter_max_frac: float = 2.0

    # Background (shot-noise limited), with an optional component proportional to excitation.
    bg_photons_flat: float = 5.0
    bg_photons_scale_exc: float = 15.0

    # PSF model (Gaussian; sigma should be comparable to the Slice0 LoG scale).
    psf_sigma_px: float = 2.9
    psf_kernel_radius_px: int = 12


cfg = EmitterSimConfig()

# Slice0 params (tune u0_min here)
# Slice0 params: match the repo's integrated config (and your local YAML) as closely as possible.
params = Slice0Params(
    pixel_size_nm=dx_um * 1e3,  # µm → nm
    spot_radius_nm=270.0,       # ~= sqrt(lambda*zR/pi) for (lambda=667 nm, zR=344.5 nm)
    q_min=3.0,
    u0_min=30.0,
)

# Precompute a Gaussian PSF kernel (normalized to sum=1)
R = cfg.psf_kernel_radius_px
yy, xx = np.mgrid[-R : R + 1, -R : R + 1]
psf = np.exp(-(xx * xx + yy * yy) / (2.0 * cfg.psf_sigma_px * cfg.psf_sigma_px))
psf /= psf.sum()


def place_emitters_nonoverlap(
    *,
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


def simulate_sparse_emitter_frame(
    *,
    I_exc: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (img_noisy, gt_yx) for one frame."""

    gt_yx = place_emitters_nonoverlap(
        n_emitters=cfg.n_emitters,
        roi_mask=inner_mask,
        min_sep_px=cfg.min_separation_px,
        rng=rng,
    )
    emit_y = gt_yx[:, 0]
    emit_x = gt_yx[:, 1]

    photons = cfg.photons_per_emitter_mean * (
        1.0 + cfg.photons_per_emitter_sigma_frac * rng.standard_normal(cfg.n_emitters)
    )
    photons = np.clip(
        photons,
        cfg.photons_per_emitter_min_frac * cfg.photons_per_emitter_mean,
        cfg.photons_per_emitter_max_frac * cfg.photons_per_emitter_mean,
    )

    exc_loc = I_exc[emit_y, emit_x]

    img = np.zeros((N_grid, N_grid), dtype=np.float64)

    # Background
    img += cfg.bg_photons_flat + cfg.bg_photons_scale_exc * I_exc

    # Add emitters
    for y0, x0, p0, e0 in zip(emit_y, emit_x, photons, exc_loc):
        amp = float(p0 * e0)
        y1 = y0 - R
        y2 = y0 + R + 1
        x1 = x0 - R
        x2 = x0 + R + 1
        if y1 < 0 or x1 < 0 or y2 > N_grid or x2 > N_grid:
            continue
        img[y1:y2, x1:x2] += amp * psf

    img_noisy = rng.poisson(img).astype(np.float32)
    return img_noisy, gt_yx


def eval_detections_confusion_proxy(
    *,
    df_det: pd.DataFrame,
    gt_yx: np.ndarray,
    rng: np.random.Generator,
    match_r_px: float = 3.0,
    n_decoy: int = 200,
) -> dict[str, int | float]:
    """Compute TP/FP/FN + decoy-site TN/FP proxy."""

    det_xy = df_det[["y_px", "x_px"]].to_numpy(dtype=float)
    gt_xy = gt_yx.astype(float)

    tp_det = np.zeros(det_xy.shape[0], dtype=bool)
    gt_matched = np.zeros(gt_xy.shape[0], dtype=bool)

    for i, (y, x) in enumerate(det_xy):
        if gt_xy.shape[0] == 0:
            break
        dy = gt_xy[:, 0] - y
        dx = gt_xy[:, 1] - x
        d2 = dy * dy + dx * dx
        j = int(np.argmin(d2))
        if d2[j] <= match_r_px * match_r_px and not gt_matched[j]:
            tp_det[i] = True
            gt_matched[j] = True

    TP = int(tp_det.sum())
    FP = int((~tp_det).sum())
    FN = int((~gt_matched).sum())

    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)

    # Finite TN proxy: decoy sites in inner ROI not near any GT emitter
    decoy_xy: list[tuple[int, int]] = []
    tries = 0
    while len(decoy_xy) < n_decoy and tries < 200000:
        tries += 1
        y = int(rng.integers(0, N_grid))
        x = int(rng.integers(0, N_grid))
        if not inner_mask[y, x]:
            continue
        # far from GT
        dy = gt_yx[:, 0] - y
        dx = gt_yx[:, 1] - x
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

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN*": int(TN),
        "FP*": int(FP_decoy),
        "precision": float(precision),
        "recall": float(recall),
    }


def run_monte_carlo_for_field(
    *,
    I_exc: np.ndarray,
    n_trials: int,
    seed0: int,
) -> dict[str, float]:
    """Aggregate confusion metrics over many random emitter/noise realizations."""

    totals = {"TP": 0, "FP": 0, "FN": 0, "TN*": 0, "FP*": 0}
    precs: list[float] = []
    recs: list[float] = []

    for t in range(n_trials):
        rng = np.random.default_rng(seed0 + 10_000 * t)
        img, gt = simulate_sparse_emitter_frame(I_exc=I_exc, rng=rng)
        df_det = detect_spots(img, params)
        out = eval_detections_confusion_proxy(df_det=df_det, gt_yx=gt, rng=rng)

        for k in totals:
            totals[k] += int(out[k])  # type: ignore[arg-type]
        precs.append(float(out["precision"]))
        recs.append(float(out["recall"]))

    TP = totals["TP"]
    FP = totals["FP"]
    FN = totals["FN"]
    TN = totals["TN*"]
    FPd = totals["FP*"]

    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)

    # Decoy-based specificity proxy
    specificity = TN / max(TN + FPd, 1)

    return {
        "TP": float(TP),
        "FP": float(FP),
        "FN": float(FN),
        "TN*": float(TN),
        "FP*": float(FPd),
        "precision": float(precision),
        "recall": float(recall),
        "F1": float(f1),
        "specificity*": float(specificity),
        "precision_mean": float(np.mean(precs)),
        "precision_std": float(np.std(precs)),
        "recall_mean": float(np.mean(recs)),
        "recall_std": float(np.std(recs)),
    }


# Run a modest Monte Carlo for each scenario (use fewer trials if this is slow on your machine)
mc_trials = 20

scenario_exc_fields: dict[str, np.ndarray] = {}

rows = []
for sc in scenarios:
    # Build (or reuse) a representative excitation field for the scenario.
    # For speed we reuse the single visualization field we already generated when available.
    if sc.name in vis_results:
        I_exc = vis_results[sc.name]["I_sum"]
    else:
        w, _neff, _label = scenario_weights_and_neff(sc, lambda0_nm=lambda0_nm, corr_width_nm=dlam_c_nm)
        uniform_weights = np.allclose(w, w[0])
        if uniform_weights:
            I_exc, _ = simulate_excitation_speckle_field(
                n=N_grid,
                dx_um=dx_um,
                roi_um=roi_um,
                lambda_um=lambda0_nm * 1e-3,
                na_illum=NA_illum,
                exposure_s=1.0,
                scrambler_hz=0.0,
                n_src=int(len(w)),
                seed=seed_vis,
            )
            I_exc = I_exc / float(I_exc[inner_mask].mean())
        else:
            I_exc, _comps = simulate_weighted_excitation_field(
                weights=w,
                seed0=seed_vis,
                n=N_grid,
                dx_um=dx_um,
                roi_um=roi_um,
                lambda_um=lambda0_nm * 1e-3,
                na_illum=NA_illum,
            )

    # Keep the representative excitation field for later visual QC.
    scenario_exc_fields[sc.name] = I_exc

    # Metrics
    C_inner = speckle_contrast_inner(I_exc)
    w, neff, label = scenario_weights_and_neff(sc, lambda0_nm=lambda0_nm, corr_width_nm=dlam_c_nm)
    C_pred = float(speckle_contrast_from_weights(w))

    stats = run_monte_carlo_for_field(I_exc=I_exc, n_trials=mc_trials, seed0=1_000_000)

    rows.append(
        {
            "scenario": sc.name,
            "spectrum_model": label,
            "N_eff": neff,
            "C_pred": C_pred,
            "C_inner_meas": C_inner,
            **stats,
        }
    )


# Positive control: perfectly uniform illumination (no speckle) to sanity-check the confusion-matrix logic.
positive_control_name = "P0: uniform illumination (positive control)"
I_exc_pc = np.ones((N_grid, N_grid), dtype=np.float64)
scenario_exc_fields[positive_control_name] = I_exc_pc

pc_stats = run_monte_carlo_for_field(I_exc=I_exc_pc, n_trials=mc_trials, seed0=2_000_000)
rows.append(
    {
        "scenario": positive_control_name,
        "spectrum_model": "uniform (control)",
        "N_eff": float("inf"),
        "C_pred": 0.0,
        "C_inner_meas": 0.0,
        **pc_stats,
    }
)

if pc_stats["recall"] < 0.95 or pc_stats["precision"] < 0.95:
    print(
        "WARNING: Positive control is not near-perfect. "
        "This usually means the synthetic photon budget / noise model is mismatched to Slice0 thresholds."
    )



df_out = pd.DataFrame(rows)

# Compact display
cols_show = [
    "scenario",
    "spectrum_model",
    "N_eff",
    "C_pred",
    "C_inner_meas",
    "precision",
    "recall",
    "F1",
    "specificity*",
    "FP",
    "FN",
]

df_out[cols_show]

# %% [markdown]
# ### 6.1 Plot: scenario → (C, detection metrics)

# %%
fig, ax = plt.subplots(figsize=(8, 4.0))
ax.plot(df_out["C_inner_meas"], df_out["recall"], marker="o", linestyle="-", label="recall")
ax.plot(df_out["C_inner_meas"], df_out["precision"], marker="s", linestyle="-", label="precision")
ax.set_xlabel("Measured C (inner ROI)")
ax.set_ylabel("metric")
ax.set_title("Scenario-level detector metrics vs measured speckle contrast")
ax.grid(True, alpha=0.3)
ax.legend()
plt.show()

# %% [markdown]
# ### 6.2 Visual QC: representative **raw pixel** frames per scenario
#
# The Monte Carlo table above is an aggregate. As a sanity check / intuition builder, we render **one
# deterministic synthetic raw frame per scenario** and show:
#
# - a **full-field** view (raw counts, colorbar, GT emitters + detections), and
# - multiple **zoom-ins** spanning dim → bright emitters, with the **pixel values written directly over the pixels**
#   (plus an outline of the Slice0 in5 aperture).
#
# This makes it visually obvious what the detector is operating on, and lets you sanity-check that the
# aggregate confusion-matrix behavior matches what you see in representative frames.

# %%
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

# --- Deterministic emitter set shared across scenarios (so only illumination changes) ---
qc_seed_emitters = 424242
_rng_emit = np.random.default_rng(qc_seed_emitters)
gt_yx_qc = place_emitters_nonoverlap(
    n_emitters=cfg.n_emitters,
    roi_mask=inner_mask,
    min_sep_px=cfg.min_separation_px,
    rng=_rng_emit,
)
photons_qc = cfg.photons_per_emitter_mean * (
    1.0 + cfg.photons_per_emitter_sigma_frac * _rng_emit.standard_normal(cfg.n_emitters)
)
photons_qc = np.clip(
    photons_qc,
    cfg.photons_per_emitter_min_frac * cfg.photons_per_emitter_mean,
    cfg.photons_per_emitter_max_frac * cfg.photons_per_emitter_mean,
).astype(np.float64)


def _mask_edge_segments(mask: np.ndarray) -> np.ndarray:
    """External pixel-edge segments for a binary mask.

    This draws *between* pixels (at half-integer coordinates) so overlays don't obscure pixel-value text.
    Coordinate convention matches imshow() default: pixel centers at integer coords, edges at half-integers.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or m.size == 0 or not np.any(m):
        return np.zeros((0, 2, 2), dtype=float)

    h, w = m.shape
    segs: list[list[tuple[float, float]]] = []

    for i in range(h):
        for j in range(w):
            if not m[i, j]:
                continue

            x0 = float(j) - 0.5
            x1 = float(j) + 0.5
            y0 = float(i) - 0.5
            y1 = float(i) + 0.5

            if i == 0 or not m[i - 1, j]:
                segs.append([(x0, y0), (x1, y0)])
            if i == h - 1 or not m[i + 1, j]:
                segs.append([(x0, y1), (x1, y1)])
            if j == 0 or not m[i, j - 1]:
                segs.append([(x0, y0), (x0, y1)])
            if j == w - 1 or not m[i, j + 1]:
                segs.append([(x1, y0), (x1, y1)])

    if not segs:
        return np.zeros((0, 2, 2), dtype=float)
    return np.asarray(segs, dtype=float)


def _add_edge_overlay(ax, segs: np.ndarray, *, color: str = "y", lw: float = 1.1):
    if segs is None or np.asarray(segs).size == 0:
        return None
    lc = LineCollection(
        np.asarray(segs, dtype=float),
        colors=[color],
        linewidths=float(lw),
        capstyle="projecting",
        joinstyle="miter",
        zorder=5,
    )
    ax.add_collection(lc)
    return lc


def _annotate_pixel_values(ax, img: np.ndarray, *, fmt: str = "{:d}", fontsize: int = 6) -> None:
    """Write pixel values over each pixel center (int formatting)."""
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            v = img[i, j]
            try:
                s = fmt.format(int(v))
            except Exception:
                s = str(v)
            ax.text(
                j,
                i,
                s,
                ha="center",
                va="center",
                fontsize=int(fontsize),
                color="white",
                path_effects=[pe.withStroke(linewidth=1.25, foreground="black")],
                zorder=6,
            )


def simulate_sparse_emitter_frame_fixed_emitters(
    *,
    I_exc: np.ndarray,
    gt_yx: np.ndarray,
    photons: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Simulate one raw noisy frame using a fixed emitter set (deterministic geometry)."""
    rng = np.random.default_rng(int(seed))

    img = np.zeros((N_grid, N_grid), dtype=np.float64)
    img += cfg.bg_photons_flat + cfg.bg_photons_scale_exc * I_exc

    # Add emitters
    for (y0, x0), p0 in zip(gt_yx, photons):
        amp = float(p0 * I_exc[int(y0), int(x0)])
        y1 = int(y0) - R
        y2 = int(y0) + R + 1
        x1 = int(x0) - R
        x2 = int(x0) + R + 1
        img[y1:y2, x1:x2] += amp * psf

    return rng.poisson(img).astype(np.float32)


def match_detections_to_gt(
    *,
    df_det: pd.DataFrame,
    gt_yx: np.ndarray,
    match_r_px: float = 3.0,
) -> dict[str, np.ndarray]:
    """Greedy 1:1 nearest-neighbor matching (detection -> GT)."""
    det_yx = df_det[["y_px", "x_px"]].to_numpy(dtype=float)
    gt_yx_f = gt_yx.astype(float)

    det_to_gt = -np.ones(det_yx.shape[0], dtype=int)
    gt_to_det = -np.ones(gt_yx_f.shape[0], dtype=int)

    for i, (y, x) in enumerate(det_yx):
        if gt_yx_f.shape[0] == 0:
            break
        dy = gt_yx_f[:, 0] - y
        dx = gt_yx_f[:, 1] - x
        d2 = dy * dy + dx * dx
        j = int(np.argmin(d2))
        if d2[j] <= match_r_px * match_r_px and gt_to_det[j] < 0:
            det_to_gt[i] = j
            gt_to_det[j] = i

    tp_det = det_to_gt >= 0
    fp_det = det_to_gt < 0
    fn_gt = gt_to_det < 0

    return {
        "det_yx": det_yx,
        "det_to_gt": det_to_gt,
        "gt_to_det": gt_to_det,
        "tp_det": tp_det,
        "fp_det": fp_det,
        "fn_gt": fn_gt,
    }


def _roi_outline_rect_px(*, half_size_um: float, edge_color: str, lw: float) -> Rectangle:
    half_px = int(round(half_size_um / dx_um))
    x0 = (N_grid // 2) - half_px - 0.5
    y0 = (N_grid // 2) - half_px - 0.5
    side = 2 * half_px
    return Rectangle((x0, y0), side, side, fill=False, edgecolor=edge_color, linewidth=float(lw))


def show_full_field_qc(
    *,
    scenario_name: str,
    img_raw: np.ndarray,
    gt_yx: np.ndarray,
    df_det: pd.DataFrame,
    match: dict[str, np.ndarray],
) -> None:
    det_yx = match["det_yx"]
    tp_det = match["tp_det"]
    fp_det = match["fp_det"]
    fn_gt = match["fn_gt"]

    TP = int(tp_det.sum())
    FP = int(fp_det.sum())
    FN = int(fn_gt.sum())

    fig, ax = plt.subplots(figsize=(7.0, 6.6))
    # Robust contrast for visibility while remaining in raw-count units.
    vmin, vmax = np.percentile(img_raw[inner_mask], [1.0, 99.9])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.min(img_raw)), float(np.max(img_raw))

    im = ax.imshow(img_raw, origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.add_patch(_roi_outline_rect_px(half_size_um=0.5 * roi_um, edge_color="w", lw=1.5))
    ax.add_patch(_roi_outline_rect_px(half_size_um=0.5 * roi_um - inner_margin_um, edge_color="w", lw=1.0))

    # GT emitters
    ax.scatter(gt_yx[:, 1], gt_yx[:, 0], s=22, facecolors="none", edgecolors="w", linewidths=1.0, label="GT emitters")

    # FN GT emitters (missed)
    if np.any(fn_gt):
        ax.scatter(
            gt_yx[fn_gt, 1],
            gt_yx[fn_gt, 0],
            s=48,
            facecolors="none",
            edgecolors="r",
            linewidths=1.6,
            label="GT missed (FN)",
        )

    # Detections
    # Use *rings* (unfilled markers) instead of "+/x" so the underlying raw pixels remain visible.
    if det_yx.size > 0:
        if np.any(tp_det):
            ax.scatter(
                det_yx[tp_det, 1],
                det_yx[tp_det, 0],
                s=70,
                marker="o",
                facecolors="none",
                edgecolors="lime",
                linewidths=1.4,
                label="detections (TP)",
            )
        if np.any(fp_det):
            ax.scatter(
                det_yx[fp_det, 1],
                det_yx[fp_det, 0],
                s=60,
                marker="s",
                facecolors="none",
                edgecolors="orange",
                linewidths=1.2,
                label="detections (FP)",
            )

    ax.set_title(f"{scenario_name}\nRepresentative raw frame | TP={TP}  FP={FP}  FN={FN}")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="raw counts / pixel")
    plt.show()


def select_emitters_dim_to_bright(
    *,
    gt_yx: np.ndarray,
    photons: np.ndarray,
    I_exc: np.ndarray,
    n_show: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose emitter indices spanning dim → bright in terms of (photons × I_exc)."""

    amp = photons * I_exc[gt_yx[:, 0], gt_yx[:, 1]]
    order = np.argsort(amp)

    if int(n_show) >= int(len(order)):
        chosen = order
    else:
        pick = np.linspace(0, len(order) - 1, int(n_show)).round().astype(int)
        chosen = order[pick]

    return chosen.astype(int), amp


def show_emitter_montage_qc(
    *,
    scenario_name: str,
    img_raw: np.ndarray,
    I_exc: np.ndarray,
    gt_yx: np.ndarray,
    photons: np.ndarray,
    df_det: pd.DataFrame,
    match: dict[str, np.ndarray],
    n_show: int = 15,
    crop_radius_px: int = 5,
) -> None:
    det_yx = match["det_yx"]
    gt_to_det = match["gt_to_det"]

    r = int(crop_radius_px)
    crop_size = 2 * r + 1

    # Choose emitters spanning dim → bright expected amplitude (intrinsic photons × local excitation)
    chosen, amp = select_emitters_dim_to_bright(
        gt_yx=gt_yx,
        photons=photons,
        I_exc=I_exc,
        n_show=n_show,
    )
    n_show = int(len(chosen))

    n_cols = 5
    n_rows = int(np.ceil(n_show / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.6 * n_rows), constrained_layout=True)

    # Shared scaling (raw counts)
    vmin, vmax = np.percentile(img_raw[inner_mask], [1.0, 99.9])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.min(img_raw)), float(np.max(img_raw))

    # Slice0 in5 aperture outline (disk mask) in crop coordinates.
    from src.slice0_kernel import _disk_mask  # local import: private helper, but matches detector exactly

    in5_mask = _disk_mask(int(params.in5_radius_px), crop_size)
    in5_segs = _mask_edge_segments(in5_mask)

    last_im = None
    for k, idx in enumerate(chosen):
        ax = axes.flat[k] if hasattr(axes, "flat") else axes[k]
        y0, x0 = (int(gt_yx[idx, 0]), int(gt_yx[idx, 1]))

        crop = img_raw[y0 - r : y0 + r + 1, x0 - r : x0 + r + 1]
        if crop.shape != (crop_size, crop_size):
            ax.axis("off")
            continue

        # Matched detection (if any)
        det_idx = int(gt_to_det[idx])
        det_off = None
        status = "FN"
        if det_idx >= 0:
            dy = int(round(float(det_yx[det_idx, 0]) - y0))
            dx = int(round(float(det_yx[det_idx, 1]) - x0))
            det_off = (dy, dx)
            status = "TP"

        last_im = ax.imshow(crop, origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax)

        # Pixel grid
        ax.set_xticks(np.arange(-0.5, crop_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, crop_size, 1), minor=True)
        ax.grid(which="minor", linestyle="-", linewidth=0.5, alpha=0.25)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

        # Pixel values
        _annotate_pixel_values(ax, crop, fmt="{:d}", fontsize=6)

        # Mark GT center *without* obscuring the central pixel value.
        # Outline the central pixel (edges are at half-integer coords).
        ax.add_patch(
            Rectangle(
                (r - 0.5, r - 0.5),
                1,
                1,
                fill=False,
                edgecolor="cyan",
                linewidth=1.2,
                zorder=7,
            )
        )

        # Mark detection center + measurement aperture outline.
        # Again: outline the detection pixel instead of drawing a marker on top of pixel text.
        segs = in5_segs
        if det_off is not None:
            ax.add_patch(
                Rectangle(
                    (r + det_off[1] - 0.5, r + det_off[0] - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="lime",
                    linewidth=1.2,
                    zorder=7,
                )
            )
            segs = segs.copy()
            segs[:, :, 0] += float(det_off[1])  # shift x
            segs[:, :, 1] += float(det_off[0])  # shift y

        _add_edge_overlay(ax, segs, color="yellow", lw=1.1)

        ax.set_title(f"{status} | amp≈{amp[idx]:.0f}", fontsize=9)

    # Turn off any unused axes
    for j in range(n_show, n_rows * n_cols):
        ax = axes.flat[j] if hasattr(axes, "flat") else axes[j]
        ax.axis("off")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes, fraction=0.015, pad=0.02, label="raw counts / pixel")

    fig.suptitle(f"{scenario_name} | emitter crops (11×11) with pixel values + in5 aperture", fontsize=12)
    plt.show()


def show_emitter_context_montage_qc(
    *,
    scenario_name: str,
    img_raw: np.ndarray,
    I_exc: np.ndarray,
    gt_yx: np.ndarray,
    photons: np.ndarray,
    df_det: pd.DataFrame,
    match: dict[str, np.ndarray],
    n_show: int = 9,
    crop_radius_px: int = 15,
) -> None:
    """Show a larger-FOV montage around emitters to visualize local background structure.

    This complements `show_emitter_montage_qc()`:
    - small crops (11×11) show exact pixel values,
    - these larger crops (e.g. 31×31) show surrounding background / speckle context.
    """

    det_yx = match["det_yx"]
    gt_to_det = match["gt_to_det"]

    r = int(crop_radius_px)
    crop_size = 2 * r + 1

    chosen, amp = select_emitters_dim_to_bright(
        gt_yx=gt_yx,
        photons=photons,
        I_exc=I_exc,
        n_show=n_show,
    )
    n_show = int(len(chosen))

    n_cols = 3
    n_rows = int(np.ceil(n_show / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.6 * n_rows), constrained_layout=True)

    # Use global scaling so crops can be compared.
    vmin, vmax = np.percentile(img_raw[inner_mask], [1.0, 99.9])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.min(img_raw)), float(np.max(img_raw))

    # in5 aperture overlay (same definition Slice0 uses)
    from src.slice0_kernel import _disk_mask  # local import: private helper, but matches detector exactly
    in5_mask = _disk_mask(int(params.in5_radius_px), crop_size)
    in5_segs = _mask_edge_segments(in5_mask)

    last_im = None
    for k, idx in enumerate(chosen):
        ax = axes.flat[k] if hasattr(axes, "flat") else axes[k]
        y0, x0 = int(gt_yx[idx, 0]), int(gt_yx[idx, 1])

        crop = img_raw[y0 - r : y0 + r + 1, x0 - r : x0 + r + 1]
        if crop.shape != (crop_size, crop_size):
            ax.axis("off")
            continue

        last_im = ax.imshow(crop, origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

        # Mark the GT emitter center in crop coordinates (ring does not obscure pixels)
        ax.plot([r], [r], marker="o", mfc="none", mec="white", ms=10, mew=1.2, zorder=6)
        _add_edge_overlay(ax, in5_segs, color="yellow", lw=1.0)

        det_i = int(gt_to_det[idx]) if gt_to_det.size > idx else -1
        status = "FN"
        if det_i >= 0:
            status = "TP"
            dy = float(det_yx[det_i, 0]) - float(y0)
            dx = float(det_yx[det_i, 1]) - float(x0)
            # Use a ring marker so the spot remains visible under the annotation.
            ax.plot([r + dx], [r + dy], marker="o", mfc="none", mec="lime", ms=10, mew=1.4, zorder=7)

        ax.set_title(f"{status} | amp≈{amp[idx]:.0f}", fontsize=10)

    for j in range(n_show, n_rows * n_cols):
        ax = axes.flat[j] if hasattr(axes, "flat") else axes[j]
        ax.axis("off")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes, fraction=0.03, pad=0.02, label="raw counts / pixel")

    fig.suptitle(f"{scenario_name} | emitter context crops ({crop_size}×{crop_size})", fontsize=12)
    plt.show()


def show_fp_montage_qc(
    *,
    scenario_name: str,
    img_raw: np.ndarray,
    df_det: pd.DataFrame,
    match: dict[str, np.ndarray],
    max_fp: int = 6,
    crop_radius_px: int = 5,
) -> None:
    fp_det = match["fp_det"]
    det_yx = match["det_yx"]
    if det_yx.size == 0 or not np.any(fp_det):
        return

    r = int(crop_radius_px)
    crop_size = 2 * r + 1
    from src.slice0_kernel import _disk_mask  # local import: private helper, but matches detector exactly

    in5_mask = _disk_mask(int(params.in5_radius_px), crop_size)
    in5_segs = _mask_edge_segments(in5_mask)

    fp_idx = np.where(fp_det)[0][: int(max_fp)]
    n_show = len(fp_idx)
    n_cols = 3
    n_rows = int(np.ceil(n_show / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.6 * n_rows), constrained_layout=True)

    vmin, vmax = np.percentile(img_raw[inner_mask], [1.0, 99.9])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.min(img_raw)), float(np.max(img_raw))

    last_im = None
    for k, det_i in enumerate(fp_idx):
        ax = axes.flat[k] if hasattr(axes, "flat") else axes[k]
        y0 = int(round(float(det_yx[det_i, 0])))
        x0 = int(round(float(det_yx[det_i, 1])))

        crop = img_raw[y0 - r : y0 + r + 1, x0 - r : x0 + r + 1]
        if crop.shape != (crop_size, crop_size):
            ax.axis("off")
            continue

        last_im = ax.imshow(crop, origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(-0.5, crop_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, crop_size, 1), minor=True)
        ax.grid(which="minor", linestyle="-", linewidth=0.5, alpha=0.25)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

        _annotate_pixel_values(ax, crop, fmt="{:d}", fontsize=6)
        # Outline the detection-center pixel (do not cover pixel text)
        ax.add_patch(
            Rectangle(
                (r - 0.5, r - 0.5),
                1,
                1,
                fill=False,
                edgecolor="orange",
                linewidth=1.3,
                zorder=7,
            )
        )
        _add_edge_overlay(ax, in5_segs, color="yellow", lw=1.1)
        ax.set_title("FP (crop)", fontsize=9)

    for j in range(n_show, n_rows * n_cols):
        ax = axes.flat[j] if hasattr(axes, "flat") else axes[j]
        ax.axis("off")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes, fraction=0.04, pad=0.02, label="raw counts / pixel")

    fig.suptitle(f"{scenario_name} | example false positives (pixel-annotated)", fontsize=12)
    plt.show()


# Render one representative raw frame per scenario.
qc_seed_frame_base = 1_234_567
scenario_qc_names = [sc.name for sc in scenarios] + [positive_control_name]
for k, scenario_name in enumerate(scenario_qc_names):
    I_exc = scenario_exc_fields[scenario_name]
    img_qc = simulate_sparse_emitter_frame_fixed_emitters(
        I_exc=I_exc,
        gt_yx=gt_yx_qc,
        photons=photons_qc,
        seed=qc_seed_frame_base + 100 * k,
    )
    df_det_qc = detect_spots(img_qc, params)
    match = match_detections_to_gt(df_det=df_det_qc, gt_yx=gt_yx_qc, match_r_px=3.0)

    # Full field: population view
    show_full_field_qc(
        scenario_name=scenario_name,
        img_raw=img_qc,
        gt_yx=gt_yx_qc,
        df_det=df_det_qc,
        match=match,
    )

    # Zoom-ins: enough emitters to build intuition for the aggregate confusion matrix
    show_emitter_montage_qc(
        scenario_name=scenario_name,
        img_raw=img_qc,
        I_exc=I_exc,
        gt_yx=gt_yx_qc,
        photons=photons_qc,
        df_det=df_det_qc,
        match=match,
        n_show=15,
        crop_radius_px=5,
    )

    show_emitter_context_montage_qc(
        scenario_name=scenario_name,
        img_raw=img_qc,
        I_exc=I_exc,
        gt_yx=gt_yx_qc,
        photons=photons_qc,
        df_det=df_det_qc,
        match=match,
        n_show=9,
        crop_radius_px=15,
    )

    # Optional: show a few example FPs to understand failure modes
    show_fp_montage_qc(
        scenario_name=scenario_name,
        img_raw=img_qc,
        df_det=df_det_qc,
        match=match,
        max_fp=6,
        crop_radius_px=5,
    )



# %% [markdown]
# ## 7) Speckle grain size and the underfill ratio
#
# Your concern (paraphrased):
#
# > “If each wavelength produces a very fine speckle pattern, then even if I add many independent patterns,
# > won’t I still see the fine structure?”
#
# Yes, you still see structure at the same correlation length — but with reduced amplitude.
# The correlation length (grain size) is set mainly by $`\mathrm{NA}_{\mathrm{illum}}`$:
#
# $$
# \Delta x_{\mathrm{speckle}} \approx \frac{\lambda}{2\thinspace\mathrm{NA}_{\mathrm{illum}}}.
# $$
#
# Underfilling the BFP reduces $`\mathrm{NA}_{\mathrm{illum}}`$ and makes speckle grains larger.
# That can be good (speckle becomes a slow multiplicative shading) or bad (big dark regions cause FNs).
#
# We do a small sweep over underfill ratios for one mid-case scenario.

# %%
# Choose a mid-case scenario to sweep
sweep_scenario = next(sc for sc in scenarios if sc.name.startswith("C2:"))

w_sweep, neff_sweep, _ = scenario_weights_and_neff(sweep_scenario, lambda0_nm=lambda0_nm, corr_width_nm=dlam_c_nm)

underfill_list = [0.05, 0.10, 0.20, 0.30]

rows = []
sweep_exc_fields: dict[float, np.ndarray] = {}
sweep_scrambler_hz_by_uf: dict[float, float] = {}
for uf in underfill_list:
    na = uf * NA_obj

    # Equal-weight spikes -> can simulate by averaging len(w) patterns.
    # Add a 10 kHz scrambler to the most plausible "lab-realistic" extremes (0.05 and 0.30 underfill).
    scr_hz = 10_000.0 if float(uf) in (0.05, 0.30) else 0.0

    I_exc, meta = simulate_excitation_speckle_field(
        n=N_grid,
        dx_um=dx_um,
        roi_um=roi_um,
        lambda_um=lambda0_nm * 1e-3,
        na_illum=na,
        exposure_s=float(exposure_us) * 1e-6,
        scrambler_hz=float(scr_hz),
        n_src=int(len(w_sweep)),
        seed=seed_vis,
    )
    I_exc = I_exc / float(I_exc[inner_mask].mean())

    # Keep fields for visual QC later
    sweep_exc_fields[float(uf)] = I_exc
    sweep_scrambler_hz_by_uf[float(uf)] = float(scr_hz)

    C_meas = speckle_contrast_inner(I_exc)
    grain_um = (lambda0_nm * 1e-3) / (2.0 * na)

    stats = run_monte_carlo_for_field(I_exc=I_exc, n_trials=10, seed0=2_000_000 + int(1e4 * uf))

    rows.append(
        {
            "underfill_ratio": uf,
            "NA_illum": na,
            "scrambler_hz": float(scr_hz),
            "N_time (scrambler)": int(meta.n_time),
            "N_eff (time*src)": int(meta.n_eff),
            "speckle_grain_um (est)": grain_um,
            "C_inner_meas": C_meas,
            "precision": stats["precision"],
            "recall": stats["recall"],
            "F1": stats["F1"],
            "FP": stats["FP"],
            "FN": stats["FN"],
        }
    )

df_sweep = pd.DataFrame(rows)
df_sweep

# %%
fig, ax = plt.subplots(figsize=(7.5, 4.0))
ax.plot(df_sweep["speckle_grain_um (est)"], df_sweep["recall"], marker="o", label="recall")
ax.plot(df_sweep["speckle_grain_um (est)"], df_sweep["precision"], marker="s", label="precision")
ax.set_xlabel("Estimated speckle grain size (µm)")
ax.set_ylabel("metric")
ax.set_title("Effect of grain size (NA_illum) at fixed spectral scenario")
ax.grid(True, alpha=0.3)
ax.legend()
plt.show()

# %% [markdown]
# ### 7.1 Visual QC: raw pixel frames across the **underfill / grain-size sweep**
#
# In section 6.2 we compared scenarios at a fixed $`\mathrm{NA}_{\mathrm{illum}}`$.
# Here we keep the spectrum fixed (one mid-case scenario) and change **grain size** via pupil underfill.
#
# We again show:
# - full-field raw pixel frames (with GT + detections), and
# - multiple pixel-annotated zoom-ins spanning dim → bright emitters.
#
# We reuse the **same deterministic emitter set** (`gt_yx_qc`, `photons_qc`) so the visual differences are driven
# primarily by the illumination field (not by different random emitter geometries).

# %%
qc_seed_uf_base = 9_876_543
for i, uf in enumerate(underfill_list):
    I_exc = sweep_exc_fields[float(uf)]
    na = float(uf) * float(NA_obj)
    grain_um = (lambda0_nm * 1e-3) / (2.0 * na)

    scr_hz = float(sweep_scrambler_hz_by_uf[float(uf)])
    scr_note = f"  scrambler={scr_hz/1000:.0f} kHz" if scr_hz > 0 else ""
    label = f"{sweep_scenario.name} | underfill={uf:.2f}  NA_illum={na:.3f}  grain≈{grain_um:.2f} µm{scr_note}"

    img_qc = simulate_sparse_emitter_frame_fixed_emitters(
        I_exc=I_exc,
        gt_yx=gt_yx_qc,
        photons=photons_qc,
        seed=qc_seed_uf_base + 100 * i,
    )
    df_det_qc = detect_spots(img_qc, params)
    match = match_detections_to_gt(df_det=df_det_qc, gt_yx=gt_yx_qc, match_r_px=3.0)

    show_full_field_qc(
        scenario_name=label,
        img_raw=img_qc,
        gt_yx=gt_yx_qc,
        df_det=df_det_qc,
        match=match,
    )

    show_emitter_montage_qc(
        scenario_name=label,
        img_raw=img_qc,
        I_exc=I_exc,
        gt_yx=gt_yx_qc,
        photons=photons_qc,
        df_det=df_det_qc,
        match=match,
        n_show=15,
        crop_radius_px=5,
    )

    show_emitter_context_montage_qc(
        scenario_name=label,
        img_raw=img_qc,
        I_exc=I_exc,
        gt_yx=gt_yx_qc,
        photons=photons_qc,
        df_det=df_det_qc,
        match=match,
        n_show=9,
        crop_radius_px=15,
    )

    show_fp_montage_qc(
        scenario_name=label,
        img_raw=img_qc,
        df_det=df_det_qc,
        match=match,
        max_fp=6,
        crop_radius_px=5,
    )



# %% [markdown]
# ## 8) The 500 µs simultaneity question: what to ask the vendor
#
# Your spectral-diversity strategy needs the spectral components to be effectively **simultaneous** within 500 µs.
# “Linewidth” on a spec sheet is often measured on an optical spectrum analyzer (OSA), which integrates over time.
# That can hide slow spectral wandering.
#
# Practical, vendor-answerable questions:
#
# 1. **Is the 2–3 nm number instantaneous?**
#    - Ask for an OSA trace with a stated sweep time / resolution bandwidth.
#    - Ask whether the source is **Fabry–Perot multi-longitudinal-mode** (many modes at once),
#      or single-frequency with drift.
#
# 2. **How many spectral peaks are present simultaneously?**
#    - If they can provide a screenshot, you can literally count peaks.
#
# 3. **What is the relative power distribution across peaks?**
#    - If one peak dominates, $N_{\mathrm{eff}}$ collapses.
#
# 4. **Is there any internal mode scrambling?**
#    - Your quote says “after homogenizing fiber”. Confirm whether the “homogenizing” effect is purely the MMF,
#      or whether there is any additional internal agitation.
#
# With these answers you can choose which of C0–C5 is closest to reality.

# %% [markdown]
# ## 9) Takeaways (how to interpret this notebook)
#
# 1. The fiber’s step-index modal dispersion makes $`\Delta\lambda_c`$ *very small*.
#    That means a 2–3 nm wide spectrum has **room** to create a large $N_\lambda$.
#
# 2. In practice, your *actual* $N_{\mathrm{eff}}$ may be limited not by the fiber, but by the **laser’s internal spectrum**:
#    - If you only have ~10–20 spikes, you are in the C1–C2 regime.
#    - To reach “display-like” low speckle (C~0.05–0.1) you likely need tens to >100 effective components.
#    - A continuous-ish band is an upper bound (C5).
#
# 3. Underfilling the BFP changes **grain size**, not **contrast** (for a fixed $N_{\mathrm{eff}}$).
#    The “best” underfill depends on whether your limiting failure mode is FP (PSF-scale structure)
#    or FN (large-scale shading).
#
# 4. The most important next step is to turn the vendor ambiguity into a concrete measurement:
#    request a spectrum screenshot and ask the “simultaneous peaks?” question.

