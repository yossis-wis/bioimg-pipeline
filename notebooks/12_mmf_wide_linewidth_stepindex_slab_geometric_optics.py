# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # MMF + wide linewidth (step-index): slab + geometrical-optics intuition + toy mode / wavelength sums
#
# This notebook is **only** about the **multimode, wide-linewidth, step-index** approach.
#
# We will deliberately avoid Fourier-optics machinery and work in the picture that even
# Fourier-optics fans tend to accept for *large-core multimode guides*:
#
# - **Rays** are a good intuition tool.
# - **Modes** can be treated as a large set of coherent contributors.
# - **Speckle** is a consequence of **interference** between those contributors.
# - **Spectral width** + **scrambling** reduces speckle by *incoherent averaging*.
#
# The user-facing goals for this notebook are:
#
# 1. Use a **slab** (planar) guide as the simplest geometry for the *core argument*.
# 2. Work through a **toy example**:
#    - 2 modes at 1 wavelength
#    - many modes at 1 wavelength
#    - 2+ wavelengths (spectral diversity)
# 3. Address the confusion:
#    - “I can count modes, but how do I know their **weights**?”
#    - “I have a linewidth, but what’s the **power per wavelength**?”
#
# Throughout, we keep the math as direct as possible and generate **lots of figures**.
#
# Notes / scope:
#
# - We assume **step-index** fiber (we skip GRIN on purpose).
# - We are not solving the exact vector eigenmodes of a step-index cylinder.
#   For 2D “mode pictures” we use the repo’s **disk-Bessel surrogate basis**.
#   That surrogate is good for building intuition about interference + averaging.

# %% [markdown]
# ## 0) Imports + repo plumbing

# %%
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


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

# Make plots a bit larger by default.
plt.rcParams.update({"figure.figsize": (8.0, 5.0), "figure.dpi": 120})

# If we are in a notebook, prefer inline backend.
if "ipykernel" in sys.modules:
    try:
        if matplotlib.get_backend().lower() == "agg":
            matplotlib.use("module://matplotlib_inline.backend_inline", force=True)
    except Exception:
        pass

from src.fiber_modes import (  # noqa: E402
    intensity_from_field,
    make_core_grid,
    precompute_mode_stack,
    random_complex_coeffs,
    speckle_contrast,
    superpose_modes,
)
from src.mmf_fiber_speckle import (  # noqa: E402
    MultimodeFiber,
    approx_num_guided_modes_step_index,
    max_guided_meridional_ray_angle_rad,
    optical_path_spread_geometric_m,
    speckle_spectral_corr_width_nm,
    v_number,
)
from src.speckle_diversity_models import (  # noqa: E402
    DiversityBudget,
    estimate_n_eff,
    speckle_contrast_from_n_eff,
)
from src.speckle_weighting import (  # noqa: E402
    effective_n_from_weights,
    gaussian_spectrum_bins,
    speckle_contrast_from_weights,
    uniform_top_hat_spectrum_bins,
)
from src.temporal_coherence import coherence_length_m_from_linewidth_nm  # noqa: E402


# %% [markdown]
# ## 1) The core geometric-optics argument in one line
#
# For a step-index multimode guide, treat propagation as rays at angles $\theta$ w.r.t. the axis.
#
# A ray at angle $\theta$ has physical path length $L/\cos\theta$ over an axial length $L$.
# The **vacuum-equivalent optical path** (the thing that matters for phase) is:
#
# $$
# \mathrm{OPL}(\theta) = n\,\frac{L}{\cos\theta}.
# $$
#
# The “axial” ray has $\mathrm{OPL}(0) = nL$.
# The **spread** across guided rays is therefore
#
# $$
# \Delta\mathrm{OPL} \;\approx\; nL\left(\frac{1}{\cos\theta_{\max}} - 1\right)
# \;\approx\; \frac{\mathrm{NA}^2}{2n}\,L.
# $$
#
# A simple speckle decorrelation condition is
#
# $$
# \Delta k\,\Delta\mathrm{OPL}\sim 2\pi \quad\Rightarrow\quad
# \Delta\lambda_c\sim \frac{\lambda_0^2}{\Delta\mathrm{OPL}}.
# $$
#
# where $\Delta\lambda_c$ is the **spectral correlation width** of the speckle pattern.
#
# If your source spans many *independent* spectral bins,
#
# $$
# N_{\lambda}\approx \frac{\Delta\lambda_{\mathrm{src}}}{\Delta\lambda_c},
# $$
#
# then the speckle contrast drops roughly as
#
# $$
# C \sim \frac{1}{\sqrt{N_{\lambda}}}.
# $$

# %% [markdown]
# ### A concrete “real” fiber spec

# %%
lambda0_nm = 640.0

fiber_real = MultimodeFiber(
    core_diameter_um=400.0,
    na=0.22,
    length_m=3.0,
    n_core=1.46,
    modal_delay_scale=1.0,  # step-index assumption
)

theta_max_rad = max_guided_meridional_ray_angle_rad(na=fiber_real.na, n_core=fiber_real.n_core)
delta_opl_real_m = optical_path_spread_geometric_m(fiber_real)
dlam_c_real_nm = speckle_spectral_corr_width_nm(lambda0_nm=lambda0_nm, delta_opl_m=delta_opl_real_m)

print("Real-fiber baseline")
print(f"  lambda0 = {lambda0_nm:.1f} nm")
print(f"  core_diameter = {fiber_real.core_diameter_um:.0f} um")
print(f"  NA = {fiber_real.na:.3f}")
print(f"  length = {fiber_real.length_m:.2f} m")
print(f"  n_core = {fiber_real.n_core:.3f}")
print(f"  theta_max (inside core) = {theta_max_rad*180/math.pi:.2f} deg")
print(f"  Delta OPL ~ {delta_opl_real_m*1e3:.1f} mm")
print(f"  Delta lambda_c ~ {dlam_c_real_nm:.4f} nm")


# %% [markdown]
# ### Sanity-check: coherence length vs path spread
#
# A convenient “reality check” is to compare the **coherence length** implied by the linewidth
# to the fiber’s **intermodal path spread** $\Delta\mathrm{OPL}$.
#
# If $L_c \ll \Delta\mathrm{OPL}$, then different modal delays cannot stay mutually coherent
# across the whole spectrum, so spectral averaging becomes very effective.

# %%
for fwhm_nm in [0.1, 2.0, 20.0]:
    lc_m = coherence_length_m_from_linewidth_nm(lambda0_nm=lambda0_nm, fwhm_nm=fwhm_nm, profile="gaussian")
    print(f"FWHM={fwhm_nm:>4.1f} nm -> coherence length Lc ~ {lc_m*1e3:>6.2f} mm")

print("")
print(f"Delta OPL (real fiber) ~ {delta_opl_real_m*1e3:.1f} mm")


# %% [markdown]
# ## 2) A picture: the slab waveguide ray model
#
# We draw a **slab** (planar) guide with thickness equal to the fiber core diameter.
# We plot:
#
# - an **axial** ray
# - a **highest-angle guided** ray ($\theta\approx\theta_{\max}$)
#
# over a short segment of the fiber.

# %%

def ray_polyline_in_slab(
    *,
    a: float,
    theta_rad: float,
    z_max: float,
    x0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear ray in a slab x in [-a, +a], z in [0, z_max].

    Parameters
    ----------
    a:
        Half-thickness (same units as x and z).
    theta_rad:
        Ray angle w.r.t. z axis.
    z_max:
        Propagation distance.
    x0:
        Initial transverse position.

    Returns
    -------
    (z, x)
        Polyline vertices (not a dense sample).
    """

    if a <= 0:
        raise ValueError("a must be > 0")

    s = math.tan(float(theta_rad))
    if abs(s) < 1e-12:
        return np.asarray([0.0, z_max]), np.asarray([float(x0), float(x0)])

    z_pts = [0.0]
    x_pts = [float(x0)]

    z = 0.0
    x = float(x0)
    slope = s

    # Safety limit on bounces.
    for _ in range(10_000):
        if z >= z_max:
            break

        # Which boundary do we hit next?
        if slope > 0:
            x_hit = +a
        else:
            x_hit = -a

        dz = (x_hit - x) / slope

        if dz <= 0:
            # Numerical edge case: we're on the boundary.
            slope *= -1
            continue

        if z + dz >= z_max:
            # We stop before the next bounce.
            dz2 = z_max - z
            z = z_max
            x = x + slope * dz2
            z_pts.append(z)
            x_pts.append(x)
            break

        # Bounce.
        z = z + dz
        x = x_hit
        z_pts.append(z)
        x_pts.append(x)
        slope *= -1

    return np.asarray(z_pts), np.asarray(x_pts)


core_diam_mm = fiber_real.core_diameter_um * 1e-3
half_mm = 0.5 * core_diam_mm

# Choose a short z segment that shows a handful of bounces.
z_seg_mm = 20.0

z_ax, x_ax = ray_polyline_in_slab(a=half_mm, theta_rad=0.0, z_max=z_seg_mm)
z_hi, x_hi = ray_polyline_in_slab(a=half_mm, theta_rad=theta_max_rad, z_max=z_seg_mm)

plt.figure()
plt.plot(z_ax, x_ax, linewidth=2, label="axial")
plt.plot(z_hi, x_hi, linewidth=2, label=r"$\\theta\\approx\\theta_{max}$")
plt.hlines([+half_mm, -half_mm], xmin=0, xmax=z_seg_mm, linestyles="--")
plt.xlabel("z (mm)")
plt.ylabel("x (mm)")
plt.title("Slab ray model (segment view)")
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Scale diagram: core diameter vs fiber length

# %%
length_m = fiber_real.length_m
core_diam_m = fiber_real.core_diameter_um * 1e-6

plt.figure()
plt.semilogx([core_diam_m], [1], "o", label=f"core diameter = {core_diam_m:.1e} m")
plt.semilogx([length_m], [1], "o", label=f"fiber length = {length_m:.1e} m")
plt.yticks([])
plt.xlabel("length scale (m, log)")
plt.title("A 3 m fiber is ~10^4× longer than a 400 µm core")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 3) Mode counting is easy; mode *weights* are the subtle part
#
# **Counting modes** (step-index, weak guidance, large V):
#
# $$
# V = \frac{2\pi a\,\mathrm{NA}}{\lambda},\qquad M\approx \frac{V^2}{2}.
# $$
#
# For a 400 µm core at 640 nm, $V\gg 1$ and the total guided-mode count is enormous.
#
# **But:** the speckle (and how well averaging works) depends on the *effective* number of
# contributors, which depends on the **weights**.
#
# A useful identity (same as an inverse-participation ratio): if you incoherently average
# independent patterns with intensity weights $w_i$,
#
# $$
# N_{\mathrm{eff}} = \frac{(\sum_i w_i)^2}{\sum_i w_i^2}.
# $$
#
# If one weight dominates, $N_{\mathrm{eff}}\to 1$ even if there are many available modes.

# %%
a_um = 0.5 * fiber_real.core_diameter_um
v = v_number(core_radius_um=a_um, na=fiber_real.na, lambda_um=lambda0_nm * 1e-3)
modes_est = approx_num_guided_modes_step_index(v)

print(f"V-number (real fiber @ {lambda0_nm:.0f} nm): V ~ {v:.1f}")
print(f"Approx guided modes M ~ V^2/2 ~ {modes_est:,d}")


# %% [markdown]
# ## 4) Toy example 1: 2 slab “modes” at one wavelength
#
# We use a deliberately simple (not physically exact) slab basis to show what the *sum of modes*
# looks like.
#
# Define two transverse patterns $u_1(x), u_2(x)$ across the core, and form
#
# $$
# E(x) = a_1 u_1(x) + a_2 u_2(x)e^{i\Delta\phi}.
# $$
#
# Changing $\Delta\phi$ changes the interference and therefore the intensity $I(x)=|E(x)|^2$.

# %%
# Toy slab coordinate (units arbitrary)
a = 1.0
x = np.linspace(-a, a, 2000)

# Two simple transverse shapes.
# (Think: two different guided transverse standing-wave patterns.)
u1 = np.cos(0.5 * math.pi * x / a)
u2 = np.cos(1.5 * math.pi * x / a)

# Normalize energies over the core.
def l2_norm(v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(v * v)))

u1 = u1 / l2_norm(u1)
u2 = u2 / l2_norm(u2)

plt.figure()
plt.plot(x, u1, label="u1(x)")
plt.plot(x, u2, label="u2(x)")
plt.xlabel("x (core coordinate)")
plt.title("Two toy slab 'modes' (transverse patterns)")
plt.legend()
plt.tight_layout()
plt.show()

# Now show intensity vs relative phase.
phases = [0.0, 0.5 * math.pi, math.pi]
a1, a2 = 1.0, 1.0

plt.figure()
for dphi in phases:
    E = a1 * u1 + a2 * u2 * np.exp(1j * dphi)
    I = (E.real * E.real + E.imag * E.imag)
    plt.plot(x, I / float(np.mean(I)), label=f"Δφ={dphi/math.pi:.1f}π")
plt.xlabel("x")
plt.ylabel("normalized intensity")
plt.title("2-mode interference at a single wavelength")
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 5) Toy example 2: many modes → speckle, and why weights matter
#
# For a *multimode* guide, we have a large set of transverse patterns $u_m(x,y)$ and coefficients
# $c_m$.
#
# $$
# U(x,y) = \sum_m c_m u_m(x,y),\qquad I(x,y)=|U(x,y)|^2.
# $$
#
# If the phases of $c_m$ are effectively random, the intensity is granular (speckle-like).
#
# Here we use the repo’s **disk-Bessel surrogate basis** for $u_m(x,y)$.
# This is not a physical LP-mode solver, but it is excellent for intuition.

# %%
# Build a disk grid representing the core cross-section.
# (We pick small grid sizes so this notebook stays fast.)
core_radius_um = 200.0
n_grid = 160
x_um, y_um, mask, dx_um = make_core_grid(n=n_grid, core_radius_um=core_radius_um)

# Precompute a stack of surrogate modes, then take the first N.
from src.fiber_modes import disk_mode_indices  # noqa: E402

modes = disk_mode_indices(max_l=10, max_m=10, include_sin=True)

n_modes = 80
modes = modes[:n_modes]
mode_stack = precompute_mode_stack(modes, x_um=x_um, y_um=y_um, core_radius_um=core_radius_um, mask=mask)

print(f"Using n_modes={n_modes} surrogate modes on a {n_grid}x{n_grid} grid")


# %%
# Three weight scenarios (intensity weights). We will convert to amplitude weights via sqrt.

wI_uniform = np.ones(n_modes, dtype=float)

# A "peaked" distribution: most power in first few modes.
# (This mimics severe underfill / poor mixing.)
k = np.arange(n_modes, dtype=float)
wI_peaked = np.exp(-k / 8.0)

# A "few-mode" distribution: only first 10 modes have power.
wI_few = np.zeros(n_modes, dtype=float)
wI_few[:10] = 1.0

# Normalize all to sum=1 so N_eff comparisons are fair.
def normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    return w / float(w.sum())

wI_uniform = normalize_weights(wI_uniform)
wI_peaked = normalize_weights(wI_peaked)
wI_few = normalize_weights(wI_few)

scenarios = {
    "uniform": wI_uniform,
    "peaked": wI_peaked,
    "few": wI_few,
}

for name, wI in scenarios.items():
    n_eff = effective_n_from_weights(wI)
    print(f"{name:>7s}: N_eff(mode weights) = {n_eff:6.1f}")


# %%
# Visualize the weight distributions.

plt.figure()
for name, wI in scenarios.items():
    plt.plot(wI, label=f"{name} (N_eff={effective_n_from_weights(wI):.1f})")
plt.xlabel("mode index (sorted by spatial frequency)")
plt.ylabel("intensity weight")
plt.title("Toy mode-weight distributions")
plt.legend()
plt.tight_layout()
plt.show()


# %%
# Generate one speckle pattern per scenario at a single wavelength.

rng = np.random.default_rng(0)

figs = []
for name, wI in scenarios.items():
    wA = np.sqrt(wI)  # amplitude weights
    coeffs = random_complex_coeffs(n_modes, seed=int(rng.integers(0, 2**31 - 1)), weights=wA)
    U = superpose_modes(mode_stack, coeffs)
    I = intensity_from_field(U)
    C = speckle_contrast(I, mask)

    plt.figure()
    plt.imshow(I, origin="lower")
    plt.title(f"{name}: single-λ intensity (C={C:.3f}, N_eff={effective_n_from_weights(wI):.1f})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# %% [markdown]
# **Takeaway:** “many modes exist” does not automatically mean “many modes contribute.”
#
# - If coupling/mixing makes weights fairly even, $N_{\mathrm{eff}}$ is large and speckle is fully developed.
# - If the launch severely underfills NA (or the fiber is static + clean + straight), weights can be dominated
#   by a small subset of modes, and the effective mode count is small.
#
# In practice, for your illumination concept, you have several knobs to avoid the pathological case:
#
# - Fill the fiber NA (diffuser / appropriate launch optics).
# - Use a **scrambler / mode mixer** (bends, agitation) so power redistributes.
# - Avoid “perfect” launch conditions that trap you in a small modal subset.

# %% [markdown]
# ## 6) Toy example 3: wavelength dependence and correlation width
#
# For a fixed coupling state, we now include **wavelength-dependent modal phase**.
#
# A minimal model that captures the correct scaling is:
#
# $$
# c_m(\lambda) = a_m\,e^{i\phi_m}\,e^{i2\pi\,\Delta\mathrm{OPL}_m/\lambda}
# $$
#
# where $\Delta\mathrm{OPL}_m$ is a mode-dependent optical-path offset.
#
# Then the speckle pattern changes with $\lambda$, and the correlation width is
# roughly $\Delta\lambda_c\sim \lambda^2/\Delta\mathrm{OPL}$.
#
# We will do this twice:
#
# 1. A **toy-short fiber** (so $\Delta\lambda_c$ is ~0.1–0.5 nm and we can visualize it).
# 2. The **real** 3 m fiber (where $\Delta\lambda_c$ is ~0.01 nm and changes are extremely fast).

# %%
# Build a short "toy" fiber so correlation widths are easy to visualize.
# Choose length so dlam_c is O(0.2 nm) at 640 nm.

fiber_toy = MultimodeFiber(
    core_diameter_um=fiber_real.core_diameter_um,
    na=fiber_real.na,
    length_m=0.12,
    n_core=fiber_real.n_core,
    modal_delay_scale=1.0,
)

delta_opl_toy_m = optical_path_spread_geometric_m(fiber_toy)
dlam_c_toy_nm = speckle_spectral_corr_width_nm(lambda0_nm=lambda0_nm, delta_opl_m=delta_opl_toy_m)

print("Toy fiber")
print(f"  length = {fiber_toy.length_m:.3f} m")
print(f"  Delta OPL ~ {delta_opl_toy_m*1e3:.3f} mm")
print(f"  Delta lambda_c ~ {dlam_c_toy_nm:.3f} nm")


# %%
# Assign each surrogate mode a delay/path offset by sampling a ray angle.
#
# For small angles:
#   DeltaOPL(theta) = n L (1/cos(theta) - 1)
# and theta is bounded by theta_max.


def sample_delta_opl_per_mode(
    *,
    fiber: MultimodeFiber,
    n_modes: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))

    theta_max = max_guided_meridional_ray_angle_rad(na=fiber.na, n_core=fiber.n_core)

    # Sample an "angle fraction" s in [0,1].
    # We map s -> theta using sqrt so we populate the high-angle end a bit more.
    s = rng.uniform(0.0, 1.0, size=int(n_modes))
    theta = theta_max * np.sqrt(s)

    # Optical-path offset relative to axial.
    delta_opl = fiber.n_core * fiber.length_m * (1.0 / np.cos(theta) - 1.0)

    return delta_opl.astype(np.float64, copy=False)


delta_opl_per_mode_toy = sample_delta_opl_per_mode(fiber=fiber_toy, n_modes=n_modes, seed=1)
delta_opl_per_mode_real = sample_delta_opl_per_mode(fiber=fiber_real, n_modes=n_modes, seed=1)


# %%
# Wavelength-dependent intensity generator.

base_wI = wI_uniform
base_wA = np.sqrt(base_wI)

phi0 = np.random.default_rng(123).uniform(0.0, 2.0 * math.pi, size=n_modes)


def intensity_for_lambda_nm(
    *,
    lambda_nm: float,
    mode_stack: np.ndarray,
    base_phase_rad: np.ndarray,
    delta_opl_m: np.ndarray,
    amp_weights: np.ndarray,
) -> np.ndarray:
    lam_m = float(lambda_nm) * 1e-9
    phase = base_phase_rad + (2.0 * math.pi) * (delta_opl_m / lam_m)
    coeffs = amp_weights * np.exp(1j * phase)
    U = superpose_modes(mode_stack, coeffs)
    return intensity_from_field(U)


# %%
# Correlation vs wavelength offset (toy fiber).

I0 = intensity_for_lambda_nm(
    lambda_nm=lambda0_nm,
    mode_stack=mode_stack,
    base_phase_rad=phi0,
    delta_opl_m=delta_opl_per_mode_toy,
    amp_weights=base_wA,
)

# Choose offsets up to a few nm.
delta_lams = np.linspace(0.0, 2.0, 81)

corrs = []
flat0 = I0[mask].ravel()
flat0 = (flat0 - float(np.mean(flat0))) / float(np.std(flat0))

for dlam in delta_lams:
    I1 = intensity_for_lambda_nm(
        lambda_nm=lambda0_nm + float(dlam),
        mode_stack=mode_stack,
        base_phase_rad=phi0,
        delta_opl_m=delta_opl_per_mode_toy,
        amp_weights=base_wA,
    )
    flat1 = I1[mask].ravel()
    flat1 = (flat1 - float(np.mean(flat1))) / float(np.std(flat1))
    corr = float(np.mean(flat0 * flat1))
    corrs.append(corr)

corrs = np.asarray(corrs)

plt.figure()
plt.plot(delta_lams, corrs, linewidth=2)
plt.axvline(dlam_c_toy_nm, linestyle="--", label=f"Δλc ~ {dlam_c_toy_nm:.2f} nm")
plt.xlabel("Δλ (nm)")
plt.ylabel("pattern correlation (toy model)")
plt.title("Speckle pattern decorrelation vs wavelength (toy fiber)")
plt.ylim(-0.2, 1.05)
plt.legend()
plt.tight_layout()
plt.show()


# %%
# Same idea for the real 3 m fiber: correlation collapses on ~0.01 nm scales.

I0r = intensity_for_lambda_nm(
    lambda_nm=lambda0_nm,
    mode_stack=mode_stack,
    base_phase_rad=phi0,
    delta_opl_m=delta_opl_per_mode_real,
    amp_weights=base_wA,
)

# Offsets up to 0.05 nm.
delta_lams_r = np.linspace(0.0, 0.05, 101)

corrs_r = []
flat0r = I0r[mask].ravel()
flat0r = (flat0r - float(np.mean(flat0r))) / float(np.std(flat0r))

for dlam in delta_lams_r:
    I1r = intensity_for_lambda_nm(
        lambda_nm=lambda0_nm + float(dlam),
        mode_stack=mode_stack,
        base_phase_rad=phi0,
        delta_opl_m=delta_opl_per_mode_real,
        amp_weights=base_wA,
    )
    flat1r = I1r[mask].ravel()
    flat1r = (flat1r - float(np.mean(flat1r))) / float(np.std(flat1r))
    corrs_r.append(float(np.mean(flat0r * flat1r)))

corrs_r = np.asarray(corrs_r)

plt.figure()
plt.plot(delta_lams_r, corrs_r, linewidth=2)
plt.axvline(dlam_c_real_nm, linestyle="--", label=f"Δλc ~ {dlam_c_real_nm:.4f} nm")
plt.xlabel("Δλ (nm)")
plt.ylabel("pattern correlation (toy model)")
plt.title("Real-fiber decorrelation is extremely fast in λ")
plt.ylim(-0.2, 1.05)
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 7) Spectral diversity: 2+ wavelengths, and why you do *not* need 0.001 nm bookkeeping
#
# You asked:
#
# - “If I have a 2 nm or 20 nm spectral span, do I need to know the power at each 0.001 nm?”
#
# The practical answer is: **no**.
#
# What matters for speckle averaging is the power per **correlation bin** of width $\Delta\lambda_c$.
# Anything finer than $\Delta\lambda_c$ is partially correlated and does not buy you independent samples.
#
# The right discretization is:
#
# - pick bins of width $\Delta\lambda_c$ (or slightly larger)
# - integrate the spectrum in each bin to get weights $w_i$
# - compute the effective count $N_{\mathrm{eff}} = (\sum w_i)^2/(\sum w_i^2)$

# %%
# Visual spectral-bin toy: take the *toy* fiber so we only need ~10 bins for a 2 nm spectrum.

corr_width_nm = dlam_c_toy_nm

bins_gauss_2 = gaussian_spectrum_bins(lambda0_nm=lambda0_nm, fwhm_nm=2.0, corr_width_nm=corr_width_nm)
bins_gauss_20 = gaussian_spectrum_bins(lambda0_nm=lambda0_nm, fwhm_nm=20.0, corr_width_nm=corr_width_nm)

print(f"Toy fiber Δλc ~ {corr_width_nm:.3f} nm")
print(f"2 nm Gaussian -> n_bins={bins_gauss_2.n_bins}, N_eff={bins_gauss_2.n_eff:.1f}")
print(f"20 nm Gaussian -> n_bins={bins_gauss_20.n_bins}, N_eff={bins_gauss_20.n_eff:.1f}")


# %%
plt.figure()
plt.stem(
    bins_gauss_2.centers_nm - lambda0_nm,
    bins_gauss_2.weights,
    basefmt=" ",
)
plt.xlabel("λ - λ0 (nm)")
plt.ylabel("bin weight")
plt.title("Gaussian spectrum binned into Δλc-sized correlation bins (2 nm FWHM, toy fiber)")
plt.tight_layout()
plt.show()


# %%
# A direct 2-wavelength example: average two speckle patterns.
# Pick two wavelengths separated by a few correlation widths -> nearly independent.

lam_a = lambda0_nm - 0.5
lam_b = lambda0_nm + 0.5

Ia = intensity_for_lambda_nm(
    lambda_nm=lam_a,
    mode_stack=mode_stack,
    base_phase_rad=phi0,
    delta_opl_m=delta_opl_per_mode_toy,
    amp_weights=base_wA,
)
Ib = intensity_for_lambda_nm(
    lambda_nm=lam_b,
    mode_stack=mode_stack,
    base_phase_rad=phi0,
    delta_opl_m=delta_opl_per_mode_toy,
    amp_weights=base_wA,
)

Iavg2 = 0.5 * (Ia + Ib)

Ca = speckle_contrast(Ia, mask)
Cb = speckle_contrast(Ib, mask)
Cavg2 = speckle_contrast(Iavg2, mask)

plt.figure()
plt.imshow(Ia, origin="lower")
plt.title(f"λ={lam_a:.1f} nm (C={Ca:.3f})")
plt.axis("off")
plt.tight_layout()
plt.show()

plt.figure()
plt.imshow(Ib, origin="lower")
plt.title(f"λ={lam_b:.1f} nm (C={Cb:.3f})")
plt.axis("off")
plt.tight_layout()
plt.show()

plt.figure()
plt.imshow(Iavg2, origin="lower")
plt.title(f"Average of 2 wavelengths (C={Cavg2:.3f})")
plt.axis("off")
plt.tight_layout()
plt.show()


# %%
# Now do a full spectral average using the binned Gaussian weights (toy fiber).


def weighted_spectral_average(
    *,
    bins_centers_nm: np.ndarray,
    bins_weights: np.ndarray,
    delta_opl_m: np.ndarray,
) -> np.ndarray:
    I_acc = np.zeros((n_grid, n_grid), dtype=np.float64)
    for lam, w in zip(bins_centers_nm, bins_weights, strict=True):
        I = intensity_for_lambda_nm(
            lambda_nm=float(lam),
            mode_stack=mode_stack,
            base_phase_rad=phi0,
            delta_opl_m=delta_opl_m,
            amp_weights=base_wA,
        )
        I_acc += float(w) * I
    return I_acc


Iavg_2nm = weighted_spectral_average(
    bins_centers_nm=bins_gauss_2.centers_nm,
    bins_weights=bins_gauss_2.weights,
    delta_opl_m=delta_opl_per_mode_toy,
)
Iavg_20nm = weighted_spectral_average(
    bins_centers_nm=bins_gauss_20.centers_nm,
    bins_weights=bins_gauss_20.weights,
    delta_opl_m=delta_opl_per_mode_toy,
)

C0 = speckle_contrast(I0, mask)
C2 = speckle_contrast(Iavg_2nm, mask)
C20 = speckle_contrast(Iavg_20nm, mask)

print(f"Single λ contrast (toy model): C0 ~ {C0:.3f}")
print(f"2 nm averaged (toy fiber):      C  ~ {C2:.3f}    (N_eff~{bins_gauss_2.n_eff:.1f})")
print(f"20 nm averaged (toy fiber):     C  ~ {C20:.3f}   (N_eff~{bins_gauss_20.n_eff:.1f})")


# %%
plt.figure()
plt.imshow(Iavg_2nm, origin="lower")
plt.title(f"Toy fiber: Gaussian 2 nm spectral average (C={C2:.3f})")
plt.axis("off")
plt.tight_layout()
plt.show()

plt.figure()
plt.imshow(Iavg_20nm, origin="lower")
plt.title(f"Toy fiber: Gaussian 20 nm spectral average (C={C20:.3f})")
plt.axis("off")
plt.tight_layout()
plt.show()


# %% [markdown]
# ### The same calculation for the real 3 m fiber (numbers only)
#
# For the real fiber, $\Delta\lambda_c$ is so small that even 2 nm spans hundreds of correlation bins.
# It’s not useful to brute-force simulate every bin in a notebook.
#
# But the bookkeeping is easy:
#
# - compute $\Delta\lambda_c$
# - bin the spectrum into $\Delta\lambda_c$ bins
# - compute the effective count $N_{\lambda,\mathrm{eff}}$
#
# Then your spectral-averaging-only contrast scale is $C\approx 1/\sqrt{N_{\lambda,\mathrm{eff}}}$.

# %%
# Real fiber spectral bins.
bins_real_2 = gaussian_spectrum_bins(lambda0_nm=lambda0_nm, fwhm_nm=2.0, corr_width_nm=dlam_c_real_nm)
bins_real_20 = gaussian_spectrum_bins(lambda0_nm=lambda0_nm, fwhm_nm=20.0, corr_width_nm=dlam_c_real_nm)

print(f"Real fiber Δλc ~ {dlam_c_real_nm:.4f} nm")
print(f"2 nm Gaussian -> N_eff_lambda ~ {bins_real_2.n_eff:.0f} -> C~{1/math.sqrt(bins_real_2.n_eff):.3f}")
print(f"20 nm Gaussian -> N_eff_lambda ~ {bins_real_20.n_eff:.0f} -> C~{1/math.sqrt(bins_real_20.n_eff):.3f}")


# %% [markdown]
# ## 8) Putting it together: time + spectral diversity (and where weights enter)
#
# A conservative diversity budget (for short exposures) is:
#
# $$
# N_{\mathrm{eff}} \approx N_t\,N_{\lambda}\,N_{\mathrm{pol}}\,N_{\mathrm{angle}}.
# $$
#
# For this notebook we keep only the two knobs you explicitly asked about:
#
# - **time scrambling**: $N_t \approx f_{\mathrm{scr}}\,T_{\mathrm{exp}}$
# - **spectral width**: $N_{\lambda}$ from $\Delta\lambda_{\mathrm{src}}/\Delta\lambda_c$
#
# Then $C \approx 1/\sqrt{N_{\mathrm{eff}}}$.
#
# Where the “distribution” confusion comes in:
#
# - If spectral power is very non-uniform across bins, use the weights $w_i$ and
#   $N_{\lambda,\mathrm{eff}} = (\sum w_i)^2/(\sum w_i^2)$.
# - If modal power is very non-uniform, the **speckle realization per wavelength** may itself
#   be less “fully developed” (and in the extreme, a single mode gives no speckle).
#   But for most real MMF illumination launches, *many* modes are excited.

# %%
# A concrete exposure/scrambler example.
exposure_us = 500.0
exposure_s = exposure_us * 1e-6
scrambler_hz = 10_000.0

# Use the real-fiber spectral effective counts we computed above.

for fwhm_nm, bins in [(2.0, bins_real_2), (20.0, bins_real_20)]:
    diversity = DiversityBudget(n_lambda=int(round(bins.n_eff)), n_pol=1, n_angle=1)
    n_eff = estimate_n_eff(
        exposure_s=exposure_s,
        scrambler_hz=scrambler_hz,
        diversity=diversity,
        successive_pattern_correlation=0.0,
    )
    c = speckle_contrast_from_n_eff(n_eff)
    print(
        f"FWHM={fwhm_nm:>4.0f} nm | Nt~{scrambler_hz*exposure_s:.1f} | Nλ_eff~{bins.n_eff:>7.0f}"
        f" -> N_eff~{n_eff:>8.0f} -> C~{c:.4f}"
    )


# %% [markdown]
# ## 9) Summary answers to the two “distribution” confusions
#
# ### (A) “How do I know each *mode’s* contribution to speckle at a given wavelength?”
#
# - The number of guided modes (from $V$) is an **available state count**, not a guarantee.
# - The **launched field** sets the initial mode weights.
# - Imperfections + bends + a scrambler couple modes and tend to **redistribute** power.
#
# What you can do in practice:
#
# - Treat mode weights as unknown, but **bound** the bad case via $N_{\mathrm{eff}}$.
# - Design the launch so the bad case is unlikely (fill NA + scrambler).
# - If needed, *measure* near-field statistics vs fiber agitation to confirm you’re not stuck
#   in a few-mode regime.
#
# ### (B) “How do I know the power at each exact wavelength within a 2 nm (or 20 nm) span?”
#
# - You don’t need 0.001 nm resolution unless $\Delta\lambda_c$ is that small.
# - The correct granularity is the **speckle correlation width** $\Delta\lambda_c$.
# - Bin your measured spectrum into $\Delta\lambda_c$ chunks and compute $N_{\lambda,\mathrm{eff}}$.
#
# This notebook used two models for that:
#
# - top-hat spectrum (uniform)
# - gaussian spectrum (typical “linewidth” spec)
#
# Both support the design intuition: for step-index fibers of meter-ish lengths,
# $\Delta\lambda_c$ is tiny in the visible, so even a few nm provides many independent
# spectral speckle realizations.
