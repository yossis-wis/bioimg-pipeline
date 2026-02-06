r"""Educational fiber-mode and multimode-speckle primitives.

This module exists to support **visual intuition building** in the repo's MMF
illumination discussions.

Important scope note
--------------------
This is **not** a full vector, step-index / graded-index eigenmode solver.

Instead, we use a mathematically clean surrogate:

- An orthonormal basis on a circular disk (the fiber core) built from Bessel
  functions :math:`J_\ell` with zeros :math:`j_{\ell m}`.
- Real angular variants ``cos(ℓφ)`` and ``sin(ℓφ)`` (degenerate for ℓ>0).

This is good enough to:

- show what "modes" look like spatially,
- show how interference between many modes produces a granular intensity pattern,
- demonstrate how tiny phase perturbations can dramatically change the pattern
  (modal noise / speckle drift),
- illustrate why averaging (time/spectral/polarization) reduces speckle contrast.

If you later want *quantitative* modal dispersion or exact LP mode fields, keep
using :mod:`src.mmf_fiber_speckle` for dispersion-scale estimates and treat this
file as a visualization helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy import special


AngularType = Literal["cos", "sin"]


@dataclass(frozen=True)
class DiskModeIndex:
    """Index for a disk-Bessel mode used as a fiber-core surrogate.

    Parameters
    ----------
    l:
        Azimuthal order (>=0).
    m:
        Radial order (>=1). Uses the m-th zero of J_l.
    angular:
        Angular variant. For l=0, only ``cos`` is meaningful.
    alpha:
        Bessel zero j_{l,m}. This controls the radial oscillation scale.
    """

    l: int
    m: int
    angular: AngularType
    alpha: float

    @property
    def label(self) -> str:
        if self.l == 0:
            return f"l={self.l}, m={self.m}"
        return f"l={self.l}, m={self.m} ({self.angular})"


def make_core_grid(
    *,
    n: int,
    core_radius_um: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return a square cartesian grid spanning the fiber core.

    Returns
    -------
    (x_um, y_um, mask, dx_um)

    Where:
    - x_um, y_um: (n, n) coordinate arrays (µm)
    - mask: boolean disk mask for r <= core_radius_um
    - dx_um: grid spacing (µm)
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    if core_radius_um <= 0:
        raise ValueError("core_radius_um must be > 0")

    coords = np.linspace(-core_radius_um, core_radius_um, int(n), dtype=np.float64)
    dx_um = float(coords[1] - coords[0]) if n > 1 else float(2.0 * core_radius_um)
    x_um, y_um = np.meshgrid(coords, coords, indexing="xy")
    r_um = np.sqrt(x_um * x_um + y_um * y_um)
    mask = r_um <= float(core_radius_um)
    return x_um, y_um, mask, dx_um


def disk_mode_indices(
    *,
    max_l: int,
    max_m: int,
    include_sin: bool = True,
) -> list[DiskModeIndex]:
    """Generate a sorted list of disk-Bessel mode indices.

    Sorting is by increasing Bessel zero ``alpha`` (roughly "increasing spatial frequency").

    Notes
    -----
    For l>0 we include both cos and sin angular variants by default to reflect the
    common degeneracy of fiber modes.
    """
    if max_l < 0:
        raise ValueError("max_l must be >= 0")
    if max_m <= 0:
        raise ValueError("max_m must be > 0")

    modes: list[DiskModeIndex] = []
    for l in range(int(max_l) + 1):
        zeros = special.jn_zeros(l, int(max_m))
        for m, alpha in enumerate(zeros, start=1):
            modes.append(DiskModeIndex(l=l, m=m, angular="cos", alpha=float(alpha)))
            if include_sin and l > 0:
                modes.append(DiskModeIndex(l=l, m=m, angular="sin", alpha=float(alpha)))

    # stable sort; cos and sin stay adjacent
    modes.sort(key=lambda k: (k.alpha, k.l, k.m, 0 if k.angular == "cos" else 1))
    return modes


def disk_bessel_mode_field(
    mode: DiskModeIndex,
    *,
    x_um: np.ndarray,
    y_um: np.ndarray,
    core_radius_um: float,
    mask: np.ndarray | None = None,
    normalize: bool = True,
) -> np.ndarray:
    r"""Compute a single disk-Bessel mode field on a cartesian grid.

    The field is defined (inside the core) as:

    .. math::

        u_{lm}(r,\phi) = J_l(j_{l,m} r/a) \\times
            \\begin{cases}
                \\cos(l\\phi), & \\text{angular='cos'} \\\\
                \\sin(l\\phi), & \\text{angular='sin'}
            \\end{cases}

    and zero outside the core radius ``a``.

    Normalization
    -------------
    If ``normalize=True``, the field is normalized to unit L2 norm on the discrete grid.
    This keeps mode coefficients comparable when you superpose modes.
    """
    if core_radius_um <= 0:
        raise ValueError("core_radius_um must be > 0")
    if x_um.shape != y_um.shape:
        raise ValueError("x_um and y_um must have the same shape")
    if mask is not None and mask.shape != x_um.shape:
        raise ValueError("mask must have same shape as x_um")

    r_um = np.sqrt(x_um * x_um + y_um * y_um)
    phi = np.arctan2(y_um, x_um)
    rho = r_um / float(core_radius_um)

    # radial term (defined for rho<=1)
    J = special.jv(int(mode.l), float(mode.alpha) * rho)

    if mode.angular == "cos":
        ang = np.cos(int(mode.l) * phi)
    elif mode.angular == "sin":
        ang = np.sin(int(mode.l) * phi)
    else:
        raise ValueError(f"unknown angular type: {mode.angular}")

    u = (J * ang).astype(np.float64, copy=False)

    if mask is None:
        mask = r_um <= float(core_radius_um)

    u = np.where(mask, u, 0.0)

    if normalize:
        denom = float(np.sqrt(np.sum(u[mask] * u[mask])))
        if denom > 0:
            u = u / denom

    return u.astype(np.float64, copy=False)


def precompute_mode_stack(
    modes: Sequence[DiskModeIndex],
    *,
    x_um: np.ndarray,
    y_um: np.ndarray,
    core_radius_um: float,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Precompute a stack of mode fields with shape (n_modes, n, n)."""
    stack = []
    for m in modes:
        stack.append(
            disk_bessel_mode_field(
                m,
                x_um=x_um,
                y_um=y_um,
                core_radius_um=core_radius_um,
                mask=mask,
                normalize=True,
            )
        )
    return np.stack(stack, axis=0)


def random_complex_coeffs(
    n_modes: int,
    *,
    seed: int | None = 0,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Generate random complex coefficients with random phases.

    If ``weights`` is provided, coefficients are scaled by it.
    """
    if n_modes <= 0:
        raise ValueError("n_modes must be > 0")

    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=int(n_modes))
    coeffs = np.exp(1j * phases)

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (int(n_modes),):
            raise ValueError("weights must have shape (n_modes,)")
        coeffs = coeffs * w

    return coeffs.astype(np.complex128, copy=False)


def superpose_modes(
    mode_stack: np.ndarray,
    coeffs: np.ndarray,
) -> np.ndarray:
    """Return complex field U(x,y) = sum_k coeffs[k] * mode_stack[k]."""
    if mode_stack.ndim != 3:
        raise ValueError("mode_stack must have shape (n_modes, n, n)")
    n_modes = mode_stack.shape[0]
    if coeffs.shape != (n_modes,):
        raise ValueError("coeffs must have shape (n_modes,)")

    # tensordot over mode axis -> (n, n)
    return np.tensordot(coeffs, mode_stack, axes=(0, 0))


def intensity_from_field(u: np.ndarray) -> np.ndarray:
    """Return intensity I=|U|^2 as float64."""
    return (u.real * u.real + u.imag * u.imag).astype(np.float64, copy=False)


def speckle_contrast(intensity: np.ndarray, mask: np.ndarray) -> float:
    """Compute speckle contrast C = std/mean over a mask."""
    if intensity.shape != mask.shape:
        raise ValueError("intensity and mask must have the same shape")
    if not np.any(mask):
        raise ValueError("mask is empty")

    region = intensity[mask]
    mu = float(np.mean(region))
    if mu == 0:
        raise ValueError("mean intensity is zero")
    return float(np.std(region) / mu)


def average_uncorrelated_intensities(
    mode_stack: np.ndarray,
    *,
    n_avg: int,
    seed: int = 0,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Average intensities from ``n_avg`` independent random-phase realizations.

    This is a proxy for time/spectral/polarization diversity when patterns are independent.
    """
    if n_avg <= 0:
        raise ValueError("n_avg must be > 0")

    rng = np.random.default_rng(int(seed))
    I_acc = np.zeros(mode_stack.shape[1:], dtype=np.float64)

    for _ in range(int(n_avg)):
        coeffs = random_complex_coeffs(mode_stack.shape[0], seed=int(rng.integers(0, 2**31 - 1)), weights=weights)
        u = superpose_modes(mode_stack, coeffs)
        I_acc += intensity_from_field(u)

    return I_acc / float(n_avg)


def apply_mode_phase_perturbation(
    base_coeffs: np.ndarray,
    *,
    delta_phase_rms_rad: float,
    seed: int = 0,
) -> np.ndarray:
    r"""Return new coeffs with an additional random phase perturbation per mode.

    We draw :math:`\delta\phi_k \\sim \\mathcal{N}(0, \\sigma^2)` and apply:

    ``coeffs_k <- coeffs_k * exp(i*delta_phi_k)``.

    This is a simple way to visualize "speckle drift" when fiber conditions change.
    """
    if delta_phase_rms_rad < 0:
        raise ValueError("delta_phase_rms_rad must be >= 0")

    rng = np.random.default_rng(int(seed))
    delta = rng.normal(loc=0.0, scale=float(delta_phase_rms_rad), size=base_coeffs.shape[0])
    return base_coeffs * np.exp(1j * delta)
