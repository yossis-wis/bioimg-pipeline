"""Multimode-fiber (MMF) models for speckle averaging with finite linewidth.

Why this module exists
----------------------
This repo already contains:

- a lightweight Fourier-optics speckle simulator (:mod:`src.excitation_speckle_sim`), and
- bookkeeping helpers for *how many independent speckle realizations* are averaged
  (:mod:`src.speckle_diversity_models`).

When discussing an MMF illumination concept with colleagues, one recurring question is:

> *How wide does the spectrum have to be, and how long does the fiber have to be,*
> for wavelength diversity to substantially reduce speckle?

This file provides a **transparent, parameterized** bridge from common fiber specs
(core diameter, NA, length) to an estimate of the **speckle spectral correlation width**
``Δλ_c``.

Key ideas (short)
-----------------
The output speckle of a multimode fiber depends on the relative phases of many guided
modes.

A change in wavelength changes those relative phases. If the spectrum spans many
*wavelengths that produce effectively uncorrelated patterns*, then a camera (or a molecule)
that integrates over that spectrum sees an **incoherent sum of intensities**, and speckle
contrast drops approximately as:

.. math::

    C \approx 1/\sqrt{N_{\lambda}}.

For step-index fibers under weak guidance, a standard geometric-optics estimate of the
**intermodal group-delay spread** is:

.. math::

    \Delta\tau \approx \frac{\mathrm{NA}^2}{2 n_{\mathrm{core}} c}\,L,

which implies an **optical path-length spread**:

.. math::

    \Delta\mathrm{OPL} \approx c\,\Delta\tau \approx \frac{\mathrm{NA}^2}{2 n_{\mathrm{core}}}\,L.

A common decorrelation condition is ``Δk·ΔOPL ~ 2π``, which leads to:

.. math::

    \Delta\lambda_c \sim \frac{\lambda_0^2}{\Delta\mathrm{OPL}}
    \approx \frac{2 n_{\mathrm{core}}\,\lambda_0^2}{\mathrm{NA}^2\,L}.

These are *order-of-magnitude* estimates, but they are good enough to falsify (or support)
claims like "you need an impractically long fiber".

Important caveat
----------------
Many "homogenizing" fibers are graded-index (GI) and can have **much lower modal
dispersion** than the step-index estimate above.

Rather than pretending we know the exact GI profile, this module exposes a single
``modal_delay_scale`` parameter:

- ``modal_delay_scale = 1``: step-index-like (high dispersion)
- ``modal_delay_scale << 1``: graded-index-like (low dispersion)

The notebook can sweep that scale to show the sensitivity.

All functions are unit-tested and intentionally lightweight.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


C_M_PER_S = 299_792_458.0


@dataclass(frozen=True)
class MultimodeFiber:
    """Simple MMF specification used for speckle / linewidth estimates.

    Parameters
    ----------
    core_diameter_um:
        Core diameter (µm). Example: 400 for a 400 µm core homogenizing fiber.
    na:
        Numerical aperture (unitless). Typical large-core silica MMF: ~0.22.
    length_m:
        Physical fiber length (m).
    n_core:
        Effective refractive index in the core. Silica around visible wavelengths is
        ~1.45–1.47.
    modal_delay_scale:
        Scale factor applied to the step-index intermodal delay estimate.

        Use this to represent reduced modal dispersion in graded-index fibers.
        Values in the range ~0.02–0.2 are plausible "strong GI" scenarios.
    """

    core_diameter_um: float
    na: float
    length_m: float
    n_core: float = 1.46
    modal_delay_scale: float = 1.0

    def __post_init__(self) -> None:
        for name, v in (
            ("core_diameter_um", self.core_diameter_um),
            ("na", self.na),
            ("length_m", self.length_m),
            ("n_core", self.n_core),
            ("modal_delay_scale", self.modal_delay_scale),
        ):
            if v <= 0:
                raise ValueError(f"{name} must be positive")


def v_number(*, core_radius_um: float, na: float, lambda_um: float) -> float:
    """Return the fiber V-number.

    For a step-index fiber in air (good approximation in most lab use):

    .. math::

        V = \frac{2\pi a\,\mathrm{NA}}{\lambda}

    where ``a`` is core radius.

    A fiber is strictly single-mode only for ``V < 2.405``.

    Notes
    -----
    This uses the standard weakly-guiding definition used in many optics texts.
    """

    if core_radius_um <= 0 or na <= 0 or lambda_um <= 0:
        raise ValueError("core_radius_um, na, and lambda_um must be > 0")

    a_m = core_radius_um * 1e-6
    lambda_m = lambda_um * 1e-6
    return (2.0 * math.pi * a_m * na) / lambda_m


def approx_num_guided_modes_step_index(v: float) -> int:
    """Approximate number of guided modes in a large-V step-index fiber.

    For ``V >> 1`` (typical for 200–400 µm cores at visible wavelengths), a
    classic estimate for the total number of guided modes is:

    .. math::

        M \approx V^2/2.

    Notes
    -----
    Different textbooks differ in whether they count polarization degeneracy
    explicitly. For the use-case in this repo (establishing that ``M`` is
    *enormous* for 400 µm-class MMFs), that factor-of-two ambiguity is
    irrelevant.
    """

    if v <= 0:
        raise ValueError("v must be positive")

    return max(1, int(round((v * v) / 2.0)))


def intermodal_group_delay_spread_step_index_s(*, length_m: float, na: float, n_core: float) -> float:
    """Worst-case intermodal group-delay spread for a step-index MMF.

    Using a geometric-optics bound for meridional rays:

    .. math::

        \Delta\tau \approx \frac{\mathrm{NA}^2}{2 n_{\mathrm{core}} c}\,L.

    This corresponds to the delay between the axial ray and the highest-angle guided ray.

    The result is a *useful upper bound* on the spread of group delays.

    References
    ----------
    This expression is standard in introductory fiber-optics treatments of intermodal
    dispersion for step-index fibers.
    """

    if length_m <= 0 or na <= 0 or n_core <= 0:
        raise ValueError("length_m, na, and n_core must be > 0")

    return (na * na) * length_m / (2.0 * n_core * C_M_PER_S)


def intermodal_group_delay_spread_s(fiber: MultimodeFiber) -> float:
    """Intermodal group-delay spread with an adjustable dispersion scale."""

    base = intermodal_group_delay_spread_step_index_s(length_m=fiber.length_m, na=fiber.na, n_core=fiber.n_core)
    return float(fiber.modal_delay_scale) * base


def optical_path_spread_m_from_delay(delta_tau_s: float) -> float:
    """Convert a group-delay spread (s) to an equivalent optical path-length spread (m)."""

    if delta_tau_s <= 0:
        raise ValueError("delta_tau_s must be > 0")
    return float(C_M_PER_S * delta_tau_s)


def optical_path_spread_m(fiber: MultimodeFiber) -> float:
    """Estimated effective optical path-length spread ``ΔOPL`` (meters)."""

    return optical_path_spread_m_from_delay(intermodal_group_delay_spread_s(fiber))


def optical_path_length_m(fiber: MultimodeFiber) -> float:
    """Return the on-axis optical path length ``OPL = n_core * L`` (meters).

    Notes
    -----
    This is **not** the same as the intermodal optical-path spread ``ΔOPL``.

    - ``OPL`` sets the *absolute* phase delay through the fiber.
    - ``ΔOPL`` sets how quickly the **relative phases between modes** decorrelate
      with wavelength, and therefore how quickly the output speckle pattern changes
      across a finite spectrum.

    Keeping both quantities explicit helps avoid common confusions in discussions
    about “is a few nm of linewidth enough?”.
    """

    return float(fiber.n_core * fiber.length_m)


def max_guided_meridional_ray_angle_rad(*, na: float, n_core: float) -> float:
    """Maximum meridional ray angle inside the core (radians), step-index estimate.

    For a weakly guiding fiber in air, a standard approximation is:

    .. math::

        \sin\theta_{\max} \approx \mathrm{NA}/n_{\mathrm{core}}.

    This is the geometric-optics picture underlying the common step-index modal
    delay bound used elsewhere in this module.
    """

    if na <= 0 or n_core <= 0:
        raise ValueError("na and n_core must be > 0")
    if na >= n_core:
        raise ValueError("na must be < n_core for this approximation")

    return float(math.asin(na / n_core))


def optical_path_spread_geometric_step_index_m(*, length_m: float, na: float, n_core: float) -> float:
    """Geometric-optics estimate of step-index intermodal optical path spread ``ΔOPL`` (m).

    Consider two limiting meridional rays in a step-index MMF:

    - An axial ray (\(\theta=0\)) with optical path length
      \(\mathrm{OPL}_0 = n\,L\).
    - The highest-angle guided meridional ray (\(\theta=\theta_{\max}\)) with
      \(\mathrm{OPL}_{\max} = n\,L/\cos\theta_{\max}\).

    The resulting spread is:

    .. math::

        \Delta\mathrm{OPL} = \mathrm{OPL}_{\max} - \mathrm{OPL}_0
        = nL\left(\frac{1}{\cos\theta_{\max}} - 1\right).

    For small angles this reduces to the familiar step-index scaling:

    .. math::

        \Delta\mathrm{OPL} \approx \frac{\mathrm{NA}^2}{2 n}\,L.

    This function is provided mainly for **intuition building**: it makes it clear
    which parts of the spec-sheet matter (\(L\), NA, and \(n\)).
    """

    if length_m <= 0 or na <= 0 or n_core <= 0:
        raise ValueError("length_m, na, and n_core must be > 0")

    theta = max_guided_meridional_ray_angle_rad(na=na, n_core=n_core)
    return float(n_core * length_m * (1.0 / math.cos(theta) - 1.0))


def optical_path_spread_geometric_m(fiber: MultimodeFiber) -> float:
    """Geometric-optics ``ΔOPL`` estimate with the same ``modal_delay_scale`` convention.

    - If ``modal_delay_scale=1``: step-index-like upper bound.
    - If ``modal_delay_scale<<1``: graded-index-like reduced dispersion proxy.
    """

    base = optical_path_spread_geometric_step_index_m(length_m=fiber.length_m, na=fiber.na, n_core=fiber.n_core)
    return float(fiber.modal_delay_scale) * base



def speckle_spectral_corr_width_nm(*, lambda0_nm: float, delta_opl_m: float) -> float:
    """Estimate speckle spectral correlation width ``Δλ_c`` (nm).

    Uses the phase decorrelation condition:

    .. math::

        \Delta k\,\Delta\mathrm{OPL} \sim 2\pi,\qquad k = 2\pi/\lambda.

    leading to the order-of-magnitude relationship:

    .. math::

        \Delta\lambda_c \sim \lambda_0^2/\Delta\mathrm{OPL}.

    Parameters
    ----------
    lambda0_nm:
        Center wavelength (nm).
    delta_opl_m:
        Effective optical path-length spread (m).

    Returns
    -------
    float
        ``Δλ_c`` in nm.
    """

    if lambda0_nm <= 0 or delta_opl_m <= 0:
        raise ValueError("lambda0_nm and delta_opl_m must be > 0")

    lambda_m = lambda0_nm * 1e-9
    delta_lambda_m = (lambda_m * lambda_m) / delta_opl_m
    return float(delta_lambda_m * 1e9)


def speckle_spectral_corr_width_nm_for_fiber(*, lambda0_nm: float, fiber: MultimodeFiber) -> float:
    """Convenience wrapper: ``Δλ_c`` from an MMF specification."""

    return speckle_spectral_corr_width_nm(lambda0_nm=lambda0_nm, delta_opl_m=optical_path_spread_m(fiber))


def estimate_n_lambda_from_fiber(
    *,
    lambda0_nm: float,
    source_span_nm: float,
    fiber: MultimodeFiber,
    n_lines: int = 1,
) -> int:
    """Estimate number of independent spectral speckle patterns from a fiber spec.

    This is a thin wrapper around the core logic:

    ``N_λ ≈ ceil(Δλ_src / Δλ_c)``.

    It exists so notebooks can stay readable.
    """

    if source_span_nm < 0:
        raise ValueError("source_span_nm must be >= 0")
    if n_lines <= 0:
        raise ValueError("n_lines must be > 0")

    dlam_c = speckle_spectral_corr_width_nm_for_fiber(lambda0_nm=lambda0_nm, fiber=fiber)

    if source_span_nm == 0:
        n_from_span = 1
    else:
        n_from_span = int(math.ceil(source_span_nm / dlam_c))

    return int(max(1, n_lines, n_from_span))


def required_fiber_length_m_for_target_n_lambda(
    *,
    lambda0_nm: float,
    na: float,
    n_core: float,
    source_span_nm: float,
    target_n_lambda: int,
    modal_delay_scale: float = 1.0,
) -> float:
    """Solve for the fiber length needed to reach a target ``N_λ``.

    We use the step-index scaling ``Δλ_c ∝ 1/L`` with an optional ``modal_delay_scale``.

    Starting from:

    .. math::

        \Delta\lambda_c \approx \frac{2 n_{\mathrm{core}}\,\lambda_0^2}{\mathrm{NA}^2\,L}\,\frac{1}{s},

    where ``s = modal_delay_scale`` (``s<1`` means less dispersion -> larger correlation width).

    Then:

    .. math::

        N_\lambda \approx \Delta\lambda_{\mathrm{src}}/\Delta\lambda_c.

    Solving for ``L`` yields:

    .. math::

        L \approx \frac{2 n_{\mathrm{core}}\,\lambda_0^2}{\mathrm{NA}^2}\,\frac{N_\lambda}{\Delta\lambda_{\mathrm{src}}}\,\frac{1}{s}.

    Returns a *design-scale estimate*, not a guarantee.
    """

    if lambda0_nm <= 0 or na <= 0 or n_core <= 0:
        raise ValueError("lambda0_nm, na, and n_core must be > 0")
    if source_span_nm <= 0:
        raise ValueError("source_span_nm must be > 0")
    if target_n_lambda <= 0:
        raise ValueError("target_n_lambda must be > 0")
    if modal_delay_scale <= 0:
        raise ValueError("modal_delay_scale must be > 0")

    lambda_m = lambda0_nm * 1e-9
    span_m = source_span_nm * 1e-9

    return float((2.0 * n_core * (lambda_m * lambda_m) * float(target_n_lambda)) / ((na * na) * span_m * modal_delay_scale))
