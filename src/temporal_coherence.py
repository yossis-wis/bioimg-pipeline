from __future__ import annotations

"""Temporal-coherence helpers (linewidth ↔ coherence length/time).

This repo's MMF speckle notebooks often need to answer questions like:

- "Is 2 nm linewidth enough? What about 20 nm?"
- "How does linewidth compare to a fiber's intermodal optical-path spread?"

To keep those discussions reproducible (and avoid re-deriving constants in
multiple notebooks), this module provides lightweight conversions.

Important caveat
----------------
"Coherence length" depends on the chosen definition (FWHM vs 1/e, Gaussian vs
Lorentzian spectrum, etc.). Here we use **simple, common approximations** that
are sufficient for order-of-magnitude engineering.

We support two spectral shapes:

- ``profile='gaussian'``: uses the Gaussian time–bandwidth product for FWHM:
  ``Δν_FWHM * τ_c,FWHM ≈ 0.44``.
- ``profile='lorentzian'``: uses ``τ_c ≈ 1/(π Δν_FWHM)``.

The main outputs are:

- coherence time ``τ_c`` (seconds)
- coherence length in *vacuum-equivalent optical path* ``L_c = c τ_c`` (meters)

That last definition is the most directly comparable to the repo's
``ΔOPL`` estimates, which are also expressed as vacuum-equivalent optical path
differences.
"""

import math


C_M_PER_S = 299_792_458.0


def delta_nu_hz_from_delta_lambda_nm(*, lambda0_nm: float, delta_lambda_nm: float) -> float:
    """Convert a small wavelength span to frequency bandwidth (Hz).

    Uses the small-signal approximation around ``lambda0``:

    Δν ≈ c·Δλ / λ₀².

    Parameters
    ----------
    lambda0_nm:
        Center wavelength (nm).
    delta_lambda_nm:
        Small wavelength span (nm), e.g. FWHM.
    """

    if lambda0_nm <= 0 or delta_lambda_nm <= 0:
        raise ValueError("lambda0_nm and delta_lambda_nm must be > 0")

    lambda0_m = float(lambda0_nm) * 1e-9
    d_lambda_m = float(delta_lambda_nm) * 1e-9
    return C_M_PER_S * d_lambda_m / (lambda0_m * lambda0_m)


def coherence_time_s_from_linewidth_nm(
    *,
    lambda0_nm: float,
    fwhm_nm: float,
    profile: str = "gaussian",
) -> float:
    """Approximate coherence time (seconds) from spectral FWHM in nm."""

    delta_nu = delta_nu_hz_from_delta_lambda_nm(lambda0_nm=lambda0_nm, delta_lambda_nm=fwhm_nm)

    p = profile.lower().strip()
    if p == "gaussian":
        # FWHM time–bandwidth product for a Gaussian.
        return 0.44 / float(delta_nu)
    if p == "lorentzian":
        return 1.0 / (math.pi * float(delta_nu))

    raise ValueError("profile must be 'gaussian' or 'lorentzian'")


def coherence_length_m_from_linewidth_nm(
    *,
    lambda0_nm: float,
    fwhm_nm: float,
    profile: str = "gaussian",
) -> float:
    """Approximate coherence length as a *vacuum-equivalent* optical path (meters)."""

    return C_M_PER_S * coherence_time_s_from_linewidth_nm(
        lambda0_nm=lambda0_nm, fwhm_nm=fwhm_nm, profile=profile
    )


def linewidth_nm_from_coherence_length_m(
    *,
    lambda0_nm: float,
    coherence_length_m: float,
    profile: str = "gaussian",
) -> float:
    """Inverse of :func:`coherence_length_m_from_linewidth_nm` (approximate).

    Parameters
    ----------
    lambda0_nm:
        Center wavelength (nm).
    coherence_length_m:
        Coherence length as vacuum-equivalent optical path (m).
    profile:
        ``'gaussian'`` or ``'lorentzian'``.
    """

    if lambda0_nm <= 0 or coherence_length_m <= 0:
        raise ValueError("lambda0_nm and coherence_length_m must be > 0")

    lambda0_m = float(lambda0_nm) * 1e-9
    lc = float(coherence_length_m)

    p = profile.lower().strip()
    if p == "gaussian":
        # Lc = 0.44 * λ^2 / Δλ
        d_lambda_m = 0.44 * (lambda0_m * lambda0_m) / lc
    elif p == "lorentzian":
        # Lc = (c/πΔν) and Δν ≈ c/λ^2 Δλ -> Lc ≈ λ^2/(πΔλ)
        d_lambda_m = (lambda0_m * lambda0_m) / (math.pi * lc)
    else:
        raise ValueError("profile must be 'gaussian' or 'lorentzian'")

    return d_lambda_m * 1e9
