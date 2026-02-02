"""First-order illumination design calculations (widefield / MMF delivery).

This module intentionally avoids microscope-brand specifics. It provides
*back-of-the-envelope* numbers that are still useful for deciding whether an
illumination concept is plausible:

* required sample power for a target irradiance over a square/rectangular ROI
* the matching field stop size in a sample-conjugate image plane
* objective BFP/pupil diameter and what it means to "fill 0.3 of the BFP"
* the collimated beam diameter out of a fiber collimator (given NA and f_coll)

All equations follow standard Fourier/Köhler optics approximations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


def _pair(x: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(x, tuple):
        if len(x) != 2:
            msg = "Expected a 2-tuple."
            raise ValueError(msg)
        return float(x[0]), float(x[1])
    return float(x), float(x)


def roi_area_cm2(roi_um: float | tuple[float, float]) -> float:
    """ROI area in :math:`\mathrm{cm}^2`.

    Notes
    -----
    :math:`1\,\mu\mathrm{m} = 10^{-4}\,\mathrm{cm}`.
    """

    wx_um, wy_um = _pair(roi_um)
    return (wx_um * 1e-4) * (wy_um * 1e-4)


def required_sample_power_mw(
    irradiance_kw_per_cm2: float,
    roi_um: float | tuple[float, float],
) -> float:
    """Required optical power at the *sample plane* in mW.

    Parameters
    ----------
    irradiance_kw_per_cm2:
        Target irradiance at the sample in :math:`\mathrm{kW}/\mathrm{cm}^2`.
    roi_um:
        ROI width/height in :math:`\mu\mathrm{m}` (float => square ROI).
    """

    # 1 kW/cm^2 = 1e3 W/cm^2; 1 W = 1e3 mW.
    return irradiance_kw_per_cm2 * 1e6 * roi_area_cm2(roi_um)


def field_stop_size_mm(
    roi_um: float | tuple[float, float],
    sample_to_stop_magnification: float,
) -> tuple[float, float]:
    """Field stop size (mm) for an ROI at the sample.

    If the field stop sits in an image plane conjugate to the sample, then the
    lateral magnification from sample to that plane defines the relationship:

    .. math::
        D_{\mathrm{stop}} = M\,D_{\mathrm{sample}}.

    For a standard infinity system, an image-plane stop often sees approximately
    the objective magnification (e.g. 100x).
    """

    wx_um, wy_um = _pair(roi_um)
    wx_mm = (wx_um * sample_to_stop_magnification) / 1000.0
    wy_mm = (wy_um * sample_to_stop_magnification) / 1000.0
    return wx_mm, wy_mm


def objective_focal_length_mm(tube_lens_f_mm: float, magnification: float) -> float:
    """Estimate objective focal length via :math:`f_{\mathrm{obj}}=f_{\mathrm{TL}}/M`."""

    if magnification <= 0:
        msg = "magnification must be positive"
        raise ValueError(msg)
    return tube_lens_f_mm / magnification


def objective_pupil_diameter_mm(na_obj: float, f_obj_mm: float) -> float:
    """Approximate pupil/BFP diameter in mm.

    Using the paraxial pupil relationship:

    .. math::
        D_{\mathrm{pupil}} \approx 2 f_{\mathrm{obj}}\,\mathrm{NA}_{\mathrm{obj}}.
    """

    if na_obj <= 0 or f_obj_mm <= 0:
        msg = "na_obj and f_obj_mm must be positive"
        raise ValueError(msg)
    return 2.0 * f_obj_mm * na_obj


def bfp_beam_diameter_mm(pupil_fill_fraction: float, pupil_diameter_mm: float) -> float:
    """Beam diameter at the BFP if you fill a fraction of the pupil."""

    if not (0.0 < pupil_fill_fraction <= 1.0):
        msg = "pupil_fill_fraction must be in (0, 1]"
        raise ValueError(msg)
    if pupil_diameter_mm <= 0:
        msg = "pupil_diameter_mm must be positive"
        raise ValueError(msg)
    return pupil_fill_fraction * pupil_diameter_mm


def illumination_na(na_obj: float, pupil_fill_fraction: float) -> float:
    """Illumination NA for a uniformly filled circular pupil fraction."""

    return na_obj * pupil_fill_fraction


def speckle_grain_size_um(lambda_nm: float, na_illum: float) -> float:
    """Approx. speckle grain (FWHM-ish) in the sample plane.

    For fully developed speckle with a circular aperture, a common rule-of-thumb
    for grain size is

    .. math::
        d \sim \frac{\lambda}{2\,\mathrm{NA}}.

    This is for *lateral* scale at the sample plane.
    """

    if lambda_nm <= 0 or na_illum <= 0:
        msg = "lambda_nm and na_illum must be positive"
        raise ValueError(msg)
    return (lambda_nm * 1e-3) / (2.0 * na_illum)


def collimated_beam_diameter_mm(f_coll_mm: float, fiber_na: float) -> float:
    """Approximate collimated beam diameter (mm) from a fiber collimator.

    Uses a more accurate mapping than :math:`D\approx 2 f \mathrm{NA}` by taking
    :math:`\theta=\arcsin(\mathrm{NA})` and :math:`D\approx 2 f \tan\theta`.
    """

    if f_coll_mm <= 0 or not (0 < fiber_na < 1):
        msg = "f_coll_mm must be positive and fiber_na must be in (0, 1)"
        raise ValueError(msg)
    theta = math.asin(fiber_na)
    return 2.0 * f_coll_mm * math.tan(theta)


@dataclass(frozen=True)
class PowerBudget:
    """A simple multiplicative power budget.

    Each factor is a transmission/coupling efficiency in [0, 1].
    """

    coupling_into_fiber: float = 0.6
    fiber_to_collimator: float = 0.95
    stop_and_relays: float = 0.7
    objective_and_misc: float = 0.8

    def total_throughput(self) -> float:
        factors: Iterable[float] = (
            self.coupling_into_fiber,
            self.fiber_to_collimator,
            self.stop_and_relays,
            self.objective_and_misc,
        )
        t = 1.0
        for f in factors:
            if not (0 <= f <= 1):
                msg = "All budget factors must be in [0, 1]."
                raise ValueError(msg)
            t *= f
        return t


def required_fiber_exit_power_mw(sample_power_mw: float, throughput: float) -> float:
    """Power required at the fiber exit to deliver ``sample_power_mw``."""

    if sample_power_mw < 0:
        msg = "sample_power_mw must be non-negative"
        raise ValueError(msg)
    if not (0 < throughput <= 1):
        msg = "throughput must be in (0, 1]"
        raise ValueError(msg)
    return sample_power_mw / throughput
