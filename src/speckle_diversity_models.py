"""Speckle averaging / diversity bookkeeping for short exposures.

This repo contains a Fourier-optics kernel that can simulate an illumination
speckle field (see :mod:`src.excitation_speckle_sim`). In practice, the speckle
contrast a protein "feels" during an exposure depends on how many *statistically
independent* speckle realizations are incoherently averaged during that
exposure.

For many imaging design conversations, you can treat the effective number of
independent patterns as

.. math::

    N_{\mathrm{eff}} \approx N_{t}\,N_{\lambda}\,N_{\mathrm{pol}}\,N_{\mathrm{angle}},

and the resulting speckle contrast as

.. math::

    C \approx \frac{1}{\sqrt{N_{\mathrm{eff}}}}.

This file provides conservative, transparent estimates for

* :math:`N_t` from a (possibly slow) scrambler frequency and the exposure time
* :math:`N_\lambda` from a laser's spectral span and the speckle spectral
  decorrelation bandwidth

The spectral decorrelation estimate uses a simple optical-path-length spread
argument:

.. math::

    \Delta \lambda_c \sim \frac{\lambda^2}{\Delta \mathrm{OPL}},

where :math:`\Delta\mathrm{OPL} \approx n\,\Delta L` is an effective spread in
optical path length between the interfering contributions.

These expressions are approximations; the goal is to parameterize the
assumptions so the notebook can sweep them.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


def required_n_eff_for_contrast(c_target: float) -> int:
    """Minimum :math:`N_{\mathrm{eff}}` to reach a target speckle contrast."""

    if not (0 < c_target <= 1):
        msg = "c_target must be in (0, 1]"
        raise ValueError(msg)
    return int(math.ceil(1.0 / (c_target * c_target)))


def speckle_contrast_from_n_eff(n_eff: float) -> float:
    """Approximate speckle contrast :math:`C\approx 1/\sqrt{N_{\mathrm{eff}}}`."""

    if n_eff <= 0:
        msg = "n_eff must be positive"
        raise ValueError(msg)
    return 1.0 / math.sqrt(n_eff)


def n_time_samples(exposure_s: float, scrambler_hz: float) -> int:
    """Number of temporally independent realizations during an exposure."""

    if exposure_s <= 0:
        msg = "exposure_s must be positive"
        raise ValueError(msg)
    if scrambler_hz <= 0:
        return 1
    return max(1, int(math.floor(exposure_s * scrambler_hz)))


def effective_n_time_samples(
    exposure_s: float,
    scrambler_hz: float,
    *,
    successive_pattern_correlation: float = 0.0,
) -> float:
    """Effective independent samples for imperfect (correlated) scrambling.

    If successive realizations have correlation coefficient :math:`\rho`, then
    averaging is less efficient than assuming full independence.

    We use a simple heuristic:

    .. math::
        N_{t,\mathrm{eff}} = 1 + (N_t - 1)\,(1-\rho).

    where :math:`N_t=\lfloor f_\mathrm{scr}\,\tau\rfloor`.
    """

    if not (0 <= successive_pattern_correlation < 1):
        msg = "successive_pattern_correlation must be in [0, 1)"
        raise ValueError(msg)
    n_t = n_time_samples(exposure_s, scrambler_hz)
    return 1.0 + (n_t - 1) * (1.0 - successive_pattern_correlation)


def estimate_speckle_spectral_corr_width_nm(
    lambda0_nm: float,
    optical_path_spread_m: float,
    *,
    n_eff: float = 1.46,
) -> float:
    """Estimate speckle spectral correlation width :math:`\Delta\lambda_c`.

    Using the phase decorrelation condition :math:`\Delta k\,\Delta\mathrm{OPL}\sim2\pi`.
    With :math:`k=2\pi/\lambda`, we get (order-of-magnitude):

    .. math::
        \Delta\lambda_c \sim \frac{\lambda^2}{\Delta\mathrm{OPL}}.

    Parameters
    ----------
    lambda0_nm:
        Center wavelength.
    optical_path_spread_m:
        Effective optical path-length spread :math:`\Delta\mathrm{OPL}`.
        If you only know a physical length spread :math:`\Delta L`, you can use
        :math:`\Delta\mathrm{OPL}\approx n\,\Delta L`.
    n_eff:
        Effective refractive index used only if you pass a physical length
        spread instead of an OPL spread.
    """

    if lambda0_nm <= 0 or optical_path_spread_m <= 0 or n_eff <= 0:
        msg = "lambda0_nm, optical_path_spread_m, and n_eff must be positive"
        raise ValueError(msg)
    lambda_m = lambda0_nm * 1e-9
    delta_lambda_m = (lambda_m**2) / optical_path_spread_m
    return delta_lambda_m * 1e9


def estimate_n_lambda(
    source_span_nm: float,
    speckle_corr_width_nm: float,
    *,
    n_lines: int = 1,
) -> int:
    """Estimate the number of independent speckle patterns from spectrum.

    Parameters
    ----------
    source_span_nm:
        Effective spectral span during the exposure. This can represent a
        diode's instantaneous linewidth (if the detector averages over it) *or*
        a deliberate multi-wavelength sweep across the exposure.
    speckle_corr_width_nm:
        Speckle spectral correlation width :math:`\Delta\lambda_c`.
    n_lines:
        If the illumination consists of several discrete lines separated by
        more than :math:`\Delta\lambda_c`, they contribute additional diversity.
    """

    if speckle_corr_width_nm <= 0:
        msg = "speckle_corr_width_nm must be positive"
        raise ValueError(msg)
    if source_span_nm < 0:
        msg = "source_span_nm must be non-negative"
        raise ValueError(msg)
    if n_lines <= 0:
        msg = "n_lines must be positive"
        raise ValueError(msg)

    if source_span_nm == 0:
        n_from_span = 1
    else:
        n_from_span = int(math.ceil(source_span_nm / speckle_corr_width_nm))

    return max(1, n_from_span, n_lines)


@dataclass(frozen=True)
class DiversityBudget:
    """Bookkeeping for diversity channels contributing to speckle averaging."""

    n_lambda: int = 1
    n_pol: int = 1
    n_angle: int = 1

    def n_source_states(self) -> int:
        for name, n in (
            ("n_lambda", self.n_lambda),
            ("n_pol", self.n_pol),
            ("n_angle", self.n_angle),
        ):
            if n <= 0:
                msg = f"{name} must be positive"
                raise ValueError(msg)
        return int(self.n_lambda * self.n_pol * self.n_angle)


def estimate_n_eff(
    *,
    exposure_s: float,
    scrambler_hz: float,
    diversity: DiversityBudget,
    successive_pattern_correlation: float = 0.0,
) -> float:
    """Estimate :math:`N_{\mathrm{eff}}` for a given exposure and diversity."""

    n_t_eff = effective_n_time_samples(
        exposure_s, scrambler_hz, successive_pattern_correlation=successive_pattern_correlation
    )
    return n_t_eff * float(diversity.n_source_states())
