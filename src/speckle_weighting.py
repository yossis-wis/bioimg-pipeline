from __future__ import annotations

"""Weighted diversity models for speckle averaging.

Motivation
----------
Many "back-of-the-envelope" speckle arguments assume **equal weights**:

- N independent speckle realizations
- equal power in each
- zero correlation between them

Under those assumptions, speckle contrast scales as::

    C \approx 1/sqrt(N).

In real optical systems, two common deviations matter:

1) **Unequal weights**
   - e.g. spectral power is not flat across the linewidth
   - or only a subset of fiber modes carry most of the power

2) **Partial correlation** between successive realizations (handled elsewhere in
   :mod:`src.speckle_diversity_models`).

This module focuses on (1): how unequal weights reduce the *effective* number of
independent averages.

Key identity
------------
If you incoherently sum independent speckle patterns with weights ``w_i``
(proportional to their mean intensity contribution), and each pattern is
"fully developed" (unit contrast), the resulting contrast is::

    C = sqrt(sum(w_i^2)) / sum(w_i)

and therefore the effective sample count is::

    N_eff = (sum(w_i))^2 / sum(w_i^2)

When all weights are equal, ``N_eff = N``.

This is the same "inverse participation ratio" used in many fields.

Notes
-----
These helpers are deliberately lightweight (NumPy + stdlib only) so they can be
used in notebooks without pulling in additional dependencies.
"""

import math
from dataclasses import dataclass

import numpy as np


def _as_positive_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError("weights must be 1D")
    if np.any(w < 0):
        raise ValueError("weights must be nonnegative")
    s = float(w.sum())
    if s <= 0:
        raise ValueError("weights must sum to > 0")
    return w


def effective_n_from_weights(weights: np.ndarray) -> float:
    """Return effective sample count ``N_eff`` implied by nonnegative weights.

    Parameters
    ----------
    weights:
        1D array-like of nonnegative weights (need not be normalized).

    Returns
    -------
    float
        ``N_eff = (sum w)^2 / sum(w^2)``.

    Examples
    --------
    - Equal weights ``[1, 1, 1, 1]`` -> ``N_eff = 4``.
    - Dominant single weight ``[1, 0, 0, 0]`` -> ``N_eff = 1``.
    """

    w = _as_positive_weights(weights)
    s1 = float(w.sum())
    s2 = float(np.sum(w * w))
    return (s1 * s1) / s2


def speckle_contrast_from_weights(weights: np.ndarray) -> float:
    """Speckle contrast for an incoherent weighted sum of independent unit-contrast speckles."""

    w = _as_positive_weights(weights)
    return float(math.sqrt(float(np.sum(w * w))) / float(w.sum()))


@dataclass(frozen=True)
class SpectralBins:
    """A discrete spectrum representation for speckle-averaging bookkeeping."""

    centers_nm: np.ndarray
    weights: np.ndarray
    bin_width_nm: float

    @property
    def n_bins(self) -> int:  # pragma: no cover (trivial)
        return int(self.weights.size)

    @property
    def n_eff(self) -> float:
        return effective_n_from_weights(self.weights)


def uniform_top_hat_spectrum_bins(*, span_nm: float, corr_width_nm: float) -> SpectralBins:
    """Discretize a uniform (top-hat) spectrum into independent correlation bins.

    This is a more numerically explicit version of::

        N_lambda = ceil(span_nm / corr_width_nm)

    that also handles a partial final bin and produces weights for
    :func:`effective_n_from_weights`.
    """

    if span_nm <= 0 or corr_width_nm <= 0:
        raise ValueError("span_nm and corr_width_nm must be > 0")

    n_full = int(span_nm // corr_width_nm)
    rem = float(span_nm - n_full * corr_width_nm)

    widths = [corr_width_nm] * n_full
    if rem > 1e-12:
        widths.append(rem)

    w = np.asarray(widths, dtype=float)
    w /= float(w.sum())

    # Centers are arbitrary for a top-hat; we place them on [-(span/2), +(span/2)].
    # The absolute λ0 is irrelevant for the weight-only calculation.
    edges = np.concatenate([[0.0], np.cumsum(np.asarray(widths, dtype=float))])
    centers = 0.5 * (edges[:-1] + edges[1:]) - 0.5 * span_nm

    return SpectralBins(centers_nm=centers, weights=w, bin_width_nm=float(corr_width_nm))


def gaussian_spectrum_bins(
    *,
    lambda0_nm: float,
    fwhm_nm: float,
    corr_width_nm: float,
    n_sigma: float = 4.0,
) -> SpectralBins:
    """Discretize a Gaussian spectrum into bins of width ``corr_width_nm``.

    Parameters
    ----------
    lambda0_nm:
        Center wavelength.
    fwhm_nm:
        Full-width at half maximum (FWHM) of the Gaussian.
    corr_width_nm:
        Speckle spectral correlation width (bin width).
    n_sigma:
        Truncation radius in standard deviations (default ±4σ).

    Returns
    -------
    SpectralBins
        Bin centers and normalized weights (integrated spectral power per bin).

    Notes
    -----
    For most purposes, the exact truncation is not critical as long as it
    captures essentially all the power (±4σ captures >99.99%).
    """

    if fwhm_nm <= 0 or corr_width_nm <= 0:
        raise ValueError("fwhm_nm and corr_width_nm must be > 0")
    if n_sigma <= 0:
        raise ValueError("n_sigma must be > 0")

    sigma = float(fwhm_nm) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    lo = float(lambda0_nm) - float(n_sigma) * sigma
    hi = float(lambda0_nm) + float(n_sigma) * sigma

    n_bins = max(1, int(math.ceil((hi - lo) / float(corr_width_nm))))
    edges = lo + float(corr_width_nm) * np.arange(n_bins + 1, dtype=float)

    # Gaussian CDF using erf.
    def cdf(x: float) -> float:
        z = (x - float(lambda0_nm)) / (math.sqrt(2.0) * sigma)
        return 0.5 * (1.0 + math.erf(z))

    weights = np.empty(n_bins, dtype=float)
    for i in range(n_bins):
        weights[i] = cdf(float(edges[i + 1])) - cdf(float(edges[i]))

    # Normalize (since we truncated tails).
    s = float(weights.sum())
    if s <= 0:
        raise RuntimeError("Gaussian binning produced zero total weight")
    weights /= s

    centers = 0.5 * (edges[:-1] + edges[1:])
    return SpectralBins(centers_nm=centers, weights=weights, bin_width_nm=float(corr_width_nm))


def effective_n_lambda_uniform(*, span_nm: float, corr_width_nm: float) -> float:
    """Convenience: effective ``N_lambda`` for a uniform top-hat spectrum."""

    return uniform_top_hat_spectrum_bins(span_nm=span_nm, corr_width_nm=corr_width_nm).n_eff


def effective_n_lambda_gaussian(
    *, lambda0_nm: float, fwhm_nm: float, corr_width_nm: float, n_sigma: float = 4.0
) -> float:
    """Convenience: effective ``N_lambda`` for a Gaussian spectrum."""

    return gaussian_spectrum_bins(
        lambda0_nm=lambda0_nm,
        fwhm_nm=fwhm_nm,
        corr_width_nm=corr_width_nm,
        n_sigma=n_sigma,
    ).n_eff
