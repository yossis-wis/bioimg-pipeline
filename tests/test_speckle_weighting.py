from __future__ import annotations

import math

import numpy as np

from src.speckle_weighting import (
    effective_n_from_weights,
    effective_n_lambda_gaussian,
    effective_n_lambda_uniform,
    speckle_contrast_from_weights,
    uniform_top_hat_spectrum_bins,
)


def test_effective_n_equal_weights() -> None:
    w = np.ones(10, dtype=float)
    n_eff = effective_n_from_weights(w)
    assert math.isclose(n_eff, 10.0, rel_tol=0, abs_tol=1e-12)

    c = speckle_contrast_from_weights(w)
    assert math.isclose(c, 1.0 / math.sqrt(10.0), rel_tol=0, abs_tol=1e-12)


def test_effective_n_single_dominant_weight() -> None:
    w = np.array([1.0, 0.0, 0.0, 0.0])
    n_eff = effective_n_from_weights(w)
    assert math.isclose(n_eff, 1.0, rel_tol=0, abs_tol=1e-12)


def test_uniform_spectrum_bins_exact_multiple() -> None:
    # span = 1.0 nm, corr width = 0.2 nm -> 5 equal bins -> N_eff = 5
    n_eff = effective_n_lambda_uniform(span_nm=1.0, corr_width_nm=0.2)
    assert math.isclose(n_eff, 5.0, rel_tol=0, abs_tol=1e-12)


def test_uniform_spectrum_bins_partial_final_bin() -> None:
    # span = 1.0 nm, corr width = 0.3 nm -> widths [0.3, 0.3, 0.3, 0.1]
    bins = uniform_top_hat_spectrum_bins(span_nm=1.0, corr_width_nm=0.3)
    assert bins.n_bins == 4
    n_eff = bins.n_eff
    assert math.isclose(n_eff, 1.0 / (3 * 0.3 * 0.3 + 0.1 * 0.1), rel_tol=0, abs_tol=1e-12)


def test_gaussian_effective_n_limits() -> None:
    # If the linewidth is far narrower than the correlation width, there is ~1 effective bin.
    n_eff = effective_n_lambda_gaussian(lambda0_nm=640.0, fwhm_nm=0.05, corr_width_nm=1.0)
    assert math.isclose(n_eff, 1.0, rel_tol=0, abs_tol=1e-12)

    # For a broader line, N_eff should increase.
    n_eff2 = effective_n_lambda_gaussian(lambda0_nm=640.0, fwhm_nm=2.0, corr_width_nm=0.2)
    assert n_eff2 > 5.0
