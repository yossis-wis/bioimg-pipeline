from __future__ import annotations

import math

from src.speckle_diversity_models import (
    DiversityBudget,
    estimate_n_eff,
    estimate_n_lambda,
    estimate_speckle_spectral_corr_width_nm,
    n_time_samples,
    required_n_eff_for_contrast,
    speckle_contrast_from_n_eff,
)


def test_required_n_eff_and_contrast_relation() -> None:
    assert required_n_eff_for_contrast(0.1) == 100
    assert math.isclose(speckle_contrast_from_n_eff(100.0), 0.1, rel_tol=0, abs_tol=1e-12)


def test_time_samples_500us_10khz() -> None:
    assert n_time_samples(exposure_s=500e-6, scrambler_hz=10e3) == 5
    assert n_time_samples(exposure_s=500e-6, scrambler_hz=0.0) == 1


def test_spectral_correlation_width_and_n_lambda() -> None:
    # λ=640 nm, ΔOPL=3.2 cm -> Δλ_c ~ λ^2/ΔOPL ~ 0.0128 nm
    dlam_c = estimate_speckle_spectral_corr_width_nm(lambda0_nm=640.0, optical_path_spread_m=0.032)
    assert math.isclose(dlam_c, 0.0128, rel_tol=0.2, abs_tol=0)  # order-of-magnitude

    nlam = estimate_n_lambda(source_span_nm=1.0, speckle_corr_width_nm=dlam_c, n_lines=1)
    assert nlam >= 10  # depends on tolerance; should be large if dlam_c is small


def test_combined_n_eff() -> None:
    # 500 µs, 10 kHz -> N_t = 5. With N_src=20 -> N_eff ≈ 100.
    div = DiversityBudget(n_lambda=20, n_pol=1, n_angle=1)
    n_eff = estimate_n_eff(exposure_s=500e-6, scrambler_hz=10e3, diversity=div)
    assert math.isclose(n_eff, 100.0, rel_tol=0, abs_tol=1e-12)
