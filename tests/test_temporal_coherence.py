from __future__ import annotations

import math

import pytest

from src.temporal_coherence import (
    coherence_length_m_from_linewidth_nm,
    coherence_time_s_from_linewidth_nm,
    delta_nu_hz_from_delta_lambda_nm,
    linewidth_nm_from_coherence_length_m,
)


def test_delta_nu_scales_with_delta_lambda() -> None:
    # Δν ≈ c Δλ / λ^2 so doubling Δλ should double Δν.
    d1 = delta_nu_hz_from_delta_lambda_nm(lambda0_nm=640.0, delta_lambda_nm=1.0)
    d2 = delta_nu_hz_from_delta_lambda_nm(lambda0_nm=640.0, delta_lambda_nm=2.0)
    assert math.isfinite(d1) and d1 > 0
    assert pytest.approx(d1 * 2.0, rel=1e-12) == d2


@pytest.mark.parametrize("profile", ["gaussian", "lorentzian"])
def test_coherence_length_inversely_scales_with_linewidth(profile: str) -> None:
    # Coherence length ~ 1/linewidth. Ratio should match linewidth ratio.
    lc_2 = coherence_length_m_from_linewidth_nm(lambda0_nm=640.0, fwhm_nm=2.0, profile=profile)
    lc_20 = coherence_length_m_from_linewidth_nm(lambda0_nm=640.0, fwhm_nm=20.0, profile=profile)
    assert lc_2 > lc_20
    assert pytest.approx(lc_2 / lc_20, rel=1e-12) == 10.0


@pytest.mark.parametrize("profile", ["gaussian", "lorentzian"])
def test_linewidth_roundtrip(profile: str) -> None:
    # linewidth -> coherence length -> linewidth should be approximately invertible
    fwhm_nm = 3.5
    lc = coherence_length_m_from_linewidth_nm(lambda0_nm=640.0, fwhm_nm=fwhm_nm, profile=profile)
    fwhm_back = linewidth_nm_from_coherence_length_m(
        lambda0_nm=640.0, coherence_length_m=lc, profile=profile
    )
    assert pytest.approx(fwhm_nm, rel=1e-12) == fwhm_back


def test_coherence_time_positive() -> None:
    t = coherence_time_s_from_linewidth_nm(lambda0_nm=640.0, fwhm_nm=2.0)
    assert t > 0
