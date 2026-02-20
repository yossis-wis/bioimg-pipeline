from __future__ import annotations

import numpy as np
import pytest

from src.instantaneous_phasor_sum import (
    InstantaneousPhasorSumConfig,
    compute_phasors,
    parse_deltaL_mm,
)


def test_parse_deltaL_mm() -> None:
    assert parse_deltaL_mm("0,25,51") == (0.0, 25.0, 51.0)
    assert parse_deltaL_mm(" 1.5 ") == (1.5,)
    with pytest.raises(ValueError):
        _ = parse_deltaL_mm("")


def test_compute_phasors_shapes_and_intensity_invariant() -> None:
    cfg = InstantaneousPhasorSumConfig(
        lambda0_nm=640.0,
        n_wavelengths=5,
        T_ps=10.0,
        dt_ps=2.0,
        deltaL_mm=(0.0, 25.0, 51.0),
        path_amp="equal_power",
        ref="lowest",
        order="by_wavelength",
        seed=0,
        add_random_initial_phase=True,
    )
    out = compute_phasors(cfg)

    assert out.f_hz.shape == (cfg.n_wavelengths,)
    assert out.lambda_m.shape == (cfg.n_wavelengths,)
    assert out.tau_s.shape == (len(cfg.deltaL_mm),)

    n_times = out.times_ps.shape[0]
    assert out.times_s.shape == (n_times,)

    n_steps = cfg.n_wavelengths * len(cfg.deltaL_mm)
    assert out.order_kp.shape == (n_steps, 2)
    assert out.phasors.shape == (n_times, n_steps)
    assert out.field.shape == (n_times,)
    assert out.intensity.shape == (n_times,)

    # I = |E|^2
    I2 = (out.field.real * out.field.real + out.field.imag * out.field.imag).astype(float)
    assert np.allclose(out.intensity, I2, rtol=0.0, atol=1e-12)


def test_equal_power_path_amplitude() -> None:
    cfg = InstantaneousPhasorSumConfig(n_wavelengths=3, deltaL_mm=(0.0, 25.0, 51.0), path_amp="equal_power")
    out = compute_phasors(cfg)
    assert pytest.approx(1.0 / np.sqrt(3.0), rel=1e-12) == out.A_path
