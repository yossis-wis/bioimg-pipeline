
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.instantaneous_phasor_sum import (
    InstantaneousPhasorSumConfig,
    compute_phasors,
    parse_deltaL_mm,
    time_average_intensity_analytic,
    time_average_intensity_numeric,
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


def test_ref_mode_does_not_change_intensity() -> None:
    cfg = InstantaneousPhasorSumConfig(
        lambda0_nm=640.0,
        n_wavelengths=10,
        T_ps=40.0,
        dt_ps=1.0,
        deltaL_mm=(0.0, 25.0, 51.0),
        path_amp="equal_power",
        ref="lowest",
        order="by_wavelength",
        seed=2,
        add_random_initial_phase=True,
    )
    out_a = compute_phasors(cfg)
    out_b = compute_phasors(replace(cfg, ref="below_lowest"))

    # Changing the envelope reference is a global phase rotation of E(t); intensity is invariant.
    assert np.allclose(out_a.intensity, out_b.intensity, rtol=0.0, atol=1e-9)


def test_time_average_analytic_matches_numeric_over_full_period() -> None:
    # Use the canonical period alignment: Δf = 1/T and sample t on an integer grid over [0, T].
    cfg = InstantaneousPhasorSumConfig(
        lambda0_nm=640.0,
        n_wavelengths=20,
        T_ps=160.0,
        dt_ps=1.0,
        deltaL_mm=(0.0, 25.0, 51.0),
        path_amp="equal_power",
        ref="lowest",
        order="by_wavelength",
        seed=7,
        add_random_initial_phase=True,
    )
    out = compute_phasors(cfg)

    avg_num = time_average_intensity_numeric(out, exclude_endpoint=True)
    avg_an = time_average_intensity_analytic(cfg)

    # With discrete sampling over a full period, this should match to numerical precision.
    assert np.isfinite(avg_num)
    assert np.isfinite(avg_an)
    assert np.allclose(avg_num, avg_an, rtol=0.0, atol=5e-10)
