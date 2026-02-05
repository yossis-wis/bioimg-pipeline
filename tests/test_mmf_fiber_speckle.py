from __future__ import annotations

import math

from src.mmf_fiber_speckle import (
    MultimodeFiber,
    approx_num_guided_modes_step_index,
    intermodal_group_delay_spread_s,
    optical_path_length_m,
    optical_path_spread_geometric_m,
    optical_path_spread_m,
    required_fiber_length_m_for_target_n_lambda,
    speckle_spectral_corr_width_nm_for_fiber,
    v_number,
)


def test_v_number_and_mode_count_400um_na022_640nm() -> None:
    # 400 µm core diameter -> 200 µm radius.
    v = v_number(core_radius_um=200.0, na=0.22, lambda_um=0.640)
    # Rough sanity: V should be a few hundred.
    assert 350 < v < 550

    m = approx_num_guided_modes_step_index(v)
    # Order-of-magnitude: ~V^2/2 ~ 1e5.
    assert 50_000 < m < 200_000


def test_intermodal_delay_and_corr_width_step_index_3m() -> None:
    fiber = MultimodeFiber(core_diameter_um=400.0, na=0.22, length_m=3.0, n_core=1.46, modal_delay_scale=1.0)

    dt = intermodal_group_delay_spread_s(fiber)
    # Expect ~ O(1e-10 s) for a few-meter, NA~0.22 step-index MMF.
    assert 1e-11 < dt < 1e-9

    dlam_c = speckle_spectral_corr_width_nm_for_fiber(lambda0_nm=640.0, fiber=fiber)
    # Step-index estimate gives ~0.008 nm at 640 nm for 3 m (order-of-magnitude check).
    assert math.isclose(dlam_c, 0.0082, rel_tol=0.35, abs_tol=0.0)


def test_required_length_for_target_n_lambda_scales_reasonably() -> None:
    # If we want N_lambda ~ 100 from a 2 nm-wide source at 640 nm, NA=0.22,
    # step-index-like dispersion, we should not need kilometers of fiber.
    L = required_fiber_length_m_for_target_n_lambda(
        lambda0_nm=640.0,
        na=0.22,
        n_core=1.46,
        source_span_nm=2.0,
        target_n_lambda=100,
        modal_delay_scale=1.0,
    )
    assert 0.1 < L < 10.0

    # If modal dispersion is 10× smaller (strong GI), required length should be 10× larger.
    L_gi = required_fiber_length_m_for_target_n_lambda(
        lambda0_nm=640.0,
        na=0.22,
        n_core=1.46,
        source_span_nm=2.0,
        target_n_lambda=100,
        modal_delay_scale=0.1,
    )
    assert math.isclose(L_gi / L, 10.0, rel_tol=0.15, abs_tol=0.0)


def test_optical_path_length_and_geometric_spread_consistency() -> None:
    fiber = MultimodeFiber(core_diameter_um=400.0, na=0.22, length_m=3.0, n_core=1.46, modal_delay_scale=1.0)

    opl = optical_path_length_m(fiber)
    # OPL = n * L -> ~4.38 m
    assert math.isclose(opl, 4.38, rel_tol=0, abs_tol=1e-12)

    # Geometric and group-delay based ΔOPL should agree within a few percent for small angles.
    d_opl_geom = optical_path_spread_geometric_m(fiber)
    d_opl_delay = optical_path_spread_m(fiber)

    # Both should be ~5 cm for these parameters.
    assert 0.03 < d_opl_geom < 0.08
    assert 0.03 < d_opl_delay < 0.08

    assert math.isclose(d_opl_geom, d_opl_delay, rel_tol=0.05, abs_tol=0.0)
