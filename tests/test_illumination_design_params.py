from __future__ import annotations

import math

from src.illumination_design_params import (
    PowerBudget,
    bfp_beam_diameter_mm,
    collimated_beam_diameter_mm,
    field_stop_size_mm,
    objective_focal_length_mm,
    objective_pupil_diameter_mm,
    required_fiber_exit_power_mw,
    required_sample_power_mw,
    roi_area_cm2,
)


def test_roi_area_and_power_scaling() -> None:
    # 10 µm × 10 µm = (1e-3 cm)^2 = 1e-6 cm^2
    area = roi_area_cm2((10.0, 10.0))
    assert math.isclose(area, 1e-6, rel_tol=0, abs_tol=1e-12)

    # 10 kW/cm^2 over 1e-6 cm^2 -> 10 mW
    p_mw = required_sample_power_mw(irradiance_kw_per_cm2=10.0, roi_um=(10.0, 10.0))
    assert math.isclose(p_mw, 10.0, rel_tol=0, abs_tol=1e-9)


def test_field_stop_size_matches_magnification() -> None:
    stop_w_mm, stop_h_mm = field_stop_size_mm((10.0, 10.0), sample_to_stop_magnification=100.0)
    assert math.isclose(stop_w_mm, 1.0, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(stop_h_mm, 1.0, rel_tol=0, abs_tol=1e-12)


def test_pupil_diameter_and_fill() -> None:
    f_obj = objective_focal_length_mm(tube_lens_f_mm=200.0, magnification=100.0)
    assert math.isclose(f_obj, 2.0, rel_tol=0, abs_tol=1e-12)

    d_pupil = objective_pupil_diameter_mm(na_obj=1.45, f_obj_mm=f_obj)
    assert math.isclose(d_pupil, 5.8, rel_tol=0, abs_tol=1e-12)

    d_bfp = bfp_beam_diameter_mm(pupil_fill_fraction=0.3, pupil_diameter_mm=d_pupil)
    assert math.isclose(d_bfp, 1.74, rel_tol=0, abs_tol=1e-12)


def test_collimated_beam_diameter_and_power_budget() -> None:
    d_coll = collimated_beam_diameter_mm(f_coll_mm=3.0, fiber_na=0.22)
    # 2 f tan(asin(NA)) ~ 1.33 mm for 3 mm and NA=0.22
    assert math.isclose(d_coll, 1.33, rel_tol=0.03, abs_tol=0)

    budget = PowerBudget(
        coupling_into_fiber=0.5,
        fiber_to_collimator=0.9,
        stop_and_relays=0.8,
        objective_and_misc=0.75,
    )
    throughput = budget.total_throughput()
    assert math.isclose(throughput, 0.5 * 0.9 * 0.8 * 0.75, rel_tol=0, abs_tol=1e-12)

    p_fiber = required_fiber_exit_power_mw(sample_power_mw=20.0, throughput=throughput)
    assert math.isclose(p_fiber * throughput, 20.0, rel_tol=0, abs_tol=1e-12)
