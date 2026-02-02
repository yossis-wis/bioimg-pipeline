"""Sweep MMF widefield 500 µs illumination assumptions.

This script reads a YAML config (see `configs/illumination_mmf_500us.yaml`) and
prints derived design numbers:

- sample power and implied fiber-exit power
- field stop size
- objective pupil and pupil-fill derived NA_illum
- speckle averaging N_eff and predicted contrast

It does **not** write outputs to disk.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml


def find_repo_root(start: Path) -> Path:
    """Find repo root by walking upward until we see (src/, environment.yml)."""

    p = start.resolve()
    for parent in [p, *p.parents]:
        if (parent / "src").is_dir() and (parent / "environment.yml").exists():
            return parent
    return p


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/illumination_mmf_500us.yaml")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Also run a tiny Fourier-optics speckle simulation and print the measured contrast.",
    )
    args = parser.parse_args()

    repo_root = find_repo_root(Path.cwd())
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Repo modules (import after sys.path adjustment so the script can be run
    # from arbitrary working directories).
    from src.excitation_speckle_sim import simulate_excitation_speckle_field  # noqa: WPS433
    from src.illumination_design_params import (  # noqa: WPS433
        PowerBudget,
        bfp_beam_diameter_mm,
        collimated_beam_diameter_mm,
        field_stop_size_mm,
        illumination_na,
        objective_focal_length_mm,
        objective_pupil_diameter_mm,
        required_fiber_exit_power_mw,
        required_sample_power_mw,
        speckle_grain_size_um,
    )
    from src.speckle_diversity_models import (  # noqa: WPS433
        DiversityBudget,
        estimate_n_eff,
        estimate_n_lambda,
        estimate_speckle_spectral_corr_width_nm,
        required_n_eff_for_contrast,
        speckle_contrast_from_n_eff,
    )

    cfg = _load_yaml(repo_root / args.config)

    target = cfg["target"]
    microscope = cfg["microscope"]
    fiber = cfg["fiber"]
    budget_cfg = cfg["power_budget"]
    speckle = cfg["speckle_averaging"]
    targets = cfg["targets"]

    lambda_nm = float(target["wavelength_nm"])
    roi_um = tuple(target["roi_um"])
    exposure_s = float(target["exposure_us"]) * 1e-6
    E = float(target["irradiance_kw_per_cm2"])

    # ---- power + stop sizing ----
    p_sample_mw = required_sample_power_mw(E, roi_um)
    stop_mm = field_stop_size_mm(roi_um, sample_to_stop_magnification=float(microscope["objective_magnification"]))

    budget = PowerBudget(**budget_cfg)
    T = budget.total_throughput()
    p_fiber_mw = required_fiber_exit_power_mw(p_sample_mw, throughput=T)

    print("=== Target ===")
    print(f"λ = {lambda_nm:.1f} nm, exposure = {exposure_s*1e6:.0f} µs, ROI = {roi_um[0]:.1f}×{roi_um[1]:.1f} µm")
    print(f"E = {E:.1f} kW/cm² -> P_sample ≈ {p_sample_mw:.2f} mW")
    print("")
    print("=== Field stop ===")
    print(f"sample→stop magnification ≈ {microscope['objective_magnification']}×")
    print(f"stop ≈ {stop_mm[0]:.3f} × {stop_mm[1]:.3f} mm")
    print("")
    print("=== Power budget ===")
    print(f"T_total ≈ {T:.3f} (from {asdict(budget)})")
    print(f"P_fiber_exit ≈ {p_fiber_mw:.1f} mW")

    # ---- pupil fill + speckle grain ----
    f_obj_mm = objective_focal_length_mm(
        tube_lens_f_mm=float(microscope["tube_lens_f_mm"]), magnification=float(microscope["objective_magnification"])
    )
    d_pupil_mm = objective_pupil_diameter_mm(na_obj=float(microscope["objective_na"]), f_obj_mm=f_obj_mm)

    pupil_fill = float(microscope["pupil_fill_fraction"])
    d_bfp_mm = bfp_beam_diameter_mm(pupil_fill_fraction=pupil_fill, pupil_diameter_mm=d_pupil_mm)
    na_illum = illumination_na(na_obj=float(microscope["objective_na"]), pupil_fill_fraction=pupil_fill)
    grain_um = speckle_grain_size_um(lambda_nm=lambda_nm, na_illum=na_illum)

    print("")
    print("=== Pupil / NA_illum ===")
    print(f"objective: NA_obj={microscope['objective_na']}, M={microscope['objective_magnification']}, tube f={microscope['tube_lens_f_mm']} mm")
    print(f"f_obj ≈ {f_obj_mm:.3f} mm, D_pupil ≈ {d_pupil_mm:.2f} mm")
    print(f"fill fraction ρ={pupil_fill:.2f} -> D_beam@BFP ≈ {d_bfp_mm:.2f} mm, NA_illum ≈ {na_illum:.3f}")
    print(f"speckle grain ~ λ/(2 NA_illum) ≈ {grain_um:.2f} µm")

    # ---- collimator size ----
    d_coll_mm = collimated_beam_diameter_mm(f_coll_mm=float(fiber["f_coll_mm"]), fiber_na=float(fiber["na"]))
    print("")
    print("=== Fiber collimator ===")
    print(f"NA_fiber={fiber['na']}, f_coll={fiber['f_coll_mm']} mm -> D_coll ≈ {d_coll_mm:.2f} mm")
    print(f"to reach D_beam@BFP ≈ {d_bfp_mm:.2f} mm: telescope ratio ≈ {d_bfp_mm/d_coll_mm:.2f}×")

    # ---- speckle averaging prediction ----
    dlam_c = estimate_speckle_spectral_corr_width_nm(
        lambda0_nm=lambda_nm, optical_path_spread_m=float(fiber["optical_path_spread_m"])
    )
    n_lambda = estimate_n_lambda(
        source_span_nm=float(speckle["source_span_nm"]),
        speckle_corr_width_nm=dlam_c,
        n_lines=int(speckle.get("n_lines", 1)),
    )

    div = DiversityBudget(n_lambda=n_lambda, n_pol=int(speckle["n_pol"]), n_angle=int(speckle["n_angle"]))
    n_eff = estimate_n_eff(
        exposure_s=exposure_s,
        scrambler_hz=float(speckle["scrambler_hz"]),
        diversity=div,
        successive_pattern_correlation=float(speckle["successive_pattern_correlation"]),
    )
    c_pred = speckle_contrast_from_n_eff(n_eff)
    n_eff_need = required_n_eff_for_contrast(float(targets["speckle_contrast_target"]))

    print("")
    print("=== Speckle averaging estimate ===")
    print(f"Δλ_c ~ {dlam_c:.4f} nm (from OPL spread {fiber['optical_path_spread_m']} m)")
    print(f"source span ~ {speckle['source_span_nm']} nm -> N_lambda ≈ {n_lambda}")
    print(f"diversity: N_pol={speckle['n_pol']}, N_angle={speckle['n_angle']} -> N_src={div.n_source_states()}")
    print(f"N_eff ≈ {n_eff:.1f} -> predicted contrast C ≈ {c_pred:.3f}")
    print(f"target: C ≤ {targets['speckle_contrast_target']} -> N_eff ≥ {n_eff_need}")

    if not args.simulate:
        return 0

    # ---- optional: small Fourier-optics simulation ----
    sim = cfg["simulation"]
    M = float(microscope["objective_magnification"])
    dx_um = float(microscope["camera_pixel_um"]) / M
    I, _ = simulate_excitation_speckle_field(
        n=int(sim["n_grid"]),
        dx_um=dx_um,
        roi_um=float(roi_um[0]),  # kernel currently assumes square ROI
        lambda_um=lambda_nm * 1e-3,
        na_illum=na_illum,
        exposure_s=exposure_s,
        scrambler_hz=float(speckle["scrambler_hz"]),
        n_src=div.n_source_states(),
        seed=int(sim["seed"]),
    )

    # measure contrast in a central patch (avoid edges)
    n = I.shape[0]
    center = n // 2
    half = int((roi_um[0] / dx_um) * 0.35)
    patch = I[center - half : center + half, center - half : center + half]
    c_meas = float(np.std(patch) / np.mean(patch))
    print("")
    print("=== Simulation (idealized) ===")
    print(f"Measured contrast in central patch: C_meas ≈ {c_meas:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
