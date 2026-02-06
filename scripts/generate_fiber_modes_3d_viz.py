#!/usr/bin/env python
"""Generate interactive 3D HTML visualizations for fiber modes + multimode speckle.

This script writes **standalone HTML** files (Plotly) to an output directory.
The HTML files are meant to be opened in a browser and rotated/zoomed.

Why this exists
---------------
The repo's MMF illumination discussions can get abstract fast ("modes", "modal
noise", "speckle decorrelation"). These interactive 3D visuals are designed to
build intuition:

1) What does a "mode" look like in a fiber core cross-section?
2) Why does superposition of many modes create speckle-like intensity?
3) Why can the pattern drift dramatically from tiny phase perturbations?
4) Why does averaging independent realizations reduce speckle contrast?

Outputs
-------
- mode_gallery_3d.html
- speckle_drift_3d.html
- speckle_averaging_3d.html

Notes
-----
- The "modes" shown here are a clean surrogate basis on a disk (Bessel modes).
  They are not an exact LP mode solver.
- For dispersion/linewidth scaling estimates, use :mod:`src.mmf_fiber_speckle`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fiber_modes import (  # noqa: E402
    average_uncorrelated_intensities,
    disk_mode_indices,
    intensity_from_field,
    make_core_grid,
    precompute_mode_stack,
    random_complex_coeffs,
    speckle_contrast,
    superpose_modes,
)
from src.fiber_modes_plotly import (  # noqa: E402
    make_surface_stack,
    surface_stack_figure,
    write_html,
)


def _nan_outside(mask: np.ndarray, z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return np.where(mask, z, np.nan)


def build_mode_gallery(
    *,
    out_dir: Path,
    n_grid: int,
    core_radius_um: float,
    max_l: int,
    max_m: int,
) -> Path:
    x_um, y_um, mask, _ = make_core_grid(n=n_grid, core_radius_um=core_radius_um)
    modes = disk_mode_indices(max_l=max_l, max_m=max_m, include_sin=True)

    z_list = []
    labels = []
    abs_max = 0.0
    for mode in modes:
        u = precompute_mode_stack([mode], x_um=x_um, y_um=y_um, core_radius_um=core_radius_um, mask=mask)[0]
        u = _nan_outside(mask, u)
        abs_max = max(abs_max, float(np.nanmax(np.abs(u))))
        z_list.append(u)
        labels.append(mode.label)

    # Symmetric color scaling for +/- amplitude.
    color_range = (-abs_max, abs_max) if abs_max > 0 else None

    surf = make_surface_stack(x_um=x_um, y_um=y_um, z_list=z_list, labels=labels)
    fig = surface_stack_figure(
        surf,
        title="Fiber-core surrogate modes (disk-Bessel basis) — rotate + use slider",
        z_title="Mode amplitude (a.u.)",
        colorscale="RdBu",
        show_colorbar=False,
        z_range=color_range,
        aspectmode="data",
    )

    out_path = out_dir / "mode_gallery_3d.html"
    write_html(fig, out_path, auto_open=False)
    return out_path


def build_speckle_drift_demo(
    *,
    out_dir: Path,
    n_grid: int,
    core_radius_um: float,
    n_modes: int,
    seed: int,
) -> Path:
    x_um, y_um, mask, _ = make_core_grid(n=n_grid, core_radius_um=core_radius_um)
    modes_all = disk_mode_indices(max_l=6, max_m=6, include_sin=True)
    modes = modes_all[: int(n_modes)]
    stack = precompute_mode_stack(modes, x_um=x_um, y_um=y_um, core_radius_um=core_radius_um, mask=mask)

    base_coeffs = random_complex_coeffs(stack.shape[0], seed=seed)
    u0 = superpose_modes(stack, base_coeffs)
    I0 = intensity_from_field(u0)
    I0n = I0 / float(np.mean(I0[mask]))

    # Use a *fixed* random perturbation direction in coefficient phase-space so the
    # slider feels continuous.
    rng = np.random.default_rng(int(seed) + 1)
    delta_unit = rng.normal(loc=0.0, scale=1.0, size=stack.shape[0])

    delta_list = [0.0, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60]

    z_list = []
    labels = []
    for delta_rms in delta_list:
        coeffs = base_coeffs * np.exp(1j * (delta_unit * float(delta_rms)))
        u = superpose_modes(stack, coeffs)
        I = intensity_from_field(u)
        In = I / float(np.mean(I[mask]))

        # Pattern correlation with baseline (on intensity, inside core).
        a = I0n[mask].ravel()
        b = In[mask].ravel()
        corr = float(np.corrcoef(a, b)[0, 1])

        C = speckle_contrast(In, mask)

        z_list.append(_nan_outside(mask, In))
        labels.append(f"σϕ={delta_rms:.2f} rad  (corr={corr:.2f}, C={C:.2f})")

    surf = make_surface_stack(x_um=x_um, y_um=y_um, z_list=z_list, labels=labels)
    fig = surface_stack_figure(
        surf,
        title="MMF speckle drift: tiny per-mode phase noise changes the intensity pattern",
        z_title="Normalized intensity (mean=1)",
        colorscale="Viridis",
        show_colorbar=False,
        z_range=None,
        aspectmode="data",
    )

    out_path = out_dir / "speckle_drift_3d.html"
    write_html(fig, out_path, auto_open=False)
    return out_path


def build_speckle_averaging_demo(
    *,
    out_dir: Path,
    n_grid: int,
    core_radius_um: float,
    n_modes: int,
    seed: int,
) -> Path:
    x_um, y_um, mask, _ = make_core_grid(n=n_grid, core_radius_um=core_radius_um)
    modes_all = disk_mode_indices(max_l=6, max_m=6, include_sin=True)
    modes = modes_all[: int(n_modes)]
    stack = precompute_mode_stack(modes, x_um=x_um, y_um=y_um, core_radius_um=core_radius_um, mask=mask)

    n_avg_list = [1, 2, 4, 8, 16, 32, 64]

    z_list = []
    labels = []

    for n_avg in n_avg_list:
        I = average_uncorrelated_intensities(stack, n_avg=int(n_avg), seed=seed)
        In = I / float(np.mean(I[mask]))
        C = speckle_contrast(In, mask)
        z_list.append(_nan_outside(mask, In))
        labels.append(f"N={n_avg}  (C={C:.2f}, 1/√N={1/np.sqrt(n_avg):.2f})")

    surf = make_surface_stack(x_um=x_um, y_um=y_um, z_list=z_list, labels=labels)
    fig = surface_stack_figure(
        surf,
        title="Speckle averaging: incoherent average of N independent patterns reduces contrast",
        z_title="Normalized intensity (mean=1)",
        colorscale="Viridis",
        show_colorbar=False,
        z_range=None,
        aspectmode="data",
    )

    out_path = out_dir / "speckle_averaging_3d.html"
    write_html(fig, out_path, auto_open=False)
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=str, default="reports/fiber_modes_3d", help="Output directory for HTML files")
    p.add_argument("--n-grid", type=int, default=201, help="Grid size (n×n) for surfaces")
    p.add_argument("--core-radius-um", type=float, default=200.0, help="Fiber core radius in µm (e.g. 200 for 400 µm core)")
    p.add_argument("--gallery-max-l", type=int, default=3, help="Max azimuthal order ℓ for mode gallery")
    p.add_argument("--gallery-max-m", type=int, default=2, help="Max radial order m for mode gallery")
    p.add_argument("--n-speckle-modes", type=int, default=30, help="How many modes to superpose for speckle demos")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility")

    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_path = build_mode_gallery(
        out_dir=out_dir,
        n_grid=int(args.n_grid),
        core_radius_um=float(args.core_radius_um),
        max_l=int(args.gallery_max_l),
        max_m=int(args.gallery_max_m),
    )
    drift_path = build_speckle_drift_demo(
        out_dir=out_dir,
        n_grid=int(args.n_grid),
        core_radius_um=float(args.core_radius_um),
        n_modes=int(args.n_speckle_modes),
        seed=int(args.seed),
    )
    avg_path = build_speckle_averaging_demo(
        out_dir=out_dir,
        n_grid=int(args.n_grid),
        core_radius_um=float(args.core_radius_um),
        n_modes=int(args.n_speckle_modes),
        seed=int(args.seed),
    )

    print("Wrote:")
    print(f"  - {mode_path}")
    print(f"  - {drift_path}")
    print(f"  - {avg_path}")
    print()
    print("Tip: open these HTML files in a browser for full 3D interaction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
