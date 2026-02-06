from __future__ import annotations

import numpy as np

from src.fiber_modes import (
    average_uncorrelated_intensities,
    disk_bessel_mode_field,
    disk_mode_indices,
    intensity_from_field,
    make_core_grid,
    precompute_mode_stack,
    random_complex_coeffs,
    speckle_contrast,
    superpose_modes,
)


def test_disk_mode_indices_count_and_sorting() -> None:
    modes = disk_mode_indices(max_l=2, max_m=2, include_sin=True)

    # l=0: m=1..2 -> 2 modes (cos only)
    # l=1: m=1..2 -> 4 modes (cos+sin)
    # l=2: m=1..2 -> 4 modes (cos+sin)
    assert len(modes) == 10

    # Should be sorted by increasing alpha (Bessel zero)
    alphas = [m.alpha for m in modes]
    assert alphas == sorted(alphas)


def test_mode_field_is_masked_and_normalized() -> None:
    x_um, y_um, mask, _ = make_core_grid(n=101, core_radius_um=1.0)
    modes = disk_mode_indices(max_l=1, max_m=1, include_sin=True)
    u = disk_bessel_mode_field(modes[0], x_um=x_um, y_um=y_um, core_radius_um=1.0, mask=mask, normalize=True)

    assert u.shape == x_um.shape
    assert np.all(u[~mask] == 0.0)

    # Discrete L2 norm inside mask should be ~1 (by construction).
    norm = float(np.sqrt(np.sum(u[mask] * u[mask])))
    assert np.isclose(norm, 1.0, rtol=1e-3, atol=0.0)


def test_speckle_contrast_drops_with_averaging() -> None:
    x_um, y_um, mask, _ = make_core_grid(n=101, core_radius_um=1.0)
    modes = disk_mode_indices(max_l=3, max_m=3, include_sin=True)[:25]
    stack = precompute_mode_stack(modes, x_um=x_um, y_um=y_um, core_radius_um=1.0, mask=mask)

    coeffs = random_complex_coeffs(stack.shape[0], seed=0)
    I1 = intensity_from_field(superpose_modes(stack, coeffs))
    I1n = I1 / float(np.mean(I1[mask]))
    C1 = speckle_contrast(I1n, mask)

    Iavg = average_uncorrelated_intensities(stack, n_avg=16, seed=0)
    Iavgn = Iavg / float(np.mean(Iavg[mask]))
    Cavg = speckle_contrast(Iavgn, mask)

    # Averaging should reduce contrast (for this fixed seed and mode set).
    assert Cavg < C1
