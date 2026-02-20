from __future__ import annotations

import numpy as np
import pytest

from src.phasor_random_walk import (
    average_over_k_uncorrelated,
    intensity_from_field,
    simulate_ensemble,
    simulate_walk,
)


def test_simulate_walk_shapes_and_invariants() -> None:
    w = simulate_walk(n_steps=10, seed=0)
    assert w.amplitudes.shape == (10,)
    assert w.phases_rad.shape == (10,)
    assert w.steps.shape == (10,)
    assert w.path.shape == (11,)

    # endpoint equals sum of steps
    assert pytest.approx(w.endpoint.real, rel=1e-12) == float(np.sum(w.steps).real)
    assert pytest.approx(w.endpoint.imag, rel=1e-12) == float(np.sum(w.steps).imag)

    # intensity is |E|^2
    assert pytest.approx(w.intensity, rel=1e-12) == float(intensity_from_field(w.endpoint))


def test_power_normalization_default_is_unit_power() -> None:
    # Default amplitude model normalizes sum |a_n|^2 to 1.
    w = simulate_walk(n_steps=37, seed=1)
    p = float(np.sum(w.amplitudes * w.amplitudes))
    assert pytest.approx(1.0, rel=1e-12) == p


def test_ensemble_mean_intensity_matches_sum_power() -> None:
    # For random phases with normalized power sum|a|^2=1, E[|E|^2] ≈ 1.
    _, I = simulate_ensemble(n_steps=60, n_realizations=8000, seed=0)
    mu = float(np.mean(I))
    assert 0.95 < mu < 1.05


def test_average_over_k_uncorrelated_groups_samples() -> None:
    x = np.arange(12, dtype=float)
    y = average_over_k_uncorrelated(x, k=3)
    assert y.shape == (4,)
    assert np.allclose(y, np.array([1.0, 4.0, 7.0, 10.0], dtype=float))

    with pytest.raises(ValueError):
        _ = average_over_k_uncorrelated(x, k=5)
