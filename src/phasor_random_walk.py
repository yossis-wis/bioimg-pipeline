from __future__ import annotations

r"""2D random-walk (phasor-sum) toy model.

This module provides a minimal, reproducible way to connect two equivalent pictures:

- A **2D random walk** in the complex plane (head-to-tail vector addition).
- A **phasor sum**: :math:`E = \sum_n a_n \exp(i\phi_n)`.

It is intentionally lightweight (NumPy only) so it can be reused in notebooks, scripts,
and unit tests throughout the repo.

Conventions
-----------
- A step is represented as a complex number ``z = x + i y``.
- The walk starts at the origin.
- "Intensity" is defined as ``I = |E|^2`` where ``E`` is the endpoint (phasor sum).

The key "obvious knobs" exposed here are:

- number of steps ``n_steps``
- amplitude model (equal, random, one-dominant)
- phase model (uniform, biased/von-Mises, correlated/persistent)
- whether to normalize total power ``sum |a_n|^2`` for comparability across models

"""  # noqa: D401

from dataclasses import dataclass
from typing import Literal

import numpy as np


AmplitudeModel = Literal["equal", "uniform", "lognormal", "one_dominant"]
PhaseModel = Literal["uniform", "vonmises", "correlated_gaussian"]


_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class RandomWalk2D:
    """One 2D random walk / phasor-sum realization."""

    amplitudes: np.ndarray  # shape (n_steps,)
    phases_rad: np.ndarray  # shape (n_steps,)
    steps: np.ndarray  # complex, shape (n_steps,)
    path: np.ndarray  # complex, shape (n_steps+1,) including origin at index 0
    endpoint: complex
    intensity: float


def normalize_power(amplitudes: np.ndarray, *, total_power: float = 1.0) -> np.ndarray:
    """Scale amplitudes so that ``sum amplitudes**2 == total_power``."""

    a = np.asarray(amplitudes, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError("amplitudes must be 1D")
    if np.any(a < 0):
        raise ValueError("amplitudes must be >= 0")
    if total_power <= 0:
        raise ValueError("total_power must be > 0")

    p = float(np.sum(a * a))
    if p == 0:
        raise ValueError("cannot normalize: total power is zero")
    return (a * np.sqrt(float(total_power) / p)).astype(np.float64, copy=False)


def sample_amplitudes(
    n_steps: int,
    *,
    model: AmplitudeModel = "equal",
    rng: np.random.Generator | None = None,
    uniform_low: float = 0.5,
    uniform_high: float = 1.5,
    lognormal_mu: float = 0.0,
    lognormal_sigma: float = 0.5,
    dominant_ratio: float = 10.0,
    dominant_index: int | None = None,
    normalize_total_power: bool = True,
    total_power: float = 1.0,
) -> np.ndarray:
    """Sample nonnegative step amplitudes.

    Parameters
    ----------
    n_steps:
        Number of steps/phasers.
    model:
        - ``'equal'``: all steps have the same amplitude.
        - ``'uniform'``: amplitudes are uniform on ``[uniform_low, uniform_high]``.
        - ``'lognormal'``: log-normal amplitudes with ``(mu, sigma)`` in log-space.
        - ``'one_dominant'``: one step is ``dominant_ratio`` times the others.
    normalize_total_power:
        If True, scale amplitudes so that ``sum |a_n|^2 == total_power``.
        This makes comparisons across different amplitude models more meaningful.
    """

    if n_steps <= 0:
        raise ValueError("n_steps must be > 0")

    r = np.random.default_rng() if rng is None else rng

    m = model.lower().strip()
    if m == "equal":
        a = np.ones(int(n_steps), dtype=np.float64)
    elif m == "uniform":
        if uniform_high <= uniform_low:
            raise ValueError("uniform_high must be > uniform_low")
        a = r.uniform(float(uniform_low), float(uniform_high), size=int(n_steps)).astype(np.float64)
    elif m == "lognormal":
        if lognormal_sigma < 0:
            raise ValueError("lognormal_sigma must be >= 0")
        a = r.lognormal(mean=float(lognormal_mu), sigma=float(lognormal_sigma), size=int(n_steps)).astype(
            np.float64
        )
    elif m == "one_dominant":
        if dominant_ratio <= 0:
            raise ValueError("dominant_ratio must be > 0")
        a = np.ones(int(n_steps), dtype=np.float64)
        idx = int(dominant_index) if dominant_index is not None else int(r.integers(0, int(n_steps)))
        if idx < 0 or idx >= int(n_steps):
            raise ValueError("dominant_index out of range")
        a[idx] = float(dominant_ratio)
    else:
        raise ValueError(f"unknown amplitude model: {model!r}")

    if normalize_total_power:
        a = normalize_power(a, total_power=total_power)

    return a


def wrap_phase_rad(phi: np.ndarray) -> np.ndarray:
    """Wrap angles to ``[0, 2π)``."""

    return np.mod(phi, _TWO_PI)


def sample_phases_rad(
    n_steps: int,
    *,
    model: PhaseModel = "uniform",
    rng: np.random.Generator | None = None,
    mu_rad: float = 0.0,
    kappa: float = 0.0,
    step_sigma_rad: float = 0.35,
    phi0_rad: float | None = None,
) -> np.ndarray:
    """Sample step phases/angles in radians.

    Parameters
    ----------
    model:
        - ``'uniform'``: i.i.d. uniform on ``[0, 2π)``.
        - ``'vonmises'``: i.i.d. von Mises with mean ``mu_rad`` and concentration ``kappa``.
          ``kappa=0`` is uniform on the circle.
        - ``'correlated_gaussian'``: "persistent" walk.
          Draw ``phi_0`` then apply Gaussian increments:
          ``phi_n = phi_{n-1} + Normal(0, step_sigma_rad)``.
    """

    if n_steps <= 0:
        raise ValueError("n_steps must be > 0")

    r = np.random.default_rng() if rng is None else rng
    m = model.lower().strip()

    if m == "uniform":
        phi = r.uniform(0.0, _TWO_PI, size=int(n_steps)).astype(np.float64)
        return phi

    if m == "vonmises":
        if kappa < 0:
            raise ValueError("kappa must be >= 0")
        # numpy's vonmises is centered on mu in [-π, π], but we wrap afterward.
        phi = r.vonmises(mu=float(mu_rad), kappa=float(kappa), size=int(n_steps)).astype(np.float64)
        return wrap_phase_rad(phi)

    if m == "correlated_gaussian":
        if step_sigma_rad < 0:
            raise ValueError("step_sigma_rad must be >= 0")
        phi = np.empty(int(n_steps), dtype=np.float64)
        phi[0] = float(phi0_rad) if phi0_rad is not None else float(r.uniform(0.0, _TWO_PI))
        if int(n_steps) > 1:
            dphi = r.normal(loc=0.0, scale=float(step_sigma_rad), size=int(n_steps) - 1).astype(np.float64)
            phi[1:] = phi[0] + np.cumsum(dphi)
        return wrap_phase_rad(phi)

    raise ValueError(f"unknown phase model: {model!r}")


def phasors_from_amplitudes_phases(
    amplitudes: np.ndarray,
    phases_rad: np.ndarray,
) -> np.ndarray:
    """Return complex steps ``a_n * exp(i*phi_n)``."""

    a = np.asarray(amplitudes, dtype=np.float64)
    phi = np.asarray(phases_rad, dtype=np.float64)
    if a.shape != phi.shape:
        raise ValueError("amplitudes and phases_rad must have the same shape")
    if a.ndim != 1:
        raise ValueError("amplitudes and phases_rad must be 1D")
    if np.any(a < 0):
        raise ValueError("amplitudes must be >= 0")

    return (a * np.exp(1j * phi)).astype(np.complex128, copy=False)


def partial_sums(steps: np.ndarray) -> np.ndarray:
    """Return head-to-tail partial sums, including the origin at index 0."""

    z = np.asarray(steps)
    if z.ndim != 1:
        raise ValueError("steps must be 1D")
    out = np.zeros(z.shape[0] + 1, dtype=np.complex128)
    out[1:] = np.cumsum(z.astype(np.complex128, copy=False))
    return out


def intensity_from_field(e: np.ndarray | complex) -> np.ndarray | float:
    """Return intensity ``|E|^2`` (real-valued)."""

    if isinstance(e, complex):
        return float(e.real * e.real + e.imag * e.imag)
    u = np.asarray(e)
    return (u.real * u.real + u.imag * u.imag).astype(np.float64, copy=False)


def speckle_contrast_1d(intensity_samples: np.ndarray) -> float:
    """Compute contrast ``C = std/mean`` for 1D intensity samples."""

    x = np.asarray(intensity_samples, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("intensity_samples must be 1D")
    if x.size == 0:
        raise ValueError("intensity_samples is empty")
    mu = float(np.mean(x))
    if mu == 0:
        raise ValueError("mean intensity is zero")
    return float(np.std(x) / mu)


def simulate_walk(
    *,
    n_steps: int,
    amplitude_model: AmplitudeModel = "equal",
    phase_model: PhaseModel = "uniform",
    seed: int | None = 0,
    normalize_total_power: bool = True,
    total_power: float = 1.0,
    # amplitude params
    uniform_low: float = 0.5,
    uniform_high: float = 1.5,
    lognormal_mu: float = 0.0,
    lognormal_sigma: float = 0.5,
    dominant_ratio: float = 10.0,
    dominant_index: int | None = None,
    # phase params
    mu_rad: float = 0.0,
    kappa: float = 0.0,
    step_sigma_rad: float = 0.35,
    phi0_rad: float | None = None,
) -> RandomWalk2D:
    """Simulate one random-walk realization."""

    rng = np.random.default_rng(seed)
    amps = sample_amplitudes(
        n_steps,
        model=amplitude_model,
        rng=rng,
        uniform_low=uniform_low,
        uniform_high=uniform_high,
        lognormal_mu=lognormal_mu,
        lognormal_sigma=lognormal_sigma,
        dominant_ratio=dominant_ratio,
        dominant_index=dominant_index,
        normalize_total_power=normalize_total_power,
        total_power=total_power,
    )
    phases = sample_phases_rad(
        n_steps,
        model=phase_model,
        rng=rng,
        mu_rad=mu_rad,
        kappa=kappa,
        step_sigma_rad=step_sigma_rad,
        phi0_rad=phi0_rad,
    )
    steps = phasors_from_amplitudes_phases(amps, phases)
    path = partial_sums(steps)
    endpoint = complex(path[-1])
    intensity = float(intensity_from_field(endpoint))
    return RandomWalk2D(
        amplitudes=amps,
        phases_rad=phases,
        steps=steps,
        path=path,
        endpoint=endpoint,
        intensity=intensity,
    )


def simulate_ensemble(
    *,
    n_steps: int,
    n_realizations: int,
    amplitude_model: AmplitudeModel = "equal",
    phase_model: PhaseModel = "uniform",
    seed: int | None = 0,
    normalize_total_power: bool = True,
    total_power: float = 1.0,
    # amplitude params
    uniform_low: float = 0.5,
    uniform_high: float = 1.5,
    lognormal_mu: float = 0.0,
    lognormal_sigma: float = 0.5,
    dominant_ratio: float = 10.0,
    # phase params
    mu_rad: float = 0.0,
    kappa: float = 0.0,
    step_sigma_rad: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate an ensemble and return ``(endpoints, intensities)``.

    Returns
    -------
    endpoints:
        Complex array with shape ``(n_realizations,)``.
    intensities:
        Float array with shape ``(n_realizations,)``.
    """

    if n_realizations <= 0:
        raise ValueError("n_realizations must be > 0")

    rng = np.random.default_rng(seed)
    endpoints = np.empty(int(n_realizations), dtype=np.complex128)

    for i in range(int(n_realizations)):
        # Derive an independent seed for each realization to keep the loop stable.
        s_i = int(rng.integers(0, 2**31 - 1))
        w = simulate_walk(
            n_steps=n_steps,
            amplitude_model=amplitude_model,
            phase_model=phase_model,
            seed=s_i,
            normalize_total_power=normalize_total_power,
            total_power=total_power,
            uniform_low=uniform_low,
            uniform_high=uniform_high,
            lognormal_mu=lognormal_mu,
            lognormal_sigma=lognormal_sigma,
            dominant_ratio=dominant_ratio,
            mu_rad=mu_rad,
            kappa=kappa,
            step_sigma_rad=step_sigma_rad,
        )
        endpoints[i] = w.endpoint

    intensities = intensity_from_field(endpoints).astype(np.float64, copy=False)
    return endpoints, intensities


def average_over_k_uncorrelated(
    intensity_samples: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    """Return averaged intensities from groups of ``k`` independent samples.

    This helper is useful for toy "averaging" experiments:

    - Generate ``M*k`` independent intensity samples.
    - Call this function to get ``M`` averaged values.

    The caller is responsible for choosing ``M`` so ``len(samples)`` is a multiple of ``k``.
    """

    x = np.asarray(intensity_samples, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("intensity_samples must be 1D")
    if k <= 0:
        raise ValueError("k must be > 0")
    if x.size % int(k) != 0:
        raise ValueError("len(intensity_samples) must be divisible by k")

    m = x.size // int(k)
    return x.reshape(m, int(k)).mean(axis=1)
