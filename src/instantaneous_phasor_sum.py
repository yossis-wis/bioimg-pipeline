
from __future__ import annotations

r"""Instantaneous phasor-sum toy model (wavelength comb × discrete paths).

This module implements the small "instantaneous phasor deck" model used in:

- ``scripts/phasor_slidertype_animation.py`` (PNG frames + optional GIF)

We model an interferometer-like sum of complex phasors indexed by:

- wavelength index ``k`` (a frequency comb)
- path index ``p`` (fixed vacuum-equivalent delays)

At each time ``t`` we compute:

$$
E_{k,p}(t) = A\exp\{ i[2\pi(f_k-f_\mathrm{ref})t - 2\pi f_k\tau_p + \phi_{k,p}] \},
$$

then sum:

$$
E(t) = \sum_{k,p} E_{k,p}(t),\qquad I(t) = |E(t)|^2.
$$

Notes
-----
- The term ``(f_k - f_ref)`` is a *global envelope reference*; it changes the plotted
  rotation rate but does not change intensity.
- ``\tau_p`` is computed from the path-length offsets as ``\tau_p = \Delta L_p / c``.

This model is intentionally a "toy":

- It does not include polarization, bandwidth envelopes, or continuous path distributions.
- It is meant to support intuition-building and visualization.

"""  # noqa: D401

from dataclasses import dataclass
from typing import Literal

import numpy as np


C0_M_PER_S = 299_792_458.0


OrderMode = Literal["by_wavelength", "by_path", "random_fixed"]
RefMode = Literal["lowest", "below_lowest"]
PathAmpMode = Literal["equal_power", "unit"]


@dataclass(frozen=True)
class InstantaneousPhasorSumConfig:
    """Configuration for the instantaneous phasor-sum model."""

    lambda0_nm: float = 640.0
    n_wavelengths: int = 20
    # Choose Δf so that the adjacent-beat period is exactly T_ps.
    T_ps: float = 160.0
    dt_ps: float = 1.0
    deltaL_mm: tuple[float, ...] = (0.0, 25.0, 51.0)
    path_amp: PathAmpMode = "equal_power"
    ref: RefMode = "lowest"
    order: OrderMode = "by_wavelength"
    seed: int = 0
    add_random_initial_phase: bool = False


@dataclass(frozen=True)
class InstantaneousPhasorSumResult:
    """Computed phasors and derived quantities."""

    times_s: np.ndarray  # shape (n_times,)
    times_ps: np.ndarray  # shape (n_times,)
    f_hz: np.ndarray  # shape (n_wavelengths,)
    lambda_m: np.ndarray  # shape (n_wavelengths,)
    tau_s: np.ndarray  # shape (n_paths,)
    order_kp: np.ndarray  # int, shape (n_steps, 2) with columns (k, p)
    phasors: np.ndarray  # complex, shape (n_times, n_steps)
    field: np.ndarray  # complex, shape (n_times,)
    intensity: np.ndarray  # float, shape (n_times,)
    A_path: float
    f_ref_hz: float
    df_hz: float


def parse_deltaL_mm(s: str) -> tuple[float, ...]:
    """Parse a comma-separated list of path-length offsets in mm."""

    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("deltaL_mm must contain at least one value, e.g. '0,25,51'")
    return tuple(float(x) for x in parts)


def make_frequency_comb(
    *,
    lambda0_nm: float,
    n_wavelengths: int,
    T_ps: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(f_hz, lambda_m, df_hz)`` for an equally-spaced frequency comb.

    We define:

    - center line: ``f0 = c / lambda0``
    - comb: ``f_k = f0 + k Δf``
    - set ``Δf = 1/T`` so the adjacent-beat period is ``T``.

    This is convenient for deck-style intuition plots because it makes the time axis
    "line up" with the comb spacing.
    """

    if n_wavelengths <= 0:
        raise ValueError("n_wavelengths must be > 0")
    if lambda0_nm <= 0:
        raise ValueError("lambda0_nm must be > 0")
    if T_ps <= 0:
        raise ValueError("T_ps must be > 0")

    lambda0_m = float(lambda0_nm) * 1e-9
    f0 = C0_M_PER_S / lambda0_m
    T_s = float(T_ps) * 1e-12
    df = 1.0 / T_s

    k = np.arange(int(n_wavelengths), dtype=np.float64)
    f = (f0 + k * df).astype(np.float64)
    lam = (C0_M_PER_S / f).astype(np.float64)
    return f, lam, float(df)


def build_order_indices(
    *,
    n_wavelengths: int,
    n_paths: int,
    order: OrderMode,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return an array of shape ``(n_steps, 2)`` listing ``(k, p)`` pairs."""

    if n_wavelengths <= 0 or n_paths <= 0:
        raise ValueError("n_wavelengths and n_paths must be > 0")

    pairs: list[tuple[int, int]] = [(k, p) for k in range(int(n_wavelengths)) for p in range(int(n_paths))]

    o = order.lower().strip()
    if o == "by_wavelength":
        pass
    elif o == "by_path":
        pairs = [(k, p) for p in range(int(n_paths)) for k in range(int(n_wavelengths))]
    elif o == "random_fixed":
        rng.shuffle(pairs)
    else:
        raise ValueError(f"unknown order: {order!r}")

    return np.array(pairs, dtype=np.int64)


def _initial_phases(
    *,
    n_wavelengths: int,
    n_paths: int,
    rng: np.random.Generator,
    add_random_initial_phase: bool,
) -> np.ndarray:
    """Return per-(k,p) initial phases in radians.

    If ``add_random_initial_phase`` is False, returns zeros.

    Note: we keep phase sampling deterministic relative to the RNG state so that
    callers can reproduce exactly the same phases as ``compute_phasors``.
    """

    phi0 = np.zeros((int(n_wavelengths), int(n_paths)), dtype=np.float64)
    if add_random_initial_phase:
        phi0 = rng.uniform(0.0, 2.0 * np.pi, size=phi0.shape).astype(np.float64, copy=False)
    return phi0


def _path_amplitude(*, n_paths: int, mode: PathAmpMode) -> float:
    if n_paths <= 0:
        raise ValueError("n_paths must be > 0")

    m = mode.lower().strip()
    if m == "equal_power":
        return float(1.0 / np.sqrt(float(n_paths)))
    if m == "unit":
        return 1.0
    raise ValueError("path_amp must be 'equal_power' or 'unit'")


def _reference_frequency(
    *,
    f0_hz: float,
    df_hz: float,
    mode: RefMode,
) -> float:
    m = mode.lower().strip()
    if m == "lowest":
        return float(f0_hz)
    if m == "below_lowest":
        # Shift the reference so that every line rotates >= 1 cycle over T.
        return float(f0_hz - df_hz)
    raise ValueError("ref must be 'lowest' or 'below_lowest'")


def compute_phasors(cfg: InstantaneousPhasorSumConfig) -> InstantaneousPhasorSumResult:
    """Compute phasors for all times and (k,p) steps."""

    if cfg.n_wavelengths <= 0:
        raise ValueError("n_wavelengths must be > 0")
    if cfg.dt_ps <= 0:
        raise ValueError("dt_ps must be > 0")
    if cfg.T_ps <= 0:
        raise ValueError("T_ps must be > 0")
    if cfg.lambda0_nm <= 0:
        raise ValueError("lambda0_nm must be > 0")
    if not cfg.deltaL_mm:
        raise ValueError("deltaL_mm must contain at least one value")

    rng = np.random.default_rng(int(cfg.seed))

    f, lam, df = make_frequency_comb(lambda0_nm=cfg.lambda0_nm, n_wavelengths=cfg.n_wavelengths, T_ps=cfg.T_ps)

    n_paths = len(cfg.deltaL_mm)
    deltaL_m = np.asarray(cfg.deltaL_mm, dtype=np.float64) * 1e-3
    tau = (deltaL_m / C0_M_PER_S).astype(np.float64)

    f_ref = _reference_frequency(f0_hz=float(f[0]), df_hz=df, mode=cfg.ref)
    A_path = _path_amplitude(n_paths=n_paths, mode=cfg.path_amp)

    # Time samples: include both endpoints [0, T] for slide-like indexing.
    times_ps = np.arange(0.0, float(cfg.T_ps) + 1e-9, float(cfg.dt_ps), dtype=np.float64)
    times_s = (times_ps * 1e-12).astype(np.float64)

    order_kp = build_order_indices(
        n_wavelengths=int(cfg.n_wavelengths),
        n_paths=int(n_paths),
        order=cfg.order,
        rng=rng,
    )
    n_steps = int(order_kp.shape[0])

    # Optional random initial phases per (k,p), fixed over time.
    phi0 = _initial_phases(
        n_wavelengths=int(cfg.n_wavelengths),
        n_paths=int(n_paths),
        rng=rng,
        add_random_initial_phase=bool(cfg.add_random_initial_phase),
    )

    fk = f[order_kp[:, 0]]  # (n_steps,)
    pk = order_kp[:, 1]  # (n_steps,)
    tau_step = tau[pk]  # (n_steps,)
    phi0_step = phi0[order_kp[:, 0], pk]  # (n_steps,)

    # phase(t; k,p) = 2π (f_k - f_ref) t  -  2π f_k τ_p + φ0[k,p]
    phase = (
        2.0 * np.pi * (fk - f_ref)[None, :] * times_s[:, None]
        - 2.0 * np.pi * fk[None, :] * tau_step[None, :]
        + phi0_step[None, :]
    )
    phasors = (A_path * np.exp(1j * phase)).astype(np.complex128, copy=False)

    field = phasors.sum(axis=1)
    intensity = (field.real * field.real + field.imag * field.imag).astype(np.float64, copy=False)

    return InstantaneousPhasorSumResult(
        times_s=times_s,
        times_ps=times_ps,
        f_hz=f,
        lambda_m=lam,
        tau_s=tau,
        order_kp=order_kp,
        phasors=phasors,
        field=field,
        intensity=intensity,
        A_path=A_path,
        f_ref_hz=f_ref,
        df_hz=df,
    )


def time_average_intensity_numeric(
    out: InstantaneousPhasorSumResult,
    *,
    exclude_endpoint: bool = True,
) -> float:
    """Return the numeric time-average of ``I(t)`` from a computed result.

    Notes
    -----
    ``compute_phasors`` includes both endpoints ``t=0`` and ``t=T`` so slide-like
    indexing has an intuitive "last frame". For combs with ``Δf = 1/T`` the field
    at ``t=T`` repeats the field at ``t=0``, so a simple mean over all samples
    slightly double-counts the first frame.

    If ``exclude_endpoint`` is True (default), we drop the final sample to estimate
    the average over ``[0, T)``.
    """

    I = out.intensity
    if exclude_endpoint and I.shape[0] > 1:
        I = I[:-1]
    return float(np.mean(I))


def time_average_intensity_analytic(cfg: InstantaneousPhasorSumConfig) -> float:
    r"""Return the time-average intensity over one comb period (analytic).

    For the comb spacing used in this toy model (``Δf = 1/T``), the average of:

    $$I(t) = |E(t)|^2,\qquad E(t)=\sum_{k,p} E_{k,p}(t)$$

    over a full window of duration ``T`` simplifies to:

    $$
    \langle I\rangle_t
    = \sum_k \left|\sum_p A\exp\left(-i2\pi f_k\tau_p + i\phi_{k,p}\right)\right|^2.
    $$

    This is the expression derived in notebook 14 (step-index MMF spectral physics),
    adapted to the discrete-path toy model here.

    Implementation details
    ----------------------
    - The envelope reference frequency (``f_ref``) does not affect intensity.
    - We intentionally sample the initial-phase RNG in the same order as
      ``compute_phasors`` so ``cfg.seed`` reproduces the same phases.
    """

    if cfg.n_wavelengths <= 0:
        raise ValueError("n_wavelengths must be > 0")
    if cfg.lambda0_nm <= 0:
        raise ValueError("lambda0_nm must be > 0")
    if not cfg.deltaL_mm:
        raise ValueError("deltaL_mm must contain at least one value")

    n_paths = len(cfg.deltaL_mm)
    rng = np.random.default_rng(int(cfg.seed))

    # Match ``compute_phasors`` RNG consumption for ``order=random_fixed``.
    _ = build_order_indices(
        n_wavelengths=int(cfg.n_wavelengths),
        n_paths=int(n_paths),
        order=cfg.order,
        rng=rng,
    )

    f, _lam, _df = make_frequency_comb(
        lambda0_nm=float(cfg.lambda0_nm),
        n_wavelengths=int(cfg.n_wavelengths),
        T_ps=float(cfg.T_ps),
    )

    deltaL_m = np.asarray(cfg.deltaL_mm, dtype=np.float64) * 1e-3
    tau = (deltaL_m / C0_M_PER_S).astype(np.float64)

    phi0 = _initial_phases(
        n_wavelengths=int(cfg.n_wavelengths),
        n_paths=int(n_paths),
        rng=rng,
        add_random_initial_phase=bool(cfg.add_random_initial_phase),
    )
    A_path = _path_amplitude(n_paths=int(n_paths), mode=cfg.path_amp)

    # Sum over paths for each wavelength line k.
    # shape: (n_wavelengths, n_paths)
    phase0 = (-2.0 * np.pi * f[:, None] * tau[None, :]) + phi0
    e_k = (A_path * np.exp(1j * phase0)).sum(axis=1)

    I_avg = np.sum(e_k.real * e_k.real + e_k.imag * e_k.imag)
    return float(I_avg)
