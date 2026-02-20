#!/usr/bin/env python3
"""Phasor "slide-type" animation generator (PNG frames + optional GIF).

This script generates the same *instantaneous head-to-tail phasor snapshots* used in
the accompanying decks and notebooks:

- big fixed Re/Im axes
- head-to-tail sum (order affects the drawn path, not the endpoint)
- "ghost" of the previous frame (t-Δt)
- color-coded by wavelength index
- marker shape by path index

Model
-----
We use a simple comb × discrete-path model:

$$
E_{k,p}(t)=A\exp\{i[2\pi(f_k-f_\mathrm{ref})t-2\pi f_k\tau_p+\phi_{k,p}]\},
$$
$$
E(t)=\sum_{k,p}E_{k,p}(t),\qquad I(t)=|E(t)|^2.
$$

See :mod:`src.instantaneous_phasor_sum` for the implementation.

Outputs
-------
- A directory of PNG frames ("slide browsing").
- Optionally a looping GIF.

Notes
-----
- The plotted phasors use a *complex-envelope reference* (``f_ref``) so motion is visible
  on picosecond scales. Global phase does not affect intensity.
- For reproducible figures, set ``--seed`` (and optionally ``--add_random_initial_phase``).

Example
-------
Generate 1 ps snapshots (161 frames) + a GIF:

.. code-block:: bash

    python scripts/phasor_slidertype_animation.py \
      --N_wl 20 --T_ps 160 --dt_ps 1 \
      --deltaL_mm 0,25,51 \
      --outdir reports/phasor_slides_60phasors \
      --make_gif --fps 25

"""  # noqa: D401

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

# Allow running from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.instantaneous_phasor_sum import (  # noqa: E402
    InstantaneousPhasorSumConfig,
    compute_phasors,
    parse_deltaL_mm,
)


def _apply_plot_style() -> None:
    """Apply a lightweight style consistent with docs/figures/STYLE_GUIDE.md."""

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "axes.linewidth": 2.0,
            "lines.linewidth": 2.5,
            "xtick.major.width": 2.0,
            "ytick.major.width": 2.0,
            "xtick.minor.width": 1.5,
            "ytick.minor.width": 1.5,
        }
    )


def compute_axis_limit(z: np.ndarray, *, margin: float = 1.10) -> float:
    """Compute a symmetric axis limit that contains all partial sums.

    Parameters
    ----------
    z:
        Complex array with shape ``(n_times, n_steps)``.
    margin:
        Multiplicative margin beyond the max abs Re/Im.
    """

    cum = np.cumsum(z, axis=1)  # (n_times, n_steps)
    max_abs = np.max(np.abs(np.concatenate([cum.real.ravel(), cum.imag.ravel()])))
    return float(max_abs * float(margin) + 1e-12)


def render_frame(
    *,
    out_png: Path,
    z_now: np.ndarray,  # (n_steps,)
    z_prev: Optional[np.ndarray],  # (n_steps,) or None
    order_kp: np.ndarray,  # (n_steps,2)
    t_ps: float,
    E_now: complex,
    I_now: float,
    axis_lim: float,
    cfg: InstantaneousPhasorSumConfig,
) -> None:
    """Create a single slide-like Re/Im snapshot and save to PNG."""

    n_steps = int(z_now.shape[0])
    n_wl = int(cfg.n_wavelengths)
    n_paths = len(cfg.deltaL_mm)

    # Build head-to-tail cumulative positions
    start = np.zeros(n_steps + 1, dtype=np.complex128)
    start[1:] = np.cumsum(z_now.astype(np.complex128, copy=False))
    segs = np.stack(
        [
            np.column_stack([start[:-1].real, start[:-1].imag]),
            np.column_stack([start[1:].real, start[1:].imag]),
        ],
        axis=1,
    )

    # Colors by wavelength index of each step
    k_idx = order_kp[:, 0].astype(np.float64)
    cmap = plt.get_cmap("viridis")
    colors = cmap((k_idx / max(n_wl - 1, 1)).astype(np.float64))

    # Marker by path index
    p_idx = order_kp[:, 1]
    markers = ["o", "^", "s", "D", "v", "P", "X"]
    if n_paths > len(markers):
        raise ValueError(f"Too many paths ({n_paths}) for built-in markers list")

    # 16:9 canvas; choose dpi for crisp text/lines.
    fig = plt.figure(figsize=(12.8, 7.2), dpi=160)
    ax = fig.add_axes([0.06, 0.10, 0.88, 0.84])

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Re [a.u.]")
    ax.set_ylabel("Im [a.u.]")
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    # Ghost previous snapshot
    if z_prev is not None:
        prev = np.zeros(n_steps + 1, dtype=np.complex128)
        prev[1:] = np.cumsum(z_prev.astype(np.complex128, copy=False))
        prev_segs = np.stack(
            [
                np.column_stack([prev[:-1].real, prev[:-1].imag]),
                np.column_stack([prev[1:].real, prev[1:].imag]),
            ],
            axis=1,
        )
        lc_prev = LineCollection(
            prev_segs,
            colors=[(0.45, 0.45, 0.45, 0.55)],
            linewidths=1.5,
            linestyles="--",
        )
        ax.add_collection(lc_prev)

    # Current segments
    lc = LineCollection(segs, colors=colors, linewidths=2.3)
    ax.add_collection(lc)

    # Per-step endpoints with path markers
    endpts = start[1:]
    for p in range(n_paths):
        mask = p_idx == p
        if not np.any(mask):
            continue
        ax.scatter(
            endpts.real[mask],
            endpts.imag[mask],
            s=22,
            marker=markers[p],
            c=colors[mask],
            edgecolors="none",
            alpha=0.95,
        )

    # Endpoint vector + marker
    ax.plot([0.0, start[-1].real], [0.0, start[-1].imag], linewidth=3.0, alpha=0.9)
    ax.scatter([start[-1].real], [start[-1].imag], s=90, marker="*", zorder=5)

    absE = float(np.abs(E_now))
    ax.set_title(
        "Instantaneous phasor sum (head-to-tail)\n"
        f"N_wl={cfg.n_wavelengths}, paths={n_paths}  |  t = {t_ps:.1f} ps  |  "
        f"|E(t)| = {absE:.4f}   I(t)=|E|^2 = {I_now:.4f}  (ghost = t−{cfg.dt_ps:g} ps)",
        fontsize=12,
    )

    # Colorbar mapping index -> wavelength index
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max(n_wl - 1, 1)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label("wavelength index k (color)", rotation=90)

    # Legend for paths (marker shapes)
    handles = []
    labels = []
    for p, dL in enumerate(cfg.deltaL_mm):
        h = ax.scatter([], [], marker=markers[p], s=70, c="k")
        handles.append(h)
        labels.append(f"ΔL = {dL:g} mm")
    ax.legend(handles, labels, loc="lower left", framealpha=0.90, fontsize=10)

    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def maybe_make_gif(*, frames_dir: Path, out_gif: Path, fps: float) -> None:
    """Write a looping GIF from PNG frames."""

    try:
        import imageio.v3 as iio
    except Exception as e:  # pragma: no cover
        raise RuntimeError("imageio not available. Install with: pip install imageio") from e

    files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(".png")])
    if not files:
        raise RuntimeError(f"No PNG frames found in {frames_dir}")
    imgs = [iio.imread(frames_dir / f) for f in files]
    iio.imwrite(out_gif, imgs, duration=1.0 / max(float(fps), 1e-6), loop=0)


def main() -> int:
    _apply_plot_style()

    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda0_nm", type=float, default=640.0)
    ap.add_argument("--N_wl", type=int, default=20, help="Number of wavelength comb lines")
    ap.add_argument("--T_ps", type=float, default=160.0, help="Total time span [ps]")
    ap.add_argument("--dt_ps", type=float, default=1.0, help="Time step [ps]")
    ap.add_argument("--deltaL_mm", type=str, default="0,25,51", help="Comma-separated ΔL list [mm]")
    ap.add_argument(
        "--path_amp",
        type=str,
        choices=["equal_power", "unit"],
        default="equal_power",
        help="Path amplitude model",
    )
    ap.add_argument(
        "--ref",
        type=str,
        choices=["lowest", "below_lowest"],
        default="lowest",
        help="Reference frequency for the plotted complex envelope",
    )
    ap.add_argument(
        "--order",
        type=str,
        choices=["by_wavelength", "by_path", "random"],
        default="by_wavelength",
        help="Drawing order for head-to-tail phasors",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--add_random_initial_phase", action="store_true")
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(Path("reports") / "phasor_slides"),
        help="Output directory for PNG frames",
    )
    ap.add_argument("--make_gif", action="store_true")
    ap.add_argument("--gif_name", type=str, default="phasor_slides.gif")
    ap.add_argument("--fps", type=float, default=20.0)
    args = ap.parse_args()

    order = "random_fixed" if args.order == "random" else args.order

    cfg = InstantaneousPhasorSumConfig(
        lambda0_nm=float(args.lambda0_nm),
        n_wavelengths=int(args.N_wl),
        T_ps=float(args.T_ps),
        dt_ps=float(args.dt_ps),
        deltaL_mm=parse_deltaL_mm(args.deltaL_mm),
        path_amp=str(args.path_amp),
        ref=str(args.ref),
        order=str(order),
        seed=int(args.seed),
        add_random_initial_phase=bool(args.add_random_initial_phase),
    )

    data = compute_phasors(cfg)
    z = data.phasors
    axis_lim = compute_axis_limit(z)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    times_ps = data.times_ps
    order_kp = data.order_kp

    print(f"Rendering {len(times_ps)} frames to: {outdir.as_posix()}")
    z_prev: Optional[np.ndarray] = None
    for i, t_ps in enumerate(times_ps):
        png = outdir / f"frame_{i:04d}_t{t_ps:07.1f}ps.png"
        render_frame(
            out_png=png,
            z_now=z[i],
            z_prev=z_prev,
            order_kp=order_kp,
            t_ps=float(t_ps),
            E_now=complex(data.field[i]),
            I_now=float(data.intensity[i]),
            axis_lim=axis_lim,
            cfg=cfg,
        )
        z_prev = z[i].copy()

        if (i + 1) % 10 == 0 or i == len(times_ps) - 1:
            print(f"  {i+1}/{len(times_ps)}")

    if args.make_gif:
        out_gif = outdir / str(args.gif_name)
        print(f"Writing GIF: {out_gif.as_posix()}")
        maybe_make_gif(frames_dir=outdir, out_gif=out_gif, fps=float(args.fps))
        print("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
