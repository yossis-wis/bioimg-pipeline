#!/usr/bin/env python3
"""Generate canonical SVG figures for instantaneous intensity toy-model visualizations.

This script writes a small set of stable, GitHub-friendly SVGs to ``docs/figures/``:

- instantaneous_intensity_traces_stacked.svg
- instantaneous_intensity_heatmap.svg
- instantaneous_timeavg_intensity_hist.svg

These are *tracked* assets: they are intentionally committed so the notebook
Markdown mirror is readable on GitHub.

Style notes
-----------
- Prefer SVG.
- Target width ~900 px (we export with a 900×... viewBox).
- Sans-serif fonts, axes ~2 px, curves ~2–3 px.
- Comfortable outer margins (≥ ~30 px).

See: ``docs/figures/STYLE_GUIDE.md``.

Run
---
    python scripts/generate_instantaneous_intensity_figures.py
"""  # noqa: D401

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Allow running from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.instantaneous_phasor_sum import (  # noqa: E402
    InstantaneousPhasorSumConfig,
    compute_phasors,
    time_average_intensity_analytic,
    time_average_intensity_numeric,
)


def _apply_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "axes.linewidth": 2.0,
            "lines.linewidth": 2.6,
            "xtick.major.width": 2.0,
            "ytick.major.width": 2.0,
            "xtick.minor.width": 1.5,
            "ytick.minor.width": 1.5,
        }
    )


def _write_svg(fig: plt.Figure, out_path: Path) -> None:
    """Save SVG and normalize root width/height to match repo conventions."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg")

    # Matplotlib writes width/height like '900pt'. Our repo SVGs use unitless pixels.
    svg = out_path.read_text(encoding="utf-8", errors="replace")

    # Replace e.g. width="900pt" -> width="900" (same for height).
    svg = re.sub(r'\bwidth="([0-9.]+)pt"', r'width="\1"', svg)
    svg = re.sub(r'\bheight="([0-9.]+)pt"', r'height="\1"', svg)

    # Remove Matplotlib metadata (timestamps, version strings) for stable diffs.
    svg = re.sub(r"<metadata>.*?</metadata>", "", svg, flags=re.S)

    out_path.write_text(svg, encoding="utf-8")


def _new_figure(*, height_in: float) -> tuple[plt.Figure, plt.Axes]:
    """Return (fig, ax) with a ~900 px wide SVG canvas."""

    fig, ax = plt.subplots(figsize=(12.5, float(height_in)))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.14, top=0.90)
    return fig, ax


def _baseline_cfg(*, seed: int, dt_ps: float = 1.0) -> InstantaneousPhasorSumConfig:
    return InstantaneousPhasorSumConfig(
        lambda0_nm=640.0,
        n_wavelengths=20,
        T_ps=160.0,
        dt_ps=float(dt_ps),
        deltaL_mm=(0.0, 25.0, 51.0),
        path_amp="equal_power",
        ref="lowest",
        order="by_wavelength",
        seed=int(seed),
        add_random_initial_phase=True,
    )


def fig_traces_stacked(*, out_path: Path, seed0: int, n_traces: int) -> None:
    outs = [compute_phasors(_baseline_cfg(seed=int(seed0 + i))) for i in range(int(n_traces))]
    t_ps = outs[0].times_ps

    traces = []
    for o in outs:
        mu = time_average_intensity_numeric(o, exclude_endpoint=True)
        traces.append(o.intensity / mu)
    I = np.asarray(traces, dtype=np.float64)

    y_max = float(np.quantile(I, 0.995))
    spacing = 1.15 * y_max

    fig, ax = _new_figure(height_in=6.2)
    for i in range(int(n_traces)):
        ax.plot(t_ps, I[i] + i * spacing, linewidth=2.2)

    ax.grid(True, alpha=0.22)
    ax.set_title("Instantaneous intensity over time: many random initializations (stacked)")
    ax.set_xlabel("time [ps]")
    ax.set_ylabel("Normalized intensity I(t)/⟨I⟩$_t$ + offset [a.u.]")
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    _write_svg(fig, out_path)
    plt.close(fig)


def fig_heatmap(*, out_path: Path, seed0: int, n_realizations: int) -> None:
    outs = [compute_phasors(_baseline_cfg(seed=int(seed0 + i))) for i in range(int(n_realizations))]

    # Use [0, T) samples to avoid the duplicated endpoint t=T.
    t_ps = outs[0].times_ps[:-1]
    dt = float(t_ps[1] - t_ps[0]) if t_ps.shape[0] >= 2 else 1.0

    Z = []
    for o in outs:
        mu = time_average_intensity_numeric(o, exclude_endpoint=True)
        Z.append((o.intensity[:-1] / mu).astype(np.float64, copy=False))
    Zm = np.asarray(Z, dtype=np.float64)  # shape (n_realizations, n_times)

    # IMPORTANT: avoid giant SVGs.
    #
    # Matplotlib's pcolormesh emits one vector patch per cell, which becomes huge even for
    # modest grids (e.g. 80×160). For the repo's "canonical" docs figure we instead
    # rasterize the heatmap inside the SVG via imshow(), which embeds a compact PNG.
    extent = [float(t_ps[0]), float(t_ps[-1] + dt), 0.0, float(n_realizations)]

    fig, ax = _new_figure(height_in=5.8)
    im = ax.imshow(
        Zm,
        aspect="auto",
        origin="lower",
        extent=extent,
        interpolation="nearest",
    )
    im.set_rasterized(True)

    ax.set_title("Time × realization heatmap: normalized instantaneous intensity")
    ax.set_xlabel("time [ps]")
    ax.set_ylabel("realization index [count]")
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label("I(t)/⟨I⟩$_t$ [a.u.]")
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    _write_svg(fig, out_path)
    plt.close(fig)


def fig_timeavg_hist(*, out_path: Path, seed0: int, n_samples: int) -> None:
    cfg0 = _baseline_cfg(seed=int(seed0))
    means = np.array(
        [time_average_intensity_analytic(replace(cfg0, seed=int(seed0 + i))) for i in range(int(n_samples))],
        dtype=np.float64,
    )
    mu = float(np.mean(means))
    x = means / mu

    fig, ax = _new_figure(height_in=4.9)
    ax.hist(x, bins=70, density=True, alpha=0.70)
    ax.axvline(1.0, linestyle="--", linewidth=2.0, alpha=0.45)
    ax.grid(True, alpha=0.25)
    ax.set_title("Distribution of time-averaged intensity across random phase initializations")
    ax.set_xlabel("⟨I⟩$_t$ / ⟨⟨I⟩$_t$⟩ [a.u.]")
    ax.set_ylabel("Probability density [a.u.^-1]")
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    _write_svg(fig, out_path)
    plt.close(fig)


def main() -> int:
    _apply_style()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(REPO_ROOT / "docs" / "figures"),
        help="Output directory (default: docs/figures)",
    )
    ap.add_argument("--seed0", type=int, default=0, help="Base seed for deterministic figures")
    ap.add_argument("--n_traces", type=int, default=25, help="Number of stacked traces")
    ap.add_argument("--n_heat", type=int, default=80, help="Number of realizations for the heatmap")
    ap.add_argument("--n_hist", type=int, default=10000, help="Histogram sample count")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    seed0 = int(args.seed0)

    fig_traces_stacked(
        out_path=out_dir / "instantaneous_intensity_traces_stacked.svg",
        seed0=seed0,
        n_traces=int(args.n_traces),
    )
    fig_heatmap(
        out_path=out_dir / "instantaneous_intensity_heatmap.svg",
        seed0=seed0,
        n_realizations=int(args.n_heat),
    )
    fig_timeavg_hist(
        out_path=out_dir / "instantaneous_timeavg_intensity_hist.svg",
        seed0=seed0,
        n_samples=int(args.n_hist),
    )

    print("OK: wrote instantaneous-intensity figures to", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
