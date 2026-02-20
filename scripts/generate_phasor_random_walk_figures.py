#!/usr/bin/env python3
"""Generate canonical SVG figures for the phasor-sum / random-walk toy model.

This script writes a small set of stable, GitHub-friendly SVGs to ``docs/figures/``:

- phasor_random_walk_single_path.svg
- phasor_random_walk_endpoints_scatter.svg
- phasor_random_walk_intensity_hist.svg
- phasor_random_walk_contrast_vs_K.svg

These figures are *tracked* assets: they are intentionally committed so the notebook
Markdown mirror is readable on GitHub.

Style notes
-----------
- Prefer SVG.
- Target width ~900 px (we export with a 900×... viewBox).
- Sans-serif fonts, axes ~2 px, curves ~2–3 px.

See: ``docs/figures/STYLE_GUIDE.md``.

Run
---
    python scripts/generate_phasor_random_walk_figures.py
"""  # noqa: D401

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Allow running from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.phasor_random_walk import (  # noqa: E402
    average_over_k_uncorrelated,
    simulate_ensemble,
    simulate_walk,
    speckle_contrast_1d,
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


def _new_figure(*, height_in: float = 4.8) -> tuple[plt.Figure, plt.Axes]:
    """Return (fig, ax) with a ~900 px wide SVG canvas.

    Matplotlib's SVG backend uses "pt" units where 1 inch = 72 pt. To get a 900-wide
    viewBox we set width_in = 900/72 = 12.5 inches.
    """

    fig, ax = plt.subplots(figsize=(12.5, float(height_in)))
    # Leave comfortable outer margins (≥~30 px once rendered).
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.14, top=0.90)
    return fig, ax


def fig_single_path(*, out_path: Path, seed: int) -> None:
    walk = simulate_walk(n_steps=60, amplitude_model="equal", phase_model="uniform", seed=seed)

    fig, ax = _new_figure(height_in=4.8)
    ax.plot(walk.path.real, walk.path.imag, marker="o", markersize=3.5, linewidth=2.6)
    ax.scatter([0.0], [0.0], s=60, marker="o", zorder=4)
    ax.scatter([walk.endpoint.real], [walk.endpoint.imag], s=90, marker="*", zorder=5)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_title("2D random walk (phasor sum): one realization")
    ax.set_xlabel("Re [a.u.]")
    ax.set_ylabel("Im [a.u.]")
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    _write_svg(fig, out_path)
    plt.close(fig)


def fig_endpoints_scatter(*, out_path: Path, seed: int) -> None:
    endpoints, _ = simulate_ensemble(
        n_steps=60,
        n_realizations=2500,
        amplitude_model="equal",
        phase_model="uniform",
        seed=seed,
    )

    fig, ax = _new_figure(height_in=4.8)
    ax.scatter(endpoints.real, endpoints.imag, s=10, alpha=0.35)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_title("Endpoint distribution over many realizations")
    ax.set_xlabel("Re [a.u.]")
    ax.set_ylabel("Im [a.u.]")
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    _write_svg(fig, out_path)
    plt.close(fig)


def fig_intensity_hist(*, out_path: Path, seed: int) -> None:
    _, intensities = simulate_ensemble(
        n_steps=60,
        n_realizations=20000,
        amplitude_model="equal",
        phase_model="uniform",
        seed=seed,
    )
    mu = float(np.mean(intensities))

    fig, ax = _new_figure(height_in=4.8)
    bins = 60
    ax.hist(intensities, bins=bins, density=True, alpha=0.65, label="simulation")

    # For a circular complex Gaussian field, intensity is exponential: p(I) = exp(-I/μ)/μ.
    x = np.linspace(0.0, float(np.quantile(intensities, 0.995)), 500)
    pdf = (1.0 / mu) * np.exp(-x / mu)
    ax.plot(x, pdf, linewidth=2.6, label="exp. pdf with same mean")

    ax.grid(True, alpha=0.25)
    ax.set_title("Intensity statistics for a random phasor sum")
    ax.set_xlabel("Intensity I = |E|^2 [a.u.]")
    ax.set_ylabel("Probability density [a.u.^-1]")
    ax.legend(framealpha=0.90)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    _write_svg(fig, out_path)
    plt.close(fig)


def fig_contrast_vs_k(*, out_path: Path, seed: int) -> None:
    # Evaluate contrast for averaged intensities I_K = (1/K) sum_{j=1..K} I_j.
    k_list = np.array([1, 2, 4, 8, 16, 32], dtype=int)
    m = 4000  # averaged samples per K (keeps runtime modest)

    C = []
    for k in k_list:
        # Need m*k independent samples.
        _, I = simulate_ensemble(
            n_steps=60,
            n_realizations=int(m * k),
            amplitude_model="equal",
            phase_model="uniform",
            seed=int(seed + 1000 + k),
        )
        I_k = average_over_k_uncorrelated(I, k=int(k))
        C.append(speckle_contrast_1d(I_k))

    C = np.asarray(C, dtype=np.float64)

    fig, ax = _new_figure(height_in=4.8)
    ax.plot(k_list, C, marker="o", linewidth=2.6, label="simulation")

    # Reference scaling for independent averaging: C ~ 1/sqrt(K).
    ax.plot(k_list, 1.0 / np.sqrt(k_list), linestyle="--", linewidth=2.2, label="1/sqrt(K)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title("Speckle contrast vs averaging factor")
    ax.set_xlabel("Averaging factor K [count]")
    ax.set_ylabel("Contrast C = std/mean [a.u.]")
    ax.legend(framealpha=0.90)
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
    ap.add_argument("--seed", type=int, default=0, help="Random seed for deterministic figures")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    seed = int(args.seed)

    fig_single_path(out_path=out_dir / "phasor_random_walk_single_path.svg", seed=seed)
    fig_endpoints_scatter(out_path=out_dir / "phasor_random_walk_endpoints_scatter.svg", seed=seed)
    fig_intensity_hist(out_path=out_dir / "phasor_random_walk_intensity_hist.svg", seed=seed)
    fig_contrast_vs_k(out_path=out_dir / "phasor_random_walk_contrast_vs_K.svg", seed=seed)

    print("OK: wrote phasor random-walk figures to", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
