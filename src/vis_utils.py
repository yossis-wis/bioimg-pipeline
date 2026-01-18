"""Visualization utilities for QC artifacts.

These functions generate visual overlays and montages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from skimage.morphology import dilation, disk
from skimage.segmentation import find_boundaries


def create_cutout_montage(
    nuclei_img: np.ndarray,
    spots_img: np.ndarray,
    spots_df: pd.DataFrame,
    *,
    crop_size: int = 80,
    max_cutouts: int = 50,
    n_cols: int = 10,
    sample_seed: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], int]:
    """Create a 2-channel montage of spot cutouts (Nuclei ch, Spot ch)."""
    if spots_df.empty:
        return None, 0

    n_spots = min(len(spots_df), max_cutouts)
    if "snr" in spots_df.columns:
        subset = spots_df.sort_values("snr", ascending=False).head(n_spots)
    else:
        seed = 0 if sample_seed is None else int(sample_seed)
        order = np.random.default_rng(seed).permutation(len(spots_df))[:n_spots]
        subset = spots_df.iloc[order]

    n_cols = max(1, min(int(n_cols), n_spots))
    n_rows = int(np.ceil(n_spots / n_cols))

    montage_h = n_rows * crop_size
    montage_w = n_cols * crop_size

    dtype = np.result_type(nuclei_img.dtype, spots_img.dtype)
    montage = np.zeros((2, montage_h, montage_w), dtype=dtype)

    height, width = nuclei_img.shape
    half = crop_size // 2

    for idx, (_, row) in enumerate(subset.iterrows()):
        r = idx // n_cols
        c = idx % n_cols

        y_c, x_c = int(row["y_px"]), int(row["x_px"])

        y0_src = max(0, y_c - half)
        y1_src = min(height, y_c + half)
        x0_src = max(0, x_c - half)
        x1_src = min(width, x_c + half)

        crop_h = y1_src - y0_src
        crop_w = x1_src - x0_src

        y0_dst = (crop_size - crop_h) // 2
        x0_dst = (crop_size - crop_w) // 2

        n_crop = nuclei_img[y0_src:y1_src, x0_src:x1_src]
        s_crop = spots_img[y0_src:y1_src, x0_src:x1_src]

        grid_y = r * crop_size
        grid_x = c * crop_size

        montage[0, grid_y + y0_dst : grid_y + y0_dst + crop_h, grid_x + x0_dst : grid_x + x0_dst + crop_w] = n_crop
        montage[1, grid_y + y0_dst : grid_y + y0_dst + crop_h, grid_x + x0_dst : grid_x + x0_dst + crop_w] = s_crop

    return montage, n_spots


def write_qc_overlay(
    spot_img: np.ndarray,
    nuclei_labels: np.ndarray,
    spots_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Write a PNG overlay of spots and nuclei contours."""
    import matplotlib.pyplot as plt

    img = spot_img.astype(float)
    if img.size:
        vmin, vmax = np.percentile(img, [1, 99])
    else:
        vmin, vmax = 0.0, 1.0

    boundaries = find_boundaries(nuclei_labels > 0, mode="outer")
    boundaries = dilation(boundaries, disk(1))

    fig = plt.figure(figsize=(10, 10), dpi=150)
    ax = fig.add_subplot(111)

    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")

    rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=float)
    rgba[boundaries, 0] = 1.0
    rgba[boundaries, 3] = 0.6
    ax.imshow(rgba, interpolation="nearest")

    if not spots_df.empty:
        ax.scatter(
            spots_df["x_px"],
            spots_df["y_px"],
            s=40,
            facecolors="none",
            edgecolors="cyan",
            linewidths=1.2,
            alpha=0.8,
        )
        ax.set_title(
            f"Integrated QC: {len(spots_df)} spots in {int(nuclei_labels.max())} nuclei"
        )
    else:
        ax.set_title("Integrated QC: 0 spots")

    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
