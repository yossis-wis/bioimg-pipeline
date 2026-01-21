"""Visualization utilities for QC artifacts.

These functions generate visual overlays and montages for human review.

Notes
-----
- QC outputs are not part of the strict *data contracts* (spots table + manifest),
  but they are critical for building trust in the analysis.
- Keep QC deterministic where possible (e.g. seed-controlled sampling).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from skimage.morphology import dilation, disk
from skimage.segmentation import find_boundaries


def _rescale_to_unit(img: np.ndarray, pmin: float, pmax: float) -> np.ndarray:
    """Robustly rescale an image to [0, 1] using percentiles.

    Parameters
    ----------
    img:
        2D array.
    pmin, pmax:
        Percentiles (0-100). Typical values: (1, 99) or (1, 99.8).

    Returns
    -------
    np.ndarray
        float32 array in [0, 1].
    """
    x = np.asarray(img, dtype=np.float32)
    if x.size == 0:
        return np.zeros_like(x, dtype=np.float32)

    lo, hi = np.percentile(x, [float(pmin), float(pmax)])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=np.float32)

    y = (x - float(lo)) / float(hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32, copy=False)


def merge_nuclei_spots_rgb(
    nuclei_img: np.ndarray,
    spots_img: np.ndarray,
    *,
    nuclei_pmin: float = 1.0,
    nuclei_pmax: float = 99.8,
    spots_pmin: float = 1.0,
    spots_pmax: float = 99.0,
    spots_to_cyan: bool = False,
) -> np.ndarray:
    """Create an RGB merge (Fiji-like composite) from nuclei + spots images.

    Mapping
    -------
    - Nuclei -> Red channel
    - Spots  -> Green channel (and optionally Blue as well for cyan)

    Returns
    -------
    rgb : np.ndarray
        Float32 image of shape (Y, X, 3) in [0, 1].
    """
    r = _rescale_to_unit(nuclei_img, nuclei_pmin, nuclei_pmax)
    g = _rescale_to_unit(spots_img, spots_pmin, spots_pmax)
    if spots_to_cyan:
        b = g
    else:
        b = np.zeros_like(g)
    rgb = np.stack([r, g, b], axis=-1)
    return rgb.astype(np.float32, copy=False)


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
    """Create a 2-channel montage of spot cutouts (Nuclei ch, Spot ch).

    Selection rule
    --------------
    - If ``sample_seed is None`` and ``snr`` exists: take highest-SNR spots.
    - Otherwise: shuffle deterministically with the provided seed.

    Cropping rule
    -------------
    Each cutout is centered on the detected spot. If the cutout extends outside
    the image, it is zero-padded so the spot stays centered.

    Returns
    -------
    montage:
        Array of shape (2, montage_h, montage_w) with axes (C, Y, X),
        or None if ``spots_df`` is empty.
    count:
        Number of cutouts actually placed.
    """
    if spots_df.empty:
        return None, 0

    crop_size = int(crop_size)
    if crop_size <= 0:
        raise ValueError("crop_size must be > 0")

    max_cutouts = int(max_cutouts)
    if max_cutouts <= 0:
        return None, 0

    n_spots = min(len(spots_df), max_cutouts)

    if sample_seed is None and "snr" in spots_df.columns:
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

    placed = 0
    for idx, (_, row) in enumerate(subset.iterrows()):
        r = idx // n_cols
        c = idx % n_cols

        y_c, x_c = int(row["y_px"]), int(row["x_px"])

        # Desired crop bounds (centered on spot)
        y0 = y_c - half
        x0 = x_c - half
        y1 = y0 + crop_size
        x1 = x0 + crop_size

        # Clip to image bounds
        y0c = max(0, y0)
        x0c = max(0, x0)
        y1c = min(height, y1)
        x1c = min(width, x1)

        # Required padding to keep the spot centered
        pad_top = max(0, -y0)
        pad_left = max(0, -x0)
        pad_bottom = max(0, y1 - height)
        pad_right = max(0, x1 - width)

        n_crop = nuclei_img[y0c:y1c, x0c:x1c]
        s_crop = spots_img[y0c:y1c, x0c:x1c]

        if pad_top or pad_bottom or pad_left or pad_right:
            n_crop = np.pad(
                n_crop,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=0,
            )
            s_crop = np.pad(
                s_crop,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=0,
            )

        if n_crop.shape != (crop_size, crop_size) or s_crop.shape != (crop_size, crop_size):
            # Defensive: skip malformed crops rather than crashing QC.
            continue

        grid_y = r * crop_size
        grid_x = c * crop_size
        montage[0, grid_y : grid_y + crop_size, grid_x : grid_x + crop_size] = n_crop
        montage[1, grid_y : grid_y + crop_size, grid_x : grid_x + crop_size] = s_crop
        placed += 1

    return montage, int(placed)


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
