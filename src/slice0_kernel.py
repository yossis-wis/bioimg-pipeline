from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from skimage.feature import peak_local_max
from skimage.filters import gaussian


REQUIRED_COLUMNS = ["frame", "y_px", "x_px", "intensity", "background", "snr"]


@dataclass(frozen=True)
class Slice0Params:
    threshold: float
    min_distance: int = 3
    smooth_sigma: float = 1.0  # 0 disables smoothing

    # background estimation window
    bg_radius: int = 4  # uses a (2*bg_radius+1)^2 patch
    bg_exclude_radius: int = 1  # excludes a (2*bg_exclude_radius+1)^2 center


def _robust_std(x: np.ndarray) -> float:
    """Robust std via MAD; returns 0.0 if too few samples."""
    x = np.asarray(x, dtype=float)
    if x.size < 5:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad)


def detect_spots(image_2d: np.ndarray, params: Slice0Params) -> pd.DataFrame:
    """Detect candidate spots in a single 2D image.

    This is intentionally simple (Slice0): local maxima above a threshold.

    Returns a DataFrame that satisfies docs/CONTRACTS.md required columns.
    """
    if image_2d.ndim != 2:
        raise ValueError(f"detect_spots expects a 2D array; got shape={image_2d.shape}")

    img = np.asarray(image_2d)
    img_f = img.astype(np.float32, copy=False)

    if params.smooth_sigma and params.smooth_sigma > 0:
        work = gaussian(img_f, sigma=float(params.smooth_sigma), preserve_range=True)
    else:
        work = img_f

    coords = peak_local_max(
        work,
        min_distance=int(params.min_distance),
        threshold_abs=float(params.threshold),
        exclude_border=False,
    )

    if coords.size == 0:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    H, W = img_f.shape
    rows = []
    r = int(params.bg_radius)
    er = int(params.bg_exclude_radius)

    for (y, x) in coords:
        y = int(y); x = int(x)
        intensity = float(img_f[y, x])

        y0 = max(0, y - r); y1 = min(H, y + r + 1)
        x0 = max(0, x - r); x1 = min(W, x + r + 1)
        patch = img_f[y0:y1, x0:x1].astype(float, copy=False)

        # Exclude a small center region so the spot itself doesn't dominate the background.
        cy0 = max(0, (y - er) - y0); cy1 = min(patch.shape[0], (y + er + 1) - y0)
        cx0 = max(0, (x - er) - x0); cx1 = min(patch.shape[1], (x + er + 1) - x0)

        mask = np.ones(patch.shape, dtype=bool)
        mask[cy0:cy1, cx0:cx1] = False
        bg_vals = patch[mask]

        background = float(np.median(bg_vals)) if bg_vals.size else float(np.median(patch))
        noise = _robust_std(bg_vals) if bg_vals.size else _robust_std(patch)
        if noise <= 1e-6:
            noise = float(np.std(bg_vals)) if bg_vals.size else float(np.std(patch))
        if noise <= 1e-6:
            noise = 1e-6

        snr = (intensity - background) / noise

        rows.append(
            dict(
                frame=0,
                y_px=float(y),
                x_px=float(x),
                intensity=float(intensity),
                background=float(background),
                snr=float(snr),
            )
        )

    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
