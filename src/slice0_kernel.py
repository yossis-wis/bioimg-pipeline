"""Slice0 kernel: spot detection.

This kernel is intentionally **pure computation** (no filesystem I/O).

The implementation here mirrors the spot-detection math used in
`2024-11-12_realtime_analysis_0.py`:

1) Build a Laplacian-of-Gaussian (LoG) filter from optics-ish parameters
2) Convolve the image with the LoG (mode='valid')
3) Find local maxima in the *negative* LoG response via a max filter
4) For each candidate maximum, compute:
     - background = median in a thin ring mask (out0)
     - mean_in5   = mean inside a small disk mask (in5)
     - mean_in7   = mean inside a slightly larger disk mask (in7)
     - u0 = mean_in5 - background
     - u1 = mean_in7 - background
   and keep spots with u0 > u0_min

Optional masks can further restrict candidate maxima (e.g. AOI/illumination mask,
or nuclei labels from Slice1).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["frame", "y_px", "x_px", "intensity", "background", "snr"]


@dataclass(frozen=True)
class Slice0Params:
    """Parameters for LoG-based spot detection.

    Defaults are chosen to closely match the realtime analysis script.
    """

    # --- Optics-ish LoG definition (mirrors realtime script) ---
    zR: float = 344.5
    lambda_nm: float = 667.0
    pixel_size_nm: float = 65.0

    # --- Candidate maxima + quality threshold ---
    q_min: float = 1.0
    se_size: int = 15

    # --- Per-spot measurements (mirrors realtime script) ---
    window_radius_px: int = 15  # -> 31x31 window
    in5_radius_px: int = 2
    in7_radius_px: int = 3
    ring_outer_radius_px: int = 10
    ring_inner_radius_px: int = 9

    # Keep spot if (mean(in5) - background) > u0_min
    u0_min: float = 30.0


def _robust_std(x: np.ndarray) -> float:
    """Robust std via MAD; returns 0.0 if too few samples."""
    x = np.asarray(x, dtype=float)
    if x.size < 5:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad)


def _laplacian_of_gaussian(size: int, sigma: float) -> np.ndarray:
    """LoG filter matching the realtime analysis implementation."""
    size = int(size)
    if size % 2 == 0:
        size += 1
    half = size // 2

    n1_grid, n2_grid = np.meshgrid(
        range(-half, half + 1),
        range(-half, half + 1),
        indexing="ij",
    )
    n1_grid = n1_grid.astype(np.float32)
    n2_grid = n2_grid.astype(np.float32)

    sigma = float(sigma)
    hg = np.exp(-(n1_grid**2 + n2_grid**2) / (2.0 * sigma**2))
    h_unnormalized = (n1_grid**2 + n2_grid**2 - 2.0 * sigma**2) * hg
    normalizing_factor = (sigma**4) * np.sum(hg)
    h_normalized = h_unnormalized / normalizing_factor
    return h_normalized.astype(np.float32, copy=False)


@lru_cache(maxsize=64)
def _cached_log_filter(size: int, sigma: float) -> np.ndarray:
    """Cached LoG filter.

    We keep the math identical to `_laplacian_of_gaussian`; this is purely a
    performance optimization when the kernel is called repeatedly with the same
    parameters.
    """
    h = _laplacian_of_gaussian(size, sigma)
    # Make accidental mutation obvious.
    h.setflags(write=False)
    return h


@lru_cache(maxsize=64)
def _disk_mask(radius_px: int, window_size: int) -> np.ndarray:
    """Return a (window_size x window_size) boolean disk mask."""
    try:
        from skimage.morphology import disk  # type: ignore
    except Exception as e:
        raise ImportError(
            "scikit-image is required for Slice0 masks (skimage.morphology.disk)."
        ) from e

    base = disk(int(radius_px)).astype(bool)
    pad = (int(window_size) - int(base.shape[0])) // 2
    if pad < 0:
        raise ValueError(
            f"disk(radius={radius_px}) is larger than window_size={window_size}."
        )
    m = np.pad(base, ((pad, pad), (pad, pad)), mode="constant").astype(bool)
    m.setflags(write=False)
    return m


@lru_cache(maxsize=64)
def _ring_mask(inner_radius_px: int, outer_radius_px: int, window_size: int) -> np.ndarray:
    """Return a thin ring mask (outer disk minus inner disk), padded to window_size."""
    outer = _disk_mask(int(outer_radius_px), int(window_size))
    inner = _disk_mask(int(inner_radius_px), int(window_size))
    m = outer & (~inner)
    m.setflags(write=False)
    return m


def detect_spots(
    image_2d: np.ndarray,
    params: Slice0Params,
    *,
    valid_mask: Optional[np.ndarray] = None,
    nuclei_labels: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Detect spots in a single 2D image using the realtime LoG method.

    Parameters
    ----------
    image_2d:
        2D image array (Y, X).
    params:
        Detection parameters.
    valid_mask:
        Optional boolean mask (Y, X). If provided, candidates are restricted
        to mask==True (e.g. illumination/AOI mask).
    nuclei_labels:
        Optional label image (Y, X). If provided, candidates are restricted
        to pixels where labels>0 (spots inside nuclei).

    Returns
    -------
    pd.DataFrame
        Spots table containing the required Slice0 contract columns.
        Additional columns are included for debugging/analysis.
    """

    if image_2d.ndim != 2:
        raise ValueError(f"detect_spots expects a 2D array; got shape={image_2d.shape}")

    img = np.asarray(image_2d)
    img_f = img.astype(np.float32, copy=False)
    H, W = img_f.shape

    if valid_mask is not None:
        vm = np.asarray(valid_mask).astype(bool, copy=False)
        if vm.shape != (H, W):
            raise ValueError(f"valid_mask shape {vm.shape} does not match image shape {(H, W)}")
    else:
        vm = None

    nl: Optional[np.ndarray]
    if nuclei_labels is not None:
        nl = np.asarray(nuclei_labels)
        if nl.shape != (H, W):
            raise ValueError(
                f"nuclei_labels shape {nl.shape} does not match image shape {(H, W)}"
            )
        nm = (nl > 0)
    else:
        nl = None
        nm = None

    # --- Build LoG filter (matching realtime script) ---
    zR = float(params.zR)
    lambda_nm = float(params.lambda_nm)
    pixel_size_nm = float(params.pixel_size_nm)
    if pixel_size_nm <= 0:
        raise ValueError("pixel_size_nm must be > 0")

    w0 = float(np.sqrt(lambda_nm * zR / np.pi))
    sigma0 = float(w0 / np.sqrt(2.0) / pixel_size_nm)
    n1 = int(np.ceil(sigma0 * 3.0) * 2.0 + 1.0)

    log_filter = _cached_log_filter(n1, sigma0)

    # --- Convolution + local maxima on -LoG (matching realtime script) ---
    try:
        from scipy.signal import convolve  # type: ignore
        from scipy.ndimage import maximum_filter  # type: ignore
    except Exception as e:
        raise ImportError("SciPy is required for LoG convolution and maximum_filter.") from e

    image_conv = convolve(img_f, log_filter, mode="valid")

    se_size = int(params.se_size)
    maxima = (-1.0 * image_conv) == maximum_filter(-1.0 * image_conv, size=se_size)

    pad_size = int(log_filter.shape[0] // 2)
    maxima_padded = np.pad(
        maxima,
        pad_width=((pad_size, pad_size), (pad_size, pad_size)),
        mode="constant",
        constant_values=0,
    )
    image_conv_padded = np.pad(
        image_conv,
        pad_width=((pad_size, pad_size), (pad_size, pad_size)),
        mode="constant",
        constant_values=0,
    )

    # Apply optional masks (mirrors `maxima_padded = maxima_padded * mask_crop` and
    # `maxima_padded = maxima_padded * nuclei_mask` in the realtime script).
    if vm is not None:
        maxima_padded = maxima_padded * vm
    if nm is not None:
        maxima_padded = maxima_padded * nm.astype(np.int32)

    ys_all, xs_all = np.where(maxima_padded)
    if ys_all.size == 0:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    quality = -1.0 * image_conv_padded[ys_all, xs_all]
    keep_q = quality > float(params.q_min)
    ys = ys_all[keep_q]
    xs = xs_all[keep_q]
    quality = quality[keep_q]

    if ys.size == 0:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    # --- Per-spot measurement masks (31x31 by default) ---
    wr = int(params.window_radius_px)
    win = 2 * wr + 1
    in5 = _disk_mask(int(params.in5_radius_px), win)
    in7 = _disk_mask(int(params.in7_radius_px), win)
    out0 = _ring_mask(int(params.ring_inner_radius_px), int(params.ring_outer_radius_px), win)

    rows = []
    window_range = np.arange(-wr, wr + 1)

    for y0, x0, q in zip(ys.tolist(), xs.tolist(), quality.tolist()):
        y0 = int(y0)
        x0 = int(x0)

        ys0 = y0 + window_range
        xs0 = x0 + window_range

        # Boundary check (exactly like realtime script): require full window inside image.
        if (
            ys0.min() < 0
            or xs0.min() < 0
            or ys0.max() >= H
            or xs0.max() >= W
        ):
            continue

        box = img_f[np.ix_(ys0, xs0)]
        bkg = float(np.median(box[out0]))
        mean_in5 = float(box[in5].mean())
        mean_in7 = float(box[in7].mean())

        u0 = mean_in5 - bkg
        u1 = mean_in7 - bkg

        if u0 <= float(params.u0_min):
            continue

        # SNR (not in realtime script, but required by Slice0 contract).
        bg_vals = box[out0].astype(float, copy=False)
        noise = _robust_std(bg_vals)
        if noise <= 1e-6:
            noise = float(np.std(bg_vals))
        if noise <= 1e-6:
            noise = 1e-6
        snr = float(u0 / noise)

        nucleus_label = int(nl[y0, x0]) if nl is not None else 0

        rows.append(
            dict(
                frame=0,
                y_px=float(y0),
                x_px=float(x0),
                intensity=float(u0),
                background=float(bkg),
                snr=float(snr),
                # extras (safe additions)
                u0=float(u0),
                u1=float(u1),
                quality=float(q),
                mean_in5=float(mean_in5),
                mean_in7=float(mean_in7),
                peak_intensity=float(img_f[y0, x0]),
                nucleus_label=int(nucleus_label),
                sigma_px=float(sigma0),
                log_size=int(log_filter.shape[0]),
            )
        )

    if not rows:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    # Keep required columns first (contract), then extras.
    df = pd.DataFrame(rows)
    ordered = REQUIRED_COLUMNS + [c for c in df.columns if c not in REQUIRED_COLUMNS]
    return df.loc[:, ordered]
