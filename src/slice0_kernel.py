"""Slice0 kernel: spot detection.

This kernel is intentionally **pure computation** (no filesystem I/O).

Spot detection is split into two conceptual stages:

A) **Candidate generation (TrackMate-style LoG detector)**

   We port the core behavior of TrackMate's *Laplacian of Gaussian (LoG) detector*:

   1) Build a calibrated LoG kernel tuned to a target blob radius.
   2) Convolve the image with this kernel (FFT-based convolution; output is same
      size as the input).
   3) Detect *strict* local maxima of the LoG response in a 3×3 neighborhood.

   In TrackMate, these steps are implemented in the `LogDetector` and
   `DetectionUtils` classes (LoG kernel + local maxima), with optional median
   filtering and optional sub-pixel localization.

   Attribution note: TrackMate is a Fiji plugin distributed under the GNU GPL v3.
   This repo contains a Python reimplementation intended to match TrackMate behavior
   for reproducibility. See docs/SPOT_DETECTION.md.

B) **Per-candidate photometry (kept unchanged from your repo / realtime script)**

   For each candidate maximum that survives optional masks (valid_mask, nuclei),
   we compute:

     - background = median in a thin ring mask (out0)
     - mean_in5   = mean inside a small disk mask (in5)
     - mean_in7   = mean inside a slightly larger disk mask (in7)
     - u0 = mean_in5 - background
     - u1 = mean_in7 - background

   and we keep spots with u0 > u0_min.

Only stage (A) is upgraded here; stage (B) is left byte-for-byte identical in
spirit (and line-for-line identical in implementation) to preserve your
experimentally calibrated intensity logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "frame",
    "y_px",
    "x_px",
    "intensity",
    "background",
    "snr",
    "nucleus_label",
]


@dataclass(frozen=True)
class Slice0Params:
    """Parameters for LoG-based spot detection.

    Notes
    -----
    - Candidate generation follows TrackMate's LoG detector: calibrated LoG kernel,
      FFT convolution, and strict local maxima in a 3×3 neighborhood.
    - The downstream photometry step (in5/out0 masks → u0 threshold) is intentionally
      unchanged.

    Parameter mapping vs TrackMate
    ------------------------------
    - TrackMate "radius" (in calibrated units) is taken to be `spot_radius_nm` if
      provided; otherwise we derive a radius from (zR, lambda_nm) via the Rayleigh
      relation and set:

        radius_nm = w0_nm,  where  z_R = π w0^2 / λ.

      For a Gaussian intensity profile I(r) ∝ exp(-2 r^2 / w0^2), the corresponding
      Gaussian σ satisfies σ = w0 / √2, which is consistent with TrackMate's internal
      choice σ = radius / √nDims (here nDims=2).

    - TrackMate "threshold" corresponds to `q_min` here (threshold on LoG response
      at the candidate peak).
    """

    # --- Optics-ish radius definition (kept for backward compatibility) ---
    zR: float = 344.5
    lambda_nm: float = 667.0
    pixel_size_nm: float = 65.0

    # Optional direct override of the TrackMate-style radius (calibrated units: nm here).
    # If None, we compute radius_nm = w0_nm from (zR, lambda_nm).
    spot_radius_nm: Optional[float] = None

    # TrackMate options (defaults match your previous pipeline unless changed in config).
    do_median_filter: bool = False
    do_subpixel_localization: bool = False  # currently only exported as extra columns

    # --- Candidate maxima + quality threshold ---
    # Threshold on LoG response at the candidate location (TrackMate "threshold").
    q_min: float = 1.0

    # Size of the non-max suppression neighborhood.
    # TrackMate uses a strict 3×3 neighborhood (RectangleShape(1)).
    se_size: int = 3

    # --- Per-spot measurements (UNCHANGED; mirrors realtime script) ---
    window_radius_px: int = 15  # -> 31x31 window
    in5_radius_px: int = 2
    in7_radius_px: int = 3
    ring_outer_radius_px: int = 10
    ring_inner_radius_px: int = 9

    # Keep spot if (mean(in5) - background) > u0_min
    u0_min: float = 30.0


@dataclass(frozen=True)
class Slice0Debug:
    """Intermediate arrays from Slice0 spot detection (for QC / notebooks).

    This is intended for *interactive* debugging of a single plane. It contains
    the LoG kernel, LoG response maps, and the candidate maxima maps before and
    after masks / thresholds.

    Notes
    -----
    - Arrays are returned by reference (no defensive copies).
    - Shapes follow the implementation in :func:`detect_spots`.
    """

    # Input image (float32 view of the provided image_2d)
    img_f: np.ndarray

    # Image shape
    H: int
    W: int

    # Optics-derived LoG parameters / TrackMate radius mapping
    w0_nm: float          # radius_nm actually used (nm)
    sigma0_px: float      # σ in pixels (σ = radius / √2 / pixel_size_nm for 2D)
    log_filter: np.ndarray
    pad_size: int         # kernel half-size (informational; convolution is "same")

    # LoG response (same size as input image)
    image_conv_valid: np.ndarray
    image_conv_padded: np.ndarray

    # Local maxima on LoG response (same size as image)
    maxima_valid: np.ndarray
    maxima_padded_raw: np.ndarray
    maxima_padded_masked: np.ndarray

    # Candidate maxima coordinates
    ys_raw: np.ndarray
    xs_raw: np.ndarray
    ys_masked: np.ndarray
    xs_masked: np.ndarray

    # Quality at masked maxima (= LoG response at the candidate location)
    quality_masked: np.ndarray

    # Quality thresholding (q_min)
    keep_q: np.ndarray
    ys_q: np.ndarray
    xs_q: np.ndarray
    quality_q: np.ndarray

    # Per-spot measurement masks (window_size = 2*window_radius_px + 1)
    window_radius_px: int
    in5_mask: np.ndarray
    in7_mask: np.ndarray
    out0_mask: np.ndarray

    # Masks applied (may be None)
    valid_mask: Optional[np.ndarray]
    nuclei_mask: Optional[np.ndarray]


def _robust_std(x: np.ndarray) -> float:
    """Robust std via MAD; returns 0.0 if too few samples."""
    x = np.asarray(x, dtype=float)
    if x.size < 5:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad)


def _trackmate_log_kernel(radius: float, calibration: tuple[float, float]) -> np.ndarray:
    """Create a 2D LoG kernel using the same formula as TrackMate.

    TrackMate builds a calibrated LoG kernel tuned for a target blob radius.
    Internally it uses:

        σ = radius / √nDims,

    and maps the kernel onto the pixel grid using `calibration` (pixel sizes).
    See TrackMate docs for the high-level behavior.
    """
    radius = float(radius)
    if radius <= 0:
        raise ValueError("radius must be > 0")

    cal_y, cal_x = (float(calibration[0]), float(calibration[1]))
    if cal_y <= 0 or cal_x <= 0:
        raise ValueError("calibration values must be > 0")

    n_dims = 2
    sigma = radius / np.sqrt(n_dims)  # calibrated units (nm here)

    sigma_px_y = sigma / cal_y
    sigma_px_x = sigma / cal_x
    sigma_px = (sigma_px_y, sigma_px_x)

    # Kernel half-sizes match TrackMate's logic:
    # hksize = max(2, int(3*sigmaPixels + 0.5) + 1)
    hks_y = max(2, int(3.0 * sigma_px_y + 0.5) + 1)
    hks_x = max(2, int(3.0 * sigma_px_x + 0.5) + 1)

    size_y = 3 + 2 * hks_y
    size_x = 3 + 2 * hks_x
    mid_y = 1 + hks_y
    mid_x = 1 + hks_x

    # Vectorized kernel computation (double precision, cast to float32 at end).
    yy, xx = np.indices((size_y, size_x), dtype=np.float64)
    y_phys = cal_y * (yy - mid_y)
    x_phys = cal_x * (xx - mid_x)

    sumx2 = y_phys * y_phys + x_phys * x_phys

    mantissa = (
        (1.0 / (sigma_px_y * sigma_px_y)) * (y_phys * y_phys / (sigma * sigma) - 1.0)
        + (1.0 / (sigma_px_x * sigma_px_x)) * (x_phys * x_phys / (sigma * sigma) - 1.0)
    )

    exponent = -sumx2 / (2.0 * sigma * sigma)

    # TrackMate uses C = 1/(π σ_px0^2) where σ_px0 is the first dimension.
    C = 1.0 / (np.pi * sigma_px[0] * sigma_px[0])

    kernel = -C * mantissa * np.exp(exponent)
    return kernel.astype(np.float32, copy=False)


@lru_cache(maxsize=64)
def _cached_trackmate_log_kernel(radius: float, cal_y: float, cal_x: float) -> np.ndarray:
    """Cached TrackMate-style LoG kernel (2D)."""
    h = _trackmate_log_kernel(radius, (cal_y, cal_x))
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


def _strict_local_maxima_2d(img: np.ndarray, se_size: int) -> np.ndarray:
    """Strict local maxima in a square neighborhood, excluding the center pixel.

    TrackMate's LoG detector finds local maxima in a 3×3 neighborhood (span=1).
    We keep `se_size` as a (mostly) backwards-compatible knob, but **TrackMate
    equivalence corresponds to se_size=3**.

    Implementation: a pixel is a strict local maximum if it is strictly greater
    than the maximum of its neighbors (excluding itself).
    """
    try:
        from scipy.ndimage import maximum_filter  # type: ignore
    except Exception as e:
        raise ImportError("SciPy is required for maximum_filter.") from e

    se_size = int(se_size)
    if se_size < 3:
        raise ValueError("se_size must be >= 3")
    if se_size % 2 == 0:
        se_size += 1

    footprint = np.ones((se_size, se_size), dtype=bool)
    footprint[se_size // 2, se_size // 2] = False

    neigh_max = maximum_filter(img, footprint=footprint, mode="mirror")
    return img > neigh_max


def _subpixel_refine_parabolic(resp: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Very small, TrackMate-inspired subpixel refinement using 1D parabolic fits.

    This is **not** a full port of ImgLib2's SubpixelLocalization, but it captures
    the common quadratic-vertex refinement used in LoG-based detection pipelines.

    Returns
    -------
    (y_sub, x_sub): float arrays of refined coordinates (same length as ys/xs).
    """
    ys = np.asarray(ys, dtype=int)
    xs = np.asarray(xs, dtype=int)

    H, W = resp.shape
    y_sub = ys.astype(np.float32)
    x_sub = xs.astype(np.float32)

    # Only refine where we have a 3×3 neighborhood inside bounds.
    ok = (ys >= 1) & (ys <= H - 2) & (xs >= 1) & (xs <= W - 2)
    if not np.any(ok):
        return y_sub, x_sub

    y0 = ys[ok]
    x0 = xs[ok]

    f0 = resp[y0, x0]
    fxm1 = resp[y0, x0 - 1]
    fxp1 = resp[y0, x0 + 1]
    fym1 = resp[y0 - 1, x0]
    fyp1 = resp[y0 + 1, x0]

    denom_x = fxm1 - 2.0 * f0 + fxp1
    denom_y = fym1 - 2.0 * f0 + fyp1

    # Vertex offset for parabola through (-1,0,+1): δ = (f(-1) - f(+1)) / (2*(f(-1) - 2f(0) + f(+1))).
    dx = np.zeros_like(f0, dtype=np.float32)
    dy = np.zeros_like(f0, dtype=np.float32)

    nzx = np.abs(denom_x) > 1e-12
    nzy = np.abs(denom_y) > 1e-12

    dx[nzx] = 0.5 * (fxm1[nzx] - fxp1[nzx]) / denom_x[nzx]
    dy[nzy] = 0.5 * (fym1[nzy] - fyp1[nzy]) / denom_y[nzy]

    # Clamp (avoid pathological jumps).
    dx = np.clip(dx, -1.0, 1.0)
    dy = np.clip(dy, -1.0, 1.0)

    x_sub[ok] = x0.astype(np.float32) + dx
    y_sub[ok] = y0.astype(np.float32) + dy
    return y_sub, x_sub


def _detect_spots_core(
    image_2d: np.ndarray,
    params: Slice0Params,
    *,
    valid_mask: Optional[np.ndarray] = None,
    nuclei_labels: Optional[np.ndarray] = None,
    return_debug: bool = False,
) -> tuple[pd.DataFrame, Optional[Slice0Debug]]:
    """Shared implementation for detect_spots() and detect_spots_debug()."""

    if image_2d.ndim != 2:
        raise ValueError(f"detect_spots expects a 2D array; got shape={image_2d.shape}")

    img = np.asarray(image_2d)
    img_f = img.astype(np.float32, copy=False)
    H, W = img_f.shape

    if valid_mask is not None:
        vm = np.asarray(valid_mask).astype(bool, copy=False)
        if vm.shape != (H, W):
            raise ValueError(
                f"valid_mask shape {vm.shape} does not match image shape {(H, W)}"
            )
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

    # --- TrackMate-style LoG kernel (calibrated) ---
    zR = float(params.zR)
    lambda_nm = float(params.lambda_nm)
    pixel_size_nm = float(params.pixel_size_nm)
    if pixel_size_nm <= 0:
        raise ValueError("pixel_size_nm must be > 0")

    # Rayleigh range relation: z_R = π w0^2 / λ  =>  w0 = sqrt(λ z_R / π)
    w0_from_optics = float(np.sqrt(lambda_nm * zR / np.pi))

    radius_nm = float(params.spot_radius_nm) if params.spot_radius_nm is not None else w0_from_optics
    if radius_nm <= 0:
        raise ValueError("spot_radius_nm (or derived w0) must be > 0")

    # TrackMate uses σ = radius / √nDims, with nDims=2 here.
    sigma0 = float(radius_nm / np.sqrt(2.0) / pixel_size_nm)

    log_filter = _cached_trackmate_log_kernel(radius_nm, pixel_size_nm, pixel_size_nm)
    pad_size = int(log_filter.shape[0] // 2)

    # --- Convolution (FFT) ---
    try:
        from scipy.signal import fftconvolve  # type: ignore
        from scipy.ndimage import median_filter  # type: ignore
    except Exception as e:
        raise ImportError("SciPy is required for LoG FFT convolution and median_filter.") from e

    img_for_conv = img_f
    if bool(params.do_median_filter):
        # TrackMate applies a simple 3×3 median filter when requested.
        img_for_conv = median_filter(img_for_conv, size=3)

    resp = fftconvolve(img_for_conv, log_filter, mode="same")

    # --- Local maxima (strict) ---
    maxima = _strict_local_maxima_2d(resp, params.se_size)

    maxima_masked = maxima
    if vm is not None:
        maxima_masked = maxima_masked & vm
    if nm is not None:
        maxima_masked = maxima_masked & nm

    ys_raw, xs_raw = np.where(maxima)
    ys_all, xs_all = np.where(maxima_masked)

    # Quality is defined on masked maxima
    quality_all = resp[ys_all, xs_all] if ys_all.size else np.asarray([])
    keep_q = quality_all > float(params.q_min) if ys_all.size else np.asarray([], dtype=bool)

    ys = ys_all[keep_q] if ys_all.size else np.asarray([], dtype=int)
    xs = xs_all[keep_q] if xs_all.size else np.asarray([], dtype=int)
    quality = quality_all[keep_q] if ys_all.size else np.asarray([], dtype=float)

    # Optional subpixel refinement (export-only; photometry remains pixel-centered)
    y_sub = x_sub = None
    if bool(params.do_subpixel_localization) and ys.size:
        y_sub, x_sub = _subpixel_refine_parabolic(resp, ys, xs)

    # --- Per-spot measurement masks (31x31 by default) ---
    wr = int(params.window_radius_px)
    win = 2 * wr + 1
    in5 = _disk_mask(int(params.in5_radius_px), win)
    in7 = _disk_mask(int(params.in7_radius_px), win)
    out0 = _ring_mask(int(params.ring_inner_radius_px), int(params.ring_outer_radius_px), win)

    debug_obj: Optional[Slice0Debug] = None
    if return_debug:
        debug_obj = Slice0Debug(
            img_f=img_f,
            H=int(H),
            W=int(W),
            w0_nm=float(radius_nm),
            sigma0_px=float(sigma0),
            log_filter=log_filter,
            pad_size=int(pad_size),
            image_conv_valid=resp,
            image_conv_padded=resp,
            maxima_valid=maxima,
            maxima_padded_raw=maxima,
            maxima_padded_masked=maxima_masked,
            ys_raw=ys_raw,
            xs_raw=xs_raw,
            ys_masked=ys_all,
            xs_masked=xs_all,
            quality_masked=quality_all,
            keep_q=keep_q,
            ys_q=ys,
            xs_q=xs,
            quality_q=quality,
            window_radius_px=int(wr),
            in5_mask=in5,
            in7_mask=in7,
            out0_mask=out0,
            valid_mask=vm,
            nuclei_mask=nm,
        )

    if ys.size == 0:
        return pd.DataFrame(columns=REQUIRED_COLUMNS), debug_obj

    rows = []
    window_range = np.arange(-wr, wr + 1)

    for idx, (y0, x0, q) in enumerate(zip(ys.tolist(), xs.tolist(), quality.tolist())):
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

        row = dict(
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
            log_radius_nm=float(radius_nm),
        )

        # Export subpixel (if computed) without changing photometry centering.
        if y_sub is not None and x_sub is not None:
            row["y_subpx"] = float(y_sub[idx])
            row["x_subpx"] = float(x_sub[idx])

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=REQUIRED_COLUMNS), debug_obj

    # Keep required columns first (contract), then extras.
    df = pd.DataFrame(rows)
    ordered = REQUIRED_COLUMNS + [c for c in df.columns if c not in REQUIRED_COLUMNS]
    return df.loc[:, ordered], debug_obj


def detect_spots(
    image_2d: np.ndarray,
    params: Slice0Params,
    *,
    valid_mask: Optional[np.ndarray] = None,
    nuclei_labels: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Detect spots in a single 2D image.

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

    df, _ = _detect_spots_core(
        image_2d,
        params,
        valid_mask=valid_mask,
        nuclei_labels=nuclei_labels,
        return_debug=False,
    )
    return df


def detect_spots_debug(
    image_2d: np.ndarray,
    params: Slice0Params,
    *,
    valid_mask: Optional[np.ndarray] = None,
    nuclei_labels: Optional[np.ndarray] = None,
) -> tuple[pd.DataFrame, Slice0Debug]:
    """Detect spots and also return intermediate arrays for QC notebooks.

    This is a convenience wrapper around the same implementation used by
    :func:`detect_spots`.

    Returns
    -------
    (df, debug)
        df: same as :func:`detect_spots`
        debug: :class:`Slice0Debug` with intermediate arrays.
    """

    df, dbg = _detect_spots_core(
        image_2d,
        params,
        valid_mask=valid_mask,
        nuclei_labels=nuclei_labels,
        return_debug=True,
    )
    assert dbg is not None
    return df, dbg

