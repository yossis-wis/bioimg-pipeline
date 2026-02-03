from __future__ import annotations

"""Illumination-field uniformity metrics for MMF + field-stop simulations.

These helpers are used by the illumination-design notebooks to quantify
*where* nonuniformity matters:

1) **Inner ROI** (exclude the edge roll-off): residual speckle contrast that
   can bias single-molecule excitation and downstream detection.
2) **Edge band** (near the ROI boundary): residual speckle/ripple at the
   illumination boundary, which can matter if molecules near the edge are used
   in analysis or if you are judging top-hat quality.

The main metric returned is the usual speckle contrast

.. math::

    C = \sigma_I / \langle I\rangle,

computed over a user-defined mask.

Notes
-----
* The ROI is assumed to be a *square* of side ``roi_um`` centered on the array.
* All geometry is computed analytically from coordinates (no morphology
  dependency), so this stays lightweight.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SquareRoiMasks:
    """Boolean masks for ROI regions (all shape ``(n, n)``)."""

    roi: np.ndarray
    inner: np.ndarray
    edge_in: np.ndarray
    edge_out: np.ndarray


def square_roi_region_masks(
    *,
    n: int,
    dx_um: float,
    roi_um: float,
    inner_margin_um: float = 0.0,
    edge_band_um: float = 1.0,
) -> SquareRoiMasks:
    """Create boolean masks for inner and edge regions of a square ROI.

    Parameters
    ----------
    n:
        Grid size (assumes a square ``n×n`` field).
    dx_um:
        Sample-plane pixel pitch (µm/px).
    roi_um:
        Side length of the square ROI at the sample (µm).
    inner_margin_um:
        Margin excluded from each ROI edge to define the "inner" region (µm).
        If ``0``, the inner region equals the ROI.
    edge_band_um:
        Thickness of the edge band (µm). The edge-in band lives *inside* the ROI
        within this distance of the boundary; the edge-out band lives *outside*
        the ROI within this distance of the boundary.

    Returns
    -------
    SquareRoiMasks
        ``roi``, ``inner``, ``edge_in``, ``edge_out`` masks.
    """

    if n <= 0:
        raise ValueError("n must be > 0")
    if dx_um <= 0:
        raise ValueError("dx_um must be > 0")
    if roi_um <= 0:
        raise ValueError("roi_um must be > 0")
    if inner_margin_um < 0:
        raise ValueError("inner_margin_um must be >= 0")
    if edge_band_um <= 0:
        raise ValueError("edge_band_um must be > 0")

    half = 0.5 * float(roi_um)
    half_inner = half - float(inner_margin_um)
    if half_inner <= 0:
        raise ValueError("inner_margin_um is too large for this roi_um")

    coords_um = (np.arange(n) - n // 2) * float(dx_um)
    x_um, y_um = np.meshgrid(coords_um, coords_um, indexing="xy")

    ax = np.abs(x_um)
    ay = np.abs(y_um)

    roi = (ax <= half) & (ay <= half)
    inner = (ax <= half_inner) & (ay <= half_inner)

    # Edge-inside band: pixels inside ROI and within edge_band_um of any edge.
    band_start = max(0.0, half - float(edge_band_um))
    edge_in = roi & ((ax >= band_start) | (ay >= band_start))

    # Edge-outside band: pixels outside ROI but within edge_band_um of the ROI boundary.
    half_out = half + float(edge_band_um)
    edge_out = (~roi) & (ax <= half_out) & (ay <= half_out)

    # Defensive: guarantee dtypes and non-overlap.
    roi = roi.astype(bool, copy=False)
    inner = inner.astype(bool, copy=False)
    edge_in = edge_in.astype(bool, copy=False)
    edge_out = edge_out.astype(bool, copy=False)

    if np.any(roi & edge_out):
        raise RuntimeError("edge_out overlaps roi; mask logic bug")

    return SquareRoiMasks(roi=roi, inner=inner, edge_in=edge_in, edge_out=edge_out)


def masked_mean(x: np.ndarray, mask: np.ndarray) -> float:
    """Mean of ``x`` over ``mask``.

    Raises if the mask is empty.
    """

    if x.shape != mask.shape:
        raise ValueError("x and mask must have the same shape")
    if not np.any(mask):
        raise ValueError("mask is empty")
    return float(np.mean(x[mask]))


def masked_std(x: np.ndarray, mask: np.ndarray) -> float:
    """Standard deviation of ``x`` over ``mask``.

    Raises if the mask is empty.
    """

    if x.shape != mask.shape:
        raise ValueError("x and mask must have the same shape")
    if not np.any(mask):
        raise ValueError("mask is empty")
    return float(np.std(x[mask]))


def speckle_contrast(intensity: np.ndarray, mask: np.ndarray) -> float:
    """Speckle contrast ``C = std/mean`` over a region."""

    mu = masked_mean(intensity, mask)
    if mu == 0:
        raise ValueError("masked mean is zero; cannot compute contrast")
    return masked_std(intensity, mask) / mu


def normalize_by_mask_mean(intensity: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return ``intensity`` normalized by the mean over ``mask``."""

    mu = masked_mean(intensity, mask)
    if mu == 0:
        raise ValueError("masked mean is zero; cannot normalize")
    return intensity / mu
