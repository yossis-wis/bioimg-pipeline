from __future__ import annotations

import numpy as np

from src.illumination_speckle_metrics import (
    normalize_by_mask_mean,
    speckle_contrast,
    square_roi_region_masks,
)


def test_square_roi_region_masks_basic_properties() -> None:
    masks = square_roi_region_masks(
        n=101,
        dx_um=0.1,
        roi_um=5.0,
        inner_margin_um=0.5,
        edge_band_um=0.3,
    )

    assert masks.roi.shape == (101, 101)
    assert masks.inner.shape == (101, 101)
    assert masks.edge_in.shape == (101, 101)
    assert masks.edge_out.shape == (101, 101)

    assert masks.roi.dtype == bool
    assert masks.inner.dtype == bool

    # Inner must be a strict subset of ROI when margin > 0.
    assert int(masks.inner.sum()) < int(masks.roi.sum())
    assert np.all(masks.roi[masks.inner])

    # Edge bands should be non-empty and not overlap.
    assert int(masks.edge_in.sum()) > 0
    assert int(masks.edge_out.sum()) > 0
    assert int((masks.roi & masks.edge_out).sum()) == 0


def test_contrast_and_normalization_on_constant_field() -> None:
    masks = square_roi_region_masks(n=51, dx_um=0.2, roi_um=4.0, inner_margin_um=0.5, edge_band_um=0.4)
    I = np.ones((51, 51), dtype=float)

    I_norm = normalize_by_mask_mean(I, masks.inner)
    assert np.allclose(I_norm[masks.inner], 1.0)

    c = speckle_contrast(I_norm, masks.inner)
    assert np.isclose(c, 0.0)


def test_contrast_increases_with_added_noise() -> None:
    masks = square_roi_region_masks(n=101, dx_um=0.1, roi_um=6.0, inner_margin_um=0.5, edge_band_um=0.5)
    rng = np.random.default_rng(0)
    I = 1.0 + 0.2 * rng.standard_normal((101, 101))
    I = np.clip(I, 1e-6, None)

    c = speckle_contrast(I, masks.inner)
    assert c > 0.0
