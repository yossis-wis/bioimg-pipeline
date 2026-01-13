"""Slice1 kernel: nucleus segmentation.

Slice1 goal
-----------
Given a single 2D fluorescence image (nuclear marker), segment nuclei and return
an integer label image.

This is intentionally minimal:
- input: 2D array
- output: 2D label image (0=background, 1..N=nuclei)

No filesystem I/O here (that's handled by drivers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Slice1NucleiParams:
    """Parameters for StarDist nucleus segmentation."""

    normalize_pmin: float = 1.0
    normalize_pmax: float = 99.8

    # If None, use model defaults (from thresholds.json)
    prob_thresh: Optional[float] = None
    nms_thresh: Optional[float] = None


def segment_nuclei_stardist(
    image_2d: np.ndarray,
    model: Any,
    params: Slice1NucleiParams,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Segment nuclei using a loaded StarDist2D model.

    Parameters
    ----------
    image_2d:
        2D input image (Y, X). Any numeric dtype.
    model:
        Loaded StarDist2D model (I/O handled by driver).
    params:
        Segmentation parameters.

    Returns
    -------
    labels:
        2D integer label image.
    meta:
        Small dict of metadata used for the run (thresholds actually used).
    """

    try:
        from csbdeep.utils import normalize  # type: ignore
    except Exception as e:
        raise ImportError(
            "csbdeep is not installed. Install Slice1 dependencies, e.g.:\n\n"
            "  pip install csbdeep\n"
        ) from e

    img = image_2d.astype(np.float32, copy=False)
    img_n = normalize(img, params.normalize_pmin, params.normalize_pmax, axis=None, clip=True)

    kwargs: Dict[str, float] = {}
    if params.prob_thresh is not None:
        kwargs["prob_thresh"] = float(params.prob_thresh)
    if params.nms_thresh is not None:
        kwargs["nms_thresh"] = float(params.nms_thresh)

    labels, _details = model.predict_instances(img_n, **kwargs)

    # Ensure predictable dtype for downstream steps
    labels = labels.astype(np.int32, copy=False)

    # What thresholds were *actually* used?
    used: Dict[str, float] = {}
    if "prob_thresh" in kwargs:
        used["prob_thresh"] = float(kwargs["prob_thresh"])
    if "nms_thresh" in kwargs:
        used["nms_thresh"] = float(kwargs["nms_thresh"])

    return labels, used
