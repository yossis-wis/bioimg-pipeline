from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import tifffile


@dataclass(frozen=True)
class PhantomParams:
    height: int = 512
    width: int = 512
    num_spots: int = 40
    sigma_px: float = 1.5
    min_separation_px: float = 8.0  # helps keep peaks visually distinct
    background_level: float = 100.0
    noise_sigma: float = 10.0
    intensity_min: float = 800.0
    intensity_max: float = 2500.0
    seed: int = 42


def _place_spots_separated(
    rng: np.random.Generator,
    height: int,
    width: int,
    num_spots: int,
    margin: int,
    min_sep: float,
    max_attempts: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rejection-sample spot centers with a minimum separation.

    Returns fewer than num_spots if it can't satisfy constraints.
    """
    ys: list[float] = []
    xs: list[float] = []
    for _ in range(num_spots):
        placed = False
        for _attempt in range(max_attempts):
            y = float(rng.uniform(margin, height - margin))
            x = float(rng.uniform(margin, width - margin))
            if not ys:
                ys.append(y); xs.append(x); placed = True; break
            d2 = (np.asarray(ys) - y) ** 2 + (np.asarray(xs) - x) ** 2
            if float(np.min(d2)) >= (min_sep ** 2):
                ys.append(y); xs.append(x); placed = True; break
        if not placed:
            break
    return np.asarray(ys, dtype=float), np.asarray(xs, dtype=float)


def generate_phantom_image(params: PhantomParams) -> np.ndarray:
    """Generate a synthetic microscopy-like 2D image as uint16.

    Spots are rendered as small 2D Gaussians + background + Gaussian noise.

    This is intentionally simple; the purpose is to validate:
      TIFF I/O -> kernel -> parquet + manifest + QC image
    """
    if params.height <= 0 or params.width <= 0:
        raise ValueError("height/width must be positive")
    if params.sigma_px <= 0:
        raise ValueError("sigma_px must be positive")
    if params.num_spots < 0:
        raise ValueError("num_spots must be >= 0")

    rng = np.random.default_rng(params.seed)

    # Keep spots away from edges so the Gaussian support doesn't clip.
    margin = int(max(1, np.ceil(3.0 * params.sigma_px)))

    # Choose centers (float centers are fine; detector will report integer maxima)
    ys, xs = _place_spots_separated(
        rng=rng,
        height=params.height,
        width=params.width,
        num_spots=params.num_spots,
        margin=margin,
        min_sep=float(params.min_separation_px),
    )

    # Background + noise in float
    img = np.full((params.height, params.width), float(params.background_level), dtype=np.float32)
    img += rng.normal(0.0, float(params.noise_sigma), size=img.shape).astype(np.float32)

    # Render each Gaussian in a small local patch for efficiency
    half = int(max(2, np.ceil(3.0 * params.sigma_px)))
    yy = np.arange(-half, half + 1, dtype=np.float32)
    xx = np.arange(-half, half + 1, dtype=np.float32)
    XX, YY = np.meshgrid(xx, yy)
    denom = 2.0 * (float(params.sigma_px) ** 2)

    for y0, x0 in zip(ys, xs):
        amp = float(rng.uniform(float(params.intensity_min), float(params.intensity_max)))
        g = amp * np.exp(-((XX ** 2 + YY ** 2) / denom)).astype(np.float32)

        y_c = int(round(y0))
        x_c = int(round(x0))
        y0i = max(0, y_c - half)
        y1i = min(params.height, y_c + half + 1)
        x0i = max(0, x_c - half)
        x1i = min(params.width, x_c + half + 1)

        gy0 = (y0i - (y_c - half))
        gx0 = (x0i - (x_c - half))
        gy1 = gy0 + (y1i - y0i)
        gx1 = gx0 + (x1i - x0i)

        img[y0i:y1i, x0i:x1i] += g[gy0:gy1, gx0:gx1]

    # Clip to uint16 range to mimic camera output
    img = np.clip(img, 0, 65535).astype(np.uint16)
    return img


def write_phantom_tiff(output_path: Path, params: PhantomParams, overwrite: bool = False) -> Path:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {output_path}. " 
            "Pass --overwrite if you intend to replace it."
        )

    img = generate_phantom_image(params)
    tifffile.imwrite(str(output_path), img)
    return output_path
