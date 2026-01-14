from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import tifffile
import yaml
from skimage.draw import disk as draw_disk

# Allow running without packaging
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _data_root() -> Path:
    root = os.environ.get("BIOIMG_DATA_ROOT")
    if not root:
        raise RuntimeError("BIOIMG_DATA_ROOT is not set (see docs/SETUP_WINDOWS.md)")
    return Path(root).expanduser().resolve()


def generate_integrated_phantom(cfg: Dict[str, Any]) -> np.ndarray:
    """Generate a (2, H, W) phantom for the integrated pipeline.

    Channel 0: nuclei-like blobs
    Channel 1: small spots, mostly inside nuclei
    """
    height = int(cfg.get("sim_height", 512))
    width = int(cfg.get("sim_width", 512))
    n_nuclei = int(cfg.get("sim_num_nuclei", 15))
    n_spots = int(cfg.get("sim_num_spots", 100))
    seed = int(cfg.get("sim_seed", 42))

    rng = np.random.default_rng(seed)

    nuclei_img = np.full((height, width), 100.0, dtype=np.float32)
    nuclei_img += rng.normal(0, 10, size=(height, width))

    nuclei_centers: list[tuple[int, int]] = []
    nuclei_radii: list[int] = []

    for _ in range(n_nuclei):
        radius = int(rng.integers(20, 40))
        y = int(rng.integers(radius, height - radius))
        x = int(rng.integers(radius, width - radius))

        rr, cc = draw_disk((y, x), radius, shape=(height, width))
        nuclei_img[rr, cc] += rng.uniform(1000, 2000)

        nuclei_centers.append((y, x))
        nuclei_radii.append(radius)

    spots_img = np.full((height, width), 100.0, dtype=np.float32)
    spots_img += rng.normal(0, 10, size=(height, width))

    def add_spot(y: float, x: float) -> None:
        sigma = 1.5
        amp = float(rng.uniform(800, 2500))
        size = int(np.ceil(3 * sigma))
        yy, xx = np.mgrid[-size : size + 1, -size : size + 1]
        g = np.exp(-(yy**2 + xx**2) / (2 * sigma**2))

        y0 = int(y) - size
        x0 = int(x) - size
        if y0 < 0 or x0 < 0 or y0 + g.shape[0] >= height or x0 + g.shape[1] >= width:
            return

        spots_img[y0 : y0 + g.shape[0], x0 : x0 + g.shape[1]] += amp * g

    num_inside = int(n_spots * 0.7)
    for _ in range(num_inside):
        idx = int(rng.integers(0, n_nuclei))
        ny, nx = nuclei_centers[idx]
        nr = nuclei_radii[idx]
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(0, nr * 0.8)
        sy = ny + dist * np.sin(angle)
        sx = nx + dist * np.cos(angle)
        add_spot(sy, sx)

    for _ in range(n_spots - num_inside):
        sy = int(rng.integers(10, height - 10))
        sx = int(rng.integers(10, width - 10))
        add_spot(sy, sx)

    stack = np.stack([nuclei_img, spots_img])
    return np.clip(stack, 0, 65535).astype(np.uint16)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate integrated phantom TIFF")
    ap.add_argument("--config", default="configs/integrated_sim.yaml")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (REPO_ROOT / cfg_path).resolve()

    cfg = _load_config(cfg_path)
    data_root = _data_root()

    input_relpath = cfg.get("input_relpath")
    if not input_relpath:
        raise ValueError("config missing required key: input_relpath")
    out_path = (data_root / input_relpath).resolve()

    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"File exists: {out_path}. Use --overwrite.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = generate_integrated_phantom(cfg)
    tifffile.imwrite(str(out_path), img, imagej=True, metadata={"axes": "CYX"})

    print(f"Wrote integrated phantom: {out_path}")
    print(f"Shape: {img.shape} (C, Y, X)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
