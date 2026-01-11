from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import yaml

# Allow running without packaging
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT / "src"))

from simulate_phantom import PhantomParams, write_phantom_tiff  # noqa: E402


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _data_root() -> Path:
    root = os.environ.get("BIOIMG_DATA_ROOT")
    if not root:
        raise RuntimeError("BIOIMG_DATA_ROOT is not set (see docs/SETUP_WINDOWS.md)")
    return Path(root).expanduser().resolve()


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a synthetic TIFF into the data bench (raw_staging/...)")
    ap.add_argument("--config", default="configs/sim.yaml", help="Path to YAML config (relative or absolute)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing TIFF if present")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (REPO_ROOT / cfg_path).resolve()

    cfg = _load_config(cfg_path)
    data_root = _data_root()

    input_relpath = cfg.get("input_relpath")
    if not input_relpath:
        raise ValueError("config missing required key: input_relpath (where the TIFF will be written)")
    out_path = (data_root / input_relpath).resolve()

    params = PhantomParams(
        height=int(cfg.get("sim_height", 512)),
        width=int(cfg.get("sim_width", 512)),
        num_spots=int(cfg.get("sim_num_spots", 40)),
        sigma_px=float(cfg.get("sim_sigma_px", 1.5)),
        min_separation_px=float(cfg.get("sim_min_separation_px", 8.0)),
        background_level=float(cfg.get("sim_background_level", 100.0)),
        noise_sigma=float(cfg.get("sim_noise_sigma", 10.0)),
        intensity_min=float(cfg.get("sim_intensity_min", 800.0)),
        intensity_max=float(cfg.get("sim_intensity_max", 2500.0)),
        seed=int(cfg.get("sim_seed", 42)),
    )

    out_path = write_phantom_tiff(out_path, params, overwrite=bool(args.overwrite))
    print(f"Wrote phantom TIFF: {out_path}")
    print(f"  shape: ({params.height}, {params.width}), dtype: uint16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
