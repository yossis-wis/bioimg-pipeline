from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
import yaml

# Parquet support (required by docs/CONTRACTS.md)
try:
    import pyarrow  # noqa: F401
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "pyarrow is required to write spots.parquet. "
        "Install the conda env from environment.yml and re-run scripts/verify_setup.py."
    ) from e

# Matplotlib: write PNGs without needing a display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Allow running without packaging
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT / "src"))

from slice0_kernel import Slice0Params, detect_spots, REQUIRED_COLUMNS  # noqa: E402


def _cfg_first(cfg: Dict[str, Any], keys: list[str], default: Any) -> Any:
    """Return the first present key in `keys`, else default.

    This helps us accept a few common alias names from notebooks/legacy scripts
    (e.g. lambda0_nm vs lambda_nm).
    """
    for k in keys:
        if k in cfg and cfg[k] is not None:
            return cfg[k]
    return default


def _load_optional_tiff_2d(path: Path, *, channel: Optional[int] = None) -> np.ndarray:
    """Load a TIFF and best-effort select a 2D plane."""
    arr = tifffile.imread(str(path))
    return _select_2d_plane(np.asarray(arr), channel=channel)


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _data_root() -> Path:
    root = os.environ.get("BIOIMG_DATA_ROOT")
    if not root:
        raise RuntimeError("BIOIMG_DATA_ROOT is not set (see docs/SETUP_WINDOWS.md)")
    return Path(root).expanduser().resolve()


def _try_git_commit(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def _select_2d_plane(arr: np.ndarray, channel: Optional[int]) -> np.ndarray:
    """Best-effort: choose a 2D plane for Slice0.

    Rules:
      - If arr is already 2D -> return it.
      - If arr is 3D and channel is provided -> treat axis0 as channel-like if possible.
      - Otherwise, fall back to the first plane (arr[0]).
    """
    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        if channel is None:
            return arr[0]
        # treat channel as 1-based if >=1; allow channel=0 to mean first channel
        ch = int(channel)
        idx = ch - 1 if ch >= 1 else 0
        if 0 <= idx < arr.shape[0]:
            return arr[idx]
        return arr[0]

    # For higher dims, flatten leading dims and take first plane
    flat = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
    return flat[0]


def _read_ims_2d(input_path: Path, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read a 2D plane from an Imaris .ims file.

    Imaris .ims is HDF5-backed. In many files, the pixel data lives at:

        DataSet/ResolutionLevel <R>/TimePoint <T>/Channel <C>/Data

    where the dataset typically has shape (Z, Y, X) (or sometimes (Y, X)).

    Config keys (all optional):
      - channel: 1-based (1 => Channel 0, 2 => Channel 1, ...)
      - ims_resolution_level: int (default 0)
      - ims_time_index: int (default 0)
      - ims_z_index: int (default 0)

    Returns:
      (image_2d, info_dict)
    """
    try:
        import h5py  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "h5py is required to read .ims files. "
            "Install it (recommended): conda install -c conda-forge h5py"
        ) from e

    # channel is treated as 1-based if >=1; allow channel=0 to mean first channel
    ch_raw = cfg.get("channel", 1)
    ch = int(ch_raw) if ch_raw is not None else 1
    ch_idx = ch - 1 if ch >= 1 else 0

    rl = int(cfg.get("ims_resolution_level", 0))
    tp = int(cfg.get("ims_time_index", 0))
    z = int(cfg.get("ims_z_index", 0))

    dset_path = f"DataSet/ResolutionLevel {rl}/TimePoint {tp}/Channel {ch_idx}/Data"

    with h5py.File(str(input_path), "r") as f:
        if dset_path not in f:
            # Provide a helpful message with a few candidate dataset paths
            candidates: list[str] = []

            def _collector(name: str, obj: Any) -> None:
                try:
                    import h5py as _h5py  # type: ignore
                    if isinstance(obj, _h5py.Dataset) and name.endswith("/Data"):
                        candidates.append(name)
                except Exception:
                    return

            try:
                f.visititems(_collector)
            except Exception:
                pass

            hint = ""
            if candidates:
                preview = "\n".join(f"  - {p}" for p in candidates[:10])
                hint = f"\nFound these /Data datasets (showing up to 10):\n{preview}"

            raise KeyError(
                f"Expected dataset not found in .ims: {dset_path}{hint}"
            )

        dset = f[dset_path]

        # Typical: (Z, Y, X)
        if dset.ndim == 3:
            z_clamped = max(0, min(z, int(dset.shape[0]) - 1))
            img2d = np.asarray(dset[z_clamped, :, :])
        elif dset.ndim == 2:
            z_clamped = 0
            img2d = np.asarray(dset[:, :])
        else:
            # Fallback: flatten leading dims and take first plane
            arr = np.asarray(dset)
            flat = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
            z_clamped = 0
            img2d = flat[0]

    info: Dict[str, Any] = {
        "input_format": "ims",
        "ims_dataset_path": dset_path,
        "ims_resolution_level": rl,
        "ims_time_index": tp,
        "ims_channel_index": ch_idx,
        "ims_z_index": z_clamped,
    }
    return img2d, info


def _read_input_image_2d(input_path: Path, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read an input image and return a 2D plane for Slice0.

    Supported:
      - TIFF (.tif/.tiff) via tifffile
      - Imaris (.ims) via h5py

    Returns:
      (image_2d, info_dict)
    """
    suf = input_path.suffix.lower()

    if suf in {".tif", ".tiff"}:
        arr = tifffile.imread(str(input_path))
        img2d = _select_2d_plane(np.asarray(arr), channel=cfg.get("channel"))
        info: Dict[str, Any] = {
            "input_format": "tiff",
            "selected_channel": cfg.get("channel", None),
        }
        return img2d, info

    if suf == ".ims":
        return _read_ims_2d(input_path, cfg)

    raise ValueError(
        f"Unsupported input file extension: {suf}. Supported: .tif/.tiff, .ims"
    )


def _write_qc_overlay(image_2d: np.ndarray, spots: pd.DataFrame, out_path: Path) -> None:
    img = np.asarray(image_2d)

    # Contrast stretch for view
    vmin, vmax = np.percentile(img.astype(float), [1, 99]) if img.size else (0, 1)

    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    if not spots.empty:
        plt.scatter(spots["x_px"], spots["y_px"], s=30, facecolors="none", edgecolors="red", linewidths=1.0)
        plt.title(f"Slice0: {len(spots)} detected spots")
    else:
        plt.title("Slice0: 0 detected spots")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def run_slice0(config_path: Path) -> Path:
    cfg = _load_config(config_path)
    data_root = _data_root()
    # Resolve input image path (TIFF or Imaris .ims)
    input_relpath = cfg.get("input_relpath")
    if not input_relpath:
        raise ValueError("config missing required key: input_relpath")
    input_path = (data_root / input_relpath).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    # Read input -> 2D plane for Slice0
    img2d, input_info = _read_input_image_2d(input_path, cfg)

    # Optional: restrict detection to an AOI/illumination mask.
    # This mirrors `maxima_padded = maxima_padded * mask_crop` in your realtime script.
    valid_mask_rel = _cfg_first(
        cfg,
        ["valid_mask_relpath", "aoi_mask_relpath", "illumination_mask_relpath"],
        None,
    )
    valid_mask = None
    valid_mask_path = None
    if valid_mask_rel:
        valid_mask_path = (data_root / str(valid_mask_rel)).resolve()
        if not valid_mask_path.exists():
            raise FileNotFoundError(f"valid_mask_relpath not found: {valid_mask_path}")
        valid_mask = _load_optional_tiff_2d(valid_mask_path)
        valid_mask = (np.asarray(valid_mask) > 0)

    # Optional: restrict detection to spots inside nuclei labels.
    # This mirrors `maxima_padded = maxima_padded * nuclei_mask` in your realtime script.
    nuclei_labels_rel = _cfg_first(cfg, ["nuclei_labels_relpath", "nuclei_mask_relpath"], None)
    nuclei_labels = None
    nuclei_labels_path = None
    if nuclei_labels_rel:
        nuclei_labels_path = (data_root / str(nuclei_labels_rel)).resolve()
        if not nuclei_labels_path.exists():
            raise FileNotFoundError(f"nuclei_labels_relpath not found: {nuclei_labels_path}")
        nuclei_labels = _load_optional_tiff_2d(nuclei_labels_path)

    # Detect (LoG-based; mirrors your realtime analysis script)
    # Back-compat note: older configs used `threshold` for peak_local_max.
    # We now interpret `threshold` as the u0 (mean(in5)-bkg) cutoff unless
    # `u0_min` is explicitly set.
    # max-filter window for local maxima detection.
    # If a legacy config provides `min_distance` but not `se_size`, we map it to an
    # odd window size (~2*min_distance+1) to preserve the rough intent.
    if cfg.get("se_size", None) is None:
        if "min_distance" in cfg:
            se_size = int(2 * int(cfg.get("min_distance", 7)) + 1)
        else:
            se_size = 15
    else:
        se_size = int(cfg.get("se_size"))

    # A few common key aliases are accepted for convenience.
    zR = float(_cfg_first(cfg, ["zR", "zR_nm"], 344.5))
    lambda_nm = float(_cfg_first(cfg, ["lambda_nm", "lambda0_nm", "lambda0"], 667.0))
    pixel_size_nm = float(_cfg_first(cfg, ["pixel_size_nm", "pixel_size"], 65.0))

    # u0 cutoff: prefer explicit u0_min/u0_threshold, else fall back to legacy `threshold`.
    u0_min = _cfg_first(cfg, ["u0_min", "u0_threshold"], None)
    if u0_min is None:
        u0_min = cfg.get("threshold", 30.0)
    u0_min = float(u0_min)

    params = Slice0Params(
        zR=zR,
        lambda_nm=lambda_nm,
        pixel_size_nm=pixel_size_nm,
        q_min=float(cfg.get("q_min", 1.0)),
        se_size=se_size,
        window_radius_px=int(cfg.get("window_radius_px", 15)),
        in5_radius_px=int(cfg.get("in5_radius_px", 2)),
        in7_radius_px=int(cfg.get("in7_radius_px", 3)),
        ring_outer_radius_px=int(cfg.get("ring_outer_radius_px", 10)),
        ring_inner_radius_px=int(cfg.get("ring_inner_radius_px", 9)),
        u0_min=u0_min,
    )

    # Optional: crop to the bounding box of the valid_mask to match the
    # realtime script's "mask_indices" behavior (and speed up detection on large images).
    crop_to_valid_bbox = bool(cfg.get("crop_to_valid_bbox", False))
    crop_bbox_yxxy: Optional[list[int]] = None
    if crop_to_valid_bbox and valid_mask is not None:
        yy, xx = np.where(valid_mask)
        if yy.size == 0:
            spots = pd.DataFrame(columns=REQUIRED_COLUMNS)
        else:
            y0, y1 = int(yy.min()), int(yy.max()) + 1
            x0, x1 = int(xx.min()), int(xx.max()) + 1
            crop_bbox_yxxy = [y0, y1, x0, x1]
            img_crop = np.asarray(img2d)[y0:y1, x0:x1]
            vm_crop = np.asarray(valid_mask)[y0:y1, x0:x1]
            nl_crop = np.asarray(nuclei_labels)[y0:y1, x0:x1] if nuclei_labels is not None else None

            spots = detect_spots(img_crop, params, valid_mask=vm_crop, nuclei_labels=nl_crop)
            if not spots.empty:
                spots["y_px"] = spots["y_px"] + float(y0)
                spots["x_px"] = spots["x_px"] + float(x0)
    else:
        spots = detect_spots(img2d, params, valid_mask=valid_mask, nuclei_labels=nuclei_labels)

    # Create run folder
    runs_dir = (data_root / cfg.get("output_runs_dir", "runs")).resolve()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (runs_dir / f"{ts}__slice0").resolve()
    out_dir.mkdir(parents=True, exist_ok=False)

    # Outputs
    spots_path = out_dir / "spots.parquet"
    spots.to_parquet(spots_path, index=False)

    qc_path = out_dir / "qc_overlay.png"
    _write_qc_overlay(img2d, spots, qc_path)

    manifest: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "input_path": str(input_path),
        "config_snapshot": cfg,
        "output_dir": str(out_dir),
        "git_commit": _try_git_commit(REPO_ROOT),
        "env_name": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        # small helpful extras (safe additions)
        "image_shape": list(np.asarray(img2d).shape),
        "image_dtype": str(np.asarray(img2d).dtype),
        "num_spots": int(len(spots)),
        "qc_overlay": str(qc_path),
        "kernel_params": asdict(params),
        # optional inputs (safe additions)
        "valid_mask_path": str(valid_mask_path) if valid_mask_path is not None else None,
        "nuclei_labels_path": str(nuclei_labels_path) if nuclei_labels_path is not None else None,
        "crop_to_valid_bbox": bool(crop_to_valid_bbox),
        "crop_bbox_yxxy": crop_bbox_yxxy,
    }

    manifest.update(input_info)

    manifest_path = out_dir / "run_manifest.yaml"
    with manifest_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Slice0 on a single image (TIFF or Imaris .ims)")
    ap.add_argument("--config", default="configs/dev.yaml", help="Path to YAML config (relative or absolute)")
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()

    out_dir = run_slice0(config_path)
    print(f"Slice0 complete: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
