from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow  # noqa: F401
import tifffile
import yaml

# Matplotlib: write PNGs without needing a display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from skimage.morphology import dilation, disk
from skimage.segmentation import find_boundaries

# Allow running without packaging
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT / "src"))

from image_io import PlaneSelection, read_image_2d  # noqa: E402
from slice0_kernel import Slice0Params, detect_spots  # noqa: E402
from slice1_nuclei_kernel import Slice1NucleiParams, segment_nuclei_stardist  # noqa: E402
from stardist_utils import StardistModelRef, load_stardist2d  # noqa: E402


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


def _resolve_model_dir(cfg: Dict[str, Any], data_root: Path) -> Path:
    model_dir_raw = cfg.get("stardist_model_dir")
    if not model_dir_raw:
        raise ValueError("Config must set stardist_model_dir")
    model_dir = Path(str(model_dir_raw))
    if not model_dir.is_absolute():
        model_dir = (data_root / model_dir).resolve()
    return model_dir


def _assert_tiff_channel(path: Path, channel: int) -> None:
    if channel <= 1:
        return
    with tifffile.TiffFile(str(path)) as tif:
        series = tif.series[0]
        axes = series.axes
        shape = series.shape
    if "C" not in axes:
        raise ValueError(
            f"Input TIFF has no channel axis but channel {channel} was requested: {path}"
        )
    c_index = axes.index("C")
    c_count = shape[c_index]
    if channel > c_count:
        raise ValueError(
            f"Input TIFF has {c_count} channel(s); requested channel {channel}: {path}"
        )


def _normalize_channels(value: Any, *, default: int) -> list[int]:
    if value is None:
        return [int(default)]
    if isinstance(value, (list, tuple)):
        channels = [int(v) for v in value]
    else:
        channels = [int(value)]
    if not channels:
        raise ValueError("channel_spots must be a non-empty int or list of ints")
    return channels


def _load_mask(path: Path) -> np.ndarray:
    mask = np.asarray(tifffile.imread(str(path)))
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"valid_mask_relpath must be 2D; got shape={mask.shape}")
    return mask.astype(bool)


def _create_cutout_montage(
    nuclei_img: np.ndarray,
    spots_img: np.ndarray,
    spots_df: pd.DataFrame,
    *,
    crop_size: int = 80,
    max_cutouts: int = 50,
    n_cols: int = 10,
    sample_seed: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], int]:
    if spots_df.empty:
        return None, 0

    n_spots = min(len(spots_df), max_cutouts)
    if "snr" in spots_df.columns:
        subset = spots_df.sort_values("snr", ascending=False).head(n_spots)
    else:
        seed = 0 if sample_seed is None else int(sample_seed)
        order = np.random.default_rng(seed).permutation(len(spots_df))[:n_spots]
        subset = spots_df.iloc[order]

    n_cols = max(1, min(int(n_cols), n_spots))
    n_rows = int(np.ceil(n_spots / n_cols))

    montage_h = n_rows * crop_size
    montage_w = n_cols * crop_size

    dtype = np.result_type(nuclei_img.dtype, spots_img.dtype)
    montage = np.zeros((2, montage_h, montage_w), dtype=dtype)

    height, width = nuclei_img.shape
    half = crop_size // 2

    for idx, (_, row) in enumerate(subset.iterrows()):
        r = idx // n_cols
        c = idx % n_cols

        y_c, x_c = int(row["y_px"]), int(row["x_px"])

        y0_src = max(0, y_c - half)
        y1_src = min(height, y_c + half)
        x0_src = max(0, x_c - half)
        x1_src = min(width, x_c + half)

        crop_h = y1_src - y0_src
        crop_w = x1_src - x0_src

        y0_dst = (crop_size - crop_h) // 2
        x0_dst = (crop_size - crop_w) // 2

        n_crop = nuclei_img[y0_src:y1_src, x0_src:x1_src]
        s_crop = spots_img[y0_src:y1_src, x0_src:x1_src]

        grid_y = r * crop_size
        grid_x = c * crop_size

        montage[0, grid_y + y0_dst : grid_y + y0_dst + crop_h, grid_x + x0_dst : grid_x + x0_dst + crop_w] = n_crop
        montage[1, grid_y + y0_dst : grid_y + y0_dst + crop_h, grid_x + x0_dst : grid_x + x0_dst + crop_w] = s_crop

    return montage, n_spots


def _write_qc_overlay(
    spot_img: np.ndarray,
    nuclei_labels: np.ndarray,
    spots_df: pd.DataFrame,
    out_path: Path,
) -> None:
    img = spot_img.astype(float)
    if img.size:
        vmin, vmax = np.percentile(img, [1, 99])
    else:
        vmin, vmax = 0.0, 1.0

    boundaries = find_boundaries(nuclei_labels > 0, mode="outer")
    boundaries = dilation(boundaries, disk(1))

    fig = plt.figure(figsize=(10, 10), dpi=150)
    ax = fig.add_subplot(111)

    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")

    rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=float)
    rgba[boundaries, 0] = 1.0
    rgba[boundaries, 3] = 0.6
    ax.imshow(rgba, interpolation="nearest")

    if not spots_df.empty:
        ax.scatter(
            spots_df["x_px"],
            spots_df["y_px"],
            s=40,
            facecolors="none",
            edgecolors="cyan",
            linewidths=1.2,
            alpha=0.8,
        )
        ax.set_title(f"Integrated QC: {len(spots_df)} spots in {int(nuclei_labels.max())} nuclei")
    else:
        ax.set_title("Integrated QC: 0 spots")

    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_integrated(config_path: Path) -> Path:
    cfg = _load_config(config_path)
    data_root = _data_root()

    input_relpath = cfg.get("input_relpath")
    if not input_relpath:
        raise ValueError("Config must set input_relpath")
    input_path = (data_root / str(input_relpath)).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    channel_nuclei_raw = cfg.get("channel_nuclei", 1)
    channel_spots = _normalize_channels(cfg.get("channel_spots", 2), default=2)
    skip_segmentation = bool(cfg.get("skip_nuclei_segmentation", False))
    have_nuclei_channel = channel_nuclei_raw not in (None, 0, "none", "None")
    channel_nuclei = int(channel_nuclei_raw) if have_nuclei_channel else None

    if input_path.suffix.lower() in {".tif", ".tiff"}:
        if have_nuclei_channel:
            _assert_tiff_channel(input_path, int(channel_nuclei))
        for ch in channel_spots:
            _assert_tiff_channel(input_path, int(ch))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runs_dir = (data_root / str(cfg.get("output_runs_dir", "runs"))).resolve()
    out_dir = runs_dir / f"{stamp}__integrated"
    out_dir.mkdir(parents=True, exist_ok=False)

    common_sel = {
        "ims_resolution_level": int(cfg.get("ims_resolution_level", 0)),
        "ims_time_index": int(cfg.get("ims_time_index", 0)),
        "ims_z_index": int(cfg.get("ims_z_index", 0)),
    }

    img_nuc: Optional[np.ndarray]
    if have_nuclei_channel:
        print(f"Loading nuclei channel ({channel_nuclei})...")
        img_nuc = read_image_2d(input_path, PlaneSelection(channel=channel_nuclei, **common_sel))
    else:
        img_nuc = None

    spot_images: list[tuple[int, np.ndarray]] = []
    for ch in channel_spots:
        print(f"Loading spots channel ({ch})...")
        spot_images.append(
            (ch, read_image_2d(input_path, PlaneSelection(channel=ch, **common_sel)))
        )

    ref_shape = spot_images[0][1].shape
    for ch, img_spot in spot_images:
        if img_spot.shape != ref_shape:
            raise ValueError(
                f"Spot channel {ch} shape {img_spot.shape} does not match {ref_shape}"
            )
    if img_nuc is not None and img_nuc.shape != ref_shape:
        raise ValueError(f"Channel shape mismatch: {img_nuc.shape} vs {ref_shape}")

    nuclei_labels_relpath = cfg.get("nuclei_labels_relpath")
    valid_mask_relpath = cfg.get("valid_mask_relpath")
    valid_mask: Optional[np.ndarray] = None
    valid_mask_path: Optional[Path] = None
    if valid_mask_relpath:
        valid_mask_path = (data_root / str(valid_mask_relpath)).resolve()
        if not valid_mask_path.exists():
            raise FileNotFoundError(f"valid_mask_relpath not found: {valid_mask_path}")
        valid_mask = _load_mask(valid_mask_path)
        if valid_mask.shape != ref_shape:
            raise ValueError(
                f"valid_mask_relpath shape {valid_mask.shape} does not match image shape {ref_shape}"
            )

    nuclei_labels: Optional[np.ndarray]
    if nuclei_labels_relpath:
        labels_source_path = (data_root / str(nuclei_labels_relpath)).resolve()
        if not labels_source_path.exists():
            raise FileNotFoundError(f"nuclei_labels_relpath not found: {labels_source_path}")
        nuclei_labels = np.asarray(tifffile.imread(str(labels_source_path)))
        nuclei_labels = np.squeeze(nuclei_labels)
        if nuclei_labels.ndim != 2:
            raise ValueError(
                f"nuclei_labels_relpath must be 2D; got shape={nuclei_labels.shape}"
            )
        if nuclei_labels.shape != ref_shape:
            raise ValueError(
                f"nuclei_labels_relpath shape {nuclei_labels.shape} does not match "
                f"image shape {ref_shape}"
            )
        nuc_meta = {
            "source": "precomputed",
            "path": str(labels_source_path),
        }
    elif skip_segmentation or not have_nuclei_channel:
        nuclei_labels = None
        nuc_meta = {"source": "none"}
    else:
        print("Running nuclei segmentation...")
        model_dir = _resolve_model_dir(cfg, data_root)
        model = load_stardist2d(StardistModelRef(model_dir=model_dir))

        nuc_params = Slice1NucleiParams(
            normalize_pmin=float(cfg.get("nuc_normalize_pmin", 1.0)),
            normalize_pmax=float(cfg.get("nuc_normalize_pmax", 99.8)),
            prob_thresh=cfg.get("nuc_prob_thresh", None),
            nms_thresh=cfg.get("nuc_nms_thresh", None),
        )

        nuclei_labels, nuc_meta = segment_nuclei_stardist(img_nuc, model, nuc_params)

    print("Running spot detection (inside nuclei)...")
    spot_params = Slice0Params(
        zR=float(cfg.get("spot_zR", 344.5)),
        lambda_nm=float(cfg.get("spot_lambda_nm", 667.0)),
        pixel_size_nm=float(cfg.get("spot_pixel_size_nm", 65.0)),
        q_min=float(cfg.get("spot_q_min", 1.0)),
        se_size=int(cfg.get("spot_se_size", 15)),
        u0_min=float(cfg.get("spot_u0_min", 30.0)),
    )

    spots_tables: list[pd.DataFrame] = []
    for ch, img_spot in spot_images:
        df = detect_spots(img_spot, spot_params, valid_mask=valid_mask, nuclei_labels=nuclei_labels)
        if not df.empty:
            df["spot_channel"] = int(ch)
        else:
            df = df.assign(spot_channel=pd.Series(dtype=int))
        spots_tables.append(df)
    spots_df = pd.concat(spots_tables, ignore_index=True) if spots_tables else pd.DataFrame()

    spots_path = out_dir / "spots.parquet"
    spots_df.to_parquet(spots_path, index=False)

    labels_path = out_dir / "nuclei_labels.tif"
    labels_out = (
        nuclei_labels
        if nuclei_labels is not None
        else np.zeros(ref_shape, dtype=np.int32)
    )
    if labels_out.max() < np.iinfo(np.uint16).max:
        tifffile.imwrite(labels_path, labels_out.astype(np.uint16))
    else:
        tifffile.imwrite(labels_path, labels_out.astype(np.uint32))

    qc_overlay_paths: list[str] = []
    qc_cutouts_paths: list[str] = []
    total_cutouts = 0
    qc_cutout_size = int(cfg.get("qc_cutout_size", 80))
    qc_max_cutouts = int(cfg.get("qc_max_cutouts", 50))
    qc_montage_cols = int(cfg.get("qc_montage_cols", 10))
    qc_sample_seed = cfg.get("qc_sample_seed", 0)

    for ch, img_spot in spot_images:
        suffix = f"_ch{ch}" if len(spot_images) > 1 else ""
        qc_overlay_path = out_dir / f"qc_overlay{suffix}.png"
        _write_qc_overlay(img_spot, labels_out, spots_df[spots_df["spot_channel"] == ch], qc_overlay_path)
        qc_overlay_paths.append(qc_overlay_path.name)

        qc_cutouts_path = out_dir / f"qc_cutouts{suffix}.tif"
        nuclei_for_montage = img_nuc if img_nuc is not None else np.zeros_like(img_spot)
        montage, montage_count = _create_cutout_montage(
            nuclei_for_montage,
            img_spot,
            spots_df[spots_df["spot_channel"] == ch],
            crop_size=qc_cutout_size,
            max_cutouts=qc_max_cutouts,
            n_cols=qc_montage_cols,
            sample_seed=qc_sample_seed,
        )
        if montage is not None:
            tifffile.imwrite(qc_cutouts_path, montage, imagej=True, metadata={"axes": "CYX"})
            qc_cutouts_paths.append(qc_cutouts_path.name)
            total_cutouts += int(montage_count)

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "output_dir": str(out_dir),
        "git_commit": _try_git_commit(REPO_ROOT),
        "config_snapshot": cfg,
        "image_shape": list(ref_shape),
        "num_nuclei": int(labels_out.max()),
        "num_spots": int(len(spots_df)),
        "nuclei_meta": nuc_meta,
        "valid_mask_path": str(valid_mask_path) if valid_mask_path else None,
        "outputs": {
            "spots": spots_path.name,
            "nuclei_labels": labels_path.name,
            "qc_overlay": qc_overlay_paths if len(qc_overlay_paths) > 1 else qc_overlay_paths[0],
            "qc_montage": qc_cutouts_paths if len(qc_cutouts_paths) > 1 else (qc_cutouts_paths[0] if qc_cutouts_paths else None),
        },
        "qc": {
            "cutout_size": qc_cutout_size,
            "max_cutouts": qc_max_cutouts,
            "montage_cols": qc_montage_cols,
            "sample_seed": qc_sample_seed,
            "cutouts_written": int(total_cutouts),
        },
    }

    with (out_dir / "run_manifest.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    print(f"Integrated run complete: {out_dir}")
    return out_dir


def main() -> int:
    p = argparse.ArgumentParser(description="Run integrated nuclei+spots workflow")
    p.add_argument("--config", type=Path, default=Path("configs/integrated_sim.yaml"))
    args = p.parse_args()

    cfg_path = args.config
    if not cfg_path.is_absolute():
        cfg_path = (REPO_ROOT / cfg_path).resolve()

    run_integrated(cfg_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
