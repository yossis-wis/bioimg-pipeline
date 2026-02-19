from __future__ import annotations

import argparse
import glob
import hashlib
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
try:
    import pyarrow  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    # Keep module importable in minimal environments; the driver will error
    # at runtime if parquet I/O is requested without pyarrow.
    pyarrow = None  # type: ignore[assignment]
import tifffile
import yaml

# Matplotlib: write PNGs without needing a display
import matplotlib

matplotlib.use("Agg")

# Allow running without packaging
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT / "src"))

from image_io import PlaneSelection, read_image_2d  # noqa: E402
from slice0_kernel import Slice0Params, detect_spots  # noqa: E402
from slice1_nuclei_kernel import Slice1NucleiParams, segment_nuclei_stardist  # noqa: E402
from stardist_utils import StardistModelRef, load_stardist2d  # noqa: E402
from vis_utils import create_cutout_montage, write_qc_overlay  # noqa: E402
from pixel_size_utils import infer_pixel_size_nm_mean  # noqa: E402



_SLICE0_CFG_KEY_TO_FIELD = {
    # Legacy config keys (top-level) -> Slice0Params field names
    "spot_zR": "zR",
    "spot_lambda_nm": "lambda_nm",
    "spot_pixel_size_nm": "pixel_size_nm",
    "spot_radius_nm": "spot_radius_nm",
    "spot_do_median_filter": "do_median_filter",
    "spot_do_subpixel_localization": "do_subpixel_localization",
    "spot_q_min": "q_min",
    "spot_se_size": "se_size",
    "spot_u0_min": "u0_min",
}


def _coerce_slice0_overrides(overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce a config override dict into Slice0Params field names.

    Supports either:
    - Slice0Params field names directly (e.g. ``q_min``)
    - The legacy config key names used in YAML (e.g. ``spot_q_min``)

    Unknown keys raise ValueError to avoid silent typos in YAML.
    """

    if overrides is None:
        return {}
    if not isinstance(overrides, dict):
        raise ValueError(f"spot_params overrides must be a dict; got {type(overrides)}")

    field_names = set(Slice0Params.__dataclass_fields__.keys())
    out: Dict[str, Any] = {}
    for k, v in overrides.items():
        if k in field_names:
            out[k] = v
            continue
        if k in _SLICE0_CFG_KEY_TO_FIELD:
            out[_SLICE0_CFG_KEY_TO_FIELD[k]] = v
            continue
        raise ValueError(
            "Unknown Slice0Params override key: "
            f"{k!r}. Expected a Slice0Params field name or one of: "
            f"{sorted(_SLICE0_CFG_KEY_TO_FIELD.keys())}"
        )
    return out


def _build_slice0_params_for_channel(
    cfg: Dict[str, Any],
    *,
    pixel_size_nm: float,
    channel: int,
) -> Slice0Params:
    """Build Slice0Params for a given spot channel.

    This implements the approved plan's requirement to support per-channel LoG
    settings by merging:

        defaults_from_cfg (top-level spot_* keys)
        → spot_params_by_channel[channel] overrides (if provided)

    The per-channel override dict may use either Slice0Params field names
    (e.g. ``q_min``) or the legacy YAML keys (e.g. ``spot_q_min``).
    """

    base = {
        "zR": float(cfg.get("spot_zR", 344.5)),
        "lambda_nm": float(cfg.get("spot_lambda_nm", 667.0)),
        "pixel_size_nm": float(pixel_size_nm),

        # Optional TrackMate-style overrides
        "spot_radius_nm": (
            float(cfg["spot_radius_nm"]) if cfg.get("spot_radius_nm", None) is not None else None
        ),
        "do_median_filter": bool(cfg.get("spot_do_median_filter", False)),
        "do_subpixel_localization": bool(cfg.get("spot_do_subpixel_localization", False)),

        # TrackMate threshold + maxima neighborhood
        "q_min": float(cfg.get("spot_q_min", 1.0)),
        "se_size": int(cfg.get("spot_se_size", 3)),

        # Downstream photometry threshold (can be set to 0 for “LoG-only” output).
        "u0_min": float(cfg.get("spot_u0_min", 30.0)),
    }

    by_ch_raw = cfg.get("spot_params_by_channel", None)
    if by_ch_raw not in (None, "", False):
        if not isinstance(by_ch_raw, dict):
            raise ValueError(
                "spot_params_by_channel must be a mapping like {1: {...}, 2: {...}}"
            )

        # Keys may arrive as strings from YAML; coerce to int.
        # Unknown keys inside overrides raise early for safety.
        for ch_key, override_raw in by_ch_raw.items():
            try:
                ch_int = int(ch_key)
            except Exception as exc:
                raise ValueError(
                    f"spot_params_by_channel key {ch_key!r} is not an int channel index"
                ) from exc
            if ch_int != int(channel):
                continue
            if override_raw in (None, ""):
                break
            overrides = _coerce_slice0_overrides(override_raw)
            base.update(overrides)
            break

    # Coerce a few known numeric fields (YAML may provide ints, strings, etc.)
    base["zR"] = float(base["zR"])
    base["lambda_nm"] = float(base["lambda_nm"])
    base["pixel_size_nm"] = float(base["pixel_size_nm"])
    if base.get("spot_radius_nm", None) is not None:
        base["spot_radius_nm"] = float(base["spot_radius_nm"])  # type: ignore[assignment]
    base["q_min"] = float(base["q_min"])
    base["se_size"] = int(base["se_size"])
    base["u0_min"] = float(base["u0_min"])
    base["do_median_filter"] = bool(base["do_median_filter"])
    base["do_subpixel_localization"] = bool(base["do_subpixel_localization"])

    return Slice0Params(**base)


def _require_pyarrow() -> None:
    if pyarrow is None:
        raise RuntimeError(
            "pyarrow is required for parquet I/O (spots.parquet / spots_aggregate.parquet). "
            "Install pyarrow in your environment to run this driver."
        )



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
    return _resolve_path(model_dir_raw, data_root)


def _resolve_path(value: Any, data_root: Path) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = (data_root / path).resolve()
    return path


def _resolve_input_paths(cfg: Dict[str, Any], data_root: Path) -> list[Path]:
    input_relpath = cfg.get("input_relpath")
    input_relpaths = cfg.get("input_relpaths")
    input_glob = cfg.get("input_glob")

    provided = [v for v in (input_relpath, input_relpaths, input_glob) if v]
    if len(provided) > 1:
        raise ValueError("Set only one of input_relpath, input_relpaths, or input_glob")

    if input_relpath:
        paths = [_resolve_path(input_relpath, data_root)]
    elif input_relpaths:
        if not isinstance(input_relpaths, (list, tuple)):
            raise ValueError("input_relpaths must be a list of paths")
        paths = [_resolve_path(path, data_root) for path in input_relpaths]
    elif input_glob:
        pattern = str(input_glob)
        pattern_path = Path(pattern)
        if not pattern_path.is_absolute():
            pattern = str((data_root / pattern_path).resolve())
        matches = sorted(glob.glob(pattern, recursive=True))
        if not matches:
            raise FileNotFoundError(f"input_glob matched no files: {pattern}")
        paths = [Path(match).resolve() for match in matches]
    else:
        raise ValueError("Config must set input_relpath, input_relpaths, or input_glob")

    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Input file(s) not found: {missing}")
    return paths


def _safe_run_name(name: str, *, max_len: int = 60) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    cleaned = cleaned or "run"
    if len(cleaned) <= max_len:
        return cleaned
    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:8]
    truncated = cleaned[: max_len - 10].rstrip("_")
    truncated = truncated or "run"
    return f"{truncated}__{digest}"


def _resolve_optional_path(value: Any, data_root: Path) -> Optional[Path]:
    if value in (None, ""):
        return None
    return _resolve_path(value, data_root)


def _publish_output_dir(
    source_dir: Path,
    *,
    input_path: Path,
    publish_dir: Path,
    publish_mirror: bool,
    input_base_dir: Optional[Path],
    publish_batch_root: Optional[str],
    publish_mode: str,
) -> Path:
    if publish_mirror:
        if not input_base_dir:
            raise ValueError("publish_mirror requires input_base_dir to be set")
        try:
            rel_parent = input_path.parent.resolve().relative_to(input_base_dir.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Input path {input_path} is not under input_base_dir {input_base_dir}"
            ) from exc
        target_root = publish_dir / rel_parent
    else:
        target_root = publish_dir
    if publish_batch_root:
        target_root = target_root / publish_batch_root
    target_root.mkdir(parents=True, exist_ok=True)
    dest_dir = target_root / source_dir.name
    mode = str(publish_mode or "error").lower()
    if mode not in {"error", "overwrite", "merge"}:
        raise ValueError(f"publish_mode must be one of: error, overwrite, merge (got {publish_mode})")
    if dest_dir.exists():
        if mode == "error":
            raise FileExistsError(f"Publish destination already exists: {dest_dir}")
        if mode == "overwrite":
            shutil.rmtree(dest_dir)
    shutil.copytree(source_dir, dest_dir, dirs_exist_ok=(mode == "merge"))
    return dest_dir


def _is_run_complete(run_dir: Path) -> bool:
    manifest_path = run_dir / "run_manifest.yaml"
    spots_path = run_dir / "spots.parquet"
    done_path = run_dir / "DONE"
    return done_path.exists() or (manifest_path.exists() and spots_path.exists())


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


def _run_integrated_single(
    cfg: Dict[str, Any],
    *,
    data_root: Path,
    input_path: Path,
    out_dir: Path,
) -> Path:
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)

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
        valid_mask_path = _resolve_path(valid_mask_relpath, data_root)
        if not valid_mask_path.exists():
            raise FileNotFoundError(f"valid_mask_relpath not found: {valid_mask_path}")
        valid_mask = _load_mask(valid_mask_path)
        if valid_mask.shape != ref_shape:
            raise ValueError(
                f"valid_mask_relpath shape {valid_mask.shape} does not match image shape {ref_shape}"
            )

    nuclei_labels: Optional[np.ndarray]
    if nuclei_labels_relpath:
        labels_source_path = _resolve_path(nuclei_labels_relpath, data_root)
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

    # --- Pixel size (nm/px): prefer explicit config, but allow 'auto' from metadata ---
    meta_px_mean_nm, meta_px_note = infer_pixel_size_nm_mean(input_path)
    if meta_px_mean_nm is not None:
        print(f"[pixel_size] metadata mean: {meta_px_mean_nm:.3f} nm/px ({meta_px_note})")
    else:
        print(f"[pixel_size] metadata unavailable: {meta_px_note}")

    cfg_px = cfg.get("spot_pixel_size_nm", None)
    px_source = ""
    if cfg_px is None:
        if meta_px_mean_nm is not None:
            pixel_size_nm = float(meta_px_mean_nm)
            px_source = "metadata(default)"
        else:
            pixel_size_nm = 65.0
            px_source = "default(65nm)"
    elif isinstance(cfg_px, str) and cfg_px.strip().lower() == "auto":
        if meta_px_mean_nm is not None:
            pixel_size_nm = float(meta_px_mean_nm)
            px_source = "metadata(auto)"
        else:
            pixel_size_nm = 65.0
            px_source = "fallback_default(65nm)"
    else:
        pixel_size_nm = float(cfg_px)
        px_source = "config"

    if px_source == "config" and meta_px_mean_nm is not None and float(meta_px_mean_nm) > 0:
        rel = abs(float(pixel_size_nm) - float(meta_px_mean_nm)) / float(meta_px_mean_nm)
        if rel > 0.02:
            print(
                f"[pixel_size] WARNING: spot_pixel_size_nm={float(pixel_size_nm):.3f} nm/px from config, "
                f"but metadata suggests {float(meta_px_mean_nm):.3f} nm/px ({100.0*rel:.1f}% difference). "
                "Consider setting spot_pixel_size_nm: auto or updating your config."
            )
    print(f"[pixel_size] using spot_pixel_size_nm={float(pixel_size_nm):.3f} nm/px (source={px_source})")

    print("Running spot detection (inside nuclei)...")
    spots_tables: list[pd.DataFrame] = []
    for ch, img_spot in spot_images:
        spot_params = _build_slice0_params_for_channel(cfg, pixel_size_nm=float(pixel_size_nm), channel=int(ch))

        # Convenience print: TrackMate GUI uses “estimated blob diameter” = 2 × radius.
        if spot_params.spot_radius_nm is not None:
            radius_nm = float(spot_params.spot_radius_nm)
        else:
            radius_nm = float(np.sqrt(float(spot_params.lambda_nm) * float(spot_params.zR) / np.pi))
        tm_diam_um = 2.0 * radius_nm / 1000.0
        print(
            f"[spots] channel {int(ch)}: TrackMate diameter ≈ {tm_diam_um:.3f} µm "
            f"(radius={radius_nm:.1f} nm), q_min={float(spot_params.q_min):g}, u0_min={float(spot_params.u0_min):g}"
        )

        df = detect_spots(
            img_spot,
            spot_params,
            valid_mask=valid_mask,
            nuclei_labels=nuclei_labels,
        )
        if not df.empty:
            df["spot_channel"] = int(ch)
        else:
            df = df.assign(spot_channel=pd.Series(dtype=int))
        spots_tables.append(df)
    spots_df = pd.concat(spots_tables, ignore_index=True) if spots_tables else pd.DataFrame()

    spots_path = out_dir / "spots.parquet"
    _require_pyarrow()
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
    qc_sample_seed_raw = cfg.get("qc_sample_seed", None)
    if qc_sample_seed_raw in (None, "", "none", "None"):
        qc_sample_seed: Optional[int] = None
    else:
        qc_sample_seed = int(qc_sample_seed_raw)

    for ch, img_spot in spot_images:
        suffix = f"_ch{ch}" if len(spot_images) > 1 else ""
        qc_overlay_path = out_dir / f"qc_overlay{suffix}.png"
        write_qc_overlay(
            img_spot,
            labels_out,
            spots_df[spots_df["spot_channel"] == ch],
            qc_overlay_path,
        )
        qc_overlay_paths.append(qc_overlay_path.name)

        qc_cutouts_path = out_dir / f"qc_cutouts{suffix}.tif"
        nuclei_for_montage = img_nuc if img_nuc is not None else np.zeros_like(img_spot)
        montage, montage_count = create_cutout_montage(
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

    with (out_dir / "DONE").open("w", encoding="utf-8") as f:
        f.write(f"completed_at: {datetime.now(timezone.utc).isoformat()}\n")

    print(f"Integrated run complete: {out_dir}")
    return out_dir


def run_integrated(config_path: Path) -> Path:
    cfg = _load_config(config_path)
    data_root = _data_root()

    input_paths = _resolve_input_paths(cfg, data_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    runs_dir = _resolve_path(cfg.get("output_runs_dir", "runs"), data_root)
    runs_dir.mkdir(parents=True, exist_ok=True)

    publish_dir = _resolve_optional_path(cfg.get("publish_dir"), data_root)
    publish_mirror = bool(cfg.get("publish_mirror", False))
    input_base_dir = _resolve_optional_path(cfg.get("input_base_dir"), data_root)
    publish_mode = cfg.get("publish_mode", "error")
    republish_skipped = bool(cfg.get("republish_skipped", False))

    if len(input_paths) == 1:
        out_dir = runs_dir / f"{stamp}__integrated"
        if out_dir.exists() and bool(cfg.get("skip_existing", False)):
            if _is_run_complete(out_dir):
                print(f"Skipping existing output: {out_dir}")
                if publish_dir and republish_skipped:
                    try:
                        _publish_output_dir(
                            out_dir,
                            input_path=input_paths[0],
                            publish_dir=publish_dir,
                            publish_mirror=publish_mirror,
                            input_base_dir=input_base_dir,
                            publish_batch_root=None,
                            publish_mode=publish_mode,
                        )
                    except Exception as exc:
                        print(f"Republish failed for {input_paths[0]}: {exc}")
                return out_dir
            raise FileExistsError(
                f"Output directory exists but is incomplete: {out_dir}"
            )
        run_dir = _run_integrated_single(
            cfg,
            data_root=data_root,
            input_path=input_paths[0],
            out_dir=out_dir,
        )
        if publish_dir:
            publish_status = "completed"
            publish_error = None
            try:
                _publish_output_dir(
                    run_dir,
                    input_path=input_paths[0],
                    publish_dir=publish_dir,
                    publish_mirror=publish_mirror,
                    input_base_dir=input_base_dir,
                    publish_batch_root=None,
                    publish_mode=publish_mode,
                )
            except Exception as exc:
                publish_status = "failed"
                publish_error = str(exc)
                print(f"Publishing failed for {input_paths[0]}: {exc}")
            if publish_error:
                print(f"Publish status: {publish_status}")
        return run_dir

    batch_dir_name = cfg.get("batch_dir_name")
    if batch_dir_name:
        batch_dir = runs_dir / _safe_run_name(str(batch_dir_name), max_len=80)
    else:
        batch_dir = runs_dir / f"{stamp}__integrated_batch"
    if batch_dir.exists():
        if bool(cfg.get("skip_existing", False)):
            print(f"Reusing existing batch dir: {batch_dir}")
        else:
            raise FileExistsError(f"Batch output directory already exists: {batch_dir}")
    else:
        batch_dir.mkdir(parents=True, exist_ok=False)

    continue_on_error = bool(cfg.get("continue_on_error", True))
    skip_existing = bool(cfg.get("skip_existing", False))
    batch_entries = []
    for idx, input_path in enumerate(input_paths, start=1):
        run_name = _safe_run_name(input_path.stem)
        run_dir = batch_dir / f"{idx:03d}__{run_name}"
        if run_dir.exists() and skip_existing:
            if _is_run_complete(run_dir):
                entry: Dict[str, Any] = {
                    "input_path": str(input_path),
                    "output_dir": str(run_dir),
                    "status": "skipped",
                }
                print(f"[{idx}/{len(input_paths)}] Skipping existing: {input_path}")
                if publish_dir and republish_skipped:
                    publish_status = "completed"
                    publish_error = None
                    published = None
                    try:
                        published = _publish_output_dir(
                            run_dir,
                            input_path=input_path,
                            publish_dir=publish_dir,
                            publish_mirror=publish_mirror,
                            input_base_dir=input_base_dir,
                            publish_batch_root=batch_dir.name,
                            publish_mode=publish_mode,
                        )
                    except Exception as exc:
                        publish_status = "failed"
                        publish_error = str(exc)
                        print(f"Republish failed for {input_path}: {exc}")
                    entry["publish_status"] = publish_status
                    if publish_error:
                        entry["publish_error"] = publish_error
                    if published:
                        entry["published_dir"] = str(published)
                batch_entries.append(entry)
                continue
            entry = {
                "input_path": str(input_path),
                "output_dir": str(run_dir),
                "status": "failed",
                "error": "Existing output directory is incomplete; remove it to rerun.",
            }
            batch_entries.append(entry)
            print(f"[{idx}/{len(input_paths)}] Incomplete output exists: {input_path}")
            if not continue_on_error:
                break
            continue

        print(f"[{idx}/{len(input_paths)}] Running integrated analysis on: {input_path}")
        try:
            output_dir = _run_integrated_single(
                cfg,
                data_root=data_root,
                input_path=input_path,
                out_dir=run_dir,
            )
            entry: Dict[str, Any] = {
                "input_path": str(input_path),
                "output_dir": str(output_dir),
                "status": "completed",
            }
            if publish_dir:
                publish_status = "completed"
                publish_error = None
                published = None
                try:
                    published = _publish_output_dir(
                        output_dir,
                        input_path=input_path,
                        publish_dir=publish_dir,
                        publish_mirror=publish_mirror,
                        input_base_dir=input_base_dir,
                        publish_batch_root=batch_dir.name,
                        publish_mode=publish_mode,
                    )
                except Exception as exc:
                    publish_status = "failed"
                    publish_error = str(exc)
                    print(f"Publishing failed for {input_path}: {exc}")
                entry["publish_status"] = publish_status
                if publish_error:
                    entry["publish_error"] = publish_error
                if published:
                    entry["published_dir"] = str(published)
            batch_entries.append(entry)
        except Exception as exc:
            entry = {
                "input_path": str(input_path),
                "output_dir": str(run_dir),
                "status": "failed",
                "error": str(exc),
            }
            batch_entries.append(entry)
            print(f"Error processing {input_path}: {exc}")
            if not continue_on_error:
                break

    batch_manifest: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runs": batch_entries,
        "config_snapshot": cfg,
    }

    if cfg.get("batch_aggregate_spots", False):
        _require_pyarrow()
        aggregate_tables = []
        for entry in batch_entries:
            if entry.get("status") not in {"completed", "skipped"}:
                continue
            spots_path = Path(entry["output_dir"]) / "spots.parquet"
            if not spots_path.exists():
                continue
            df = pd.read_parquet(spots_path)
            df["input_path"] = entry["input_path"]
            df["output_dir"] = entry["output_dir"]
            df["run_status"] = entry.get("status", "unknown")
            try:
                df["condition"] = Path(entry["input_path"]).parent.name
            except Exception:
                df["condition"] = ""
            aggregate_tables.append(df)
        if aggregate_tables:
            aggregate_df = pd.concat(aggregate_tables, ignore_index=True)
            aggregate_path = batch_dir / "spots_aggregate.parquet"
            aggregate_df.to_parquet(aggregate_path, index=False)
            batch_manifest["spots_aggregate"] = str(aggregate_path)

    with (batch_dir / "batch_manifest.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(batch_manifest, f, sort_keys=False)

    if publish_dir:
        publish_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir = publish_dir / "batch_manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{batch_dir.name}.yaml"
        shutil.copy2(batch_dir / "batch_manifest.yaml", manifest_path)
        batch_manifest["published_manifest_path"] = str(manifest_path)
        if "spots_aggregate" in batch_manifest:
            aggregate_src = Path(batch_manifest["spots_aggregate"])
            publish_root = publish_dir / batch_dir.name
            publish_root.mkdir(parents=True, exist_ok=True)
            aggregate_dest = publish_root / aggregate_src.name
            shutil.copy2(aggregate_src, aggregate_dest)
            batch_manifest["published_spots_aggregate"] = str(aggregate_dest)
        with (batch_dir / "batch_manifest.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(batch_manifest, f, sort_keys=False)
        shutil.copy2(batch_dir / "batch_manifest.yaml", manifest_path)

    print(f"Integrated batch complete: {batch_dir}")
    return batch_dir


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



