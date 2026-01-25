#!/usr/bin/env python3
"""Interactive spot-detection QC (matplotlib sliders).

This driver launches a matplotlib-only GUI that lets you *babysit* spot detection
thresholds (LoG quality ``q_min`` and mean-threshold ``u0_min``) on a single
2D plane.

It is intended to work well in:
- Spyder (Qt backend) for pixel-value inspection
- VS Code (interactive window)

Config support
--------------
If you pass ``--config`` (YAML), this driver understands the same input keys as
``drivers/run_integrated.py``:

- Exactly ONE of: ``input_relpath`` | ``input_relpaths`` | ``input_glob``
- ``stardist_model_dir`` (optional; resolved under ``$BIOIMG_DATA_ROOT`` unless absolute)
- ``channel_nuclei`` and ``channel_spots`` (1-based; ``channel_spots`` may be a list)
- ``ims_resolution_level``, ``ims_time_index``, ``ims_z_index``
- ``nuc_*`` and ``spot_*`` parameters

Notes
-----
- If ``input_glob`` matches multiple files, we select the first match (sorted)
  unless ``--input`` is provided.
- Paths that are not absolute are resolved under ``$BIOIMG_DATA_ROOT`` (if set).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(10):
        if (cur / "src").exists() and (cur / "drivers").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve()


REPO_ROOT = _find_repo_root(Path(__file__).parent)
sys.path.insert(0, str(REPO_ROOT / "src"))

from image_io import PlaneSelection  # noqa: E402
from qc_spot_interactive import run_interactive_spot_qc  # noqa: E402


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config YAML must be a mapping/dict: {path}")
    return data


def _resolve_under_data_root(value: str | Path, data_root: Optional[Path]) -> Path:
    """Resolve a possibly-relative path.

    If ``value`` is absolute -> use as-is.
    Else if ``data_root`` is provided -> resolve under it.
    Else resolve relative to current working directory.
    """
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    if data_root is not None:
        return (data_root / path).resolve()
    return path.resolve()


def _resolve_input_from_cfg(cfg: Dict[str, Any], data_root: Optional[Path]) -> Optional[Path]:
    """Return ONE input path chosen from config, or None if not specified."""
    input_relpath = cfg.get("input_relpath")
    input_relpaths = cfg.get("input_relpaths")
    input_glob = cfg.get("input_glob")

    provided = [v for v in (input_relpath, input_relpaths, input_glob) if v]
    if len(provided) > 1:
        raise ValueError("Set only one of input_relpath, input_relpaths, or input_glob")

    if input_relpath:
        return _resolve_under_data_root(input_relpath, data_root)

    if input_relpaths:
        if not isinstance(input_relpaths, (list, tuple)) or not input_relpaths:
            raise ValueError("input_relpaths must be a non-empty list")
        return _resolve_under_data_root(input_relpaths[0], data_root)

    if input_glob:
        pattern = str(input_glob)
        pat_path = Path(pattern)
        if not pat_path.is_absolute() and data_root is not None:
            pattern = str((data_root / pat_path).resolve())
        matches = sorted(glob.glob(pattern, recursive=True))
        if not matches:
            raise FileNotFoundError(f"input_glob matched no files: {pattern}")
        return Path(matches[0]).resolve()

    return None


def _coerce_channel_1based(value: Any, default: int) -> int:
    """channel_spots can be an int or list. For this interactive tool we use ONE channel."""
    if value is None:
        return int(default)
    if isinstance(value, (list, tuple)):
        if not value:
            return int(default)
        return int(value[0])
    return int(value)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input image path (.ims/.tif/.tiff). If relative, resolved vs --data-root (or $BIOIMG_DATA_ROOT).",
    )
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config (same key conventions as run_integrated).",
    )
    ap.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Optional data root for resolving relative paths (defaults to $BIOIMG_DATA_ROOT).",
    )

    ap.add_argument("--channel-nuclei", type=int, default=None, help="1-based channel index for nuclei image.")
    ap.add_argument("--channel-spots", type=int, default=None, help="1-based channel index for spot image.")

    ap.add_argument("--ims-resolution-level", type=int, default=None)
    ap.add_argument("--ims-time-index", type=int, default=None)
    ap.add_argument("--ims-z-index", type=int, default=None)

    ap.add_argument(
        "--stardist-model-dir",
        type=str,
        default=None,
        help="StarDist model directory (relative to $BIOIMG_DATA_ROOT by default).",
    )

    ap.add_argument("--nuc-prob-thresh", type=float, default=None, help="StarDist prob_thresh used when running the model.")
    ap.add_argument("--nuc-nms-thresh", type=float, default=None, help="StarDist nms_thresh used when running the model.")
    ap.add_argument("--nuc-normalize-pmin", type=float, default=None)
    ap.add_argument("--nuc-normalize-pmax", type=float, default=None)

    ap.add_argument("--spot-pixel-size-nm", type=float, default=None)
    ap.add_argument("--spot-lambda-nm", type=float, default=None)
    ap.add_argument("--spot-zR", type=float, default=None)
    ap.add_argument("--spot-se-size", type=int, default=None)

    ap.add_argument("--q-min-init", type=float, default=None, help="Initial q_min slider value.")
    ap.add_argument("--u0-min-init", type=float, default=None, help="Initial u0_min slider value.")
    ap.add_argument("--nuc-prob-min-init", type=float, default=None, help="Initial nuc_prob_min slider value (post-hoc filtering).")

    ap.add_argument("--backend", type=str, default=None, help="Optional matplotlib backend (e.g. QtAgg).")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = _load_yaml(Path(args.config))

    # Data root
    data_root: Optional[Path] = None
    if args.data_root:
        data_root = Path(args.data_root).expanduser().resolve()
    else:
        env_root = os.environ.get("BIOIMG_DATA_ROOT")
        if env_root:
            data_root = Path(env_root).expanduser().resolve()

    # Matplotlib backend (must happen before importing pyplot inside qc_spot_interactive)
    if args.backend:
        import matplotlib

        matplotlib.use(args.backend, force=True)

    # Input path (CLI overrides config)
    input_path: Optional[Path] = None
    if args.input:
        input_path = _resolve_under_data_root(args.input, data_root)
    else:
        input_path = _resolve_input_from_cfg(cfg, data_root)

    if input_path is None:
        raise SystemExit(
            "Need --input or a config that sets one of: input_relpath | input_relpaths | input_glob."
        )

    if not input_path.exists():
        raise SystemExit(
            f"Input file not found: {input_path}\n"
            "Fix by either:\n"
            "  1) passing --input <path>, or\n"
            "  2) editing your config to set input_relpath/input_relpaths/input_glob."
        )

    # Channels (1-based)
    channel_nuclei = int(args.channel_nuclei) if args.channel_nuclei is not None else _coerce_channel_1based(cfg.get("channel_nuclei"), 1)
    channel_spots = int(args.channel_spots) if args.channel_spots is not None else _coerce_channel_1based(cfg.get("channel_spots"), 2)

    # Plane selection defaults
    ims_resolution_level = int(args.ims_resolution_level) if args.ims_resolution_level is not None else int(cfg.get("ims_resolution_level", 0))
    ims_time_index = int(args.ims_time_index) if args.ims_time_index is not None else int(cfg.get("ims_time_index", 0))
    ims_z_index = int(args.ims_z_index) if args.ims_z_index is not None else int(cfg.get("ims_z_index", 0))

    # Base PlaneSelection (channel is overridden inside run_interactive_spot_qc)
    plane = PlaneSelection(
        channel=1,
        ims_resolution_level=ims_resolution_level,
        ims_time_index=ims_time_index,
        ims_z_index=ims_z_index,
    )

    # StarDist model dir: treat as relative to BIOIMG_DATA_ROOT unless absolute
    model_dir_str = args.stardist_model_dir if args.stardist_model_dir is not None else cfg.get("stardist_model_dir")
    stardist_model_dir: Optional[Path] = None
    if model_dir_str not in (None, ""):
        stardist_model_dir = _resolve_under_data_root(model_dir_str, data_root)
        if not stardist_model_dir.exists():
            raise SystemExit(
                f"StarDist model_dir not found: {stardist_model_dir}\n"
                "If you used a relative path, make sure BIOIMG_DATA_ROOT is set correctly."
            )

    # Nuclei params
    nuc_prob_thresh = float(args.nuc_prob_thresh) if args.nuc_prob_thresh is not None else float(cfg.get("nuc_prob_thresh", 0.10))
    nuc_nms_thresh = float(args.nuc_nms_thresh) if args.nuc_nms_thresh is not None else float(cfg.get("nuc_nms_thresh", 0.00))
    nuc_normalize_pmin = float(args.nuc_normalize_pmin) if args.nuc_normalize_pmin is not None else float(cfg.get("nuc_normalize_pmin", 1.0))
    nuc_normalize_pmax = float(args.nuc_normalize_pmax) if args.nuc_normalize_pmax is not None else float(cfg.get("nuc_normalize_pmax", 99.8))

    # Spot params
    spot_pixel_size_nm = float(args.spot_pixel_size_nm) if args.spot_pixel_size_nm is not None else float(cfg.get("spot_pixel_size_nm", 65.0))
    spot_lambda_nm = float(args.spot_lambda_nm) if args.spot_lambda_nm is not None else float(cfg.get("spot_lambda_nm", 667.0))
    spot_zR = float(args.spot_zR) if args.spot_zR is not None else float(cfg.get("spot_zR", 344.5))
    spot_se_size = int(args.spot_se_size) if args.spot_se_size is not None else int(cfg.get("spot_se_size", 15))

    q_min_init = float(args.q_min_init) if args.q_min_init is not None else float(cfg.get("spot_q_min", cfg.get("spot_q_min_min", 1.0) if isinstance(cfg.get("spot_q_min_min"), (int, float)) else 1.0))
    u0_min_init = float(args.u0_min_init) if args.u0_min_init is not None else float(cfg.get("spot_u0_min", 30.0))
    nuc_prob_min_init = float(args.nuc_prob_min_init) if args.nuc_prob_min_init is not None else float(cfg.get("nuc_prob_thresh", 0.10))

    print("interactive_spot_qc input:", input_path)
    print("channels: nuclei=", channel_nuclei, "spots=", channel_spots)
    print("ims selection:", dict(rl=ims_resolution_level, t=ims_time_index, z=ims_z_index))

    run_interactive_spot_qc(
        input_path=input_path,
        nuclei_channel_1based=channel_nuclei,
        spot_channel_1based=channel_spots,
        plane=plane,
        stardist_model_dir=stardist_model_dir,
        nuc_prob_thresh=nuc_prob_thresh,
        nuc_nms_thresh=nuc_nms_thresh,
        nuc_normalize_pmin=nuc_normalize_pmin,
        nuc_normalize_pmax=nuc_normalize_pmax,
        spot_pixel_size_nm=spot_pixel_size_nm,
        spot_lambda_nm=spot_lambda_nm,
        spot_zR=spot_zR,
        spot_se_size=spot_se_size,
        q_min_init=q_min_init,
        u0_min_init=u0_min_init,
        nuc_prob_min_init=nuc_prob_min_init,
    )


if __name__ == "__main__":
    main()
