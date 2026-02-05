"""StarDist utilities.

We keep StarDist-specific concerns here so Slice1 drivers/kernels stay clean.

Model storage
-------------
The model files (config.json, thresholds.json, weights_*.h5) are expected to live
**outside the git repo** in your data bench, e.g.

    $BIOIMG_DATA_ROOT/models/y22m01d12_model_0/

This avoids bloating the repo while keeping runs reproducible via the manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class StardistModelRef:
    """Reference to a StarDist model on disk."""

    model_dir: Path

    @property
    def name(self) -> str:
        return self.model_dir.name

    @property
    def basedir(self) -> Path:
        return self.model_dir.parent


def assert_stardist_model_dir(model_dir: Path) -> None:
    """Lightweight sanity checks for a StarDist model folder."""

    required = ["config.json", "thresholds.json", "weights_best.h5"]
    missing = [f for f in required if not (model_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"StarDist model folder is missing required files: {missing}\n"
            f"model_dir={model_dir}"
        )


def load_stardist2d(model_ref: StardistModelRef) -> Any:
    """Load a StarDist2D model.

    Notes
    -----
    This function performs disk I/O (loads weights). Keep it in the *driver*,
    not in kernels, to preserve the Kernel/Driver separation.
    """

    try:
        from stardist.models import StarDist2D  # type: ignore
    except Exception as e:
        msg = str(e)

        # Common failure mode: StarDist is installed, but numba is missing.
        # (StarDist uses numba-accelerated geometry ops.)
        if "no module named" in msg.lower() and "numba" in msg.lower():
            raise ImportError(
                "StarDist imported but its dependency 'numba' is missing.\n\n"
                "Fix (recommended):\n"
                "  conda activate bioimg-pipeline\n"
                "  conda env update -f environment.yml --prune\n\n"
                "If you need a one-liner:\n"
                "  conda install -c conda-forge numba\n\n"
                "Then re-run:\n"
                "  python scripts/verify_setup.py"
            ) from e

        raise ImportError(
            "StarDist is not usable (import failed).\n\n"
            "Fix (recommended):\n"
            "  conda activate bioimg-pipeline\n"
            "  conda env update -f environment.yml --prune\n\n"
            f"Original import error:\n  {type(e).__name__}: {e}"
        ) from e

    assert_stardist_model_dir(model_ref.model_dir)

    # StarDist expects a (basedir, name) pair.
    return StarDist2D(None, name=model_ref.name, basedir=str(model_ref.basedir))


def get_model_thresholds(model: Any) -> Dict[str, float]:
    """Return model thresholds if available."""

    out: Dict[str, float] = {}
    thr = getattr(model, "thresholds", None)
    if isinstance(thr, dict):
        # stardist typically uses {'prob': ..., 'nms': ...}
        for k, v in thr.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
    return out

