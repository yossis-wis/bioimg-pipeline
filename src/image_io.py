"""Image I/O helpers.

Goal: keep format quirks (TIFF vs Imaris .ims) out of kernels.

Design choice (matches docs/ARCHITECTURE.md):
- Drivers handle filesystem paths + choose which plane/channel to load.
- Kernels operate on in-memory 2D arrays.

The functions here intentionally return **only a single 2D plane**.
Slice0 and Slice1 are defined as operating on a single frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class PlaneSelection:
    """How to select a single 2D plane from a multi-dimensional file.

    Notes
    -----
    * `channel` is **1-based** for user friendliness:
        - channel=1 corresponds to "Channel 0" in Imaris
        - channel=2 corresponds to "Channel 1" in Imaris
    """

    channel: int = 1

    # Only used for .ims inputs
    ims_resolution_level: int = 0
    ims_time_index: int = 0
    ims_z_index: int = 0


def read_image_2d(input_path: Path, selection: PlaneSelection) -> np.ndarray:
    """Read a single 2D plane from TIFF or Imaris .ims.

    Parameters
    ----------
    input_path:
        Path to input image (.tif/.tiff or .ims).
    selection:
        Plane/channel selection.

    Returns
    -------
    np.ndarray
        2D array (Y, X). The dtype is preserved from file.
    """

    suffix = input_path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        return _read_tiff_2d(input_path, channel=selection.channel)
    if suffix == ".ims":
        return _read_ims_2d(
            input_path,
            channel=selection.channel,
            resolution_level=selection.ims_resolution_level,
            time_index=selection.ims_time_index,
            z_index=selection.ims_z_index,
        )

    raise ValueError(f"Unsupported input format: {input_path}")


def _read_tiff_2d(path: Path, channel: int) -> np.ndarray:
    """Read TIFF and select a 2D plane.

    For Slice0/Slice1 we primarily expect either:
    * (Y, X)
    * (C, Y, X) where C is small (1-4)

    If you later hit real data with T/Z/C, we can extend this function.
    """

    try:
        import tifffile
    except ImportError as e:
        raise ImportError("tifffile is required to read TIFF inputs") from e

    arr = np.asarray(tifffile.imread(str(path)))

    # Selection rules:
    # - If already 2D, return.
    # - If 3D, treat axis0 as channel-like when possible.
    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        ch = int(channel)
        idx = ch - 1 if ch >= 1 else 0
        if 0 <= idx < arr.shape[0]:
            return arr[idx]
        raise ValueError(
            f"TIFF has {arr.shape[0]} channel(s); requested channel {ch}: {path}"
        )

    raise ValueError(
        f"TIFF has {arr.ndim} dimensions {arr.shape}, which is ambiguous for 2D reading. "
        "Please ensure input is (Y,X) or (C,Y,X)."
    )


def _read_ims_2d(
    path: Path,
    channel: int,
    resolution_level: int,
    time_index: int,
    z_index: int,
) -> np.ndarray:
    """Read a single 2D plane from an Imaris .ims file.

    Imaris .ims files are HDF5 containers.

    We follow the common dataset layout:
        /DataSet/ResolutionLevel {r}/TimePoint {t}/Channel {c}/Data

    Where `c` is 0-based.
    """

    try:
        import h5py
    except ImportError as e:
        raise ImportError("h5py is required to read .ims inputs") from e

    # channel is treated as 1-based if >=1; allow channel=0 to mean first channel
    ch = int(channel)
    c = ch - 1 if ch >= 1 else 0

    ds_path = (
        f"DataSet/ResolutionLevel {int(resolution_level)}/"
        f"TimePoint {int(time_index)}/"
        f"Channel {c}/Data"
    )

    with h5py.File(str(path), "r") as f:
        if ds_path not in f:
            # Provide helpful debug info
            found = []
            def _visit(name, obj):
                if isinstance(obj, h5py.Dataset) and name.endswith("/Data"):
                    found.append(name)
            f.visititems(_visit)
            preview = "\n".join(found[:10])
            raise KeyError(
                f"Dataset path not found: {ds_path}\n"
                f"Found these /Data datasets (showing up to 10):\n{preview}"
            )

        dset = f[ds_path]

        # Typical: (Z, Y, X) or (Y, X)
        if dset.ndim == 3:
            z = int(z_index)
            if z < 0 or z >= dset.shape[0]:
                raise IndexError(
                    f"z_index={z_index} out of range for IMS data shape={tuple(dset.shape)}"
                )
            arr = np.asarray(dset[z, :, :])
        elif dset.ndim == 2:
            arr = np.asarray(dset[:, :])
        else:
            # Fallback (may load more data than needed)
            raw = np.asarray(dset)
            raw = np.squeeze(raw)
            if raw.ndim == 2:
                arr = raw
            else:
                flat = raw.reshape(-1, raw.shape[-2], raw.shape[-1])
                arr = flat[0]

    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(
            f"Unsupported IMS data shape for Slice0/Slice1: shape={arr.shape}. "
            "Expected (Y,X) or (Z,Y,X)."
        )
    return arr
