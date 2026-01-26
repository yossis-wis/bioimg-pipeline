"""Pixel-size inference utilities for microscopy files.

This module centralizes the logic used by:
- scripts/inspect_pixel_size.py
- drivers that want to auto-validate or auto-infer pixel size

Supported formats
-----------------
- TIFF / OME-TIFF: reads OME-XML PhysicalSizeX/Y where available
- Imaris .ims (HDF5): reads DataSetInfo/Image extents and pixel counts

Why this exists
---------------
Imaris sometimes stores numeric HDF5 attributes (e.g. ExtMin0) as *arrays of single-character
strings* (dtype ``|S1``), e.g. [b'6', b'3', b'.', b'1', ...]. A naive ``float(attr)`` fails.
We therefore include robust parsers for scalars encoded as character arrays.

All returned sizes are in **nm/px**.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _as_str(x) -> Optional[str]:
    """Best-effort conversion of HDF5/TIFF metadata scalars to string."""
    if x is None:
        return None

    # numpy arrays are common for h5py attrs
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return None

        # Character arrays: join all elements
        if x.dtype.kind in ("S", "U"):
            flat = x.ravel().tolist()
            if not flat:
                return None
            if x.dtype.kind == "S":
                # bytes elements
                try:
                    return b"".join(flat).decode("utf-8")
                except Exception:
                    try:
                        return "".join([b.decode("utf-8") for b in flat])
                    except Exception:
                        return None
            # unicode elements
            try:
                return "".join(flat)
            except Exception:
                return None

        # Non-string array: if singleton, stringify that element
        if x.size == 1:
            try:
                return str(x.reshape(()))
            except Exception:
                return None

        # Vector numeric metadata is not a scalar; don't guess
        return None

    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8")
        except Exception:
            return None

    try:
        return str(x)
    except Exception:
        return None


def _as_float(x) -> Optional[float]:
    """Best-effort conversion of metadata scalar to float."""
    if x is None:
        return None

    # numpy scalar
    if isinstance(x, (np.generic,)):
        try:
            return float(x)
        except Exception:
            return None

    # numpy array attr
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return None
        if x.size == 1:
            try:
                return float(x.reshape(()))
            except Exception:
                # Might be a 1-element string array
                s = _as_str(x)
                if s is None:
                    return None
                try:
                    return float(s)
                except Exception:
                    return None

        # Some Imaris attrs are stored as arrays of single-character strings.
        s = _as_str(x)
        if s is None:
            return None
        try:
            return float(s)
        except Exception:
            return None

    if isinstance(x, (bytes, bytearray)):
        try:
            return float(x.decode("utf-8"))
        except Exception:
            return None

    if isinstance(x, str):
        try:
            return float(x.strip())
        except Exception:
            return None

    try:
        return float(x)
    except Exception:
        return None


def _unit_to_nm_factor(unit: str) -> Optional[float]:
    u = unit.strip().lower()
    u = u.replace("μ", "u").replace("µ", "u")
    if u in ("nm", "nanometer", "nanometers"):
        return 1.0
    if u in ("um", "micrometer", "micrometers", "micron", "microns"):
        return 1000.0
    if u in ("mm", "millimeter", "millimeters"):
        return 1_000_000.0
    if u in ("m", "meter", "meters"):
        return 1e9
    return None


def inspect_tiff_pixel_size(path: Path) -> Tuple[Optional[float], Optional[float], str]:
    """Return (px_size_y_nm, px_size_x_nm, note) from a TIFF."""
    import tifffile
    import xml.etree.ElementTree as ET

    with tifffile.TiffFile(str(path)) as tif:
        # 1) OME-XML (most reliable for microscopy TIFFs).
        ome = tif.ome_metadata
        if ome:
            try:
                root = ET.fromstring(ome)
                pixels = root.find(".//{*}Pixels")
                if pixels is not None:
                    psx = pixels.attrib.get("PhysicalSizeX")
                    psy = pixels.attrib.get("PhysicalSizeY")
                    ux = pixels.attrib.get("PhysicalSizeXUnit", pixels.attrib.get("PhysicalSizeUnit", "µm"))
                    uy = pixels.attrib.get("PhysicalSizeYUnit", pixels.attrib.get("PhysicalSizeUnit", "µm"))
                    fx = _unit_to_nm_factor(ux or "µm")
                    fy = _unit_to_nm_factor(uy or "µm")
                    if psx is not None and psy is not None and fx and fy:
                        px_x_nm = float(psx) * fx
                        px_y_nm = float(psy) * fy
                        return px_y_nm, px_x_nm, f"OME metadata (PhysicalSizeX/Y in {ux}/{uy})"
            except Exception:
                pass

        # 2) TIFF resolution tags (common for non-OME TIFFs, but often absent for microscopy).
        page = tif.pages[0]
        tags = page.tags
        if "XResolution" in tags and "YResolution" in tags and "ResolutionUnit" in tags:
            try:
                xres = tags["XResolution"].value
                yres = tags["YResolution"].value
                unit = tags["ResolutionUnit"].value  # 2=inches, 3=cm

                def _to_float(res) -> float:
                    if isinstance(res, tuple) and len(res) == 2:
                        return float(res[0]) / float(res[1])
                    return float(res)

                xres_f = _to_float(xres)
                yres_f = _to_float(yres)
                if xres_f > 0 and yres_f > 0:
                    if unit == 2:  # inch
                        um_per_unit = 25400.0
                        note_unit = "inch"
                    elif unit == 3:  # cm
                        um_per_unit = 10000.0
                        note_unit = "cm"
                    else:
                        um_per_unit = None
                        note_unit = str(unit)
                    if um_per_unit is not None:
                        px_x_um = um_per_unit / xres_f
                        px_y_um = um_per_unit / yres_f
                        return (
                            px_y_um * 1000.0,
                            px_x_um * 1000.0,
                            f"TIFF resolution tags (ResolutionUnit={note_unit})",
                        )
            except Exception:
                pass

    return None, None, "No usable pixel size metadata found in TIFF."


def inspect_ims_pixel_size(path: Path) -> Tuple[Optional[float], Optional[float], str]:
    """Return (px_size_y_nm, px_size_x_nm, note) for Imaris .ims (HDF5)."""
    import h5py

    with h5py.File(str(path), "r") as f:
        g = f.get("DataSetInfo/Image")
        if g is None:
            return None, None, "Missing group DataSetInfo/Image in .ims."

        unit_raw = g.attrs.get("Unit", g.attrs.get("unit", None))
        unit = _as_str(unit_raw)

        ext_min0 = _as_float(g.attrs.get("ExtMin0"))
        ext_max0 = _as_float(g.attrs.get("ExtMax0"))
        ext_min1 = _as_float(g.attrs.get("ExtMin1"))
        ext_max1 = _as_float(g.attrs.get("ExtMax1"))

        size_x = _as_float(g.attrs.get("X", g.attrs.get("SizeX", None)))
        size_y = _as_float(g.attrs.get("Y", g.attrs.get("SizeY", None)))

        if None in (ext_min0, ext_max0, ext_min1, ext_max1, size_x, size_y):
            return None, None, "Could not parse ExtMin/ExtMax and X/Y attributes from DataSetInfo/Image."

        nx = int(round(float(size_x)))
        ny = int(round(float(size_y)))
        if nx <= 0 or ny <= 0:
            return None, None, "Invalid X/Y sizes in DataSetInfo/Image."

        dx = (float(ext_max0) - float(ext_min0)) / float(nx)
        dy = (float(ext_max1) - float(ext_min1)) / float(ny)

        if unit is None:
            factor = 1000.0
            unit_note = "unknown; assuming µm"
        else:
            factor = _unit_to_nm_factor(unit)
            if factor is None:
                factor = 1000.0
                unit_note = f"{unit} (unrecognized; assuming µm)"
            else:
                unit_note = unit

        px_x_nm = dx * factor
        px_y_nm = dy * factor
        return px_y_nm, px_x_nm, f"Imaris HDF5 extents (Unit={unit_note})"


def infer_pixel_size_nm(path: Path) -> Tuple[Optional[float], Optional[float], str]:
    """Infer XY pixel size (nm/px) from an image file."""
    ext = path.suffix.lower()
    if ext in (".tif", ".tiff"):
        return inspect_tiff_pixel_size(path)
    if ext == ".ims":
        return inspect_ims_pixel_size(path)
    return None, None, f"Unsupported extension: {ext}"


def infer_pixel_size_nm_mean(path: Path) -> Tuple[Optional[float], str]:
    """Return (mean_xy_nm, note)."""
    py_nm, px_nm, note = infer_pixel_size_nm(path)
    if py_nm is None or px_nm is None:
        return None, note
    return 0.5 * (float(py_nm) + float(px_nm)), note
