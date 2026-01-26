from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _as_float(x) -> Optional[float]:
    """Best-effort conversion of HDF5/TIFF metadata scalars to float."""
    if x is None:
        return None
    # h5py attrs can be numpy scalars/arrays, bytes, or str
    if isinstance(x, (np.generic,)):
        try:
            return float(x)
        except Exception:
            return None
    if isinstance(x, (np.ndarray,)):
        if x.size == 1:
            try:
                return float(x.reshape(()))
            except Exception:
                return None
        # Sometimes stored as a 1-element bytes array.
        try:
            if x.size == 1:
                return float(x.flat[0])
        except Exception:
            return None
        return None
    if isinstance(x, (bytes, bytearray)):
        try:
            return float(x.decode("utf-8"))
        except Exception:
            return None
    if isinstance(x, str):
        try:
            return float(x)
        except Exception:
            return None
    try:
        return float(x)
    except Exception:
        return None


def _unit_to_nm_factor(unit: str) -> Optional[float]:
    u = unit.strip().lower()
    # Normalize common micro symbols.
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
    """Return (px_size_y_nm, px_size_x_nm, note)."""
    import tifffile
    import xml.etree.ElementTree as ET

    with tifffile.TiffFile(str(path)) as tif:
        # 1) OME-XML (most reliable for microscopy TIFFs).
        ome = tif.ome_metadata
        if ome:
            try:
                root = ET.fromstring(ome)
                # OME namespace handling: search any Pixels element.
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
            except Exception as exc:
                # fall through to other methods
                pass

        # 2) TIFF resolution tags (common for non-OME TIFFs, but often absent for microscopy).
        page = tif.pages[0]
        tags = page.tags
        if "XResolution" in tags and "YResolution" in tags and "ResolutionUnit" in tags:
            try:
                xres = tags["XResolution"].value  # (num, den) rational or float
                yres = tags["YResolution"].value
                unit = tags["ResolutionUnit"].value  # 2=inches, 3=cm
                def _to_float(res):
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
                        return px_y_um * 1000.0, px_x_um * 1000.0, f"TIFF resolution tags (ResolutionUnit={note_unit})"
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

        # Units (Imaris typically uses µm).
        unit_raw = g.attrs.get("Unit", g.attrs.get("unit", None))
        unit = None
        if isinstance(unit_raw, (bytes, bytearray)):
            unit = unit_raw.decode("utf-8")
        elif isinstance(unit_raw, str):
            unit = unit_raw
        elif unit_raw is not None:
            try:
                unit = str(unit_raw)
            except Exception:
                unit = None

        # Extents in physical units:
        ext_min0 = _as_float(g.attrs.get("ExtMin0"))
        ext_max0 = _as_float(g.attrs.get("ExtMax0"))
        ext_min1 = _as_float(g.attrs.get("ExtMin1"))
        ext_max1 = _as_float(g.attrs.get("ExtMax1"))

        # Image sizes (pixels):
        size_x = _as_float(g.attrs.get("X", g.attrs.get("SizeX", None)))
        size_y = _as_float(g.attrs.get("Y", g.attrs.get("SizeY", None)))

        if None in (ext_min0, ext_max0, ext_min1, ext_max1, size_x, size_y):
            return None, None, "Could not parse ExtMin/ExtMax and X/Y attributes from DataSetInfo/Image."

        # Convert to float ints
        nx = int(round(float(size_x)))
        ny = int(round(float(size_y)))
        if nx <= 0 or ny <= 0:
            return None, None, "Invalid X/Y sizes in DataSetInfo/Image."

        dx = (float(ext_max0) - float(ext_min0)) / float(nx)
        dy = (float(ext_max1) - float(ext_min1)) / float(ny)

        if unit is None:
            # Assume µm (most common in Imaris)
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


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Inspect XY pixel size (nm/px) from TIFF or Imaris .ims metadata.",
    )
    ap.add_argument(
        "--input",
        required=True,
        help="Input image path (absolute or relative to $BIOIMG_DATA_ROOT). Supports .tif/.tiff and .ims.",
    )
    args = ap.parse_args()

    p = Path(args.input)
    if not p.is_absolute():
        root = os.environ.get("BIOIMG_DATA_ROOT")
        if not root:
            raise RuntimeError("BIOIMG_DATA_ROOT is not set; provide an absolute --input path.")
        p = (Path(root) / p).resolve()

    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p}")

    ext = p.suffix.lower()
    if ext in (".tif", ".tiff"):
        py_nm, px_nm, note = inspect_tiff_pixel_size(p)
    elif ext == ".ims":
        py_nm, px_nm, note = inspect_ims_pixel_size(p)
    else:
        raise ValueError(f"Unsupported extension: {ext}")

    print("=== inspect_pixel_size ===")
    print("input:", p)
    print("format:", ext)
    print("note:", note)
    print()

    if py_nm is None or px_nm is None:
        print("ERROR: could not determine pixel size from metadata.")
        print("You can still set spot_pixel_size_nm manually if you know the calibration.")
        return 2

    print(f"pixel_size_y: {py_nm:.3f} nm/px")
    print(f"pixel_size_x: {px_nm:.3f} nm/px")
    print(f"pixel_size_xy_mean: {(0.5*(py_nm+px_nm)):.3f} nm/px")
    print()
    print("Suggested config entry (nm/px):")
    print(f"  spot_pixel_size_nm: {0.5*(py_nm+px_nm):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
