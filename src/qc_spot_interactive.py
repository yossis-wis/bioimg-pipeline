"""
Interactive (matplotlib-only) babysitting tool for spot detection.

Design goals:
- Works in Spyder (Qt backend) and VS Code (interactive window), not Napari.
- Lets you *visually* tune thresholds (LoG quality q_min, u0_min, optional nucleus prob min)
  without re-running the expensive LoG convolution every time.
- Allows pixel-value inspection via a custom `ax.format_coord` so the status bar shows intensity.

This is intentionally a lightweight "QC / debug" tool, not a production pipeline step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from image_io import PlaneSelection, read_image_2d
from slice0_kernel import Slice0Params, detect_spots_debug
from slice1_nuclei_kernel import Slice1NucleiParams, segment_nuclei_stardist
from stardist_utils import StardistModelRef, load_stardist2d


@dataclass(frozen=True)
class InteractiveSpotQCResult:
    """What we computed (useful for unit tests / debugging)."""

    nuclei_labels: Optional[np.ndarray]
    nuclei_meta: Optional[Dict[str, Any]]
    spots_df_all: pd.DataFrame
    spot_debug: Any  # Slice0Debug (kept as Any to avoid import cycles)


def _resolve_path(p: Path) -> Path:
    """Resolve a path without requiring it to exist at import-time."""
    try:
        return p.expanduser().resolve()
    except Exception:
        return p.expanduser()


def _format_coord_factory(img: np.ndarray) -> Any:
    """
    Make a matplotlib `Axes.format_coord` function that shows pixel intensity.

    This is the simplest way to get "hover to see value" in Qt backends / Spyder.
    """
    h, w = img.shape[:2]

    def _fmt(x: float, y: float) -> str:
        # Data coords in imshow (origin='upper'): integer centers at (col,row).
        col = int(np.round(x))
        row = int(np.round(y))
        if 0 <= row < h and 0 <= col < w:
            z = img[row, col]
            if np.issubdtype(type(z), np.floating):
                return f"x={col:d}, y={row:d}, I={float(z):.4g}"
            return f"x={col:d}, y={row:d}, I={z}"
        return f"x={col:d}, y={row:d}"

    return _fmt


def _robust_clim(img: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.8) -> Tuple[float, float]:
    vals = np.asarray(img, dtype=float)
    vmin, vmax = np.percentile(vals, [p_lo, p_hi])
    if not np.isfinite(vmin):
        vmin = float(np.nanmin(vals))
    if not np.isfinite(vmax):
        vmax = float(np.nanmax(vals))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def _mask_edge_segments(mask: np.ndarray) -> np.ndarray:
    """Return external pixel-edge segments for a binary mask.

    This draws *between* pixels (at half-integer coordinates) so the overlay
    does not paint over the pixel values themselves.

    Coordinate convention
    ---------------------
    With matplotlib ``imshow(..., origin='upper')`` default limits, pixel centers
    are at integer coordinates (0..W-1), and pixel edges lie on half-integers.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or m.size == 0 or not np.any(m):
        return np.zeros((0, 2, 2), dtype=float)

    h, w = m.shape
    segs: list[list[tuple[float, float]]] = []

    for i in range(h):
        for j in range(w):
            if not m[i, j]:
                continue

            x0 = float(j) - 0.5
            x1 = float(j) + 0.5
            y0 = float(i) - 0.5
            y1 = float(i) + 0.5

            # Top edge
            if i == 0 or not m[i - 1, j]:
                segs.append([(x0, y0), (x1, y0)])
            # Bottom edge
            if i == h - 1 or not m[i + 1, j]:
                segs.append([(x0, y1), (x1, y1)])
            # Left edge
            if j == 0 or not m[i, j - 1]:
                segs.append([(x0, y0), (x0, y1)])
            # Right edge
            if j == w - 1 or not m[i, j + 1]:
                segs.append([(x1, y0), (x1, y1)])

    if not segs:
        return np.zeros((0, 2, 2), dtype=float)
    return np.asarray(segs, dtype=float)


def _add_edge_overlay(ax: Any, segs: np.ndarray, *, color: str, lw: float = 1.2) -> Optional[Any]:
    """Add a LineCollection overlay for precomputed mask-edge segments."""
    if segs is None or np.asarray(segs).size == 0:
        return None

    # Import lazily so callers can set their backend before importing matplotlib.
    from matplotlib.collections import LineCollection

    lc = LineCollection(
        np.asarray(segs, dtype=float),
        colors=[color],
        linewidths=float(lw),
        capstyle="projecting",
        joinstyle="miter",
        zorder=5,
    )
    ax.add_collection(lc)
    return lc


def run_interactive_spot_qc(
    *,
    input_path: Path,
    nuclei_channel_1based: int,
    spot_channel_1based: int,
    plane: PlaneSelection,
    stardist_model_dir: Optional[Path],
    # nuclei segmentation params (run once)
    nuc_prob_thresh: float = 0.10,
    nuc_nms_thresh: float = 0.00,
    nuc_normalize_pmin: float = 1.0,
    nuc_normalize_pmax: float = 99.8,
    # spot detection physical params
    spot_pixel_size_nm: float = 65.0,
    spot_lambda_nm: float = 667.0,
    spot_zR: float = 344.5,
    spot_se_size: int = 3,
    # Optional TrackMate-style controls (must be set before launching the UI)
    spot_radius_nm: Optional[float] = None,
    spot_do_median_filter: bool = False,
    spot_do_subpixel_localization: bool = False,
    # initial slider values
    q_min_init: float = 1.0,
    u0_min_init: float = 30.0,
    nuc_prob_min_init: float = 0.10,
    # optional masks
    valid_mask: Optional[np.ndarray] = None,
) -> InteractiveSpotQCResult:
    """
    Launch the interactive QC UI.

    Notes:
    - `nuc_prob_thresh` / `nuc_nms_thresh` are used when RUNNING StarDist.
      If you want to *interactively filter* nuclei by confidence without rerunning,
      use the "nuc_prob_min" slider (based on per-instance probabilities).
    - Spot detection is run once with permissive thresholds to precompute per-candidate
      (q, u0, nucleus_id). Sliders then only change filtering/visualization.
    """
    # Import matplotlib lazily so callers can set a backend before import if they want.
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button

    input_path = _resolve_path(Path(input_path))
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    # --- Load planes ---
    nuclei_plane = read_image_2d(
        input_path,
        PlaneSelection(
            channel=int(nuclei_channel_1based),
            ims_resolution_level=plane.ims_resolution_level,
            ims_time_index=plane.ims_time_index,
            ims_z_index=plane.ims_z_index,
        ),
    )
    spot_plane = read_image_2d(
        input_path,
        PlaneSelection(
            channel=int(spot_channel_1based),
            ims_resolution_level=plane.ims_resolution_level,
            ims_time_index=plane.ims_time_index,
            ims_z_index=plane.ims_z_index,
        ),
    )

    # --- Segment nuclei (optional) ---
    nuclei_labels: Optional[np.ndarray] = None
    nuclei_meta: Optional[Dict[str, Any]] = None
    if stardist_model_dir is None:
        print("[nuclei] stardist_model_dir not provided -> skipping nuclei segmentation.")
    else:
        print(f"[nuclei] Running StarDist nuclei segmentation (model_dir={stardist_model_dir}) ...")

    if stardist_model_dir is not None:
        model_dir = _resolve_path(Path(stardist_model_dir))
        model = load_stardist2d(StardistModelRef(model_dir=model_dir))

        nuc_params = Slice1NucleiParams(
            prob_thresh=float(nuc_prob_thresh),
            nms_thresh=float(nuc_nms_thresh),
            normalize_pmin=float(nuc_normalize_pmin),
            normalize_pmax=float(nuc_normalize_pmax),
        )
        seg_out = segment_nuclei_stardist(nuclei_plane, model=model, params=nuc_params)
        if isinstance(seg_out, tuple) and len(seg_out) == 2:
            nuclei_labels, nuclei_meta = seg_out
        else:
            nuclei_labels = seg_out  # type: ignore[assignment]
            nuclei_meta = None

    if nuclei_labels is not None:
        try:
            n_labels = int(np.max(nuclei_labels))
        except Exception:
            n_labels = 0
        print(f"[nuclei] Segmentation result: {n_labels} labels")

    # Build a label->prob lookup (label IDs are 1..N).
    label_prob: Optional[np.ndarray] = None
    if nuclei_meta and isinstance(nuclei_meta, dict) and "instance_probs" in nuclei_meta:
        try:
            probs = np.asarray(nuclei_meta["instance_probs"], dtype=float).ravel()
            if probs.size:
                label_prob = np.concatenate([[0.0], probs])  # index by label_id directly
        except Exception:
            label_prob = None

    # --- Run spot detection ONCE with permissive thresholds to precompute candidate metrics ---
    spot_params = Slice0Params(
        pixel_size_nm=float(spot_pixel_size_nm),
        lambda_nm=float(spot_lambda_nm),
        zR=float(spot_zR),
        se_size=int(spot_se_size),
        spot_radius_nm=(float(spot_radius_nm) if spot_radius_nm is not None else None),
        do_median_filter=bool(spot_do_median_filter),
        do_subpixel_localization=bool(spot_do_subpixel_localization),
        # permissive so we compute u0 for all candidates in the masked maxima set
        q_min=-1e9,
        u0_min=-1e9,
    )

    spots_df_all, dbg = detect_spots_debug(
        spot_plane,
        spot_params,
        valid_mask=valid_mask,
        nuclei_labels=nuclei_labels,
    )

    # Convenience print for QC: TrackMate GUI uses *estimated blob diameter* = 2 * radius.
    tm_diameter_nm = 2.0 * float(dbg.w0_nm)
    tm_diameter_um = tm_diameter_nm / 1000.0
    tm_diameter_px = tm_diameter_nm / float(spot_pixel_size_nm)
    print(f"TrackMate GUI blob diameter ≈ {tm_diameter_um:.4f} µm ({tm_diameter_nm:.1f} nm; {tm_diameter_px:.2f} px)")

    if spots_df_all.empty:
        raise RuntimeError("No spot candidates found (even with permissive thresholds).")

    # Pull arrays for fast filtering / plotting
    xs = spots_df_all["x_px"].to_numpy(dtype=float)
    ys = spots_df_all["y_px"].to_numpy(dtype=float)
    qv = spots_df_all["quality"].to_numpy(dtype=float)
    u0 = spots_df_all["u0"].to_numpy(dtype=float)

    nuc_id = None
    nuc_p = None
    if "nucleus_label" in spots_df_all.columns:
        nuc_id = spots_df_all["nucleus_label"].to_numpy(dtype=int)
        if label_prob is not None:
            nuc_p = label_prob[np.clip(nuc_id, 0, label_prob.size - 1)]
        else:
            # If we have nuclei_labels but no per-instance probabilities, treat as 1.0
            nuc_p = np.where(nuc_id > 0, 1.0, 0.0)
    else:
        nuc_id = np.zeros_like(qv, dtype=int)
        nuc_p = np.ones_like(qv, dtype=float)

    # For LoG response visualization (unpadded)
    pad_y = int(getattr(dbg, "pad_y", 0) or 0)
    pad_x = int(getattr(dbg, "pad_x", 0) or 0)
    conv_pad = np.asarray(dbg.image_conv_padded)
    if pad_y > 0 and pad_x > 0:
        conv = conv_pad[pad_y:-pad_y, pad_x:-pad_x]
    else:
        conv = conv_pad

    # LoG response (TrackMate-style kernel yields bright spots directly)
    resp = conv

    # Slider ranges (robust)
    q_lo, q_hi = np.percentile(qv[np.isfinite(qv)], [0.5, 99.5])
    u_lo, u_hi = np.percentile(u0[np.isfinite(u0)], [0.5, 99.5])
    q_lo = float(min(q_lo, q_min_init))
    q_hi = float(max(q_hi, q_min_init))
    u_lo = float(min(u_lo, u0_min_init))
    u_hi = float(max(u_hi, u0_min_init))

    # --- Matplotlib UI (main window) ---
    plt.close("all")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax_img, ax_resp = axes

    vmin_i, vmax_i = _robust_clim(spot_plane, 1.0, 99.8)
    im0 = ax_img.imshow(spot_plane, cmap="gray", vmin=vmin_i, vmax=vmax_i, interpolation="nearest", origin="upper")
    ax_img.set_title("Spot channel (raw)")
    ax_img.set_aspect("equal")
    ax_img.set_axis_off()
    ax_img.format_coord = _format_coord_factory(spot_plane)

    vmin_r, vmax_r = _robust_clim(resp, 1.0, 99.8)
    im1 = ax_resp.imshow(resp, cmap="gray", vmin=vmin_r, vmax=vmax_r, interpolation="nearest", origin="upper")
    ax_resp.set_title("LoG response (spots bright)")
    ax_resp.set_aspect("equal")
    ax_resp.set_axis_off()
    ax_resp.format_coord = _format_coord_factory(resp)

    # Scatter overlays (updated by sliders)
    sc0 = ax_img.scatter([], [], s=20, facecolors="none", edgecolors="yellow", linewidths=0.8)
    sc1 = ax_resp.scatter([], [], s=20, facecolors="none", edgecolors="yellow", linewidths=0.8)

    # Status text
    txt = fig.text(0.01, 0.01, "", ha="left", va="bottom", fontsize=10)

    # Slider layout (bottom)
    fig.subplots_adjust(bottom=0.20)

    ax_q = fig.add_axes([0.10, 0.13, 0.80, 0.03])
    s_q = Slider(ax_q, "q_min", q_lo, q_hi, valinit=float(q_min_init))

    ax_u = fig.add_axes([0.10, 0.08, 0.80, 0.03])
    s_u = Slider(ax_u, "u0_min", u_lo, u_hi, valinit=float(u0_min_init))

    ax_np = fig.add_axes([0.10, 0.03, 0.80, 0.03])
    s_np = Slider(ax_np, "nuc_prob_min", 0.0, 1.0, valinit=float(nuc_prob_min_init))

    # Button: reset
    ax_reset = fig.add_axes([0.92, 0.205, 0.07, 0.04])
    b_reset = Button(ax_reset, "Reset")

    # --- ROI window (updates on pick) ---
    wr = int(getattr(dbg, "window_radius_px", 15) or 15)
    win = 2 * wr + 1

    fig_roi, axes_roi = plt.subplots(1, 3, figsize=(12, 4))
    ax_nroi, ax_sroi, ax_rroi = axes_roi

    # Precompute in5 / out0 mask edge segments (fixed; drawn along pixel edges)
    in5_mask = np.asarray(dbg.in5_mask, dtype=bool)
    out0_mask = np.asarray(dbg.out0_mask, dtype=bool)

    in5_segs = _mask_edge_segments(in5_mask)
    out0_segs = _mask_edge_segments(out0_mask)

    # Create ROI images with placeholder data
    vmin_n, vmax_n = _robust_clim(nuclei_plane, 1.0, 99.8)
    im_nroi = ax_nroi.imshow(np.zeros((win, win), dtype=nuclei_plane.dtype), cmap="gray", vmin=vmin_n, vmax=vmax_n, interpolation="nearest", origin="upper")
    ax_nroi.set_title("Nuclei ROI")
    ax_nroi.set_aspect("equal")
    ax_nroi.set_axis_off()

    im_sroi = ax_sroi.imshow(np.zeros((win, win), dtype=spot_plane.dtype), cmap="gray", vmin=vmin_i, vmax=vmax_i, interpolation="nearest", origin="upper")
    ax_sroi.set_title("Spot ROI + masks")
    ax_sroi.set_aspect("equal")
    ax_sroi.set_axis_off()

    im_rroi = ax_rroi.imshow(np.zeros((win, win), dtype=resp.dtype), cmap="gray", vmin=vmin_r, vmax=vmax_r, interpolation="nearest", origin="upper")
    ax_rroi.set_title("LoG ROI")
    ax_rroi.set_aspect("equal")
    ax_rroi.set_axis_off()

    # Fixed mask edge overlays (drawn *between* pixels, so values remain readable)
    _add_edge_overlay(ax_sroi, in5_segs, color="yellow", lw=1.4)
    _add_edge_overlay(ax_sroi, out0_segs, color="cyan", lw=1.2)
    _add_edge_overlay(ax_rroi, in5_segs, color="yellow", lw=1.4)
    _add_edge_overlay(ax_rroi, out0_segs, color="cyan", lw=1.2)

    # Center marker (fixed)
    for axx in [ax_nroi, ax_sroi, ax_rroi]:
        axx.scatter([wr], [wr], s=50, facecolors="none", edgecolors="red", linewidths=1.2)

    # Nucleus contour artists (updated on selection)
    nuc_lines = []

    def _update_roi(i: int) -> None:
        nonlocal nuc_lines

        x0 = int(round(xs[i]))
        y0 = int(round(ys[i]))

        # Crop coordinates (detection excluded borders, but be defensive)
        y1 = max(0, y0 - wr)
        y2 = min(spot_plane.shape[0], y0 + wr + 1)
        x1 = max(0, x0 - wr)
        x2 = min(spot_plane.shape[1], x0 + wr + 1)

        # If near edges (shouldn't happen), pad with zeros
        def _crop_pad(img: np.ndarray) -> np.ndarray:
            crop = img[y1:y2, x1:x2]
            out = np.zeros((win, win), dtype=img.dtype)
            oy1 = wr - (y0 - y1)
            ox1 = wr - (x0 - x1)
            out[oy1 : oy1 + crop.shape[0], ox1 : ox1 + crop.shape[1]] = crop
            return out

        nroi = _crop_pad(nuclei_plane)
        sroi = _crop_pad(spot_plane)
        rroi = _crop_pad(resp)

        im_nroi.set_data(nroi)
        im_sroi.set_data(sroi)
        im_rroi.set_data(rroi)

        ax_nroi.format_coord = _format_coord_factory(nroi)
        ax_sroi.format_coord = _format_coord_factory(sroi)
        ax_rroi.format_coord = _format_coord_factory(rroi)

        # Remove old nucleus outlines
        for ln in nuc_lines:
            try:
                ln.remove()
            except Exception:
                pass
        nuc_lines = []

        nid = int(nuc_id[i]) if nuc_id is not None else 0
        nprob = float(nuc_p[i]) if nuc_p is not None else 1.0

        if nuclei_labels is not None and nid > 0:
            mask = (nuclei_labels[y1:y2, x1:x2] == nid)
            # pad to win x win (same offset as crop)
            mask_full = np.zeros((win, win), dtype=bool)
            oy1 = wr - (y0 - y1)
            ox1 = wr - (x0 - x1)
            mask_full[oy1 : oy1 + mask.shape[0], ox1 : ox1 + mask.shape[1]] = mask

            segs = _mask_edge_segments(mask_full)
            for axx in (ax_nroi, ax_sroi, ax_rroi):
                art = _add_edge_overlay(axx, segs, color="red", lw=1.3)
                if art is not None:
                    nuc_lines.append(art)

        # Titles with metrics
        q_i = float(qv[i])
        u0_i = float(u0[i])
        ax_sroi.set_title(f"Spot ROI | q={q_i:.3g}, u0={u0_i:.3g}")
        ax_nroi.set_title(f"Nuclei ROI | nucleus_id={nid} (prob={nprob:.3g})")
        ax_rroi.set_title("LoG ROI")

        fig_roi.canvas.draw_idle()

    # --- Filtering / plotting update ---
    kept_idx: np.ndarray = np.array([], dtype=int)

    def _update(_=None) -> None:
        nonlocal kept_idx
        q_thr = float(s_q.val)
        u_thr = float(s_u.val)
        np_thr = float(s_np.val)

        keep = (qv >= q_thr) & (u0 >= u_thr) & (nuc_p >= np_thr)
        kept_idx = np.flatnonzero(keep)

        sc0.set_offsets(np.c_[xs[kept_idx], ys[kept_idx]])
        sc1.set_offsets(np.c_[xs[kept_idx], ys[kept_idx]])

        txt.set_text(
            f"Candidates kept: {kept_idx.size} / {qv.size} | "
            f"q_min={q_thr:.3g}, u0_min={u_thr:.3g}, nuc_prob_min={np_thr:.3g}"
        )
        fig.canvas.draw_idle()

        # If we have something kept and no ROI yet, pick the first
        if kept_idx.size and (fig_roi.number in plt.get_fignums()):
            _update_roi(int(kept_idx[0]))

    def _reset(_event=None) -> None:
        s_q.reset()
        s_u.reset()
        s_np.reset()

    s_q.on_changed(_update)
    s_u.on_changed(_update)
    s_np.on_changed(_update)
    b_reset.on_clicked(_reset)

    # Pick handling (click a spot in either main axis)
    def _on_pick(event) -> None:
        # event.ind are indices into the *scatter offsets* array, i.e. into kept_idx
        if kept_idx.size == 0:
            return
        if not hasattr(event, "ind") or event.ind is None or len(event.ind) == 0:
            return
        j = int(event.ind[0])
        if 0 <= j < kept_idx.size:
            _update_roi(int(kept_idx[j]))

    sc0.set_picker(5)  # 5 px tolerance
    sc1.set_picker(5)
    fig.canvas.mpl_connect("pick_event", _on_pick)

    # Initial draw
    _update()

    plt.show()

    return InteractiveSpotQCResult(
        nuclei_labels=nuclei_labels,
        nuclei_meta=nuclei_meta,
        spots_df_all=spots_df_all,
        spot_debug=dbg,
    )

