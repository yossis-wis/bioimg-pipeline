"""Generate simple block-diagram figures for laser delivery concepts.

The intent is *communication*, not optical CAD:

- Quickly convey topology (lasers -> combining -> fiber -> field shaping -> objective).
- Make "need now" vs "future" additions visually distinct.
- Produce figures that can be exported as PNG/PDF and pasted into an email to a vendor.

The notebook `notebooks/08_cni_laser_system_diagrams.py` is the primary consumer.

Design principles
-----------------
- No dependencies beyond matplotlib (already in the repo environment).
- Keep the coordinate system normalized (0..1) so layouts are deterministic.
- Avoid interactive requirements (works in headless CI and in notebooks).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

LaserStatus = Literal["need_now", "future"]


@dataclass(frozen=True)
class LaserChannel:
    """A single laser channel to be shown in a diagram."""

    label: str
    status: LaserStatus = "need_now"


def make_laser_label(
    *,
    wavelength_nm: float,
    power_label: str,
    extra: str | None = None,
) -> str:
    """Helper to format a multi-line laser label."""

    lines = [f"{wavelength_nm:.0f} nm", power_label]
    if extra:
        lines.append(str(extra))
    return "\n".join(lines)


def _setup_canvas(*, figsize: Tuple[float, float] = (14.0, 6.0)) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()
    return fig, ax


def _add_box(
    ax: Axes,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    dashed: bool = False,
    fontsize: float = 9.0,
) -> FancyBboxPatch:
    """Draw a rounded rectangle with centered text."""

    linestyle = "--" if dashed else "-"
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="black",
        facecolor="white",
        linestyle=linestyle,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=float(fontsize),
        wrap=True,
    )
    return patch


def _add_arrow(
    ax: Axes,
    *,
    start: Tuple[float, float],
    end: Tuple[float, float],
    text: str | None = None,
    text_offset: Tuple[float, float] = (0.0, 0.0),
    fontsize: float = 8.0,
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "->",
            "lw": 1.2,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )

    if text:
        xm = 0.5 * (start[0] + end[0]) + float(text_offset[0])
        ym = 0.5 * (start[1] + end[1]) + float(text_offset[1])
        ax.text(xm, ym, text, ha="center", va="center", fontsize=float(fontsize))


def _y_positions(n: int, *, top: float = 0.80, bottom: float = 0.35) -> list[float]:
    """Evenly spaced y-centers for row blocks.

    Defaults leave bottom space for legend + explanatory boxes.
    """

    if n <= 0:
        return []
    if n == 1:
        return [(top + bottom) / 2]
    step = (top - bottom) / (n - 1)
    return [top - i * step for i in range(n)]


def _legend(ax: Axes, *, x: float = 0.02, y: float = 0.04) -> None:
    """Add a small legend explaining solid vs dashed boxes."""

    _add_box(ax, x=x, y=y, w=0.16, h=0.06, text="Need now", dashed=False, fontsize=8.5)
    _add_box(
        ax,
        x=x + 0.18,
        y=y,
        w=0.22,
        h=0.06,
        text="Future / user-upgradable",
        dashed=True,
        fontsize=8.5,
    )


def _bottom_notes_common(
    ax: Axes,
    *,
    y: float = 0.11,
    h: float = 0.18,
    need_now_text: str,
    future_text: str,
    field_text: str,
) -> None:
    """Shared bottom-note layout (3 boxes) used by both diagrams."""

    _add_box(ax, x=0.02, y=y, w=0.30, h=h, text=need_now_text, dashed=False, fontsize=9.0)
    _add_box(ax, x=0.35, y=y, w=0.30, h=h, text=future_text, dashed=True, fontsize=9.0)
    _add_box(ax, x=0.68, y=y, w=0.30, h=h, text=field_text, dashed=False, fontsize=9.0)


def draw_single_mode_fiber_system_diagram(
    lasers: Sequence[LaserChannel],
    *,
    title: str = "A) Single-mode fiber approach (separate SM fibers, free-space combine)",
) -> Figure:
    """Diagram for the single-mode fiber concept.

    Parameters
    ----------
    lasers:
        Laser channels (each drawn as a row). Use ``status='future'`` to draw dashed boxes.
    title:
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """

    fig, ax = _setup_canvas(figsize=(15.5, 6.4))
    ax.set_title(title, fontsize=13, pad=12)

    n = len(lasers)
    ys = _y_positions(n)

    # Column positions and sizes
    box_w = 0.13
    box_h = 0.11

    x_laser = 0.02
    x_aom = 0.17
    x_fiber = 0.32
    x_coll = 0.47
    x_comb = 0.62
    x_relay = 0.79

    # Per-channel blocks
    aom_text = "Optional\nAOM\n(+ driver)"
    fiber_text = "SM fiber\nFC/APC"
    coll_text = "Fiber\ncollimator"

    for y_center, ch in zip(ys, lasers, strict=True):
        y0 = y_center - box_h / 2

        _add_box(ax, x=x_laser, y=y0, w=box_w, h=box_h, text=ch.label, dashed=(ch.status == "future"))
        _add_box(ax, x=x_aom, y=y0, w=box_w, h=box_h, text=aom_text, dashed=True, fontsize=8.7)
        _add_box(ax, x=x_fiber, y=y0, w=box_w, h=box_h, text=fiber_text, dashed=(ch.status == "future"))
        _add_box(ax, x=x_coll, y=y0, w=box_w, h=box_h, text=coll_text, dashed=(ch.status == "future"))

        # Arrows within row
        _add_arrow(ax, start=(x_laser + box_w, y_center), end=(x_aom, y_center))
        _add_arrow(ax, start=(x_aom + box_w, y_center), end=(x_fiber, y_center))
        _add_arrow(ax, start=(x_fiber + box_w, y_center), end=(x_coll, y_center))

    # Combiner block (common)
    comb_w = 0.14
    _add_box(
        ax,
        x=x_comb,
        y=0.41,
        w=comb_w,
        h=0.18,
        text="Dichroic /\nbeam combiner(s)",
        dashed=False,
        fontsize=9.5,
    )

    # Merge arrows from each collimator into combiner
    for y_center in ys:
        _add_arrow(ax, start=(x_coll + box_w, y_center), end=(x_comb, 0.50))

    # Relay / BFP / sample block
    relay_w = 0.19
    _add_box(
        ax,
        x=x_relay,
        y=0.38,
        w=relay_w,
        h=0.24,
        text=(
            "Relay / focus\n"
            "to objective BFP\n"
            "(sets Gaussian\n"
            "field size)\n\n"
            "Objective → sample"
        ),
        dashed=False,
        fontsize=9.0,
    )
    _add_arrow(ax, start=(x_comb + comb_w, 0.50), end=(x_relay, 0.50))

    # Bottom explanatory boxes
    _bottom_notes_common(
        ax,
        need_now_text=(
            "Need now control:\n"
            "• native diode TTL + analog\n"
            "• ~5 ms exposures (on/off +\n"
            "  power setpoint changes)"
        ),
        future_text=(
            "Future (user-upgradable):\n"
            "• add per-laser AOM(s) +\n"
            "  RF driver(s)\n"
            "• for ≤500 µs gating\n"
            "  (esp. 640)\n"
            "• DPSS 561 (if added)\n"
            "  likely needs an AOM"
        ),
        field_text=(
            "Field goal:\n"
            "• time-constant Gaussian\n"
            "• stable over exposure + days\n"
            "• focus to objective BFP\n"
            "  sets field size"
        ),
    )

    _legend(ax)
    fig.tight_layout()
    return fig


def draw_multimode_fiber_system_diagram(
    lasers: Sequence[LaserChannel],
    *,
    title: str = "B) Multimode fiber approach (wide-linewidth lasers -> common MMF)",
) -> Figure:
    """Diagram for the multimode fiber concept."""

    fig, ax = _setup_canvas(figsize=(15.5, 6.4))
    ax.set_title(title, fontsize=13, pad=12)

    n = len(lasers)
    ys = _y_positions(n)

    # Row blocks
    laser_w = 0.14
    aom_w = 0.12
    box_h = 0.11

    x_laser = 0.02
    x_aom = 0.18
    x_comb = 0.32
    x_couple = 0.47
    x_mmf = 0.62
    x_out = 0.77

    # Laser blocks (+ optional AOM column)
    for y_center, ch in zip(ys, lasers, strict=True):
        y0 = y_center - box_h / 2
        _add_box(ax, x=x_laser, y=y0, w=laser_w, h=box_h, text=ch.label, dashed=(ch.status == "future"))
        _add_box(ax, x=x_aom, y=y0, w=aom_w, h=box_h, text="Optional\nAOM", dashed=True, fontsize=8.8)

        _add_arrow(ax, start=(x_laser + laser_w, y_center), end=(x_aom, y_center))
        _add_arrow(ax, start=(x_aom + aom_w, y_center), end=(x_comb, 0.50))

    # Combiner
    comb_w = 0.13
    _add_box(
        ax,
        x=x_comb,
        y=0.41,
        w=comb_w,
        h=0.18,
        text="Dichroic /\nbeam combiner(s)",
        dashed=False,
        fontsize=9.5,
    )
    _add_arrow(ax, start=(x_comb + comb_w, 0.50), end=(x_couple, 0.50))

    # Coupling into MMF
    mid_w = 0.13
    _add_box(
        ax,
        x=x_couple,
        y=0.41,
        w=mid_w,
        h=0.18,
        text="Couple\ninto MMF\n(TBD connector)",
        dashed=False,
        fontsize=9.5,
    )
    _add_arrow(ax, start=(x_couple + mid_w, 0.50), end=(x_mmf, 0.50))

    # MMF + scrambler
    _add_box(
        ax,
        x=x_mmf,
        y=0.41,
        w=mid_w,
        h=0.18,
        text="MMF +\n~10 kHz\nscrambler",
        dashed=False,
        fontsize=9.5,
    )
    _add_arrow(ax, start=(x_mmf + mid_w, 0.50), end=(x_out, 0.50))

    # Output / field shaping
    out_w = 0.21
    _add_box(
        ax,
        x=x_out,
        y=0.34,
        w=out_w,
        h=0.32,
        text=(
            "MMF output\ncollimator\n\n"
            "Square field stop\n(image plane)\n\n"
            "Relay to objective BFP\n(underfill ~0.3)\n\n"
            "Objective → sample"
        ),
        dashed=False,
        fontsize=9.0,
    )

    # Bottom explanatory boxes
    _bottom_notes_common(
        ax,
        need_now_text=(
            "Need now control:\n"
            "• native diode TTL + analog\n"
            "• ~5 ms exposures (on/off +\n"
            "  power setpoint changes)"
        ),
        future_text=(
            "Future (user-upgradable):\n"
            "• add per-laser AOM(s) +\n"
            "  RF driver(s)\n"
            "• for ≤500 µs gating\n"
            "  (esp. 640)\n"
            "• DPSS 561 (if added)\n"
            "  likely needs an AOM"
        ),
        field_text=(
            "Field goal:\n"
            "• time-constant field over\n"
            "  exposure + stable day-to-day\n"
            "• wide linewidth + scrambling\n"
            "  to reduce speckle\n"
            "• square stop defines ROI"
        ),
    )

    _legend(ax)
    fig.tight_layout()
    return fig


def to_laser_channels(rows: Iterable[dict]) -> list[LaserChannel]:
    """Convert a list of dicts (e.g. from YAML) to LaserChannel objects.

    Expected keys
    -------------
    label: str
    status: "need_now" | "future" (optional; defaults to "need_now")
    """

    out: list[LaserChannel] = []
    for r in rows:
        label = str(r.get("label", ""))
        if not label:
            raise ValueError("LaserChannel requires non-empty 'label'.")
        status = r.get("status", "need_now")
        if status not in ("need_now", "future"):
            raise ValueError(f"Invalid laser status: {status!r}")
        out.append(LaserChannel(label=label, status=status))
    return out
