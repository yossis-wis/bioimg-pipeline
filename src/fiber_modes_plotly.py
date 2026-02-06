"""Plotly helpers for interactive 3D fiber-mode / speckle figures.

These helpers intentionally keep Plotly usage lightweight:

- no Dash dependency
- writes standalone HTML via :func:`plotly.graph_objects.Figure.write_html`
- slider-driven frames for "choose mode / choose averaging level" interactions

The notebook `notebooks/11_fiber_modes_speckle_interactive_3d.py` is the primary consumer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SurfaceStack:
    """A stack of 2D surfaces (z) sharing the same x/y grid."""
    x_um_1d: np.ndarray  # shape (nx,)
    y_um_1d: np.ndarray  # shape (ny,)
    z_stack: np.ndarray  # shape (n_frames, ny, nx)
    labels: list[str]


def _require_plotly():
    try:
        import plotly.graph_objects as go  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Plotly is required for interactive 3D figures. "
            "Install via conda: `conda env update -f environment.yml --prune` "
            "or pip: `pip install plotly`."
        ) from exc


def make_surface_stack(
    *,
    x_um: np.ndarray,
    y_um: np.ndarray,
    z_list: Sequence[np.ndarray],
    labels: Sequence[str],
) -> SurfaceStack:
    """Convert a list of (ny,nx) arrays into a stack suitable for sliders."""
    if x_um.shape != y_um.shape:
        raise ValueError("x_um and y_um must have the same shape")
    if len(z_list) == 0:
        raise ValueError("z_list must be non-empty")
    if len(z_list) != len(labels):
        raise ValueError("labels must match z_list length")

    z0 = np.asarray(z_list[0])
    if z0.shape != x_um.shape:
        raise ValueError("z arrays must have the same shape as x_um/y_um")

    stack = np.stack([np.asarray(z) for z in z_list], axis=0)
    x1d = np.asarray(x_um[0, :], dtype=np.float64)
    y1d = np.asarray(y_um[:, 0], dtype=np.float64)
    return SurfaceStack(x_um_1d=x1d, y_um_1d=y1d, z_stack=stack, labels=[str(s) for s in labels])


def surface_stack_figure(
    surf: SurfaceStack,
    *,
    title: str,
    z_title: str,
    colorscale: str = "Viridis",
    show_colorbar: bool = False,
    z_range: tuple[float, float] | None = None,  # interpreted as (cmin, cmax) for color scaling
    aspectmode: str = "data",
) -> "object":
    """Build an interactive Plotly figure with a slider over the surface stack."""
    _require_plotly()
    import plotly.graph_objects as go

    if surf.z_stack.ndim != 3:
        raise ValueError("z_stack must have shape (n_frames, ny, nx)")
    n_frames = surf.z_stack.shape[0]
    if n_frames != len(surf.labels):
        raise ValueError("labels length must match z_stack[0]")

    z0 = surf.z_stack[0]

    surface_kwargs: dict = dict(
        x=surf.x_um_1d,
        y=surf.y_um_1d,
        z=z0,
        colorscale=colorscale,
        showscale=bool(show_colorbar),
    )
    if z_range is not None:
        zmin, zmax = z_range
        surface_kwargs.update(dict(cmin=float(zmin), cmax=float(zmax)))

    fig = go.Figure(data=[go.Surface(**surface_kwargs)])

    frames = []
    for i in range(n_frames):
        frames.append(go.Frame(name=str(i), data=[go.Surface(z=surf.z_stack[i])]))
    fig.frames = frames

    steps = []
    for i, label in enumerate(surf.labels):
        steps.append(
            dict(
                method="animate",
                args=[
                    [str(i)],
                    dict(
                        mode="immediate",
                        frame=dict(duration=0, redraw=True),
                        transition=dict(duration=0),
                    ),
                ],
                label=str(label),
            )
        )

    fig.update_layout(
        title=title,
        sliders=[
            dict(
                active=0,
                pad=dict(t=50),
                steps=steps,
                currentvalue=dict(prefix=""),
            )
        ],
        scene=dict(
            xaxis_title="x (µm)",
            yaxis_title="y (µm)",
            zaxis_title=str(z_title),
            aspectmode=str(aspectmode),
        ),
        margin=dict(l=0, r=0, b=0, t=60),
        height=720,
    )

    # Optional play/pause controls.
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.10,
                x=0.0,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, dict(frame=dict(duration=250, redraw=True), transition=dict(duration=0), fromcurrent=True)],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), transition=dict(duration=0), mode="immediate")],
                    ),
                ],
            )
        ]
    )

    return fig


def write_html(fig: "object", out_path: Path, *, auto_open: bool = False) -> None:
    """Write a Plotly figure to a standalone HTML file."""
    _require_plotly()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # `include_plotlyjs='cdn'` keeps the HTML small and is fine for lab use.
    fig.write_html(str(out_path), include_plotlyjs="cdn", auto_open=bool(auto_open))
