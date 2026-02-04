from __future__ import annotations

import matplotlib

# Force a non-interactive backend for test environments.
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

from src.laser_system_diagrams import (
    LaserChannel,
    draw_multimode_fiber_system_diagram,
    draw_single_mode_fiber_system_diagram,
)


def test_diagram_functions_smoke() -> None:
    lasers = [
        LaserChannel(label="640 nm\n500 mW\nSM FC/APC", status="need_now"),
        LaserChannel(label="488 nm\n50 mW\nSM FC/APC", status="need_now"),
        LaserChannel(label="561 nm\n200 mW", status="future"),
    ]

    fig_a = draw_single_mode_fiber_system_diagram(lasers)
    fig_b = draw_multimode_fiber_system_diagram(lasers)

    # Basic sanity: we got figures and they have at least one axes.
    assert fig_a is not None
    assert fig_b is not None
    assert len(fig_a.axes) == 1
    assert len(fig_b.axes) == 1

    # Cleanup to avoid memory leaks in long test runs.
    plt.close(fig_a)
    plt.close(fig_b)
