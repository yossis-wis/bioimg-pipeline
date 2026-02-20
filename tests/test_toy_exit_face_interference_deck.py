from __future__ import annotations

from pathlib import Path

import matplotlib

# Force a non-interactive backend for test environments.
matplotlib.use("Agg", force=True)


def test_generate_toy_exit_face_interference_deck_smoke(tmp_path: Path) -> None:
    from src.toy_exit_face_interference_deck import (
        ToyExitFaceInterferenceDeckConfig,
        generate_toy_exit_face_interference_deck,
    )

    repo_root = Path(__file__).resolve().parents[1]
    svg = repo_root / "docs" / "figures" / "speckle_toy_exit_face_interference.svg"
    assert svg.exists()

    out = tmp_path / "toy_exit_face_interference.pptx"

    cfg = ToyExitFaceInterferenceDeckConfig(
        svg_figure_path=svg,
        out_pptx_path=out,
        svg_raster_width_px=1600,
        n_components=(1, 10),
    )

    written = generate_toy_exit_face_interference_deck(config=cfg)
    assert written.exists()
    assert written.stat().st_size > 10_000
