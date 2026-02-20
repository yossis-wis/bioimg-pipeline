"""Generate a slide-ready PowerPoint deck for the exit-face interference toy model.

This writes a `.pptx` (gitignored) that embeds:

- `docs/figures/speckle_toy_exit_face_interference.svg`
- a quantitative spectral-diversity slide

Example
-------

```bash
python scripts/generate_toy_exit_face_interference_deck.py \
  --out reports/toy_exit_face_interference.pptx
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a PowerPoint deck illustrating the exit-face interference toy model."
    )
    p.add_argument(
        "--out",
        type=str,
        default="reports/toy_exit_face_interference.pptx",
        help="Output .pptx path (gitignored).",
    )
    p.add_argument(
        "--svg",
        type=str,
        default="docs/figures/speckle_toy_exit_face_interference.svg",
        help="Input SVG figure to embed.",
    )
    p.add_argument(
        "--svg-width-px",
        type=int,
        default=2400,
        help="Rasterization width for the embedded SVG (higher = crisper in PPT).",
    )
    p.add_argument(
        "--n-components",
        type=str,
        default="1,5,20",
        help="Comma-separated list of N values for the spectral-diversity PDF slide.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    # Make `import src...` work even when the script is executed as
    # `python scripts/...` from arbitrary working directories.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.toy_exit_face_interference_deck import (  # noqa: WPS433
        ToyExitFaceInterferenceDeckConfig,
        generate_toy_exit_face_interference_deck,
    )

    n_list = tuple(int(x.strip()) for x in str(args.n_components).split(",") if x.strip())
    if not n_list:
        raise SystemExit("--n-components must contain at least one integer")

    svg_path = Path(args.svg)
    if not svg_path.is_absolute():
        svg_path = repo_root / svg_path

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path

    cfg = ToyExitFaceInterferenceDeckConfig(
        svg_figure_path=svg_path,
        out_pptx_path=out_path,
        svg_raster_width_px=int(args.svg_width_px),
        n_components=n_list,
    )

    written = generate_toy_exit_face_interference_deck(config=cfg)
    print(f"Wrote: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
