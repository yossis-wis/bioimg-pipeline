# Notebook Markdown mirrors

This folder contains **generated** Markdown mirrors of the documentation-style optics notebooks.

Why this exists:

- The canonical notebooks live in `notebooks/*.py` (Jupytext percent format).
- GitHub does not render the Markdown cells inside those `.py` files.
- These mirrors make the narrative + equations readable in the GitHub web UI.

Regenerate:

```bash
python scripts/export_notebooks_markdown.py
```

Verify mirrors are up to date (no writes):

```bash
python scripts/export_notebooks_markdown.py --check
```

## Mirrors

- `06_mmf_illumination_500us_design.md`
- `07_linewidth_speckle_mechanism_500us.md`
- `08_cni_laser_system_diagrams.md`
- `09_mmf_wide_linewidth_scrambling_fourier_optics.md`
- `10_mmf_robust_setup_linewidth_stepindex_kohler.md`
- `11_fiber_modes_speckle_interactive_3d.md`
- `12_mmf_wide_linewidth_stepindex_slab_geometric_optics.md`
