# Notebooks

This repo stores notebooks as **Jupytext percent-format `.py` files**. Keeping notebooks in plain Python:

- produces clean diffs and code reviews
- keeps notebooks runnable in VS Code or JupyterLab
- avoids large binary `.ipynb` blobs in git

The canonical notebooks live in `notebooks/`:

- `_TEMPLATE__jupytext_percent.py` — copy this to start new analyses
- `01_step_by_step_integrated_qc.py` — load a single plane, segment nuclei, detect spots, and visualize (prints TrackMate-equivalent blob diameter)
- `02_review_run_folder.py` — review a run folder (manifests, spots table, QC images)
- `03_generate_batch_spot_atlas_qc.py` — generate a MATLAB-style **spot atlas** PowerPoint for batch QC
- `04_babysit_spot_detection.py` — "babysit" Slice0 spot detection step-by-step (TrackMate-style LoG, nonmax, u0 mask/threshold, nucleus assignment; prints TrackMate-equivalent blob diameter)
- `05_excitation_speckle_fpfn_proxy.py`: simulate excitation (speckle + field stop) and quantify Slice0 TP/FP/FN risk.
- `06_mmf_illumination_500us_design.py`: 500 µs design sweeps for MMF widefield illumination (power budget, stop sizing, pupil fill, and speckle-averaging strategies).
- `07_linewidth_speckle_mechanism_500us.py`: visualize how spectral linewidth reduces speckle at 500 µs (inner ROI vs edge metrics + Fourier-optics mechanism).
- `08_cni_laser_system_diagrams.py`: generate block diagrams + quote-request checklists for two laser delivery concepts (single-mode fibers vs common multimode fiber) to communicate with CNI Laser.
- `09_mmf_wide_linewidth_scrambling_fourier_optics.py`: step-by-step Fourier-optics + fiber-dispersion analysis answering common objections to “MMF + wide-linewidth + scrambler” as a homogeneous illuminator.
- `10_mmf_robust_setup_linewidth_stepindex_kohler.py`: practical design Q&A for a robust MMF illuminator (2 nm vs 20 nm, OPL from vendor specs, step-index vs graded-index, Köhler-like relay, and failure modes).
- `11_fiber_modes_speckle_interactive_3d.py`: interactive 3D intuition builder for fiber modes + MMF speckle (mode shapes, speckle drift, and averaging).
- `12_mmf_wide_linewidth_stepindex_slab_geometric_optics.py`: **multimode-only** wide-linewidth argument using a **slab + geometrical-optics** picture, with toy examples showing (1) mode superposition → speckle, (2) mode-weight effects, and (3) spectral diversity → speckle averaging.
- `13_cni_2nm_stepindex_spectral_diversity_500us.py`: evaluate a **CNI FC-640-class** (2–3 nm linewidth) laser + **3 m step-index MMF** for **500 µs** exposures, focusing on **spectral diversity only** and mapping spectral scenarios → (speckle contrast C, Slice0 confusion-matrix proxy).
- `14_stepindex_mmf_spectral_linewidth_physics.py`: derivations + intuition for **step-index MMF + spectral linewidth** (answers the margin math questions from notebook 13, adds coherence-time/length intuition from the whiteboard discussion, and includes a staff-scientist meeting checklist).
- `15_phasor_random_walk_toy_model.py`: toy model for **phasor sums as a 2D random walk** (endpoint statistics, intensity distribution, and averaging), plus a bridge to the slide-type instantaneous phasor-sum script.

## Running notebooks

### VS Code

1. Install the Jupyter extension.
2. Open any `notebooks/*.py` file (percent format).
3. Use the **Run Cell** / **Run All** commands.

### JupyterLab

1. Launch JupyterLab and open a percent-format `.py` file.
2. If Jupytext is missing, install it once in your environment:
   `pip install jupytext`.
   - Note: `environment.yml` already includes Jupytext.
3. Jupytext will render the `.py` file as a notebook with cells.



## Math in notebook Markdown cells

When writing Markdown cells in `notebooks/*.py` (Jupytext percent format), use the notebook-safe math delimiters so equations render in both **JupyterLab** and **VS Code Jupyter**:

- Inline: `$...$`
- Display: `$$ ... $$` (use `$$` on their own lines for multi-line)
- Multi-line: use `\begin{aligned}...\end{aligned}` inside `$$ ... $$`

Do **not** use:

- `\(...\)` / `\[...\]`
- fenced ```math blocks
- `align` / `equation` environments (use `aligned`)

Canonical reference: `docs/MATH_STYLE.md`.

## GitHub-friendly Markdown mirrors (stable reading)

GitHub does **not** render the Markdown cells inside Jupytext percent-format `notebooks/*.py` files.
For the documentation-style optics notebooks (06–15), we keep a **generated** Markdown mirror (no cell outputs)
in:

- `docs/notebooks_md/`

Regenerate the mirrors after editing a notebook:

```bash
python scripts/export_notebooks_markdown.py
```

CI/local verification (no writes):

```bash
python scripts/export_notebooks_markdown.py --check
```

These mirrors are intended for **stable, cross-platform reading** (including math) in the GitHub web UI.

## Exporting to `.ipynb` and HTML (local-only)

This repo intentionally does **not** commit `.ipynb` files (they are ignored in `.gitignore`).
But sometimes you want an `.ipynb` for sharing, or an HTML export for printing.

Convert a percent-format notebook to an `.ipynb` (writes next to the `.py`):

```bash
# Single notebook
jupytext --to ipynb notebooks/13_cni_2nm_stepindex_spectral_diversity_500us.py

# All notebooks (writes adjacent .ipynb files)
jupytext --to ipynb notebooks/*.py
```

Convert that `.ipynb` to HTML:

```bash
jupyter nbconvert --to html notebooks/13_cni_2nm_stepindex_spectral_diversity_500us.ipynb
```

If you want nbconvert to **execute** the notebook first (so the HTML includes figures), add `--execute`:

```bash
jupyter nbconvert --to html --execute notebooks/13_cni_2nm_stepindex_spectral_diversity_500us.ipynb
```

Tip: keep exports out of git by putting them under a build folder:

```bash
mkdir -p notebooks/_build/html
jupyter nbconvert --to html --execute \
  --output-dir notebooks/_build/html \
  notebooks/13_cni_2nm_stepindex_spectral_diversity_500us.ipynb
```

