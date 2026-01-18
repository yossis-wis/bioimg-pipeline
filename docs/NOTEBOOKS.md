# Notebooks

This repo stores notebooks as **Jupytext percent-format `.py` files**. Keeping notebooks in plain Python:

- produces clean diffs and code reviews
- keeps notebooks runnable in VS Code or JupyterLab
- avoids large binary `.ipynb` blobs in git

The canonical notebooks live in `notebooks/`:

- `_TEMPLATE__jupytext_percent.py` — copy this to start new analyses
- `01_step_by_step_integrated_qc.py` — load a single plane, segment nuclei, detect spots, and visualize
- `02_review_run_folder.py` — review a run folder (manifests, spots table, QC images)

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

## Exporting HTML/PDF

One recommended workflow (with execution):

1. Convert to `.ipynb` (temporary artifact):
   ```bash
   jupytext --to ipynb notebooks/01_step_by_step_integrated_qc.py
   ```
2. Create the reports output directory:
   ```bash
   mkdir -p reports
   ```
3. Execute and export HTML (or PDF) with `nbconvert` (writes to `reports/`):
   ```bash
   jupyter nbconvert --execute --to html --output-dir reports notebooks/01_step_by_step_integrated_qc.ipynb
   ```

The generated `.ipynb` files are intentionally ignored by git.
