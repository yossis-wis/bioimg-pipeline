# Figures style guide

This repo aims for figures that are readable **in Jupyter, exported HTML/PDF, and on GitHub**.

## Quick consistency checklist

- **Canvas / margins**
  - Prefer SVG.
  - Default size: **~900 px wide** (height as needed).
  - Keep a comfortable outer margin (**≥ 30 px**) so nothing touches the edges.

- **Fonts**
  - Use **sans-serif** everywhere.
  - Suggested sizes (SVG):  
    - `label`: **16 px** (titles, section labels)  
    - `eq`: **15 px** (equations)  
    - `small`: **13 px** (annotations, axis labels)
  - Avoid long lines of tiny text. If it gets dense, split into multiple figures.

- **Line / marker styles**
  - Axes: solid, **2 px**.
  - Curves/paths: **2–3 px**.
  - Reference/guide lines: dashed, lower opacity (e.g. `opacity="0.3"`).

- **Axis labels + units**
  - Label every axis that appears.
  - Include units in brackets, e.g. **`time [ps]`**, **`x [µm]`**, **`phase (rad)`**, **`amplitude [a.u.]`**.
  - If units are not meaningful, use **`[a.u.]`** (arbitrary units).

- **Math notation**
  - Use consistent subscripts: `n_{\mathrm{core}}`, `n_{\mathrm{clad}}`, `\theta_{\max}`.
  - Prefer explicit parentheses for trig: `\sin(\theta)`, `\cos(\theta)`.

- **Clutter control**
  - Do not let text overlap shapes/curves.
  - If a figure needs > ~6 annotations, split it into smaller “Picture 1 / 2 / 3 …” steps.

- **Filenames**
  - Use descriptive, stable names (snake_case).
  - Prefer prefixing with the concept/section, e.g. `k_wavenumber_...`, `phasor_...`, `modal_dispersion_...`.



## Repo size guardrails (important)

This repo is often **flattened and attached to LLM prompts**, so figure files must remain **small**.

- **Preferred:** SVG figures under **~150 KB**.
- **Hard limit:** keep committed figure assets under **~250 KB** unless there is a very strong reason.

### Avoid “exploded” SVGs from Matplotlib

Some Matplotlib artists (notably `pcolormesh` and large `scatter`) can emit **one SVG element per cell/point**.
Even modest grids (e.g. 80×160) can become multi‑MB SVGs.

Preferred patterns:

- **Heatmaps:** use `imshow(...)` (embeds a compact raster inside the SVG) rather than `pcolormesh(...)`.
- **Large scatters:** keep point counts modest *or* set `rasterized=True` on the scatter artist so the SVG stays compact.

### Where to put large / publication outputs

If you need high‑resolution or publication‑grade figures:

- Write them to `docs/figures/generated/` (gitignored) or to the data bench under `$BIOIMG_DATA_ROOT/reports/`.
- Commit the **script/notebook** that generates them, plus (optionally) a small “preview” SVG that is readable on GitHub.

See also: `scripts/check_repo_bloat.py`.
