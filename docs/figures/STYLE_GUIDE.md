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

