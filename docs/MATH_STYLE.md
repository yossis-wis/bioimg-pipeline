# Math style and delimiters

This repo needs LaTeX math to render correctly in **two different surfaces**:

1) **GitHub-rendered Markdown** (`*.md`) in the browser.
2) **Notebook Markdown cells** in **JupyterLab** *and* **VS Code Jupyter** (`notebooks/*.py` in Jupytext percent format).

The syntax that works well on GitHub is **not** identical to what works reliably inside notebook renderers.  
This document is the **canonical contract** for how to write math in this repo.

---

## 1) GitHub-rendered Markdown (`*.md`)

### Inline math

Use:

- `$...$`

If Markdown punctuation makes inline math fragile (underscores, asterisks, etc.), you can use the safer GitHub pattern:

- `$`\`...\`$`  (math delimiters around a code span)

### Display (block) math

Prefer fenced math blocks:

```math
E = mc^2
```

`$$...$$` also works on GitHub, but keep the `$$` on **their own lines** to avoid Markdown edge cases.

### Never use in `.md`

- `\(...\)` or `\[...\]` (GitHub will show backslashes literally)

---

## 2) Notebook Markdown cells (`notebooks/*.py` via Jupytext)

Notebook math must render in both **JupyterLab** and **VS Code Jupyter**.  
VS Code is pickier about delimiters, so notebooks use the **intersection** syntax:

### Inline math

Use:

- `$...$`

### Display (block) math

Use:

- `$$ ... $$`

Put `$$` on **their own lines** for multi-line blocks:

```text
$$
\sigma_0 = \frac{w_0/\sqrt{2}}{p}
$$
```

### Multi-line equations

Do **not** use `align` / `equation` environments in notebooks.  
Instead:

```text
$$
\begin{aligned}
b &= \mathrm{median}\{ I(\mathbf{r}) : \mathbf{r}\in\mathrm{out0}\},\\
\langle I\rangle_{\mathrm{in5}} &= \frac{1}{|\mathrm{in5}|}\sum_{\mathbf{r}\in\mathrm{in5}} I(\mathbf{r}),\\
u_0 &= \langle I\rangle_{\mathrm{in5}} - b.
\end{aligned}
$$
```

### Never use in notebooks

- `\(...\)` or `\[...\]`
- fenced ```math blocks (they render as code blocks in notebooks)
- `\begin{equation}...\end{equation}`
- `\begin{align}...\end{align}` (use `aligned` instead)
- tab characters inside math (can corrupt `\text{...}`)

---

## 3) Optics-journal typography conventions

Use the conventions typical of optics journals (e.g. *Optics Express*, *Applied Optics*):

- **Variables** italic by default (math mode default).
- **Units** in roman with a thin space: `10\,\mu\mathrm{m}`, `500\,\mu\mathrm{s}`, `30\,\mathrm{kW}/\mathrm{cm}^2`.
- **GitHub note**: In `.md` math, use `\,` for thin spaces. Avoid `\thinspace` (GitHub renders it as an unknown macro).
- **Named quantities / labels** in roman: `\mathrm{NA}`, `\mathrm{PSF}`, `\mathrm{SNR}`, `\mathrm{LoG}`.
- Prefer `\exp(\cdot)` and roman differential `\mathrm{d}x` when relevant.
- Vectors in bold: `\mathbf{r}`.

---

## 4) Checklist for humans and LLMs

When editing or creating content:

- If the file is `*.md`: use `$...$` and fenced ```math blocks.
- If the file is under `notebooks/` and is Jupytext percent-format: use `$...$` and `$$...$$` (with `aligned` for multi-line).
- Run `python scripts/verify_math_syntax.py` before committing.
