# Math style and delimiters

This repo needs LaTeX math to render correctly in **two different surfaces**:

1) **GitHub-rendered Markdown** (`*.md`) in the browser.
2) **Notebook Markdown cells** in **JupyterLab** *and* **VS Code Jupyter** (`notebooks/*.py` in Jupytext percent format).

The syntax that works well on GitHub is **not** identical to what works reliably inside notebook renderers.  
This document is the **canonical contract** for how to write math in this repo.

---

## 1) GitHub-rendered Markdown (`*.md`)

### Inline math

GitHub's Markdown parser can be fragile when inline math contains characters that overlap with Markdown
syntax (most commonly **subscripts** with `_`, but also `*`, `|`, etc.), especially inside **lists**, **tables**,
and **headings**.

In this repo:

- Use plain `$...$` only for **simple** expressions that do **not** contain `_` (e.g. `$M^2$`, `$p$`).
- For anything with subscripts / Markdown-sensitive characters, treat the safer GitHub-only form as **required**:

  - `$`\`...\`$` (math delimiters around a code span)

Examples:

- `$`\`\mathrm{NA}_{\mathrm{illum}}\`$`
- `$`\`\Delta x_{\mathrm{speckle}} \approx \lambda/(2\thinspace\mathrm{NA}_{\mathrm{illum}})\`$`

**Note on spacing macros:** `\thinspace` is a control word. If the next token is a letter/digit, you must
separate it (e.g. write `\thinspace p`, not `\thinspacep`).

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
- **Units** in roman with a thin space: `10\thinspace\mu\mathrm{m}`, `500\thinspace\mu\mathrm{s}`, `30\thinspace\mathrm{kW}/\mathrm{cm}^2`.
- **Markdown note**: In `.md`, prefer `\thinspace` (word macro) for thin spaces in math; some Markdown parsers can treat the backslash-comma thinspace macro as an escaped comma.
- **Named quantities / labels** in roman: `\mathrm{NA}`, `\mathrm{PSF}`, `\mathrm{SNR}`, `\mathrm{LoG}`.
- Prefer `\exp(\cdot)` and roman differential `\mathrm{d}x` when relevant.
- Vectors in bold: `\mathbf{r}`.

---

## 4) Checklist for humans and LLMs

When editing or creating content:

- If the file is `*.md`: use `$...$` and fenced ```math blocks.
- If the file is under `notebooks/` and is Jupytext percent-format: use `$...$` and `$$...$$` (with `aligned` for multi-line).
- Run `python scripts/verify_math_syntax.py` before committing.
