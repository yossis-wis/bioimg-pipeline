# Generated figures (untracked)

This folder is for **large** or **publication-grade** figures that should **not** be committed.

Why:

- The repo is intentionally kept **small and prompt-attachable** (often flattened for LLM context).
- Some figure formats (especially Matplotlib SVGs with many elements) can bloat the repo.

Recommended workflow:

1) Keep the **generating code** (script/notebook) committed.
2) Keep a small “preview” figure committed under `docs/figures/` if helpful.
3) Write big outputs here (or to `$BIOIMG_DATA_ROOT/reports/`).

Nothing in this directory is tracked except this README and `.gitignore`.
