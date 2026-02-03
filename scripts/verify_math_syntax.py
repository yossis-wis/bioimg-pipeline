"""Verify math syntax conventions for this repo.

This is a lightweight regression guard to keep LaTeX math rendering consistent across:

- GitHub-rendered Markdown (.md files), and
- Notebook Markdown cells in JupyterLab + VS Code Jupyter (Jupytext percent-format notebooks/*.py).

Usage:
    python scripts/verify_math_syntax.py

Exit code:
    0  if all checks pass
    1  if any violations are found
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def iter_files(glob_pattern: str) -> list[Path]:
    return sorted(REPO_ROOT.glob(glob_pattern))


def find_control_chars(s: str) -> list[tuple[int, str]]:
    """Return (index, char) for ASCII control characters excluding newline and carriage return."""
    bad = []
    for i, ch in enumerate(s):
        o = ord(ch)
        if o < 0x20 and ch not in ("\n", "\r"):
            bad.append((i, ch))
    return bad


def main() -> int:
    violations: list[str] = []

    # --- Notebooks: notebook-safe math only
    notebook_paths = iter_files("notebooks/*.py")
    notebook_bad_substrings = [
        "```math",  # fenced math blocks break in notebooks
        "\\(",
        "\\)",
        "\\[",
        "\\]",
        "\\begin{equation}",
        "\\end{equation}",
        "\\begin{align}",
        "\\end{align}",
        "\t",  # tabs can corrupt LaTeX (e.g., \text)
    ]

    for path in notebook_paths:
        s = path.read_text(encoding="utf-8", errors="replace")
        for sub in notebook_bad_substrings:
            if sub in s:
                violations.append(f"{path}: forbidden substring: {sub!r}")

    # --- Markdown docs: no ASCII control chars, no tabs
    md_paths = iter_files("**/*.md")
    for path in md_paths:
        s = path.read_text(encoding="utf-8", errors="replace")
        if "\t" in s:
            violations.append(f"{path}: contains tab character(s)")
        bad = find_control_chars(s.replace("\t", ""))  # tabs handled separately above
        if bad:
            # show first few
            sample = ", ".join([f"0x{ord(ch):02x}" for _, ch in bad[:5]])
            violations.append(f"{path}: contains ASCII control character(s): {sample}")

    if violations:
        print("Math syntax verification FAILED:\n", file=sys.stderr)
        for v in violations:
            print(f"- {v}", file=sys.stderr)
        print("\nSee docs/MATH_STYLE.md for the canonical rules.", file=sys.stderr)
        return 1

    print("Math syntax verification OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
