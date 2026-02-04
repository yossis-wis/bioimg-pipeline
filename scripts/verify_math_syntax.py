#!/usr/bin/env python3
"""verify_math_syntax.py

Repository math hygiene checks.

Why this exists
---------------
We render math in two places:

1) Jupytext notebooks (`notebooks/*.py`) in Jupyter/VS Code.
2) GitHub Markdown (`*.md`) in the GitHub web UI.

These renderers differ. This script enforces repo-wide rules that keep
math readable and renderable across both.

Checks performed
----------------
- No ASCII control characters (except \n and \r) in scanned files
- No hard tabs (use spaces)
- Notebooks:
  - forbid fenced ```math blocks (GitHub-only feature)
  - forbid LaTeX delimiters \(...\) and \[...\] (GitHub shows backslashes literally)
- Markdown:
  - forbid the TeX thin-space macro written as backslash-comma (`\,`) in math (some Markdown parsers
    treat it as an escaped comma). Prefer the word macro `\thinspace` instead.
  - forbid `\thinspace` accidentally glued to the next symbol (e.g. `\thinspacep`); add a space.
  - for inline math that contains `_` (subscripts), require the safer GitHub form: `$`\`...\`$`.

The Markdown check ignores inline code spans and non-math fenced code blocks.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "data",
    "data_bench",
    "reports",
    ".virtual_documents",
}

INLINE_CODE_RE = re.compile(r"`[^`]*`")  # simple + sufficient for this repo
FENCE_OPEN_RE = re.compile(r"^\s*(```+)\s*([A-Za-z0-9_-]+)?\s*$")
FENCE_CLOSE_RE_TEMPLATE = r"^\s*{delim}\s*$"

# GitHub-safe inline math uses math delimiters around an inline code span: $`...`$
SAFE_GH_INLINE_MATH_RE = re.compile(r"\$`[^`]*`\$")
# Plain $...$ inline math containing '_' is fragile on GitHub; require the safe form instead.
UNSAFE_INLINE_MATH_UNDERSCORE_RE = re.compile(r"(?<!\$)\$(?!\$|`)(?P<body>[^$]*?_[^$]*?)\$(?!\$)")
# TeX control word: \thinspace must be separated from following letters/digits (avoid '\thinspacep').
THINSPACE_GLUE_RE = re.compile(r"\\thinspace(?=[0-9A-Za-z])")



def is_excluded(path: Path) -> bool:
    return any(part in EXCLUDE_DIRS for part in path.parts)


def iter_markdown_files() -> list[Path]:
    return sorted(p for p in REPO_ROOT.rglob("*.md") if not is_excluded(p))


def iter_notebook_files() -> list[Path]:
    nb_dir = REPO_ROOT / "notebooks"
    if not nb_dir.exists():
        return []
    return sorted(p for p in nb_dir.glob("*.py") if p.is_file())


def find_control_chars(text: str) -> list[tuple[int, str]]:
    """Return (index, char) for ASCII control characters excluding \n and \r."""
    bad: list[tuple[int, str]] = []
    for i, ch in enumerate(text):
        o = ord(ch)
        if o < 0x20 and ch not in ("\n", "\r"):
            bad.append((i, ch))
    return bad


def check_no_control_chars(path: Path, text: str, errors: list[str]) -> None:
    bad = find_control_chars(text)
    if not bad:
        return

    # Show first few occurrences with line/col.
    for idx, ch in bad[:10]:
        line_no = text.count("\n", 0, idx) + 1
        col_no = idx - (text.rfind("\n", 0, idx) + 1) + 1
        errors.append(
            f"{path.as_posix()}:{line_no}:{col_no}: control char U+{ord(ch):04X}"
        )

    remaining = len(bad) - 10
    if remaining > 0:
        errors.append(f"{path.as_posix()}: ... and {remaining} more control chars")


def check_no_tabs(path: Path, text: str, errors: list[str]) -> None:
    if "\t" in text:
        # show first few
        for m in re.finditer(r"\t", text):
            idx = m.start()
            line_no = text.count("\n", 0, idx) + 1
            col_no = idx - (text.rfind("\n", 0, idx) + 1) + 1
            errors.append(f"{path.as_posix()}:{line_no}:{col_no}: tab character")
            if len(errors) >= 10:
                break


def check_notebook_math_syntax(path: Path, text: str, errors: list[str]) -> None:
    forbidden = [
        ("```math", "Use $...$ / $$...$$ in notebooks; fenced ```math is GitHub-only."),
        ("\\(", "Do not use \\( ... \\) in this repo; GitHub shows backslashes literally."),
        ("\\)", "Do not use \\( ... \\) in this repo; GitHub shows backslashes literally."),
        ("\\[", "Do not use \\[ ... \\] in this repo; GitHub shows backslashes literally."),
        ("\\]", "Do not use \\[ ... \\] in this repo; GitHub shows backslashes literally."),
    ]
    for token, msg in forbidden:
        if token in text:
            errors.append(f"{path.as_posix()}: forbidden notebook token '{token}'. {msg}")


def check_markdown_no_backslash_comma(markdown_path: Path) -> list[str]:
    """Markdown math hygiene checks.

    Enforces GitHub-facing rules that avoid known Markdown edge cases:

    - Prefer `\thinspace` over `\,` (some Markdown parsers treat backslash-comma as an escaped comma).
    - Ensure `\thinspace` is not accidentally glued to the next symbol (e.g. `\thinspacep`).
    - For inline math that contains `_` (subscripts), require the safer GitHub form: `$`\`...\`$`.

    This check ignores inline code spans and skips non-math fenced code blocks.
    """

    text = markdown_path.read_text(encoding="utf-8", errors="replace")
    errors: list[str] = []

    in_fence = False
    fence_delim = ""
    fence_lang = ""

    for lineno, raw in enumerate(text.splitlines(), start=1):
        if not in_fence:
            m_open = FENCE_OPEN_RE.match(raw)
            if m_open:
                in_fence = True
                fence_delim = m_open.group(1)
                fence_lang = (m_open.group(2) or "").lower()
                continue
        else:
            close_re = re.compile(FENCE_CLOSE_RE_TEMPLATE.format(delim=re.escape(fence_delim)))
            if close_re.match(raw):
                in_fence = False
                fence_delim = ""
                fence_lang = ""
                continue
            if fence_lang and fence_lang != "math":
                # Skip content in non-math fenced blocks
                continue

        # Checks that should ignore inline code spans (including GitHub's $`...`$ form).
        line_no_code = INLINE_CODE_RE.sub("", raw)

        if "\," in line_no_code:
            errors.append(
                f"{markdown_path.as_posix()}:{lineno}: found '\,' in Markdown. Prefer '\thinspace'."
            )

        if THINSPACE_GLUE_RE.search(line_no_code):
            errors.append(
                f"{markdown_path.as_posix()}:{lineno}: found '\thinspace' immediately followed by a letter/digit. "
                "Write '\thinspace p' (or '\thinspace{}p'), not '\thinspacep'."
            )

        # Inline-math robustness: outside fenced blocks, require the safe GitHub form for subscripts.
        if not in_fence:
            masked = SAFE_GH_INLINE_MATH_RE.sub("", raw)
            masked = INLINE_CODE_RE.sub("", masked)
            if UNSAFE_INLINE_MATH_UNDERSCORE_RE.search(masked):
                errors.append(
                    f"{markdown_path.as_posix()}:{lineno}: inline math contains '_' (subscript) but is not in the "
                    "GitHub-safe $`...`$ form. Use $`...`$ for inline math with subscripts."
                )

    return errors

def main() -> int:
    errors: list[str] = []

    # Notebooks
    for path in iter_notebook_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        check_no_control_chars(path, text, errors)
        check_no_tabs(path, text, errors)
        check_notebook_math_syntax(path, text, errors)

    # Markdown
    for path in iter_markdown_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        check_no_control_chars(path, text, errors)
        check_no_tabs(path, text, errors)
        check_markdown_no_backslash_comma(path, text, errors)

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1

    print("OK: math syntax checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
