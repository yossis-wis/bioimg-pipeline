"""Flatten committed repo text files into a single snapshot for LLM context.

This script generates an LLM-friendly *flat text* snapshot of the repository.

Key behavior (important)
------------------------
The snapshot section is generated from a **committed git ref** (default: HEAD),
so it is stable and deterministic even if you have local uncommitted changes.

Optionally (default: enabled), the snapshot also appends an **annotated unified
diff section** representing local changes in your working tree vs the base ref.
This is meant to be comparable to the "Copy patch" experience in GitHub/Codex.

If you need an *apply-able* patch file (for `git apply`), use `--patch-out` or
`--patch-only`. Those outputs contain **raw unified diff text only** (no headers
or prose), so they can be used with:

    git apply --check --reverse <patchfile>

Default output
--------------
Writes a timestamped snapshot into:

    repo_snapshots/repo_context__<timestamp>__<commit>.txt

Snapshots are intended to be **artifacts** (ignored by git), not a replacement
for git history. See `repo_snapshots/README.md` for the rationale and usage.
"""

from __future__ import annotations

import argparse
import difflib
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Tuple


# --- Defaults ---
DEFAULT_OUTPUT_DIR = "repo_snapshots"
DEFAULT_PREFIX = "repo_context"
DEFAULT_PATCH_PREFIX = "repo_patch"
DEFAULT_BASE_REF = "HEAD"
MAX_FILE_SIZE_KB_DEFAULT = 250  # Skip files larger than this (keep snapshots prompt-friendly)

# Even for tracked files, we explicitly skip file types that are commonly binary
# or huge when accidentally tracked.
BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".tif",
    ".tiff",
    ".pyc",
    ".iso",
    ".bin",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".pdf",
    ".parquet",
    ".zip",
    ".tar",
    ".gz",
    ".ipynb",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_env() -> dict[str, str]:
    """Return environment variables for non-interactive git calls."""
    env = dict(os.environ)
    # Avoid pagers in any shell.
    env.setdefault("GIT_PAGER", "cat")
    env.setdefault("PAGER", "cat")
    return env


def _to_text(value: str | bytes | None) -> str:
    """Coerce subprocess outputs to text safely.

    On Windows, `subprocess.run(..., text=True)` defaults to a legacy codepage
    (e.g. cp1252). Git output can contain bytes that are not decodable in that
    codepage, causing `UnicodeDecodeError` inside subprocess reader threads.

    We force UTF-8 decoding with replacement to keep this script robust.
    """
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _run_git(
    repo_root: Path,
    args: list[str],
    *,
    text: bool = True,
    ok_returncodes: tuple[int, ...] = (0,),
) -> subprocess.CompletedProcess:
    """Run a git command and return CompletedProcess.

    Parameters
    ----------
    ok_returncodes:
        Return codes that should be treated as success.
        Note: `git diff` returns 1 when differences are found.
    """
    try:
        kwargs: dict = {
            "cwd": str(repo_root),
            "env": _git_env(),
            "capture_output": True,
            "check": False,
            # Avoid any interactive prompts (credentials, etc.)
            "stdin": subprocess.DEVNULL,
        }
        if text:
            # Force UTF-8 decoding so we don't crash on Windows default codepages.
            kwargs.update({"text": True, "encoding": "utf-8", "errors": "replace"})
        else:
            kwargs.update({"text": False})

        proc = subprocess.run(["git", *args], **kwargs)
    except FileNotFoundError as exc:
        raise RuntimeError("git is not installed or not on PATH") from exc

    if proc.returncode not in ok_returncodes:
        stderr = _to_text(proc.stderr).strip()
        stdout = _to_text(proc.stdout).strip()
        msg = stderr or stdout or f"(no output; returncode={proc.returncode})"
        raise RuntimeError(f"git {' '.join(args)} failed: {msg}")

    return proc


def _try_git_rev_parse(repo_root: Path, ref: str, *, short: bool) -> Optional[str]:
    try:
        args = ["rev-parse"]
        if short:
            args.append("--short")
        args.append(str(ref))
        out = (_run_git(repo_root, args, text=True).stdout or "").strip()
        return out or None
    except Exception:
        return None


def _git_ls_tree_files(repo_root: Path, ref: str) -> list[str]:
    """List repo-relative files in a commit/tree (stable vs working tree)."""
    out = _run_git(
        repo_root,
        ["ls-tree", "-r", "--name-only", str(ref)],
        text=True,
    ).stdout or ""
    files = [line.strip() for line in out.splitlines() if line.strip()]
    return sorted(files)


def _git_cat_file_size(repo_root: Path, ref: str, rel_path: str) -> Optional[int]:
    spec = f"{ref}:{rel_path}"
    try:
        out = (_run_git(repo_root, ["cat-file", "-s", spec], text=True).stdout or "").strip()
        return int(out)
    except Exception:
        return None


def _git_show_bytes(repo_root: Path, ref: str, rel_path: str) -> Optional[bytes]:
    spec = f"{ref}:{rel_path}"
    try:
        proc = _run_git(repo_root, ["show", spec], text=False)
        return proc.stdout
    except Exception:
        return None


def _git_status_porcelain(repo_root: Path) -> Tuple[str, Optional[str]]:
    try:
        status = _run_git(repo_root, ["status", "--porcelain"], text=True).stdout or ""
        return status, None
    except Exception as exc:
        return "", str(exc)


def _git_diff_bytes(repo_root: Path, base_ref: str) -> Tuple[bytes, Optional[str]]:
    """Return raw diff bytes for tracked changes (base_ref -> working tree)."""
    try:
        proc = _run_git(
            repo_root,
            ["diff", "--no-color", "--no-ext-diff", "--patch", str(base_ref)],
            text=False,
            ok_returncodes=(0, 1),
        )
        out = proc.stdout or b""
        return out, None
    except Exception as exc:
        return b"", str(exc)


def _is_likely_text_path(rel_path: str) -> bool:
    return Path(rel_path).suffix.lower() not in BINARY_EXTENSIONS


def _is_text_blob(rel_path: str, blob: bytes) -> bool:
    """Heuristic: extension blocklist + null-byte check."""
    if not _is_likely_text_path(rel_path):
        return False
    return b"\0" not in blob[:1024]


def _is_text_worktree_file(path: Path) -> bool:
    """Heuristic: extension blocklist + null-byte check."""
    if path.suffix.lower() in BINARY_EXTENSIONS:
        return False
    try:
        with path.open("rb") as f:
            chunk = f.read(1024)
        return b"\0" not in chunk
    except Exception:
        return False


def _should_skip_relpath(rel_path: str) -> bool:
    """Skip snapshot/patch artifacts if they somehow become tracked/untracked."""
    rel = rel_path.replace("\\", "/")

    # If someone mistakenly tracks snapshot/patch artifacts, skip them to avoid runaway growth.
    if rel.startswith("repo_snapshots/") and (
        rel.endswith(".txt") or rel.endswith(".diff") or rel.endswith(".patch")
    ):
        return True

    # Historical default location (kept here defensively)
    if rel.startswith("reports/") and "repo_context" in rel and rel.endswith(".txt"):
        return True

    return False


def _generate_tree(files: list[str]) -> str:
    tree = ["Project Directory Structure:"]
    for f in files:
        tree.append(f"├── {f}")
    return "\n".join(tree)


def _generate_untracked_file_patch_text(
    repo_root: Path, rel_path: str, *, max_size_bytes: int
) -> str:
    """Generate a unified diff that adds an untracked text file.

    We intentionally generate this in pure Python (difflib) to avoid platform
    differences around `/dev/null` (especially on Windows).
    """
    full_path = repo_root / rel_path
    if not full_path.exists() or not full_path.is_file():
        return ""
    try:
        size = full_path.stat().st_size
    except Exception:
        return ""
    if size > max_size_bytes:
        return ""
    if not _is_text_worktree_file(full_path):
        return ""

    content = full_path.read_text(encoding="utf-8", errors="replace")
    new_lines = content.splitlines(keepends=False)

    diff_lines = list(
        difflib.unified_diff(
            [],
            new_lines,
            fromfile="/dev/null",
            tofile=f"b/{rel_path}",
            lineterm="",
        )
    )
    if not diff_lines:
        return ""

    # Add a git-style header so tooling recognizes it.
    header = [
        f"diff --git a/{rel_path} b/{rel_path}",
        "new file mode 100644",
    ]
    return "\n".join(header + diff_lines) + "\n"


def _iter_untracked_paths(repo_root: Path) -> Iterable[str]:
    """Yield untracked, non-ignored paths (repo-relative)."""
    out = _run_git(
        repo_root,
        ["ls-files", "--others", "--exclude-standard"],
        text=True,
    ).stdout or ""
    for line in out.splitlines():
        rel = line.strip()
        if rel:
            yield rel


def _ensure_trailing_newline_bytes(data: bytes) -> bytes:
    if not data:
        return data
    return data if data.endswith(b"\n") else (data + b"\n")


def _collect_local_patch(
    repo_root: Path,
    *,
    base_ref: str,
    include_untracked: bool,
    max_file_size_kb: int,
) -> Tuple[bytes, str]:
    """Return (raw_patch_bytes, annotated_patch_text)."""

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    base_commit = _try_git_rev_parse(repo_root, base_ref, short=True) or "(unavailable)"

    status, status_err = _git_status_porcelain(repo_root)
    diff_bytes, diff_err = _git_diff_bytes(repo_root, base_ref)

    untracked_patches: list[str] = []
    if include_untracked:
        max_size_bytes = int(max_file_size_kb) * 1024
        untracked = list(_iter_untracked_paths(repo_root))
        # Avoid including our own artifacts even if they are not ignored.
        untracked = [p for p in untracked if not _should_skip_relpath(p)]
        for rel in sorted(untracked):
            if not _is_likely_text_path(rel):
                continue
            patch = _generate_untracked_file_patch_text(repo_root, rel, max_size_bytes=max_size_bytes)
            if patch:
                untracked_patches.append(patch)

    # --- Raw patch bytes (apply-able) ---
    raw_parts: list[bytes] = []
    if diff_bytes.strip():
        raw_parts.append(_ensure_trailing_newline_bytes(diff_bytes))
    for p in untracked_patches:
        raw_parts.append(_ensure_trailing_newline_bytes(p.encode("utf-8", errors="replace")))
    raw_patch = b"".join(raw_parts)

    # --- Annotated patch text (LLM-friendly) ---
    lines: list[str] = []
    lines.append("=" * 40)
    lines.append("LOCAL CHANGES (PATCH)")
    lines.append(f"Generated: {stamp}")
    lines.append(f"Base ref: {base_ref}")
    lines.append(f"Base commit: {base_commit}")
    lines.append("=" * 40)
    lines.append("")

    if status_err:
        lines.append(f"(git status failed: {status_err})")
        lines.append("")

    status_clean = not status.strip()
    if status_clean:
        lines.append("Git status: clean (no local changes detected).")
        lines.append("")
    else:
        lines.append("Git status (porcelain):")
        lines.append(status.rstrip())
        lines.append("")

    if diff_err:
        lines.append(f"(git diff failed: {diff_err})")
        lines.append("")
        diff_text = ""
    else:
        diff_text = _to_text(diff_bytes).rstrip()

    if diff_text.strip():
        lines.append(f"git diff --patch {base_ref} (working tree vs base):")
        lines.append(diff_text)
        lines.append("")
    elif status_clean and not untracked_patches:
        lines.append("git diff: (empty)")
        lines.append("")
    else:
        lines.append("git diff: (empty for tracked files)")
        lines.append("")

    if untracked_patches:
        lines.append("Untracked files (text-only) included as new-file patches:")
        for p in untracked_patches:
            lines.append(p.rstrip())
        lines.append("")

    lines.append("END LOCAL CHANGES (PATCH)")
    lines.append("=" * 40)
    lines.append("")
    annotated = "\n".join(lines)

    return raw_patch, annotated


def _write_bytes(path: Path, data: bytes) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(data)


def flatten_repo(
    *,
    out_path: Path,
    max_file_size_kb: int = MAX_FILE_SIZE_KB_DEFAULT,
    base_ref: str = DEFAULT_BASE_REF,
    include_patch: bool = True,
    include_untracked: bool = True,
    patch_out_path: Optional[Path] = None,
) -> None:
    repo_root = _repo_root()
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute patch *before* writing the snapshot file to avoid the output
    # itself showing up as an untracked change (for custom --out locations).
    patch_raw = b""
    patch_annotated = ""
    if include_patch:
        patch_raw, patch_annotated = _collect_local_patch(
            repo_root,
            base_ref=base_ref,
            include_untracked=include_untracked,
            max_file_size_kb=int(max_file_size_kb),
        )
        if patch_out_path is not None:
            _write_bytes(patch_out_path, patch_raw)

    # Base commit metadata
    base_commit = _try_git_rev_parse(repo_root, base_ref, short=True) or "(unavailable)"

    # Stable file listing (from the commit tree)
    try:
        all_commit_files = _git_ls_tree_files(repo_root, base_ref)
    except Exception as exc:
        print(f"❌ Error: could not list files for ref={base_ref}: {exc}")
        return

    if not all_commit_files:
        print(f"No files found in ref={base_ref}. Are you in a git repo?")
        return

    processed_files: list[str] = []
    skipped_files: list[str] = []
    total_chars = 0

    print(f"🔍 Found {len(all_commit_files)} files in {base_ref} ({base_commit}). Processing...")

    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        # --- Header ---
        out.write("REPO CONTEXT SNAPSHOT\n")
        out.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
        out.write(f"Source: {repo_root.name}\n")
        out.write(f"Base ref: {base_ref}\n")
        out.write(f"Git commit: {base_commit}\n")
        out.write("=" * 40 + "\n\n")

        # --- Directory tree ---
        out.write(_generate_tree(all_commit_files))
        out.write("\n\n" + "=" * 40 + "\n\n")

        # --- File contents (from commit) ---
        max_size_bytes = int(max_file_size_kb) * 1024
        for rel_path in all_commit_files:
            if _should_skip_relpath(rel_path):
                continue

            if not _is_likely_text_path(rel_path):
                skipped_files.append(f"{rel_path} (Binary extension)")
                continue

            size_bytes = _git_cat_file_size(repo_root, base_ref, rel_path)
            if size_bytes is None:
                skipped_files.append(f"{rel_path} (Could not read blob size)")
                continue

            if size_bytes > max_size_bytes:
                skipped_files.append(f"{rel_path} (Too large: {size_bytes/1024:.1f} KB)")
                continue

            blob = _git_show_bytes(repo_root, base_ref, rel_path)
            if blob is None:
                skipped_files.append(f"{rel_path} (Could not read blob)")
                continue

            if not _is_text_blob(rel_path, blob):
                skipped_files.append(f"{rel_path} (Binary detected)")
                continue

            content = blob.decode("utf-8", errors="replace")
            out.write(f"----- START FILE: {rel_path} -----\n")
            out.write(f"<file path=\"{rel_path}\">\n")
            out.write(content)
            out.write("\n</file>\n")
            out.write(f"----- END FILE: {rel_path} -----\n\n")

            processed_files.append(rel_path)
            total_chars += len(content)

        # --- Skipped log ---
        if skipped_files:
            out.write("\n" + "=" * 40 + "\n")
            out.write("SKIPPED FILES LOG:\n")
            for item in skipped_files:
                out.write(f"- {item}\n")
            out.write("\n")

        # --- Patch (annotated; appended) ---
        if include_patch and patch_annotated:
            out.write(patch_annotated)

    print(f"✅ Wrote snapshot: {out_path}")
    if patch_out_path is not None and include_patch:
        print(f"✅ Wrote patch (raw diff): {Path(patch_out_path).expanduser().resolve()}")
    elif patch_out_path is not None and not include_patch:
        print("ℹ️  Note: --patch-out was provided but --no-patch disabled patch generation.")
    print(f"✅ Flattened {len(processed_files)} files from {base_ref}")
    if skipped_files:
        print(f"🚫 Skipped {len(skipped_files)} files (see end of snapshot)")
    print(f"📊 Total content size: {total_chars/1024:.1f} KB")


def write_patch_only(
    *,
    out_path: Path,
    max_file_size_kb: int = MAX_FILE_SIZE_KB_DEFAULT,
    base_ref: str = DEFAULT_BASE_REF,
    include_untracked: bool = True,
) -> None:
    """Write only a raw unified diff patch (apply-able by git)."""
    repo_root = _repo_root()
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    patch_raw, patch_annotated = _collect_local_patch(
        repo_root,
        base_ref=base_ref,
        include_untracked=include_untracked,
        max_file_size_kb=int(max_file_size_kb),
    )

    _write_bytes(out_path, patch_raw)

    # Helpful console feedback (without polluting the patch file)
    if patch_raw.strip():
        print(f"✅ Wrote patch (raw diff): {out_path}")
    else:
        print(f"✅ Wrote patch (raw diff): {out_path} (empty; no local changes detected)")
    # If you want an LLM-friendly annotated patch, it is included in the full snapshot output.
    # We keep this patch file pure so `git apply` works.


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Flatten committed repo files into a single text snapshot for LLM context, "
            "optionally appending an annotated patch of local changes. "
            "Also supports writing a raw apply-able patch file."
        )
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Optional explicit output path. If omitted, writes a timestamped file under "
            f"{DEFAULT_OUTPUT_DIR}/"
        ),
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (used only when --out is omitted). Default: {DEFAULT_OUTPUT_DIR}",
    )
    ap.add_argument(
        "--prefix",
        type=str,
        default=DEFAULT_PREFIX,
        help=(
            f"Output filename prefix (used only when --out is omitted). "
            f"Default: {DEFAULT_PREFIX} (or {DEFAULT_PATCH_PREFIX} for --patch-only)."
        ),
    )
    ap.add_argument(
        "--base",
        type=str,
        default=DEFAULT_BASE_REF,
        help=f"Git ref to snapshot/diff from (default: {DEFAULT_BASE_REF}).",
    )
    ap.add_argument(
        "--no-patch",
        action="store_true",
        help="Do not append a local-changes patch section to the snapshot.",
    )
    ap.add_argument(
        "--patch-out",
        type=str,
        default=None,
        help=(
            "Optional path to also write the local-changes patch as a separate file "
            "(raw unified diff ONLY; apply-able with git apply). "
            "The snapshot still includes an annotated patch section unless --no-patch is set."
        ),
    )
    ap.add_argument(
        "--patch-only",
        action="store_true",
        help=(
            "Write ONLY the patch (raw unified diff; apply-able with git apply) "
            "and do not write a snapshot file."
        ),
    )
    ap.add_argument(
        "--no-include-untracked",
        action="store_true",
        help="Do not include untracked text files as new-file patches.",
    )
    ap.add_argument(
        "--max-file-kb",
        type=int,
        default=MAX_FILE_SIZE_KB_DEFAULT,
        help=f"Skip files larger than this. Default: {MAX_FILE_SIZE_KB_DEFAULT}",
    )
    args = ap.parse_args()

    repo_root = _repo_root()
    base_ref = str(args.base or DEFAULT_BASE_REF)
    commit = _try_git_rev_parse(repo_root, base_ref, short=True) or "nogit"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    include_untracked = not bool(args.no_include_untracked)
    max_file_kb = int(args.max_file_kb)

    # --- Patch-only mode ---
    if bool(args.patch_only):
        if args.out:
            out_path = Path(args.out)
            if not out_path.is_absolute():
                out_path = repo_root / out_path
        else:
            out_dir = Path(args.out_dir)
            if not out_dir.is_absolute():
                out_dir = repo_root / out_dir

            prefix = str(args.prefix or "")
            # If user didn't override --prefix, pick a patch-specific default.
            if prefix == DEFAULT_PREFIX:
                prefix = DEFAULT_PATCH_PREFIX

            filename = f"{prefix}__{stamp}__{commit}.diff"
            out_path = out_dir / filename

        write_patch_only(
            out_path=out_path,
            max_file_size_kb=max_file_kb,
            base_ref=base_ref,
            include_untracked=include_untracked,
        )
        return 0

    # --- Snapshot mode ---
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = repo_root / out_path
    else:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = repo_root / out_dir
        filename = f"{args.prefix}__{stamp}__{commit}.txt"
        out_path = out_dir / filename

    patch_out_path = None
    if args.patch_out:
        patch_out_path = Path(args.patch_out)
        if not patch_out_path.is_absolute():
            patch_out_path = repo_root / patch_out_path

    flatten_repo(
        out_path=out_path,
        max_file_size_kb=max_file_kb,
        base_ref=base_ref,
        include_patch=(not bool(args.no_patch)),
        include_untracked=include_untracked,
        patch_out_path=patch_out_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
