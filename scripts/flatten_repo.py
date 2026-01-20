"""Flatten tracked repo text files into a single snapshot for LLM context.

This script is intentionally simple and conservative:
- It only considers files returned by `git ls-files`.
- It skips large files and likely-binary files.
- It writes a single text file with a directory tree + file contents.

Default behavior writes a **timestamped** snapshot into:

    repo_snapshots/repo_context__<timestamp>__<commit>.txt

Snapshots are intended to be **artifacts** (ignored by git), not a replacement for git history.
See `repo_snapshots/README.md` for the rationale and usage.
"""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# --- Defaults ---
DEFAULT_OUTPUT_DIR = "repo_snapshots"
DEFAULT_PREFIX = "repo_context"
MAX_FILE_SIZE_KB_DEFAULT = 500  # Skip files larger than this (avoid data dumps)

# Even with git ls-files, we explicitly skip file types that are commonly binary
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


def _try_git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def get_git_files(repo_root: Path) -> list[str]:
    """Return tracked files (repo-relative) using `git ls-files`."""
    try:
        out = subprocess.check_output(
            ["git", "ls-files"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        files = [line.strip() for line in out.splitlines() if line.strip()]
        return sorted(files)
    except subprocess.CalledProcessError:
        print("❌ Error: not a git repository (or git is not installed).")
        return []


def _is_text_file(path: Path) -> bool:
    """Heuristic: extension blocklist + null-byte check."""
    if path.suffix.lower() in BINARY_EXTENSIONS:
        return False

    try:
        with path.open("rb") as f:
            chunk = f.read(1024)
        if b"\0" in chunk:
            return False
    except Exception:
        return False

    return True


def _should_skip_relpath(rel_path: str) -> bool:
    """Skip snapshot artifacts if they somehow become tracked."""
    rel = rel_path.replace("\\", "/")

    # If someone mistakenly tracks snapshot artifacts, skip them to avoid runaway growth.
    if rel.startswith("repo_snapshots/") and rel.endswith(".txt"):
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


def flatten_repo(
    *,
    out_path: Path,
    max_file_size_kb: int = MAX_FILE_SIZE_KB_DEFAULT,
) -> None:
    repo_root = _repo_root()
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_git_files = get_git_files(repo_root)
    if not all_git_files:
        print("No tracked files found. Run from a cloned git repo.")
        return

    processed_files: list[str] = []
    skipped_files: list[str] = []
    total_chars = 0

    print(f"🔍 Found {len(all_git_files)} tracked files. Processing...")

    with out_path.open("w", encoding="utf-8") as out:
        commit = _try_git_commit(repo_root) or "(unavailable)"

        # --- Header ---
        out.write("REPO CONTEXT SNAPSHOT\n")
        out.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
        out.write(f"Source: {repo_root.name}\n")
        out.write(f"Git commit: {commit}\n")
        out.write("=" * 40 + "\n\n")

        # --- Directory tree ---
        out.write(_generate_tree(all_git_files))
        out.write("\n\n" + "=" * 40 + "\n\n")

        # --- File contents ---
        for rel_path in all_git_files:
            if _should_skip_relpath(rel_path):
                continue

            full_path = repo_root / rel_path

            # Size check
            try:
                size_bytes = full_path.stat().st_size
            except Exception as exc:
                skipped_files.append(f"{rel_path} (Stat error: {exc})")
                continue

            if size_bytes > (int(max_file_size_kb) * 1024):
                skipped_files.append(f"{rel_path} (Too large: {size_bytes/1024:.1f} KB)")
                continue

            # Binary check
            if not _is_text_file(full_path):
                skipped_files.append(f"{rel_path} (Binary detected)")
                continue

            try:
                content = full_path.read_text(encoding="utf-8", errors="replace")

                out.write(f"----- START FILE: {rel_path} -----\n")
                out.write(f"<file path=\"{rel_path}\">\n")
                out.write(content)
                out.write("\n</file>\n")
                out.write(f"----- END FILE: {rel_path} -----\n\n")

                processed_files.append(rel_path)
                total_chars += len(content)
            except Exception as exc:
                skipped_files.append(f"{rel_path} (Read error: {exc})")

        # --- Skipped log ---
        if skipped_files:
            out.write("\n" + "=" * 40 + "\n")
            out.write("SKIPPED FILES LOG:\n")
            for item in skipped_files:
                out.write(f"- {item}\n")

    print(f"✅ Wrote snapshot: {out_path}")
    print(f"✅ Flattened {len(processed_files)} files")
    if skipped_files:
        print(f"🚫 Skipped {len(skipped_files)} files (see end of snapshot)")
    print(f"📊 Total content size: {total_chars/1024:.1f} KB")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Flatten tracked repo files into a single text snapshot for LLM context"
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
        help=f"Output filename prefix (used only when --out is omitted). Default: {DEFAULT_PREFIX}",
    )
    ap.add_argument(
        "--max-file-kb",
        type=int,
        default=MAX_FILE_SIZE_KB_DEFAULT,
        help=f"Skip files larger than this. Default: {MAX_FILE_SIZE_KB_DEFAULT}",
    )
    args = ap.parse_args()

    repo_root = _repo_root()
    commit = _try_git_commit(repo_root) or "nogit"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

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

    flatten_repo(out_path=out_path, max_file_size_kb=int(args.max_file_kb))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
