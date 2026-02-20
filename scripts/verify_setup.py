from __future__ import annotations

import os
import sys
import platform
import subprocess
from pathlib import Path


# Modules required for integrated workflow + file I/O
REQUIRED_IMPORTS = [
    "numpy",
    "pandas",
    "tifffile",
    "yaml",
    "pyarrow",
    "skimage",
    "matplotlib",
    "h5py",  # needed for reading Imaris .ims
    "pptx",  # python-pptx (spot atlas QC)
    "cairosvg",  # rasterize SVG figures for slide export
    "numba",  # required by StarDist
    "stardist",
    "csbdeep",
    "tensorflow",
]


def try_git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        return out
    except Exception:
        return None


def _check_typing_extensions() -> tuple[bool, str]:
    """Return (ok, info) where ok means TypeAliasType is available."""
    try:
        import typing_extensions as te  # noqa: F401
        from typing_extensions import TypeAliasType  # noqa: F401

        ver = getattr(te, "__version__", "unknown")
        return True, f"TypeAliasType available; version={ver}"
    except Exception as exc:
        try:
            import typing_extensions as te  # type: ignore

            ver = getattr(te, "__version__", "unknown")
        except Exception:
            ver = "not installed"
        return False, f"version={ver}; error={exc}"


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    print("=== bioimg-pipeline :: verify_setup ===")
    print(f"repo_root: {repo_root}")
    print(f"python: {sys.executable}")
    print(f"python_version: {sys.version.split()[0]}")
    print(f"conda_env: {os.environ.get('CONDA_DEFAULT_ENV', '(not set)')}")
    print(f"platform: {platform.platform()}")
    print(f"machine: {platform.machine()}")

    # 1) Env var
    data_root = os.environ.get("BIOIMG_DATA_ROOT")
    if not data_root:
        print("ERROR: BIOIMG_DATA_ROOT is not set.")
        print()
        if sys.platform.startswith("win"):
            print(
                r'Fix (PowerShell): [Environment]::SetEnvironmentVariable("BIOIMG_DATA_ROOT","D:\\bioimg-data","User")'
            )
            print("Then close/reopen terminals (and VS Code) so the variable is visible.")
        elif sys.platform == "darwin":
            print("Fix (macOS zsh): add this line to ~/.zshrc:")
            print('  export BIOIMG_DATA_ROOT="$HOME/bioimg-data"')
            print("Then run:  source ~/.zshrc")
        else:
            print("Fix (bash): set and export BIOIMG_DATA_ROOT, e.g.:")
            print('  export BIOIMG_DATA_ROOT="$HOME/bioimg-data"')
        return 2

    data_root = Path(data_root).expanduser().resolve()
    print(f"BIOIMG_DATA_ROOT: {data_root}")
    if not data_root.exists():
        print(f"ERROR: BIOIMG_DATA_ROOT path does not exist: {data_root}")
        return 2

    # 2) Expected bench folders
    expected = ["raw_staging", "runs", "cache", "models"]
    for name in expected:
        p = data_root / name
        if not p.exists():
            print(f"ERROR: missing bench folder: {p}")
            return 2
    print("bench folders: OK")

    # 3) Required repo files
    required_repo_files = [
        "environment.yml",
        "docs/CONTRACTS.md",
        "configs/integrated_sim.yaml",
        "configs/integrated_ims.example.yaml",
    ]
    for rel in required_repo_files:
        p = repo_root / rel
        if not p.exists():
            print(f"ERROR: missing repo file: {p}")
            return 2
    print("repo files: OK")

    # 4) Required imports
    for mod in REQUIRED_IMPORTS:
        try:
            __import__(mod)
            print(f"import {mod}: OK")
        except Exception as e:
            print(f"ERROR: import {mod} failed: {e}")
            msg = str(e)
            if mod in ("stardist", "numba") and ("no module named" in msg.lower()) and ("numba" in msg.lower()):
                print()
                print("Hint: StarDist requires numba. Fix:")
                print("  conda activate bioimg-pipeline")
                print("  conda env update -f environment.yml --prune")
                print("  python scripts/verify_setup.py")
            return 2

    # 4b) typing_extensions sanity check (VS Code Jupyter / Spyder)
    ok, info = _check_typing_extensions()
    if ok:
        print(f"typing_extensions: OK ({info})")
    else:
        print(f"ERROR: typing_extensions check failed ({info})")
        print()
        print("This usually happens after `conda env create/update` because the TensorFlow pip install")
        print("downgrades typing_extensions to <4.6. Recent VS Code Jupyter / Spyder expects")
        print("typing_extensions to provide TypeAliasType.")
        print()
        print("Fix (recommended):")
        print("  python scripts/fix_typing_extensions.py")
        print()
        print("Alternative (direct pip):")
        print('  python -m pip install -U "typing-extensions>=4.7"')
        print()
        print("After fixing:")
        print("  - Re-run: python scripts/verify_setup.py")
        reload_keys = "Cmd+Shift+P" if sys.platform == "darwin" else "Ctrl+Shift+P"
        print(f"  - Reload VS Code: {reload_keys} -> Developer: Reload Window")
        return 2

    # 5) Write test to runs/
    test_dir = data_root / "runs" / "_setup_write_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "touch.txt").write_text("ok\n", encoding="utf-8")
    print(f"\nwrite test: OK ({test_dir})")

    # 6) Git commit (optional)
    commit = try_git_commit(repo_root)
    if commit:
        print(f"git_commit: {commit[:12]}")
    else:
        print("git_commit: (unavailable in this shell)")

    print("SETUP OK ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


