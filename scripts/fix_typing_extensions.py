from __future__ import annotations

import subprocess
import sys


MIN_VERSION = "4.7"


def _has_type_alias_type() -> tuple[bool, str]:
    """Return (ok, info).

    ok=True  -> TypeAliasType import works.
    ok=False -> info contains the installed version (if any) + the exception.
    """
    try:
        import typing_extensions as te  # noqa: F401
        from typing_extensions import TypeAliasType  # noqa: F401

        version = getattr(te, "__version__", "unknown")
        return True, version
    except Exception as exc:
        try:
            import typing_extensions as te  # type: ignore

            version = getattr(te, "__version__", "unknown")
        except Exception:
            version = "not installed"
        return False, f"{version} ({exc})"


def _run(cmd: list[str]) -> int:
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def main() -> int:
    ok, info = _has_type_alias_type()
    if ok:
        print(f"typing_extensions OK (TypeAliasType available; version={info}).")
        return 0

    print("typing_extensions appears too old or missing TypeAliasType:", info)
    print()
    print("Fixing by upgrading typing-extensions...")
    rc = _run([sys.executable, "-m", "pip", "install", "-U", f"typing-extensions>={MIN_VERSION}"])
    if rc != 0:
        print("ERROR: pip install failed.")
        return rc

    ok2, info2 = _has_type_alias_type()
    if ok2:
        print(f"typing_extensions OK after upgrade (version={info2}).")
        print()
        print("Note:")
        print("  pip may warn that TensorFlow pins typing-extensions<4.6 in its metadata.")
        print("  In practice, this repo requires a newer typing-extensions for VS Code Jupyter/Spyder.")
        print()
        print("Next steps:")
        print("  - Re-run: python scripts/verify_setup.py")
        print("  - If VS Code was open: Ctrl+Shift+P -> Developer: Reload Window")
        return 0

    print("ERROR: typing_extensions still missing TypeAliasType after upgrade:", info2)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
