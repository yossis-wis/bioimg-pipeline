# Repo snapshots (flattened context for LLM + debugging)

This folder stores **flattened snapshots** of the tracked text files in this repository.
Each snapshot is a single text file containing:

- a directory tree of tracked files
- the contents of each tracked text file (wrapped with clear START/END markers)
- (by default) an appended **LOCAL CHANGES (PATCH)** section showing your current working-tree
  changes as a unified diff vs a base git ref (default: `HEAD`)

## Why this exists

### 1) LLM context
When asking an LLM to plan or implement changes (often delivered as a zip overlay), a single snapshot file makes the model’s view **deterministic**:
it sees exactly what the repo looked like at that moment.

### 2) Debugging over time
A timestamped snapshot can be attached to a chat to analyze “what changed?”
without requiring the LLM to have direct access to your git history.

## Important rules

- Snapshots are **artifacts**, not the source of truth.
  The source of truth is the git commit history.
- By default, snapshot `.txt` files in this folder are **ignored by git** (see `repo_snapshots/.gitignore`).
  This keeps the repo small and avoids bloating diffs.
- Patch artifacts (`*.diff`, `*.patch`) written by `flatten_repo.py --patch-out` or `--patch-only`
  are also ignored by git.
- Snapshots should **not** include run outputs or data bench files.

## How to create a snapshot

From the repo root:

```bash
python scripts/flatten_repo.py
```

By default this writes a new timestamped file under `repo_snapshots/`, for example:

```
repo_snapshots/repo_context__20260120T131425Z__a1b2c3d.txt
```

Useful modes:

```bash
# Snapshot only (no local patch section appended)
python scripts/flatten_repo.py --no-patch

# Snapshot + ALSO write a raw patch file (apply-able with `git apply`)
python scripts/flatten_repo.py --patch-out repo_snapshots/local_changes.diff

# Change the base ref (snapshot + diff are relative to this)
python scripts/flatten_repo.py --base HEAD
python scripts/flatten_repo.py --base main
```

Notes:

- The snapshot section is generated from the **committed tree** at the base ref (default: `HEAD`).
  This makes the snapshot deterministic even if you have uncommitted edits.
- The appended **LOCAL CHANGES (PATCH)** section is meant for *human/LLM context* and includes some
  headings and `git status` output. It is **not** intended to be applied via `git apply`.
  If you need an apply-able patch, use `--patch-only` or `--patch-out`.

## How to create a patch file (no snapshot)

If you only want the raw patch (for example, you already sent the repo snapshot to an LLM once and
now you want to iterate by sending diffs), run:

```bash
python scripts/flatten_repo.py --patch-only
```

This writes a `*.diff` file (raw unified diff only).

You can also choose a stable filename:

```bash
python scripts/flatten_repo.py --patch-only --out repo_snapshots/local_changes.diff
```

Options you may want:

```bash
# Diff against a different base
python scripts/flatten_repo.py --patch-only --base main

# Exclude untracked files (default includes untracked *text* files as new-file patches)
python scripts/flatten_repo.py --patch-only --no-include-untracked
```

Patch sanity checks:

```bash
# If you already have the local changes in your working tree, this should succeed.
git apply --check --reverse repo_snapshots/local_changes.diff

# On a clean checkout of the base ref, this should succeed.
git apply --check repo_snapshots/local_changes.diff
```


## Applying a patch you got from an LLM

If an LLM prints a raw unified diff in chat (instead of sending a zip), you can save it and apply it locally:

1. Create `repo_snapshots/llm_patch.diff` (this folder ignores `*.diff` in git).
2. Copy/paste the diff text into the file.
   - Remove any Markdown code fences.
   - Remove any prose before the first `diff --git ...` line.
3. Dry-run + inspect:

```bash
git apply --stat repo_snapshots/llm_patch.diff
git apply --check repo_snapshots/llm_patch.diff
```

4. Apply + review:

```bash
git apply repo_snapshots/llm_patch.diff
git status
git diff
```

If the patch was generated against a different base commit/branch, `git apply --check` will fail; checkout the intended base first (or ask the LLM to rebase the patch).

Note: the **annotated** `LOCAL CHANGES (PATCH)` section inside a full snapshot is for reading only and is not meant to be applied. If you need an apply-able patch, request (or generate) **raw unified diff text only**.

## When to attach a snapshot to an LLM

Attach a snapshot when:

- you have local changes that are not pushed to GitHub
- you want the LLM to propose multi-file changes
- you want maximum auditability and minimal “implicit assumptions”

If you are using a GitHub connector *and* your repo is pushed and clean, you may not need to attach snapshots.
