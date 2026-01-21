# Repo snapshots (flattened context for LLM + debugging)

This folder stores **flattened snapshots** of the tracked text files in this repository.
Each snapshot is a single text file containing:

- a directory tree of tracked files
- the contents of each tracked text file (wrapped with clear START/END markers)
- (optional, default) a **LOCAL CHANGES (PATCH)** section at the end that captures
  your uncommitted work as a unified diff (similar to "Copy patch" in GitHub/Codex)

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
- Snapshots should **not** include run outputs or data bench files.

## How to create a snapshot

From the repo root:

```bash
python scripts/flatten_repo.py
```

By default the snapshot is generated from the committed `HEAD` state, even if you
have local modifications, and it appends a patch section capturing your local
changes.

If you want *only* the committed snapshot (no patch), run:

```bash
python scripts/flatten_repo.py --no-patch
```

By default this writes a new timestamped file under `repo_snapshots/`, for example:

```
repo_snapshots/repo_context__20260120T131425Z__a1b2c3d.txt
```

## When to attach a snapshot to an LLM

Attach a snapshot when:

- you have local changes that are not pushed to GitHub
- you want the LLM to propose multi-file changes
- you want maximum auditability and minimal “implicit assumptions”

If you are using a GitHub connector *and* your repo is pushed and clean, you may not need to attach snapshots.
