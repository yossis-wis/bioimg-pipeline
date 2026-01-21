# LLM workflow

This repo is intentionally kept **small and “zip-able”**: code + configs + docs live in git, while large inputs/outputs/models live in the **data bench** referenced by `BIOIMG_DATA_ROOT`.

We sometimes use an LLM (ChatGPT / GPT‑5.2 Pro) to **plan and implement** changes. The key is to stay **human-in-the-loop**:

- the LLM proposes changes (often as a zip overlay)
- you review diffs in GitHub Desktop
- you run verification scripts + QC
- only then do you commit

This document is written for **humans** and for the **LLM** (as repo-native instructions).

---

## Quickstart (human)

1) (Optional, recommended) Generate a flattened repo snapshot for the LLM context:

```bash
python scripts/flatten_repo.py
```

This produces a single **flat text** file under `repo_snapshots/`.

Important behavior:

- The snapshot section is built from the **committed tree** at `HEAD` (or `--base <ref>`), so it is stable even if you have uncommitted local edits.
- By default it also appends a **LOCAL CHANGES (PATCH)** section showing your current working-tree diff vs the base ref.
  - This is meant for *LLM context* (human-readable + annotated), not for `git apply`.

Useful modes:

```bash
# Snapshot only (no local patch section appended)
python scripts/flatten_repo.py --no-patch

# Patch only (raw unified diff; safe to use with `git apply`)
python scripts/flatten_repo.py --patch-only

# Snapshot + ALSO write a raw patch file alongside it
python scripts/flatten_repo.py --patch-out repo_snapshots/local_changes.diff

# Exclude untracked files from the patch (default: include untracked *text* files)
python scripts/flatten_repo.py --no-include-untracked

# Change the base ref (snapshot + diff are relative to this)
python scripts/flatten_repo.py --base HEAD
python scripts/flatten_repo.py --base main
```

Patch sanity checks:

```bash
# If you already have the local changes in your working tree,
# this should succeed (verifies the patch matches what you have now).
git apply --check --reverse repo_snapshots/local_changes.diff

# On a clean checkout of the base ref, this should succeed.
git apply --check repo_snapshots/local_changes.diff
```

2) Ask the LLM for a **plan** (no code), including:
- files to modify/add/delete
- verification commands you will run

3) Ask the LLM to implement and deliver a **zip overlay**.

4) Extract the zip **over the repo root**, then review diffs in GitHub Desktop.

5) Run the verification ladder:

```bash
python scripts/verify_setup.py
ruff check .
pytest -q
python scripts/generate_phantom_tiff.py --config configs/integrated_sim.yaml --overwrite
python drivers/run_integrated.py --config configs/integrated_sim.yaml
```

6) Visually inspect QC artifacts under `$BIOIMG_DATA_ROOT/runs/...`.

7) Commit only after verification.

---

## Repo invariants (non‑negotiable)

### 1) Code vs data bench separation

- **Repo (git)**: code + tracked configs + docs
- **Data bench (NOT in git)**: inputs, run outputs, caches, and StarDist models

Never include in LLM zips and never commit:

- `runs/`, `raw_staging/`, `cache/`, `models/`, `reports/`
- `configs/local/` or `configs/*.local.yaml`

`BIOIMG_DATA_ROOT` must point to the data bench (see platform setup docs).

### 2) Driver vs Kernel separation

- `drivers/` handle filesystem paths, config parsing, and writing outputs.
- `src/` kernels operate on **in-memory 2D arrays** and should not do filesystem I/O.

### 3) Contracts must hold

- Spot table contract: `docs/CONTRACTS.md`
- Run manifest contract: `docs/CONTRACTS.md`

Do not remove/rename required columns or manifest keys.

### 4) Notebooks are stored as text

Notebooks are stored as **Jupytext percent-format `.py` files** under `notebooks/`.
Do not commit `.ipynb`.

---

## Zip overlay workflow (human)

> This repo is shallow and changes are often committed directly to `main`.
> Branches are optional. If you skip branches, use a **checkpoint commit** so rollback is easy.

### 0) Pre-flight

- Make sure your working tree is clean (no uncommitted changes).
- (Recommended) create a checkpoint commit before applying an LLM zip.

### 1) Apply the zip overlay

- Extract the zip **over the repo root**.
- Ensure the zip contains repo-relative paths (e.g. `docs/LLM_WORKFLOW.md`), not an extra top-level folder.

### 2) Review diffs in GitHub Desktop

Before running anything:

- scan the changed file list
- open each changed file
- sanity-check any new scripts, subprocess calls, or filesystem deletes

### 3) Run verification + QC

Use the verification ladder above. For real data, start with a **mini-batch** (3–10 representative `.ims` files) before a full batch.

### 4) Commit

Commit message should capture:
- what changed
- what you verified (commands/notebooks/QC)

---

## Important limitation: zip overlays do not delete files

A zip overlay typically **adds/overwrites** files but does not remove old ones.

If deletions are needed:
- the LLM must list them explicitly
- you delete them manually (then verify + commit)

---

## Rules for the LLM (read before proposing changes)

### A) Required interaction pattern

1) **Plan first (no code)**
   - goal (1–3 bullets)
   - files to modify/add/delete
   - verification commands the human will run

2) **Implement second**
   - deliver either:
     - a **zip overlay** (preferred for this repo), or
     - a single **unified diff**

### B) Zip content rules (strict)

The zip MUST:

- use **repo-relative paths** (no extra top-level directory)
- include only **tracked, text-based repo files** (code/docs/config templates)
- avoid “format churn” and unrelated refactors

The zip MUST NOT include:

- `.git/` or git internals
- any data-bench paths (`runs/`, `raw_staging/`, `cache/`, `models/`, `reports/`)
- local configs (`configs/local/`, `configs/*.local.yaml`)
- binary artifacts (`*.tif`, `*.png`, `*.parquet`, `*.zip`, etc.)

### C) Change discipline

- Keep changes minimal and reviewable.
- Preserve Driver/Kernel separation.
- Do not silently break contracts in `docs/CONTRACTS.md`.
- If you add a config key, document it and give it a safe default.

### D) Definition of “done”

A change is acceptable only if the human can:

- review it cleanly in GitHub Desktop diffs
- run:
  - `python scripts/verify_setup.py`
  - `ruff check .`
  - `pytest -q`
  - simulated phantom run + integrated driver run
- visually verify QC outputs

---

## Prompt templates (copy/paste)

### 1) Planning only

> Read the attached flattened repo snapshot (or GitHub state). Propose a plan to achieve: <goal>.
> List files to change/create/delete and exact verification commands.
> Do not write code yet.

### 2) Implementation (zip overlay)

> Implement the approved plan.
> Output a zip overlay containing only changed/new tracked files.
> Also list every file included in the zip and the verification commands.
> Do not include any outputs, local configs, or data-bench files.

### 3) Implementation (unified diff)

> Implement the approved plan.
> Output a single unified diff (`git diff` style).
> No refactors unless necessary.
> Include verification commands.

---

## Getting repo context into the LLM

### Preferred: flattened snapshot (optionally with local patch appended)

Generate a snapshot with:

```bash
python scripts/flatten_repo.py
```

By default, this writes a timestamped snapshot under `repo_snapshots/` (which is ignored by git).
If you have local changes, the snapshot will also include an appended **LOCAL CHANGES (PATCH)** section.

### Patch-only updates (iterative)

If the LLM already has a recent snapshot and you’re iterating quickly, you can send only your latest diff:

```bash
python scripts/flatten_repo.py --patch-only --out repo_snapshots/local_changes.diff
```

This output is **raw unified diff text** (apply-able with `git apply`).
It is also usually the best thing to paste into an LLM when you want it to reason about your latest local changes.

### If using a GitHub connector

A connector can be convenient when:

- the repo is pushed and clean
- you want quick per-file lookups

If you have **local changes not pushed**, attach a flattened snapshot (or at least the relevant diffs/files). The snapshot should be treated as the source of truth.

---

## Repo snapshots

See `repo_snapshots/README.md` for the purpose and retention policy of timestamped flattened snapshots.
