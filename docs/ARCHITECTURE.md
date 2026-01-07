# One-Page White Paper: “Slice 0” Plan

**Goal:** a minimal, end-to-end working pipeline that can evolve into the full system.

## Objective

Produce a reproducible, portable pipeline “vertical slice” that:

- Reads a single microscopy image (TIFF)
- Detects candidate spots with a simple method (baseline detector)
- Outputs a standardized table of spot coordinates and basic metrics
- Writes a run manifest capturing metadata and reproducibility details

## Architecture: Kernel – Driver – Manifest

**Kernel:** pure computation (no file I/O).  
Input: image array + parameters. Output: spot table.

Example signature:

- `detect_spots(I, θ) -> T_spots`

**Driver:** orchestrates file I/O and run structure.  
Loads config, reads image, calls kernel, writes outputs.

Example:

- `run_slice0(config.yaml) -> run_folder`

**Manifest:** stored with outputs for reproducibility.  
Contains timestamp, input file path, config snapshot, git commit hash, environment info.

## Storage and Compute Geography

Separate concerns:

- **Code repo** (small, versioned): lives in Git (e.g. `C:\Code\bioimg-pipeline`)
- **Data bench** (large, not versioned): lives outside Git (e.g. `D:\bioimg-data`)
- **Docs** (human-readable): can live in OneDrive separately

| Layer | Location | Rule |
|---|---|---|
| Code | Git repo | Only source + configs + docs |
| Raw/Intermediate data | Data bench | Never committed, not synced |
| Outputs (runs) | Data bench / `runs/` | Timestamped, reproducible |
| Docs | OneDrive | Notes, plots, papers |

## Slice 0 Contracts

### A) Spots table schema (minimal but future-compatible)

- `frame` (int)
- `y_px`, `x_px` (float)
- `intensity_adu` (float)
- `background_adu` (float)
- `snr` (float)

### B) Manifest keys

- `timestamp`
- `input_path`
- `config_snapshot`
- `git_commit`
- `output_folder`

## Incremental Steps to Implement Slice 0

1. Create repo structure: `src/`, `drivers/`, `configs/`, `docs/`
2. Create data bench folders: `raw_staging/`, `runs/`, `cache/`
3. Set environment variable: `BIOIMG_DATA_ROOT = D:\bioimg-data`
4. Write `configs/dev.yaml` for threshold + paths
5. Implement kernel: simple detector (threshold or LoG)
6. Implement driver: read TIFF, call kernel, write `spots.parquet` + `manifest.yaml`
7. Verify end-to-end run on one image, commit code
