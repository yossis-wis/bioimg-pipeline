# bioimg-pipeline

## Integrated Slice (Nuclei + Spots)

This pipeline performs **nuclear segmentation** (StarDist) and **spot detection** in a single pass.

Spot detection uses a **TrackMate-style LoG detector** for candidate generation, followed by a
**fixed-mask photometry step** (`in5/out0` → `u0_min`) that is tuned to the microscope/dye. It produces quality control artifacts including spot cutouts and visual overlays.
Each run processes **one 2D plane per file**, controlled by `ims_resolution_level`, `ims_time_index`, and `ims_z_index` for `.ims` inputs.

### Setup

Follow the platform guide first:

- Windows: [docs/SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md)
- macOS: [docs/SETUP_MAC.md](docs/SETUP_MAC.md)

### Docs map

- [Architecture](docs/ARCHITECTURE.md)
- [Spot detection (TrackMate-style LoG + u0 photometry)](docs/SPOT_DETECTION.md)
- [Data contracts](docs/CONTRACTS.md)
- [Notebooks (QC workflows)](docs/NOTEBOOKS.md)
- [LLM workflow (human-in-the-loop zip overlays)](docs/LLM_WORKFLOW.md)

### 1) Setup & Verification

```bash
python scripts/verify_setup.py
```

### 2) Run on Simulated Data

Generate a synthetic 2-channel TIFF (Ch1: Nuclei, Ch2: Spots):

```bash
python scripts/generate_phantom_tiff.py --config configs/integrated_sim.yaml
```

Run the integrated analysis:

```bash
python drivers/run_integrated.py --config configs/integrated_sim.yaml
```

### 3) Run on Real Data (`.ims` or `.tif`)

1. Ensure you have your StarDist model in `$BIOIMG_DATA_ROOT/models/`.
2. Create a **local** config from the tracked template:
   - `mkdir -p configs/local`
   - `cp configs/integrated_ims.example.yaml configs/local/integrated_ims.local.yaml`
   - PowerShell:
     - `New-Item -ItemType Directory -Force configs/local`
     - `Copy-Item configs/integrated_ims.example.yaml configs/local/integrated_ims.local.yaml`
   - Edit the local file (do not commit anything under `configs/local/`).
3. Update your config with:
   - `input_relpath` (single file), `input_relpaths` (list), or `input_glob` (batch).
     - For storWIS: point `input_glob` at the network share (e.g. `S:/BIC/.../*.ims`)
     - Relative paths resolve under `$BIOIMG_DATA_ROOT`; absolute paths are used as-is.
   - `stardist_model_dir` (path to your StarDist model folder)
   - `channel_nuclei` and `channel_spots` (1-based indices; `channel_spots` can be a list like `[2, 3]`)
   - Detector parameters
     - **Important:** set `spot_pixel_size_nm` in **nm/px** using your file metadata.
       You can print it with:
       `python scripts/inspect_pixel_size.py --input <your_file.tif|.ims>`
4. (Optional) If you want local intermediates + storWIS publish:
   - Keep `output_runs_dir` under `BIOIMG_DATA_ROOT` for fast local writes.
   - Set `publish_dir` to a writable storWIS location to copy final outputs.
   - Set `publish_mirror: true` and `input_base_dir` to mirror subfolders (e.g. `.../5ms`, `.../45ms`).
     - Mirror mode publishes to `publish_dir/<condition>/<batch_root>/<run_dir>`.
     - Non-mirror mode publishes to `publish_dir/<batch_root>/<run_dir>`.
   - Set `publish_mode` to control collisions (`error`, `overwrite`, `merge`).
5. (Optional) For batch robustness:
   - `continue_on_error: true` to keep processing after a failed file.
   - `skip_existing: true` to resume without redoing completed outputs (requires a prior `run_manifest.yaml`).
   - `republish_skipped: true` to retry publishing for skipped runs.
   - `batch_dir_name` to reuse a stable output folder name across reruns.
   - `batch_aggregate_spots: true` to write a combined `spots_aggregate.parquet`.
6. (Optional) If you already have nuclei labels from another tool, set
   `nuclei_labels_relpath` to skip StarDist segmentation.
7. (Optional) If you have no nuclear channel, set `skip_nuclei_segmentation: true`
   and omit `channel_nuclei`.
8. (Optional) To restrict detection to a region of interest, set
   `valid_mask_relpath` to a 2D mask (nonzero = valid).

```bash
python drivers/run_integrated.py --config configs/local/integrated_ims.local.yaml
```

### Quickstart: storWIS batch (recommended)

Copy the template config first:

```bash
mkdir -p configs/local
cp configs/integrated_ims.example.yaml configs/local/integrated_ims.local.yaml
```

PowerShell:

```powershell
New-Item -ItemType Directory -Force configs/local
Copy-Item configs/integrated_ims.example.yaml configs/local/integrated_ims.local.yaml
```

Then **edit these keys in the copied template** (`configs/local/integrated_ims.local.yaml`).

**1) Single folder of `.ims` files**

```yaml
input_glob: "S:/BIC/<user>/equipment/<instrument>/<date>/*.ims"
stardist_model_dir: "models/y22m01d12_model_0"
output_runs_dir: runs
publish_dir: "S:/bioimg-results/<user>/<date>"
publish_mode: "error"
# batch_aggregate_spots: true
```

**2) Subfolders (e.g. `5ms/`, `45ms/`) + mirror publish**

```yaml
input_glob: "S:/BIC/<user>/equipment/<instrument>/<date>/*/*.ims"
stardist_model_dir: "models/y22m01d12_model_0"
output_runs_dir: runs
publish_dir: "S:/bioimg-results/<user>/<date>"
publish_mirror: true
input_base_dir: "S:/BIC/<user>/equipment/<instrument>/<date>"
publish_mode: "error"
# batch_aggregate_spots: true
```

Run:

```bash
python drivers/run_integrated.py --config configs/local/integrated_ims.local.yaml
```

### Outputs

Located in `$BIOIMG_DATA_ROOT/runs/<timestamp>__integrated/`:

- `spots.parquet`: Table of detected spots (inside nuclei).
- `nuclei_labels.tif`: Integer labels of segmented nuclei.
- `qc_overlay.png`: Spot channel image with nuclei outlines (red) and spots (cyan).
- `qc_cutouts.tif`: Multi-channel TIFF montage of spot crops (open in Fiji).
  For multiple spot channels, QC outputs are written per channel (e.g. `qc_overlay_ch2.png`).
- `run_manifest.yaml`: Run metadata.

For batch runs, outputs live under `$BIOIMG_DATA_ROOT/runs/<timestamp>__integrated_batch/`
with a `batch_manifest.yaml` and optional `spots_aggregate.parquet`.

### Batch spot-atlas PowerPoint QC (MATLAB-style)

To generate a single aggregate PPTX that shows **all detected spots** (multi-zoom, pixel-level),
run:

```bash
python drivers/generate_spot_atlas_pptx.py --batch-dir <timestamp>__integrated_batch
```

This writes:

- `<batch_dir>/qc_spot_atlas_batch.pptx`

The recommended notebook workflow is in `notebooks/03_generate_batch_spot_atlas_qc.py`.

### LLM-assisted development

We sometimes use an LLM (ChatGPT / GPT‑5.2 Pro) to plan and implement changes (often delivered as a zip overlay), then review diffs in GitHub Desktop and run verification + QC before committing. Full workflow and rules: [docs/LLM_WORKFLOW.md](docs/LLM_WORKFLOW.md).

