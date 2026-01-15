# bioimg-pipeline

## Integrated Slice (Nuclei + Spots)

This pipeline performs **nuclear segmentation** (StarDist) and **spot detection** (LoG) in a single pass. It produces quality control artifacts including spot cutouts and visual overlays.

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
2. Create a config (copy `configs/integrated_ims.yaml`) and update:
   - `input_relpath` (single file), `input_relpaths` (list), or `input_glob` (batch).
     - For storWIS: point `input_glob` at the network share (e.g. `S:/BIC/.../*.ims`)
     - Relative paths resolve under `$BIOIMG_DATA_ROOT`; absolute paths are used as-is.
   - `channel_nuclei` and `channel_spots` (1-based indices; `channel_spots` can be a list like `[2, 3]`)
   - Detector parameters
3. (Optional) If you want local intermediates + storWIS publish:
   - Keep `output_runs_dir` under `BIOIMG_DATA_ROOT` for fast local writes.
   - Set `publish_dir` to a writable storWIS location to copy final outputs.
   - Set `publish_mirror: true` and `input_base_dir` to mirror subfolders (e.g. `.../5ms`, `.../45ms`).
   - Batch publishes land under `publish_dir/<batch_dir_name-or-timestamp>/...` to avoid collisions.
4. (Optional) For batch robustness:
   - `continue_on_error: true` to keep processing after a failed file.
   - `skip_existing: true` to resume without redoing completed outputs.
   - `batch_dir_name` to reuse a stable output folder name across reruns.
   - `batch_aggregate_spots: true` to write a combined `spots_aggregate.parquet`.
5. (Optional) If you already have nuclei labels from another tool, set
   `nuclei_labels_relpath` to skip StarDist segmentation.
6. (Optional) If you have no nuclear channel, set `skip_nuclei_segmentation: true`
   and omit `channel_nuclei`.
7. (Optional) To restrict detection to a region of interest, set
   `valid_mask_relpath` to a 2D mask (nonzero = valid).

```bash
python drivers/run_integrated.py --config configs/integrated_ims.yaml
```

### Outputs

Located in `$BIOIMG_DATA_ROOT/runs/<timestamp>__integrated/`:

- `spots.parquet`: Table of detected spots (inside nuclei).
- `nuclei_labels.tif`: Integer labels of segmented nuclei.
- `qc_overlay.png`: Spot channel image with nuclei outlines (red) and spots (cyan).
- `qc_cutouts.tif`: Multi-channel TIFF montage of spot crops (open in Fiji).
  For multiple spot channels, QC outputs are written per channel (e.g. `qc_overlay_ch2.png`).
- By default the montage samples the highest-SNR spots; set `qc_sample_seed` to shuffle instead.
- `run_manifest.yaml`: Run metadata.
For batch runs, outputs live under `$BIOIMG_DATA_ROOT/runs/<timestamp>__integrated_batch/`
with a `batch_manifest.yaml` and optional `spots_aggregate.parquet` (includes a `condition`
column derived from the input folder name).
