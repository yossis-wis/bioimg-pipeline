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

1. Copy your image to `$BIOIMG_DATA_ROOT/raw_staging/`.
2. Ensure you have your StarDist model in `$BIOIMG_DATA_ROOT/models/`.
3. Create a config (copy `configs/integrated_ims.yaml`) and update:
   - `input_relpath`
   - `channel_nuclei` and `channel_spots` (1-based indices; `channel_spots` can be a list like `[2, 3]`)
   - Detector parameters
4. (Optional) If you already have nuclei labels from another tool, set
   `nuclei_labels_relpath` to skip StarDist segmentation.
5. (Optional) If you have no nuclear channel, set `skip_nuclei_segmentation: true`
   and omit `channel_nuclei`.
6. (Optional) To restrict detection to a region of interest, set
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
