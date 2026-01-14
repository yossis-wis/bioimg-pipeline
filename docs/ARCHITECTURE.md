# One-Page White Paper: Integrated Slice

**Goal:** An end-to-end pipeline that segments nuclei and detects spots within them.

## Workflow

1. **Input:** Multi-channel image (TIFF or Imaris .ims).
   - Optional Channel A: Nuclear marker (e.g., DAPI)
   - One or more spot channels (e.g., FISH, protein)
2. **Kernel 1 (Segmentation):** StarDist runs on Channel A -> `nuclei_labels.tif` (or reuse a precomputed label image).
3. **Kernel 2 (Detection):** LoG detector runs on each spot channel, masked by `nuclei_labels` if provided -> `spots.parquet`.
4. **QC Generation:**
   - **Overlay:** Full frame image showing nuclei contours and spot locations.
   - **Montage:** 80x80 pixel cutouts of spots from both channels, stitched into a multi-channel TIFF.

## Contracts

### Spots Table (`spots.parquet`)
- `y_px`, `x_px`: Spot coordinates.
- `intensity`: Spot intensity.
- `nucleus_label`: ID of the nucleus containing the spot.
- `snr`: Signal-to-noise ratio.

### Run Manifest (`run_manifest.yaml`)
- `timestamp`, `git_commit`, `input_path`.
- `outputs`: Filenames of all generated artifacts.
