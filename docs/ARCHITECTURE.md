# One-Page White Paper: Integrated Slice

**Goal:** An end-to-end pipeline that segments nuclei and detects spots within them.

## Workflow

1. **Input:** Multi-channel image (TIFF or Imaris .ims).
   - Optional Channel A: Nuclear marker (e.g., DAPI)
   - One or more spot channels (e.g., FISH, protein)
2. **Kernel 1 (Segmentation):** StarDist runs on Channel A -> `nuclei_labels.tif` (or reuse a precomputed label image).
3. **Kernel 2 (Detection / Slice0):** Spot detection runs on each spot channel -> `spots.parquet`.
   - **Stage A:** TrackMate-style LoG candidate generation (FFT convolution + 3×3 local maxima + `q_min`).
   - **Stage B:** Per-candidate photometry using the fixed in5/out0 masks and the `u0_min` threshold (**unchanged**).
   - Optional masks can restrict candidates to `nuclei_labels > 0` (inside nuclei) and/or a `valid_mask`.
   - Details + parameter mapping: see [`docs/SPOT_DETECTION.md`](SPOT_DETECTION.md).
4. **QC Generation:**
   - **Overlay:** Full frame image showing nuclei contours and spot locations.
   - **Montage:** 80×80 pixel cutouts of spots from both channels, stitched into a multi-channel TIFF.
   - **Spot atlas (optional):** a MATLAB-style PowerPoint deck with multi-zoom spot cutouts for batch-scale review.

## Contracts

### Spots Table (`spots.parquet`)
- `y_px`, `x_px`: Spot coordinates.
- `intensity`: Spot intensity.
- `nucleus_label`: ID of the nucleus containing the spot.
- `snr`: Signal-to-noise ratio.

### Run Manifest (`run_manifest.yaml`)
- `timestamp`, `git_commit`, `input_path`.
- `outputs`: Filenames of all generated artifacts.

