# Contracts

## Integrated slice: spots table contract

Output file: `spots.parquet`

Required columns (minimal, future-compatible):

| column        | type   | meaning |
|---------------|--------|---------|
| frame         | int    | frame index (0 for single-frame runs) |
| y_px          | float  | y coordinate in pixels |
| x_px          | float  | x coordinate in pixels |
| intensity     | float  | spot intensity (ADU/counts unless calibrated) |
| background    | float  | local background estimate |
| snr           | float  | signal-to-noise ratio |
| nucleus_label | int    | nucleus label containing the spot (0 if none) |

Rules:
- These columns must exist.
- Additional columns are allowed, but do not remove/rename these.
- Optional columns may include `spot_channel` (1-based spot channel index) for multi-channel runs.
- Optional metadata: `valid_mask_path` in the run manifest records any detection mask used.

- Optional/debug columns currently written by Slice0 include:
  - `u0`, `u1`: fixed-mask photometry values.
  - `quality`: LoG response at the candidate maximum (TrackMate-style “quality”).
  - `mean_in5`, `mean_in7`, `peak_intensity`: raw intensity summaries.
  - `sigma_px`, `log_size`, `log_radius_nm`: LoG-kernel diagnostics.
  - `y_subpx`, `x_subpx`: optional subpixel localization outputs (if enabled).

## Integrated slice: run manifest contract

Each run writes `run_manifest.yaml` next to outputs.

Minimal keys:
- timestamp
- input_path (resolved absolute path)
- config_snapshot (the YAML used for the run)
- output_dir (resolved absolute path)
- git_commit (if available)
- outputs (filenames for spots, nuclei labels, and QC artifacts)

