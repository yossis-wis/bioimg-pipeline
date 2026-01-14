# Contracts

## Slice 0: spots table contract

Output file: `spots.parquet`

Required columns (minimal, future-compatible):

| column       | type   | meaning |
|--------------|--------|---------|
| frame        | int    | frame index (0 for slice 0 if single frame) |
| y_px         | float  | y coordinate in pixels |
| x_px         | float  | x coordinate in pixels |
| intensity    | float  | spot intensity (ADU/counts unless calibrated) |
| background   | float  | local background estimate |
| snr          | float  | signal-to-noise ratio |

Rules:
- These columns must exist, even if some are placeholders for Slice 0.
- Additional columns are allowed, but do not remove/rename these.

## Slice 0: run manifest contract

Each run writes `run_manifest.yaml` next to outputs.

Minimal keys:
- timestamp
- input_path (resolved absolute path)
- config_snapshot (the YAML used for the run)
- output_dir (resolved absolute path)
- git_commit (if available)
- env_name (e.g. `bioimg-slice0`)
