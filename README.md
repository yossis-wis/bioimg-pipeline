# bioimg-pipeline

## Slice0 (simulated TIFF first)

1) Verify your bench + environment:

```bash
python scripts/verify_setup.py
```

2) Generate a deterministic synthetic TIFF in your data bench:

```bash
python scripts/generate_phantom_tiff.py --config configs/sim.yaml
```

3) Run Slice0 on that TIFF (same path you'd use for real data later):

```bash
python drivers/run_slice0.py --config configs/sim.yaml
```

Outputs are written under:

`$BIOIMG_DATA_ROOT/runs/<timestamp>__slice0/`

Expected artifacts:
- `spots.parquet`
- `run_manifest.yaml`
- `qc_overlay.png`
