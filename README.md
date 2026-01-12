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
## Slice0 on a real Imaris `.ims` file

Your Andor Dragonfly/Imaris `.ims` files are HDF5-backed. This repo now supports reading a **single 2D plane**
directly from `.ims` (no manual conversion required).

Prerequisite (only needed for `.ims`):
```bash
conda install -c conda-forge h5py
```

1) Copy your `.ims` into your data bench, e.g.:
`$BIOIMG_DATA_ROOT/raw_staging/my_image.ims`

2) Create a config (or edit the example):

LINUX
```bash
# copy the example and edit input_relpath + detector params
cp configs/ims_example.yaml configs/ims.yaml
```
WINDOWS CMD
```bash
# copy the example and edit input_relpath + detector params
copy configs/ims_example.yaml configs/ims.yaml
```
3) Run Slice0:
```bash
python drivers/run_slice0.py --config configs/ims.yaml
```

### `.ims` config keys

- `channel` (1-based): `1 => Channel 0`, `2 => Channel 1`, etc.
- `ims_resolution_level` (default `0`)
- `ims_time_index` (default `0`)
- `ims_z_index` (default `0`)

The output artifacts are the same as the TIFF path:
- `spots.parquet`
- `run_manifest.yaml`
- `qc_overlay.png` (overlay of predicted spots on the 2D plane)
