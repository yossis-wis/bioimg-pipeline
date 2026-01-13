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

*If you created your environment from the current `environment.yml`, `h5py` is
already included.*

If you're using an older environment that doesn't have `h5py` yet:

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

---

## Slice1 (StarDist nuclei segmentation)

Slice1 segments nuclei from a single 2D plane (TIFF or Imaris `.ims`) using a
StarDist2D model.

### 1) Put inputs in your data bench

1. Put an input image under your bench, e.g.:
   * `$BIOIMG_DATA_ROOT/raw_staging/my_image.ims`
   * or `$BIOIMG_DATA_ROOT/raw_staging/my_image.tif`

2. Put your StarDist model folder under your bench (outside git), e.g.:
   * `$BIOIMG_DATA_ROOT/models/y22m01d12_model_0/`

The model folder should contain files like `config.json`, `thresholds.json`, and
`weights_best.h5`.

### 2) Configure

Edit `configs/slice1_nuclei.yaml`:

* `input_relpath`: relative to `$BIOIMG_DATA_ROOT`
* `channel` / `ims_*` keys if you are reading `.ims`
* `stardist_model_dir`: absolute or relative to `$BIOIMG_DATA_ROOT`

### 3) Run

```bash
python drivers/run_slice1_nuclei.py --config configs/slice1_nuclei.yaml
```

Outputs are written under:

`$BIOIMG_DATA_ROOT/runs/<timestamp>__slice1_nuclei/`

Expected artifacts:

* `nuclei_labels.tif`
* `qc_overlay.png`
* `run_manifest.yaml`
* (optional) `nuclei.parquet`
